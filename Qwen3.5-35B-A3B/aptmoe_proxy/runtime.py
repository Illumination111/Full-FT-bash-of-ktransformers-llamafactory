"""APTMoE pipeline construction, full-update audit, and canonical timing."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from Runtime.OffloadRuntime.comm_scheduler import CommScheduler
from Runtime.PipelineRuntime.generate_action_list import (
    generate_action_Mobius_APTMoE,
)
from Runtime.PipelineRuntime.pipeline_runtime import PipelineRuntime

from .model import (
    APTQwen35Stage,
    Qwen35ModelShard,
    categorized_parameter_counts,
    parameter_category,
)
from .placement import ProxyPlacementSolver
from .routes import RouteController


TIMING_MODE = "coarse_host_wall_no_cuda_sync"
PHASE_KEYS = ("forward_sec", "backward_sec", "optimizer_sec")
REQUIRED_GRADIENT_CATEGORIES = {
    "embedding",
    "lm_head",
    "norm",
    "token_mixer",
    "router",
    "routed_experts",
    "shared_expert_and_gate",
}
REQUIRED_NUMERICAL_UPDATE_CATEGORIES = (
    REQUIRED_GRADIENT_CATEGORIES - {"norm"}
)


class ProxyPipelineRuntime(PipelineRuntime):
    """APTMoE runtime variant that does not call loss.item() inside each step."""

    def forward_pass(
        self,
        mod_rank: int,
        source_tensor: torch.Tensor,
        chunk_id: int,
    ) -> None:
        target_module = self.module_list[mod_rank]
        next_module = (
            self.module_list[mod_rank + 1]
            if mod_rank != self.num_stages - 1
            else None
        )
        is_last_sft = self.sft_mode and self._is_last_stage_module(mod_rank)
        labels = self._get_labels_for_chunk(chunk_id) if is_last_sft else None
        num_items = self.step_num_items_in_batch

        target_module.FwdStageLoad(
            chunk_id,
            self.num_chunks,
            sft_mode=self.sft_mode,
        )
        if mod_rank + self.world_size < len(self.module_list):
            next_target_module = self.module_list[mod_rank + self.world_size]
            next_target_module.FwdStageLoad(
                chunk_id,
                self.num_chunks,
                sft_mode=self.sft_mode,
            )

        self.input_batch_list[mod_rank].append(source_tensor)
        if is_last_sft and labels is not None:
            result = target_module(
                source_tensor,
                chunk_id,
                next_module,
                labels=labels,
                num_items_in_batch=num_items,
            )
        else:
            result = target_module(source_tensor, chunk_id, next_module)
        if self.sft_mode:
            torch.cuda.current_stream().wait_event(
                target_module.StageCompEvent
            )
        target_module.FwdStageDrop(
            chunk_id,
            self.num_chunks,
            self.fwd_only,
            sft_mode=self.sft_mode,
        )
        self.batch_activation_list[mod_rank].append(result)


def split_action_list(actions: list[str]) -> tuple[list[str], list[str]]:
    split_at = None
    for index, action in enumerate(actions):
        if action.startswith("backward ") or action.startswith("send_grad "):
            split_at = index
            break
    if split_at is None:
        raise ValueError("APTMoE action list does not contain a backward phase")
    forward = actions[:split_at]
    backward = actions[split_at:]
    if not forward or not backward:
        raise ValueError("forward/backward action split is empty")
    return forward, backward


def build_proxy_pipeline(
    *,
    config: Any,
    world_size: int,
    local_rank: int,
    global_rank: int,
    routes: RouteController,
    placement_solver: ProxyPlacementSolver,
    seed: int,
) -> tuple[list[Qwen35ModelShard | None], CommScheduler]:
    if config.num_hidden_layers != 40:
        raise ValueError("Qwen3.5 proxy requires exactly 40 decoder layers")
    num_stages = config.num_hidden_layers
    layers_per_stage = 1
    comm_scheduler = CommScheduler(device_id=local_rank)
    module_list: list[Qwen35ModelShard | None] = []
    for stage_id in range(num_stages):
        if stage_id % world_size != global_rank:
            module_list.append(None)
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + stage_id)
            stage = APTQwen35Stage(
                config=config,
                stage_id=stage_id,
                global_layer_offset=stage_id * layers_per_stage,
                num_layers=layers_per_stage,
                routes=routes,
                comm_scheduler=comm_scheduler,
                placement_solver=placement_solver,
                is_first_stage=stage_id == 0,
                is_last_stage=stage_id == num_stages - 1,
            )
        module_list.append(
            Qwen35ModelShard(
                model_shard=stage,
                compute_device_id=local_rank,
                offload_device="cpu",
                index=stage_id,
                offload_grained="fine",
                inter_stage_only=True,
                sft_mode=True,
            )
        )
    return module_list, comm_scheduler


def local_parameter_counts(
    module_list: Iterable[Qwen35ModelShard | None],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for shard in module_list:
        if shard is None:
            continue
        for category, count in categorized_parameter_counts(
            shard.model_shard
        ).items():
            result[category] = result.get(category, 0) + count
    return result


def global_parameter_counts(
    local_counts: dict[str, int],
) -> dict[str, int]:
    gathered: list[dict[str, int] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local_counts)
    result: dict[str, int] = {}
    for rank_counts in gathered:
        if rank_counts is None:
            continue
        for category, count in rank_counts.items():
            result[category] = result.get(category, 0) + int(count)
    return result


def wait_for_local_stage_drops(
    module_list: Iterable[Qwen35ModelShard | None],
) -> None:
    """Wait only on each owned stage's drop event, never on the whole device."""
    for shard in module_list:
        if shard is not None:
            shard._StageDropEvent.synchronize()


class FullUpdateAudit:
    """Prove optimizer scope, gradients, BF16 moments, and numerical updates."""

    def __init__(
        self,
        module_list: list[Qwen35ModelShard | None],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.module_list = module_list
        self.optimizer = optimizer
        self.local_parameters: list[tuple[str, torch.nn.Parameter]] = []
        for shard in module_list:
            if shard is None:
                continue
            self.local_parameters.extend(
                shard.model_shard.named_parameters()
            )
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        model_ids = {id(parameter) for _, parameter in self.local_parameters}
        if optimizer_ids != model_ids:
            raise RuntimeError(
                "optimizer parameter objects do not exactly match local proxy parameters"
            )
        self.samples: dict[
            str,
            list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]],
        ] = {}
        self.sample_counts: dict[str, int] = {}
        self.gradient_seen: dict[str, bool] = {}
        self.optimizer_state_dtypes: set[str] = set()
        self.optimizer_state_devices: set[str] = set()

    def observe_gradients_before_step(self) -> None:
        for name, parameter in self.local_parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            category = parameter_category(name)
            remaining = 4096 - self.sample_counts.get(category, 0)
            if remaining <= 0:
                continue
            flat_gradient = gradient.detach().reshape(-1)
            max_index = int(flat_gradient.abs().argmax().item())
            if flat_gradient[max_index].item() == 0:
                continue
            self.gradient_seen[category] = True
            sample_size = min(remaining, parameter.numel())
            start = min(
                max(0, max_index - sample_size // 2),
                parameter.numel() - sample_size,
            )
            indices = torch.arange(
                start,
                start + sample_size,
                device=parameter.device,
            )
            before = (
                parameter.detach()
                .reshape(-1)
                .index_select(0, indices)
                .to(device="cpu", copy=True)
            )
            self.samples.setdefault(category, []).append(
                (
                    parameter,
                    indices.to(device="cpu"),
                    before,
                )
            )
            self.sample_counts[category] = (
                self.sample_counts.get(category, 0) + sample_size
            )

    def observe_optimizer_state(self) -> None:
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    self.optimizer_state_dtypes.add(str(value.dtype))
                    self.optimizer_state_devices.add(value.device.type)

    def local_result(self) -> dict[str, Any]:
        changed: dict[str, bool] = {}
        for category, samples in self.samples.items():
            changed[category] = False
            for parameter, indices, before in samples:
                after = (
                    parameter.detach()
                    .reshape(-1)
                    .index_select(0, indices.to(parameter.device))
                    .to(device="cpu", copy=True)
                )
                changed[category] = (
                    changed[category] or not torch.equal(before, after)
                )
        return {
            "optimizer_parameter_count": sum(
                parameter.numel()
                for _, parameter in self.local_parameters
            ),
            "gradient_seen": dict(self.gradient_seen),
            "weight_changed": changed,
            "optimizer_state_dtypes": sorted(self.optimizer_state_dtypes),
            "optimizer_state_devices": sorted(self.optimizer_state_devices),
        }

    def gather_result(self) -> dict[str, Any] | None:
        local = self.local_result()
        gathered: list[dict[str, Any] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, local)
        if dist.get_rank() != 0:
            return None
        merged_grad: dict[str, bool] = {}
        merged_changed: dict[str, bool] = {}
        state_dtypes: set[str] = set()
        state_devices: set[str] = set()
        parameter_count = 0
        for result in gathered:
            assert result is not None
            parameter_count += int(result["optimizer_parameter_count"])
            for category, value in result["gradient_seen"].items():
                merged_grad[category] = merged_grad.get(category, False) or bool(
                    value
                )
            for category, value in result["weight_changed"].items():
                merged_changed[category] = merged_changed.get(
                    category,
                    False,
                ) or bool(value)
            state_dtypes.update(result["optimizer_state_dtypes"])
            state_devices.update(result["optimizer_state_devices"])

        missing_gradients = sorted(
            category
            for category in REQUIRED_GRADIENT_CATEGORIES
            if not merged_grad.get(category, False)
        )
        unchanged = sorted(
            category
            for category in REQUIRED_NUMERICAL_UPDATE_CATEGORIES
            if not merged_changed.get(category, False)
        )
        valid = (
            not missing_gradients
            and not unchanged
            and state_dtypes == {"torch.bfloat16"}
            and state_devices == {"cpu"}
        )
        return {
            "schema_version": 1,
            "optimizer_scope": "all_proxy_parameters",
            "optimizer_parameter_count": parameter_count,
            "gradient_seen": merged_grad,
            "weight_changed": merged_changed,
            "required_gradient_categories": sorted(
                REQUIRED_GRADIENT_CATEGORIES
            ),
            "required_numerical_update_categories": sorted(
                REQUIRED_NUMERICAL_UPDATE_CATEGORIES
            ),
            "missing_gradient_categories": missing_gradients,
            "unchanged_weight_categories": unchanged,
            "optimizer_state_dtypes": sorted(state_dtypes),
            "optimizer_state_devices": sorted(state_devices),
            "valid_full_update": valid,
        }


def _stats(rows: list[dict[str, float | int]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {
            "count": 0,
            "mean_sec": None,
            "min_sec": None,
            "max_sec": None,
        }
    return {
        "count": len(values),
        "mean_sec": statistics.fmean(values),
        "min_sec": min(values),
        "max_sec": max(values),
    }


def _merge_rank_rows(
    gathered: list[list[dict[str, float | int]]],
    tokens_per_step: int,
) -> list[dict[str, float | int]]:
    step_count = len(gathered[0])
    if any(len(rows) != step_count for rows in gathered):
        raise RuntimeError("ranks produced different timing row counts")
    merged: list[dict[str, float | int]] = []
    for index in range(step_count):
        source_rows = [rows[index] for rows in gathered]
        total = max(float(row["step_total_sec"]) for row in source_rows)
        merged.append(
            {
                "global_step": int(source_rows[0]["global_step"]),
                "microbatches": int(source_rows[0]["microbatches"]),
                "forward_sec": max(
                    float(row["forward_sec"]) for row in source_rows
                ),
                "backward_sec": max(
                    float(row["backward_sec"]) for row in source_rows
                ),
                "optimizer_sec": max(
                    float(row["optimizer_sec"]) for row in source_rows
                ),
                "step_total_sec": total,
                "step_tps": tokens_per_step / total if total > 0 else 0.0,
            }
        )
    return merged


def gather_and_write_timing(
    *,
    local_rows: list[dict[str, float | int]],
    output_dir: Path,
    warmup_steps: int,
    tokens_per_step: int,
) -> dict[str, Any] | None:
    gathered: list[list[dict[str, float | int]] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local_rows)
    if dist.get_rank() != 0:
        return None
    rank_rows = [rows for rows in gathered if rows is not None]
    merged = _merge_rank_rows(rank_rows, tokens_per_step)
    stable = [
        row
        for row in merged
        if int(row["global_step"]) > warmup_steps
    ]
    aggregate_all = {
        key: _stats(merged, key)
        for key in (*PHASE_KEYS, "step_total_sec")
    }
    aggregate_stable = {
        key: _stats(stable, key)
        for key in (*PHASE_KEYS, "step_total_sec")
    }
    mean_stable = aggregate_stable["step_total_sec"]["mean_sec"]
    stable_tps = (
        tokens_per_step / float(mean_stable)
        if mean_stable is not None and float(mean_stable) > 0
        else None
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "timing_mode": TIMING_MODE,
        "backend": "aptmoe",
        "precision": "bf16",
        "benchmark_class": "deployment_proxy",
        "instrumentation": {
            "forced_cuda_synchronize": False,
            "localized_stage_drop_event_wait": True,
            "backend_internal_probes": False,
            "system_resource_monitor": False,
            "per_step_file_io": False,
        },
        "rank_reduction": "per-step maximum host wall time across pipeline ranks",
        "warmup_steps": warmup_steps,
        "tokens_per_step": tokens_per_step,
        "num_steps": len(merged),
        "num_stable_steps": len(stable),
        "steps": merged,
        "aggregate_all": aggregate_all,
        "aggregate_stable": aggregate_stable,
        "tps_attribution": {
            "tokens_per_step": tokens_per_step,
            "mean_stable_step_sec": mean_stable,
            "stable_tps": stable_tps,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "step_timing.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "global_step",
        "microbatches",
        *PHASE_KEYS,
        "step_total_sec",
        "step_tps",
    ]
    with (output_dir / "step_timing.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    return summary


def run_full_update_steps(
    *,
    runtime: ProxyPipelineRuntime,
    module_list: list[Qwen35ModelShard | None],
    routes: RouteController,
    steps: int,
    warmup_steps: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    tokens_per_step: int,
    timing_output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    actions = generate_action_Mobius_APTMoE(
        world_size=dist.get_world_size(),
        num_stages=runtime.num_stages,
        num_chunks=runtime.num_chunks,
    )[dist.get_rank()]
    forward_actions, backward_actions = split_action_list(actions)
    runtime._checked_stages = set(range(runtime.num_stages))
    audit = FullUpdateAudit(module_list, runtime.optimizer_)
    audit_steps = max(1, warmup_steps)
    local_rows: list[dict[str, float | int]] = []

    for step_index in range(steps):
        step_started = time.perf_counter()
        runtime.optimizer_.zero_grad(set_to_none=True)

        runtime.prefetch_step_data(gradient_accumulation_steps)
        forward_seconds = 0.0
        backward_seconds = 0.0
        for accumulation_index in range(gradient_accumulation_steps):
            routes.set_position(step_index, accumulation_index)
            forward_started = time.perf_counter()
            runtime.run_pipeline(action_list=forward_actions)
            forward_seconds += time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            runtime.run_pipeline(action_list=backward_actions)
            wait_for_local_stage_drops(module_list)
            backward_seconds += time.perf_counter() - backward_started
        if step_index < audit_steps:
            audit.observe_gradients_before_step()

        optimizer_started = time.perf_counter()
        if max_grad_norm > 0:
            parameters_with_grad = [
                parameter
                for group in runtime.optimizer_.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if parameters_with_grad:
                torch.nn.utils.clip_grad_norm_(
                    parameters_with_grad,
                    max_grad_norm,
                )
        runtime.optimizer_.step()
        runtime.scheduler_step()
        optimizer_seconds = time.perf_counter() - optimizer_started
        if step_index < audit_steps:
            audit.observe_optimizer_state()
        step_total = time.perf_counter() - step_started
        local_rows.append(
            {
                "global_step": step_index + 1,
                "microbatches": (
                    gradient_accumulation_steps * runtime.num_chunks
                ),
                "forward_sec": forward_seconds,
                "backward_sec": backward_seconds,
                "optimizer_sec": optimizer_seconds,
                "step_total_sec": step_total,
                "step_tps": (
                    tokens_per_step / step_total
                    if step_total > 0
                    else 0.0
                ),
            }
        )

    timing = gather_and_write_timing(
        local_rows=local_rows,
        output_dir=timing_output_dir,
        warmup_steps=warmup_steps,
        tokens_per_step=tokens_per_step,
    )
    verification = audit.gather_result()
    return timing, verification
