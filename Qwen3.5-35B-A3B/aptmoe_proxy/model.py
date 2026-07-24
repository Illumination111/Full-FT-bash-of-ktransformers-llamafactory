"""Parameter-exact Qwen3.5 proxy stages wired to APTMoE expert offload."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
)

from model.transformer_lm import OffloadInputBegin, OffloadInputEnd
from Runtime.OffloadRuntime.offload import ModelShard, random_stageload_list

from qwen35_aptmoe_proxy_components import (
    Qwen35RoutedExpert,
    Qwen35SharedExpert,
    Qwen35TokenMixer,
)

from .placement import ProxyPlacementSolver
from .routes import RouteController


class APTQwen35RoutedExpert(Qwen35RoutedExpert):
    """Qwen3.5 expert with APTMoE's differentiable CPU input/output bridge."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.on_GPU:
            return super().forward(hidden_states)
        gpu_device = hidden_states.device
        cpu_states = OffloadInputBegin.apply(
            hidden_states,
            gpu_device,
            self.layer_id,
            self.expert_id,
        )
        cpu_output = super().forward(cpu_states)
        return OffloadInputEnd.apply(
            cpu_output,
            gpu_device,
            self.layer_id,
            self.expert_id,
        )


class APTQwen35Router(nn.Module):
    """Exact Qwen3.5 router projection with optional top-k route replay."""

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_idx: int,
        routes: RouteController,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size)
        )
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)
        self.layer_idx = layer_idx
        self.routes = routes
        self.on_GPU = False
        self.last_topk_scores: torch.Tensor | None = None
        self.last_topk_indices: torch.Tensor | None = None
        self.last_counts: list[int] | None = None

    def forward(self, hidden_states: torch.Tensor) -> list[int]:
        logits = F.linear(hidden_states, self.weight)
        scores, indices, counts = self.routes.select(
            layer_idx=self.layer_idx,
            logits=logits,
        )
        self.last_topk_scores = scores
        self.last_topk_indices = indices
        self.last_counts = counts
        return counts


class APTQwen35RoutePredictor(nn.Module):
    """Parameter-free replay predictor; it does not inflate target parameters."""

    def __init__(self, layer_idx: int, routes: RouteController) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.routes = routes
        self.on_GPU = False

    def forward(self, hidden_states: torch.Tensor) -> list[int]:
        del hidden_states
        return self.routes.predicted_counts(self.layer_idx)


class APTQwen35MoELayer(nn.Module):
    """Top-8 weighted dispatch over independently movable 6 MiB experts."""

    def __init__(
        self,
        *,
        config: Qwen3_5MoeTextConfig,
        layer_idx: int,
        stage_id: int,
        gate: APTQwen35Router,
        next_gate: APTQwen35RoutePredictor,
        comp_stream: torch.cuda.Stream,
        comm_scheduler: Any,
        placement_solver: ProxyPlacementSolver,
        gate_event: torch.cuda.Event,
        expert_events: list[torch.cuda.Event],
    ) -> None:
        super().__init__()
        self.gate = gate
        self.next_gate = next_gate
        self.layer_id = layer_idx
        self.comp_stream = comp_stream
        self.CommScheduler = comm_scheduler
        self.R_solver = placement_solver
        self.Gate_event = gate_event
        self.expert_events = expert_events
        self.pipeline = "APTMoE"
        self.is_dense = False
        self.layer_type = config.layer_types[layer_idx]
        self.is_first_stage = layer_idx == 0
        self.is_last_stage = layer_idx == config.num_hidden_layers - 1
        self.num_local_experts = config.num_experts
        self.experts = nn.ModuleList(
            [
                APTQwen35RoutedExpert(
                    config.hidden_size,
                    config.moe_intermediate_size,
                    layer_id=layer_idx,
                    expert_id=expert_id,
                    device="cpu",
                    dtype=torch.bfloat16,
                )
                for expert_id in range(config.num_experts)
            ]
        )
        self.shared_experts = Qwen35SharedExpert(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            device="cpu",
            dtype=torch.bfloat16,
        )
        self.stage_id = stage_id
        self.predicted_expert_selection_list: list[list[int]] = []
        self.next_layer_expert_selection_prediction: list[int] | None = None
        self.real_expert_selection: list[int] = []
        self.assigned_tokens_list = [0] * config.num_experts
        self.historical_assigned_tokens_list = [0] * config.num_experts

    def _queue_hot_experts(self, counts: list[int]) -> None:
        gpu_expert_ids = self.R_solver.solve(
            counts,
            layer_type=self.layer_type,
            is_first_stage=self.is_first_stage,
            is_last_stage=self.is_last_stage,
        )
        # CommScheduler.load_execute_with_priority() deliberately launches only
        # one queued transfer when its event is not complete yet.  Submit every
        # missing expert explicitly so tasks from a cold-start route cannot
        # leak into later stages or the stable timing window.
        self.CommScheduler.clear_priority(0)
        queued = 0
        for expert_id in gpu_expert_ids:
            expert = self.experts[expert_id]
            if not expert.on_GPU:
                self.CommScheduler.add_model_to_queue(
                    expert,
                    self.expert_events[expert_id],
                    0,
                )
                queued += 1
        for _ in range(queued):
            self.CommScheduler.load_execute_with_priority()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"MoE input must be [batch, sequence, hidden], got {hidden_states.shape}"
            )
        self.comp_stream.wait_event(self.Gate_event)
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        counts = self.gate(flat_states)
        scores = self.gate.last_topk_scores
        indices = self.gate.last_topk_indices
        if scores is None or indices is None:
            raise RuntimeError("router did not expose top-k scores and indices")
        self.real_expert_selection = counts
        for expert_id, count in enumerate(counts):
            self.assigned_tokens_list[expert_id] += count
        self.next_layer_expert_selection_prediction = self.next_gate(flat_states)
        self.predicted_expert_selection_list.clear()
        self._queue_hot_experts(self.assigned_tokens_list)

        final_output = torch.zeros_like(flat_states)
        dispatch: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for expert_id, expert in enumerate(self.experts):
            positions = torch.nonzero(
                indices == expert_id,
                as_tuple=False,
            )
            if positions.numel() == 0:
                continue
            token_indices = positions[:, 0]
            topk_slots = positions[:, 1]
            dispatch.append((expert_id, token_indices, topk_slots))

        # Queue all GPU work before a CPU expert can block the host.  This
        # preserves the inter-expert CPU/GPU overlap used by APTMoE.
        for execute_on_gpu in (True, False):
            for expert_id, token_indices, topk_slots in dispatch:
                expert = self.experts[expert_id]
                if expert.on_GPU is not execute_on_gpu:
                    continue
                current_states = flat_states.index_select(0, token_indices)
                if execute_on_gpu:
                    self.comp_stream.wait_event(
                        self.expert_events[expert_id]
                    )
                expert_output = expert(current_states)
                weighted_output = expert_output * scores[
                    token_indices,
                    topk_slots,
                ].unsqueeze(-1)
                final_output.index_add_(
                    0,
                    token_indices,
                    weighted_output.to(final_output.dtype),
                )

        final_output = final_output + self.shared_experts(flat_states)
        return final_output.reshape_as(hidden_states)


class APTQwen35DecoderLayer(nn.Module):
    """One exact token mixer plus APTMoE-decomposed Qwen3.5 experts."""

    def __init__(
        self,
        *,
        config: Qwen3_5MoeTextConfig,
        layer_idx: int,
        stage_id: int,
        routes: RouteController,
        comm_scheduler: Any,
        placement_solver: ProxyPlacementSolver,
    ) -> None:
        super().__init__()
        self.layer_id = layer_idx
        self.stage_id = stage_id
        self.comp_stream = torch.cuda.Stream()
        self.CommScheduler = comm_scheduler
        self.self_attn = Qwen35TokenMixer(config, layer_idx)
        self.norm1 = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm2 = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.dropout = nn.Identity()
        self.MHA_event = torch.cuda.Event()
        self.Gate_event = torch.cuda.Event()
        self.expert_events = [
            torch.cuda.Event() for _ in range(config.num_experts)
        ]
        self.gate = APTQwen35Router(config, layer_idx, routes)
        self.next_gate = APTQwen35RoutePredictor(layer_idx, routes)
        self.moe_layer = APTQwen35MoELayer(
            config=config,
            layer_idx=layer_idx,
            stage_id=stage_id,
            gate=self.gate,
            next_gate=self.next_gate,
            comp_stream=self.comp_stream,
            comm_scheduler=comm_scheduler,
            placement_solver=placement_solver,
            gate_event=self.Gate_event,
            expert_events=self.expert_events,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.moe_layer(hidden_states)
        return residual + hidden_states


class APTQwen35Stage(nn.Sequential):
    """One or more Qwen3.5 layers forming an APTMoE pipeline stage."""

    def __init__(
        self,
        *,
        config: Qwen3_5MoeTextConfig,
        stage_id: int,
        global_layer_offset: int,
        num_layers: int,
        routes: RouteController,
        comm_scheduler: Any,
        placement_solver: ProxyPlacementSolver,
        is_first_stage: bool,
        is_last_stage: bool,
    ) -> None:
        comp_stream = torch.cuda.Stream()
        layers = [
            APTQwen35DecoderLayer(
                config=config,
                layer_idx=global_layer_offset + local_idx,
                stage_id=stage_id,
                routes=routes,
                comm_scheduler=comm_scheduler,
                placement_solver=placement_solver,
            )
            for local_idx in range(num_layers)
        ]
        super().__init__(*layers)
        self._ndecoder = len(layers)
        self.comp_stream = comp_stream
        for layer in layers:
            layer.comp_stream = comp_stream
            layer.moe_layer.comp_stream = comp_stream
        self.comm_scheduler = comm_scheduler
        self.R_solver = placement_solver
        self.stage_id = stage_id
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        self.hidden_size = config.hidden_size
        self.embed_tokens: nn.Embedding | None = None
        self.final_norm: nn.Module | None = None
        self.lm_head: nn.Linear | None = None
        if is_first_stage:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
            )
            nn.init.normal_(
                self.embed_tokens.weight,
                mean=0.0,
                std=config.initializer_range,
            )
        if is_last_stage:
            self.final_norm = Qwen3_5MoeRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=config.initializer_range,
            )
        self.to(device="cpu", dtype=torch.bfloat16)

    def __iter__(self) -> Iterator[APTQwen35DecoderLayer]:
        return iter(
            [self._modules[str(index)] for index in range(self._ndecoder)]
        )

    def __len__(self) -> int:
        return self._ndecoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if self.embed_tokens is not None and hidden_states.dtype == torch.long:
            hidden_states = self.embed_tokens(hidden_states)
        for layer in self:
            hidden_states = layer(hidden_states)
        if self.lm_head is None:
            return hidden_states

        assert self.final_norm is not None
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if labels is None:
            return logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction=reduction,
        )
        if num_items_in_batch is not None:
            loss = loss / max(1, num_items_in_batch)
        return loss


class Qwen35ModelShard(ModelShard):
    """APTMoE shard with route-aware stage load and safe route snapshots."""

    def FwdStageLoad(
        self,
        chunk_id: int | None = None,
        num_chunks: int | None = None,
        sft_mode: bool = False,
    ) -> None:
        del num_chunks, sft_mode
        if chunk_id != 0:
            return

        # There is exactly one decoder layer per pipeline stage.  Loading it as
        # an inter-stage unit avoids APTMoE's multi-layer prediction queue while
        # preserving its CPU-home expert placement and asynchronous H2D stream.
        # Preload decisions use the previous microbatch's measured popularity;
        # the current replayed route can still add missing hot experts after
        # the router runs.  This avoids an oracle look-ahead TPS advantage.
        load_model: list[nn.Module] = []
        hot_by_layer: dict[int, list[int]] = {}
        load_model.extend(self._get_extra_modules())
        for layer in self.model_shard:
            load_model.extend(
                (
                    layer.self_attn,
                    layer.norm1,
                    layer.norm2,
                    layer.dropout,
                    layer.gate,
                    layer.next_gate,
                    layer.moe_layer.shared_experts,
                )
            )
            counts = list(
                layer.moe_layer.historical_assigned_tokens_list
            )
            hot_expert_ids = self.R_solver.solve(
                counts,
                layer_type=layer.self_attn.layer_type,
                is_first_stage=self.model_shard.is_first_stage,
                is_last_stage=self.model_shard.is_last_stage,
            )
            if not hot_expert_ids:
                hot_expert_ids = random_stageload_list(
                    len(layer.moe_layer.experts),
                    portion=self.R_solver.prefetch_portion,
                )
            hot_by_layer[layer.layer_id] = hot_expert_ids
            for expert_id in hot_expert_ids:
                load_model.append(layer.moe_layer.experts[expert_id])
        self.CommScheduler.load_execute(
            load_model,
            waitEvent=self._StageDropEvent,
            recordEvent=self.StageLoadEvent,
        )
        # The common StageLoadEvent is sufficient for stage correctness.  Also
        # refresh the per-expert events because the MoE dispatch checks them
        # before launching a hot expert, including during checkpoint recompute.
        with torch.cuda.stream(self.CommScheduler.load_stream):
            for layer in self.model_shard:
                for expert_id in hot_by_layer[layer.layer_id]:
                    layer.expert_events[expert_id].record()

    def FwdStageDrop(
        self,
        chunk_id: int | None = None,
        num_chunks: int | None = None,
        fwd_only: bool = False,
        sft_mode: bool = False,
    ) -> None:
        del sft_mode
        if chunk_id != (num_chunks or 1) - 1:
            return
        # Capture the original forward route before re-entrant checkpoint
        # backward recomputes the layer.  The recompute executes the router a
        # second time and would otherwise double every popularity count used
        # for the next stage load.
        for layer in self.model_shard:
            assigned = layer.moe_layer.assigned_tokens_list
            layer.moe_layer.historical_assigned_tokens_list = list(assigned)
        self.CommScheduler.drop_execute(
            self.model_shard,
            waitEvent=self.StageCompEvent,
            recordEvent=self._StageDropEvent,
        )
        if fwd_only:
            for layer in self.model_shard:
                assigned = layer.moe_layer.assigned_tokens_list
                for index in range(len(assigned)):
                    assigned[index] = 0

    def BwdStageLoad(
        self,
        chunk_id: int | None = None,
        num_chunks: int | None = None,
        sft_mode: bool = False,
    ) -> None:
        del num_chunks, sft_mode
        if chunk_id != 0:
            return
        load_model: list[nn.Module] = []
        load_model.extend(self._get_extra_modules())
        for layer in self.model_shard:
            load_model.extend(
                (
                    layer.self_attn,
                    layer.norm1,
                    layer.norm2,
                    layer.dropout,
                    layer.gate,
                    layer.next_gate,
                    layer.moe_layer.shared_experts,
                )
            )
            hot_expert_ids = self.R_solver.solve(
                layer.moe_layer.assigned_tokens_list,
                layer_type=layer.self_attn.layer_type,
                is_first_stage=self.model_shard.is_first_stage,
                is_last_stage=self.model_shard.is_last_stage,
            )
            if not hot_expert_ids:
                hot_expert_ids = random_stageload_list(
                    len(layer.moe_layer.experts),
                    portion=self.R_solver.prefetch_portion,
                )
            for expert_id in hot_expert_ids:
                load_model.append(layer.moe_layer.experts[expert_id])
        self.CommScheduler.load_execute(
            load_model,
            waitEvent=self._StageDropEvent,
            recordEvent=self.StageLoadEvent,
        )

    def BwdStageDrop(
        self,
        chunk_id: int | None = None,
        num_chunks: int | None = None,
        sft_mode: bool = False,
    ) -> None:
        del sft_mode
        if chunk_id != (num_chunks or 1) - 1:
            return
        self.CommScheduler.drop_execute(
            self.model_shard,
            waitEvent=self.StageCompEvent,
            recordEvent=self._StageDropEvent,
        )
        for layer in self.model_shard:
            assigned = layer.moe_layer.assigned_tokens_list
            for index in range(len(assigned)):
                assigned[index] = 0


def parameter_category(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "final_norm" in name or ".norm1." in name or ".norm2." in name:
        return "norm"
    if ".self_attn." in name:
        return "token_mixer"
    if ".moe_layer.experts." in name:
        return "routed_experts"
    if ".moe_layer.shared_experts." in name:
        return "shared_expert_and_gate"
    if ".gate.weight" in name:
        return "router"
    return "other"


def categorized_parameter_counts(module: nn.Module) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        category = parameter_category(name)
        result[category] = result.get(category, 0) + parameter.numel()
    return result
