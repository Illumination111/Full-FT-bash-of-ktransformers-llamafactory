#!/usr/bin/env python3
"""Run the Qwen3.5 component-isomorphic BF16 full-update proxy on APTMoE."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_PARAMETERS = 34_660_610_688


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aptmoe-root",
        type=Path,
        default=Path("/mnt/data2/wbw/APTMoE-baseline"),
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=Path("/mnt/data2/wbw/FFTtest/APTMoE-simulate"),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--step-timing-output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--route-trace", type=Path)
    parser.add_argument("--lookup-table", type=Path)
    parser.add_argument(
        "--deployment-profile",
        choices=("server", "consumer"),
        required=True,
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--prefetch-portion", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16"], default="bf16")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--allow-synthetic-routing", action="store_true")
    parser.add_argument(
        "--allow-unprofiled-placement",
        action="store_true",
    )
    parser.add_argument(
        "--allow-linear-attention-fallback",
        action="store_true",
    )
    parser.add_argument("--save-random-weights", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "sequence_length",
        "num_gpus",
        "global_batch_size",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "steps",
    ):
        _validate_positive(name, getattr(args, name))
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup_steps must be in [0, steps)")
    if not args.audit_only and args.warmup_steps == 0:
        raise ValueError(
            "APTMoE proxy training requires at least one excluded warmup step"
        )
    _validate_positive("learning_rate", args.learning_rate)
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    if not args.text_only:
        raise ValueError("--text-only is required")
    if args.global_batch_size != (
        args.num_gpus * args.per_device_batch_size
    ):
        raise ValueError(
            "global batch must equal num_gpus * per_device_batch_size"
        )
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    required_aptmoe = (
        args.aptmoe_root / "Runtime/PipelineRuntime/pipeline_runtime.py",
        args.aptmoe_root / "Runtime/OffloadRuntime/offload.py",
        args.aptmoe_root / "data/sft_dataset.py",
    )
    missing = [str(path) for path in required_aptmoe if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"APTMoE runtime files are missing: {missing}"
        )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = completed.stdout.strip().splitlines()
    return lines[0] if lines else None


def _runtime_versions() -> dict[str, Any]:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "flash_linear_attention": _package_version(
            "flash-linear-attention"
        ),
        "causal_conv1d": _package_version("causal-conv1d"),
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": _nvidia_driver_version(),
        "gpu": gpu_name,
        "qwen35_linear_attention_fastpath": bool(
            modeling_qwen3_5_moe.is_fast_path_available
        ),
        "full_attention_implementation": "sdpa",
    }


def _git_identity(repository: Path) -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    head = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_head": head,
        "git_dirty": bool(status),
        "git_status_sha256": (
            hashlib.sha256(status.encode()).hexdigest()
            if status is not None
            else None
        ),
    }


def _expected_category_counts(
    model_path: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from qwen35_proxy_spec import build_manifest

    manifest = build_manifest(model_path, None)
    components = manifest["target"]["components"]
    expected = {
        "embedding": components["embedding"]["parameters"],
        "lm_head": components["lm_head"]["parameters"],
        "norm": components["norm"]["parameters"],
        "token_mixer": components["attention_total"]["parameters"],
        "router": components["router"]["parameters"],
        "routed_experts": components["routed_experts"]["parameters"],
        "shared_expert_and_gate": (
            components["shared_expert"]["parameters"]
            + components["shared_expert_gate"]["parameters"]
        ),
    }
    return expected, manifest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _audit_only(args: argparse.Namespace) -> None:
    expected, manifest = _expected_category_counts(args.model_path)
    print(
        json.dumps(
            {
                "benchmark_class": "deployment_proxy",
                "expected_parameters": sum(expected.values()),
                "expected_categories": expected,
                "training_state_projection": manifest["target"][
                    "training_state_projection"
                ],
                "random_weight_checkpoint_required": False,
            },
            indent=2,
        )
    )


def _configure_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for an APTMoE proxy training run")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != args.num_gpus:
        raise RuntimeError(
            f"torchrun world_size={world_size}, --num-gpus={args.num_gpus}"
        )
    torch.cuda.set_device(local_rank)
    thread_count = int(
        os.environ.get(
            "FFT_CPU_THREADS",
            os.environ.get("OMP_NUM_THREADS", "1"),
        )
    )
    torch.set_num_threads(max(1, thread_count))
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    return rank, world_size, local_rank


def _save_random_weight_shard(
    *,
    args: argparse.Namespace,
    module_list: list[Any],
    model_bytes: int,
    rank: int,
) -> None:
    if not args.save_random_weights:
        return
    from aptmoe_proxy.storage import (
        require_free_space,
        require_within_simulation_root,
    )

    output = require_within_simulation_root(
        args.output_dir,
        args.simulation_root,
    )
    if rank == 0:
        require_free_space(output, model_bytes)
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    local_state: dict[str, torch.Tensor] = {}
    for stage_id, shard in enumerate(module_list):
        if shard is None:
            continue
        for name, tensor in shard.model_shard.state_dict().items():
            local_state[f"stage_{stage_id}.{name}"] = tensor.detach().cpu()
    torch.save(
        {
            "benchmark_class": "deployment_proxy",
            "checkpoint_compatible": False,
            "rank": rank,
            "state_dict": local_state,
        },
        output / f"random_proxy_rank_{rank:02d}.pt",
    )
    dist.barrier()


def run(args: argparse.Namespace) -> None:
    aptmoe_root = args.aptmoe_root.expanduser().resolve()
    sys.path.insert(0, str(aptmoe_root))
    sys.path.insert(0, str(SCRIPT_DIR))

    from data import SFTDataLoader, load_sft_dataset, load_tokenizer
    from data.sft_dataset import tokenize_and_mask
    from qwen35_aptmoe_proxy_components import (
        load_text_config,
        require_linear_attention_fastpath,
    )
    from Runtime.OffloadRuntime import offload as offload_runtime

    from aptmoe_proxy import ProxyPlacementSolver, RouteController
    from aptmoe_proxy.runtime import (
        ProxyPipelineRuntime,
        build_proxy_pipeline,
        global_parameter_counts,
        local_parameter_counts,
        run_full_update_steps,
    )
    from aptmoe_proxy.storage import resolve_simulation_root

    rank, world_size, local_rank = _configure_distributed(args)
    simulation_root = resolve_simulation_root(args.simulation_root)
    if rank == 0:
        simulation_root.mkdir(parents=True, exist_ok=True)

    text_config = load_text_config(args.model_path)
    if not args.allow_linear_attention_fallback:
        require_linear_attention_fastpath()
    expected_categories, target_manifest = _expected_category_counts(
        args.model_path
    )
    tokens_per_microbatch = (
        args.global_batch_size * args.sequence_length
    )
    routes = RouteController(
        num_layers=text_config.num_hidden_layers,
        num_experts=text_config.num_experts,
        top_k=text_config.num_experts_per_tok,
        sequence_length=args.sequence_length,
        tokens_per_microbatch=tokens_per_microbatch,
        microbatches_per_step=args.gradient_accumulation_steps,
        expected_patterns=(
            args.warmup_steps * args.gradient_accumulation_steps
        ),
        trace_path=args.route_trace,
        allow_synthetic=args.allow_synthetic_routing,
    )
    placement = ProxyPlacementSolver(
        text_config.num_experts,
        1,
        lookup_path=args.lookup_table,
        prefetch_portion=args.prefetch_portion,
        allow_unprofiled=args.allow_unprofiled_placement,
        expected_profile=args.deployment_profile,
        required_max_tokens=tokens_per_microbatch,
    )
    offload_runtime.prefetch_portion = args.prefetch_portion

    tokenizer = load_tokenizer(str(args.model_path), trust_remote_code=True)
    raw_dataset = load_sft_dataset(
        args.dataset_name,
        str(args.dataset_dir),
        max_samples=-1,
    )
    tokenized = tokenize_and_mask(
        raw_dataset,
        tokenizer,
        args.sequence_length,
        "qwen",
    )
    data_loader = SFTDataLoader(
        tokenized_examples=tokenized,
        batch_size=args.global_batch_size,
        cutoff_len=args.sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        seed=args.seed,
        num_workers=0,
    )
    if len(data_loader) == 0:
        raise RuntimeError(
            "dataset has fewer examples than the APTMoE pipeline global batch"
        )

    module_list, _ = build_proxy_pipeline(
        config=text_config,
        world_size=world_size,
        local_rank=local_rank,
        global_rank=rank,
        routes=routes,
        placement_solver=placement,
        seed=args.seed,
    )
    actual_categories = global_parameter_counts(
        local_parameter_counts(module_list)
    )
    actual_total = sum(actual_categories.values())
    if actual_total != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"proxy parameters={actual_total:,}, expected={EXPECTED_PARAMETERS:,}"
        )
    if actual_categories != expected_categories:
        raise RuntimeError(
            "proxy parameter categories do not match target: "
            f"actual={actual_categories}, expected={expected_categories}"
        )

    runtime_config = SimpleNamespace(
        bf16=True,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        warmup_steps=0,
        warmup_ratio=0.0,
        total_training_steps=args.steps,
    )
    runtime = ProxyPipelineRuntime(
        batch_size=args.global_batch_size,
        num_chunks=1,
        seq_length=args.sequence_length,
        model_dim=text_config.hidden_size,
        hidden_dim=text_config.moe_intermediate_size,
        module_list=module_list,
        world_size=world_size,
        local_size=world_size,
        global_rank=rank,
        num_stages=text_config.num_hidden_layers,
        pipeline="APTMoE",
        fwd_only=False,
        lora_mode=False,
        data_loader=data_loader,
        config=runtime_config,
        sft_mode=True,
    )
    if runtime.total_params != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"optimizer scope={runtime.total_params:,}, "
            f"expected={EXPECTED_PARAMETERS:,}"
        )

    run_dir = args.step_timing_output_dir.resolve().parent
    if rank == 0:
        fallback_requested = any(
            (
                args.allow_synthetic_routing,
                args.allow_unprofiled_placement,
                args.allow_linear_attention_fallback,
            )
        )
        proxy_manifest = {
            "schema_version": 1,
            "benchmark_class": "deployment_proxy",
            "result_validity": (
                "smoke_only"
                if fallback_requested
                else "formal_deployment_proxy"
            ),
            "target_model": "Qwen3.5-35B-A3B-text",
            "proxy_architecture": "qwen35_component_isomorphic",
            "weight_source": "deterministic_random_initialization",
            "random_seed": args.seed,
            "checkpoint_compatible": False,
            "model_quality_metrics_allowed": False,
            "llamafactory_backend": False,
            "real_forward_backward_optimizer_update": True,
            "precision": "bf16",
            "parameter_count": actual_total,
            "parameter_categories": actual_categories,
            "route": routes.manifest(),
            "placement": placement.manifest(),
            "runtime_versions": _runtime_versions(),
            "aptmoe_root": str(aptmoe_root),
            "aptmoe_source": _git_identity(aptmoe_root),
            "simulation_root": str(simulation_root),
            "random_weights_saved": bool(args.save_random_weights),
            "target_training_state_projection": target_manifest["target"][
                "training_state_projection"
            ],
        }
        _write_json(run_dir / "proxy_manifest.json", proxy_manifest)

    tokens_per_step = (
        args.global_batch_size
        * args.sequence_length
        * args.gradient_accumulation_steps
    )
    _, verification = run_full_update_steps(
        runtime=runtime,
        module_list=module_list,
        routes=routes,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        tokens_per_step=tokens_per_step,
        timing_output_dir=args.step_timing_output_dir.resolve(),
    )
    if rank == 0:
        assert verification is not None
        _write_json(
            run_dir / "full_update_verification.json",
            verification,
        )
    _save_random_weight_shard(
        args=args,
        module_list=module_list,
        model_bytes=target_manifest["target"]["components"]["model_total"][
            "bf16_bytes"
        ],
        rank=rank,
    )
    valid_tensor = torch.tensor(
        [
            1
            if rank != 0
            or (
                verification is not None
                and verification["valid_full_update"]
            )
            else 0
        ],
        device="cuda",
        dtype=torch.int32,
    )
    dist.broadcast(valid_tensor, src=0)
    if valid_tensor.item() != 1:
        raise RuntimeError(
            "full-update verification failed; see full_update_verification.json"
        )
    dist.barrier()
    if rank == 0:
        print(
            "[aptmoe_qwen35_proxy] completed "
            f"steps={args.steps} stable={args.steps - args.warmup_steps} "
            f"timing={args.step_timing_output_dir}",
            flush=True,
        )
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.audit_only:
        _audit_only(args)
        return
    run(args)


if __name__ == "__main__":
    main()
