#!/usr/bin/env python3
"""Profile Qwen3.5 6 MiB experts and token mixers for APTMoE placement."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from aptmoe_proxy.storage import (
    DEFAULT_SIMULATION_ROOT,
    require_within_simulation_root,
)
from qwen35_aptmoe_proxy_components import (
    Qwen35Router,
    Qwen35RoutedExpert,
    Qwen35SharedExpert,
    Qwen35TokenMixer,
    load_text_config,
    require_linear_attention_fastpath,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_driver_version() -> str | None:
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


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def time_module_transfer(
    module: torch.nn.Module,
    target: str,
    iterations: int,
) -> float:
    values: list[float] = []
    source = "cpu" if target == "cuda" else "cuda"
    for _ in range(iterations):
        module.to(source)
        torch.cuda.synchronize()
        started = time.perf_counter()
        module.to(target, non_blocking=True)
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
    return median(values)


def time_cpu_expert(
    expert: Qwen35RoutedExpert,
    tokens: int,
    hidden_size: int,
    iterations: int,
) -> tuple[float, float]:
    if tokens == 0:
        return 0.0, 0.0
    forward_values: list[float] = []
    full_values: list[float] = []
    for iteration in range(iterations + 1):
        expert.zero_grad(set_to_none=True)
        hidden = torch.randn(
            tokens,
            hidden_size,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        started = time.perf_counter()
        output = expert(hidden)
        forward_elapsed = time.perf_counter() - started
        output.float().square().mean().backward()
        full_elapsed = time.perf_counter() - started
        if iteration > 0:
            forward_values.append(forward_elapsed)
            full_values.append(full_elapsed)
    return median(forward_values), median(full_values)


def interpolate_curve(
    samples: dict[int, float],
    max_tokens: int,
) -> list[float]:
    points = sorted(samples)
    result = [0.0] * (max_tokens + 1)
    for token_count in range(1, max_tokens + 1):
        right = next(
            (point for point in points if point >= token_count),
            points[-1],
        )
        left = max(point for point in points if point <= token_count)
        if left == right:
            result[token_count] = samples[left]
        else:
            fraction = (token_count - left) / (right - left)
            result[token_count] = (
                samples[left]
                + fraction * (samples[right] - samples[left])
            )
    return result


def time_token_mixer(
    mixer: Qwen35TokenMixer,
    sequence_length: int,
    hidden_size: int,
    iterations: int,
) -> float:
    mixer.to(device="cuda", dtype=torch.bfloat16)
    values: list[float] = []
    for iteration in range(iterations + 1):
        mixer.zero_grad(set_to_none=True)
        hidden = torch.randn(
            1,
            sequence_length,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = mixer(hidden)
        output.float().square().mean().backward()
        end.record()
        end.synchronize()
        if iteration > 0:
            values.append(start.elapsed_time(end) / 1000.0)
    mixer.to("cpu")
    return median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/data3/models/Qwen3.5-35B-A3B"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deployment-profile",
        choices=("server", "consumer"),
        required=True,
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))),
    )
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=DEFAULT_SIMULATION_ROOT,
    )
    args = parser.parse_args()
    if min(
        args.sequence_length,
        args.max_tokens,
        args.iterations,
        args.cpu_threads,
    ) <= 0:
        raise SystemExit("profile sizes, iterations, and threads must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to profile APTMoE transfers and attention")
    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    require_linear_attention_fastpath()
    output = require_within_simulation_root(
        args.output,
        args.simulation_root,
    )
    config = load_text_config(args.model_path)

    expert = Qwen35RoutedExpert(
        config.hidden_size,
        config.moe_intermediate_size,
        layer_id=0,
        expert_id=0,
        device="cpu",
        dtype=torch.bfloat16,
    )
    expert_h2d = time_module_transfer(expert, "cuda", args.iterations)
    expert_d2h = time_module_transfer(expert, "cpu", args.iterations)
    expert.to("cpu")

    sample_points = {0, 1, args.max_tokens}
    exponent = 1
    while 2**exponent < args.max_tokens:
        sample_points.add(2**exponent)
        exponent += 1
    cpu_sample_pairs = {
        point: time_cpu_expert(
            expert,
            point,
            config.hidden_size,
            args.iterations,
        )
        for point in sorted(sample_points)
    }
    cpu_forward_samples = {
        point: values[0] for point, values in cpu_sample_pairs.items()
    }
    cpu_full_samples = {
        point: values[1] for point, values in cpu_sample_pairs.items()
    }
    cpu_forward_curve = interpolate_curve(
        cpu_forward_samples,
        args.max_tokens,
    )
    cpu_full_curve = interpolate_curve(
        cpu_full_samples,
        args.max_tokens,
    )

    router = Qwen35Router(
        config.hidden_size,
        config.num_experts,
        config.num_experts_per_tok,
        device="cpu",
        dtype=torch.bfloat16,
    )
    torch.nn.init.normal_(
        router.weight,
        mean=0.0,
        std=config.initializer_range,
    )
    router_h2d = time_module_transfer(router, "cuda", args.iterations)
    router.to("cpu")
    shared = Qwen35SharedExpert(
        config.hidden_size,
        config.shared_expert_intermediate_size,
        device="cpu",
        dtype=torch.bfloat16,
    )
    shared_h2d = time_module_transfer(shared, "cuda", args.iterations)
    shared.to("cpu")
    norms = torch.nn.ModuleList(
        [
            Qwen3_5MoeRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            ),
            Qwen3_5MoeRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            ),
        ]
    ).to(device="cpu", dtype=torch.bfloat16)
    norms_h2d = time_module_transfer(norms, "cuda", args.iterations)
    norms.to("cpu")

    embedding = torch.nn.Embedding(
        config.vocab_size,
        config.hidden_size,
        device="cpu",
        dtype=torch.bfloat16,
    )
    embedding_h2d = time_module_transfer(
        embedding,
        "cuda",
        args.iterations,
    )
    embedding.to("cpu")
    del embedding
    final_norm = Qwen3_5MoeRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
    ).to(device="cpu", dtype=torch.bfloat16)
    final_norm_h2d = time_module_transfer(
        final_norm,
        "cuda",
        args.iterations,
    )
    final_norm.to("cpu")
    del final_norm
    lm_head = torch.nn.Linear(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        device="cpu",
        dtype=torch.bfloat16,
    )
    lm_head_h2d = time_module_transfer(
        lm_head,
        "cuda",
        args.iterations,
    )
    lm_head.to("cpu")
    del lm_head

    mixer_profiles: dict[str, dict[str, float | int]] = {}
    non_mixer_load_seconds = router_h2d + shared_h2d + norms_h2d
    for layer_idx in (0, 3):
        mixer = Qwen35TokenMixer(config, layer_idx).to(
            device="cpu",
            dtype=torch.bfloat16,
        )
        h2d = time_module_transfer(mixer, "cuda", args.iterations)
        mixer.to("cpu")
        fwd_bwd = time_token_mixer(
            mixer,
            args.sequence_length,
            config.hidden_size,
            args.iterations,
        )
        mixer_profiles[mixer.layer_type] = {
            "representative_layer": layer_idx,
            "sequence_length": args.sequence_length,
            "h2d_seconds": h2d,
            "forward_backward_seconds": fwd_bwd,
        }
    control_load_seconds = max(
        non_mixer_load_seconds
        + float(profile["h2d_seconds"])
        for profile in mixer_profiles.values()
    )
    control_load_seconds = max(
        control_load_seconds,
        non_mixer_load_seconds
        + float(mixer_profiles["linear_attention"]["h2d_seconds"])
        + embedding_h2d,
        non_mixer_load_seconds
        + float(mixer_profiles["full_attention"]["h2d_seconds"])
        + final_norm_h2d
        + lm_head_h2d,
    )

    lookup = {
        "schema_version": 1,
        "benchmark_class": "aptmoe_qwen35_proxy_lookup",
        "deployment_profile": args.deployment_profile,
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_driver": nvidia_driver_version(),
            "transformers": package_version("transformers"),
            "flash_linear_attention": package_version(
                "flash-linear-attention"
            ),
            "causal_conv1d": package_version("causal-conv1d"),
            "cpu_threads": args.cpu_threads,
            "cpu_affinity_count": len(os.sched_getaffinity(0)),
        },
        "expert": {
            "num_experts": config.num_experts,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.moe_intermediate_size,
            "bf16_bytes": 6 * (1 << 20),
            "transfer_timing": "host_wall_module_to_plus_cuda_synchronize",
            "h2d_seconds": expert_h2d,
            "d2h_seconds": expert_d2h,
        },
        "control_plane": {
            "router_h2d_seconds": router_h2d,
            "shared_expert_h2d_seconds": shared_h2d,
            "two_norms_h2d_seconds": norms_h2d,
            "non_mixer_load_seconds": non_mixer_load_seconds,
            "load_seconds": control_load_seconds,
        },
        "extra_modules": {
            "embedding_h2d_seconds": embedding_h2d,
            "final_norm_h2d_seconds": final_norm_h2d,
            "lm_head_h2d_seconds": lm_head_h2d,
        },
        "cpu_expert": {
            "max_tokens": args.max_tokens,
            "sample_forward_seconds": {
                str(key): value
                for key, value in cpu_forward_samples.items()
            },
            "sample_forward_backward_seconds": {
                str(key): value
                for key, value in cpu_full_samples.items()
            },
            "forward_seconds_by_tokens": cpu_forward_curve,
            "forward_backward_seconds_by_tokens": cpu_full_curve,
        },
        "token_mixers": mixer_profiles,
    }
    if not math.isfinite(control_load_seconds) or control_load_seconds <= 0:
        raise SystemExit("invalid lookup timing")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(lookup, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[aptmoe_lookup] -> {output}")


if __name__ == "__main__":
    main()
