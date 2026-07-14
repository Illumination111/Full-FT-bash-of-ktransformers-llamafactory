#!/usr/bin/env python3
"""Compare stable step timings with a theoretical Qwen3-MoE FLOPs model.

The report is a roofline-style sanity check, not a hardware benchmark.  The
model deliberately separates routed-expert CPU work from non-expert GPU work
and uses configurable peak throughput/bandwidth values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASE_KEYS = {
    "forward": "forward_sec",
    "backward": "backward_sec",
    "optimizer": "optimizer_sec",
}


def _linear_lora_dims(in_features: int, out_features: int) -> int:
    """Parameter dimension multiplier for one rank-r LoRA pair."""
    return in_features + out_features


def _phase_assessment(
    phase: str,
    observed_sec: float,
    lower_bound_sec: float,
    trainable_params: int,
) -> dict[str, Any]:
    if observed_sec <= 0:
        return {
            "status": "INVALID",
            "status_cn": "计时无效",
            "reason": "observed mean time is not positive",
        }

    efficiency = lower_bound_sec / observed_sec if lower_bound_sec > 0 else 0.0
    slowdown = observed_sec / lower_bound_sec if lower_bound_sec > 0 else None
    if lower_bound_sec > 0 and observed_sec < lower_bound_sec * 0.8:
        status, status_cn = "CHECK", "需检查"
        reason = "实测快于理论下界，通常表示峰值参数、FLOPs 假设或计时范围不匹配"
    elif phase == "optimizer" and trainable_params < 20_000_000 and observed_sec <= 2.0:
        status, status_cn = "NORMAL_OVERHEAD_BOUND", "正常（固定开销主导）"
        reason = "LoRA 优化器规模较小，kernel launch、Python 和参数分组开销会主导"
    elif efficiency >= 0.10:
        status, status_cn = "NORMAL", "正常"
        reason = "达到理论 roofline 下界的 10% 以上"
    elif efficiency >= 0.01:
        status, status_cn = "LOW", "偏低"
        reason = "仅达到理论下界的 1%–10%；混合 CPU/GPU、MoE 路由碎片和通信可能解释部分差距"
    else:
        status, status_cn = "VERY_LOW", "异常偏慢"
        reason = "低于理论下界的 1%；建议检查负载不均、同步、数据搬运和实现热点"

    return {
        "status": status,
        "status_cn": status_cn,
        "reason": reason,
        "roofline_efficiency": efficiency,
        "slowdown_vs_lower_bound": slowdown,
    }


def build_flops_analysis(
    timing: dict[str, Any],
    model_config: dict[str, Any],
    *,
    mode: str,
    seq_len: int,
    batch_size: int,
    gas: int,
    num_gpus: int,
    lora_rank: int,
    gpu_bf16_tflops: float,
    cpu_bf16_tflops: float,
    gpu_memory_gbps: float,
    cpu_memory_gbps: float,
) -> dict[str, Any]:
    stable = timing.get("aggregate_stable") or {}
    if not stable:
        return {
            "status": "NO_STABLE_STEPS",
            "reason": "aggregate_stable is empty; increase max_steps or reduce warmup_skip",
            "mode": mode,
        }

    h = int(model_config["hidden_size"])
    shared_i = int(model_config.get("intermediate_size", 0) or 0)
    moe_i = int(model_config["moe_intermediate_size"])
    layers = int(model_config["num_hidden_layers"])
    experts = int(model_config["num_experts"])
    top_k = int(model_config["num_experts_per_tok"])
    heads = int(model_config["num_attention_heads"])
    kv_heads = int(model_config["num_key_value_heads"])
    head_dim = int(model_config.get("head_dim", h // heads))
    vocab = int(model_config["vocab_size"])
    tied_embeddings = bool(model_config.get("tie_word_embeddings", False))

    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    attn_matrix_params_per_layer = h * q_out + h * kv_out * 2 + q_out * h
    shared_matrix_params_per_layer = 3 * h * shared_i if shared_i else 0
    routed_matrix_params_per_expert = 3 * h * moe_i
    routed_active_params = layers * top_k * routed_matrix_params_per_expert
    routed_total_params = layers * experts * routed_matrix_params_per_expert
    router_params = layers * h * experts
    lm_head_params = h * vocab

    gpu_active_linear_params = (
        layers * (attn_matrix_params_per_layer + shared_matrix_params_per_layer)
        + router_params
        + lm_head_params
    )
    embedding_params = h * vocab
    norm_params = layers * 2 * h + h
    gpu_full_trainable_params = (
        gpu_active_linear_params
        + embedding_params
        + norm_params
        - (embedding_params if tied_embeddings else 0)
    )

    # lora_target=all: every Linear except lm_head.  Expert adapters reside in
    # the KT CPU path; attention/shared/router adapters reside on the GPU path.
    attn_lora_dims_per_layer = (
        _linear_lora_dims(h, q_out)
        + 2 * _linear_lora_dims(h, kv_out)
        + _linear_lora_dims(q_out, h)
    )
    shared_lora_dims_per_layer = 2 * _linear_lora_dims(h, shared_i) + _linear_lora_dims(shared_i, h)
    expert_lora_dims_per_expert = 2 * _linear_lora_dims(h, moe_i) + _linear_lora_dims(moe_i, h)
    router_lora_dims_per_layer = _linear_lora_dims(h, experts)
    gpu_lora_params = lora_rank * layers * (
        attn_lora_dims_per_layer + shared_lora_dims_per_layer + router_lora_dims_per_layer
    )
    cpu_lora_params = lora_rank * layers * experts * expert_lora_dims_per_expert
    active_gpu_lora_dims = layers * (
        attn_lora_dims_per_layer + shared_lora_dims_per_layer + router_lora_dims_per_layer
    )
    active_cpu_lora_dims = layers * top_k * expert_lora_dims_per_expert

    sequences_per_step = num_gpus * batch_size * gas
    tokens_per_step = sequences_per_step * seq_len

    # Multiply-accumulate counts as two FLOPs.  Attention QK^T and AV are
    # modeled densely; causal kernels can execute fewer operations, so this is
    # intentionally a conservative approximation.
    cpu_base_forward = 2.0 * tokens_per_step * routed_active_params
    gpu_linear_forward = 2.0 * tokens_per_step * gpu_active_linear_params
    gpu_attention_forward = 4.0 * layers * heads * head_dim * (seq_len**2) * sequences_per_step

    cpu_lora_forward = 0.0
    gpu_lora_forward = 0.0
    if mode == "lora":
        cpu_lora_forward = 2.0 * tokens_per_step * lora_rank * active_cpu_lora_dims
        gpu_lora_forward = 2.0 * tokens_per_step * lora_rank * active_gpu_lora_dims

    forward_cpu = cpu_base_forward + cpu_lora_forward
    forward_gpu = gpu_linear_forward + gpu_attention_forward + gpu_lora_forward

    if mode == "full":
        # For trainable matmuls, dX and dW together are approximately 2x the
        # forward GEMM work.
        backward_cpu = 2.0 * cpu_base_forward
        backward_gpu = 2.0 * (gpu_linear_forward + gpu_attention_forward)
        optimizer_cpu_params = routed_total_params
        optimizer_gpu_params = gpu_full_trainable_params
    else:
        # Frozen base matrices still compute dX (~one forward GEMM); trainable
        # LoRA A/B backward is approximated as 2x LoRA forward.
        backward_cpu = cpu_base_forward + 2.0 * cpu_lora_forward
        backward_gpu = gpu_linear_forward + 2.0 * gpu_attention_forward + 2.0 * gpu_lora_forward
        optimizer_cpu_params = cpu_lora_params
        optimizer_gpu_params = gpu_lora_params

    # AdamW arithmetic is approximately 14 scalar FLOPs/parameter.  Its time
    # is normally memory-bound, so the roofline also includes 32 B/parameter.
    adamw_flops_per_param = 14.0
    adamw_bytes_per_param = 32.0
    optimizer_cpu_flops = adamw_flops_per_param * optimizer_cpu_params
    optimizer_gpu_flops = adamw_flops_per_param * optimizer_gpu_params
    optimizer_cpu_bytes = adamw_bytes_per_param * optimizer_cpu_params
    optimizer_gpu_bytes = adamw_bytes_per_param * optimizer_gpu_params

    theoretical = {
        "forward": {"cpu_flops": forward_cpu, "gpu_flops": forward_gpu},
        "backward": {"cpu_flops": backward_cpu, "gpu_flops": backward_gpu},
        "optimizer": {"cpu_flops": optimizer_cpu_flops, "gpu_flops": optimizer_gpu_flops},
    }

    cpu_peak = cpu_bf16_tflops * 1e12
    gpu_peak = gpu_bf16_tflops * num_gpus * 1e12
    cpu_bw = cpu_memory_gbps * 1e9
    gpu_bw = gpu_memory_gbps * num_gpus * 1e9
    observed: dict[str, Any] = {}
    trainable_params = optimizer_cpu_params + optimizer_gpu_params

    for phase, timing_key in PHASE_KEYS.items():
        mean_sec = float(stable[timing_key]["mean_sec"])
        phase_flops = theoretical[phase]
        cpu_compute_lb = phase_flops["cpu_flops"] / cpu_peak if cpu_peak > 0 else 0.0
        gpu_compute_lb = phase_flops["gpu_flops"] / gpu_peak if gpu_peak > 0 else 0.0
        lower_bound = max(cpu_compute_lb, gpu_compute_lb)
        cpu_memory_lb = 0.0
        gpu_memory_lb = 0.0
        if phase == "optimizer":
            cpu_memory_lb = optimizer_cpu_bytes / cpu_bw if cpu_bw > 0 else 0.0
            gpu_memory_lb = optimizer_gpu_bytes / gpu_bw if gpu_bw > 0 else 0.0
            lower_bound = max(lower_bound, cpu_memory_lb, gpu_memory_lb)

        total_flops = phase_flops["cpu_flops"] + phase_flops["gpu_flops"]
        assessment = _phase_assessment(phase, mean_sec, lower_bound, trainable_params)
        observed[phase] = {
            "mean_sec_after_warmup": mean_sec,
            "theoretical_flops": total_flops,
            "theoretical_tflop": total_flops / 1e12,
            "effective_system_tflops": total_flops / mean_sec / 1e12 if mean_sec > 0 else None,
            "roofline_lower_bound_sec": lower_bound,
            "cpu_compute_lower_bound_sec": cpu_compute_lb,
            "gpu_compute_lower_bound_sec": gpu_compute_lb,
            "cpu_memory_lower_bound_sec": cpu_memory_lb,
            "gpu_memory_lower_bound_sec": gpu_memory_lb,
            **assessment,
        }

    for phase, values in theoretical.items():
        values["total_flops"] = values["cpu_flops"] + values["gpu_flops"]
        values["total_tflop"] = values["total_flops"] / 1e12

    return {
        "status": "OK",
        "mode": mode,
        "measurement": {
            "warmup_skip": int(timing.get("warmup_skip", 0)),
            "stable_steps": int(timing.get("num_stable_steps", 0)),
            "seq_len": seq_len,
            "per_device_batch_size": batch_size,
            "gradient_accumulation_steps": gas,
            "num_gpus": num_gpus,
            "sequences_per_optimizer_step": sequences_per_step,
            "tokens_per_optimizer_step": tokens_per_step,
        },
        "hardware_assumptions": {
            "gpu_bf16_peak_tflops_per_gpu": gpu_bf16_tflops,
            "cpu_amx_bf16_peak_tflops_total": cpu_bf16_tflops,
            "gpu_memory_bandwidth_gbps_per_gpu": gpu_memory_gbps,
            "cpu_memory_bandwidth_gbps_total": cpu_memory_gbps,
        },
        "model_counts": {
            "routed_active_matrix_params_per_token": routed_active_params,
            "routed_total_matrix_params": routed_total_params,
            "gpu_active_linear_params_per_token": gpu_active_linear_params,
            "optimizer_cpu_trainable_params": optimizer_cpu_params,
            "optimizer_gpu_trainable_params": optimizer_gpu_params,
            "optimizer_total_trainable_params": trainable_params,
            "lora_rank": lora_rank if mode == "lora" else 0,
        },
        "theoretical_flops_per_optimizer_step": theoretical,
        "observed_phase_sanity": observed,
        "notes": [
            "只读取 aggregate_stable，因此所有均值都已排除 warmup steps。",
            "理论峰值是乐观 roofline；低利用率可定位热点，但不能单独证明实现存在错误。",
            "判断阈值：下界利用率 >=10% 为正常，1%–10% 为偏低，<1% 为异常偏慢。",
            "CPU expert 与 GPU non-expert 的下界取 max，允许二者理想重叠。",
            "optimizer 同时使用 AdamW FLOPs 与每参数 32 字节访存近似；小规模 LoRA optimizer 另考虑固定开销。",
            "FLOPs 未计 embedding lookup、norm、激活函数、softmax、routing sort、通信和主机/设备拷贝。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# 理论 FLOPs 与稳定阶段耗时校验", ""]
    if report.get("status") != "OK":
        lines += [f"- 状态: `{report.get('status')}`", f"- 原因: {report.get('reason', '')}", ""]
        return "\n".join(lines)

    measurement = report["measurement"]
    hardware = report["hardware_assumptions"]
    lines += [
        f"- 微调方式: **{report['mode']}**",
        f"- 稳定区间: 去除前 {measurement['warmup_skip']} steps，统计 {measurement['stable_steps']} steps",
        f"- 每 optimizer step: {measurement['tokens_per_optimizer_step']} tokens "
        f"({measurement['num_gpus']} GPU × batch {measurement['per_device_batch_size']} × "
        f"GAS {measurement['gradient_accumulation_steps']} × seq {measurement['seq_len']})",
        f"- 峰值假设: GPU {hardware['gpu_bf16_peak_tflops_per_gpu']:.2f} TFLOPS/卡，"
        f"CPU AMX {hardware['cpu_amx_bf16_peak_tflops_total']:.2f} TFLOPS",
        "",
        "| 环节 | 理论 FLOPs/step | 去 warmup 平均耗时 | 有效吞吐 | roofline 下界 | 下界利用率 | 判断 |",
        "|------|----------------:|-------------------:|---------:|---------------:|-----------:|------|",
    ]
    labels = {"forward": "forward", "backward": "backward", "optimizer": "optimizer"}
    for phase in ("forward", "backward", "optimizer"):
        row = report["observed_phase_sanity"][phase]
        lines.append(
            f"| {labels[phase]} | {row['theoretical_tflop']:.3f} TFLOP | "
            f"{row['mean_sec_after_warmup']:.4f} s | {row['effective_system_tflops']:.3f} TFLOPS | "
            f"{row['roofline_lower_bound_sec']:.6f} s | {row['roofline_efficiency'] * 100:.3f}% | "
            f"{row['status_cn']} |"
        )
    lines += ["", "## 判断说明", ""]
    for phase in ("forward", "backward", "optimizer"):
        row = report["observed_phase_sanity"][phase]
        lines.append(f"- **{phase}**: {row['status_cn']}。{row['reason']}。")
    lines += ["", "## 口径与限制", ""]
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-json", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("full", "lora"))
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--gas", required=True, type=int)
    parser.add_argument("--gpus", required=True, type=int)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--gpu-bf16-tflops", type=float, default=82.58)
    parser.add_argument("--cpu-bf16-tflops", type=float, default=373.56)
    parser.add_argument("--gpu-memory-gbps", type=float, default=1008.0)
    parser.add_argument("--cpu-memory-gbps", type=float, default=614.4)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.timing_json.exists():
        report: dict[str, Any] = {
            "status": "MISSING_TIMING",
            "reason": f"timing file not found: {args.timing_json}",
            "mode": args.mode,
        }
    else:
        timing = json.loads(args.timing_json.read_text())
        model_config = json.loads(args.model_config.read_text())
        report = build_flops_analysis(
            timing,
            model_config,
            mode=args.mode,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            gas=args.gas,
            num_gpus=args.gpus,
            lora_rank=args.lora_rank,
            gpu_bf16_tflops=args.gpu_bf16_tflops,
            cpu_bf16_tflops=args.cpu_bf16_tflops,
            gpu_memory_gbps=args.gpu_memory_gbps,
            cpu_memory_gbps=args.cpu_memory_gbps,
        )

    json_path = args.output_dir / "flops_analysis.json"
    md_path = args.output_dir / "flops_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(report))
    print(render_markdown(report), end="")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if report.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
