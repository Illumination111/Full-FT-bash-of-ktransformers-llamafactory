"""In-training probe for KT expert base-weight buffers (gate/up/down_proj_buf).

HF save_pretrained writes zero-storage expert placeholders, so post-hoc
checkpoint diffs are meaningless. This callback samples the authoritative
CPU buffers during training instead.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _find_kt_wrappers(model: torch.nn.Module):
    try:
        from kt_kernel.sft.lora import _find_kt_wrappers as _kt_find
    except Exception:
        _kt_find = None

    if _kt_find is not None:
        wrappers = _kt_find(model)
        if wrappers:
            return wrappers

    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers:
        return wrappers

    current = model
    for attr in ("module", "model", "base_model"):
        if hasattr(current, attr):
            current = getattr(current, attr)
            wrappers = getattr(current, "_kt_wrappers", None)
            if wrappers:
                return wrappers
    return None


def _buf_map(inner) -> dict[str, torch.Tensor]:
    out = {}
    if getattr(inner, "gate_proj_buf", None) is not None:
        out["gate_proj"] = inner.gate_proj_buf.data
    if getattr(inner, "up_proj_buf", None) is not None:
        out["up_proj"] = inner.up_proj_buf.data
    if getattr(inner, "down_proj_buf", None) is not None:
        out["down_proj"] = inner.down_proj_buf.data
    return out


def _tensor_stats(t: torch.Tensor) -> dict[str, float | bool | int]:
    x = t.detach().float().cpu()
    finite = bool(torch.isfinite(x).all().item())
    if not finite:
        finite_frac = float(torch.isfinite(x).float().mean().item())
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        finite_frac = 1.0
    return {
        "numel": int(x.numel()),
        "finite": finite,
        "finite_frac": finite_frac,
        "l2": float(torch.linalg.vector_norm(x).item()),
        "max_abs": float(x.abs().max().item()) if x.numel() else 0.0,
        "mean_abs": float(x.abs().mean().item()) if x.numel() else 0.0,
    }


def _delta_stats(before: torch.Tensor, after: torch.Tensor, atol: float) -> dict[str, Any]:
    b = before.detach().float().cpu()
    a = after.detach().float().cpu()
    finite_b = bool(torch.isfinite(b).all().item())
    finite_a = bool(torch.isfinite(a).all().item())
    if not (finite_b and finite_a):
        return {
            "status": "NONFINITE",
            "finite_before": finite_b,
            "finite_after": finite_a,
            "changed": False,
            "max_abs_delta": float("nan"),
            "mean_abs_delta": float("nan"),
            "rel_l2_delta": float("nan"),
            "changed_element_fraction": 0.0,
        }

    delta = a - b
    abs_delta = delta.abs()
    old_norm = float(torch.linalg.vector_norm(b).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    max_abs = float(abs_delta.max().item())
    mean_abs = float(abs_delta.mean().item())
    changed_elems = int((abs_delta > atol).sum().item())
    return {
        "status": "OK",
        "finite_before": True,
        "finite_after": True,
        "changed": changed_elems > 0 and max_abs > atol,
        "max_abs_delta": max_abs,
        "mean_abs_delta": mean_abs,
        "rel_l2_delta": delta_norm / max(old_norm, 1e-30),
        "changed_element_fraction": changed_elems / max(abs_delta.numel(), 1),
        "base_l2_norm": old_norm,
        "delta_l2_norm": delta_norm,
        "after_l2_norm": float(torch.linalg.vector_norm(a).item()),
    }


from transformers import TrainerCallback


class ExpertBufProbeCallback(TrainerCallback):
    """Sample KT expert base-weight buffers during training."""

    def __init__(self) -> None:
        super().__init__()
        self.out_path = Path(os.environ.get("KT_EXPERT_BUF_PROBE_OUT", "expert_buf_probe.json"))
        self.sample_n = _env_int("KT_EXPERT_BUF_PROBE_SAMPLES", 12)
        self.seed = _env_int("KT_EXPERT_BUF_PROBE_SEED", 20260709)
        self.atol = _env_float("KT_EXPERT_BUF_PROBE_ATOL", 0.0)
        self.baseline: dict[str, torch.Tensor] = {}
        self.targets: list[tuple[int, int, str]] = []
        self.meta: dict[str, Any] = {
            "sample_n": self.sample_n,
            "seed": self.seed,
            "atol": self.atol,
            "out_path": str(self.out_path),
        }

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        wrappers = _find_kt_wrappers(model)
        if not wrappers:
            self._emit(
                {
                    "status": "ERROR",
                    "reason": "no KT wrappers found on trainer model",
                    "sampled_tensors": 0,
                    "changed_tensors": 0,
                }
            )
            return

        candidates: list[tuple[int, int, str]] = []
        for w in wrappers:
            if not getattr(w, "_full_weight_grad", False) or w.wrapper is None:
                continue
            layer_idx = int(w.layer_idx)
            for proj_name, buf in _buf_map(w.wrapper).items():
                e = int(buf.shape[0])
                for expert_idx in range(e):
                    candidates.append((layer_idx, expert_idx, proj_name))

        if not candidates:
            self._emit(
                {
                    "status": "ERROR",
                    "reason": "no full_weight_grad expert buffers found (gate/up/down_proj_buf)",
                    "sampled_tensors": 0,
                    "changed_tensors": 0,
                }
            )
            return

        rng = random.Random(self.seed)
        self.targets = rng.sample(candidates, min(self.sample_n, len(candidates)))
        wrapper_by_layer = {int(w.layer_idx): w for w in wrappers if w.wrapper is not None}

        for layer_idx, expert_idx, proj_name in self.targets:
            w = wrapper_by_layer[layer_idx]
            buf = _buf_map(w.wrapper)[proj_name]
            key = f"layer{layer_idx}.expert{expert_idx}.{proj_name}"
            # Clone one expert slice only — lightweight.
            self.baseline[key] = buf[expert_idx].detach().to("cpu", dtype=torch.float32).clone()

        self.meta.update(
            {
                "candidate_tensors": len(candidates),
                "sampled_tensors": len(self.targets),
                "num_kt_wrappers": len(wrappers),
                "baseline_step": int(getattr(state, "global_step", 0) or 0),
            }
        )
        print(
            f"[expert_buf_probe] baseline captured: {len(self.targets)} expert slices "
            f"(from {len(candidates)} candidates)",
            flush=True,
        )

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.baseline or model is None:
            if not self.out_path.exists():
                self._emit(
                    {
                        "status": "ERROR",
                        "reason": "probe ended without baseline snapshot",
                        "sampled_tensors": 0,
                        "changed_tensors": 0,
                    }
                )
            return

        wrappers = _find_kt_wrappers(model) or []
        wrapper_by_layer = {int(w.layer_idx): w for w in wrappers if w.wrapper is not None}

        records = []
        for layer_idx, expert_idx, proj_name in self.targets:
            key = f"layer{layer_idx}.expert{expert_idx}.{proj_name}"
            w = wrapper_by_layer.get(layer_idx)
            if w is None:
                records.append({"tensor": key, "status": "ERROR", "reason": "wrapper missing at train_end"})
                continue
            buf = _buf_map(w.wrapper).get(proj_name)
            if buf is None:
                records.append({"tensor": key, "status": "ERROR", "reason": f"{proj_name}_buf missing"})
                continue
            after = buf[expert_idx].detach().to("cpu", dtype=torch.float32).clone()
            before = self.baseline[key]
            stats = _delta_stats(before, after, self.atol)
            before_stats = _tensor_stats(before)
            after_stats = _tensor_stats(after)
            records.append(
                {
                    "tensor": key,
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx,
                    "proj": proj_name,
                    "shape": list(after.shape),
                    "before": before_stats,
                    "after": after_stats,
                    **stats,
                }
            )

        ok_records = [r for r in records if r.get("status") == "OK"]
        nonfinite = [r for r in records if r.get("status") == "NONFINITE"]
        changed = [r for r in ok_records if r.get("changed")]

        if nonfinite:
            status = "FAIL_NUMERIC"
            reason = f"{len(nonfinite)} sampled expert buffers became non-finite during training"
        elif not ok_records:
            status = "ERROR"
            reason = "all sampled tensors failed to compare"
        elif not changed:
            status = "FAIL"
            reason = "none of the sampled expert base buffers changed beyond atol"
        elif len(changed) == len(ok_records):
            status = "PASS"
            reason = "all successfully sampled expert base buffers changed and stayed finite"
        else:
            status = "PARTIAL"
            reason = (
                "some sampled expert buffers changed; short runs / routing may miss some experts"
            )

        if ok_records:
            aggregate = {
                "changed_tensor_fraction": len(changed) / len(ok_records),
                "mean_rel_l2_delta": sum(r["rel_l2_delta"] for r in ok_records) / len(ok_records),
                "max_rel_l2_delta": max(r["rel_l2_delta"] for r in ok_records),
                "mean_max_abs_delta": sum(r["max_abs_delta"] for r in ok_records) / len(ok_records),
                "max_abs_delta": max(r["max_abs_delta"] for r in ok_records),
            }
        else:
            aggregate = {
                "changed_tensor_fraction": 0.0,
                "mean_rel_l2_delta": 0.0,
                "max_rel_l2_delta": 0.0,
                "mean_max_abs_delta": 0.0,
                "max_abs_delta": 0.0,
            }

        result = {
            "status": status,
            "reason": reason,
            "method": "in_training_proj_buf_sample",
            "note": (
                "Compares KT gate/up/down_proj_buf slices at train_begin vs train_end. "
                "Does not use HF checkpoint expert weights (those are zero-storage placeholders)."
            ),
            "meta": {
                **self.meta,
                "final_step": int(getattr(state, "global_step", 0) or 0),
            },
            "sampled_tensors": len(records),
            "changed_tensors": len(changed),
            "nonfinite_tensors": len(nonfinite),
            "aggregate": aggregate,
            "records": records,
        }
        self._emit(result)
        print(
            f"[expert_buf_probe] done status={status} changed={len(changed)}/{len(ok_records)} "
            f"nonfinite={len(nonfinite)} -> {self.out_path}",
            flush=True,
        )

    def _emit(self, obj: dict[str, Any]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def install_probe() -> None:
    """Monkey-patch transformers.Trainer to attach ExpertBufProbeCallback."""
    if os.environ.get("KT_EXPERT_BUF_PROBE", "0") not in ("1", "true", "TRUE", "yes", "YES"):
        return

    from transformers import Trainer

    if getattr(Trainer, "_kt_expert_buf_probe_installed", False):
        return

    original_init = Trainer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            self.add_callback(ExpertBufProbeCallback())
            print("[expert_buf_probe] callback installed on Trainer", flush=True)
        except Exception as exc:
            print(f"[expert_buf_probe] failed to install callback: {exc}", flush=True)

    Trainer.__init__ = patched_init
    Trainer._kt_expert_buf_probe_installed = True
