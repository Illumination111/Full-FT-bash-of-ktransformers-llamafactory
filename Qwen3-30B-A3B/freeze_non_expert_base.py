"""Freeze everything except KT expert base weights (gate/up/down_proj_buf).

Ablation for Full-FT: if only expert base is trainable and loss stays flat,
that supports the diagnosis that base-weight grads are not written (e.g.
full_weight_grad lost in TP object slicing).
"""

from __future__ import annotations

import os
from typing import Any


def _enabled() -> bool:
    return os.environ.get("KT_FREEZE_NON_EXPERT_BASE", "0") in ("1", "true", "TRUE", "yes", "YES")


def _unwrap(model):
    try:
        from accelerate.utils import extract_model_from_parallel

        return extract_model_from_parallel(model)
    except Exception:
        return model


def _collect_expert_base(model) -> list:
    from kt_kernel.sft.lora import _collect_kt_full_weight_params, _find_kt_wrappers

    wrappers = _find_kt_wrappers(model) or []
    params = _collect_kt_full_weight_params(wrappers)
    for p in params:
        p.requires_grad_(True)
    return params


def _collect_kt_lora(model) -> list:
    from kt_kernel.sft.lora import _collect_kt_lora_params, _find_kt_wrappers

    wrappers = _find_kt_wrappers(model) or []
    return _collect_kt_lora_params(wrappers)


def _freeze_all_named(model) -> tuple[int, int]:
    n_params, n_elems = 0, 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            n_params += 1
            n_elems += int(p.numel())
        p.requires_grad_(False)
    return n_params, n_elems


def _freeze_list(params) -> tuple[int, int]:
    n_params, n_elems = 0, 0
    for p in params:
        if p.requires_grad:
            n_params += 1
            n_elems += int(p.numel())
        p.requires_grad_(False)
    return n_params, n_elems


def _set_optimizer_params(optimizer, params: list) -> dict[str, int]:
    """Replace optimizer contents with exactly `params` (in last group)."""
    old_n = sum(len(g["params"]) for g in optimizer.param_groups)
    if not optimizer.param_groups:
        optimizer.add_param_group({"params": list(params)})
    else:
        # Keep group metadata; put all kept params in the last group.
        for g in optimizer.param_groups[:-1]:
            g["params"] = []
        optimizer.param_groups[-1]["params"] = list(params)
    return {"optimizer_old": old_n, "optimizer_new": len(params)}


def apply_freeze(model, optimizer=None) -> dict[str, Any]:
    model = _unwrap(model)
    named_n, named_e = _freeze_all_named(model)

    lora_n, lora_e = 0, 0
    try:
        lora_n, lora_e = _freeze_list(_collect_kt_lora(model))
    except Exception as exc:
        print(f"[freeze_non_expert_base] LoRA freeze skipped: {exc}", flush=True)

    expert = []
    try:
        expert = _collect_expert_base(model)
    except Exception as exc:
        print(f"[freeze_non_expert_base] expert base collect failed: {exc}", flush=True)

    stats: dict[str, Any] = {
        "named_frozen_params": named_n,
        "named_frozen_elems": named_e,
        "lora_frozen_params": lora_n,
        "lora_frozen_elems": lora_e,
        "expert_base_params": len(expert),
        "expert_base_elems": int(sum(p.numel() for p in expert)),
        "named_trainable_after": sum(1 for _, p in model.named_parameters() if p.requires_grad),
    }
    if optimizer is not None:
        stats.update(_set_optimizer_params(optimizer, expert))
    return stats


def _log(tag: str, stats: dict[str, Any]) -> None:
    print(
        f"[freeze_non_expert_base@{tag}] "
        f"expert_base={stats.get('expert_base_params')} "
        f"(elems={stats.get('expert_base_elems')}) | "
        f"froze_named={stats.get('named_frozen_params')} "
        f"froze_lora={stats.get('lora_frozen_params')} | "
        f"opt {stats.get('optimizer_old', '-')}->{stats.get('optimizer_new', '-')} | "
        f"named_trainable_after={stats.get('named_trainable_after')}",
        flush=True,
    )
    if stats.get("expert_base_params", 0) <= 0 and tag == "train_begin":
        print(
            "[freeze_non_expert_base] WARNING: no expert base params in optimizer; "
            "a flat loss would be inconclusive",
            flush=True,
        )


def install_freeze() -> None:
    if not _enabled():
        return

    from transformers import Trainer, TrainerCallback

    if getattr(Trainer, "_kt_freeze_non_expert_base_installed", False):
        return

    # --- Restrict KT inject to expert base only (exclude LoRA) ---
    try:
        import kt_kernel.sft.lora as kt_lora

        def get_kt_params_expert_base_only(model):
            wrappers = kt_lora._find_kt_wrappers(model)
            if not wrappers:
                return []
            if not any(getattr(w, "_full_weight_grad", False) for w in wrappers):
                print(
                    "[freeze_non_expert_base] get_kt_lora_params -> [] "
                    "(full_weight_grad not enabled)",
                    flush=True,
                )
                return []
            params = kt_lora._collect_kt_full_weight_params(wrappers)
            for p in params:
                p.requires_grad_(True)
            print(
                f"[freeze_non_expert_base] inject {len(params)} expert base params only",
                flush=True,
            )
            return params

        kt_lora.get_kt_lora_params = get_kt_params_expert_base_only
        try:
            import transformers.trainer as tr_trainer

            if getattr(tr_trainer, "get_kt_lora_params", None) is not None:
                tr_trainer.get_kt_lora_params = get_kt_params_expert_base_only
        except Exception:
            pass
    except Exception as exc:
        print(f"[freeze_non_expert_base] patch get_kt_lora_params failed: {exc}", flush=True)

    # --- After create_optimizer: freeze GPU/named params, clear optimizer ---
    orig_create_optimizer = Trainer.create_optimizer

    def create_optimizer_frozen(self, *args, **kwargs):
        opt = orig_create_optimizer(self, *args, **kwargs)
        model = _unwrap(self.model)
        stats = apply_freeze(model, optimizer=self.optimizer)
        _log("create_optimizer", stats)
        return opt

    Trainer.create_optimizer = create_optimizer_frozen

    # --- At train begin: re-freeze LoRA, keep only expert base in optimizer ---
    class ExpertBaseOnlyCallback(TrainerCallback):
        def __init__(self, trainer_ref_holder: dict):
            super().__init__()
            self._holder = trainer_ref_holder

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            trainer = self._holder.get("trainer")
            if trainer is None:
                print("[freeze_non_expert_base] train_begin: no trainer ref", flush=True)
                return
            model_u = _unwrap(trainer.model)
            stats = apply_freeze(model_u, optimizer=trainer.optimizer)
            _log("train_begin", stats)

    holder: dict[str, Any] = {}
    orig_init = Trainer.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        holder["trainer"] = self
        self.add_callback(ExpertBaseOnlyCallback(holder))
        print("[freeze_non_expert_base] callback installed", flush=True)

    Trainer.__init__ = patched_init
    Trainer._kt_freeze_non_expert_base_installed = True
    print("[freeze_non_expert_base] install complete", flush=True)
