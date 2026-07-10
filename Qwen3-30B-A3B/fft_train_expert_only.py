"""Accelerate entrypoint: expert-base-only freeze + probes, then LLaMA-Factory train.

Freezes all non-expert parameters so Full-FT only attempts to train
gate/up/down_proj_buf. Used to verify whether expert base grads propagate.
"""

from __future__ import annotations


def main() -> None:
    try:
        import freeze_non_expert_base

        freeze_non_expert_base.install_freeze()
    except Exception as exc:  # pragma: no cover
        print(f"[fft_train_expert_only] freeze install failed: {exc}", flush=True)
        raise

    try:
        import expert_buf_probe

        expert_buf_probe.install_probe()
    except Exception as exc:  # pragma: no cover
        print(f"[fft_train_expert_only] expert probe install skipped: {exc}", flush=True)

    try:
        import step_timing_probe

        step_timing_probe.install_step_timing()
    except Exception as exc:  # pragma: no cover
        print(f"[fft_train_expert_only] step timing install skipped: {exc}", flush=True)

    try:
        import stall_watchdog

        stall_watchdog.install_stall_watch()
    except Exception as exc:  # pragma: no cover
        print(f"[fft_train_expert_only] stall watch install skipped: {exc}", flush=True)

    from llamafactory.cli import main as lf_main

    lf_main()


if __name__ == "__main__":
    main()
