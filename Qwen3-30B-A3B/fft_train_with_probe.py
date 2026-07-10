"""Accelerate entrypoint: install probes, then run LLaMA-Factory train.

Usage (via accelerate):
  PYTHONPATH=<this_dir>:$PYTHONPATH \\
  accelerate launch ... -m fft_train_with_probe train <config.yaml>
"""

from __future__ import annotations


def main() -> None:
    try:
        import expert_buf_probe

        expert_buf_probe.install_probe()
    except Exception as exc:  # pragma: no cover
        print(f"[fft_train_with_probe] expert probe install skipped: {exc}", flush=True)

    try:
        import step_timing_probe

        step_timing_probe.install_step_timing()
    except Exception as exc:  # pragma: no cover
        print(f"[fft_train_with_probe] step timing install skipped: {exc}", flush=True)

    try:
        import stall_watchdog

        stall_watchdog.install_stall_watch()
    except Exception as exc:  # pragma: no cover
        print(f"[fft_train_with_probe] stall watch install skipped: {exc}", flush=True)

    from llamafactory.cli import main as lf_main

    lf_main()


if __name__ == "__main__":
    main()
