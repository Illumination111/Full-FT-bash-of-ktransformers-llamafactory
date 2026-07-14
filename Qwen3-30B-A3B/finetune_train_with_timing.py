"""Accelerate entrypoint for lightweight Full/LoRA performance tests.

Only per-step timing is enabled. Expert weight/gradient probes and the stall
watchdog are intentionally excluded so they cannot distort throughput.
"""

from __future__ import annotations


def main() -> None:
    try:
        import step_timing_probe

        step_timing_probe.install_step_timing()
    except Exception as exc:  # pragma: no cover
        print(f"[finetune_train_with_timing] step timing install failed: {exc}", flush=True)
        raise

    from llamafactory.cli import main as lf_main

    lf_main()


if __name__ == "__main__":
    main()
