#!/usr/bin/env python3
"""Reuse the FFT monitor implementation shared with the Qwen3 tests."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[1] / "Qwen3-30B-A3B" / "monitor.py"
runpy.run_path(str(TARGET), run_name="__main__")
