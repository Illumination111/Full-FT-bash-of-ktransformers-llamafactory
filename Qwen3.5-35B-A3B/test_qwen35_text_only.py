#!/usr/bin/env python3
"""CPU-only tests for the Qwen3.5 text-only Full/LoRA contract."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from qwen35_text_only import (
    TEXT_ARCHITECTURE,
    TEXT_FSDP_LAYER_CLASS,
    TEXT_MODEL_TYPE,
    _normalize_text_only_distributed_metadata,
    assert_text_only_model,
)


class Qwen3_5MoeDecoderLayer(torch.nn.Module):
    pass


class Qwen3_5MoeForCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type=TEXT_MODEL_TYPE,
            architectures=[TEXT_ARCHITECTURE],
        )
        self.model = torch.nn.Module()
        self.model.layer = Qwen3_5MoeDecoderLayer()
        self.weight = torch.nn.Parameter(torch.ones(1))


class PeftModelForCausalLM(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model_for_test = base_model
        self.config = base_model.config

    def get_base_model(self) -> torch.nn.Module:
        return self.base_model_for_test


class TextOnlyContractTest(unittest.TestCase):
    def test_full_accepts_direct_text_model(self) -> None:
        assert_text_only_model(Qwen3_5MoeForCausalLM(), "full")

    def test_lora_accepts_peft_wrapped_text_model(self) -> None:
        model = PeftModelForCausalLM(Qwen3_5MoeForCausalLM())
        assert_text_only_model(model, "lora")

    def test_lora_rejects_unwrapped_model(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "adapter-wrapped"):
            assert_text_only_model(Qwen3_5MoeForCausalLM(), "lora")

    def test_full_rejects_adapter_wrapper(self) -> None:
        model = PeftModelForCausalLM(Qwen3_5MoeForCausalLM())
        with self.assertRaisesRegex(RuntimeError, "unexpectedly constructed"):
            assert_text_only_model(model, "full")

    def test_fsdp_metadata_reaches_peft_base_model(self) -> None:
        base_model = Qwen3_5MoeForCausalLM()
        model = PeftModelForCausalLM(base_model)
        _normalize_text_only_distributed_metadata(model)
        self.assertEqual(model._no_split_modules, [TEXT_FSDP_LAYER_CLASS])
        self.assertEqual(base_model._no_split_modules, [TEXT_FSDP_LAYER_CLASS])
        self.assertEqual(
            base_model.model._no_split_modules,
            [TEXT_FSDP_LAYER_CLASS],
        )


if __name__ == "__main__":
    unittest.main()
