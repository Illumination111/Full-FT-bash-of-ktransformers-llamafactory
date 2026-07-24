#!/usr/bin/env python3
"""CPU-only contract tests for the Qwen3.5 APTMoE deployment proxy."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from aggregate_sweep_results import aggregate_run
from aptmoe_proxy.placement import (
    EXPECTED_EXPERT_BF16_BYTES,
    ProxyPlacementSolver,
)
from aptmoe_proxy.routes import RouteController
from aptmoe_proxy.storage import require_within_simulation_root
from qwen35_aptmoe_proxy_components import (
    Qwen35RoutedExpert,
    component_parameter_counts,
    load_text_config,
)
from qwen35_route_capture import RouteTraceCapture


MODEL_PATH = Path("/mnt/data3/models/Qwen3.5-35B-A3B")


class ComponentContractTest(unittest.TestCase):
    @unittest.skipUnless(MODEL_PATH.is_dir(), "local Qwen3.5 config unavailable")
    def test_representative_layer_parameter_counts(self) -> None:
        config = load_text_config(MODEL_PATH)
        linear = component_parameter_counts(config, 0)
        full = component_parameter_counts(config, 3)
        self.assertEqual(linear["token_mixer"], 33_718_464)
        self.assertEqual(full["token_mixer"], 27_263_488)
        for counts in (linear, full):
            self.assertEqual(counts["router"], 524_288)
            self.assertEqual(counts["routed_experts"], 805_306_368)
            self.assertEqual(counts["shared_expert_and_gate"], 3_147_776)
            self.assertEqual(counts["norms"], 4_096)

    def test_routed_expert_uses_target_fused_tensor_layout(self) -> None:
        expert = Qwen35RoutedExpert(
            2048,
            512,
            layer_id=0,
            expert_id=0,
            device="meta",
            dtype=torch.bfloat16,
        )
        shapes = {
            name: tuple(parameter.shape)
            for name, parameter in expert.named_parameters()
        }
        self.assertEqual(
            shapes,
            {
                "gate_up_proj.weight": (1024, 2048),
                "down_proj.weight": (2048, 512),
            },
        )


class RouteContractTest(unittest.TestCase):
    def _write_trace(self, path: Path, source: str) -> None:
        routes = np.empty((2, 3, 2), dtype=np.int16)
        routes[0] = [[0, 1], [1, 2], [2, 3]]
        routes[1] = [[3, 2], [2, 1], [1, 0]]
        metadata = {
            "schema_version": 1,
            "source": source,
            "source_backend": "kt",
            "sequence_length": 3,
            "global_batch_size": 1,
            "patterns": 1,
            "layers": 2,
            "tokens": 3,
            "top_k": 2,
        }
        np.savez_compressed(
            path,
            topk_indices=routes[None, ...],
            metadata_json=np.asarray(json.dumps(metadata)),
        )

    def test_exact_trace_replay_preserves_router_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.npz"
            self._write_trace(trace, "merged_exact_qwen35_router_trace")
            controller = RouteController(
                num_layers=2,
                num_experts=4,
                top_k=2,
                sequence_length=3,
                tokens_per_microbatch=3,
                trace_path=trace,
                allow_synthetic=False,
            )
            logits = torch.randn(3, 4, requires_grad=True)
            scores, indices, counts = controller.select(
                layer_idx=0,
                logits=logits,
            )
            (scores[:, 0] * torch.arange(1, 4)).sum().backward()
            self.assertEqual(indices.tolist(), [[0, 1], [1, 2], [2, 3]])
            self.assertEqual(counts, [1, 2, 2, 1])
            self.assertIsNotNone(logits.grad)
            self.assertGreater(int(torch.count_nonzero(logits.grad)), 0)

    def test_synthetic_trace_requires_explicit_smoke_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.npz"
            self._write_trace(trace, "synthetic_zipf_smoke_only")
            arguments = {
                "num_layers": 2,
                "num_experts": 4,
                "top_k": 2,
                "sequence_length": 3,
                "tokens_per_microbatch": 3,
                "trace_path": trace,
            }
            with self.assertRaisesRegex(ValueError, "formal APTMoE"):
                RouteController(**arguments, allow_synthetic=False)
            controller = RouteController(
                **arguments,
                allow_synthetic=True,
            )
            self.assertEqual(controller.mode, "synthetic_trace_smoke_only")

    def test_multi_pattern_trace_cycles_by_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.npz"
            first = np.empty((2, 3, 2), dtype=np.int16)
            first[0] = [[0, 1], [1, 2], [2, 3]]
            first[1] = [[3, 2], [2, 1], [1, 0]]
            second = np.flip(first, axis=1).copy()
            metadata = {
                "schema_version": 1,
                "source": "merged_exact_qwen35_router_trace",
                "source_backend": "deepspeed",
                "sequence_length": 3,
                "global_batch_size": 1,
                "patterns": 2,
                "layers": 2,
                "tokens": 3,
                "top_k": 2,
            }
            np.savez_compressed(
                trace,
                topk_indices=np.stack((first, second), axis=0),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            controller = RouteController(
                num_layers=2,
                num_experts=4,
                top_k=2,
                sequence_length=3,
                tokens_per_microbatch=3,
                microbatches_per_step=2,
                trace_path=trace,
                allow_synthetic=False,
            )
            logits = torch.randn(3, 4)
            controller.set_position(step=0, microbatch=0)
            _, step_zero, _ = controller.select(layer_idx=0, logits=logits)
            controller.set_position(step=0, microbatch=1)
            _, step_one, _ = controller.select(layer_idx=0, logits=logits)
            controller.set_position(step=1, microbatch=0)
            _, step_two, _ = controller.select(layer_idx=0, logits=logits)
            self.assertEqual(step_zero.tolist(), first[0].tolist())
            self.assertEqual(step_one.tolist(), second[0].tolist())
            self.assertEqual(step_two.tolist(), first[0].tolist())
            with self.assertRaisesRegex(
                ValueError,
                r"expected warmup_steps\*GAS=1",
            ):
                RouteController(
                    num_layers=2,
                    num_experts=4,
                    top_k=2,
                    sequence_length=3,
                    tokens_per_microbatch=3,
                    expected_patterns=1,
                    trace_path=trace,
                    allow_synthetic=False,
                )

    def test_capture_writes_all_warmup_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {"FFT_APTMOE_SIMULATION_ROOT": str(root)},
            ):
                capture = RouteTraceCapture(
                    output_dir=root / "routes",
                    sequence_length=3,
                    max_patterns=2,
                    expected_layers=2,
                    top_k=2,
                )
            for offset in (0, 1):
                capture.begin_model_forward(None, ())
                for layer_idx in range(2):
                    selected = torch.tensor(
                        [
                            [offset, 2],
                            [1, 3],
                            [2, offset],
                        ],
                        dtype=torch.long,
                    )
                    capture.hook(layer_idx)(
                        None,
                        (),
                        (None, None, selected),
                    )
                capture.end_model_forward(None, (), None)
            capture.write()
            with np.load(
                root / "routes" / "rank_00.npz",
                allow_pickle=False,
            ) as data:
                routes = np.asarray(data["topk_indices"])
                metadata = json.loads(str(data["metadata_json"].item()))
            self.assertEqual(routes.shape, (2, 2, 3, 2))
            self.assertEqual(metadata["patterns"], 2)


class PlacementAndStorageTest(unittest.TestCase):
    def test_profiled_solver_uses_aptmoe_compute_load_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lookup = Path(directory) / "lookup.json"
            lookup.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_class": "aptmoe_qwen35_proxy_lookup",
                        "expert": {
                            "bf16_bytes": EXPECTED_EXPERT_BF16_BYTES,
                            "num_experts": 4,
                            "h2d_seconds": 1.0,
                        },
                        "control_plane": {
                            "load_seconds": 1.0,
                            "non_mixer_load_seconds": 0.5,
                        },
                        "token_mixers": {
                            "linear_attention": {"h2d_seconds": 0.5},
                            "full_attention": {"h2d_seconds": 0.5},
                        },
                        "extra_modules": {
                            "embedding_h2d_seconds": 1.0,
                            "final_norm_h2d_seconds": 0.1,
                            "lm_head_h2d_seconds": 1.0,
                        },
                        "cpu_expert": {
                            "max_tokens": 2,
                            "forward_seconds_by_tokens": [
                                0.0,
                                0.5,
                                10.0,
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            solver = ProxyPlacementSolver(
                4,
                1,
                lookup_path=lookup,
                prefetch_portion=0.25,
                allow_unprofiled=False,
            )
            # The formal solver must not apply the 25% smoke-only cap.
            self.assertGreater(
                len(
                    solver.solve(
                        [0, 1, 2, 2],
                        layer_type="linear_attention",
                    )
                ),
                1,
            )
            with self.assertRaisesRegex(ValueError, "required at least 3"):
                ProxyPlacementSolver(
                    4,
                    1,
                    lookup_path=lookup,
                    prefetch_portion=0.25,
                    allow_unprofiled=False,
                    required_max_tokens=3,
                )

    def test_large_artifact_path_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sim"
            inside = require_within_simulation_root(root / "weights", root)
            self.assertEqual(inside, (root / "weights").resolve())
            with self.assertRaises(ValueError):
                require_within_simulation_root(
                    Path(directory) / "outside",
                    root,
                )


class ResultIsolationTest(unittest.TestCase):
    def test_formal_proxy_is_never_labeled_exact_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "seq_32"
            timing_dir = run_dir / "step_timing"
            timing_dir.mkdir(parents=True)
            config = {
                "backend": "aptmoe",
                "profile": "server",
                "benchmark_class": "deployment_proxy",
                "result_validity": "formal_deployment_proxy",
                "weight_source": "deterministic_random_initialization",
                "checkpoint_compatible": False,
                "llamafactory_backend": False,
                "allow_end_to_end_qwen35_tps_claim": False,
                "model_load_architecture": (
                    "Qwen35ComponentIsomorphicAPTMoEProxy"
                ),
                "precision": "bf16",
                "steps": 2,
                "warmup_steps": 1,
            }
            (run_dir / "run_config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            (timing_dir / "step_timing.json").write_text(
                json.dumps(
                    {
                        "timing_mode": "coarse_host_wall_no_cuda_sync",
                        "num_stable_steps": 1,
                        "aggregate_stable": {
                            key: {"mean_sec": 1.0}
                            for key in (
                                "step_total_sec",
                                "forward_sec",
                                "backward_sec",
                                "optimizer_sec",
                            )
                        },
                        "tps_attribution": {"stable_tps": 32.0},
                        "instrumentation": {
                            "forced_cuda_synchronize": False,
                            "backend_internal_probes": False,
                            "system_resource_monitor": False,
                            "per_step_file_io": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "proxy_manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark_class": "deployment_proxy",
                        "result_validity": "formal_deployment_proxy",
                        "proxy_architecture": "qwen35_component_isomorphic",
                        "parameter_count": 34_660_610_688,
                        "checkpoint_compatible": False,
                        "real_forward_backward_optimizer_update": True,
                        "route": {
                            "mode": "replayed_qwen35_topk_indices",
                            "trace_sha256": "route-hash",
                        },
                        "placement": {
                            "mode": "profiled_compute_load",
                            "deployment_profile": "server",
                            "lookup_sha256": "lookup-hash",
                        },
                        "runtime_versions": {
                            "qwen35_linear_attention_fastpath": True
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "full_update_verification.json").write_text(
                json.dumps({"valid_full_update": True}),
                encoding="utf-8",
            )
            row = aggregate_run(run_dir / "run_config.json")
            self.assertEqual(row["status"], "OK_PROXY")
            self.assertEqual(row["benchmark_class"], "deployment_proxy")


if __name__ == "__main__":
    unittest.main()
