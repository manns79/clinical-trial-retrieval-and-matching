from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from clinical_trial_matching.benchmarking.serving import (
    PRIMARY_MODES,
    dominant_startup_resource_phase,
    latency_summary,
    load_serving_benchmark,
    percentile,
    run_serving_benchmark,
    startup_phase_report,
    summarize_mode_measurements,
)
from clinical_trial_matching.io import read_json, write_json


class FakeServingRuntime:
    def __init__(self) -> None:
        self.preload_calls = 0
        self.search_calls: list[tuple[str, str]] = []

    def preload(self, on_phase_complete: Any = None) -> None:
        self.preload_calls += 1
        for phase in ("corpus", "sqlite_fts5", "dense_index_and_model"):
            if on_phase_complete is not None:
                on_phase_complete(phase)

    def search(
        self,
        *,
        query: str,
        mode: str,
        top_k: int,
        snippet_chars: int,
    ) -> dict[str, Any]:
        self.search_calls.append((query, mode))
        stage_values = {
            "sqlite-fts5": {"lexical": 1.0, "embedding": 0.0, "fusion": 0.0},
            "dense": {"lexical": 0.0, "embedding": 2.0, "fusion": 0.0},
            "hybrid": {"lexical": 1.0, "embedding": 2.0, "fusion": 0.5},
        }[mode]
        total = sum(stage_values.values())
        return {
            "latency_ms": {**stage_values, "total": total},
            "results": [{"nct_id": "NCT1"}] * min(top_k, 1),
        }


class ServingBenchmarkTest(unittest.TestCase):
    def test_tracked_config_pins_primary_modes_and_local_output(self) -> None:
        benchmark = load_serving_benchmark(
            Path("configs/benchmarks/trec_2021_local_serving.json")
        )

        self.assertEqual(benchmark.modes, PRIMARY_MODES)
        self.assertEqual(benchmark.warmup_rounds, 1)
        self.assertEqual(benchmark.measurement_rounds, 5)
        self.assertIn("/outputs/", benchmark.output_path.as_posix())
        self.assertNotIn("/holdout/", benchmark.corpus_path.as_posix())

    def test_loader_rejects_paths_outside_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "benchmark.json"
            payload = serving_benchmark_payload()
            payload["serving"]["dense_index"] = "../dense.npz"
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "cannot resolve outside project_root"):
                load_serving_benchmark(config_path)

    def test_loader_requires_all_three_primary_modes_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "benchmark.json"
            payload = serving_benchmark_payload()
            payload["benchmark"]["modes"] = ["dense", "hybrid"]
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "three primary modes"):
                load_serving_benchmark(config_path)

    def test_benchmark_writes_complete_report_with_fake_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, content in (
                ("trials.jsonl", "{}\n"),
                ("fts.sqlite", "lexical"),
                ("dense.npz", "dense"),
            ):
                (root / name).write_text(content, encoding="utf-8")
            config_path = root / "benchmark.json"
            write_json(config_path, serving_benchmark_payload())
            benchmark = load_serving_benchmark(config_path)
            runtime = FakeServingRuntime()

            report = run_serving_benchmark(
                benchmark,
                runtime_factory=lambda: runtime,
            )
            persisted = read_json(root / "report.json")

        self.assertEqual(runtime.preload_calls, 1)
        self.assertEqual(len(runtime.search_calls), 9)
        self.assertEqual(report["warm"]["modes"]["dense"]["requests"], 2)
        self.assertEqual(
            report["warm"]["modes"]["hybrid"]["stage_latency_ms"]["fusion"]["p95"],
            0.5,
        )
        self.assertEqual(report["cost"]["hosted_service_cost_usd"], 0.0)
        self.assertEqual(
            [phase["name"] for phase in report["cold_start"]["phases"]],
            ["api_import", "corpus", "sqlite_fts5", "dense_index_and_model"],
        )
        self.assertEqual(persisted["benchmark"]["name"], "fixture_serving")
        self.assertEqual(persisted["artifacts"]["files"]["dense_index"]["bytes"], 5)

    def test_latency_statistics_use_linear_interpolation(self) -> None:
        self.assertEqual(percentile([10.0, 20.0, 30.0], 0.5), 20.0)
        self.assertEqual(percentile([10.0, 20.0], 0.95), 19.5)
        self.assertEqual(
            latency_summary([10.0, 20.0]),
            {
                "minimum": 10.0,
                "mean": 15.0,
                "p50": 15.0,
                "p95": 19.5,
                "maximum": 20.0,
            },
        )

    def test_mode_summary_reports_sequential_throughput(self) -> None:
        rows = [
            {
                "handler_wall_ms": 10.0,
                "latency_ms": {
                    "lexical": 8.0,
                    "embedding": 0.0,
                    "fusion": 0.0,
                    "total": 9.0,
                },
                "result_count": 3,
                "rss_bytes": 100 * 1024 * 1024,
            },
            {
                "handler_wall_ms": 20.0,
                "latency_ms": {
                    "lexical": 18.0,
                    "embedding": 0.0,
                    "fusion": 0.0,
                    "total": 19.0,
                },
                "result_count": 3,
                "rss_bytes": 110 * 1024 * 1024,
            },
        ]

        summary = summarize_mode_measurements(rows)

        self.assertEqual(summary["handler_latency_ms"]["p50"], 15.0)
        self.assertEqual(summary["sequential_requests_per_second"], 66.667)
        self.assertEqual(summary["sampled_process_rss"]["maximum"]["mib"], 110.0)

    def test_startup_phase_reports_retained_and_peak_rss_deltas(self) -> None:
        phase = startup_phase_report(
            name="sqlite_fts5",
            elapsed_ms=123.4567,
            before={"rss_bytes": 100, "peak_rss_bytes": 120},
            after={"rss_bytes": 160, "peak_rss_bytes": 210},
        )

        self.assertEqual(phase["milliseconds"], 123.457)
        self.assertEqual(phase["retained_rss_delta"]["bytes"], 60)
        self.assertEqual(phase["peak_rss_delta"]["bytes"], 90)

    def test_dominant_resource_phase_excludes_api_import(self) -> None:
        phases = [
            {"name": "api_import", "retained_rss_delta": {"bytes": 500}},
            {"name": "corpus", "retained_rss_delta": {"bytes": 100}},
            {"name": "sqlite_fts5", "retained_rss_delta": {"bytes": 300}},
            {
                "name": "dense_index_and_model",
                "retained_rss_delta": {"bytes": 100},
            },
        ]

        dominant = dominant_startup_resource_phase(phases)

        self.assertEqual(dominant["name"], "sqlite_fts5")
        self.assertEqual(dominant["share_of_positive_resource_delta"], 0.6)


def serving_benchmark_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "fixture_serving",
        "description": "Fixture serving benchmark.",
        "project_root": ".",
        "serving": {
            "corpus": "trials.jsonl",
            "sqlite_fts_index": "fts.sqlite",
            "dense_index": "dense.npz",
            "dense_model_name": "fixture-model",
            "dense_text_representation": "title_summary_conditions",
            "dense_batch_size": 2,
            "dense_device": "cpu",
            "dense_max_seq_length": 128,
            "rrf_k": 60,
            "rrf_candidate_depth": 100,
        },
        "benchmark": {
            "modes": list(PRIMARY_MODES),
            "queries": ["synthetic asthma summary"],
            "warmup_rounds": 1,
            "measurement_rounds": 2,
            "top_k": 3,
            "snippet_chars": 120,
        },
        "artifacts": {"output": "report.json"},
    }


if __name__ == "__main__":
    unittest.main()
