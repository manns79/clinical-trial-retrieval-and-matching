from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.evaluation.reranking_comparison import (
    build_cross_encoder_comparison,
    load_cross_encoder_comparison,
)
from clinical_trial_matching.io import read_json, write_json


class CrossEncoderComparisonTest(unittest.TestCase):
    def test_tracked_suite_uses_depth_ten_and_strict_quality_gates(self) -> None:
        comparison = load_cross_encoder_comparison(
            Path(
                "configs/experiments/trec_2021/"
                "development_cross_encoder_depth10_optimization.json"
            )
        )

        self.assertEqual(comparison.candidate_depth, 10)
        self.assertEqual(comparison.eligible_ndcg_tolerance, 0.0)
        self.assertEqual(comparison.broad_ndcg_tolerance, 0.0)
        self.assertEqual(comparison.hybrid_p95_ms, 250.0)
        self.assertEqual(comparison.reranked_p95_budget_ms, 500.0)
        self.assertEqual(comparison.baseline.candidate_depth, 10)

    def test_small_depth_suite_compares_each_depth_with_depth_ten(self) -> None:
        comparison = load_cross_encoder_comparison(
            Path(
                "configs/experiments/trec_2021/"
                "development_cross_encoder_small_depths_comparison.json"
            )
        )

        self.assertEqual(comparison.baseline.candidate_depth, 10)
        self.assertEqual(
            tuple(candidate.candidate_depth for candidate in comparison.candidates),
            (3, 5, 8),
        )

    def test_comparison_requires_quality_and_latency_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.json"
            quality_only = root / "quality_only.json"
            passing = root / "passing.json"
            write_json(baseline, _report(eligible=0.01, broad=0.02, p95=300.0))
            write_json(quality_only, _report(eligible=0.01, broad=0.02, p95=251.0))
            write_json(passing, _report(eligible=0.011, broad=0.021, p95=200.0))
            config = root / "comparison.json"
            write_json(
                config,
                {
                    "schema_version": 1,
                    "name": "fixture",
                    "description": "fixture comparison",
                    "project_root": ".",
                    "candidate_depth": 10,
                    "baseline": {"label": "baseline", "report": baseline.name},
                    "candidates": [
                        {"label": "quality_only", "report": quality_only.name},
                        {"label": "passing", "report": passing.name},
                    ],
                    "quality_gate": {
                        "eligible_ndcg_at_10_tolerance": 0.0,
                        "broad_ndcg_at_10_tolerance": 0.0,
                    },
                    "latency_budget": {
                        "hybrid_p95_ms": 250.0,
                        "reranked_mode_p95_ms": 500.0,
                    },
                    "artifacts": {"report": "output.json", "table": "output.md"},
                },
            )

            report = build_cross_encoder_comparison(load_cross_encoder_comparison(config))

            self.assertFalse(report["rows"][1]["latency_gate_passed"])
            self.assertTrue(report["rows"][1]["quality_gate_passed"])
            self.assertEqual(report["rows"][1]["candidate_depth"], 10)
            self.assertEqual(report["passing_candidates"], ["passing"])
            saved_report = read_json(root / "output.json")
            self.assertEqual(saved_report["serving_candidate_selected"], "passing")


def _report(*, eligible: float, broad: float, p95: float) -> dict[str, object]:
    return {
        "model": {
            "precision": "int8",
            "max_length": 128,
            "text_representation": "title_summary_conditions",
            "artifact_bytes": 1024,
        },
        "depths": {
            "10": {
                "metric_deltas": {
                    "metrics": {
                        "eligible_only": {"ndcg_at_10": eligible, "mrr": 0.03},
                        "excluded_or_eligible": {"ndcg_at_10": broad},
                    }
                },
                "latency_ms_per_topic": {
                    "total": {"mean": p95 - 20, "p50": p95 - 10, "p95": p95}
                },
            }
        },
        "process_peak_rss_mib": 200.0,
    }


if __name__ == "__main__":
    unittest.main()
