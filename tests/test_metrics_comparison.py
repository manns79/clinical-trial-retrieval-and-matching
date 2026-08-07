from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.evaluation.comparison import (
    build_metrics_comparison,
    infer_comparison_format,
    parse_metrics_spec,
    write_metrics_comparison,
)
from clinical_trial_matching.io import read_json, write_json


class MetricsComparisonTest(unittest.TestCase):
    def test_build_metrics_comparison_flattens_nested_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "fielded.json"
            write_json(
                metrics_path,
                {
                    "run_name": "fielded_bm25",
                    "retriever": "fielded-bm25",
                    "top_k": 100,
                    "topics": 75,
                    "trials": 26150,
                    "run_rows": 7500,
                    "metrics": {
                        "eligible_only": {
                            "precision_at_10": 0.244,
                            "recall_at_100": 0.2687835568,
                            "mrr": 0.4291612787,
                            "ndcg_at_10": 0.2471696679,
                            "ndcg_at_100": 0.2683941284,
                        },
                        "excluded_or_eligible": {
                            "precision_at_10": 0.6253333333,
                            "recall_at_100": 0.3142983387,
                            "mrr": 0.7701855724,
                            "ndcg_at_10": 0.6322739329,
                            "ndcg_at_100": 0.5070301693,
                        },
                    },
                },
            )

            comparison = build_metrics_comparison(
                [parse_metrics_spec(f"fielded={metrics_path}")],
                views=["eligible_only"],
            )

        self.assertEqual(comparison["reports"], 1)
        self.assertEqual(len(comparison["rows"]), 1)
        row = comparison["rows"][0]
        self.assertEqual(row["label"], "fielded")
        self.assertEqual(row["view"], "eligible_only")
        self.assertEqual(row["precision_at_10"], 0.244)
        self.assertEqual(row["recall_at_100"], 0.268784)

    def test_build_metrics_comparison_supports_flat_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "flat.json"
            write_json(
                metrics_path,
                {
                    "run_name": "legacy_bm25",
                    "retriever": "bm25",
                    "metrics": {"precision_at_10": 0.5, "recall_at_100": 1.0, "mrr": 1.0},
                },
            )

            comparison = build_metrics_comparison([parse_metrics_spec(str(metrics_path))])

        self.assertEqual(comparison["rows"][0]["label"], "legacy_bm25")
        self.assertEqual(comparison["rows"][0]["view"], "metrics")
        self.assertEqual(comparison["rows"][0]["ndcg_at_10"], "")

    def test_write_metrics_comparison_outputs_markdown_csv_and_json(self) -> None:
        comparison = {
            "reports": 1,
            "columns": ["label"],
            "rows": [
                {
                    "label": "fielded",
                    "run_name": "fielded_bm25",
                    "retriever": "fielded-bm25",
                    "view": "eligible_only",
                    "top_k": 100,
                    "topics": 75,
                    "trials": 26150,
                    "run_rows": 7500,
                    "precision_at_10": 0.244,
                    "recall_at_100": 0.268784,
                    "mrr": 0.429161,
                    "ndcg_at_10": 0.24717,
                    "ndcg_at_100": 0.268394,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            markdown_path = tmp_path / "comparison.md"
            csv_path = tmp_path / "comparison.csv"
            json_path = tmp_path / "comparison.json"
            write_metrics_comparison(markdown_path, comparison, "markdown")
            write_metrics_comparison(csv_path, comparison, "csv")
            write_metrics_comparison(json_path, comparison, "json")

            markdown = markdown_path.read_text(encoding="utf-8")
            csv_text = csv_path.read_text(encoding="utf-8")
            json_payload = read_json(json_path)

        self.assertIn("| label | run_name | retriever | view |", markdown)
        self.assertIn("fielded_bm25", csv_text)
        self.assertEqual(json_payload["rows"][0]["label"], "fielded")

    def test_infer_comparison_format_uses_suffix(self) -> None:
        self.assertEqual(infer_comparison_format(Path("report.csv"), None), "csv")
        self.assertEqual(infer_comparison_format(Path("report.json"), None), "json")
        self.assertEqual(infer_comparison_format(Path("report.md"), None), "markdown")
        self.assertEqual(infer_comparison_format(Path("report.txt"), None), "markdown")


if __name__ == "__main__":
    unittest.main()
