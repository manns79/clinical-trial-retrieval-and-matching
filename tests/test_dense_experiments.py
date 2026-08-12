from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.evaluation.experiments import load_dense_experiment
from clinical_trial_matching.io import write_json


class DenseExperimentTest(unittest.TestCase):
    def test_tracked_development_experiment_is_development_only(self) -> None:
        experiment = load_dense_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "development_dense_all_minilm_l6_v2.json"
            )
        )

        self.assertEqual(
            experiment.model_name,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertEqual(experiment.text_representation, "title_summary_conditions")
        self.assertIn("/development/", experiment.topics_path.as_posix())
        self.assertNotIn("/holdout/", experiment.topics_path.as_posix())
        self.assertEqual(experiment.index_path.suffix, ".npz")

    def test_loader_rejects_unknown_text_representation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dense.json"
            payload = dense_experiment_payload()
            payload["text_representation"] = "unknown"
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "Unknown dense text representation"):
                load_dense_experiment(config_path)

    def test_loader_rejects_index_outside_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dense.json"
            payload = dense_experiment_payload()
            payload["artifacts"]["index"] = "../dense.npz"
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "cannot resolve outside project_root"):
                load_dense_experiment(config_path)


def dense_experiment_payload() -> dict:
    return {
        "schema_version": 1,
        "name": "fixture_dense",
        "description": "Dense fixture.",
        "project_root": ".",
        "model_name": "fixture-model",
        "text_representation": "title_summary_conditions",
        "batch_size": 2,
        "device": "cpu",
        "max_seq_length": 128,
        "benchmark": {
            "trials": "data/trials.jsonl",
            "topics": "data/topics.jsonl",
            "qrels": "data/qrels.jsonl",
            "top_k": 100,
        },
        "artifacts": {
            "index": "data/indexes/dense.npz",
            "run": "outputs/dense.run",
            "metrics": "outputs/dense_metrics.json",
            "diagnostics": "outputs/dense_diagnostics.json",
        },
    }


if __name__ == "__main__":
    unittest.main()
