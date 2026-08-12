from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.cli import run_bm25_experiment
from clinical_trial_matching.evaluation.experiments import load_bm25_experiment
from clinical_trial_matching.io import read_json, write_json
from clinical_trial_matching.retrieval.bm25 import DEFAULT_FIELD_WEIGHTS


class Bm25ExperimentTest(unittest.TestCase):
    def test_frozen_holdout_weights_match_selected_development_profile(self) -> None:
        development = load_bm25_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "fielded_bm25_condition_title_v1.json"
            )
        )
        holdout = load_bm25_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "holdout_fielded_bm25_condition_title_v1.json"
            )
        )

        self.assertEqual(holdout.field_weights, development.field_weights)
        self.assertIn("/development/", development.topics_path.as_posix())
        self.assertIn("/holdout/", holdout.topics_path.as_posix())
        self.assertEqual(holdout.index_path, development.index_path)

    def test_load_experiment_resolves_paths_and_pins_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "configs" / "fielded.json"
            write_json(config_path, experiment_payload(project_root=".."))

            experiment = load_bm25_experiment(config_path)

        self.assertEqual(experiment.name, "fixture_fielded_bm25")
        self.assertEqual(experiment.field_weights, DEFAULT_FIELD_WEIGHTS)
        self.assertEqual(experiment.trials_path, root / "data" / "trials.jsonl")
        self.assertEqual(experiment.config_label, "configs/fielded.json")
        self.assertEqual(len(experiment.config_sha256), 64)

    def test_fielded_experiment_requires_every_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            payload = experiment_payload(project_root=".")
            payload["field_weights"] = {"conditions": 2.0}
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "must pin every field weight"):
                load_bm25_experiment(config_path)

    def test_experiment_paths_cannot_escape_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            payload = experiment_payload(project_root=".")
            payload["artifacts"]["metrics"] = "../metrics.json"
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "cannot resolve outside project_root"):
                load_bm25_experiment(config_path)

    def test_run_experiment_writes_traceable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            shutil.copyfile("data/fixtures/trials.sample.jsonl", data_dir / "trials.jsonl")
            shutil.copyfile("data/fixtures/topics.sample.jsonl", data_dir / "topics.jsonl")
            shutil.copyfile("data/fixtures/qrels.sample.tsv", data_dir / "qrels.tsv")
            config_path = root / "configs" / "fielded.json"
            write_json(config_path, experiment_payload(project_root=".."))

            run_bm25_experiment(config_path)

            metrics = read_json(root / "outputs" / "metrics.json")
            diagnostics = read_json(root / "outputs" / "diagnostics.json")

        self.assertEqual(metrics["run_name"], "fixture_fielded_bm25")
        self.assertEqual(metrics["experiment"]["config_path"], "configs/fielded.json")
        self.assertEqual(
            metrics["experiment"]["config_sha256"],
            diagnostics["experiment"]["config_sha256"],
        )


def experiment_payload(*, project_root: str) -> dict:
    return {
        "schema_version": 1,
        "name": "fixture_fielded_bm25",
        "description": "Fixture experiment for tests.",
        "project_root": project_root,
        "retriever": "fielded-bm25",
        "field_weights": dict(DEFAULT_FIELD_WEIGHTS),
        "benchmark": {
            "trials": "data/trials.jsonl",
            "topics": "data/topics.jsonl",
            "qrels": "data/qrels.tsv",
            "top_k": 100,
        },
        "artifacts": {
            "index": "data/index.pkl",
            "run": "outputs/run.trec",
            "metrics": "outputs/metrics.json",
            "diagnostics": "outputs/diagnostics.json",
        },
    }


if __name__ == "__main__":
    unittest.main()
