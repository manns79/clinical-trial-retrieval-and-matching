from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

from clinical_trial_matching.cli import run_sqlite_fts_experiment
from clinical_trial_matching.evaluation.experiments import load_sqlite_fts_experiment
from clinical_trial_matching.io import read_json, write_json
from clinical_trial_matching.retrieval.sqlite_fts import (
    DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
)


class SQLiteFtsExperimentTest(unittest.TestCase):
    def test_tracked_experiment_is_development_only(self) -> None:
        experiment = load_sqlite_fts_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "development_sqlite_fts5_condition_title_v1.json"
            )
        )

        self.assertIn("/development/", experiment.topics_path.as_posix())
        self.assertNotIn("/holdout/", experiment.topics_path.as_posix())
        self.assertEqual(
            experiment.field_weights,
            DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
        )
        self.assertEqual(experiment.index_path.suffix, ".sqlite")

    def test_loader_rejects_missing_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment.json"
            payload = experiment_payload()
            del payload["field_weights"]["conditions"]
            write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "Missing SQLite FTS5"):
                load_sqlite_fts_experiment(path)

    def test_runner_writes_traceable_development_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data").mkdir()
            (root / "outputs").mkdir()
            copyfile("data/fixtures/trials.sample.jsonl", root / "data/trials.jsonl")
            copyfile("data/fixtures/topics.sample.jsonl", root / "data/topics.jsonl")
            copyfile("data/fixtures/qrels.sample.tsv", root / "data/qrels.tsv")
            config_path = root / "experiment.json"
            write_json(config_path, experiment_payload())

            run_sqlite_fts_experiment(config_path)
            metrics = read_json(root / "outputs/metrics.json")
            diagnostics = read_json(root / "outputs/diagnostics.json")

        self.assertEqual(metrics["retriever"], "sqlite-fts5")
        self.assertEqual(metrics["metrics"]["eligible_only"]["recall_at_100"], 1.0)
        self.assertEqual(
            metrics["experiment"]["config_sha256"],
            diagnostics["experiment"]["config_sha256"],
        )
        self.assertEqual(len(diagnostics["topics"]), 2)


def experiment_payload() -> dict:
    return {
        "schema_version": 1,
        "name": "fixture_sqlite_fts5",
        "description": "Fixture SQLite FTS5 experiment.",
        "project_root": ".",
        "field_weights": dict(DEFAULT_SQLITE_FTS_FIELD_WEIGHTS),
        "benchmark": {
            "trials": "data/trials.jsonl",
            "topics": "data/topics.jsonl",
            "qrels": "data/qrels.tsv",
            "top_k": 100,
        },
        "artifacts": {
            "index": "data/index.sqlite",
            "run": "outputs/run.trec",
            "metrics": "outputs/metrics.json",
            "diagnostics": "outputs/diagnostics.json",
        },
    }


if __name__ == "__main__":
    unittest.main()
