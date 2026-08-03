from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.cli import check_retrieval_regression
from clinical_trial_matching.evaluation.regression import run_bm25_regression_check
from clinical_trial_matching.ingestion.clinicaltrials import trial_from_flat_record
from clinical_trial_matching.ingestion.trec import parse_qrels, qrel_from_json_record, topic_from_json_record
from clinical_trial_matching.io import read_json, read_jsonl


class RetrievalRegressionTest(unittest.TestCase):
    def test_bm25_regression_check_passes_fixture_thresholds(self) -> None:
        trials = [
            trial_from_flat_record(row) for row in read_jsonl(Path("data/fixtures/trials.sample.jsonl"))
        ]
        topics = [
            topic_from_json_record(row) for row in read_jsonl(Path("data/fixtures/topics.sample.jsonl"))
        ]
        qrels = parse_qrels(Path("data/fixtures/qrels.sample.tsv"))

        report = run_bm25_regression_check(trials=trials, topics=topics, qrels=qrels)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["metrics"]["recall_at_100"], 1.0)
        self.assertEqual(report["metrics"]["mrr"], 1.0)
        self.assertEqual(report["run"]["1"][0], "NCT99990001")
        self.assertEqual(report["run"]["2"][0], "NCT99990002")

    def test_bm25_regression_check_reports_threshold_failures(self) -> None:
        trials = [
            trial_from_flat_record(row) for row in read_jsonl(Path("data/fixtures/trials.sample.jsonl"))
        ]
        topics = [
            topic_from_json_record(row) for row in read_jsonl(Path("data/fixtures/topics.sample.jsonl"))
        ]
        qrels = parse_qrels(Path("data/fixtures/qrels.sample.tsv"))

        report = run_bm25_regression_check(
            trials=trials,
            topics=topics,
            qrels=qrels,
            thresholds={"precision_at_10": 0.9},
        )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(
            report["failures"],
            [{"metric": "precision_at_10", "observed": 0.3333333333333333, "threshold": 0.9}],
        )

    def test_cli_writes_passing_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "regression.json"
            check_retrieval_regression(
                trials_path=Path("data/fixtures/trials.sample.jsonl"),
                topics_path=Path("data/fixtures/topics.sample.jsonl"),
                qrels_path=Path("data/fixtures/qrels.sample.tsv"),
                output_path=output,
                top_k=100,
                thresholds={"recall_at_100": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0},
            )
            report = read_json(output)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["topics"], 2)
        self.assertEqual(report["trials"], 3)

    def test_jsonl_qrels_records_can_be_used_by_regression_cli(self) -> None:
        qrels_rows = [
            {"topic_id": "1", "nct_id": "NCT99990001", "relevance": 2, "year": None},
            {"topic_id": "2", "nct_id": "NCT99990002", "relevance": 2, "year": None},
        ]
        qrels = [qrel_from_json_record(row) for row in qrels_rows]
        self.assertEqual([qrel.nct_id for qrel in qrels], ["NCT99990001", "NCT99990002"])


if __name__ == "__main__":
    unittest.main()
