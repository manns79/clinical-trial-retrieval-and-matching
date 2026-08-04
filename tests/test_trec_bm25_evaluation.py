from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.cli import evaluate_trec_bm25
from clinical_trial_matching.evaluation.trec import (
    bm25_trec_topic_diagnostics,
    bm25_trec_evaluation_report,
    build_bm25_trec_run,
    evaluate_trec_binary_views,
    evaluate_trec_run,
    TrecRunRow,
    write_trec_run,
)
from clinical_trial_matching.ingestion.clinicaltrials import trial_from_flat_record
from clinical_trial_matching.ingestion.trec import parse_qrels, topic_from_json_record
from clinical_trial_matching.io import read_json, read_jsonl


class TrecBm25EvaluationTest(unittest.TestCase):
    def test_build_bm25_trec_run_uses_trec_format_fields(self) -> None:
        trials = [
            trial_from_flat_record(row) for row in read_jsonl(Path("data/fixtures/trials.sample.jsonl"))
        ]
        topics = [
            topic_from_json_record(row) for row in read_jsonl(Path("data/fixtures/topics.sample.jsonl"))
        ]

        rows = build_bm25_trec_run(
            trials=trials,
            topics=topics,
            run_name="bm25_fixture",
            top_k=2,
        )

        self.assertEqual(rows[0].topic_id, "1")
        self.assertEqual(rows[0].nct_id, "NCT99990001")
        self.assertEqual(rows[0].rank, 1)
        self.assertEqual(rows[0].run_name, "bm25_fixture")
        self.assertRegex(rows[0].to_trec_line(), r"^1 Q0 NCT99990001 1 [0-9.]+ bm25_fixture$")

    def test_evaluate_trec_run_reports_metrics(self) -> None:
        trials = [
            trial_from_flat_record(row) for row in read_jsonl(Path("data/fixtures/trials.sample.jsonl"))
        ]
        topics = [
            topic_from_json_record(row) for row in read_jsonl(Path("data/fixtures/topics.sample.jsonl"))
        ]
        qrels = parse_qrels(Path("data/fixtures/qrels.sample.tsv"))
        rows = build_bm25_trec_run(trials=trials, topics=topics, run_name="bm25_fixture")

        metrics = evaluate_trec_run(rows, qrels)
        binary_views = evaluate_trec_binary_views(rows, qrels)
        report = bm25_trec_evaluation_report(
            rows=rows,
            qrels=qrels,
            run_name="bm25_fixture",
            top_k=100,
            topics_count=len(topics),
            trials_count=len(trials),
        )

        self.assertEqual(metrics["recall_at_100"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(binary_views["excluded_or_eligible"]["recall_at_100"], 1.0)
        self.assertEqual(binary_views["eligible_only"]["recall_at_100"], 1.0)
        self.assertEqual(report["run_name"], "bm25_fixture")
        self.assertEqual(report["metrics"]["excluded_or_eligible"]["ndcg_at_10"], 1.0)
        self.assertEqual(report["metrics"]["eligible_only"]["ndcg_at_10"], 1.0)
        self.assertEqual(report["graded_metrics"]["ndcg_at_10"], 1.0)
        self.assertEqual(report["topics_with_results"], 2)
        self.assertEqual(report["run_rows"], len(rows))

    def test_topic_diagnostics_identify_weak_topics(self) -> None:
        topics = [
            topic_from_json_record(row) for row in read_jsonl(Path("data/fixtures/topics.sample.jsonl"))
        ]
        qrels = parse_qrels(Path("data/fixtures/qrels.sample.tsv"))
        rows = [
            TrecRunRow("1", "NCT99990002", 1, 2.0, "bm25_fixture"),
            TrecRunRow("2", "NCT99990002", 1, 2.0, "bm25_fixture"),
        ]

        diagnostics = bm25_trec_topic_diagnostics(
            rows=rows,
            qrels=qrels,
            topics=topics,
            run_name="bm25_fixture",
            top_k=1,
        )

        topic_1 = diagnostics["topics"][0]
        topic_2 = diagnostics["topics"][1]
        self.assertTrue(topic_1["weak_retrieval"])
        self.assertEqual(topic_1["first_eligible_rank"], None)
        self.assertIn("no_eligible_result_in_top_k", topic_1["weak_retrieval_reasons"])
        self.assertFalse(topic_2["weak_retrieval"])
        self.assertEqual(diagnostics["weak_topics"][0]["topic_id"], "1")

    def test_write_trec_run(self) -> None:
        trials = [
            trial_from_flat_record(row) for row in read_jsonl(Path("data/fixtures/trials.sample.jsonl"))
        ]
        topics = [
            topic_from_json_record(row) for row in read_jsonl(Path("data/fixtures/topics.sample.jsonl"))
        ]
        rows = build_bm25_trec_run(
            trials=trials,
            topics=topics,
            run_name="bm25_fixture",
            top_k=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run.txt"
            write_trec_run(output, rows)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("1 Q0 NCT99990001 1 "))
        self.assertTrue(lines[1].startswith("2 Q0 NCT99990002 1 "))

    def test_cli_writes_run_and_metrics_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_output = Path(tmpdir) / "bm25.run"
            metrics_output = Path(tmpdir) / "metrics.json"
            evaluate_trec_bm25(
                trials_path=Path("data/fixtures/trials.sample.jsonl"),
                topics_path=Path("data/fixtures/topics.sample.jsonl"),
                qrels_path=Path("data/fixtures/qrels.sample.tsv"),
                run_output_path=run_output,
                metrics_output_path=metrics_output,
                diagnostics_output_path=Path(tmpdir) / "diagnostics.json",
                run_name="bm25_fixture",
                top_k=100,
            )
            run_lines = run_output.read_text(encoding="utf-8").splitlines()
            metrics = read_json(metrics_output)
            diagnostics = read_json(Path(tmpdir) / "diagnostics.json")

        self.assertGreater(len(run_lines), 0)
        self.assertEqual(metrics["retriever"], "bm25")
        self.assertEqual(metrics["metrics"]["excluded_or_eligible"]["ndcg_at_10"], 1.0)
        self.assertEqual(metrics["metrics"]["eligible_only"]["ndcg_at_10"], 1.0)
        self.assertEqual(metrics["graded_metrics"]["ndcg_at_10"], 1.0)
        self.assertEqual(len(diagnostics["topics"]), 2)


if __name__ == "__main__":
    unittest.main()
