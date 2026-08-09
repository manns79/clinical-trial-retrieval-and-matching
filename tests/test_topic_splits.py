from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.cli import write_trec_topic_split
from clinical_trial_matching.evaluation.splits import (
    TOPIC_SPLIT_STRATEGY,
    build_trec_topic_split,
    topic_split_report,
)
from clinical_trial_matching.io import read_json, read_jsonl
from clinical_trial_matching.models import Qrel, Topic


class TrecTopicSplitTest(unittest.TestCase):
    def test_split_is_exact_disjoint_and_independent_of_input_order(self) -> None:
        topics = [Topic(topic_id=str(topic_id), text=f"Topic {topic_id}") for topic_id in range(10)]
        qrels = [
            Qrel(topic_id=str(topic_id), nct_id=f"NCT{topic_id:08d}", relevance=2)
            for topic_id in range(10)
        ]

        observed = build_trec_topic_split(
            topics,
            qrels,
            seed="fixture-v1",
            holdout_fraction=0.2,
        )
        reordered = build_trec_topic_split(
            list(reversed(topics)),
            list(reversed(qrels)),
            seed="fixture-v1",
            holdout_fraction=0.2,
        )

        development_ids = set(observed.development_topic_ids)
        holdout_ids = set(observed.holdout_topic_ids)
        self.assertEqual(len(development_ids), 8)
        self.assertEqual(len(holdout_ids), 2)
        self.assertFalse(development_ids & holdout_ids)
        self.assertEqual(holdout_ids, set(reordered.holdout_topic_ids))
        self.assertEqual(
            {qrel.topic_id for qrel in observed.holdout_qrels},
            holdout_ids,
        )

    def test_split_report_records_provenance_and_integrity(self) -> None:
        topics = [Topic(topic_id=str(topic_id), text="Text") for topic_id in range(5)]
        qrels = [
            Qrel(topic_id=str(topic_id), nct_id="NCT1", relevance=topic_id % 3)
            for topic_id in range(5)
        ]
        split = build_trec_topic_split(topics, qrels, seed="report-v1", holdout_fraction=0.2)

        report = topic_split_report(
            split,
            topics_source={"path": "topics.jsonl", "sha256": "abc", "bytes": 10},
            qrels_source={"path": "qrels.jsonl", "sha256": "def", "bytes": 20},
        )

        self.assertEqual(report["strategy"], TOPIC_SPLIT_STRATEGY)
        self.assertEqual(report["partitions"]["development"]["topics"], 4)
        self.assertEqual(report["partitions"]["holdout"]["topics"], 1)
        self.assertEqual(report["integrity"]["topic_overlap"], [])
        self.assertEqual(report["integrity"]["assigned_topics"], 5)
        self.assertEqual(report["integrity"]["assigned_qrels"], 5)

    def test_split_rejects_qrels_for_unknown_topics(self) -> None:
        topics = [Topic(topic_id="1", text="One"), Topic(topic_id="2", text="Two")]
        qrels = [Qrel(topic_id="3", nct_id="NCT1", relevance=2)]

        with self.assertRaisesRegex(ValueError, "absent from topics"):
            build_trec_topic_split(topics, qrels, seed="fixture-v1")

    def test_cli_writes_normalized_partition_files_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            development_topics = root / "development" / "topics.jsonl"
            development_qrels = root / "development" / "qrels.jsonl"
            holdout_topics = root / "holdout" / "topics.jsonl"
            holdout_qrels = root / "holdout" / "qrels.jsonl"
            report_path = root / "split_report.json"

            write_trec_topic_split(
                topics_path=Path("data/fixtures/topics.sample.jsonl"),
                qrels_path=Path("data/fixtures/qrels.sample.tsv"),
                development_topics_output=development_topics,
                development_qrels_output=development_qrels,
                holdout_topics_output=holdout_topics,
                holdout_qrels_output=holdout_qrels,
                report_output=report_path,
                seed="fixture-v1",
                holdout_fraction=0.5,
            )

            development_topic_rows = read_jsonl(development_topics)
            development_qrel_rows = read_jsonl(development_qrels)
            holdout_topic_rows = read_jsonl(holdout_topics)
            holdout_qrel_rows = read_jsonl(holdout_qrels)
            report = read_json(report_path)

        self.assertEqual(len(development_topic_rows), 1)
        self.assertEqual(len(holdout_topic_rows), 1)
        self.assertEqual(len(development_qrel_rows), 2)
        self.assertEqual(len(holdout_qrel_rows), 2)
        self.assertRegex(report["sources"]["topics"]["sha256"], r"^[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
