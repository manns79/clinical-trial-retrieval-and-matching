from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.ingestion.trec import (
    parse_qrels,
    parse_topics_xml,
    qrel_from_json_record,
    qrel_to_json_record,
    topic_from_json_record,
    topic_to_json_record,
    validate_topics_and_qrels,
)
from clinical_trial_matching.io import read_jsonl, write_jsonl


FIXTURES = Path("data/fixtures")


class TrecIngestionTest(unittest.TestCase):
    def test_parse_free_text_topics_xml(self) -> None:
        topics = parse_topics_xml(FIXTURES / "topics2021.sample.xml", year=2021)

        self.assertEqual(len(topics), 2)
        self.assertEqual(topics[0].topic_id, "1")
        self.assertEqual(topics[0].year, 2021)
        self.assertIn("persistent asthma", topics[0].text)
        self.assertEqual(topics[0].fields, ())

    def test_parse_field_topics_xml(self) -> None:
        topics = parse_topics_xml(FIXTURES / "topics2023.sample.xml", year=2023)

        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0].topic_id, "101")
        self.assertEqual(topics[0].template, "glaucoma")
        self.assertEqual(topics[0].fields[0].name, "diagnosis")
        self.assertIn("diagnosis: Primary open-angle glaucoma", topics[0].text)

    def test_parse_trec_qrels(self) -> None:
        qrels = parse_qrels(FIXTURES / "qrels2021.sample.txt", year=2021)

        self.assertEqual(len(qrels), 4)
        self.assertEqual(qrels[0].nct_id, "NCT99990001")
        self.assertEqual(qrels[0].label, "eligible")

    def test_jsonl_round_trip(self) -> None:
        topics = parse_topics_xml(FIXTURES / "topics2023.sample.xml", year=2023)
        qrels = parse_qrels(FIXTURES / "qrels2021.sample.txt", year=2021)

        with tempfile.TemporaryDirectory() as tmpdir:
            topics_path = Path(tmpdir) / "topics.jsonl"
            qrels_path = Path(tmpdir) / "qrels.jsonl"
            write_jsonl(topics_path, (topic_to_json_record(topic) for topic in topics))
            write_jsonl(qrels_path, (qrel_to_json_record(qrel) for qrel in qrels))

            loaded_topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
            loaded_qrels = [qrel_from_json_record(row) for row in read_jsonl(qrels_path)]

        self.assertEqual(loaded_topics, topics)
        self.assertEqual(loaded_qrels, qrels)

    def test_validate_topics_and_qrels(self) -> None:
        topics = parse_topics_xml(FIXTURES / "topics2021.sample.xml", year=2021)
        qrels = parse_qrels(FIXTURES / "qrels2021.sample.txt", year=2021)

        summary = validate_topics_and_qrels(topics, qrels)

        self.assertEqual(summary["topics"], 2)
        self.assertEqual(summary["qrels"], 4)
        self.assertEqual(summary["topics_without_qrels"], [])
        self.assertEqual(summary["qrels_with_unknown_topics"], [])
        self.assertEqual(summary["relevance_distribution"]["eligible"], 2)


if __name__ == "__main__":
    unittest.main()
