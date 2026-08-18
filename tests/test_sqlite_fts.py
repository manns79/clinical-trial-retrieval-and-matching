from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.models import Trial
from clinical_trial_matching.retrieval.sqlite_fts import (
    DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
    build_sqlite_fts_index,
    load_sqlite_fts_retriever,
    normalize_sqlite_fts_field_weights,
)


class SQLiteFtsRetrieverTest(unittest.TestCase):
    def test_build_load_and_weighted_search(self) -> None:
        trials = [
            Trial(
                nct_id="NCT1",
                title="Asthma inhaler study",
                conditions=("Asthma",),
                interventions=("Inhaled corticosteroid",),
                eligibility_criteria="Adults with persistent wheezing.",
            ),
            Trial(
                nct_id="NCT2",
                title="Migraine prevention study",
                conditions=("Migraine",),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "trials.sqlite"
            metadata = build_sqlite_fts_index(index_path, trials)
            retriever = load_sqlite_fts_retriever(index_path, trials)
            results = retriever.search("adult asthma inhaled corticosteroid", top_k=2)

        self.assertEqual(metadata["retriever"], "sqlite-fts5")
        self.assertEqual(results[0].nct_id, "NCT1")
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].score, 0)

    def test_query_quotes_fts_syntax_characters_safely(self) -> None:
        trials = [Trial(nct_id="NCT1", title="HER2 positive breast cancer")]
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "trials.sqlite"
            build_sqlite_fts_index(index_path, trials)
            retriever = load_sqlite_fts_retriever(index_path, trials)

            results = retriever.search('HER2-positive "breast" cancer OR', top_k=1)

        self.assertEqual(results[0].nct_id, "NCT1")

    def test_loader_rejects_corpus_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "trials.sqlite"
            build_sqlite_fts_index(index_path, [Trial(nct_id="NCT1", title="Asthma")])

            with self.assertRaisesRegex(ValueError, "corpus fingerprint"):
                load_sqlite_fts_retriever(
                    index_path,
                    [Trial(nct_id="NCT1", title="Different")],
                )

    def test_index_contains_requested_weighted_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "trials.sqlite"
            build_sqlite_fts_index(index_path, [Trial(nct_id="NCT1", title="Asthma")])
            with sqlite3.connect(index_path) as connection:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(trials_fts)")]

        self.assertEqual(
            columns,
            [
                "nct_id",
                "title",
                "brief_summary",
                "conditions",
                "interventions",
                "eligibility_criteria",
            ],
        )

    def test_weights_must_pin_every_supported_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing SQLite FTS5"):
            normalize_sqlite_fts_field_weights({"title": 1.0})

        self.assertEqual(
            normalize_sqlite_fts_field_weights(None),
            DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
        )


if __name__ == "__main__":
    unittest.main()
