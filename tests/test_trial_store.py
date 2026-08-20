from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.ingestion.clinicaltrials import trial_to_flat_record
from clinical_trial_matching.io import write_jsonl
from clinical_trial_matching.models import Trial
from clinical_trial_matching.trial_store import build_trial_store, load_trial_store


class SQLiteTrialStoreTest(unittest.TestCase):
    def test_build_load_and_ordered_batch_lookup(self) -> None:
        trials = [
            Trial(
                nct_id="NCT1",
                title="Asthma study",
                conditions=("Asthma",),
                source={"kind": "fixture"},
            ),
            Trial(nct_id="NCT2", title="Diabetes study"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_path = root / "trials.jsonl"
            store_path = root / "trials.sqlite"
            write_jsonl(corpus_path, (trial_to_flat_record(trial) for trial in trials))
            metadata = build_trial_store(store_path, iter(trials), corpus_path=corpus_path)
            store = load_trial_store(store_path, corpus_path=corpus_path)

            observed = store.get_many(["nct2", "NCT1"])
            single = store.get("nct1")

        self.assertEqual(metadata["corpus"]["trials"], 2)
        self.assertEqual(store.count, 2)
        self.assertEqual([trial.nct_id for trial in observed], ["NCT2", "NCT1"])
        self.assertIsNotNone(single)
        assert single is not None
        self.assertEqual(single.source, {"kind": "fixture"})

    def test_load_rejects_changed_corpus_file(self) -> None:
        trial = Trial(nct_id="NCT1", title="Asthma study")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_path = root / "trials.jsonl"
            store_path = root / "trials.sqlite"
            write_jsonl(corpus_path, [trial_to_flat_record(trial)])
            build_trial_store(store_path, [trial], corpus_path=corpus_path)
            write_jsonl(
                corpus_path,
                [trial_to_flat_record(Trial(nct_id="NCT1", title="Changed"))],
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_trial_store(store_path, corpus_path=corpus_path)

    def test_dense_id_order_validation_detects_drift(self) -> None:
        trials = [Trial(nct_id="NCT1", title="One"), Trial(nct_id="NCT2", title="Two")]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_path = root / "trials.jsonl"
            store_path = root / "trials.sqlite"
            write_jsonl(corpus_path, (trial_to_flat_record(trial) for trial in trials))
            build_trial_store(store_path, trials, corpus_path=corpus_path)
            store = load_trial_store(store_path, corpus_path=corpus_path)

            with self.assertRaisesRegex(ValueError, "order"):
                store.validate_nct_id_order(("NCT2", "NCT1"))


if __name__ == "__main__":
    unittest.main()
