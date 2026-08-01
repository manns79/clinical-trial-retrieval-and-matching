from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.ingestion.clinicaltrials import (
    parse_studies_json,
    trial_from_ctgov_v2_record,
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.io import read_jsonl, write_jsonl


FIXTURES = Path("data/fixtures")


class ClinicalTrialsIngestionTest(unittest.TestCase):
    def test_parse_ctgov_v2_response(self) -> None:
        trials = parse_studies_json(FIXTURES / "ctgov_v2_studies.sample.json")

        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0].nct_id, "NCT99991001")
        self.assertEqual(trials[0].title, "Synthetic Asthma Controller Therapy Study")
        self.assertEqual(trials[0].status, "RECRUITING")
        self.assertEqual(trials[0].conditions, ("Asthma",))
        self.assertEqual(
            trials[0].interventions,
            ("Inhaled corticosteroid", "Asthma education"),
        )
        self.assertEqual(trials[0].sex, "ALL")
        self.assertEqual(trials[0].minimum_age, "18 Years")
        self.assertEqual(trials[0].maximum_age, "65 Years")
        self.assertEqual(trials[0].phases, ("PHASE2",))
        self.assertEqual(trials[0].study_type, "INTERVENTIONAL")
        self.assertEqual(
            trials[0].locations,
            ("Synthetic Medical Center, Boston, Massachusetts, United States",),
        )

    def test_parse_ctgov_v2_single_record_with_missing_optional_fields(self) -> None:
        trial = trial_from_ctgov_v2_record(
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT99991003",
                        "officialTitle": "Synthetic Official Title",
                    }
                }
            }
        )

        self.assertEqual(trial.nct_id, "NCT99991003")
        self.assertEqual(trial.title, "Synthetic Official Title")
        self.assertEqual(trial.conditions, ())
        self.assertEqual(trial.interventions, ())
        self.assertEqual(trial.locations, ())

    def test_flat_record_round_trip_preserves_structured_fields(self) -> None:
        trial = parse_studies_json(FIXTURES / "ctgov_v2_studies.sample.json")[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trials.jsonl"
            write_jsonl(path, [trial_to_flat_record(trial)])
            loaded = trial_from_flat_record(read_jsonl(path)[0])

        self.assertEqual(loaded, trial)

    def test_searchable_text_includes_structured_and_eligibility_fields(self) -> None:
        trial = parse_studies_json(FIXTURES / "ctgov_v2_studies.sample.json")[0]

        self.assertIn("persistent asthma", trial.searchable_text)
        self.assertIn("Inhaled corticosteroid", trial.searchable_text)
        self.assertIn("18 Years", trial.searchable_text)
        self.assertIn("PHASE2", trial.searchable_text)


if __name__ == "__main__":
    unittest.main()
