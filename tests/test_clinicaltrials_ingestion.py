from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clinical_trial_matching.cli import download_ctgov_studies
from clinical_trial_matching.ingestion.clinicaltrials import (
    CtgovDownloadResult,
    fetch_ctgov_studies,
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

    def test_fetch_ctgov_studies_uses_query_and_status_parameters(self) -> None:
        class FakeResponse:
            url = "https://clinicaltrials.gov/api/v2/studies?query.term=asthma"

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "studies": [
                        {
                            "protocolSection": {
                                "identificationModule": {"nctId": "NCT99991004"}
                            }
                        }
                    ],
                    "totalCount": 123,
                    "nextPageToken": "next-token",
                }

        class FakeClient:
            def __init__(self) -> None:
                self.params: dict[str, str | int] = {}

            def get(self, url: str, params: dict[str, str | int]) -> FakeResponse:
                assert url == "https://clinicaltrials.gov/api/v2/studies"
                self.params = params
                return FakeResponse()

        client = FakeClient()
        result = fetch_ctgov_studies(query="asthma", status="RECRUITING", page_size=25, client=client)

        self.assertEqual(client.params["query.term"], "asthma")
        self.assertEqual(client.params["filter.overallStatus"], "RECRUITING")
        self.assertEqual(client.params["pageSize"], 25)
        self.assertEqual(client.params["countTotal"], "true")
        self.assertEqual(result.study_count, 1)
        self.assertEqual(result.total_count, 123)
        self.assertEqual(result.next_page_token, "next-token")
        self.assertIn("query.term=asthma", result.request_url)

    def test_fetch_ctgov_studies_rejects_large_page_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 1000"):
            fetch_ctgov_studies(query="asthma", page_size=1001)

    def test_download_ctgov_studies_writes_raw_processed_and_manifest(self) -> None:
        payload = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT99991005",
                            "briefTitle": "Synthetic Live Asthma Study",
                        },
                        "conditionsModule": {"conditions": ["Asthma"]},
                    }
                }
            ],
            "totalCount": 1,
        }
        result = CtgovDownloadResult(
            payload=payload,
            request_url="https://clinicaltrials.gov/api/v2/studies?query.term=asthma",
            study_count=1,
            total_count=1,
            next_page_token="",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_output = Path(tmpdir) / "raw.json"
            processed_output = Path(tmpdir) / "processed.jsonl"
            manifest_output = Path(tmpdir) / "manifest.json"
            with patch("clinical_trial_matching.cli.fetch_ctgov_studies", return_value=result):
                download_ctgov_studies(
                    query="asthma",
                    status="RECRUITING",
                    page_size=25,
                    raw_output=raw_output,
                    manifest_output=manifest_output,
                    processed_output=processed_output,
                    base_url="https://clinicaltrials.gov/api/v2",
                    timeout_seconds=30.0,
                )

            parsed = parse_studies_json(raw_output)
            processed_rows = read_jsonl(processed_output)
            manifest = json.loads(manifest_output.read_text(encoding="utf-8"))

        self.assertEqual(parsed[0].nct_id, "NCT99991005")
        self.assertEqual(processed_rows[0]["nct_id"], "NCT99991005")
        self.assertEqual(manifest["metadata"]["query"], "asthma")
        self.assertEqual(manifest["metadata"]["study_count"], "1")


if __name__ == "__main__":
    unittest.main()
