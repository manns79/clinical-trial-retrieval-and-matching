from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clinical_trial_matching.cli import build_trec_trial_corpus, unique_nct_ids_from_qrels
from clinical_trial_matching.ingestion.clinicaltrials import CtgovIdCorpusDownloadResult
from clinical_trial_matching.ingestion.trec import parse_qrels
from clinical_trial_matching.io import read_json, read_jsonl


class TrecTrialCorpusBuildTest(unittest.TestCase):
    def test_unique_nct_ids_from_qrels_preserves_first_seen_order(self) -> None:
        qrels = parse_qrels(Path("data/fixtures/qrels.sample.tsv"))

        self.assertEqual(
            unique_nct_ids_from_qrels(qrels),
            ["NCT99990001", "NCT99990002", "NCT99990003"],
        )

    def test_build_trec_trial_corpus_writes_artifacts(self) -> None:
        payload = {
            "source": "clinicaltrials_gov_v2",
            "requested_nct_ids": ["NCT99990001", "NCT99990002"],
            "found_nct_ids": ["NCT99990001"],
            "missing_nct_ids": ["NCT99990002"],
            "request_urls": ["https://clinicaltrials.gov/api/v2/studies?query.id=NCT99990001"],
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT99990001",
                            "briefTitle": "Synthetic Trial of Asthma Controller Therapy",
                        },
                        "conditionsModule": {"conditions": ["Asthma"]},
                    }
                }
            ],
        }
        result = CtgovIdCorpusDownloadResult(
            payload=payload,
            request_urls=("https://clinicaltrials.gov/api/v2/studies?query.id=NCT99990001",),
            requested_nct_ids=("NCT99990001", "NCT99990002"),
            found_nct_ids=("NCT99990001",),
            missing_nct_ids=("NCT99990002",),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_output = Path(tmpdir) / "raw.json"
            processed_output = Path(tmpdir) / "processed.jsonl"
            manifest_output = Path(tmpdir) / "manifest.json"
            report_output = Path(tmpdir) / "report.json"
            with patch("clinical_trial_matching.cli.fetch_ctgov_studies_by_ids", return_value=result):
                build_trec_trial_corpus(
                    qrels_path=Path("data/fixtures/qrels.sample.tsv"),
                    raw_output=raw_output,
                    processed_output=processed_output,
                    manifest_output=manifest_output,
                    report_output=report_output,
                    dataset="trec_clinical_trials",
                    year=2021,
                    batch_size=100,
                    limit=2,
                    delay_seconds=0.0,
                    max_retries=5,
                    retry_initial_delay_seconds=2.0,
                    retry_max_delay_seconds=60.0,
                    base_url="https://clinicaltrials.gov/api/v2",
                    timeout_seconds=30.0,
                )

            raw = read_json(raw_output)
            processed = read_jsonl(processed_output)
            manifest = read_json(manifest_output)
            report = read_json(report_output)

        self.assertEqual(raw["requested_nct_ids"], ["NCT99990001", "NCT99990002"])
        self.assertEqual(processed[0]["nct_id"], "NCT99990001")
        self.assertEqual(manifest["metadata"]["found_nct_ids"], "1")
        self.assertEqual(report["requested_nct_ids"], 2)
        self.assertEqual(report["found_nct_ids"], 1)
        self.assertEqual(report["missing_nct_ids_sample"], ["NCT99990002"])


if __name__ == "__main__":
    unittest.main()
