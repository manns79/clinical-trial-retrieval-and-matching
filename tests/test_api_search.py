from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.ingestion.clinicaltrials import trial_to_flat_record
from clinical_trial_matching.io import write_jsonl
from clinical_trial_matching.models import Trial


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "FastAPI is not installed")
class ApiSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        from clinical_trial_matching.api.main import load_trial_corpus

        load_trial_corpus.cache_clear()
        self.previous_corpus_path = os.environ.get("TRIAL_CORPUS_PATH")

    def tearDown(self) -> None:
        from clinical_trial_matching.api.main import load_trial_corpus

        if self.previous_corpus_path is None:
            os.environ.pop("TRIAL_CORPUS_PATH", None)
        else:
            os.environ["TRIAL_CORPUS_PATH"] = self.previous_corpus_path
        load_trial_corpus.cache_clear()

    def test_search_returns_traceable_bm25_records_from_configured_corpus(self) -> None:
        from clinical_trial_matching.api.main import SearchRequest, load_trial_corpus, search

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "trials.jsonl"
            write_jsonl(
                corpus_path,
                [
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT1",
                            title="Asthma inhaler study",
                            status="RECRUITING",
                            conditions=("Asthma",),
                            interventions=("Inhaled corticosteroid",),
                            eligibility_criteria="Adults with persistent asthma and wheezing.",
                        )
                    ),
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT2",
                            title="Migraine prevention study",
                            status="COMPLETED",
                            conditions=("Migraine",),
                        )
                    ),
                ],
            )
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            load_trial_corpus.cache_clear()

            response = search(
                SearchRequest(
                    query="adult persistent asthma inhaled corticosteroid",
                    top_k=1,
                )
            )

        payload = response.model_dump()
        self.assertEqual(payload["retriever"], "bm25")
        self.assertEqual(payload["corpus"], {"trials": 2, "unique_nct_ids": 2})
        self.assertEqual(payload["results"][0]["nct_id"], "NCT1")
        self.assertEqual(payload["results"][0]["rank"], 1)
        self.assertIn("asthma", payload["results"][0]["matched_terms"])
        self.assertIn("snippet", payload["results"][0])

    def test_get_trial_returns_normalized_trial_details(self) -> None:
        from clinical_trial_matching.api.main import get_trial, load_trial_corpus

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "trials.jsonl"
            write_jsonl(
                corpus_path,
                [
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT1",
                            title="Asthma inhaler study",
                            status="RECRUITING",
                            conditions=("Asthma",),
                            interventions=("Inhaled corticosteroid",),
                            eligibility_criteria="Adults with persistent asthma and wheezing.",
                            sex="ALL",
                            minimum_age="18 Years",
                            maximum_age="65 Years",
                            phases=("PHASE2",),
                            study_type="INTERVENTIONAL",
                            locations=("Boston, Massachusetts, United States",),
                            source={"kind": "test"},
                        )
                    )
                ],
            )
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            load_trial_corpus.cache_clear()

            response = get_trial("nct1")

        payload = response.model_dump()
        self.assertEqual(payload["nct_id"], "NCT1")
        self.assertEqual(payload["title"], "Asthma inhaler study")
        self.assertEqual(payload["conditions"], ["Asthma"])
        self.assertEqual(payload["interventions"], ["Inhaled corticosteroid"])
        self.assertEqual(payload["minimum_age"], "18 Years")
        self.assertEqual(payload["phases"], ["PHASE2"])
        self.assertEqual(payload["source"], {"kind": "test"})

    def test_get_trial_returns_404_when_nct_id_is_missing(self) -> None:
        from fastapi import HTTPException

        from clinical_trial_matching.api.main import get_trial, load_trial_corpus

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "trials.jsonl"
            write_jsonl(
                corpus_path,
                [
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT1",
                            title="Asthma inhaler study",
                        )
                    )
                ],
            )
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            load_trial_corpus.cache_clear()

            with self.assertRaises(HTTPException) as context:
                get_trial("NCT404")

        self.assertEqual(context.exception.status_code, 404)
        self.assertIn("Trial not found", context.exception.detail)

    def test_search_returns_503_when_configured_corpus_is_missing(self) -> None:
        from fastapi import HTTPException

        from clinical_trial_matching.api.main import SearchRequest, load_trial_corpus, search

        os.environ["TRIAL_CORPUS_PATH"] = "/tmp/clinical-trial-missing-corpus.jsonl"
        load_trial_corpus.cache_clear()

        with self.assertRaises(HTTPException) as context:
            search(SearchRequest(query="asthma"))

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("Trial corpus not found", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
