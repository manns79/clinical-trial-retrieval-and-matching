from __future__ import annotations

import unittest

from clinical_trial_matching.models import Trial
from clinical_trial_matching.retrieval.bm25 import make_snippet, matched_terms, search_trials


class BM25SearchReportTest(unittest.TestCase):
    def test_search_trials_returns_traceable_result_records(self) -> None:
        trials = [
            Trial(
                nct_id="NCT1",
                title="Asthma inhaler study",
                status="RECRUITING",
                conditions=("Asthma",),
                interventions=("Inhaled corticosteroid",),
                eligibility_criteria="Adults with persistent asthma and wheezing.",
            ),
            Trial(
                nct_id="NCT2",
                title="Migraine prevention study",
                status="COMPLETED",
                conditions=("Migraine",),
                interventions=("Preventive therapy",),
            ),
        ]

        payload = search_trials(trials, query="persistent asthma inhaler", top_k=1)
        result = payload["results"][0]

        self.assertEqual(payload["query"], "persistent asthma inhaler")
        self.assertEqual(payload["corpus"], {"trials": 2, "unique_nct_ids": 2})
        self.assertEqual(result["nct_id"], "NCT1")
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["status"], "RECRUITING")
        self.assertEqual(result["conditions"], ["Asthma"])
        self.assertIn("asthma", result["matched_terms"])
        self.assertIn("persistent asthma", result["snippet"])

    def test_matched_terms_are_sorted_unique_intersections(self) -> None:
        self.assertEqual(matched_terms("asthma asthma adult", "adult asthma trial"), ["adult", "asthma"])

    def test_make_snippet_centers_first_query_match(self) -> None:
        text = "one two three four five six seven eight nine ten"
        snippet = make_snippet(text, query="seven", max_chars=25)

        self.assertIn("seven", snippet)
        self.assertTrue(snippet.startswith("... "))


if __name__ == "__main__":
    unittest.main()
