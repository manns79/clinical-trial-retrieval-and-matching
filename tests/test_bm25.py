from __future__ import annotations

import unittest

from clinical_trial_matching.models import Trial
from clinical_trial_matching.retrieval.bm25 import (
    BM25Retriever,
    FieldAwareBM25Retriever,
    normalized_field_weights,
    search_trials,
    tokenize,
)


class BM25RetrieverTest(unittest.TestCase):
    def test_tokenize_lowercases_and_removes_punctuation(self) -> None:
        self.assertEqual(tokenize("Type-2 Diabetes, HbA1c!"), ["type", "2", "diabetes", "hba1c"])

    def test_search_ranks_matching_trial_first(self) -> None:
        trials = [
            Trial(nct_id="NCT1", title="Asthma inhaler study", conditions=("Asthma",)),
            Trial(nct_id="NCT2", title="Migraine prevention study", conditions=("Migraine",)),
        ]
        results = BM25Retriever(trials).search("asthma wheezing", top_k=2)
        self.assertEqual(results[0].nct_id, "NCT1")

    def test_field_aware_search_can_boost_structured_conditions(self) -> None:
        trials = [
            Trial(
                nct_id="NCT_TITLE",
                title="Asthma asthma asthma study",
                conditions=("Migraine",),
            ),
            Trial(
                nct_id="NCT_CONDITION",
                title="Respiratory study",
                conditions=("Asthma",),
            ),
        ]
        weights = {field: 0.0 for field in normalized_field_weights()}
        weights["conditions"] = 10.0

        results = FieldAwareBM25Retriever(trials, field_weights=weights).search("asthma", top_k=2)

        self.assertEqual(results[0].nct_id, "NCT_CONDITION")

    def test_search_trials_defaults_to_fielded_bm25(self) -> None:
        payload = search_trials(
            [Trial(nct_id="NCT1", title="Asthma study", conditions=("Asthma",))],
            query="asthma",
            top_k=1,
        )

        self.assertEqual(payload["retriever"], "fielded-bm25")
        self.assertIn("conditions", payload["parameters"]["field_weights"])


if __name__ == "__main__":
    unittest.main()
