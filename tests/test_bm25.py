from __future__ import annotations

import unittest

from clinical_trial_matching.models import Trial
from clinical_trial_matching.retrieval.bm25 import BM25Retriever, tokenize


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


if __name__ == "__main__":
    unittest.main()
