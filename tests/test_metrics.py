from __future__ import annotations

import unittest

from clinical_trial_matching.evaluation.metrics import mrr, ndcg_at_k, recall_at_k


class MetricsTest(unittest.TestCase):
    def test_recall_at_k(self) -> None:
        run = {"1": ["NCT2", "NCT1"]}
        qrels = {"1": {"NCT1": 2, "NCT3": 1}}
        self.assertEqual(recall_at_k(run, qrels, 2), 0.5)

    def test_mrr(self) -> None:
        run = {"1": ["NCT0", "NCT1"]}
        qrels = {"1": {"NCT1": 2}}
        self.assertEqual(mrr(run, qrels), 0.5)

    def test_ndcg_at_k_is_bounded(self) -> None:
        run = {"1": ["NCT1", "NCT2"]}
        qrels = {"1": {"NCT1": 2, "NCT2": 1}}
        self.assertGreaterEqual(ndcg_at_k(run, qrels, 2), 0.0)
        self.assertLessEqual(ndcg_at_k(run, qrels, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
