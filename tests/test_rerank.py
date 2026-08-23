from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from clinical_trial_matching.models import Topic, Trial
from clinical_trial_matching.retrieval.rerank import rerank_topic


class FakePairScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs: Any, *, batch_size: int) -> Any:
        self.calls.append((list(pairs), batch_size))
        return self.scores


class RerankTest(unittest.TestCase):
    def test_reranks_only_candidate_window_and_preserves_baseline_tail(self) -> None:
        scorer = FakePairScorer([0.1, 0.9])
        clock_values = iter([0.0, 0.001, 0.011, 0.012])

        result = rerank_topic(
            topic=Topic(topic_id="1", text="adult asthma"),
            baseline_nct_ids=("NCT1", "NCT2", "NCT3"),
            candidate_trials=(
                Trial(nct_id="NCT1", title="Asthma observation"),
                Trial(nct_id="NCT2", title="Adult asthma treatment"),
            ),
            reranker=scorer,
            candidate_depth=2,
            top_k=3,
            text_representation="title",
            batch_size=8,
            run_name="fixture_reranker",
            clock=lambda: next(clock_values),
        )

        self.assertEqual([row.nct_id for row in result.rows], ["NCT2", "NCT1", "NCT3"])
        self.assertEqual(result.rows[2].score, 1.0)
        self.assertEqual(scorer.calls[0][1], 8)
        self.assertAlmostEqual(result.inference_ms, 10.0)

    def test_rounded_score_ties_resolve_by_nct_id(self) -> None:
        scorer = FakePairScorer([0.5000001, 0.5000002])
        clock_values = iter([0.0, 0.0, 0.0, 0.0])

        result = rerank_topic(
            topic=Topic(topic_id="1", text="query"),
            baseline_nct_ids=("NCT2", "NCT1"),
            candidate_trials=(
                Trial(nct_id="NCT2", title="Second"),
                Trial(nct_id="NCT1", title="First"),
            ),
            reranker=scorer,
            candidate_depth=2,
            top_k=2,
            text_representation="title",
            batch_size=2,
            run_name="fixture_reranker",
            clock=lambda: next(clock_values),
        )

        self.assertEqual([row.nct_id for row in result.rows], ["NCT1", "NCT2"])

    def test_rejects_candidate_metadata_in_wrong_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "Candidate trial order"):
            rerank_topic(
                topic=Topic(topic_id="1", text="query"),
                baseline_nct_ids=("NCT1", "NCT2"),
                candidate_trials=(Trial(nct_id="NCT2", title="Wrong"),),
                reranker=FakePairScorer([0.5]),
                candidate_depth=1,
                top_k=2,
                text_representation="title",
                batch_size=2,
                run_name="fixture_reranker",
                clock=lambda: 0.0,
            )


if __name__ == "__main__":
    unittest.main()
