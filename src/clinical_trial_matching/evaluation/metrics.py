from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def precision_at_k(run: Run, qrels: Qrels, k: int, min_relevance: int = 1) -> float:
    scores: list[float] = []
    for topic_id, ranked_ids in run.items():
        judged = qrels.get(topic_id, {})
        top_ids = ranked_ids[:k]
        if not top_ids:
            scores.append(0.0)
            continue
        relevant = sum(1 for nct_id in top_ids if judged.get(nct_id, 0) >= min_relevance)
        scores.append(relevant / len(top_ids))
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(run: Run, qrels: Qrels, k: int, min_relevance: int = 1) -> float:
    scores: list[float] = []
    for topic_id, judged in qrels.items():
        relevant_ids = {nct_id for nct_id, relevance in judged.items() if relevance >= min_relevance}
        if not relevant_ids:
            continue
        retrieved = set(run.get(topic_id, [])[:k])
        scores.append(len(relevant_ids & retrieved) / len(relevant_ids))
    return sum(scores) / len(scores) if scores else 0.0


def mrr(run: Run, qrels: Qrels, min_relevance: int = 1) -> float:
    reciprocal_ranks: list[float] = []
    for topic_id, ranked_ids in run.items():
        judged = qrels.get(topic_id, {})
        rank_score = 0.0
        for rank, nct_id in enumerate(ranked_ids, start=1):
            if judged.get(nct_id, 0) >= min_relevance:
                rank_score = 1 / rank
                break
        reciprocal_ranks.append(rank_score)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg_at_k(run: Run, qrels: Qrels, k: int) -> float:
    scores: list[float] = []
    for topic_id, ranked_ids in run.items():
        judged = qrels.get(topic_id, {})
        gains = [judged.get(nct_id, 0) for nct_id in ranked_ids[:k]]
        ideal_gains = sorted(judged.values(), reverse=True)[:k]
        ideal = _dcg(ideal_gains)
        scores.append(_dcg(gains) / ideal if ideal > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _dcg(gains: Sequence[int]) -> float:
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))


def summarize_run(run: Run, qrels: Qrels) -> dict[str, float]:
    return {
        "precision_at_10": precision_at_k(run, qrels, 10),
        "recall_at_100": recall_at_k(run, qrels, 100),
        "mrr": mrr(run, qrels),
        "ndcg_at_10": ndcg_at_k(run, qrels, 10),
        "ndcg_at_100": ndcg_at_k(run, qrels, 100),
    }
