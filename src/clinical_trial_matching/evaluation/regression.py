from __future__ import annotations

from typing import Any

from clinical_trial_matching.evaluation.metrics import summarize_run
from clinical_trial_matching.ingestion.trec import qrels_to_mapping
from clinical_trial_matching.models import Qrel, Topic, Trial
from clinical_trial_matching.retrieval.bm25 import BM25Retriever


DEFAULT_THRESHOLDS = {
    "recall_at_100": 1.0,
    "mrr": 1.0,
    "ndcg_at_10": 1.0,
}


def run_bm25_regression_check(
    *,
    trials: list[Trial],
    topics: list[Topic],
    qrels: list[Qrel],
    thresholds: dict[str, float] | None = None,
    top_k: int = 100,
) -> dict[str, Any]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    retriever = BM25Retriever(trials)
    run = {
        topic.topic_id: [result.nct_id for result in retriever.search(topic.text, top_k=top_k)]
        for topic in topics
    }
    metrics = summarize_run(run, qrels_to_mapping(qrels))
    failures = [
        {
            "metric": metric,
            "observed": metrics.get(metric, 0.0),
            "threshold": threshold,
        }
        for metric, threshold in thresholds.items()
        if metrics.get(metric, 0.0) < threshold
    ]
    return {
        "check": "bm25_regression",
        "status": "pass" if not failures else "fail",
        "metrics": metrics,
        "thresholds": thresholds,
        "failures": failures,
        "run": run,
        "topics": len(topics),
        "trials": len(trials),
        "qrels": len(qrels),
    }
