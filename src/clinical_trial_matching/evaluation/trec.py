from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_trial_matching.evaluation.metrics import summarize_binary_run, summarize_run
from clinical_trial_matching.ingestion.trec import qrels_to_mapping
from clinical_trial_matching.models import Qrel, Topic, Trial
from clinical_trial_matching.retrieval.bm25 import BM25Retriever


@dataclass(frozen=True)
class TrecRunRow:
    topic_id: str
    nct_id: str
    rank: int
    score: float
    run_name: str

    def to_trec_line(self) -> str:
        return f"{self.topic_id} Q0 {self.nct_id} {self.rank} {self.score:.6f} {self.run_name}"


def build_bm25_trec_run(
    *,
    trials: Iterable[Trial],
    topics: Iterable[Topic],
    run_name: str,
    top_k: int = 100,
) -> list[TrecRunRow]:
    if top_k < 1:
        raise ValueError("Top-K must be at least 1")
    if not run_name.strip():
        raise ValueError("Run name cannot be empty")

    retriever = BM25Retriever(trials)
    rows: list[TrecRunRow] = []
    for topic in topics:
        for result in retriever.search(topic.text, top_k=top_k):
            rows.append(
                TrecRunRow(
                    topic_id=topic.topic_id,
                    nct_id=result.nct_id,
                    rank=result.rank,
                    score=result.score,
                    run_name=run_name,
                )
            )
    return rows


def write_trec_run(path: Path, rows: Iterable[TrecRunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.to_trec_line() + "\n")


def evaluate_trec_run(rows: Iterable[TrecRunRow], qrels: list[Qrel]) -> dict[str, float]:
    return summarize_run(_rows_to_run(rows), qrels_to_mapping(qrels))


def evaluate_trec_binary_views(rows: Iterable[TrecRunRow], qrels: list[Qrel]) -> dict[str, dict[str, float]]:
    run = _rows_to_run(rows)
    qrels_mapping = qrels_to_mapping(qrels)
    return {
        "excluded_or_eligible": summarize_binary_run(run, qrels_mapping, min_relevance=1),
        "eligible_only": summarize_binary_run(run, qrels_mapping, min_relevance=2),
    }


def bm25_trec_evaluation_report(
    *,
    rows: list[TrecRunRow],
    qrels: list[Qrel],
    run_name: str,
    top_k: int,
    topics_count: int,
    trials_count: int,
) -> dict[str, Any]:
    topic_ids_with_results = {row.topic_id for row in rows}
    return {
        "run_name": run_name,
        "retriever": "bm25",
        "top_k": top_k,
        "metrics": evaluate_trec_binary_views(rows, qrels),
        "graded_metrics": evaluate_trec_run(rows, qrels),
        "topics": topics_count,
        "topics_with_results": len(topic_ids_with_results),
        "trials": trials_count,
        "qrels": len(qrels),
        "run_rows": len(rows),
    }


def bm25_trec_topic_diagnostics(
    *,
    rows: list[TrecRunRow],
    qrels: list[Qrel],
    topics: list[Topic],
    run_name: str,
    top_k: int,
    weak_recall_threshold: float = 0.05,
) -> dict[str, Any]:
    run = _rows_to_run(rows)
    qrels_mapping = qrels_to_mapping(qrels)
    topic_diagnostics = [
        _topic_diagnostic(
            topic_id=topic.topic_id,
            ranked_ids=run.get(topic.topic_id, []),
            judged=qrels_mapping.get(topic.topic_id, {}),
            top_k=top_k,
            weak_recall_threshold=weak_recall_threshold,
        )
        for topic in topics
    ]
    weak_topics = sorted(
        [
            {
                "topic_id": item["topic_id"],
                "eligible_recall_at_k": item["eligible_recall_at_k"],
                "excluded_or_eligible_recall_at_k": item["excluded_or_eligible_recall_at_k"],
                "first_eligible_rank": item["first_eligible_rank"],
                "weak_retrieval_reasons": item["weak_retrieval_reasons"],
            }
            for item in topic_diagnostics
            if item["weak_retrieval"]
        ],
        key=_weak_topic_sort_key,
    )
    return {
        "run_name": run_name,
        "retriever": "bm25",
        "top_k": top_k,
        "weak_retrieval_policy": {
            "eligible_recall_at_k_below": weak_recall_threshold,
            "missing_first_eligible_rank": True,
        },
        "topics": topic_diagnostics,
        "weak_topics_count": len(weak_topics),
        "weak_topics": weak_topics,
    }


def _rows_to_run(rows: Iterable[TrecRunRow]) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for row in rows:
        run.setdefault(row.topic_id, []).append(row.nct_id)
    return run


def _topic_diagnostic(
    *,
    topic_id: str,
    ranked_ids: list[str],
    judged: dict[str, int],
    top_k: int,
    weak_recall_threshold: float,
) -> dict[str, Any]:
    top_ids = ranked_ids[:top_k]
    eligible_ids = {nct_id for nct_id, relevance in judged.items() if relevance >= 2}
    excluded_or_eligible_ids = {nct_id for nct_id, relevance in judged.items() if relevance >= 1}
    retrieved_ids = set(top_ids)
    eligible_retrieved = eligible_ids & retrieved_ids
    excluded_or_eligible_retrieved = excluded_or_eligible_ids & retrieved_ids
    eligible_recall = _topic_recall(eligible_retrieved, eligible_ids)
    excluded_or_eligible_recall = _topic_recall(
        excluded_or_eligible_retrieved,
        excluded_or_eligible_ids,
    )
    first_eligible_rank = _first_rank(top_ids, eligible_ids)
    weak_reasons = _weak_retrieval_reasons(
        eligible_total=len(eligible_ids),
        eligible_recall=eligible_recall,
        first_eligible_rank=first_eligible_rank,
        weak_recall_threshold=weak_recall_threshold,
    )
    return {
        "topic_id": topic_id,
        "retrieved": len(top_ids),
        "judged": len(judged),
        "eligible_total": len(eligible_ids),
        "excluded_or_eligible_total": len(excluded_or_eligible_ids),
        "eligible_retrieved_at_k": len(eligible_retrieved),
        "excluded_or_eligible_retrieved_at_k": len(excluded_or_eligible_retrieved),
        "eligible_recall_at_k": eligible_recall,
        "excluded_or_eligible_recall_at_k": excluded_or_eligible_recall,
        "first_eligible_rank": first_eligible_rank,
        "first_excluded_or_eligible_rank": _first_rank(top_ids, excluded_or_eligible_ids),
        "weak_retrieval": bool(weak_reasons),
        "weak_retrieval_reasons": weak_reasons,
    }


def _topic_recall(retrieved_relevant_ids: set[str], relevant_ids: set[str]) -> float | None:
    if not relevant_ids:
        return None
    return len(retrieved_relevant_ids) / len(relevant_ids)


def _first_rank(ranked_ids: list[str], relevant_ids: set[str]) -> int | None:
    for rank, nct_id in enumerate(ranked_ids, start=1):
        if nct_id in relevant_ids:
            return rank
    return None


def _weak_retrieval_reasons(
    *,
    eligible_total: int,
    eligible_recall: float | None,
    first_eligible_rank: int | None,
    weak_recall_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if eligible_total == 0:
        reasons.append("no_eligible_qrels")
    elif eligible_recall is not None and eligible_recall < weak_recall_threshold:
        reasons.append("low_eligible_recall")
    if eligible_total > 0 and first_eligible_rank is None:
        reasons.append("no_eligible_result_in_top_k")
    return reasons


def _weak_topic_sort_key(topic: dict[str, Any]) -> tuple[float, int, str]:
    recall = topic["eligible_recall_at_k"]
    first_rank = topic["first_eligible_rank"]
    return (
        recall if recall is not None else -1.0,
        first_rank if first_rank is not None else 1_000_000,
        topic["topic_id"],
    )
