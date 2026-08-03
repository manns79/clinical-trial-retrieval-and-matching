from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_trial_matching.evaluation.metrics import summarize_run
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
    run: dict[str, list[str]] = {}
    for row in rows:
        run.setdefault(row.topic_id, []).append(row.nct_id)
    return summarize_run(run, qrels_to_mapping(qrels))


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
        "metrics": evaluate_trec_run(rows, qrels),
        "topics": topics_count,
        "topics_with_results": len(topic_ids_with_results),
        "trials": trials_count,
        "qrels": len(qrels),
        "run_rows": len(rows),
    }
