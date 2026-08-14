from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from clinical_trial_matching.evaluation.trec import TrecRunRow
from clinical_trial_matching.models import SearchResult

RRF_RETRIEVER_NAME = "reciprocal-rank-fusion"


@dataclass(frozen=True)
class RankedRun:
    name: str
    weight: float
    rankings: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class RankedResults:
    name: str
    weight: float
    results: tuple[SearchResult, ...]


@dataclass(frozen=True)
class FusedResults:
    results: tuple[SearchResult, ...]
    component_ranks: dict[str, dict[str, int]]


def reciprocal_rank_fuse_results(
    runs: list[RankedResults],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
    candidate_depth: int = 100,
) -> FusedResults:
    if len(runs) < 2:
        raise ValueError("Reciprocal-rank fusion requires at least two component rankings")
    if rrf_k < 1 or top_k < 1 or candidate_depth < 1:
        raise ValueError("RRF k, top-k, and candidate depth must be positive")
    if any(run.weight <= 0 for run in runs):
        raise ValueError("RRF component weights must be positive")
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("RRF component names must be unique")

    scores: dict[str, float] = defaultdict(float)
    best_ranks: dict[str, int] = {}
    titles: dict[str, str] = {}
    component_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for run in runs:
        for rank, result in enumerate(run.results[:candidate_depth], start=1):
            scores[result.nct_id] += run.weight / (rrf_k + rank)
            best_ranks[result.nct_id] = min(best_ranks.get(result.nct_id, rank), rank)
            titles[result.nct_id] = result.title
            component_ranks[result.nct_id][run.name] = rank

    ranked_ids = sorted(
        scores,
        key=lambda nct_id: (-scores[nct_id], best_ranks[nct_id], nct_id),
    )[:top_k]
    results = tuple(
        SearchResult(
            nct_id=nct_id,
            score=scores[nct_id],
            rank=rank,
            title=titles[nct_id],
        )
        for rank, nct_id in enumerate(ranked_ids, start=1)
    )
    return FusedResults(
        results=results,
        component_ranks={nct_id: component_ranks[nct_id] for nct_id in ranked_ids},
    )


def read_trec_rankings(path: Path) -> dict[str, tuple[str, ...]]:
    rankings: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 6:
                raise ValueError(f"Invalid TREC run row at {path}:{line_number}")
            topic_id, iteration, nct_id, rank_text, score_text, _run_name = parts
            if iteration != "Q0":
                raise ValueError(f"Invalid TREC iteration at {path}:{line_number}: {iteration}")
            try:
                rank = int(rank_text)
                float(score_text)
            except ValueError as exc:
                raise ValueError(f"Invalid TREC rank/score at {path}:{line_number}") from exc
            if rank < 1:
                raise ValueError(f"TREC rank must be positive at {path}:{line_number}")
            if nct_id in seen[topic_id]:
                raise ValueError(
                    f"Duplicate NCT ID {nct_id} for topic {topic_id} in {path}"
                )
            seen[topic_id].add(nct_id)
            rankings[topic_id].append((rank, nct_id))

    if not rankings:
        raise ValueError(f"TREC run contains no rows: {path}")
    return {
        topic_id: tuple(nct_id for _rank, nct_id in sorted(rows))
        for topic_id, rows in rankings.items()
    }


def reciprocal_rank_fusion(
    runs: list[RankedRun],
    *,
    run_name: str,
    rrf_k: int = 60,
    top_k: int = 100,
    candidate_depth: int = 100,
) -> list[TrecRunRow]:
    if len(runs) < 2:
        raise ValueError("Reciprocal-rank fusion requires at least two component runs")
    if not run_name.strip():
        raise ValueError("Hybrid run name cannot be empty")
    if rrf_k < 1 or top_k < 1 or candidate_depth < 1:
        raise ValueError("RRF k, top-k, and candidate depth must be positive")
    if any(run.weight <= 0 for run in runs):
        raise ValueError("RRF component weights must be positive")
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("RRF component names must be unique")

    topic_sets = [set(run.rankings) for run in runs]
    if any(topic_ids != topic_sets[0] for topic_ids in topic_sets[1:]):
        raise ValueError("RRF component runs must contain identical topic sets")

    rows: list[TrecRunRow] = []
    for topic_id in sorted(topic_sets[0], key=_topic_sort_key):
        scores: dict[str, float] = defaultdict(float)
        best_ranks: dict[str, int] = {}
        for run in runs:
            for rank, nct_id in enumerate(
                run.rankings[topic_id][:candidate_depth],
                start=1,
            ):
                scores[nct_id] += run.weight / (rrf_k + rank)
                best_ranks[nct_id] = min(best_ranks.get(nct_id, rank), rank)
        ranked_ids = sorted(
            scores,
            key=lambda nct_id: (-scores[nct_id], best_ranks[nct_id], nct_id),
        )[:top_k]
        rows.extend(
            TrecRunRow(
                topic_id=topic_id,
                nct_id=nct_id,
                rank=rank,
                score=scores[nct_id],
                run_name=run_name,
            )
            for rank, nct_id in enumerate(ranked_ids, start=1)
        )
    return rows


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic_id))
    except ValueError:
        return (1, topic_id)
