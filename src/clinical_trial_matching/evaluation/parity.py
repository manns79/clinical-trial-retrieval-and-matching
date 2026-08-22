from __future__ import annotations

from pathlib import Path
from typing import Any

from clinical_trial_matching.retrieval.hybrid import read_trec_rankings


def trec_run_parity_report(
    baseline_path: Path,
    candidate_path: Path,
    *,
    depth: int,
) -> dict[str, Any]:
    if depth < 1:
        raise ValueError("Parity depth must be at least 1")
    baseline = read_trec_rankings(baseline_path)
    candidate = read_trec_rankings(candidate_path)
    baseline_topics = set(baseline)
    candidate_topics = set(candidate)
    shared_topics = sorted(baseline_topics & candidate_topics, key=_topic_sort_key)
    mismatches = []
    matching_topics = 0
    for topic_id in shared_topics:
        baseline_ids = baseline[topic_id][:depth]
        candidate_ids = candidate[topic_id][:depth]
        if baseline_ids == candidate_ids:
            matching_topics += 1
            continue
        first_mismatch = next(
            (
                rank
                for rank, values in enumerate(
                    zip(baseline_ids, candidate_ids, strict=False),
                    start=1,
                )
                if values[0] != values[1]
            ),
            min(len(baseline_ids), len(candidate_ids)) + 1,
        )
        mismatches.append(
            {
                "topic_id": topic_id,
                "first_mismatch_rank": first_mismatch,
                "baseline_nct_id": (
                    baseline_ids[first_mismatch - 1]
                    if first_mismatch <= len(baseline_ids)
                    else None
                ),
                "candidate_nct_id": (
                    candidate_ids[first_mismatch - 1]
                    if first_mismatch <= len(candidate_ids)
                    else None
                ),
            }
        )

    missing_candidate_topics = sorted(baseline_topics - candidate_topics, key=_topic_sort_key)
    unexpected_candidate_topics = sorted(candidate_topics - baseline_topics, key=_topic_sort_key)
    passed = not mismatches and not missing_candidate_topics and not unexpected_candidate_topics
    return {
        "schema_version": 1,
        "comparison": "exact_nct_id_order",
        "depth": depth,
        "baseline_run": str(baseline_path),
        "candidate_run": str(candidate_path),
        "passed": passed,
        "topics": {
            "baseline": len(baseline_topics),
            "candidate": len(candidate_topics),
            "matching": matching_topics,
            "mismatched": len(mismatches),
            "missing_from_candidate": missing_candidate_topics,
            "unexpected_in_candidate": unexpected_candidate_topics,
        },
        "mismatch_sample": mismatches[:20],
    }


def _topic_sort_key(topic_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(topic_id))
    except ValueError:
        return (1, topic_id)
