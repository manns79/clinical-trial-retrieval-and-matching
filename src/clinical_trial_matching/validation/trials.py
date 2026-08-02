from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from clinical_trial_matching.models import Trial


def summarize_trial_corpus(
    trials: Iterable[Trial],
    *,
    sample_size: int = 5,
    top_n: int = 10,
) -> dict[str, Any]:
    trial_list = list(trials)
    nct_counts = Counter(trial.nct_id for trial in trial_list)
    duplicate_ids = [
        {"nct_id": nct_id, "count": count}
        for nct_id, count in sorted(nct_counts.items())
        if count > 1
    ]

    missing_eligibility = [
        trial.nct_id for trial in trial_list if not trial.eligibility_criteria.strip()
    ]
    missing_conditions = [trial.nct_id for trial in trial_list if not trial.conditions]
    missing_interventions = [trial.nct_id for trial in trial_list if not trial.interventions]

    total = len(trial_list)
    return {
        "trials": total,
        "unique_nct_ids": len(nct_counts),
        "duplicate_nct_ids": duplicate_ids,
        "status_distribution": dict(Counter(_label_or_missing(trial.status) for trial in trial_list)),
        "missing_eligibility_criteria": _missing_summary(missing_eligibility, total),
        "condition_coverage": _coverage_summary(
            missing_ids=missing_conditions,
            total=total,
            top_values=Counter(
                condition for trial in trial_list for condition in trial.conditions if condition
            ),
            top_n=top_n,
            value_key="condition",
        ),
        "intervention_coverage": _coverage_summary(
            missing_ids=missing_interventions,
            total=total,
            top_values=Counter(
                intervention for trial in trial_list for intervention in trial.interventions if intervention
            ),
            top_n=top_n,
            value_key="intervention",
        ),
        "sample_records": [
            {
                "nct_id": trial.nct_id,
                "title": trial.title,
                "status": trial.status,
                "conditions": list(trial.conditions),
                "interventions": list(trial.interventions),
                "has_eligibility_criteria": bool(trial.eligibility_criteria.strip()),
            }
            for trial in trial_list[:sample_size]
        ],
    }


def _missing_summary(missing_ids: list[str], total: int) -> dict[str, Any]:
    return {
        "count": len(missing_ids),
        "rate": _rate(len(missing_ids), total),
        "nct_ids_sample": missing_ids[:10],
    }


def _coverage_summary(
    *,
    missing_ids: list[str],
    total: int,
    top_values: Counter[str],
    top_n: int,
    value_key: str,
) -> dict[str, Any]:
    with_values = total - len(missing_ids)
    return {
        "with_values": with_values,
        "missing_values": len(missing_ids),
        "coverage_rate": _rate(with_values, total),
        "missing_nct_ids_sample": missing_ids[:10],
        f"top_{value_key}s": [
            {value_key: value, "count": count}
            for value, count in top_values.most_common(top_n)
        ],
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _label_or_missing(value: str) -> str:
    return value if value else "MISSING"
