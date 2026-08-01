from __future__ import annotations

from typing import Any

from clinical_trial_matching.models import Trial


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def trial_from_flat_record(record: dict[str, Any]) -> Trial:
    return Trial(
        nct_id=str(record["nct_id"]),
        title=str(record.get("title", "")),
        status=str(record.get("status", "")),
        conditions=_as_tuple(record.get("conditions")),
        interventions=_as_tuple(record.get("interventions")),
        eligibility_criteria=str(record.get("eligibility_criteria", "")),
        locations=_as_tuple(record.get("locations")),
        source={str(k): str(v) for k, v in dict(record.get("source", {})).items()},
    )


def trial_to_flat_record(trial: Trial) -> dict[str, Any]:
    return {
        "nct_id": trial.nct_id,
        "title": trial.title,
        "status": trial.status,
        "conditions": list(trial.conditions),
        "interventions": list(trial.interventions),
        "eligibility_criteria": trial.eligibility_criteria,
        "locations": list(trial.locations),
        "source": trial.source,
    }
