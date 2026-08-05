from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any

from clinical_trial_matching.io import read_json
from clinical_trial_matching.models import Trial

CTGOV_API_BASE_URL = "https://clinicaltrials.gov/api/v2"
CTGOV_STUDIES_ENDPOINT = "/studies"
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CtgovDownloadResult:
    payload: dict[str, Any]
    request_url: str
    study_count: int
    total_count: int | None
    next_page_token: str


@dataclass(frozen=True)
class CtgovIdCorpusDownloadResult:
    payload: dict[str, Any]
    request_urls: tuple[str, ...]
    requested_nct_ids: tuple[str, ...]
    found_nct_ids: tuple[str, ...]
    missing_nct_ids: tuple[str, ...]


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
        brief_summary=str(record.get("brief_summary", "")),
        status=str(record.get("status", "")),
        conditions=_as_tuple(record.get("conditions")),
        interventions=_as_tuple(record.get("interventions")),
        eligibility_criteria=str(record.get("eligibility_criteria", "")),
        sex=str(record.get("sex", "")),
        minimum_age=str(record.get("minimum_age", "")),
        maximum_age=str(record.get("maximum_age", "")),
        phases=_as_tuple(record.get("phases")),
        study_type=str(record.get("study_type", "")),
        locations=_as_tuple(record.get("locations")),
        source={str(k): str(v) for k, v in dict(record.get("source", {})).items()},
    )


def trial_to_flat_record(trial: Trial) -> dict[str, Any]:
    return {
        "nct_id": trial.nct_id,
        "title": trial.title,
        "brief_summary": trial.brief_summary,
        "status": trial.status,
        "conditions": list(trial.conditions),
        "interventions": list(trial.interventions),
        "eligibility_criteria": trial.eligibility_criteria,
        "sex": trial.sex,
        "minimum_age": trial.minimum_age,
        "maximum_age": trial.maximum_age,
        "phases": list(trial.phases),
        "study_type": trial.study_type,
        "locations": list(trial.locations),
        "source": trial.source,
    }


def parse_studies_json(path: Path) -> list[Trial]:
    payload = read_json(path)
    return trials_from_v2_payload(payload, source_path=path)


def fetch_ctgov_studies(
    *,
    query: str,
    status: str | None = None,
    page_size: int = 25,
    base_url: str = CTGOV_API_BASE_URL,
    timeout_seconds: float = 30.0,
    client: Any = None,
) -> CtgovDownloadResult:
    if not query.strip():
        raise ValueError("ClinicalTrials.gov query cannot be empty")
    if page_size < 1 or page_size > 1000:
        raise ValueError("ClinicalTrials.gov page size must be between 1 and 1000")

    params: dict[str, str | int] = {
        "format": "json",
        "query.term": query,
        "pageSize": page_size,
        "countTotal": "true",
    }
    if status:
        params["filter.overallStatus"] = status

    close_client = client is None
    if client is None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Install project dependencies with `python3 -m pip install -e .` before live downloads."
            ) from exc
        http_client = httpx.Client(timeout=timeout_seconds)
    else:
        http_client = client
    try:
        response = http_client.get(f"{base_url.rstrip('/')}{CTGOV_STUDIES_ENDPOINT}", params=params)
        response.raise_for_status()
        payload = response.json()
    finally:
        if close_client:
            http_client.close()

    if not isinstance(payload, dict):
        raise ValueError("ClinicalTrials.gov response was not a JSON object")
    studies = payload.get("studies", [])
    if not isinstance(studies, list):
        raise ValueError("ClinicalTrials.gov response field 'studies' was not a list")

    total_count = payload.get("totalCount")
    return CtgovDownloadResult(
        payload=payload,
        request_url=str(response.url),
        study_count=len(studies),
        total_count=int(total_count) if isinstance(total_count, int | float | str) else None,
        next_page_token=str(payload.get("nextPageToken", "")),
    )


def fetch_ctgov_studies_by_ids(
    *,
    nct_ids: list[str],
    batch_size: int = 100,
    base_url: str = CTGOV_API_BASE_URL,
    timeout_seconds: float = 30.0,
    delay_seconds: float = 0.0,
    max_retries: int = 5,
    retry_initial_delay_seconds: float = 2.0,
    retry_max_delay_seconds: float = 60.0,
    client: Any = None,
) -> CtgovIdCorpusDownloadResult:
    requested_ids = normalize_nct_ids(nct_ids)
    if not requested_ids:
        raise ValueError("At least one NCT ID is required")
    if batch_size < 1 or batch_size > 1000:
        raise ValueError("ClinicalTrials.gov batch size must be between 1 and 1000")
    if delay_seconds < 0:
        raise ValueError("Delay seconds cannot be negative")
    if max_retries < 0:
        raise ValueError("Max retries cannot be negative")
    if retry_initial_delay_seconds < 0:
        raise ValueError("Retry initial delay seconds cannot be negative")
    if retry_max_delay_seconds < retry_initial_delay_seconds:
        raise ValueError("Retry max delay seconds must be at least retry initial delay seconds")

    close_client = client is None
    if client is None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Install project dependencies with `python3 -m pip install -e .` before live downloads."
            ) from exc
        http_client = httpx.Client(timeout=timeout_seconds)
    else:
        http_client = client

    studies_by_id: dict[str, dict[str, Any]] = {}
    request_urls: list[str] = []
    try:
        batches = list(_chunks(requested_ids, batch_size))
        for batch_index, batch in enumerate(batches):
            page_token = ""
            while True:
                params: dict[str, str | int] = {
                    "format": "json",
                    "query.id": " OR ".join(batch),
                    "pageSize": min(max(len(batch), 1), 1000),
                    "countTotal": "true",
                }
                if page_token:
                    params["pageToken"] = page_token
                response = _get_with_retries(
                    http_client,
                    f"{base_url.rstrip('/')}{CTGOV_STUDIES_ENDPOINT}",
                    params=params,
                    max_retries=max_retries,
                    retry_initial_delay_seconds=retry_initial_delay_seconds,
                    retry_max_delay_seconds=retry_max_delay_seconds,
                )
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("ClinicalTrials.gov response was not a JSON object")
                studies = payload.get("studies", [])
                if not isinstance(studies, list):
                    raise ValueError("ClinicalTrials.gov response field 'studies' was not a list")
                request_urls.append(str(response.url))
                for study in studies:
                    if not isinstance(study, dict):
                        continue
                    nct_id = _nct_id_from_v2_record(study)
                    if nct_id:
                        studies_by_id[nct_id] = study
                page_token = str(payload.get("nextPageToken", ""))
                if not page_token:
                    break
                if delay_seconds:
                    sleep(delay_seconds)
            if delay_seconds and batch_index < len(batches) - 1:
                sleep(delay_seconds)
    finally:
        if close_client:
            http_client.close()

    found_ids = tuple(nct_id for nct_id in requested_ids if nct_id in studies_by_id)
    missing_ids = tuple(nct_id for nct_id in requested_ids if nct_id not in studies_by_id)
    return CtgovIdCorpusDownloadResult(
        payload={
            "source": "clinicaltrials_gov_v2",
            "requested_nct_ids": list(requested_ids),
            "found_nct_ids": list(found_ids),
            "missing_nct_ids": list(missing_ids),
            "request_urls": request_urls,
            "studies": [studies_by_id[nct_id] for nct_id in found_ids],
        },
        request_urls=tuple(request_urls),
        requested_nct_ids=tuple(requested_ids),
        found_nct_ids=found_ids,
        missing_nct_ids=missing_ids,
    )


def _get_with_retries(
    client: Any,
    url: str,
    *,
    params: dict[str, str | int],
    max_retries: int,
    retry_initial_delay_seconds: float,
    retry_max_delay_seconds: float,
) -> Any:
    attempts = max_retries + 1
    for attempt in range(attempts):
        response = client.get(url, params=params)
        status_code = int(getattr(response, "status_code", 200))
        if status_code not in RETRYABLE_HTTP_STATUS_CODES:
            response.raise_for_status()
            return response
        if attempt == max_retries:
            response.raise_for_status()
            return response
        sleep(_retry_delay_seconds(response, attempt, retry_initial_delay_seconds, retry_max_delay_seconds))
    raise RuntimeError("ClinicalTrials.gov request retry loop exited unexpectedly")


def _retry_delay_seconds(
    response: Any,
    attempt: int,
    retry_initial_delay_seconds: float,
    retry_max_delay_seconds: float,
) -> float:
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), retry_max_delay_seconds)
        except ValueError:
            pass
    return min(retry_initial_delay_seconds * (2**attempt), retry_max_delay_seconds)


def normalize_nct_ids(nct_ids: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for nct_id in nct_ids:
        value = str(nct_id).strip().upper()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)


def trials_from_v2_payload(payload: Any, source_path: Path | None = None) -> list[Trial]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("studies"), list):
        records = payload["studies"]
    elif isinstance(payload, dict) and "protocolSection" in payload:
        records = [payload]
    else:
        raise ValueError("Expected a ClinicalTrials.gov v2 study, list of studies, or response with studies")

    return [
        trial_from_ctgov_v2_record(record, source_path=source_path)
        for record in records
        if isinstance(record, dict)
    ]


def trial_from_ctgov_v2_record(record: dict[str, Any], source_path: Path | None = None) -> Trial:
    protocol = _dict(record.get("protocolSection"))
    identification = _dict(protocol.get("identificationModule"))
    status_module = _dict(protocol.get("statusModule"))
    description = _dict(protocol.get("descriptionModule"))
    conditions_module = _dict(protocol.get("conditionsModule"))
    design = _dict(protocol.get("designModule"))
    interventions_module = _dict(protocol.get("armsInterventionsModule"))
    eligibility = _dict(protocol.get("eligibilityModule"))
    contacts_locations = _dict(protocol.get("contactsLocationsModule"))

    nct_id = str(identification.get("nctId", "")).strip()
    if not nct_id:
        raise ValueError("ClinicalTrials.gov study record is missing protocolSection.identificationModule.nctId")

    return Trial(
        nct_id=nct_id,
        title=_first_text(identification.get("briefTitle"), identification.get("officialTitle")),
        brief_summary=str(description.get("briefSummary", "")),
        status=str(status_module.get("overallStatus", "")),
        conditions=_as_tuple(conditions_module.get("conditions")),
        interventions=_intervention_names(interventions_module.get("interventions")),
        eligibility_criteria=str(eligibility.get("eligibilityCriteria", "")),
        sex=str(eligibility.get("sex", "")),
        minimum_age=str(eligibility.get("minimumAge", "")),
        maximum_age=str(eligibility.get("maximumAge", "")),
        phases=_as_tuple(design.get("phases")),
        study_type=str(design.get("studyType", "")),
        locations=_location_labels(contacts_locations.get("locations")),
        source={
            "kind": "clinicaltrials_gov_v2",
            "nct_id": nct_id,
            "path": str(source_path) if source_path else "",
        },
    )


def _nct_id_from_v2_record(record: dict[str, Any]) -> str:
    protocol = _dict(record.get("protocolSection"))
    identification = _dict(protocol.get("identificationModule"))
    return str(identification.get("nctId", "")).strip().upper()


def _chunks(values: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _intervention_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for intervention in value:
        if not isinstance(intervention, dict):
            continue
        name = intervention.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


def _location_labels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    labels: list[str] = []
    for location in value:
        if not isinstance(location, dict):
            continue
        parts = [
            location.get("facility"),
            location.get("city"),
            location.get("state"),
            location.get("country"),
        ]
        label = ", ".join(str(part) for part in parts if part)
        if label:
            labels.append(label)
    return tuple(labels)
