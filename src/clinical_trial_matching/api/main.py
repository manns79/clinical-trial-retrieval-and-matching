from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from clinical_trial_matching.ingestion.clinicaltrials import (
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.io import read_jsonl
from clinical_trial_matching.models import Trial
from clinical_trial_matching.observability import configure_logging, elapsed_ms, log_event, now_ms
from clinical_trial_matching.retrieval.bm25 import search_trials

try:
    from fastapi import FastAPI, HTTPException, Request, Response
except ImportError as exc:  # pragma: no cover - import-time developer guidance
    raise RuntimeError("Install API dependencies with `python3 -m pip install -e .`.") from exc

DEFAULT_TRIAL_CORPUS_PATH = "data/processed/clinicaltrials/studies.sample.jsonl"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGGER = logging.getLogger("clinical_trial_matching.api")
configure_logging(LOG_LEVEL)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    snippet_chars: int = Field(default=240, ge=1, le=1000)


class SearchResponse(BaseModel):
    query: str
    retriever: str
    parameters: dict[str, Any]
    corpus: dict[str, int]
    latency_ms: dict[str, float]
    results: list[dict[str, Any]]


class TrialResponse(BaseModel):
    nct_id: str
    title: str
    status: str
    conditions: list[str]
    interventions: list[str]
    eligibility_criteria: str
    sex: str
    minimum_age: str
    maximum_age: str
    phases: list[str]
    study_type: str
    locations: list[str]
    source: dict[str, str]


app = FastAPI(
    title="Clinical Trial Retrieval and Matching",
    version="0.1.0",
    description="Research demo API for retrieving potentially relevant clinical trials.",
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next: Any) -> Response:
    start_ms = now_ms()
    response = await call_next(request)
    duration_ms = elapsed_ms(start_ms)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    log_event(
        LOGGER,
        "http_request",
        fields={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    total_start_ms = now_ms()
    load_start_ms = now_ms()
    trials = load_trial_corpus()
    corpus_load_ms = elapsed_ms(load_start_ms)
    retrieval_start_ms = now_ms()
    payload = search_trials(
        trials,
        query=request.query,
        top_k=request.top_k,
        snippet_chars=request.snippet_chars,
    )
    retrieval_ms = elapsed_ms(retrieval_start_ms)
    payload["latency_ms"] = {
        "corpus_load": corpus_load_ms,
        "retrieval": retrieval_ms,
        "total": elapsed_ms(total_start_ms),
    }
    log_event(
        LOGGER,
        "search",
        fields={
            "query_length": len(request.query),
            "top_k": request.top_k,
            "corpus_trials": payload["corpus"]["trials"],
            "result_count": len(payload["results"]),
            "latency_ms": payload["latency_ms"],
        },
    )
    return SearchResponse(**payload)


@app.get("/trial/{nct_id}", response_model=TrialResponse)
def get_trial(nct_id: str) -> TrialResponse:
    trial = get_trial_by_nct_id(nct_id)
    if trial is None:
        raise HTTPException(status_code=404, detail=f"Trial not found: {nct_id}")
    return TrialResponse(**trial_to_flat_record(trial))


@app.get("/metrics/health")
def metrics_health() -> dict[str, object]:
    corpus_path = get_trial_corpus_path()
    return {
        "status": "ok",
        "checks": {
            "api": True,
            "trial_corpus_exists": corpus_path.exists(),
        },
        "trial_corpus_path": str(corpus_path),
    }


def get_trial_corpus_path() -> Path:
    return Path(os.getenv("TRIAL_CORPUS_PATH", DEFAULT_TRIAL_CORPUS_PATH))


@lru_cache(maxsize=1)
def load_trial_corpus() -> tuple[Trial, ...]:
    corpus_path = get_trial_corpus_path()
    if not corpus_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trial corpus not found at {corpus_path}. "
                "Set TRIAL_CORPUS_PATH or run `make ingest-ctgov-sample`."
            ),
        )
    return tuple(trial_from_flat_record(row) for row in read_jsonl(corpus_path))


def get_trial_by_nct_id(nct_id: str) -> Trial | None:
    normalized_nct_id = nct_id.strip().upper()
    for trial in load_trial_corpus():
        if trial.nct_id.upper() == normalized_nct_id:
            return trial
    return None
