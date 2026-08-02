from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from clinical_trial_matching.ingestion.clinicaltrials import trial_from_flat_record
from clinical_trial_matching.io import read_jsonl
from clinical_trial_matching.models import Trial
from clinical_trial_matching.retrieval.bm25 import search_trials

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover - import-time developer guidance
    raise RuntimeError("Install API dependencies with `python3 -m pip install -e .`.") from exc

DEFAULT_TRIAL_CORPUS_PATH = "data/processed/clinicaltrials/studies.sample.jsonl"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    snippet_chars: int = Field(default=240, ge=1, le=1000)


class SearchResponse(BaseModel):
    query: str
    retriever: str
    parameters: dict[str, Any]
    corpus: dict[str, int]
    results: list[dict[str, Any]]


app = FastAPI(
    title="Clinical Trial Retrieval and Matching",
    version="0.1.0",
    description="Research demo API for retrieving potentially relevant clinical trials.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    trials = load_trial_corpus()
    payload = search_trials(
        trials,
        query=request.query,
        top_k=request.top_k,
        snippet_chars=request.snippet_chars,
    )
    return SearchResponse(**payload)


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
