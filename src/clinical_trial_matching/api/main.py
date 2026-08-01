from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from clinical_trial_matching.ingestion.clinicaltrials import trial_from_flat_record
from clinical_trial_matching.io import read_jsonl
from clinical_trial_matching.retrieval.bm25 import BM25Retriever

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover - import-time developer guidance
    raise RuntimeError("Install API dependencies with `python3 -m pip install -e .`.") from exc


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, object]]


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
    trials = [
        trial_from_flat_record(row)
        for row in read_jsonl(Path("data/fixtures/trials.sample.jsonl"))
    ]
    retriever = BM25Retriever(trials)
    results = [
        {
            "nct_id": result.nct_id,
            "title": result.title,
            "rank": result.rank,
            "score": round(result.score, 4),
        }
        for result in retriever.search(request.query, top_k=request.top_k)
    ]
    return SearchResponse(query=request.query, results=results)


@app.get("/metrics/health")
def metrics_health() -> dict[str, object]:
    return {"status": "ok", "checks": {"api": True}}
