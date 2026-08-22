from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from clinical_trial_matching.ingestion.clinicaltrials import (
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.io import read_jsonl
from clinical_trial_matching.models import Trial
from clinical_trial_matching.observability import configure_logging, elapsed_ms, log_event, now_ms
from clinical_trial_matching.retrieval.bm25 import (
    QUERY_STOPWORDS,
    format_search_results,
    load_or_build_bm25_retriever,
    normalized_field_weights,
)
from clinical_trial_matching.retrieval.dense import (
    DENSE_SCORE_TIE_DECIMALS,
    DenseRetriever,
    construct_text_encoder,
    load_dense_index_for_corpus,
    load_encoder_framework,
    warm_up_text_encoder,
)
from clinical_trial_matching.retrieval.hybrid import (
    RankedResults,
    reciprocal_rank_fuse_results,
)
from clinical_trial_matching.retrieval.sqlite_fts import (
    DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
    load_sqlite_fts_retriever_for_corpus,
)
from clinical_trial_matching.trial_store import SQLiteTrialStore, load_trial_store

try:
    from fastapi import FastAPI, HTTPException, Request, Response
except ImportError as exc:  # pragma: no cover - import-time developer guidance
    raise RuntimeError("Install API dependencies with `python3 -m pip install -e .`.") from exc

DEFAULT_TRIAL_CORPUS_PATH = "data/processed/clinicaltrials/studies.sample.jsonl"
DEFAULT_SQLITE_FTS_INDEX_PATH = "data/indexes/studies_sample_sqlite_fts5.sqlite"
DEFAULT_TRIAL_STORE_PATH = "data/indexes/studies_sample_trial_store.sqlite"
DEFAULT_DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DENSE_TEXT_REPRESENTATION = "title_summary_conditions"
DEFAULT_DENSE_BATCH_SIZE = 64
DEFAULT_DENSE_DEVICE = "cpu"
DEFAULT_DENSE_MAX_SEQ_LENGTH = 256
DEFAULT_DENSE_ENCODER_BACKEND = "sentence-transformers"
DEFAULT_RRF_K = 60
DEFAULT_RRF_CANDIDATE_DEPTH = 100
SERVING_FIELD_WEIGHTS = {
    "all_text": 1.0,
    "title": 1.25,
    "brief_summary": 0.75,
    "conditions": 1.5,
    "interventions": 0.25,
    "eligibility_criteria": 0.5,
    "demographics": 0.1,
    "status": 0.05,
    "locations": 0.05,
}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGGER = logging.getLogger("clinical_trial_matching.api")
configure_logging(LOG_LEVEL)


@dataclass(frozen=True)
class DenseServingConfig:
    index_path: Path
    model_name: str
    text_representation: str
    batch_size: int
    device: str
    max_seq_length: int | None
    dynamic_quantization: bool
    encoder_backend: str
    onnx_model_path: Path | None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    snippet_chars: int = Field(default=240, ge=1, le=1000)
    retriever: Literal["bm25", "fielded-bm25", "sqlite-fts5", "dense", "hybrid"] = (
        "sqlite-fts5"
    )


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
    brief_summary: str
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


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    preload_search_resources()
    yield


app = FastAPI(
    title="Clinical Trial Retrieval and Matching",
    version="0.1.0",
    description="Research demo API for retrieving potentially relevant clinical trials.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next: Any) -> Response:
    start_ms = now_ms()
    response = cast(Response, await call_next(request))
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
    trial_store = load_trial_metadata_store()
    corpus_load_ms = elapsed_ms(load_start_ms)
    index_load_start_ms = now_ms()
    lexical_retriever = None
    dense_retriever = None
    if request.retriever in {"bm25", "fielded-bm25"}:
        lexical_retriever = load_search_retriever(request.retriever)
    elif request.retriever in {"sqlite-fts5", "hybrid"}:
        lexical_retriever = load_sqlite_search_retriever()
    if request.retriever in {"dense", "hybrid"}:
        dense_retriever = load_dense_search_retriever()
    index_load_ms = elapsed_ms(index_load_start_ms)

    lexical_ms = 0.0
    embedding_ms = 0.0
    fusion_ms = 0.0
    result_metadata: dict[str, dict[str, Any]] = {}
    if request.retriever in {"bm25", "fielded-bm25", "sqlite-fts5"}:
        assert lexical_retriever is not None
        lexical_start_ms = now_ms()
        results = lexical_retriever.search(request.query, top_k=request.top_k)
        lexical_ms = elapsed_ms(lexical_start_ms)
        parameters = (
            sqlite_fts_parameters(lexical_retriever, request.top_k)
            if request.retriever == "sqlite-fts5"
            else lexical_parameters(request.retriever, request.top_k)
        )
    elif request.retriever == "dense":
        assert dense_retriever is not None
        embedding_start_ms = now_ms()
        results = dense_retriever.search(request.query, top_k=request.top_k)
        embedding_ms = elapsed_ms(embedding_start_ms)
        parameters = dense_parameters(dense_retriever, request.top_k)
    else:
        assert lexical_retriever is not None
        assert dense_retriever is not None
        candidate_depth = min(
            max(request.top_k, get_rrf_candidate_depth()),
            trial_store.count,
        )
        lexical_start_ms = now_ms()
        lexical_results = lexical_retriever.search(request.query, top_k=candidate_depth)
        lexical_ms = elapsed_ms(lexical_start_ms)
        embedding_start_ms = now_ms()
        dense_results = dense_retriever.search(request.query, top_k=candidate_depth)
        embedding_ms = elapsed_ms(embedding_start_ms)
        fusion_start_ms = now_ms()
        fused = reciprocal_rank_fuse_results(
            [
                RankedResults("sqlite-fts5", 1.0, tuple(lexical_results)),
                RankedResults("dense", 1.0, tuple(dense_results)),
            ],
            rrf_k=get_rrf_k(),
            top_k=request.top_k,
            candidate_depth=candidate_depth,
        )
        fusion_ms = elapsed_ms(fusion_start_ms)
        results = list(fused.results)
        result_metadata = {
            nct_id: {"component_ranks": ranks}
            for nct_id, ranks in fused.component_ranks.items()
        }
        parameters = hybrid_parameters(dense_retriever, request.top_k, candidate_depth)

    metadata_start_ms = now_ms()
    result_trials = trial_store.get_many([result.nct_id for result in results])
    formatted_results = format_search_results(
        result_trials,
        results,
        query=request.query,
        snippet_chars=request.snippet_chars,
        result_metadata=result_metadata,
    )
    metadata_ms = elapsed_ms(metadata_start_ms)
    payload: dict[str, Any] = {
        "query": request.query,
        "retriever": request.retriever,
        "parameters": parameters,
        "corpus": {
            "trials": trial_store.count,
            "unique_nct_ids": trial_store.unique_nct_ids,
        },
        "results": formatted_results,
    }
    retrieval_ms = lexical_ms + embedding_ms + fusion_ms
    payload["latency_ms"] = {
        "corpus_load": corpus_load_ms,
        "index_load": index_load_ms,
        "lexical": lexical_ms,
        "embedding": embedding_ms,
        "fusion": fusion_ms,
        "metadata": metadata_ms,
        "retrieval": retrieval_ms,
        "total": elapsed_ms(total_start_ms),
    }
    log_event(
        LOGGER,
        "search",
        fields={
            "query_length": len(request.query),
            "top_k": request.top_k,
            "retriever": request.retriever,
            "corpus_trials": payload["corpus"]["trials"],
            "result_count": len(payload["results"]),
            "latency_ms": payload["latency_ms"],
            "bm25_index_path": get_bm25_index_path(
                "bm25" if request.retriever == "bm25" else "fielded-bm25"
            ),
            "sqlite_fts_index_path": str(get_sqlite_fts_index_path()),
            "trial_store_path": str(get_trial_store_path()),
            "dense_index_path": str(get_dense_index_path() or ""),
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
    trial_store_path = get_trial_store_path()
    index_path = get_bm25_index_path()
    dense_index_path = get_dense_index_path()
    dense_encoder_backend = os.getenv(
        "DENSE_ENCODER_BACKEND",
        DEFAULT_DENSE_ENCODER_BACKEND,
    )
    dense_onnx_model_path_value = os.getenv("DENSE_ONNX_MODEL_PATH", "").strip()
    dense_encoder_artifact_exists = (
        dense_encoder_backend != "onnxruntime"
        or bool(dense_onnx_model_path_value)
        and Path(dense_onnx_model_path_value).exists()
    )
    sqlite_fts_index_path = get_sqlite_fts_index_path()
    corpus_exists = corpus_path.exists()
    trial_store_exists = trial_store_path.exists()
    sqlite_fts_index_exists = sqlite_fts_index_path.exists()
    dense_index_exists = dense_index_path is not None and dense_index_path.exists()
    available_retrievers = ["sqlite-fts5", "fielded-bm25", "bm25"]
    if (
        corpus_exists
        and trial_store_exists
        and dense_index_exists
        and dense_encoder_artifact_exists
    ):
        available_retrievers.extend(["dense", "hybrid"])
    return {
        "status": "ok",
        "checks": {
            "api": True,
            "trial_corpus_exists": corpus_exists,
            "trial_store_exists": trial_store_exists,
            "trial_store_loaded": load_trial_metadata_store.cache_info().currsize > 0,
            "bm25_index_exists": Path(index_path).exists() if index_path else False,
            "sqlite_fts_index_exists": sqlite_fts_index_exists,
            "sqlite_fts_retriever_loaded": load_sqlite_search_retriever.cache_info().currsize > 0,
            "dense_index_configured": dense_index_path is not None,
            "dense_index_exists": dense_index_exists,
            "dense_index_loaded": load_dense_search_index.cache_info().currsize > 0,
            "dense_encoder_artifact_exists": dense_encoder_artifact_exists,
            "dense_encoder_framework_loaded": (
                load_dense_encoder_framework.cache_info().currsize > 0
            ),
            "dense_encoder_loaded": load_dense_search_encoder.cache_info().currsize > 0,
            "dense_encoder_warmed": warm_dense_search_encoder.cache_info().currsize > 0,
            "dense_retriever_loaded": load_dense_search_retriever.cache_info().currsize > 0,
        },
        "available_retrievers": available_retrievers,
        "trial_corpus_path": str(corpus_path),
        "trial_store_path": str(trial_store_path),
        "bm25_index_path": index_path,
        "sqlite_fts_index_path": str(sqlite_fts_index_path),
        "dense_index_path": str(dense_index_path or ""),
        "dense_encoder_backend": dense_encoder_backend,
        "dense_onnx_model_path": dense_onnx_model_path_value,
    }


def get_trial_corpus_path() -> Path:
    return Path(os.getenv("TRIAL_CORPUS_PATH", DEFAULT_TRIAL_CORPUS_PATH))


def get_bm25_index_path(retriever_name: str = "fielded-bm25") -> str:
    if retriever_name == "bm25":
        return os.getenv("PLAIN_BM25_INDEX_PATH", "")
    return os.getenv("BM25_INDEX_PATH", "")


def get_sqlite_fts_index_path() -> Path:
    return Path(os.getenv("SQLITE_FTS_INDEX_PATH", DEFAULT_SQLITE_FTS_INDEX_PATH))


def get_trial_store_path() -> Path:
    return Path(os.getenv("TRIAL_STORE_PATH", DEFAULT_TRIAL_STORE_PATH))


def get_dense_index_path() -> Path | None:
    value = os.getenv("DENSE_INDEX_PATH", "").strip()
    return Path(value) if value else None


def get_dense_serving_config() -> DenseServingConfig:
    index_path = get_dense_index_path()
    if index_path is None:
        raise HTTPException(
            status_code=503,
            detail="Dense retrieval is not configured. Set DENSE_INDEX_PATH.",
        )
    return DenseServingConfig(
        index_path=index_path,
        model_name=os.getenv("DENSE_MODEL_NAME", DEFAULT_DENSE_MODEL_NAME),
        text_representation=os.getenv(
            "DENSE_TEXT_REPRESENTATION",
            DEFAULT_DENSE_TEXT_REPRESENTATION,
        ),
        batch_size=_positive_env_int("DENSE_BATCH_SIZE", DEFAULT_DENSE_BATCH_SIZE),
        device=os.getenv("DENSE_DEVICE", DEFAULT_DENSE_DEVICE),
        max_seq_length=_optional_positive_env_int(
            "DENSE_MAX_SEQ_LENGTH",
            DEFAULT_DENSE_MAX_SEQ_LENGTH,
        ),
        dynamic_quantization=_boolean_env("DENSE_DYNAMIC_QUANTIZATION", False),
        encoder_backend=os.getenv(
            "DENSE_ENCODER_BACKEND",
            DEFAULT_DENSE_ENCODER_BACKEND,
        ),
        onnx_model_path=(
            Path(value)
            if (value := os.getenv("DENSE_ONNX_MODEL_PATH", "").strip())
            else None
        ),
    )


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


@lru_cache(maxsize=1)
def load_trial_metadata_store() -> SQLiteTrialStore:
    corpus_path = get_trial_corpus_path()
    if not corpus_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Trial corpus not found at {corpus_path}. Set TRIAL_CORPUS_PATH.",
        )
    try:
        return load_trial_store(
            get_trial_store_path(),
            corpus_path=corpus_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trial metadata store could not be loaded: {exc}. "
                "Run `ctmatch build-trial-store` for the configured corpus."
            ),
        ) from exc


@lru_cache(maxsize=4)
def load_search_retriever(retriever_name: str) -> Any:
    if retriever_name not in {"bm25", "fielded-bm25"}:
        raise ValueError(f"Unsupported lexical retriever: {retriever_name}")
    corpus_path = get_trial_corpus_path()
    index_path_value = get_bm25_index_path(retriever_name)
    return load_or_build_bm25_retriever(
        trials=load_trial_corpus(),
        retriever_name=retriever_name,
        field_weights=SERVING_FIELD_WEIGHTS if retriever_name == "fielded-bm25" else None,
        corpus_path=corpus_path,
        index_path=Path(index_path_value) if index_path_value else None,
    )


@lru_cache(maxsize=1)
def load_sqlite_search_retriever() -> Any:
    try:
        return load_sqlite_fts_retriever_for_corpus(
            get_sqlite_fts_index_path(),
            corpus=load_trial_metadata_store().corpus,
            field_weights=DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"SQLite FTS5 retriever could not be loaded: {exc}",
        ) from exc


@lru_cache(maxsize=1)
def load_dense_search_index() -> Any:
    config = get_dense_serving_config()
    if not config.index_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Dense index not found at {config.index_path}.",
        )
    try:
        trial_store = load_trial_metadata_store()
        index = load_dense_index_for_corpus(
            config.index_path,
            trials_count=trial_store.count,
            corpus_fingerprint_value=str(trial_store.corpus["fingerprint"]),
            model_name=config.model_name,
            text_representation=config.text_representation,
            max_seq_length=config.max_seq_length,
        )
        trial_store.validate_nct_id_order(index.nct_ids)
        return index
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dense index could not be loaded: {exc}",
        ) from exc


@lru_cache(maxsize=1)
def load_dense_encoder_framework() -> Any:
    config = get_dense_serving_config()
    try:
        return load_encoder_framework(config.encoder_backend)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dense encoder framework could not be loaded: {exc}",
        ) from exc


@lru_cache(maxsize=1)
def load_dense_search_encoder() -> Any:
    config = get_dense_serving_config()
    try:
        encoder = construct_text_encoder(
            backend=config.encoder_backend,
            framework=load_dense_encoder_framework(),
            model_name=config.model_name,
            device=config.device,
            max_seq_length=config.max_seq_length,
            onnx_model_path=config.onnx_model_path,
        )
        if config.dynamic_quantization:
            quantize = getattr(encoder, "quantize_dynamic_int8", None)
            if quantize is None:
                raise ValueError(
                    "DENSE_DYNAMIC_QUANTIZATION is only supported by sentence-transformers"
                )
            quantize()
        return encoder
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dense encoder could not be loaded: {exc}",
        ) from exc


@lru_cache(maxsize=1)
def warm_dense_search_encoder() -> bool:
    config = get_dense_serving_config()
    try:
        warm_up_text_encoder(
            load_dense_search_encoder(),
            batch_size=config.batch_size,
        )
        return True
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dense encoder first inference failed: {exc}",
        ) from exc


@lru_cache(maxsize=1)
def load_dense_search_retriever() -> Any:
    config = get_dense_serving_config()
    return DenseRetriever(
        None,
        index=load_dense_search_index(),
        encoder=load_dense_search_encoder(),
        batch_size=config.batch_size,
    )


def preload_search_resources(
    on_phase_complete: Callable[[str], None] | None = None,
) -> None:
    start_ms = now_ms()
    trial_store = load_trial_metadata_store()
    if on_phase_complete is not None:
        on_phase_complete("trial_metadata_store")
    load_sqlite_search_retriever()
    if on_phase_complete is not None:
        on_phase_complete("sqlite_fts5")
    dense_configured = get_dense_index_path() is not None
    if dense_configured:
        load_dense_search_index()
        if on_phase_complete is not None:
            on_phase_complete("dense_embedding_index")
        load_dense_encoder_framework()
        if on_phase_complete is not None:
            on_phase_complete("dense_encoder_framework")
        load_dense_search_encoder()
        if on_phase_complete is not None:
            on_phase_complete("dense_encoder_model")
        warm_dense_search_encoder()
        if on_phase_complete is not None:
            on_phase_complete("dense_encoder_first_inference_thread_pool")
        load_dense_search_retriever()
        if on_phase_complete is not None:
            on_phase_complete("dense_retriever_assembly")
    log_event(
        LOGGER,
        "search_resources_loaded",
        fields={
            "trials": trial_store.count,
            "dense_configured": dense_configured,
            "duration_ms": elapsed_ms(start_ms),
        },
    )


def lexical_parameters(retriever_name: str, top_k: int) -> dict[str, Any]:
    return {
        "top_k": top_k,
        "k1": 1.5,
        "b": 0.75,
        "field_weights": (
            normalized_field_weights(SERVING_FIELD_WEIGHTS)
            if retriever_name == "fielded-bm25"
            else {}
        ),
        "query_stopwords": sorted(QUERY_STOPWORDS),
    }


def sqlite_fts_parameters(retriever: Any, top_k: int) -> dict[str, Any]:
    return {
        "top_k": top_k,
        "field_weights": retriever.field_weights,
        "tokenizer": retriever.metadata["tokenizer"],
        "query_operator": retriever.metadata["query_operator"],
        "query_stopwords": sorted(QUERY_STOPWORDS),
    }


def dense_parameters(retriever: Any, top_k: int) -> dict[str, Any]:
    metadata = retriever.index.metadata
    return {
        "top_k": top_k,
        "model_name": metadata["model_name"],
        "text_representation": metadata["text_representation"],
        "max_seq_length": metadata["max_seq_length"],
        "embedding_dimension": metadata["embedding_dimension"],
        "normalize_embeddings": metadata["normalize_embeddings"],
        "index_storage_format": metadata.get("storage_format", "compressed_npz"),
        "query_encoder_quantization": getattr(
            getattr(retriever, "encoder", None),
            "quantization",
            "unknown",
        ),
        "query_encoder_backend": getattr(
            getattr(retriever, "encoder", None),
            "backend",
            "unknown",
        ),
        "score_tie_decimals": DENSE_SCORE_TIE_DECIMALS,
    }


def hybrid_parameters(
    dense_retriever: Any,
    top_k: int,
    candidate_depth: int,
) -> dict[str, Any]:
    return {
        "top_k": top_k,
        "rrf_k": get_rrf_k(),
        "candidate_depth": candidate_depth,
        "components": {
            "sqlite-fts5": {
                "weight": 1.0,
                "field_weights": dict(DEFAULT_SQLITE_FTS_FIELD_WEIGHTS),
            },
            "dense": {"weight": 1.0, **dense_parameters(dense_retriever, candidate_depth)},
        },
    }


def get_rrf_k() -> int:
    return _positive_env_int("RRF_K", DEFAULT_RRF_K)


def get_rrf_candidate_depth() -> int:
    return _positive_env_int("RRF_CANDIDATE_DEPTH", DEFAULT_RRF_CANDIDATE_DEPTH)


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _optional_positive_env_int(name: str, default: int | None) -> int | None:
    raw_value = os.getenv(name, "" if default is None else str(default)).strip()
    if not raw_value:
        return None
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _boolean_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def get_trial_by_nct_id(nct_id: str) -> Trial | None:
    return load_trial_metadata_store().get(nct_id)
