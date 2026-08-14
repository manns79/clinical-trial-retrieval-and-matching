# Architecture

## Modes

The project is intentionally split into two operating modes:

1. Full reproducible mode: local Docker Compose, frozen TREC/ClinicalTrials.gov snapshots, full evaluation, and larger indexes.
2. Public demo mode: a compact subset or prebuilt index, deployed on a free tier with transparent limitations.

This keeps the portfolio artifact deployable without forcing paid database or model hosting.

## Components

- Ingestion: downloads or reads source records, records provenance, and normalizes trial fields.
- Storage: Postgres for structured fields, full-text search, and eventually pgvector embeddings.
- Retrieval: BM25 baseline, metadata filters, dense retrieval, hybrid fusion, and reranking.
- Evaluation: TREC-style qrels, reproducible run files, and metrics such as Recall@100, nDCG, MRR, and Precision@k.
- API: FastAPI service for search, trial lookup, health, and later explanation endpoints.
- UI: Streamlit client over the FastAPI search and trial-detail endpoints.
- Observability: structured logs, latency timing, health checks, and retrieval regression tests.

## Initial API Surface

- `GET /health`
- `POST /search`
- `GET /trial/{nct_id}`
- `GET /metrics/health` for lightweight operational checks

`POST /search` supports plain BM25, the frozen field-aware BM25 profile, the selected local dense
bi-encoder, and equal-weight reciprocal-rank fusion. It reads normalized trial JSONL from
`TRIAL_CORPUS_PATH`. Dense and hybrid modes require a compatible ignored NumPy index configured
through `DENSE_INDEX_PATH`; the model and index are validated and loaded once in the FastAPI
lifespan before requests are accepted. The service never builds corpus embeddings on a request.

Search responses include stage latency for lexical retrieval, query embedding/dense scoring,
fusion, and total handler time. Hybrid records retain each component rank alongside the fused
score. `/metrics/health` advertises only retrievers supported by the currently mounted artifacts.

`GET /trial/{nct_id}` returns the full normalized trial record from the same configured corpus. It is intended as the detail endpoint that search results can link to before the project moves to database-backed serving.

## Streamlit UI

The Streamlit UI is a thin client over FastAPI. It reads `API_BASE_URL`, defaults to
`http://localhost:8000`, discovers available retrievers through `/metrics/health`, posts searches
to `/search`, displays stage latency, and fetches details from `/trial/{nct_id}`. It does not run
models or retrieval locally.

## Docker Compose

Docker Compose runs FastAPI, Streamlit, and Postgres/pgvector. The API image includes the optional
dense dependencies, while the default Compose startup still seeds the small synthetic corpus and
offers lexical search without downloading a model. Mounting compatible corpus/BM25/dense index
paths enables dense and hybrid modes; Hugging Face model files are cached under ignored
`data/models/`. The UI uses `API_BASE_URL=http://api:8000` inside the Compose network.

CI runs a Docker smoke check that builds the image, starts the API container with fixture data mounted, and verifies `/health`, `/search`, and `/trial/NCT99991001`.

## Observability

FastAPI includes request timing middleware that adds `X-Process-Time-Ms` to every response and
logs structured JSON events for each HTTP request. The `/search` handler logs query length,
requested `top_k`, retriever, corpus size, result count, artifact paths, and stage latency. Logs
are written to stdout/stderr so Docker, Compose, and future hosted platforms can collect them
without a paid observability service.

## Retrieval Regression

CI runs a tiny deterministic BM25 regression check over synthetic trial/topic/qrels fixtures. It records the full run and metrics, then fails if configured thresholds such as Recall@100, MRR, or nDCG@10 drop. This is intentionally small and fast; larger TREC evaluations remain a separate benchmark workflow.

## TREC-Style Evaluation

The BM25 benchmark command reads normalized trial JSONL, normalized TREC topic JSONL, and qrels, then writes both a six-column TREC run file and a metrics JSON report. Generated run files and reports belong under ignored `outputs/` paths.

The dense benchmark path uses a local sentence-transformer bi-encoder, batches normalized trial
text into L2-normalized embeddings, and persists embeddings plus NCT IDs and compatibility
metadata in an ignored NumPy `.npz` index. Query embeddings use the same encoder and cosine
similarity is computed as a matrix product. Model files stay in the local model cache and are not
committed.

## Benchmark Corpus Build

The TREC corpus builder extracts unique NCT IDs from qrels, fetches matching ClinicalTrials.gov v2 records with `query.id`, writes the raw response envelope under ignored `data/raw/`, normalizes records to trial JSONL under ignored `data/processed/`, and records manifest/report metadata. This keeps downloaded benchmark assets local while making the build process reproducible.

## Guardrails

- Always distinguish benchmark snapshots from live ClinicalTrials.gov data.
- Treat explanations as grounded summaries, not eligibility determinations.
- Require citations to trial fields for generated explanations.
- Keep LLM evaluation separate from retrieval evaluation.
