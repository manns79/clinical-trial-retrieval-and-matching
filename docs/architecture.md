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

`POST /search` currently uses the same in-memory BM25 path as the CLI command. It reads a normalized trial JSONL file from `TRIAL_CORPUS_PATH`, defaulting to `data/processed/clinicaltrials/studies.sample.jsonl`. The response includes retriever parameters, corpus size, ranked results, matched query terms, and snippets.

Search responses also include `latency_ms` for corpus loading, retrieval, and total handler time.

`GET /trial/{nct_id}` returns the full normalized trial record from the same configured corpus. It is intended as the detail endpoint that search results can link to before the project moves to database-backed serving.

## Streamlit UI

The Streamlit UI is a thin client over FastAPI. It reads `API_BASE_URL`, defaults to `http://localhost:8000`, calls `/metrics/health`, posts searches to `/search`, and fetches details from `/trial/{nct_id}`. It does not run retrieval locally.

## Docker Compose

Docker Compose runs FastAPI, Streamlit, and Postgres/pgvector. The API service seeds the small synthetic ClinicalTrials.gov fixture into the mounted `data/processed/` directory before starting so a fresh local demo has a searchable corpus. The UI service uses `API_BASE_URL=http://api:8000` inside the Compose network.

CI runs a Docker smoke check that builds the image, starts the API container with fixture data mounted, and verifies `/health`, `/search`, and `/trial/NCT99991001`.

## Observability

FastAPI includes request timing middleware that adds `X-Process-Time-Ms` to every response and logs structured JSON events for each HTTP request. The `/search` handler logs query length, requested `top_k`, corpus size, result count, and latency breakdown. Logs are written to stdout/stderr so Docker, Compose, and future hosted platforms can collect them without a paid observability service.

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
