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
- UI: Streamlit or React frontend once the API contracts stabilize.
- Observability: structured logs, latency timing, health checks, and retrieval regression tests.

## Initial API Surface

- `GET /health`
- `POST /search`
- `GET /trial/{nct_id}` eventually backed by the database
- `GET /metrics/health` for lightweight operational checks

`POST /search` currently uses the same in-memory BM25 path as the CLI command. It reads a normalized trial JSONL file from `TRIAL_CORPUS_PATH`, defaulting to `data/processed/clinicaltrials/studies.sample.jsonl`. The response includes retriever parameters, corpus size, ranked results, matched query terms, and snippets.

## Guardrails

- Always distinguish benchmark snapshots from live ClinicalTrials.gov data.
- Treat explanations as grounded summaries, not eligibility determinations.
- Require citations to trial fields for generated explanations.
- Keep LLM evaluation separate from retrieval evaluation.
