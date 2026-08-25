# Architecture

## Modes

The project is intentionally split into two operating modes:

1. Full reproducible mode: local Docker Compose, frozen TREC/ClinicalTrials.gov snapshots, full evaluation, and larger indexes.
2. Public demo mode: a compact subset or prebuilt index, deployed on a free tier with transparent limitations.

This keeps the portfolio artifact deployable without forcing paid database or model hosting.

## Components

- Ingestion: downloads or reads source records, records provenance, and normalizes trial fields.
- Storage: SQLite for disk-backed serving metadata and FTS5 retrieval; Postgres remains an
  optional later deployment path for structured fields and vector search.
- Retrieval: BM25 evaluation baseline, SQLite FTS5 serving, metadata filters, dense retrieval,
  hybrid fusion, and reranking.
- Evaluation: TREC-style qrels, reproducible run files, and metrics such as Recall@100, nDCG, MRR, and Precision@k.
- API: FastAPI service for search, trial lookup, health, and later explanation endpoints.
- UI: Streamlit client over the FastAPI search and trial-detail endpoints.
- Observability: structured logs, latency timing, health checks, and retrieval regression tests.

## Initial API Surface

- `GET /health`
- `POST /search`
- `GET /trial/{nct_id}`
- `GET /metrics/health` for lightweight operational checks

`POST /search` supports SQLite FTS5, plain BM25, the frozen field-aware BM25 profile, the selected
local dense bi-encoder, and equal-weight reciprocal-rank fusion. SQLite FTS5 is the default and
the lexical component of hybrid retrieval; legacy Python BM25 remains available for comparison
but is not preloaded. Primary serving modes validate a SQLite metadata store configured through
`TRIAL_STORE_PATH` against the normalized JSONL snapshot at `TRIAL_CORPUS_PATH`. Dense and hybrid
modes require a compatible ignored NumPy index configured
through `DENSE_INDEX_PATH`; the model and index are validated and loaded once in the FastAPI
lifespan before requests are accepted. The service never builds corpus embeddings on a request.

Search responses include stage latency for lexical retrieval, query embedding/dense scoring,
fusion, on-demand metadata loading, and total handler time. Hybrid records retain each component rank alongside the fused
score. `/metrics/health` advertises only retrievers supported by the currently mounted artifacts.

`GET /trial/{nct_id}` loads one full normalized record from the SQLite metadata store. Search
loads only the top result records in one bounded query for snippets and traceable fields.

## Streamlit UI

The Streamlit UI is a thin client over FastAPI. It reads `API_BASE_URL`, defaults to
`http://localhost:8000`, discovers available retrievers through `/metrics/health`, posts searches
to `/search`, displays stage latency, and fetches details from `/trial/{nct_id}`. It does not run
models or retrieval locally.

## Docker Compose

Docker Compose runs FastAPI, Streamlit, and Postgres/pgvector. The API image includes the optional
dense dependencies, while the default Compose startup still seeds the small synthetic corpus and
offers SQLite lexical search without downloading a model. Mounting compatible corpus/SQLite/dense index
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

## Serving Performance

The tracked local serving benchmark uses the same environment contract and cached resources as
FastAPI. It measures API import plus resource preload as cold start, performs deterministic
warmups, and interleaves SQLite FTS5, dense, and hybrid requests over synthetic queries. Reports
capture handler and stage p50/p95 latency, sequential throughput, RSS, artifact sizes, and system
metadata under ignored `outputs/` paths.

Startup profiling checkpoints the process after API import, trial-store validation, SQLite FTS5
loading, dense index loading, encoder-framework import, model/session construction,
first-inference/thread-pool initialization, and retriever assembly. Each checkpoint reports
elapsed time, retained RSS change, and
observed peak RSS change. These deltas describe one process on one host and can be influenced by
allocator behavior and operating-system filesystem caches.

This benchmark intentionally excludes HTTP transport and concurrency. It answers whether the
single-process retrieval stack has plausible memory and latency headroom before adding a
cross-encoder. A later deployment/load test should measure network serialization, concurrent
requests, queueing, and worker-level memory duplication separately.

Development-only reranking reads the selected hybrid TREC run, fetches candidate metadata from
the SQLite trial store, and scores query/trial pairs with a pinned ONNX cross-encoder. Candidate
depths are evaluated independently; the untouched baseline tail is appended after each reranked
window. A separate headroom command loads the selected API retrieval stack before the reranker and
executes real depth probes, preventing standalone model memory from being mistaken for combined
serving memory. Reranking is not wired into `/search` until quality and latency gates pass.

The depth-10 optimization comparison treats serving as a two-part gate. Candidate reports must
preserve both eligible-only and excluded-or-eligible nDCG@10 gains relative to FP32, while a
provisional 500 ms reranked-mode p95 budget reserves 250 ms for hybrid retrieval and 250 ms for
incremental reranking. Quantized model selection, context length, and text representation remain
versioned experiment inputs rather than API settings until one profile passes both gates.

The final small-window comparison supports per-row candidate depths against a fixed reference.
Depths 3 and 5 met latency but missed quality; depth 8 met quality but missed latency. Because no
profile passed both predeclared gates, the API has no reranker lifecycle, model load, request mode,
or response contract. This keeps evaluation code from silently becoming production behavior.

## TREC-Style Evaluation

The BM25 benchmark command reads normalized trial JSONL, normalized TREC topic JSONL, and qrels, then writes both a six-column TREC run file and a metrics JSON report. Generated run files and reports belong under ignored `outputs/` paths.

The dense benchmark path uses a local sentence-transformer to build L2-normalized corpus
embeddings and persists them with NCT IDs and compatibility metadata in an ignored NumPy `.npz`
index. The selected serving backend exports the same MiniLM query encoder to an ignored,
checksum-validated ONNX artifact and runs it with `tokenizers` plus ONNX Runtime, without loading
PyTorch. Query/corpus cosine similarity remains a matrix product. Six-decimal score ties resolve
by NCT ID so numerically equivalent backends produce deterministic rankings.

## Benchmark Corpus Build

The TREC corpus builder extracts unique NCT IDs from qrels, fetches matching ClinicalTrials.gov v2 records with `query.id`, writes the raw response envelope under ignored `data/raw/`, normalizes records to trial JSONL under ignored `data/processed/`, and records manifest/report metadata. This keeps downloaded benchmark assets local while making the build process reproducible.

## Guardrails

- Always distinguish benchmark snapshots from live ClinicalTrials.gov data.
- Treat explanations as grounded summaries, not eligibility determinations.
- Require citations to trial fields for generated explanations.
- Keep LLM evaluation separate from retrieval evaluation.
