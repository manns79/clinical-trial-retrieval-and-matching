# Roadmap

## Milestone 1: Reproducible Baseline

- Create source manifests and checksums.
- Parse TREC topics and qrels.
- Normalize trial text fields.
- Produce BM25 run files and metrics.
- Add CI regression tests on synthetic fixtures.

Current TREC ingestion status:

- 2021/2022-style free-text topic XML parsing is scaffolded.
- 2023-style fielded topic XML parsing is scaffolded.
- TREC qrels normalization and validation are scaffolded.
- Real benchmark files are expected under ignored local `data/raw/` paths.
- Source manifest/checksum writing is scaffolded under ignored local `data/manifests/` paths.
- ClinicalTrials.gov v2 JSON study parsing is scaffolded for small API responses and fixtures.
- Small live ClinicalTrials.gov query downloads are scaffolded with raw JSON, manifest, and normalized JSONL outputs.
- Trial corpus validation reports are scaffolded for normalized ClinicalTrials.gov JSONL files.
- BM25 command-line search is scaffolded for normalized ClinicalTrials.gov JSONL files.
- FastAPI `/search` serves lexical, dense bi-encoder, and reciprocal-rank-fusion hybrid modes.
- FastAPI `/trial/{nct_id}` returns full normalized trial details from the configured corpus.
- Streamlit UI is scaffolded as a thin client over `/search` and `/trial/{nct_id}`.
- Docker Compose runs API, Streamlit UI, and Postgres/pgvector with one command for the local demo.
- GitHub Actions includes a Docker build and API smoke check.
- API observability includes request timing headers, structured logs, and `/search` latency fields.
- Tiny BM25 retrieval-quality regression checks run in CI over fixed synthetic fixtures.
- TREC-style BM25 evaluation writes run files and metrics reports from normalized local benchmark files.
- TREC qrels-linked ClinicalTrials.gov corpus building is scaffolded with raw, normalized, manifest, and report outputs.

## Milestone 2: Structured Retrieval

- Add age, sex, recruitment status, and location normalization.
- Use structured filters before ranking.
- Report ablations against BM25-only retrieval.

## Milestone 3: Dense and Hybrid Retrieval

- Generate biomedical sentence embeddings.
- Add pgvector or a local vector index.
- Implement hybrid fusion and latency measurements.

Current dense retrieval status:

- A configurable local sentence-transformer bi-encoder baseline is scaffolded and served by API.
- Batched corpus embeddings persist in a validated NumPy index.
- Dense TREC evaluation and lexical/dense development comparison commands are scaffolded.
- A biomedical profile ablation selected MiniLM `title_summary_conditions` on development topics.
- Equal-weight RRF is evaluated and served with traceable component ranks and stage latency.
- A repeatable local serving benchmark reports cold start, warm p50/p95 latency, sequential
  throughput, RSS, and artifact/model-cache sizes for lexical, dense, and hybrid modes.
- Staged startup profiling identifies the expanded Python fielded BM25 index as the dominant
  serving-memory contributor.
- A local SQLite FTS5 prototype preserves competitive development quality, reduces lexical
  retained RSS from about 2.8 GiB to about 3 MiB, and is now the default/hybrid serving backend.

## Milestone 4: Reranking

- Add a biomedical cross-encoder over top candidates.
- Track candidate count, latency, and metric gains.
- Store experiment metadata locally with MLflow.

## Milestone 5: Grounded Explanations

- Extract eligibility criteria into structured JSON.
- Generate explanations with field-level citations.
- Evaluate hallucination and extraction quality separately from retrieval quality.

## Milestone 6: Public Demo

- Deploy a compact demo API/UI on a free tier.
- Keep the full benchmark workflow reproducible locally.
- Publish architecture, limitations, and benchmark reports.
