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
- FastAPI `/search` is wired to the same BM25 path over a configured normalized trial JSONL corpus.
- FastAPI `/trial/{nct_id}` returns full normalized trial details from the configured corpus.
- Streamlit UI is scaffolded as a thin client over `/search` and `/trial/{nct_id}`.
- Docker Compose runs API, Streamlit UI, and Postgres/pgvector with one command for the local demo.
- GitHub Actions includes a Docker build and API smoke check.
- API observability includes request timing headers, structured logs, and `/search` latency fields.
- Tiny BM25 retrieval-quality regression checks run in CI over fixed synthetic fixtures.

## Milestone 2: Structured Retrieval

- Add age, sex, recruitment status, and location normalization.
- Use structured filters before ranking.
- Report ablations against BM25-only retrieval.

## Milestone 3: Dense and Hybrid Retrieval

- Generate biomedical sentence embeddings.
- Add pgvector or a local vector index.
- Implement hybrid fusion and latency measurements.

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
