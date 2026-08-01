# Roadmap

## Milestone 1: Reproducible Baseline

- Create source manifests and checksums.
- Parse TREC topics and qrels.
- Normalize trial text fields.
- Produce BM25 run files and metrics.
- Add CI regression tests on synthetic fixtures.

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
