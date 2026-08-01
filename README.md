# Clinical Trial Retrieval and Matching

A public, deployable ML/IR portfolio project for matching short patient descriptions to potentially relevant clinical trials using public ClinicalTrials.gov data and TREC Clinical Trials relevance judgments.

This project is a research and engineering demo. It is not a medical device, does not provide medical advice, and should not be used to determine eligibility for real clinical trial enrollment.

## Project Goals

- Reproducible data ingestion from public clinical trial sources.
- Strong retrieval baselines before heavier ML components.
- Traceable ranking explanations grounded in trial fields.
- API serving, Dockerized local infrastructure, CI, and monitoring hooks.
- A usable interface and public demo path that can run on free tiers where possible.

## Quickstart

```bash
make compile
make test
make ingest-sample
make evaluate-baseline
make ingest-trec-sample
make validate-trec-sample
```

The initial sample commands use synthetic fixture data so they can run without downloading large benchmark corpora.

## TREC Ingestion

Real TREC inputs should live under ignored local paths such as `data/raw/trec/2021/`. Do not commit downloaded benchmark corpora, normalized benchmark outputs, or evaluation artifacts.

Example commands once you have downloaded the official files locally:

```bash
ctmatch ingest-trec-topics \
  --year 2021 \
  --input data/raw/trec/2021/topics2021.xml \
  --output data/processed/trec/2021/topics.jsonl

ctmatch ingest-trec-qrels \
  --year 2021 \
  --input data/raw/trec/2021/qrels2021.txt \
  --output data/processed/trec/2021/qrels.jsonl

ctmatch validate-trec \
  --topics data/processed/trec/2021/topics.jsonl \
  --qrels data/processed/trec/2021/qrels.jsonl \
  --output outputs/trec_2021_validation.json
```

To run the API after installing dependencies:

```bash
python3 -m pip install -e .
make api
```

Then visit `http://localhost:8000/health`.

## Repository Layout

```text
configs/                         Runtime and pipeline configuration
data/fixtures/                   Tiny synthetic fixtures for tests and examples
docs/                            Architecture, data, and roadmap notes
scripts/                         Database and operational scripts
src/clinical_trial_matching/     Application package
tests/                           Unit tests
```

## Milestones

1. Frozen benchmark ingestion and BM25 evaluation.
2. Metadata filters and structured normalization.
3. Dense retrieval and hybrid ranking.
4. Cross-encoder reranking and ablations.
5. Grounded LLM extraction/explanation with separate hallucination checks.
6. Public demo deployment with a compact index and clear scope limits.

## Cost Posture

The default path is local-first and free-first:

- Local Docker Compose for Postgres/pgvector and API development.
- Public GitHub repository for free standard GitHub Actions CI.
- Local MLflow file tracking before any hosted experiment service.
- Small public demo dataset for free-tier deployment rather than hosting the full corpus.
