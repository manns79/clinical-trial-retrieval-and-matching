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
make ingest-ctgov-sample
make report-ctgov-sample
make search-ctgov-sample
make evaluate-baseline
make evaluate-trec-bm25-sample
make check-retrieval-regression
make ingest-trec-sample
make validate-trec-sample
make write-manifest-sample
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

ctmatch write-manifest \
  --name trec_2021_topics \
  --dataset trec_clinical_trials \
  --year 2021 \
  --parser trec_topics_xml \
  --source-url https://trec.nist.gov/data/trials/topics2021.xml \
  --input data/raw/trec/2021/topics2021.xml \
  --output data/manifests/trec_2021_topics.json

ctmatch write-manifest \
  --name trec_2021_qrels \
  --dataset trec_clinical_trials \
  --year 2021 \
  --parser trec_qrels \
  --source-url https://trec.nist.gov/data/trials/qrels2021.txt \
  --input data/raw/trec/2021/qrels2021.txt \
  --output data/manifests/trec_2021_qrels.json
```

## ClinicalTrials.gov Ingestion

ClinicalTrials.gov v2 records should be stored under ignored local paths such as `data/raw/clinicaltrials/`. Do not commit full API responses, full JSON downloads, normalized trial corpora, or vector indexes.

Example command for a small live API query:

```bash
ctmatch download-ctgov-studies \
  --query asthma \
  --status RECRUITING \
  --page-size 25 \
  --raw-output data/raw/clinicaltrials/asthma_recruiting_25.json \
  --manifest-output data/manifests/clinicaltrials_asthma_recruiting_25.json \
  --processed-output data/processed/clinicaltrials/asthma_recruiting_25.jsonl
```

This command writes the raw API response, a source manifest with request/checksum metadata, and normalized trial JSONL. The equivalent Make target is:

```bash
make download-ctgov-small
```

After normalizing a trial corpus, generate a lightweight validation report:

```bash
ctmatch report-trial-corpus \
  --trials data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
  --output outputs/clinicaltrials_asthma_recruiting_25_report.json
```

Search a normalized trial corpus with BM25:

```bash
ctmatch search-trials-bm25 \
  --trials data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
  --query "adult persistent asthma inhaled corticosteroid" \
  --top-k 10 \
  --output outputs/clinicaltrials_asthma_recruiting_25_search.json
```

Run the tiny BM25 retrieval-quality regression check:

```bash
make check-retrieval-regression
```

This uses fixed synthetic trial/topic/qrels fixtures and fails if recall, MRR, or nDCG drops below the configured thresholds.

Write a TREC-format BM25 run file and metrics report:

```bash
ctmatch evaluate-trec-bm25 \
  --trials data/processed/clinicaltrials/benchmark_trials.jsonl \
  --topics data/processed/trec/2021/topics.jsonl \
  --qrels data/processed/trec/2021/qrels.jsonl \
  --run-output outputs/trec_2021_bm25.run \
  --metrics-output outputs/trec_2021_bm25_metrics.json \
  --diagnostics-output outputs/trec_2021_bm25_diagnostics.json \
  --retriever fielded-bm25 \
  --run-name bm25_2021
```

Use `--retriever bm25` for the older single-text BM25 baseline. The field-aware
baseline keeps an `all_text` BM25 backbone and adds separate boosts for title,
brief summary, conditions, interventions, eligibility criteria, demographics,
status, and locations via repeated `--field-weight field=value` options.

The metrics report includes separate `excluded_or_eligible` and `eligible_only` views so
topical retrieval quality is not confused with true eligibility matching. The diagnostics
file records per-topic recall, first eligible rank, and a compact list of weak topics.

If you built a ClinicalTrials.gov corpus before `brief_summary` was added, re-normalize
from the ignored raw JSON before benchmarking so summary boosts are populated:

```bash
ctmatch ingest-ctgov-studies \
  --input data/raw/clinicaltrials/trec_2021_qrels_trials_raw.json \
  --output data/processed/clinicaltrials/trec_2021_qrels_trials.jsonl
```

The run file uses standard TREC format:

```text
topic_id Q0 nct_id rank score run_name
```

Build a TREC 2021 benchmark trial corpus from qrels NCT IDs:

```bash
ctmatch build-trec-trial-corpus \
  --qrels data/processed/trec/2021/qrels.jsonl \
  --year 2021 \
  --raw-output data/raw/clinicaltrials/trec_2021_qrels_trials_raw.json \
  --processed-output data/processed/clinicaltrials/trec_2021_qrels_trials.jsonl \
  --manifest-output data/manifests/trec_2021_qrels_trials.json \
  --report-output outputs/trec_2021_qrels_trials_report.json \
  --batch-size 100 \
  --delay-seconds 0.2
```

For a cheap first smoke run, add `--limit 25`.

Example command for a small local v2 API response:

```bash
ctmatch ingest-ctgov-studies \
  --input data/raw/clinicaltrials/studies.sample.json \
  --output data/processed/clinicaltrials/studies.jsonl
```

The parser currently supports:

- a single v2 study record with `protocolSection`
- a JSON array of study records
- a v2 API response object with a top-level `studies` list

To run the API after installing dependencies:

```bash
python3 -m pip install -e .
make ingest-ctgov-sample
make api
```

Then visit `http://localhost:8000/health` or `http://localhost:8000/metrics/health`.

Search the configured corpus:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"adult persistent asthma inhaled corticosteroid","top_k":5}'
```

Search responses include `latency_ms` fields for corpus loading, retrieval, and total search handling. Every HTTP response also includes an `X-Process-Time-Ms` header, and the API writes structured JSON logs for HTTP requests and search events.

Fetch a full normalized trial record:

```bash
curl http://localhost:8000/trial/NCT99991001
```

By default, the API reads `data/processed/clinicaltrials/studies.sample.jsonl`. To use a live downloaded corpus:

```bash
TRIAL_CORPUS_PATH=data/processed/clinicaltrials/asthma_recruiting_25.jsonl make api
```

## Streamlit UI

Install the UI extra:

```bash
python3 -m pip install -e ".[ui]"
```

Start the API:

```bash
make ingest-ctgov-sample
make api
```

In another terminal, start Streamlit:

```bash
make ui
```

The UI opens at `http://localhost:8501` and calls the FastAPI service at `API_BASE_URL`, defaulting to `http://localhost:8000`.

## Docker Compose Demo

Run the local API + Streamlit UI demo with one command:

```bash
make docker-up
```

Compose starts:

- FastAPI at `http://localhost:8000`
- Streamlit at `http://localhost:8501`
- Postgres/pgvector for the later database-backed milestone

The API service seeds the small synthetic ClinicalTrials.gov fixture into `data/processed/clinicaltrials/studies.sample.jsonl` before starting. Generated data remains ignored by git.

Run the Docker image smoke check:

```bash
make docker-smoke
```

The smoke check builds the image, starts the API with the synthetic fixture mounted, and verifies `/health`, `/search`, and `/trial/NCT99991001`.

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
