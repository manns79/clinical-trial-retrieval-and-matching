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
ctmatch build-bm25-index \
  --trials data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
  --output data/indexes/asthma_recruiting_25_fielded_bm25.pkl \
  --retriever fielded-bm25

ctmatch search-trials-bm25 \
  --trials data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
  --query "adult persistent asthma inhaled corticosteroid" \
  --top-k 10 \
  --index-path data/indexes/asthma_recruiting_25_fielded_bm25.pkl \
  --output outputs/clinicaltrials_asthma_recruiting_25_search.json
```

BM25 index files are ignored local artifacts. Prefer `.pkl` for speed and only load indexes
you generated locally from this project.

Run the tiny BM25 retrieval-quality regression check:

```bash
make check-retrieval-regression
```

This uses fixed synthetic trial/topic/qrels fixtures and fails if recall, MRR, or nDCG drops below the configured thresholds.

Write a TREC-format BM25 run file and metrics report:

```bash
ctmatch build-bm25-index \
  --trials data/processed/clinicaltrials/benchmark_trials.jsonl \
  --output data/indexes/trec_2021_fielded_bm25.pkl \
  --retriever fielded-bm25

ctmatch evaluate-trec-bm25 \
  --trials data/processed/clinicaltrials/benchmark_trials.jsonl \
  --topics data/processed/trec/2021/topics.jsonl \
  --qrels data/processed/trec/2021/qrels.jsonl \
  --run-output outputs/trec_2021_bm25.run \
  --metrics-output outputs/trec_2021_bm25_metrics.json \
  --diagnostics-output outputs/trec_2021_bm25_diagnostics.json \
  --index-path data/indexes/trec_2021_fielded_bm25.pkl \
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

Compare several metrics reports in one compact table:

```bash
ctmatch compare-metrics \
  --metrics plain_bm25=outputs/trec_2021_plain_bm25_metrics.json \
  --metrics fielded_bm25=outputs/trec_2021_bm25_metrics.json \
  --output outputs/trec_2021_baseline_comparison.md \
  --view eligible_only \
  --view excluded_or_eligible
```

The output format is inferred from the suffix: `.md`, `.csv`, or `.json`.

Reproduce named BM25 experiments from the tracked registry instead of repeating paths and
field-weight flags:

```bash
make split-trec-2021-topics
make run-trec-2021-bm25
make run-trec-2021-fielded-bm25
make run-trec-2021-fielded-bm25-candidate
make compare-trec-2021-bm25
make run-trec-2021-sqlite-fts5
make compare-trec-2021-lexical-backends
```

The split command uses seeded SHA-256 ranking to assign exactly 20% of topics to holdout. For
TREC 2021 this produces 60 development topics and 15 holdout topics, partitions their qrels with
the same assignment, and writes `outputs/trec_2021_topic_split.json` with source checksums,
topic IDs, relevance distributions, and overlap checks. Re-running it with the same inputs and
seed produces the same assignments.

The underlying command accepts any compatible experiment spec:

```bash
ctmatch run-bm25-experiment \
  --config configs/experiments/trec_2021/fielded_bm25.json
```

Specs under `configs/experiments/` pin the retriever, all field weights, benchmark inputs,
and local artifact paths. Metrics and diagnostics include the config path and SHA-256 checksum
that produced them. The current tuning specs read only the development partition. Select and
freeze one lexical profile before evaluating it on holdout; do not use holdout metrics for
iterative weight selection. Data, split records, and generated artifacts remain ignored and
must not be committed.

This project inspected full TREC 2021 metrics before introducing the split, so the 2021 holdout
is a workflow guardrail rather than a claim of a historically unseen test set. A later TREC year
should provide the stronger external evaluation.

The lexical selection rationale, frozen weights, and aggregate holdout result are recorded in
[`docs/lexical_baseline_selection.md`](docs/lexical_baseline_selection.md). The holdout config is
kept for reproducibility; its result is not used for further lexical tuning.

The serving optimization experiment persists title, brief summary, conditions, interventions,
and eligibility criteria in a local SQLite FTS5 index. It uses the selected condition/title-heavy
weights, reads development topics only, and writes all database/run/report artifacts under ignored
paths. Build and evaluate it with `make run-trec-2021-sqlite-fts5`; compare isolated cold start,
RSS, warm latency, throughput, and disk size with `make benchmark-trec-2021-lexical-backends`.
The aggregate selection record is in
[`docs/sqlite_fts5_selection.md`](docs/sqlite_fts5_selection.md).

Run the first free, local sentence-transformer baseline on development topics:

```bash
make install-dense
make run-trec-2021-dense
make run-trec-2021-dense-biomedical
make compare-trec-2021-dense-ablation
make compare-trec-2021-lexical-dense
make run-trec-2021-hybrid
make run-trec-2021-hybrid-sqlite
make compare-trec-2021-hybrid-backends
make compare-trec-2021-retrievers

cat outputs/trec_2021_development_dense_ablation_comparison.md
cat outputs/trec_2021_development_retriever_comparison.md
```

The first run downloads the public `sentence-transformers/all-MiniLM-L6-v2` model, embeds the
26,150-trial corpus in batches on CPU, and writes an ignored NumPy `.npz` index. Later runs reuse
that index after validating its model, text representation, sequence limit, NCT order, embedding
dimension, and corpus fingerprint. Loading uses `allow_pickle=False`.

The selected dense profile uses `title_summary_conditions`, which labels and concatenates the
trial title, brief summary, and conditions. The tracked biomedical ablation combines
`abhinand/MedEmbed-small-v0.1` with an `eligibility_snapshot` representation. Other supported
representations are `title`, `clinical_core`, and `all_fields`; use a distinct index/output name
for each ablation. Dense experiments remain development-only. Do not evaluate dense or hybrid
candidates on the lexical holdout.

The hybrid experiment performs deterministic, equal-weight reciprocal-rank fusion over the
frozen lexical and selected dense development run files. Its metrics report includes component
paths, weights, and run checksums for traceability. The aggregate dense selection and RRF results
are recorded in [`docs/dense_hybrid_selection.md`](docs/dense_hybrid_selection.md).

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

The request accepts `sqlite-fts5`, `fielded-bm25`, `bm25`, `dense`, or `hybrid` as its
`retriever`; SQLite FTS5 is the default. Dense and
hybrid are advertised by `/metrics/health` only when a configured dense index exists. Search
responses report `lexical`, `embedding`, `fusion`, and `total` latency separately, alongside
corpus/index loading and combined retrieval timing. Every HTTP response also includes an
`X-Process-Time-Ms` header, and the API writes structured JSON logs for search events.

Set `SQLITE_FTS_INDEX_PATH` when running the API to reuse the selected disk-backed index:

```bash
ctmatch build-trial-store \
  --trials data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
  --output data/indexes/asthma_recruiting_25_trial_store.sqlite

TRIAL_CORPUS_PATH=data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
TRIAL_STORE_PATH=data/indexes/asthma_recruiting_25_trial_store.sqlite \
SQLITE_FTS_INDEX_PATH=data/indexes/asthma_recruiting_25_sqlite_fts5.sqlite \
make api
```

Search responses include `latency_ms.index_load`; after the first request this should usually
drop near zero because the API caches the validated retriever. Using a new index path lets startup
build and persist a compatible SQLite index automatically. Legacy fielded and plain BM25 remain
explicit comparison modes but are not preloaded because their expanded Python postings consume
substantially more memory.

Serve the selected dense and hybrid profiles over the local TREC corpus:

```bash
make install-onnx
make export-trec-2021-onnx-encoder

TRIAL_CORPUS_PATH=data/processed/clinicaltrials/trec_2021_qrels_trials.jsonl \
TRIAL_STORE_PATH=data/indexes/trec_2021_trial_metadata.sqlite \
SQLITE_FTS_INDEX_PATH=data/indexes/trec_2021_sqlite_fts5_condition_title_v1.sqlite \
DENSE_INDEX_PATH=data/indexes/trec_2021_dense_all_minilm_l6_v2_title_summary_conditions.npz \
DENSE_ENCODER_BACKEND=onnxruntime \
DENSE_ONNX_MODEL_PATH=data/models/onnx/all-MiniLM-L6-v2 \
make api
```

The export command uses the locally cached sentence-transformer and writes an ignored ONNX model,
tokenizer, and checksum metadata under `data/models/`. Serving then validates the metadata store,
dense index, and ONNX artifact once during startup. It does not import PyTorch, rebuild corpus
embeddings, or call a hosted service. Full records and snippets are loaded from SQLite only for
returned results; `/trial` loads one record by NCT ID.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"adult persistent asthma","top_k":10,"retriever":"hybrid"}'
```

Hybrid results include `component_ranks` for the SQLite FTS5 and dense rankings. Model files,
NumPy indexes, and generated search outputs remain local ignored artifacts.

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

## Serving Benchmark

Measure the selected lexical, dense, and hybrid profiles over the local 26,150-trial corpus:

```bash
make benchmark-trec-2021-serving
```

The tracked spec at `configs/benchmarks/trec_2021_local_serving.json` pins five synthetic queries,
one warmup round, five measurement rounds, `top_k=10`, the selected indexes, the ONNX Runtime
MiniLM query encoder, and equal-weight RRF settings. The command writes the ignored report to
`outputs/trec_2021_local_serving_benchmark.json`.

The report includes:

- API import and metadata-store/index/framework/model/first-inference/thread-pool preload time
- Per-phase retained and peak RSS deltas for the trial store, SQLite FTS5, and dense resources
- The startup resource phase responsible for the largest positive retained RSS increase
- Warm handler latency with minimum, mean, p50, p95, and maximum values
- Lexical, embedding, fusion, metadata, and total stage latency for each retriever
- Per-mode and aggregate sequential requests per second
- RSS before startup, after startup, after measurement, and process peak
- Corpus, lexical index, dense index, and local model-cache sizes
- Python, platform, CPU-count, and key package-version metadata

Warm measurements call the FastAPI search handler in-process after deterministic warmup. They
exclude HTTP transport and response serialization. Throughput is sequential single-process
capacity, not a concurrent load-test claim. The exported ONNX artifact must already exist because
the benchmark forces offline mode; no hosted service or paid API is used.

## Cross-Encoder Reranking Experiment

Run the development-only local reranking sweep after building the selected hybrid run:

```bash
make download-trec-2021-cross-encoder
make run-trec-2021-cross-encoder
make benchmark-trec-2021-cross-encoder-headroom
```

The tracked experiment reranks the top 10, 25, and 50 hybrid candidates with a pinned FP32 ONNX
MiniLM cross-encoder. It writes per-depth TREC runs, metrics, diagnostics, metadata/inference/total
latency, and a separate memory report that loads the selected serving stack first. Trial records,
model files, run files, and detailed reports stay in ignored directories. The experiment reads
only development topics and does not change the API retriever modes.

Run the depth-10 quantization/context/text optimization suite with:

```bash
make download-trec-2021-cross-encoder-optimization
make run-trec-2021-cross-encoder-optimization
```

The comparison requires both eligible-only and broad nDCG@10 gains to match the FP32 reference
and reserves 250 ms p95 for reranking inside a separate 500 ms optional-mode budget. It records a
compact ignored JSON/Markdown gate report. No current profile passes both gates, so reranking
remains outside the API.

The final small-window gate can be reproduced with:

```bash
make download-trec-2021-cross-encoder-small-depths
make run-trec-2021-cross-encoder-small-depths
```

Depths 3 and 5 passed latency but did not preserve both reference nDCG gains. Depth 8 preserved
both gains but exceeded the incremental reranking allowance. The reranking investigation is now
closed for this project phase rather than weakening a predeclared gate.

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
