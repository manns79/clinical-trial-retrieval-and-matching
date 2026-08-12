PYTHON ?= python3
PYTHONPATH ?= src

.PHONY: install install-dev install-ui install-dense test compile ingest-sample ingest-ctgov-sample report-ctgov-sample search-ctgov-sample download-ctgov-small build-trec-corpus-smoke build-bm25-index-sample evaluate-baseline evaluate-trec-bm25-sample compare-metrics-sample split-trec-2021-topics run-trec-2021-bm25 run-trec-2021-fielded-bm25 run-trec-2021-fielded-bm25-candidate compare-trec-2021-bm25 evaluate-trec-2021-lexical-holdout run-trec-2021-dense compare-trec-2021-lexical-dense check-retrieval-regression ingest-trec-sample validate-trec-sample write-manifest-sample api ui docker-up docker-down docker-smoke clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,ml]"

install-ui:
	$(PYTHON) -m pip install -e ".[ui]"

install-dense:
	$(PYTHON) -m pip install -e ".[dense]"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall -q src tests

ingest-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli ingest-sample \
		--trials data/fixtures/trials.sample.jsonl \
		--output data/processed/sample_trials.jsonl

ingest-ctgov-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli ingest-ctgov-studies \
		--input data/fixtures/ctgov_v2_studies.sample.json \
		--output data/processed/clinicaltrials/studies.sample.jsonl

report-ctgov-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli report-trial-corpus \
		--trials data/processed/clinicaltrials/studies.sample.jsonl \
		--output outputs/clinicaltrials_sample_report.json

search-ctgov-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli search-trials-bm25 \
		--trials data/processed/clinicaltrials/studies.sample.jsonl \
		--query "adult persistent asthma inhaled corticosteroid" \
		--top-k 5 \
		--retriever fielded-bm25 \
		--index-path data/indexes/clinicaltrials_sample_fielded_bm25.pkl \
		--output outputs/clinicaltrials_sample_bm25_search.json

download-ctgov-small:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli download-ctgov-studies \
		--query asthma \
		--status RECRUITING \
		--page-size 25 \
		--raw-output data/raw/clinicaltrials/asthma_recruiting_25.json \
		--manifest-output data/manifests/clinicaltrials_asthma_recruiting_25.json \
		--processed-output data/processed/clinicaltrials/asthma_recruiting_25.jsonl

build-trec-corpus-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli build-trec-trial-corpus \
		--qrels data/fixtures/qrels.sample.tsv \
		--year 2021 \
		--limit 2 \
		--raw-output data/raw/clinicaltrials/trec_2021_smoke_raw.json \
		--processed-output data/processed/clinicaltrials/trec_2021_smoke_trials.jsonl \
		--manifest-output data/manifests/trec_2021_smoke_trials.json \
		--report-output outputs/trec_2021_smoke_corpus_report.json

build-bm25-index-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli build-bm25-index \
		--trials data/fixtures/trials.sample.jsonl \
		--output data/indexes/sample_fielded_bm25.pkl \
		--retriever fielded-bm25

evaluate-baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli evaluate-baseline \
		--trials data/fixtures/trials.sample.jsonl \
		--topics data/fixtures/topics.sample.jsonl \
		--qrels data/fixtures/qrels.sample.tsv \
		--output outputs/sample_bm25_metrics.json

evaluate-trec-bm25-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli evaluate-trec-bm25 \
		--trials data/fixtures/trials.sample.jsonl \
		--topics data/fixtures/topics.sample.jsonl \
		--qrels data/fixtures/qrels.sample.tsv \
		--run-output outputs/sample_bm25.trec \
		--metrics-output outputs/sample_bm25_trec_metrics.json \
		--diagnostics-output outputs/sample_bm25_trec_diagnostics.json \
		--index-path data/indexes/sample_fielded_bm25.pkl \
		--retriever fielded-bm25 \
		--run-name bm25_fixture

compare-metrics-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics sample=outputs/sample_bm25_trec_metrics.json \
		--output outputs/sample_metrics_comparison.md \
		--view eligible_only

split-trec-2021-topics:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli split-trec-topics \
		--topics data/processed/trec/2021/topics.jsonl \
		--qrels data/processed/trec/2021/qrels.jsonl \
		--development-topics-output data/processed/trec/2021/splits/development/topics.jsonl \
		--development-qrels-output data/processed/trec/2021/splits/development/qrels.jsonl \
		--holdout-topics-output data/processed/trec/2021/splits/holdout/topics.jsonl \
		--holdout-qrels-output data/processed/trec/2021/splits/holdout/qrels.jsonl \
		--report-output outputs/trec_2021_topic_split.json \
		--seed ctmatch-trec-2021-v1 \
		--holdout-fraction 0.2

run-trec-2021-bm25:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-bm25-experiment \
		--config configs/experiments/trec_2021/plain_bm25.json

run-trec-2021-fielded-bm25:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-bm25-experiment \
		--config configs/experiments/trec_2021/fielded_bm25.json

run-trec-2021-fielded-bm25-candidate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-bm25-experiment \
		--config configs/experiments/trec_2021/fielded_bm25_condition_title_v1.json

compare-trec-2021-bm25:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics plain_bm25=outputs/trec_2021_development_plain_bm25_metrics.json \
		--metrics fielded_bm25=outputs/trec_2021_development_fielded_bm25_metrics.json \
		--metrics condition_title_v1=outputs/trec_2021_development_fielded_bm25_condition_title_v1_metrics.json \
		--output outputs/trec_2021_development_bm25_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

evaluate-trec-2021-lexical-holdout:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-bm25-experiment \
		--config configs/experiments/trec_2021/holdout_fielded_bm25_condition_title_v1.json

run-trec-2021-dense:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-dense-experiment \
		--config configs/experiments/trec_2021/development_dense_all_minilm_l6_v2.json

compare-trec-2021-lexical-dense:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics frozen_lexical=outputs/trec_2021_development_fielded_bm25_condition_title_v1_metrics.json \
		--metrics dense_all_minilm=outputs/trec_2021_development_dense_all_minilm_l6_v2_metrics.json \
		--output outputs/trec_2021_development_lexical_dense_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

check-retrieval-regression:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli check-retrieval-regression \
		--trials data/fixtures/trials.sample.jsonl \
		--topics data/fixtures/topics.sample.jsonl \
		--qrels data/fixtures/qrels.sample.tsv \
		--output outputs/retrieval_regression.json

ingest-trec-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli ingest-trec-topics \
		--year 2021 \
		--input data/fixtures/topics2021.sample.xml \
		--output data/processed/trec/2021/topics.jsonl
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli ingest-trec-qrels \
		--year 2021 \
		--input data/fixtures/qrels2021.sample.txt \
		--output data/processed/trec/2021/qrels.jsonl

validate-trec-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli validate-trec \
		--topics data/processed/trec/2021/topics.jsonl \
		--qrels data/processed/trec/2021/qrels.jsonl \
		--output outputs/sample_trec_validation.json

write-manifest-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli write-manifest \
		--name sample_trec_2021_topics \
		--dataset trec_clinical_trials \
		--year 2021 \
		--parser trec_topics_xml \
		--source-url https://trec.nist.gov/data/trials/topics2021.xml \
		--input data/fixtures/topics2021.sample.xml \
		--output data/manifests/sample_trec_2021_topics.json \
		--metadata fixture=synthetic

api:
	PYTHONPATH=$(PYTHONPATH) uvicorn clinical_trial_matching.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m streamlit run src/clinical_trial_matching/ui/streamlit_app.py --server.port 8501

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-smoke:
	sh scripts/docker_smoke_check.sh

clean:
	rm -rf data/processed data/manifests outputs .pytest_cache .mypy_cache .ruff_cache
