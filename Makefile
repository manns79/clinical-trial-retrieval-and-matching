PYTHON ?= python3
PYTHONPATH ?= src

.PHONY: install install-dev install-ui test compile ingest-sample ingest-ctgov-sample report-ctgov-sample search-ctgov-sample download-ctgov-small evaluate-baseline ingest-trec-sample validate-trec-sample write-manifest-sample api ui docker-up docker-down clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,ml]"

install-ui:
	$(PYTHON) -m pip install -e ".[ui]"

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
		--output outputs/clinicaltrials_sample_bm25_search.json

download-ctgov-small:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli download-ctgov-studies \
		--query asthma \
		--status RECRUITING \
		--page-size 25 \
		--raw-output data/raw/clinicaltrials/asthma_recruiting_25.json \
		--manifest-output data/manifests/clinicaltrials_asthma_recruiting_25.json \
		--processed-output data/processed/clinicaltrials/asthma_recruiting_25.jsonl

evaluate-baseline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli evaluate-baseline \
		--trials data/fixtures/trials.sample.jsonl \
		--topics data/fixtures/topics.sample.jsonl \
		--qrels data/fixtures/qrels.sample.tsv \
		--output outputs/sample_bm25_metrics.json

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

clean:
	rm -rf data/processed data/manifests outputs .pytest_cache .mypy_cache .ruff_cache
