PYTHON ?= python3
PYTHONPATH ?= src

.PHONY: install install-dev test compile ingest-sample evaluate-baseline ingest-trec-sample validate-trec-sample write-manifest-sample api docker-up docker-down clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,ml]"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall -q src tests

ingest-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli ingest-sample \
		--trials data/fixtures/trials.sample.jsonl \
		--output data/processed/sample_trials.jsonl

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

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf data/processed data/manifests outputs .pytest_cache .mypy_cache .ruff_cache
