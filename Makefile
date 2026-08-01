PYTHON ?= python3
PYTHONPATH ?= src

.PHONY: install install-dev test compile ingest-sample evaluate-baseline api docker-up docker-down clean

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

api:
	PYTHONPATH=$(PYTHONPATH) uvicorn clinical_trial_matching.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf data/processed outputs .pytest_cache .mypy_cache .ruff_cache
