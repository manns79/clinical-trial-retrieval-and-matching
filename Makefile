PYTHON ?= python3
PYTHONPATH ?= src

.PHONY: install install-dev install-ui install-dense install-onnx test compile ingest-sample ingest-ctgov-sample build-trial-store-sample build-trec-2021-trial-store report-ctgov-sample search-ctgov-sample download-ctgov-small build-trec-corpus-smoke build-bm25-index-sample evaluate-baseline evaluate-trec-bm25-sample compare-metrics-sample split-trec-2021-topics run-trec-2021-bm25 run-trec-2021-fielded-bm25 run-trec-2021-fielded-bm25-candidate run-trec-2021-sqlite-fts5 compare-trec-2021-bm25 compare-trec-2021-lexical-backends evaluate-trec-2021-lexical-holdout run-trec-2021-dense run-trec-2021-dense-biomedical export-trec-2021-onnx-encoder run-trec-2021-dense-onnx check-trec-2021-dense-onnx-parity convert-trec-2021-dense-mmap run-trec-2021-dense-mmap-int8 compare-trec-2021-dense-optimization compare-trec-2021-dense-ablation compare-trec-2021-lexical-dense run-trec-2021-hybrid run-trec-2021-hybrid-sqlite compare-trec-2021-retrievers compare-trec-2021-hybrid-backends download-trec-2021-cross-encoder run-trec-2021-cross-encoder benchmark-trec-2021-cross-encoder-headroom download-trec-2021-cross-encoder-optimization run-trec-2021-cross-encoder-optimization download-trec-2021-cross-encoder-small-depths run-trec-2021-cross-encoder-small-depths benchmark-trec-2021-serving benchmark-trec-2021-serving-sentence-transformers benchmark-trec-2021-serving-mmap benchmark-trec-2021-serving-mmap-int8 assess-trec-2021-serving-budget benchmark-trec-2021-lexical-backends check-retrieval-regression ingest-trec-sample validate-trec-sample write-manifest-sample api ui docker-up docker-down docker-smoke clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,ml]"

install-ui:
	$(PYTHON) -m pip install -e ".[ui]"

install-dense:
	$(PYTHON) -m pip install -e ".[dense]"

install-onnx:
	$(PYTHON) -m pip install -e ".[dense,onnx]"

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

build-trial-store-sample: ingest-ctgov-sample
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli build-trial-store \
		--trials data/processed/clinicaltrials/studies.sample.jsonl \
		--output data/indexes/studies_sample_trial_store.sqlite

build-trec-2021-trial-store:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli build-trial-store \
		--trials data/processed/clinicaltrials/trec_2021_qrels_trials.jsonl \
		--output data/indexes/trec_2021_trial_metadata.sqlite

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

run-trec-2021-sqlite-fts5:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-sqlite-fts-experiment \
		--config configs/experiments/trec_2021/development_sqlite_fts5_condition_title_v1.json

compare-trec-2021-lexical-backends:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics frozen_lexical=outputs/trec_2021_development_fielded_bm25_condition_title_v1_metrics.json \
		--metrics sqlite_fts5=outputs/trec_2021_development_sqlite_fts5_condition_title_v1_metrics.json \
		--output outputs/trec_2021_development_lexical_backend_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

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
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-dense-experiment \
		--config configs/experiments/trec_2021/development_dense_all_minilm_l6_v2.json

export-trec-2021-onnx-encoder:
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli export-onnx-encoder \
		--model-name sentence-transformers/all-MiniLM-L6-v2 \
		--max-seq-length 256 \
		--output data/models/onnx/all-MiniLM-L6-v2

run-trec-2021-dense-onnx:
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-dense-experiment \
		--config configs/experiments/trec_2021/development_dense_all_minilm_l6_v2_onnx.json

check-trec-2021-dense-onnx-parity:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli check-trec-run-parity \
		--baseline outputs/trec_2021_development_dense_all_minilm_l6_v2.run \
		--candidate outputs/trec_2021_development_dense_all_minilm_l6_v2_onnx.run \
		--depth 100 \
		--output outputs/trec_2021_dense_onnx_parity.json

convert-trec-2021-dense-mmap:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli convert-dense-index-mmap \
		--config configs/experiments/trec_2021/development_dense_all_minilm_l6_v2.json \
		--output data/indexes/trec_2021_dense_all_minilm_l6_v2_title_summary_conditions.mmap

run-trec-2021-dense-mmap-int8:
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-dense-experiment \
		--config configs/experiments/trec_2021/development_dense_all_minilm_l6_v2_mmap_int8.json

compare-trec-2021-dense-optimization:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics fp32_npz=outputs/trec_2021_development_dense_all_minilm_l6_v2_metrics.json \
		--metrics mmap_int8=outputs/trec_2021_development_dense_all_minilm_l6_v2_mmap_int8_metrics.json \
		--output outputs/trec_2021_development_dense_optimization_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

run-trec-2021-dense-biomedical:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-dense-experiment \
		--config configs/experiments/trec_2021/development_dense_medembed_small_eligibility_snapshot.json

compare-trec-2021-dense-ablation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics all_minilm=outputs/trec_2021_development_dense_all_minilm_l6_v2_metrics.json \
		--metrics medembed_eligibility=outputs/trec_2021_development_dense_medembed_small_eligibility_snapshot_metrics.json \
		--output outputs/trec_2021_development_dense_ablation_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

compare-trec-2021-lexical-dense:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics frozen_lexical=outputs/trec_2021_development_fielded_bm25_condition_title_v1_metrics.json \
		--metrics dense_all_minilm=outputs/trec_2021_development_dense_all_minilm_l6_v2_metrics.json \
		--output outputs/trec_2021_development_lexical_dense_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

run-trec-2021-hybrid:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-rrf-experiment \
		--config configs/experiments/trec_2021/development_hybrid_rrf_lexical_minilm.json

run-trec-2021-hybrid-sqlite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli run-rrf-experiment \
		--config configs/experiments/trec_2021/development_hybrid_rrf_sqlite_minilm.json

download-trec-2021-cross-encoder:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli download-cross-encoder \
		--config configs/experiments/trec_2021/development_cross_encoder_minilm_l6_v2.json

run-trec-2021-cross-encoder: run-trec-2021-hybrid-sqlite
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-cross-encoder-experiment \
		--config configs/experiments/trec_2021/development_cross_encoder_minilm_l6_v2.json

benchmark-trec-2021-cross-encoder-headroom:
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli benchmark-cross-encoder-headroom \
		--config configs/experiments/trec_2021/development_cross_encoder_minilm_l6_v2.json

download-trec-2021-cross-encoder-optimization:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli download-cross-encoder \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_fp32_256_core.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli download-cross-encoder \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_int8_256_core.json

run-trec-2021-cross-encoder-optimization: run-trec-2021-hybrid-sqlite
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-cross-encoder-experiment \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_fp32_256_core.json
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-cross-encoder-experiment \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_int8_256_core.json
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-cross-encoder-experiment \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_int8_128_core.json
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-cross-encoder-experiment \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_int8_128_short.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-cross-encoder-experiments \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_optimization.json

download-trec-2021-cross-encoder-small-depths: download-trec-2021-cross-encoder-optimization

run-trec-2021-cross-encoder-small-depths: run-trec-2021-hybrid-sqlite
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-cross-encoder-experiment \
		--config configs/experiments/trec_2021/development_cross_encoder_depth10_fp32_256_core.json
	PYTHONPATH=$(PYTHONPATH) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m clinical_trial_matching.cli run-cross-encoder-experiment \
		--config configs/experiments/trec_2021/development_cross_encoder_int8_256_core_small_depths.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-cross-encoder-experiments \
		--config configs/experiments/trec_2021/development_cross_encoder_small_depths_comparison.json

compare-trec-2021-hybrid-backends:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics bm25_hybrid=outputs/trec_2021_development_hybrid_rrf_lexical_minilm_metrics.json \
		--metrics sqlite_hybrid=outputs/trec_2021_development_hybrid_rrf_sqlite_minilm_metrics.json \
		--output outputs/trec_2021_development_hybrid_backend_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

compare-trec-2021-retrievers:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-metrics \
		--metrics frozen_lexical=outputs/trec_2021_development_fielded_bm25_condition_title_v1_metrics.json \
		--metrics selected_dense=outputs/trec_2021_development_dense_all_minilm_l6_v2_metrics.json \
		--metrics hybrid_rrf=outputs/trec_2021_development_hybrid_rrf_lexical_minilm_metrics.json \
		--output outputs/trec_2021_development_retriever_comparison.md \
		--view eligible_only \
		--view excluded_or_eligible

benchmark-trec-2021-serving:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli benchmark-serving \
		--config configs/benchmarks/trec_2021_local_serving.json

benchmark-trec-2021-serving-sentence-transformers:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli benchmark-serving \
		--config configs/benchmarks/trec_2021_local_serving_sentence_transformers.json

benchmark-trec-2021-serving-mmap-int8:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli benchmark-serving \
		--config configs/benchmarks/trec_2021_local_serving_mmap_int8.json

benchmark-trec-2021-serving-mmap:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli benchmark-serving \
		--config configs/benchmarks/trec_2021_local_serving_mmap.json

assess-trec-2021-serving-budget:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli assess-serving-budget \
		--report outputs/trec_2021_local_serving_benchmark.json \
		--budget configs/benchmarks/deployment_budget_1gib.json \
		--output outputs/trec_2021_local_serving_budget_assessment.json

benchmark-trec-2021-lexical-backends:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli benchmark-lexical-backend \
		--serving-config configs/benchmarks/trec_2021_local_serving.json \
		--backend fielded-bm25 \
		--experiment-config configs/experiments/trec_2021/fielded_bm25_condition_title_v1.json \
		--output outputs/trec_2021_fielded_bm25_resource_benchmark.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli benchmark-lexical-backend \
		--serving-config configs/benchmarks/trec_2021_local_serving.json \
		--backend sqlite-fts5 \
		--experiment-config configs/experiments/trec_2021/development_sqlite_fts5_condition_title_v1.json \
		--output outputs/trec_2021_sqlite_fts5_resource_benchmark.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m clinical_trial_matching.cli compare-lexical-backends \
		--baseline outputs/trec_2021_fielded_bm25_resource_benchmark.json \
		--candidate outputs/trec_2021_sqlite_fts5_resource_benchmark.json \
		--output outputs/trec_2021_lexical_backend_resource_comparison.md

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
