from __future__ import annotations

import argparse
from pathlib import Path

from clinical_trial_matching.benchmarking.lexical import (
    benchmark_lexical_backend,
    compare_lexical_backend_reports,
    write_lexical_backend_comparison,
)
from clinical_trial_matching.benchmarking.serving import (
    assess_serving_budget,
    load_serving_benchmark,
    run_serving_benchmark,
)
from clinical_trial_matching.evaluation.comparison import (
    build_metrics_comparison,
    infer_comparison_format,
    parse_metrics_spec,
    write_metrics_comparison,
)
from clinical_trial_matching.evaluation.experiments import (
    load_bm25_experiment,
    load_dense_experiment,
    load_rrf_experiment,
    load_sqlite_fts_experiment,
)
from clinical_trial_matching.evaluation.metrics import summarize_run
from clinical_trial_matching.evaluation.parity import trec_run_parity_report
from clinical_trial_matching.evaluation.regression import (
    DEFAULT_THRESHOLDS,
    run_bm25_regression_check,
)
from clinical_trial_matching.evaluation.splits import (
    build_trec_topic_split,
    topic_split_report,
)
from clinical_trial_matching.evaluation.trec import (
    bm25_retriever_parameters,
    bm25_trec_evaluation_report,
    bm25_trec_topic_diagnostics,
    build_bm25_trec_run,
    build_dense_trec_run,
    build_sqlite_fts_trec_run,
    trec_evaluation_report,
    trec_topic_diagnostics,
    write_trec_run,
)
from clinical_trial_matching.ingestion.clinicaltrials import (
    CTGOV_API_BASE_URL,
    fetch_ctgov_studies,
    fetch_ctgov_studies_by_ids,
    parse_studies_json,
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.ingestion.manifest import (
    build_source_manifest,
    manifest_to_json_record,
    sha256_file,
)
from clinical_trial_matching.ingestion.trec import (
    parse_qrels,
    parse_topics_xml,
    qrel_from_json_record,
    qrel_to_json_record,
    qrels_to_mapping,
    topic_from_json_record,
    topic_to_json_record,
    validate_topics_and_qrels,
)
from clinical_trial_matching.io import iter_jsonl, read_jsonl, write_json, write_jsonl
from clinical_trial_matching.models import Qrel
from clinical_trial_matching.retrieval.bm25 import (
    BM25Retriever,
    load_or_build_bm25_retriever,
    save_bm25_index,
    search_trials,
)
from clinical_trial_matching.retrieval.dense import (
    DENSE_RETRIEVER_NAME,
    DENSE_SCORE_TIE_DECIMALS,
    ENCODER_BACKENDS,
    TEXT_REPRESENTATIONS,
    SentenceTransformerEncoder,
    build_dense_index,
    export_onnx_encoder,
    load_dense_index,
    load_or_build_dense_retriever,
    save_dense_index,
)
from clinical_trial_matching.retrieval.hybrid import (
    RRF_RETRIEVER_NAME,
    RankedRun,
    read_trec_rankings,
    reciprocal_rank_fusion,
)
from clinical_trial_matching.retrieval.sqlite_fts import (
    SQLITE_FTS_RETRIEVER_NAME,
    build_sqlite_fts_index,
    load_or_build_sqlite_fts_retriever,
)
from clinical_trial_matching.trial_store import build_trial_store
from clinical_trial_matching.validation.trials import summarize_trial_corpus


def main() -> None:
    parser = argparse.ArgumentParser(prog="ctmatch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-sample", help="Normalize synthetic fixture trials.")
    ingest.add_argument("--trials", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)

    ctgov = subparsers.add_parser(
        "ingest-ctgov-studies", help="Normalize ClinicalTrials.gov v2 JSON studies to JSONL."
    )
    ctgov.add_argument("--input", type=Path, required=True)
    ctgov.add_argument("--output", type=Path, required=True)

    ctgov_live = subparsers.add_parser(
        "download-ctgov-studies",
        help="Fetch a small ClinicalTrials.gov v2 query, write raw JSON, manifest, and JSONL.",
    )
    ctgov_live.add_argument("--query", required=True)
    ctgov_live.add_argument("--status", default="RECRUITING")
    ctgov_live.add_argument("--page-size", type=int, default=25)
    ctgov_live.add_argument("--raw-output", type=Path, required=True)
    ctgov_live.add_argument("--manifest-output", type=Path, required=True)
    ctgov_live.add_argument("--processed-output", type=Path, required=True)
    ctgov_live.add_argument("--base-url", default=CTGOV_API_BASE_URL)
    ctgov_live.add_argument("--timeout-seconds", type=float, default=30.0)

    trec_corpus = subparsers.add_parser(
        "build-trec-trial-corpus",
        help=(
            "Extract NCT IDs from qrels, fetch ClinicalTrials.gov records, and normalize a trial "
            "corpus."
        ),
    )
    trec_corpus.add_argument("--qrels", type=Path, required=True)
    trec_corpus.add_argument("--raw-output", type=Path, required=True)
    trec_corpus.add_argument("--processed-output", type=Path, required=True)
    trec_corpus.add_argument("--manifest-output", type=Path, required=True)
    trec_corpus.add_argument("--report-output", type=Path, required=True)
    trec_corpus.add_argument("--dataset", default="trec_clinical_trials")
    trec_corpus.add_argument("--year", type=int, required=True)
    trec_corpus.add_argument("--batch-size", type=int, default=100)
    trec_corpus.add_argument("--limit", type=int)
    trec_corpus.add_argument("--delay-seconds", type=float, default=0.0)
    trec_corpus.add_argument("--max-retries", type=int, default=5)
    trec_corpus.add_argument("--retry-initial-delay-seconds", type=float, default=2.0)
    trec_corpus.add_argument("--retry-max-delay-seconds", type=float, default=60.0)
    trec_corpus.add_argument("--base-url", default=CTGOV_API_BASE_URL)
    trec_corpus.add_argument("--timeout-seconds", type=float, default=30.0)

    evaluate = subparsers.add_parser("evaluate-baseline", help="Evaluate BM25 on fixture data.")
    evaluate.add_argument("--trials", type=Path, required=True)
    evaluate.add_argument("--topics", type=Path, required=True)
    evaluate.add_argument("--qrels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--top-k", type=int, default=100)

    topic_split = subparsers.add_parser(
        "split-trec-topics",
        help="Create deterministic development and holdout topic/qrels partitions.",
    )
    topic_split.add_argument("--topics", type=Path, required=True)
    topic_split.add_argument("--qrels", type=Path, required=True)
    topic_split.add_argument("--development-topics-output", type=Path, required=True)
    topic_split.add_argument("--development-qrels-output", type=Path, required=True)
    topic_split.add_argument("--holdout-topics-output", type=Path, required=True)
    topic_split.add_argument("--holdout-qrels-output", type=Path, required=True)
    topic_split.add_argument("--report-output", type=Path, required=True)
    topic_split.add_argument("--seed", default="ctmatch-trec-2021-v1")
    topic_split.add_argument("--holdout-fraction", type=float, default=0.2)

    trec_bm25 = subparsers.add_parser(
        "evaluate-trec-bm25",
        help=(
            "Write a TREC-format BM25 run file and metrics report from normalized benchmark "
            "files."
        ),
    )
    trec_bm25.add_argument("--trials", type=Path, required=True)
    trec_bm25.add_argument("--topics", type=Path, required=True)
    trec_bm25.add_argument("--qrels", type=Path, required=True)
    trec_bm25.add_argument("--run-output", type=Path, required=True)
    trec_bm25.add_argument("--metrics-output", type=Path, required=True)
    trec_bm25.add_argument("--diagnostics-output", type=Path)
    trec_bm25.add_argument("--index-path", type=Path)
    trec_bm25.add_argument("--rebuild-index", action="store_true")
    trec_bm25.add_argument("--run-name", default="bm25")
    trec_bm25.add_argument("--top-k", type=int, default=100)
    trec_bm25.add_argument("--retriever", choices=["bm25", "fielded-bm25"], default="fielded-bm25")
    trec_bm25.add_argument(
        "--field-weight",
        action="append",
        default=[],
        help="Optional field=weight override for fielded-bm25. May be provided multiple times.",
    )

    regression = subparsers.add_parser(
        "check-retrieval-regression",
        help="Run tiny BM25 retrieval-quality regression checks and fail on metric regressions.",
    )
    regression.add_argument("--trials", type=Path, required=True)
    regression.add_argument("--topics", type=Path, required=True)
    regression.add_argument("--qrels", type=Path, required=True)
    regression.add_argument("--output", type=Path, required=True)
    regression.add_argument("--top-k", type=int, default=100)
    regression.add_argument(
        "--min-recall-at-100",
        type=float,
        default=DEFAULT_THRESHOLDS["recall_at_100"],
    )
    regression.add_argument("--min-mrr", type=float, default=DEFAULT_THRESHOLDS["mrr"])
    regression.add_argument(
        "--min-ndcg-at-10",
        type=float,
        default=DEFAULT_THRESHOLDS["ndcg_at_10"],
    )

    trial_report = subparsers.add_parser(
        "report-trial-corpus", help="Summarize a normalized trial JSONL corpus."
    )
    trial_report.add_argument("--trials", type=Path, required=True)
    trial_report.add_argument("--output", type=Path)
    trial_report.add_argument("--sample-size", type=int, default=5)
    trial_report.add_argument("--top-n", type=int, default=10)

    trial_search = subparsers.add_parser(
        "search-trials-bm25", help="Search a normalized trial JSONL corpus with BM25."
    )
    trial_search.add_argument("--trials", type=Path, required=True)
    trial_search.add_argument("--query", required=True)
    trial_search.add_argument("--output", type=Path)
    trial_search.add_argument("--index-path", type=Path)
    trial_search.add_argument("--rebuild-index", action="store_true")
    trial_search.add_argument("--top-k", type=int, default=10)
    trial_search.add_argument("--snippet-chars", type=int, default=240)
    trial_search.add_argument(
        "--retriever",
        choices=["bm25", "fielded-bm25"],
        default="fielded-bm25",
    )
    trial_search.add_argument(
        "--field-weight",
        action="append",
        default=[],
        help="Optional field=weight override for fielded-bm25. May be provided multiple times.",
    )

    bm25_index = subparsers.add_parser(
        "build-bm25-index",
        help="Build and persist a reusable BM25 index for a normalized trial corpus.",
    )
    bm25_index.add_argument("--trials", type=Path, required=True)
    bm25_index.add_argument("--output", type=Path, required=True)
    bm25_index.add_argument("--retriever", choices=["bm25", "fielded-bm25"], default="fielded-bm25")
    bm25_index.add_argument(
        "--field-weight",
        action="append",
        default=[],
        help="Optional field=weight override for fielded-bm25. May be provided multiple times.",
    )

    bm25_experiment = subparsers.add_parser(
        "run-bm25-experiment",
        help="Run a reproducible BM25 benchmark from a versioned experiment config.",
    )
    bm25_experiment.add_argument("--config", type=Path, required=True)
    bm25_experiment.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the configured index even when a compatible cached index exists.",
    )

    sqlite_fts_experiment = subparsers.add_parser(
        "run-sqlite-fts-experiment",
        help="Run a reproducible SQLite FTS5 benchmark from a versioned experiment config.",
    )
    sqlite_fts_experiment.add_argument("--config", type=Path, required=True)
    sqlite_fts_experiment.add_argument("--rebuild-index", action="store_true")

    sqlite_fts_index = subparsers.add_parser(
        "build-sqlite-fts-index",
        help="Build a disk-backed SQLite FTS5 index for a normalized trial corpus.",
    )
    sqlite_fts_index.add_argument("--trials", type=Path, required=True)
    sqlite_fts_index.add_argument("--output", type=Path, required=True)
    sqlite_fts_index.add_argument(
        "--field-weight",
        action="append",
        default=[],
        help="Required field=weight values for the five FTS5 text fields.",
    )

    trial_store = subparsers.add_parser(
        "build-trial-store",
        help="Stream normalized trial JSONL into a disk-backed SQLite metadata store.",
    )
    trial_store.add_argument("--trials", type=Path, required=True)
    trial_store.add_argument("--output", type=Path, required=True)

    dense_index = subparsers.add_parser(
        "build-dense-index",
        help="Embed a normalized trial corpus and persist a reusable NumPy dense index.",
    )
    _add_dense_model_arguments(dense_index)
    dense_index.add_argument("--trials", type=Path, required=True)
    dense_index.add_argument("--output", type=Path, required=True)

    dense_index_conversion = subparsers.add_parser(
        "convert-dense-index-mmap",
        help="Convert a configured NumPy dense index to a memory-mapped directory artifact.",
    )
    dense_index_conversion.add_argument("--config", type=Path, required=True)
    dense_index_conversion.add_argument("--output", type=Path, required=True)

    onnx_export = subparsers.add_parser(
        "export-onnx-encoder",
        help="Export a local sentence-transformer query encoder for ONNX Runtime.",
    )
    onnx_export.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    onnx_export.add_argument("--output", type=Path, required=True)
    onnx_export.add_argument("--device", default="cpu")
    onnx_export.add_argument("--max-seq-length", type=int, default=256)

    dense_evaluation = subparsers.add_parser(
        "evaluate-trec-dense",
        help="Evaluate a sentence-transformer bi-encoder on normalized TREC benchmark files.",
    )
    _add_dense_model_arguments(dense_evaluation)
    dense_evaluation.add_argument("--trials", type=Path, required=True)
    dense_evaluation.add_argument("--topics", type=Path, required=True)
    dense_evaluation.add_argument("--qrels", type=Path, required=True)
    dense_evaluation.add_argument("--index-path", type=Path, required=True)
    dense_evaluation.add_argument("--run-output", type=Path, required=True)
    dense_evaluation.add_argument("--metrics-output", type=Path, required=True)
    dense_evaluation.add_argument("--diagnostics-output", type=Path)
    dense_evaluation.add_argument("--run-name", default="dense_bi_encoder")
    dense_evaluation.add_argument("--top-k", type=int, default=100)
    dense_evaluation.add_argument("--rebuild-index", action="store_true")
    dense_evaluation.add_argument("--dynamic-quantization", action="store_true")
    dense_evaluation.add_argument(
        "--encoder-backend",
        choices=sorted(ENCODER_BACKENDS),
        default="sentence-transformers",
    )
    dense_evaluation.add_argument("--onnx-model-path", type=Path)

    dense_experiment = subparsers.add_parser(
        "run-dense-experiment",
        help="Run a reproducible dense benchmark from a versioned experiment config.",
    )
    dense_experiment.add_argument("--config", type=Path, required=True)
    dense_experiment.add_argument("--rebuild-index", action="store_true")

    run_parity = subparsers.add_parser(
        "check-trec-run-parity",
        help="Require exact per-topic NCT ordering between two TREC run files.",
    )
    run_parity.add_argument("--baseline", type=Path, required=True)
    run_parity.add_argument("--candidate", type=Path, required=True)
    run_parity.add_argument("--depth", type=int, default=100)
    run_parity.add_argument("--output", type=Path, required=True)

    rrf_experiment = subparsers.add_parser(
        "run-rrf-experiment",
        help="Fuse existing TREC runs with reciprocal-rank fusion and evaluate the result.",
    )
    rrf_experiment.add_argument("--config", type=Path, required=True)

    compare_metrics_parser = subparsers.add_parser(
        "compare-metrics",
        help="Compare multiple retrieval metrics JSON reports in a compact table.",
    )
    compare_metrics_parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        help="Metrics JSON path, optionally label=path. May be provided multiple times.",
    )
    compare_metrics_parser.add_argument("--output", type=Path, required=True)
    compare_metrics_parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        help="Output format. Defaults from output suffix.",
    )
    compare_metrics_parser.add_argument(
        "--view",
        action="append",
        default=[],
        help=(
            "Optional metric view to include, such as eligible_only. May be provided multiple "
            "times."
        ),
    )

    serving_benchmark = subparsers.add_parser(
        "benchmark-serving",
        help="Measure local serving startup, latency, throughput, memory, and artifact sizes.",
    )
    serving_benchmark.add_argument("--config", type=Path, required=True)

    serving_budget = subparsers.add_parser(
        "assess-serving-budget",
        help="Check a serving benchmark report against a versioned deployment budget.",
    )
    serving_budget.add_argument("--report", type=Path, required=True)
    serving_budget.add_argument("--budget", type=Path, required=True)
    serving_budget.add_argument("--output", type=Path, required=True)

    lexical_benchmark = subparsers.add_parser(
        "benchmark-lexical-backend",
        help="Measure cold start, memory, latency, and size for one lexical backend.",
    )
    lexical_benchmark.add_argument("--serving-config", type=Path, required=True)
    lexical_benchmark.add_argument(
        "--backend",
        choices=["fielded-bm25", "sqlite-fts5"],
        required=True,
    )
    lexical_benchmark.add_argument("--experiment-config", type=Path, required=True)
    lexical_benchmark.add_argument("--output", type=Path, required=True)

    lexical_comparison = subparsers.add_parser(
        "compare-lexical-backends",
        help="Compare fielded BM25 and SQLite FTS5 resource benchmark reports.",
    )
    lexical_comparison.add_argument("--baseline", type=Path, required=True)
    lexical_comparison.add_argument("--candidate", type=Path, required=True)
    lexical_comparison.add_argument("--output", type=Path, required=True)

    trec_topics = subparsers.add_parser(
        "ingest-trec-topics", help="Normalize TREC Clinical Trials topics XML to JSONL."
    )
    trec_topics.add_argument("--year", type=int, required=True)
    trec_topics.add_argument("--input", type=Path, required=True)
    trec_topics.add_argument("--output", type=Path, required=True)

    trec_qrels = subparsers.add_parser(
        "ingest-trec-qrels", help="Normalize TREC Clinical Trials qrels to JSONL."
    )
    trec_qrels.add_argument("--year", type=int, required=True)
    trec_qrels.add_argument("--input", type=Path, required=True)
    trec_qrels.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate-trec", help="Validate normalized TREC topics and qrels JSONL files."
    )
    validate.add_argument("--topics", type=Path, required=True)
    validate.add_argument("--qrels", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    manifest = subparsers.add_parser(
        "write-manifest", help="Write a reproducibility manifest for a local source file."
    )
    manifest.add_argument("--name", required=True)
    manifest.add_argument("--source-url", required=True)
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--dataset", default="")
    manifest.add_argument("--year", type=int)
    manifest.add_argument("--parser", default="")
    manifest.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Optional key=value metadata. May be provided multiple times.",
    )

    args = parser.parse_args()

    if args.command == "ingest-sample":
        ingest_sample(args.trials, args.output)
    elif args.command == "ingest-ctgov-studies":
        ingest_ctgov_studies(args.input, args.output)
    elif args.command == "download-ctgov-studies":
        download_ctgov_studies(
            query=args.query,
            status=args.status,
            page_size=args.page_size,
            raw_output=args.raw_output,
            manifest_output=args.manifest_output,
            processed_output=args.processed_output,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "build-trec-trial-corpus":
        build_trec_trial_corpus(
            qrels_path=args.qrels,
            raw_output=args.raw_output,
            processed_output=args.processed_output,
            manifest_output=args.manifest_output,
            report_output=args.report_output,
            dataset=args.dataset,
            year=args.year,
            batch_size=args.batch_size,
            limit=args.limit,
            delay_seconds=args.delay_seconds,
            max_retries=args.max_retries,
            retry_initial_delay_seconds=args.retry_initial_delay_seconds,
            retry_max_delay_seconds=args.retry_max_delay_seconds,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "evaluate-baseline":
        evaluate_baseline(args.trials, args.topics, args.qrels, args.output, args.top_k)
    elif args.command == "split-trec-topics":
        write_trec_topic_split(
            topics_path=args.topics,
            qrels_path=args.qrels,
            development_topics_output=args.development_topics_output,
            development_qrels_output=args.development_qrels_output,
            holdout_topics_output=args.holdout_topics_output,
            holdout_qrels_output=args.holdout_qrels_output,
            report_output=args.report_output,
            seed=args.seed,
            holdout_fraction=args.holdout_fraction,
        )
    elif args.command == "evaluate-trec-bm25":
        evaluate_trec_bm25(
            trials_path=args.trials,
            topics_path=args.topics,
            qrels_path=args.qrels,
            run_output_path=args.run_output,
            metrics_output_path=args.metrics_output,
            diagnostics_output_path=args.diagnostics_output,
            run_name=args.run_name,
            top_k=args.top_k,
            retriever_name=args.retriever,
            field_weights=parse_field_weights(args.field_weight),
            index_path=args.index_path,
            rebuild_index=args.rebuild_index,
        )
    elif args.command == "check-retrieval-regression":
        check_retrieval_regression(
            trials_path=args.trials,
            topics_path=args.topics,
            qrels_path=args.qrels,
            output_path=args.output,
            top_k=args.top_k,
            thresholds={
                "recall_at_100": args.min_recall_at_100,
                "mrr": args.min_mrr,
                "ndcg_at_10": args.min_ndcg_at_10,
            },
        )
    elif args.command == "report-trial-corpus":
        report_trial_corpus(args.trials, args.output, args.sample_size, args.top_n)
    elif args.command == "search-trials-bm25":
        search_trials_bm25(
            args.trials,
            args.query,
            args.output,
            args.top_k,
            args.snippet_chars,
            args.retriever,
            parse_field_weights(args.field_weight),
            args.index_path,
            args.rebuild_index,
        )
    elif args.command == "build-bm25-index":
        build_bm25_index(
            trials_path=args.trials,
            output_path=args.output,
            retriever_name=args.retriever,
            field_weights=parse_field_weights(args.field_weight),
        )
    elif args.command == "run-bm25-experiment":
        run_bm25_experiment(args.config, rebuild_index=args.rebuild_index)
    elif args.command == "run-sqlite-fts-experiment":
        run_sqlite_fts_experiment(args.config, rebuild_index=args.rebuild_index)
    elif args.command == "build-sqlite-fts-index":
        build_sqlite_fts_index_command(
            trials_path=args.trials,
            output_path=args.output,
            field_weights=parse_field_weights(args.field_weight),
        )
    elif args.command == "build-trial-store":
        build_trial_store_command(args.trials, args.output)
    elif args.command == "build-dense-index":
        build_dense_index_command(
            trials_path=args.trials,
            output_path=args.output,
            model_name=args.model_name,
            text_representation=args.text_representation,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )
    elif args.command == "convert-dense-index-mmap":
        convert_dense_index_mmap(args.config, args.output)
    elif args.command == "export-onnx-encoder":
        export_onnx_encoder_command(
            model_name=args.model_name,
            output_path=args.output,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )
    elif args.command == "evaluate-trec-dense":
        evaluate_trec_dense(
            trials_path=args.trials,
            topics_path=args.topics,
            qrels_path=args.qrels,
            index_path=args.index_path,
            run_output_path=args.run_output,
            metrics_output_path=args.metrics_output,
            diagnostics_output_path=args.diagnostics_output,
            run_name=args.run_name,
            top_k=args.top_k,
            model_name=args.model_name,
            text_representation=args.text_representation,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.max_seq_length,
            rebuild_index=args.rebuild_index,
            dynamic_quantization=args.dynamic_quantization,
            encoder_backend=args.encoder_backend,
            onnx_model_path=args.onnx_model_path,
        )
    elif args.command == "run-dense-experiment":
        run_dense_experiment(args.config, rebuild_index=args.rebuild_index)
    elif args.command == "check-trec-run-parity":
        check_trec_run_parity(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            depth=args.depth,
            output_path=args.output,
        )
    elif args.command == "run-rrf-experiment":
        run_rrf_experiment(args.config)
    elif args.command == "compare-metrics":
        compare_metrics(
            metrics_specs=args.metrics,
            output_path=args.output,
            output_format=args.format,
            views=args.view,
        )
    elif args.command == "benchmark-serving":
        benchmark_serving(args.config)
    elif args.command == "assess-serving-budget":
        assessment = assess_serving_budget(args.report, args.budget)
        write_json(args.output, assessment)
        print(f"Wrote serving budget assessment to {args.output}")
    elif args.command == "benchmark-lexical-backend":
        benchmark_lexical_backend_command(
            serving_config_path=args.serving_config,
            backend=args.backend,
            experiment_config_path=args.experiment_config,
            output_path=args.output,
        )
    elif args.command == "compare-lexical-backends":
        compare_lexical_backends(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            output_path=args.output,
        )
    elif args.command == "ingest-trec-topics":
        ingest_trec_topics(args.year, args.input, args.output)
    elif args.command == "ingest-trec-qrels":
        ingest_trec_qrels(args.year, args.input, args.output)
    elif args.command == "validate-trec":
        validate_trec(args.topics, args.qrels, args.output)
    elif args.command == "write-manifest":
        write_manifest(
            name=args.name,
            source_url=args.source_url,
            input_path=args.input,
            output_path=args.output,
            dataset=args.dataset,
            year=args.year,
            parser=args.parser,
            metadata_items=args.metadata,
        )


def ingest_sample(trials_path: Path, output_path: Path) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    write_jsonl(output_path, (trial_to_flat_record(trial) for trial in trials))
    print(f"Wrote {len(trials)} normalized trials to {output_path}")


def ingest_ctgov_studies(input_path: Path, output_path: Path) -> None:
    trials = parse_studies_json(input_path)
    write_jsonl(output_path, (trial_to_flat_record(trial) for trial in trials))
    print(f"Wrote {len(trials)} normalized ClinicalTrials.gov studies to {output_path}")


def download_ctgov_studies(
    *,
    query: str,
    status: str,
    page_size: int,
    raw_output: Path,
    manifest_output: Path,
    processed_output: Path,
    base_url: str,
    timeout_seconds: float,
) -> None:
    result = fetch_ctgov_studies(
        query=query,
        status=status,
        page_size=page_size,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    write_json(raw_output, result.payload)

    trials = parse_studies_json(raw_output)
    write_jsonl(processed_output, (trial_to_flat_record(trial) for trial in trials))

    manifest = build_source_manifest(
        name=f"clinicaltrials_gov_{slugify(query)}_{slugify(status)}_{page_size}",
        source_url=result.request_url,
        input_path=raw_output,
        dataset="clinicaltrials_gov",
        parser="clinicaltrials_gov_v2_studies",
        metadata={
            "query": query,
            "status": status,
            "page_size": str(page_size),
            "study_count": str(result.study_count),
            "total_count": str(result.total_count) if result.total_count is not None else "",
            "next_page_token": result.next_page_token,
        },
    )
    write_json(manifest_output, manifest_to_json_record(manifest))

    print(f"Wrote raw ClinicalTrials.gov response to {raw_output}")
    print(f"Wrote {len(trials)} normalized ClinicalTrials.gov studies to {processed_output}")
    print(f"Wrote source manifest to {manifest_output}")


def build_trec_trial_corpus(
    *,
    qrels_path: Path,
    raw_output: Path,
    processed_output: Path,
    manifest_output: Path,
    report_output: Path,
    dataset: str,
    year: int,
    batch_size: int,
    limit: int | None,
    delay_seconds: float,
    max_retries: int,
    retry_initial_delay_seconds: float,
    retry_max_delay_seconds: float,
    base_url: str,
    timeout_seconds: float,
) -> None:
    qrels = read_qrels_records(qrels_path)
    nct_ids = unique_nct_ids_from_qrels(qrels)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be at least 1 when provided")
        nct_ids = nct_ids[:limit]
    result = fetch_ctgov_studies_by_ids(
        nct_ids=nct_ids,
        batch_size=batch_size,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        delay_seconds=delay_seconds,
        max_retries=max_retries,
        retry_initial_delay_seconds=retry_initial_delay_seconds,
        retry_max_delay_seconds=retry_max_delay_seconds,
    )
    write_json(raw_output, result.payload)

    trials = parse_studies_json(raw_output)
    write_jsonl(processed_output, (trial_to_flat_record(trial) for trial in trials))

    manifest = build_source_manifest(
        name=f"{dataset}_{year}_clinicaltrials_corpus",
        source_url=f"{base_url.rstrip('/')}/studies",
        input_path=raw_output,
        dataset=dataset,
        year=year,
        parser="clinicaltrials_gov_v2_id_corpus",
        metadata={
            "qrels_path": str(qrels_path),
            "requested_nct_ids": str(len(result.requested_nct_ids)),
            "found_nct_ids": str(len(result.found_nct_ids)),
            "missing_nct_ids": str(len(result.missing_nct_ids)),
            "batch_size": str(batch_size),
            "limit": str(limit or ""),
            "request_count": str(len(result.request_urls)),
            "max_retries": str(max_retries),
            "retry_initial_delay_seconds": str(retry_initial_delay_seconds),
            "retry_max_delay_seconds": str(retry_max_delay_seconds),
        },
    )
    write_json(manifest_output, manifest_to_json_record(manifest))

    report = {
        "dataset": dataset,
        "year": year,
        "qrels": len(qrels),
        "requested_nct_ids": len(result.requested_nct_ids),
        "found_nct_ids": len(result.found_nct_ids),
        "missing_nct_ids": len(result.missing_nct_ids),
        "missing_nct_ids_sample": list(result.missing_nct_ids[:20]),
        "raw_output": str(raw_output),
        "processed_output": str(processed_output),
        "manifest_output": str(manifest_output),
        "request_count": len(result.request_urls),
    }
    write_json(report_output, report)

    print(f"Wrote raw ClinicalTrials.gov corpus to {raw_output}")
    print(f"Wrote {len(trials)} normalized benchmark trials to {processed_output}")
    print(f"Wrote source manifest to {manifest_output}")
    print(f"Wrote corpus build report to {report_output}")


def evaluate_baseline(
    trials_path: Path,
    topics_path: Path,
    qrels_path: Path,
    output_path: Path,
    top_k: int,
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
    qrels = read_qrels(qrels_path)

    retriever = BM25Retriever(trials)
    run = {
        topic.topic_id: [result.nct_id for result in retriever.search(topic.text, top_k=top_k)]
        for topic in topics
    }
    metrics = summarize_run(run, qrels)
    payload = {
        "run_name": "sample_bm25",
        "metrics": metrics,
        "topics": len(topics),
        "trials": len(trials),
    }
    write_json(output_path, payload)
    print(f"Wrote baseline metrics to {output_path}")


def evaluate_trec_bm25(
    *,
    trials_path: Path,
    topics_path: Path,
    qrels_path: Path,
    run_output_path: Path,
    metrics_output_path: Path,
    diagnostics_output_path: Path | None,
    run_name: str,
    top_k: int,
    retriever_name: str = "fielded-bm25",
    field_weights: dict[str, float] | None = None,
    index_path: Path | None = None,
    rebuild_index: bool = False,
    experiment_metadata: dict[str, str | int] | None = None,
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
    qrels = read_qrels_records(qrels_path)
    rows = build_bm25_trec_run(
        trials=trials,
        topics=topics,
        run_name=run_name,
        top_k=top_k,
        retriever_name=retriever_name,
        field_weights=field_weights,
        corpus_path=trials_path,
        index_path=index_path,
        rebuild_index=rebuild_index,
    )
    write_trec_run(run_output_path, rows)
    retriever_parameters = bm25_retriever_parameters(
        retriever_name,
        field_weights=field_weights,
    )
    report = bm25_trec_evaluation_report(
        rows=rows,
        qrels=qrels,
        run_name=run_name,
        top_k=top_k,
        topics_count=len(topics),
        trials_count=len(trials),
        retriever_name=retriever_name,
        retriever_parameters=retriever_parameters,
    )
    if experiment_metadata:
        report["experiment"] = experiment_metadata
    write_json(metrics_output_path, report)
    if diagnostics_output_path:
        diagnostics = bm25_trec_topic_diagnostics(
            rows=rows,
            qrels=qrels,
            topics=topics,
            run_name=run_name,
            top_k=top_k,
            retriever_name=retriever_name,
            retriever_parameters=retriever_parameters,
        )
        if experiment_metadata:
            diagnostics["experiment"] = experiment_metadata
        write_json(diagnostics_output_path, diagnostics)
    print(f"Wrote TREC run file to {run_output_path}")
    print(f"Wrote TREC BM25 metrics report to {metrics_output_path}")
    if diagnostics_output_path:
        print(f"Wrote TREC BM25 topic diagnostics to {diagnostics_output_path}")


def write_trec_topic_split(
    *,
    topics_path: Path,
    qrels_path: Path,
    development_topics_output: Path,
    development_qrels_output: Path,
    holdout_topics_output: Path,
    holdout_qrels_output: Path,
    report_output: Path,
    seed: str,
    holdout_fraction: float,
) -> None:
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
    qrels = read_qrels_records(qrels_path)
    split = build_trec_topic_split(
        topics,
        qrels,
        seed=seed,
        holdout_fraction=holdout_fraction,
    )

    write_jsonl(
        development_topics_output,
        (topic_to_json_record(topic) for topic in split.development_topics),
    )
    write_jsonl(
        development_qrels_output,
        (qrel_to_json_record(qrel) for qrel in split.development_qrels),
    )
    write_jsonl(
        holdout_topics_output,
        (topic_to_json_record(topic) for topic in split.holdout_topics),
    )
    write_jsonl(
        holdout_qrels_output,
        (qrel_to_json_record(qrel) for qrel in split.holdout_qrels),
    )
    report = topic_split_report(
        split,
        topics_source=_split_source_record(topics_path),
        qrels_source=_split_source_record(qrels_path),
    )
    write_json(report_output, report)
    print(
        f"Wrote {len(split.development_topics)} development topics and "
        f"{len(split.holdout_topics)} holdout topics"
    )
    print(f"Wrote TREC topic split report to {report_output}")


def _split_source_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def check_retrieval_regression(
    *,
    trials_path: Path,
    topics_path: Path,
    qrels_path: Path,
    output_path: Path,
    top_k: int,
    thresholds: dict[str, float],
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
    qrels = read_qrels_records(qrels_path)
    report = run_bm25_regression_check(
        trials=trials,
        topics=topics,
        qrels=qrels,
        thresholds=thresholds,
        top_k=top_k,
    )
    write_json(output_path, report)
    if report["failures"]:
        failures = ", ".join(
            f"{failure['metric']}={failure['observed']} < {failure['threshold']}"
            for failure in report["failures"]
        )
        raise SystemExit(f"Retrieval regression check failed: {failures}")
    print(f"Wrote passing retrieval regression report to {output_path}")


def report_trial_corpus(
    trials_path: Path,
    output_path: Path | None,
    sample_size: int,
    top_n: int,
) -> None:
    if sample_size < 0:
        raise ValueError("Sample size must be non-negative")
    if top_n < 1:
        raise ValueError("Top-N must be at least 1")
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    report = summarize_trial_corpus(trials, sample_size=sample_size, top_n=top_n)
    if output_path:
        write_json(output_path, report)
        print(f"Wrote trial corpus report to {output_path}")
    else:
        for key, value in report.items():
            print(f"{key}: {value}")


def search_trials_bm25(
    trials_path: Path,
    query: str,
    output_path: Path | None,
    top_k: int,
    snippet_chars: int,
    retriever_name: str,
    field_weights: dict[str, float],
    index_path: Path | None,
    rebuild_index: bool,
) -> None:
    if top_k < 1:
        raise ValueError("Top-K must be at least 1")
    if snippet_chars < 1:
        raise ValueError("Snippet chars must be at least 1")
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    retriever = load_or_build_bm25_retriever(
        trials=trials,
        retriever_name=retriever_name,
        field_weights=field_weights,
        corpus_path=trials_path,
        index_path=index_path,
        rebuild_index=rebuild_index,
    )
    payload = search_trials(
        trials,
        query=query,
        top_k=top_k,
        snippet_chars=snippet_chars,
        retriever_name=retriever_name,
        field_weights=field_weights,
        retriever=retriever,
    )
    if output_path:
        write_json(output_path, payload)
        print(f"Wrote BM25 search results to {output_path}")
    else:
        for result in payload["results"]:
            print(
                f"{result['rank']}. {result['nct_id']} "
                f"score={result['score']} status={result['status']} title={result['title']}"
            )
            print(f"   matched_terms={', '.join(result['matched_terms'])}")
            print(f"   snippet={result['snippet']}")


def build_bm25_index(
    *,
    trials_path: Path,
    output_path: Path,
    retriever_name: str,
    field_weights: dict[str, float],
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    record = save_bm25_index(
        output_path,
        trials,
        retriever_name=retriever_name,
        field_weights=field_weights,
        corpus_path=trials_path,
    )
    print(
        f"Wrote {record['index']['retriever']} index for "
        f"{record['corpus']['trials']} trials to {output_path}"
    )


def run_bm25_experiment(config_path: Path, *, rebuild_index: bool = False) -> None:
    experiment = load_bm25_experiment(config_path)
    print(f"Running BM25 experiment {experiment.name} from {experiment.config_label}")
    evaluate_trec_bm25(
        trials_path=experiment.trials_path,
        topics_path=experiment.topics_path,
        qrels_path=experiment.qrels_path,
        run_output_path=experiment.run_output_path,
        metrics_output_path=experiment.metrics_output_path,
        diagnostics_output_path=experiment.diagnostics_output_path,
        run_name=experiment.name,
        top_k=experiment.top_k,
        retriever_name=experiment.retriever,
        field_weights=experiment.field_weights,
        index_path=experiment.index_path,
        rebuild_index=rebuild_index,
        experiment_metadata=experiment.metadata(),
    )


def run_sqlite_fts_experiment(
    config_path: Path,
    *,
    rebuild_index: bool = False,
) -> None:
    experiment = load_sqlite_fts_experiment(config_path)
    print(
        f"Running SQLite FTS5 experiment {experiment.name} "
        f"from {experiment.config_label}"
    )
    trials = [trial_from_flat_record(row) for row in read_jsonl(experiment.trials_path)]
    topics = [topic_from_json_record(row) for row in read_jsonl(experiment.topics_path)]
    qrels = read_qrels_records(experiment.qrels_path)
    retriever = load_or_build_sqlite_fts_retriever(
        trials=trials,
        index_path=experiment.index_path,
        field_weights=experiment.field_weights,
        corpus_path=experiment.trials_path,
        rebuild_index=rebuild_index,
    )
    rows = build_sqlite_fts_trec_run(
        retriever=retriever,
        topics=topics,
        run_name=experiment.name,
        top_k=experiment.top_k,
    )
    write_trec_run(experiment.run_output_path, rows)
    parameters = {
        "field_weights": experiment.field_weights,
        "index_schema_version": retriever.metadata["schema_version"],
        "tokenizer": retriever.metadata["tokenizer"],
        "query_operator": retriever.metadata["query_operator"],
        "corpus_fingerprint": retriever.metadata["corpus"]["fingerprint"],
    }
    report = trec_evaluation_report(
        rows=rows,
        qrels=qrels,
        run_name=experiment.name,
        top_k=experiment.top_k,
        topics_count=len(topics),
        trials_count=len(trials),
        retriever_name=SQLITE_FTS_RETRIEVER_NAME,
        retriever_parameters=parameters,
    )
    report["experiment"] = experiment.metadata()
    write_json(experiment.metrics_output_path, report)
    diagnostics = trec_topic_diagnostics(
        rows=rows,
        qrels=qrels,
        topics=topics,
        run_name=experiment.name,
        top_k=experiment.top_k,
        retriever_name=SQLITE_FTS_RETRIEVER_NAME,
        retriever_parameters=parameters,
    )
    diagnostics["experiment"] = experiment.metadata()
    write_json(experiment.diagnostics_output_path, diagnostics)
    print(f"Wrote TREC SQLite FTS5 run file to {experiment.run_output_path}")
    print(f"Wrote TREC SQLite FTS5 metrics report to {experiment.metrics_output_path}")
    print(f"Wrote TREC SQLite FTS5 diagnostics to {experiment.diagnostics_output_path}")


def build_sqlite_fts_index_command(
    *,
    trials_path: Path,
    output_path: Path,
    field_weights: dict[str, float],
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    metadata = build_sqlite_fts_index(
        output_path,
        trials,
        field_weights=field_weights or None,
        corpus_path=trials_path,
    )
    print(
        f"Wrote SQLite FTS5 index for {metadata['corpus']['trials']} trials "
        f"to {output_path}"
    )


def build_trial_store_command(trials_path: Path, output_path: Path) -> None:
    metadata = build_trial_store(
        output_path,
        (trial_from_flat_record(row) for row in iter_jsonl(trials_path)),
        corpus_path=trials_path,
    )
    print(
        f"Wrote SQLite trial metadata store for {metadata['corpus']['trials']} trials "
        f"to {output_path}"
    )


def build_dense_index_command(
    *,
    trials_path: Path,
    output_path: Path,
    model_name: str,
    text_representation: str,
    batch_size: int,
    device: str,
    max_seq_length: int | None,
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    encoder = SentenceTransformerEncoder(
        model_name,
        device=device,
        max_seq_length=max_seq_length,
    )
    index = build_dense_index(
        trials,
        encoder=encoder,
        model_name=model_name,
        text_representation=text_representation,
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
    )
    save_dense_index(output_path, index)
    print(
        f"Wrote {index.metadata['embedding_dimension']}-dimensional dense index for "
        f"{index.metadata['trials']} trials to {output_path}"
    )


def export_onnx_encoder_command(
    *,
    model_name: str,
    output_path: Path,
    device: str,
    max_seq_length: int,
) -> None:
    metadata = export_onnx_encoder(
        model_name=model_name,
        output_path=output_path,
        device=device,
        max_seq_length=max_seq_length,
    )
    print(
        f"Wrote {metadata['embedding_dimension']}-dimensional ONNX encoder "
        f"to {output_path}"
    )


def evaluate_trec_dense(
    *,
    trials_path: Path,
    topics_path: Path,
    qrels_path: Path,
    index_path: Path,
    run_output_path: Path,
    metrics_output_path: Path,
    diagnostics_output_path: Path | None,
    run_name: str,
    top_k: int,
    model_name: str,
    text_representation: str,
    batch_size: int,
    device: str,
    max_seq_length: int | None,
    rebuild_index: bool = False,
    dynamic_quantization: bool = False,
    encoder_backend: str = "sentence-transformers",
    onnx_model_path: Path | None = None,
    experiment_metadata: dict[str, str | int | bool] | None = None,
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
    qrels = read_qrels_records(qrels_path)
    retriever = load_or_build_dense_retriever(
        trials=trials,
        model_name=model_name,
        text_representation=text_representation,
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
        index_path=index_path,
        rebuild_index=rebuild_index,
        dynamic_quantization=dynamic_quantization,
        encoder_backend=encoder_backend,
        onnx_model_path=onnx_model_path,
    )
    rows = build_dense_trec_run(
        retriever=retriever,
        topics=topics,
        run_name=run_name,
        top_k=top_k,
    )
    write_trec_run(run_output_path, rows)
    parameters = {
        "model_name": model_name,
        "text_representation": text_representation,
        "batch_size": batch_size,
        "device": device,
        "max_seq_length": max_seq_length,
        "normalize_embeddings": True,
        "query_encoder_quantization": (
            "dynamic_int8" if dynamic_quantization else "fp32"
        ),
        "query_encoder_backend": encoder_backend,
        "score_tie_decimals": DENSE_SCORE_TIE_DECIMALS,
        "embedding_dimension": retriever.index.metadata["embedding_dimension"],
        "index_schema_version": retriever.index.metadata["schema_version"],
        "corpus_fingerprint": retriever.index.metadata["corpus_fingerprint"],
    }
    report = trec_evaluation_report(
        rows=rows,
        qrels=qrels,
        run_name=run_name,
        top_k=top_k,
        topics_count=len(topics),
        trials_count=len(trials),
        retriever_name=DENSE_RETRIEVER_NAME,
        retriever_parameters=parameters,
    )
    if experiment_metadata:
        report["experiment"] = experiment_metadata
    write_json(metrics_output_path, report)
    if diagnostics_output_path:
        diagnostics = trec_topic_diagnostics(
            rows=rows,
            qrels=qrels,
            topics=topics,
            run_name=run_name,
            top_k=top_k,
            retriever_name=DENSE_RETRIEVER_NAME,
            retriever_parameters=parameters,
        )
        if experiment_metadata:
            diagnostics["experiment"] = experiment_metadata
        write_json(diagnostics_output_path, diagnostics)

    print(f"Wrote TREC dense run file to {run_output_path}")
    print(f"Wrote TREC dense metrics report to {metrics_output_path}")
    if diagnostics_output_path:
        print(f"Wrote TREC dense topic diagnostics to {diagnostics_output_path}")


def run_dense_experiment(config_path: Path, *, rebuild_index: bool = False) -> None:
    experiment = load_dense_experiment(config_path)
    print(f"Running dense experiment {experiment.name} from {experiment.config_label}")
    evaluate_trec_dense(
        trials_path=experiment.trials_path,
        topics_path=experiment.topics_path,
        qrels_path=experiment.qrels_path,
        index_path=experiment.index_path,
        run_output_path=experiment.run_output_path,
        metrics_output_path=experiment.metrics_output_path,
        diagnostics_output_path=experiment.diagnostics_output_path,
        run_name=experiment.name,
        top_k=experiment.top_k,
        model_name=experiment.model_name,
        text_representation=experiment.text_representation,
        batch_size=experiment.batch_size,
        device=experiment.device,
        max_seq_length=experiment.max_seq_length,
        rebuild_index=rebuild_index,
        dynamic_quantization=experiment.dynamic_quantization,
        encoder_backend=experiment.encoder_backend,
        onnx_model_path=experiment.onnx_model_path,
        experiment_metadata=experiment.metadata(),
    )


def check_trec_run_parity(
    *,
    baseline_path: Path,
    candidate_path: Path,
    depth: int,
    output_path: Path,
) -> None:
    report = trec_run_parity_report(
        baseline_path,
        candidate_path,
        depth=depth,
    )
    write_json(output_path, report)
    if not report["passed"]:
        topics = report["topics"]
        raise SystemExit(
            "TREC run parity failed: "
            f"{topics['mismatched']} ranking mismatches, "
            f"{len(topics['missing_from_candidate'])} missing topics, and "
            f"{len(topics['unexpected_in_candidate'])} unexpected topics"
        )
    print(
        f"TREC run parity passed for {report['topics']['matching']} topics "
        f"through rank {depth}; wrote {output_path}"
    )


def convert_dense_index_mmap(config_path: Path, output_path: Path) -> None:
    experiment = load_dense_experiment(config_path)
    if output_path.suffix.lower() != ".mmap":
        raise ValueError("Memory-mapped dense index output must use the .mmap suffix")
    trials = [trial_from_flat_record(row) for row in read_jsonl(experiment.trials_path)]
    index = load_dense_index(
        experiment.index_path,
        trials,
        model_name=experiment.model_name,
        text_representation=experiment.text_representation,
        max_seq_length=experiment.max_seq_length,
    )
    save_dense_index(output_path, index)
    print(
        f"Converted {index.metadata['trials']} dense embeddings to memory-mapped index "
        f"at {output_path}"
    )


def run_rrf_experiment(config_path: Path) -> None:
    experiment = load_rrf_experiment(config_path)
    topics = [topic_from_json_record(row) for row in read_jsonl(experiment.topics_path)]
    qrels = read_qrels_records(experiment.qrels_path)
    trials_count = len(read_jsonl(experiment.trials_path))
    runs = [
        RankedRun(
            name=component.name,
            weight=component.weight,
            rankings=read_trec_rankings(component.run_path),
        )
        for component in experiment.components
    ]
    expected_topic_ids = {topic.topic_id for topic in topics}
    for run in runs:
        actual_topic_ids = set(run.rankings)
        if actual_topic_ids != expected_topic_ids:
            missing = sorted(expected_topic_ids - actual_topic_ids)
            extra = sorted(actual_topic_ids - expected_topic_ids)
            raise ValueError(
                f"RRF component {run.name!r} does not match benchmark topics; "
                f"missing={missing}, extra={extra}"
            )
    rows = reciprocal_rank_fusion(
        runs,
        run_name=experiment.name,
        rrf_k=experiment.rrf_k,
        top_k=experiment.top_k,
        candidate_depth=experiment.candidate_depth,
    )
    write_trec_run(experiment.run_output_path, rows)
    parameters = {
        "rrf_k": experiment.rrf_k,
        "candidate_depth": experiment.candidate_depth,
        "components": [
            {
                "name": component.name,
                "weight": component.weight,
                "run_path": str(component.run_path),
                "run_sha256": sha256_file(component.run_path),
            }
            for component in experiment.components
        ],
    }
    report = trec_evaluation_report(
        rows=rows,
        qrels=qrels,
        run_name=experiment.name,
        top_k=experiment.top_k,
        topics_count=len(topics),
        trials_count=trials_count,
        retriever_name=RRF_RETRIEVER_NAME,
        retriever_parameters=parameters,
    )
    report["experiment"] = experiment.metadata()
    write_json(experiment.metrics_output_path, report)
    diagnostics = trec_topic_diagnostics(
        rows=rows,
        qrels=qrels,
        topics=topics,
        run_name=experiment.name,
        top_k=experiment.top_k,
        retriever_name=RRF_RETRIEVER_NAME,
        retriever_parameters=parameters,
    )
    diagnostics["experiment"] = experiment.metadata()
    write_json(experiment.diagnostics_output_path, diagnostics)
    print(f"Wrote TREC RRF run file to {experiment.run_output_path}")
    print(f"Wrote TREC RRF metrics report to {experiment.metrics_output_path}")
    print(f"Wrote TREC RRF topic diagnostics to {experiment.diagnostics_output_path}")


def compare_metrics(
    *,
    metrics_specs: list[str],
    output_path: Path,
    output_format: str | None,
    views: list[str],
) -> None:
    specs = [parse_metrics_spec(spec) for spec in metrics_specs]
    comparison = build_metrics_comparison(specs, views=views)
    resolved_format = infer_comparison_format(output_path, output_format)
    write_metrics_comparison(output_path, comparison, resolved_format)
    print(f"Wrote {len(comparison['rows'])} comparison rows to {output_path}")


def benchmark_serving(config_path: Path) -> None:
    benchmark = load_serving_benchmark(config_path)
    print(f"Running serving benchmark {benchmark.name} from {benchmark.config_label}")
    report = run_serving_benchmark(benchmark)
    print(f"Wrote serving benchmark report to {benchmark.output_path}")
    print(
        f"Cold start: {report['cold_start']['seconds']} s | "
        f"Peak RSS: {report['memory']['peak']['mib']} MiB"
    )
    for phase in report["cold_start"]["phases"]:
        print(
            f"startup/{phase['name']}: {phase['milliseconds']} ms | "
            f"retained RSS delta={phase['retained_rss_delta']['mib']} MiB | "
            f"peak RSS delta={phase['peak_rss_delta']['mib']} MiB"
        )
    dominant = report["cold_start"]["dominant_resource_phase"]
    print(
        f"Dominant startup resource: {dominant['name']} "
        f"({dominant['retained_rss_delta']['mib']} MiB retained RSS delta)"
    )
    for mode in benchmark.modes:
        mode_report = report["warm"]["modes"][mode]
        latency = mode_report["handler_latency_ms"]
        print(
            f"{mode}: p50={latency['p50']} ms p95={latency['p95']} ms "
            f"throughput={mode_report['sequential_requests_per_second']} req/s"
        )


def benchmark_lexical_backend_command(
    *,
    serving_config_path: Path,
    backend: str,
    experiment_config_path: Path,
    output_path: Path,
) -> None:
    serving = load_serving_benchmark(serving_config_path)
    if backend == "fielded-bm25":
        experiment = load_bm25_experiment(experiment_config_path)
        index_path = experiment.index_path
        field_weights = experiment.field_weights
    elif backend == "sqlite-fts5":
        experiment = load_sqlite_fts_experiment(experiment_config_path)
        index_path = experiment.index_path
        field_weights = experiment.field_weights
    else:
        raise ValueError(f"Unsupported lexical backend: {backend}")
    if experiment.trials_path != serving.corpus_path:
        raise ValueError("Lexical experiment and serving benchmark corpus paths must match")
    report = benchmark_lexical_backend(
        backend=backend,
        corpus_path=serving.corpus_path,
        index_path=index_path,
        field_weights=field_weights,
        queries=serving.queries,
        warmup_rounds=serving.warmup_rounds,
        measurement_rounds=serving.measurement_rounds,
        top_k=serving.top_k,
        output_path=output_path,
    )
    retriever_phase = report["cold_start"]["phases"][1]
    latency = report["warm"]["latency_ms"]
    print(f"Wrote {backend} lexical benchmark to {output_path}")
    print(
        f"{backend}: cold={report['cold_start']['milliseconds']} ms | "
        f"retriever RSS delta={retriever_phase['retained_rss_delta']['mib']} MiB | "
        f"p50={latency['p50']} ms | p95={latency['p95']} ms"
    )


def compare_lexical_backends(
    *,
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> None:
    comparison = compare_lexical_backend_reports(baseline_path, candidate_path)
    write_lexical_backend_comparison(output_path, comparison)
    print(f"Wrote lexical backend comparison to {output_path}")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    if path.suffix == ".jsonl":
        return qrels_to_mapping([qrel_from_json_record(row) for row in read_jsonl(path)])
    return qrels_to_mapping(parse_qrels(path))


def read_qrels_records(path: Path) -> list[Qrel]:
    if path.suffix == ".jsonl":
        return [qrel_from_json_record(row) for row in read_jsonl(path)]
    return parse_qrels(path)


def unique_nct_ids_from_qrels(qrels: list[Qrel]) -> list[str]:
    nct_ids: list[str] = []
    seen: set[str] = set()
    for qrel in qrels:
        nct_id = qrel.nct_id.strip().upper()
        if not nct_id or nct_id in seen:
            continue
        nct_ids.append(nct_id)
        seen.add(nct_id)
    return nct_ids


def ingest_trec_topics(year: int, input_path: Path, output_path: Path) -> None:
    topics = parse_topics_xml(input_path, year=year)
    write_jsonl(output_path, (topic_to_json_record(topic) for topic in topics))
    print(f"Wrote {len(topics)} normalized TREC topics to {output_path}")


def ingest_trec_qrels(year: int, input_path: Path, output_path: Path) -> None:
    qrels = parse_qrels(input_path, year=year)
    write_jsonl(output_path, (qrel_to_json_record(qrel) for qrel in qrels))
    print(f"Wrote {len(qrels)} normalized TREC qrels to {output_path}")


def validate_trec(topics_path: Path, qrels_path: Path, output_path: Path | None) -> None:
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
    qrels = [qrel_from_json_record(row) for row in read_jsonl(qrels_path)]
    summary = validate_topics_and_qrels(topics, qrels)
    if output_path:
        write_json(output_path, summary)
        print(f"Wrote TREC validation summary to {output_path}")
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")


def write_manifest(
    *,
    name: str,
    source_url: str,
    input_path: Path,
    output_path: Path,
    dataset: str,
    year: int | None,
    parser: str,
    metadata_items: list[str],
) -> None:
    manifest = build_source_manifest(
        name=name,
        source_url=source_url,
        input_path=input_path,
        dataset=dataset,
        year=year,
        parser=parser,
        metadata=parse_metadata_items(metadata_items),
    )
    write_json(output_path, manifest_to_json_record(manifest))
    print(f"Wrote source manifest to {output_path}")


def parse_metadata_items(items: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid metadata item {item!r}; expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid metadata item {item!r}; key cannot be empty")
        metadata[key] = value.strip()
    return metadata


def parse_field_weights(items: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid field weight {item!r}; expected field=weight")
        field, value = item.split("=", 1)
        field = field.strip()
        if not field:
            raise ValueError(f"Invalid field weight {item!r}; field cannot be empty")
        try:
            weights[field] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight for field {field!r}: {value!r}") from exc
    return weights


def _add_dense_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--text-representation",
        choices=sorted(TEXT_REPRESENTATIONS),
        default="title_summary_conditions",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seq-length", type=int, default=256)


def slugify(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in slug.split("_") if part) or "all"


if __name__ == "__main__":
    main()
