from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clinical_trial_matching.benchmarking.serving import (
    ApiServingRuntime,
    ServingRuntime,
    latency_summary,
    load_serving_benchmark,
    process_memory,
    startup_phase_report,
    temporary_environment,
)
from clinical_trial_matching.evaluation.experiments import CrossEncoderExperiment
from clinical_trial_matching.evaluation.trec import (
    trec_evaluation_report,
    trec_topic_diagnostics,
    write_trec_run,
)
from clinical_trial_matching.ingestion.trec import qrel_from_json_record, topic_from_json_record
from clinical_trial_matching.io import read_json, read_jsonl, write_json
from clinical_trial_matching.retrieval.hybrid import read_trec_rankings
from clinical_trial_matching.retrieval.rerank import (
    CROSS_ENCODER_RETRIEVER_NAME,
    OnnxCrossEncoder,
    RerankerFramework,
    load_reranker_framework,
    rerank_topic,
)
from clinical_trial_matching.trial_store import load_trial_store


def run_cross_encoder_experiment(
    experiment: CrossEncoderExperiment,
    *,
    clock: Callable[[], float] = time.perf_counter,
    framework_loader: Callable[[], RerankerFramework] = load_reranker_framework,
    reranker_factory: Callable[..., Any] = OnnxCrossEncoder,
) -> dict[str, Any]:
    _validate_experiment_inputs(experiment)
    topics = [topic_from_json_record(row) for row in read_jsonl(experiment.topics_path)]
    qrels = [qrel_from_json_record(row) for row in read_jsonl(experiment.qrels_path)]
    baseline_rankings = read_trec_rankings(experiment.baseline_run_path)
    topic_ids = {topic.topic_id for topic in topics}
    if set(baseline_rankings) != topic_ids:
        raise ValueError("Baseline run and development topics contain different topic IDs")
    if any(len(baseline_rankings[topic_id]) < experiment.top_k for topic_id in topic_ids):
        raise ValueError("Baseline run does not contain benchmark.top_k results for every topic")
    trial_store = load_trial_store(
        experiment.trial_store_path,
        corpus_path=experiment.corpus_path,
    )
    baseline_metrics = read_json(experiment.baseline_metrics_path)

    phases = []
    phase_start = clock()
    phase_before = process_memory()
    framework = framework_loader()
    phase_after = process_memory()
    phases.append(
        startup_phase_report(
            name="cross_encoder_framework",
            elapsed_ms=(clock() - phase_start) * 1000,
            before=phase_before,
            after=phase_after,
        )
    )
    phase_start = clock()
    phase_before = phase_after
    reranker = reranker_factory(
        experiment.model_artifact_path,
        model_name=experiment.model_name,
        model_revision=experiment.model_revision,
        model_file=experiment.model_file,
        max_length=experiment.max_length,
        framework=framework,
    )
    phase_after = process_memory()
    phases.append(
        startup_phase_report(
            name="cross_encoder_model",
            elapsed_ms=(clock() - phase_start) * 1000,
            before=phase_before,
            after=phase_after,
        )
    )
    first_topic = topics[0]
    first_trial = trial_store.get(baseline_rankings[first_topic.topic_id][0])
    if first_trial is None:
        raise ValueError("Trial store is missing the first reranking candidate")
    from clinical_trial_matching.retrieval.dense import trial_text

    phase_start = clock()
    phase_before = phase_after
    reranker.predict(
        [(first_topic.text, trial_text(first_trial, experiment.text_representation))],
        batch_size=1,
    )
    phase_after = process_memory()
    phases.append(
        startup_phase_report(
            name="cross_encoder_first_inference_thread_pool",
            elapsed_ms=(clock() - phase_start) * 1000,
            before=phase_before,
            after=phase_after,
        )
    )

    depth_reports = {}
    for candidate_depth in experiment.candidate_depths:
        rows = []
        metadata_values = []
        inference_values = []
        total_values = []
        rss_values = []
        run_name = f"{experiment.name}_depth_{candidate_depth}"
        for topic in topics:
            baseline_ids = baseline_rankings[topic.topic_id][: experiment.top_k]
            total_start = clock()
            metadata_start = clock()
            candidate_trials = trial_store.get_many(baseline_ids[:candidate_depth])
            metadata_ms = (clock() - metadata_start) * 1000
            result = rerank_topic(
                topic=topic,
                baseline_nct_ids=baseline_ids,
                candidate_trials=candidate_trials,
                reranker=reranker,
                candidate_depth=candidate_depth,
                top_k=experiment.top_k,
                text_representation=experiment.text_representation,
                batch_size=experiment.batch_size,
                run_name=run_name,
                clock=clock,
            )
            rows.extend(result.rows)
            metadata_values.append(metadata_ms)
            inference_values.append(result.inference_ms)
            total_values.append((clock() - total_start) * 1000)
            rss_values.append(process_memory()["rss_bytes"])

        parameters = _reranker_parameters(experiment, candidate_depth)
        metrics = trec_evaluation_report(
            rows=rows,
            qrels=qrels,
            run_name=run_name,
            top_k=experiment.top_k,
            topics_count=len(topics),
            trials_count=trial_store.count,
            retriever_name=CROSS_ENCODER_RETRIEVER_NAME,
            retriever_parameters=parameters,
        )
        metrics["experiment"] = experiment.metadata()
        metrics["baseline"] = {
            "run": str(experiment.baseline_run_path),
            "metrics": str(experiment.baseline_metrics_path),
        }
        run_path = experiment.run_path(candidate_depth)
        metrics_path = experiment.metrics_path(candidate_depth)
        diagnostics_path = experiment.diagnostics_path(candidate_depth)
        write_trec_run(run_path, rows)
        write_json(metrics_path, metrics)
        diagnostics = trec_topic_diagnostics(
            rows=rows,
            qrels=qrels,
            topics=topics,
            run_name=run_name,
            top_k=experiment.top_k,
            retriever_name=CROSS_ENCODER_RETRIEVER_NAME,
            retriever_parameters=parameters,
        )
        diagnostics["experiment"] = experiment.metadata()
        write_json(diagnostics_path, diagnostics)
        depth_reports[str(candidate_depth)] = {
            "candidate_depth": candidate_depth,
            "metrics": metrics["metrics"],
            "graded_metrics": metrics["graded_metrics"],
            "metric_deltas": _metric_deltas(baseline_metrics, metrics),
            "latency_ms_per_topic": {
                "metadata": latency_summary(metadata_values),
                "inference": latency_summary(inference_values),
                "total": latency_summary(total_values),
            },
            "sampled_process_rss_mib": {
                "minimum": round(min(rss_values) / (1024 * 1024), 3),
                "maximum": round(max(rss_values) / (1024 * 1024), 3),
            },
            "artifacts": {
                "run": str(run_path),
                "metrics": str(metrics_path),
                "diagnostics": str(diagnostics_path),
            },
        }

    final_memory = process_memory()
    report = {
        "schema_version": 1,
        "experiment": experiment.metadata(),
        "scope": "Development topics only; holdout topics are not read.",
        "baseline": {
            "run": str(experiment.baseline_run_path),
            "metrics": str(experiment.baseline_metrics_path),
        },
        "model": {
            "name": experiment.model_name,
            "revision": experiment.model_revision,
            "file": experiment.model_file,
            "precision": experiment.model_precision,
            "backend": "onnxruntime",
            "artifact": str(experiment.model_artifact_path),
            "artifact_bytes": _directory_bytes(experiment.model_artifact_path),
            "device": experiment.device,
            "max_length": experiment.max_length,
            "batch_size": experiment.batch_size,
            "text_representation": experiment.text_representation,
        },
        "model_loading": {
            "phases": phases,
            "rss_after_first_inference_mib": round(
                phase_after["rss_bytes"] / (1024 * 1024), 3
            ),
        },
        "depths": depth_reports,
        "process_peak_rss_mib": round(
            max(final_memory["peak_rss_bytes"], final_memory["rss_bytes"])
            / (1024 * 1024),
            3,
        ),
        "cost": {"hosted_service_cost_usd": 0.0, "external_api_calls": 0},
    }
    write_json(experiment.report_output_path, report)
    return report


def benchmark_cross_encoder_headroom(
    experiment: CrossEncoderExperiment,
    *,
    clock: Callable[[], float] = time.perf_counter,
    runtime_factory: Callable[[], ServingRuntime] = ApiServingRuntime,
    framework_loader: Callable[[], RerankerFramework] = load_reranker_framework,
    reranker_factory: Callable[..., Any] = OnnxCrossEncoder,
) -> dict[str, Any]:
    _validate_experiment_inputs(experiment)
    serving = load_serving_benchmark(experiment.serving_config_path)
    topics = [topic_from_json_record(row) for row in read_jsonl(experiment.topics_path)]
    rankings = read_trec_rankings(experiment.baseline_run_path)
    store = load_trial_store(experiment.trial_store_path, corpus_path=experiment.corpus_path)
    first_trial = store.get(rankings[topics[0].topic_id][0])
    if first_trial is None:
        raise ValueError("Trial store is missing the first reranking candidate")
    from clinical_trial_matching.retrieval.dense import trial_text

    before = process_memory()
    phases = []
    with temporary_environment(serving.environment()):
        phase_start = clock()
        phase_before = before
        runtime = runtime_factory()
        runtime.preload()
        phase_after = process_memory()
        phases.append(
            startup_phase_report(
                name="selected_retrieval_stack",
                elapsed_ms=(clock() - phase_start) * 1000,
                before=phase_before,
                after=phase_after,
            )
        )
        phase_start = clock()
        phase_before = phase_after
        framework = framework_loader()
        phase_after = process_memory()
        phases.append(
            startup_phase_report(
                name="cross_encoder_framework",
                elapsed_ms=(clock() - phase_start) * 1000,
                before=phase_before,
                after=phase_after,
            )
        )
        phase_start = clock()
        phase_before = phase_after
        reranker = reranker_factory(
            experiment.model_artifact_path,
            model_name=experiment.model_name,
            model_revision=experiment.model_revision,
            model_file=experiment.model_file,
            max_length=experiment.max_length,
            framework=framework,
        )
        phase_after = process_memory()
        phases.append(
            startup_phase_report(
                name="cross_encoder_model",
                elapsed_ms=(clock() - phase_start) * 1000,
                before=phase_before,
                after=phase_after,
            )
        )
        phase_start = clock()
        phase_before = phase_after
        reranker.predict(
            [(topics[0].text, trial_text(first_trial, experiment.text_representation))],
            batch_size=1,
        )
        after = process_memory()
        phases.append(
            startup_phase_report(
                name="cross_encoder_first_inference_thread_pool",
                elapsed_ms=(clock() - phase_start) * 1000,
                before=phase_before,
                after=after,
            )
        )
        depth_probes = {}
        baseline_ids = rankings[topics[0].topic_id][: experiment.top_k]
        for candidate_depth in experiment.candidate_depths:
            probe_start = clock()
            trials = store.get_many(baseline_ids[:candidate_depth])
            reranker.predict(
                [
                    (
                        topics[0].text,
                        trial_text(trial, experiment.text_representation),
                    )
                    for trial in trials
                ],
                batch_size=experiment.batch_size,
            )
            after = process_memory()
            observed_peak_bytes = max(after["peak_rss_bytes"], after["rss_bytes"])
            depth_probes[str(candidate_depth)] = {
                "candidate_depth": candidate_depth,
                "milliseconds": round((clock() - probe_start) * 1000, 3),
                "rss_after_mib": round(after["rss_bytes"] / (1024 * 1024), 3),
                "peak_process_rss_mib": round(
                    observed_peak_bytes / (1024 * 1024), 3
                ),
            }
    peak_mib = round(
        max(after["peak_rss_bytes"], after["rss_bytes"]) / (1024 * 1024),
        3,
    )
    selected_rss = int(phases[0]["rss_after"]["bytes"])
    final_rss = int(after["rss_bytes"])
    report = {
        "schema_version": 1,
        "experiment": experiment.metadata(),
        "serving_profile": serving.metadata(),
        "phases": phases,
        "depth_probes": depth_probes,
        "selected_stack_rss_mib": round(selected_rss / (1024 * 1024), 3),
        "combined_rss_mib": round(final_rss / (1024 * 1024), 3),
        "incremental_reranker_rss_mib": round(
            (final_rss - selected_rss) / (1024 * 1024), 3
        ),
        "peak_process_rss_mib": peak_mib,
        "budget": {
            "peak_process_rss_mib": experiment.peak_process_rss_mib,
            "passed": peak_mib <= experiment.peak_process_rss_mib,
        },
        "cost": {"hosted_service_cost_usd": 0.0, "external_api_calls": 0},
    }
    write_json(experiment.headroom_output_path, report)
    return report


def _reranker_parameters(
    experiment: CrossEncoderExperiment,
    candidate_depth: int,
) -> dict[str, Any]:
    return {
        "model_name": experiment.model_name,
        "model_revision": experiment.model_revision,
        "model_file": experiment.model_file,
        "model_precision": experiment.model_precision,
        "backend": "onnxruntime",
        "candidate_depth": candidate_depth,
        "batch_size": experiment.batch_size,
        "max_length": experiment.max_length,
        "text_representation": experiment.text_representation,
        "tail_policy": "append_baseline_after_reranked_candidates",
    }


def _metric_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": {
            view: {
                metric: float(value) - float(baseline["metrics"][view][metric])
                for metric, value in values.items()
            }
            for view, values in candidate["metrics"].items()
        },
        "graded_metrics": {
            metric: float(value) - float(baseline["graded_metrics"][metric])
            for metric, value in candidate["graded_metrics"].items()
        },
    }


def _validate_experiment_inputs(experiment: CrossEncoderExperiment) -> None:
    missing = [
        path
        for path in (
            experiment.corpus_path,
            experiment.trial_store_path,
            experiment.topics_path,
            experiment.qrels_path,
            experiment.baseline_run_path,
            experiment.baseline_metrics_path,
            experiment.model_artifact_path,
            experiment.serving_config_path,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cross-encoder experiment input(s) not found: "
            + ", ".join(str(path) for path in missing)
        )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
