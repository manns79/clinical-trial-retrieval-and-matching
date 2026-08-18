from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from clinical_trial_matching.benchmarking.serving import (
    latency_summary,
    process_memory,
    startup_phase_report,
)
from clinical_trial_matching.ingestion.clinicaltrials import trial_from_flat_record
from clinical_trial_matching.io import read_json, read_jsonl, write_json
from clinical_trial_matching.models import SearchResult
from clinical_trial_matching.retrieval.bm25 import load_or_build_bm25_retriever
from clinical_trial_matching.retrieval.sqlite_fts import (
    load_or_build_sqlite_fts_retriever,
)

LexicalBackend = Literal["fielded-bm25", "sqlite-fts5"]


class LexicalRetriever(Protocol):
    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]: ...


def benchmark_lexical_backend(
    *,
    backend: LexicalBackend,
    corpus_path: Path,
    index_path: Path,
    field_weights: Mapping[str, float],
    queries: Sequence[str],
    warmup_rounds: int,
    measurement_rounds: int,
    top_k: int,
    output_path: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Trial corpus not found: {corpus_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Lexical index not found: {index_path}")
    if not queries:
        raise ValueError("At least one benchmark query is required")
    if warmup_rounds < 0 or measurement_rounds < 1 or top_k < 1:
        raise ValueError("Benchmark rounds and top_k are invalid")

    benchmark_start = clock()
    memory_before = process_memory()
    corpus_start = clock()
    trials = [trial_from_flat_record(row) for row in read_jsonl(corpus_path)]
    memory_after_corpus = process_memory()
    corpus_phase = startup_phase_report(
        name="corpus",
        elapsed_ms=(clock() - corpus_start) * 1000,
        before=memory_before,
        after=memory_after_corpus,
    )

    retriever_start = clock()
    retriever: LexicalRetriever
    if backend == "fielded-bm25":
        retriever = load_or_build_bm25_retriever(
            trials=trials,
            retriever_name="fielded-bm25",
            field_weights=dict(field_weights),
            corpus_path=corpus_path,
            index_path=index_path,
        )
    elif backend == "sqlite-fts5":
        retriever = load_or_build_sqlite_fts_retriever(
            trials=trials,
            field_weights=field_weights,
            corpus_path=corpus_path,
            index_path=index_path,
        )
    else:
        raise ValueError(f"Unsupported lexical backend: {backend}")
    memory_after_retriever = process_memory()
    retriever_phase = startup_phase_report(
        name=backend,
        elapsed_ms=(clock() - retriever_start) * 1000,
        before=memory_after_corpus,
        after=memory_after_retriever,
    )
    cold_start_ms = (clock() - benchmark_start) * 1000

    for _ in range(warmup_rounds):
        for query in queries:
            retriever.search(query, top_k=top_k)

    latencies: list[float] = []
    memory_before_measurement = process_memory()
    measurement_start = clock()
    for _ in range(measurement_rounds):
        for query in queries:
            request_start = clock()
            retriever.search(query, top_k=top_k)
            latencies.append((clock() - request_start) * 1000)
    measurement_ms = (clock() - measurement_start) * 1000
    memory_after_measurement = process_memory()
    requests = len(latencies)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": backend,
        "configuration": {
            "corpus": str(corpus_path),
            "index": str(index_path),
            "field_weights": dict(field_weights),
            "queries": len(queries),
            "warmup_rounds": warmup_rounds,
            "measurement_rounds": measurement_rounds,
            "top_k": top_k,
        },
        "measurement_scope": {
            "cold_start": "Normalized corpus plus persisted retriever load in a fresh CLI process",
            "warm_latency": "Sequential in-process retrieval after deterministic warmup",
            "memory": "Process RSS; SQLite file pages may also use the operating-system cache",
        },
        "cold_start": {
            "milliseconds": round(cold_start_ms, 3),
            "phases": [corpus_phase, retriever_phase],
        },
        "warm": {
            "requests": requests,
            "measurement_wall_ms": round(measurement_ms, 3),
            "latency_ms": latency_summary(latencies),
            "sequential_requests_per_second": _throughput(requests, measurement_ms),
        },
        "memory": {
            "before": _memory_value(memory_before["rss_bytes"]),
            "after_corpus": _memory_value(memory_after_corpus["rss_bytes"]),
            "after_retriever": _memory_value(memory_after_retriever["rss_bytes"]),
            "before_measurement": _memory_value(memory_before_measurement["rss_bytes"]),
            "after_measurement": _memory_value(memory_after_measurement["rss_bytes"]),
            "peak": _memory_value(
                max(
                    memory_after_measurement["rss_bytes"],
                    memory_after_measurement["peak_rss_bytes"],
                )
            ),
        },
        "artifacts": {
            "corpus": _file_value(corpus_path),
            "index": _file_value(index_path),
        },
        "cost": {
            "external_api_calls": 0,
            "hosted_service_cost_usd": 0.0,
        },
    }
    write_json(output_path, report)
    return report


def compare_lexical_backend_reports(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    reports = [read_json(baseline_path), read_json(candidate_path)]
    rows = [_comparison_row(report) for report in reports]
    baseline = rows[0]
    candidate = rows[1]
    return {
        "baseline": baseline["backend"],
        "candidate": candidate["backend"],
        "rows": rows,
        "candidate_ratios": {
            "cold_start": _ratio(candidate["cold_start_ms"], baseline["cold_start_ms"]),
            "retriever_rss_delta": _ratio(
                candidate["retriever_rss_delta_mib"],
                baseline["retriever_rss_delta_mib"],
            ),
            "warm_p50": _ratio(candidate["warm_p50_ms"], baseline["warm_p50_ms"]),
            "warm_p95": _ratio(candidate["warm_p95_ms"], baseline["warm_p95_ms"]),
            "throughput": _ratio(
                candidate["requests_per_second"],
                baseline["requests_per_second"],
            ),
            "index_size": _ratio(candidate["index_mib"], baseline["index_mib"]),
        },
    }


def write_lexical_backend_comparison(path: Path, comparison: Mapping[str, Any]) -> None:
    columns = (
        "backend",
        "cold_start_ms",
        "retriever_load_ms",
        "retriever_rss_delta_mib",
        "warm_p50_ms",
        "warm_p95_ms",
        "requests_per_second",
        "index_mib",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in comparison["rows"]:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_row(report: Mapping[str, Any]) -> dict[str, Any]:
    retriever_phase = report["cold_start"]["phases"][1]
    return {
        "backend": report["backend"],
        "cold_start_ms": report["cold_start"]["milliseconds"],
        "retriever_load_ms": retriever_phase["milliseconds"],
        "retriever_rss_delta_mib": retriever_phase["retained_rss_delta"]["mib"],
        "warm_p50_ms": report["warm"]["latency_ms"]["p50"],
        "warm_p95_ms": report["warm"]["latency_ms"]["p95"],
        "requests_per_second": report["warm"]["sequential_requests_per_second"],
        "index_mib": report["artifacts"]["index"]["mib"],
    }


def _memory_value(value: int) -> dict[str, int | float]:
    return {"bytes": value, "mib": round(value / (1024 * 1024), 3)}


def _file_value(path: Path) -> dict[str, int | float | str]:
    return {"path": str(path), **_memory_value(path.stat().st_size)}


def _throughput(requests: int, elapsed_ms: float) -> float:
    return 0.0 if elapsed_ms <= 0 else round(requests / (elapsed_ms / 1000), 3)


def _ratio(candidate: float, baseline: float) -> float:
    return 0.0 if baseline == 0 else round(candidate / baseline, 4)
