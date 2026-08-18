from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from clinical_trial_matching.io import write_json

SERVING_BENCHMARK_SCHEMA_VERSION = 1
PRIMARY_MODES = ("sqlite-fts5", "dense", "hybrid")
STAGE_NAMES = ("lexical", "embedding", "fusion", "total")
STARTUP_RESOURCE_PHASES = ("corpus", "sqlite_fts5", "dense_index_and_model")


@dataclass(frozen=True)
class ServingBenchmark:
    name: str
    description: str
    project_root: Path
    corpus_path: Path
    sqlite_fts_index_path: Path
    dense_index_path: Path
    dense_model_name: str
    dense_text_representation: str
    dense_batch_size: int
    dense_device: str
    dense_max_seq_length: int | None
    rrf_k: int
    rrf_candidate_depth: int
    modes: tuple[str, ...]
    queries: tuple[str, ...]
    warmup_rounds: int
    measurement_rounds: int
    top_k: int
    snippet_chars: int
    output_path: Path
    config_path: Path
    config_label: str
    config_sha256: str

    def environment(self) -> dict[str, str]:
        return {
            "TRIAL_CORPUS_PATH": str(self.corpus_path),
            "SQLITE_FTS_INDEX_PATH": str(self.sqlite_fts_index_path),
            "DENSE_INDEX_PATH": str(self.dense_index_path),
            "DENSE_MODEL_NAME": self.dense_model_name,
            "DENSE_TEXT_REPRESENTATION": self.dense_text_representation,
            "DENSE_BATCH_SIZE": str(self.dense_batch_size),
            "DENSE_DEVICE": self.dense_device,
            "DENSE_MAX_SEQ_LENGTH": (
                "" if self.dense_max_seq_length is None else str(self.dense_max_seq_length)
            ),
            "RRF_K": str(self.rrf_k),
            "RRF_CANDIDATE_DEPTH": str(self.rrf_candidate_depth),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LOG_LEVEL": "WARNING",
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SERVING_BENCHMARK_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "config_path": self.config_label,
            "config_sha256": self.config_sha256,
        }


class ServingRuntime(Protocol):
    def preload(
        self,
        on_phase_complete: Callable[[str], None] | None = None,
    ) -> None: ...

    def search(
        self,
        *,
        query: str,
        mode: str,
        top_k: int,
        snippet_chars: int,
    ) -> dict[str, Any]: ...


class ApiServingRuntime:
    def __init__(self) -> None:
        from clinical_trial_matching.api.main import (
            SearchRequest,
            preload_search_resources,
            search,
        )

        self._request_type = SearchRequest
        self._preload = preload_search_resources
        self._search = search

    def preload(
        self,
        on_phase_complete: Callable[[str], None] | None = None,
    ) -> None:
        self._preload(on_phase_complete=on_phase_complete)

    def search(
        self,
        *,
        query: str,
        mode: str,
        top_k: int,
        snippet_chars: int,
    ) -> dict[str, Any]:
        response = self._search(
            self._request_type(
                query=query,
                top_k=top_k,
                snippet_chars=snippet_chars,
                retriever=cast(
                    Literal[
                        "bm25",
                        "fielded-bm25",
                        "sqlite-fts5",
                        "dense",
                        "hybrid",
                    ],
                    mode,
                ),
            )
        )
        return dict(response.model_dump())


def load_serving_benchmark(path: Path) -> ServingBenchmark:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid serving benchmark JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Serving benchmark config must be a JSON object")
    _reject_unknown_fields(
        payload,
        "config",
        {
            "schema_version",
            "name",
            "description",
            "project_root",
            "serving",
            "benchmark",
            "artifacts",
        },
    )
    if payload.get("schema_version") != SERVING_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported serving benchmark schema_version "
            f"{payload.get('schema_version')!r}; expected {SERVING_BENCHMARK_SCHEMA_VERSION}"
        )

    config_path = path.resolve()
    project_root_value = _required_string(payload, "project_root")
    project_root = (config_path.parent / project_root_value).resolve()
    name = _required_string(payload, "name")
    description = _required_string(payload, "description")
    serving = _required_mapping(payload, "serving")
    benchmark = _required_mapping(payload, "benchmark")
    artifacts = _required_mapping(payload, "artifacts")
    _reject_unknown_fields(
        serving,
        "serving",
        {
            "corpus",
            "sqlite_fts_index",
            "dense_index",
            "dense_model_name",
            "dense_text_representation",
            "dense_batch_size",
            "dense_device",
            "dense_max_seq_length",
            "rrf_k",
            "rrf_candidate_depth",
        },
    )
    _reject_unknown_fields(
        benchmark,
        "benchmark",
        {
            "modes",
            "queries",
            "warmup_rounds",
            "measurement_rounds",
            "top_k",
            "snippet_chars",
        },
    )
    _reject_unknown_fields(artifacts, "artifacts", {"output"})

    modes = _string_list(benchmark, "modes", "benchmark")
    if tuple(modes) != PRIMARY_MODES:
        raise ValueError(
            "benchmark.modes must list the three primary modes in order: "
            + ", ".join(PRIMARY_MODES)
        )
    queries = _string_list(benchmark, "queries", "benchmark")
    if not queries:
        raise ValueError("benchmark.queries must contain at least one query")

    try:
        config_label = config_path.relative_to(project_root).as_posix()
    except ValueError:
        config_label = config_path.name
    max_seq_length_value = serving.get("dense_max_seq_length")
    if max_seq_length_value is not None:
        max_seq_length = _positive_integer(
            serving,
            "dense_max_seq_length",
            prefix="serving.",
        )
    else:
        max_seq_length = None

    return ServingBenchmark(
        name=name,
        description=description,
        project_root=project_root,
        corpus_path=_project_path(project_root, serving, "corpus", "serving"),
        sqlite_fts_index_path=_project_path(
            project_root,
            serving,
            "sqlite_fts_index",
            "serving",
        ),
        dense_index_path=_project_path(project_root, serving, "dense_index", "serving"),
        dense_model_name=_required_string(serving, "dense_model_name", "serving."),
        dense_text_representation=_required_string(
            serving,
            "dense_text_representation",
            "serving.",
        ),
        dense_batch_size=_positive_integer(
            serving,
            "dense_batch_size",
            prefix="serving.",
        ),
        dense_device=_required_string(serving, "dense_device", "serving."),
        dense_max_seq_length=max_seq_length,
        rrf_k=_positive_integer(serving, "rrf_k", prefix="serving."),
        rrf_candidate_depth=_positive_integer(
            serving,
            "rrf_candidate_depth",
            prefix="serving.",
        ),
        modes=tuple(modes),
        queries=tuple(queries),
        warmup_rounds=_non_negative_integer(
            benchmark,
            "warmup_rounds",
            prefix="benchmark.",
        ),
        measurement_rounds=_positive_integer(
            benchmark,
            "measurement_rounds",
            prefix="benchmark.",
        ),
        top_k=_positive_integer(benchmark, "top_k", prefix="benchmark."),
        snippet_chars=_positive_integer(
            benchmark,
            "snippet_chars",
            prefix="benchmark.",
        ),
        output_path=_project_path(project_root, artifacts, "output", "artifacts"),
        config_path=config_path,
        config_label=config_label,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def run_serving_benchmark(
    benchmark: ServingBenchmark,
    *,
    runtime_factory: Callable[[], ServingRuntime] = ApiServingRuntime,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    _validate_artifacts_exist(benchmark)
    memory_before = process_memory()
    with temporary_environment(benchmark.environment()):
        cold_start = clock()
        phase_start = cold_start
        phase_memory_before = memory_before
        startup_phases: list[dict[str, Any]] = []
        runtime = runtime_factory()
        phase_memory_after = process_memory()
        startup_phases.append(
            startup_phase_report(
                name="api_import",
                elapsed_ms=_elapsed_ms(phase_start, clock),
                before=phase_memory_before,
                after=phase_memory_after,
            )
        )
        phase_start = clock()
        phase_memory_before = phase_memory_after

        def record_startup_phase(name: str) -> None:
            nonlocal phase_start, phase_memory_before
            phase_memory_after = process_memory()
            startup_phases.append(
                startup_phase_report(
                    name=name,
                    elapsed_ms=_elapsed_ms(phase_start, clock),
                    before=phase_memory_before,
                    after=phase_memory_after,
                )
            )
            phase_start = clock()
            phase_memory_before = phase_memory_after

        runtime.preload(on_phase_complete=record_startup_phase)
        cold_start_ms = _elapsed_ms(cold_start, clock)
        memory_after_startup = process_memory()
        validate_startup_phases(startup_phases)

        for _ in range(benchmark.warmup_rounds):
            for query in benchmark.queries:
                for mode in benchmark.modes:
                    runtime.search(
                        query=query,
                        mode=mode,
                        top_k=benchmark.top_k,
                        snippet_chars=benchmark.snippet_chars,
                    )

        measurements: dict[str, list[dict[str, Any]]] = {
            mode: [] for mode in benchmark.modes
        }
        measurement_start = clock()
        for _ in range(benchmark.measurement_rounds):
            for query in benchmark.queries:
                for mode in benchmark.modes:
                    request_start = clock()
                    response = runtime.search(
                        query=query,
                        mode=mode,
                        top_k=benchmark.top_k,
                        snippet_chars=benchmark.snippet_chars,
                    )
                    handler_wall_ms = _elapsed_ms(request_start, clock)
                    measurements[mode].append(
                        {
                            "handler_wall_ms": handler_wall_ms,
                            "latency_ms": response["latency_ms"],
                            "result_count": len(response["results"]),
                            "rss_bytes": process_memory()["rss_bytes"],
                        }
                    )
        measurement_wall_ms = _elapsed_ms(measurement_start, clock)
        memory_after_benchmark = process_memory()

    artifacts = serving_artifact_sizes(benchmark)
    mode_reports = {
        mode: summarize_mode_measurements(rows)
        for mode, rows in measurements.items()
    }
    total_requests = sum(report["requests"] for report in mode_reports.values())
    report = {
        "benchmark": benchmark.metadata(),
        "generated_at": datetime.now(UTC).isoformat(),
        "measurement_scope": {
            "cold_start": (
                "API module import plus corpus, lexical index, dense index, and model preload "
                "inside a fresh benchmark CLI process; excludes model download time and may "
                "benefit from the operating system filesystem cache"
            ),
            "warm_latency": (
                "In-process FastAPI search handler wall time after deterministic warmup; "
                "excludes HTTP transport and JSON serialization"
            ),
            "throughput": "Single-process sequential requests per second; not a concurrency test",
            "memory": (
                "Shared serving-process RSS with lexical and dense resources both preloaded; "
                "per-mode values are sampled immediately after each request"
            ),
        },
        "configuration": {
            "modes": list(benchmark.modes),
            "queries": len(benchmark.queries),
            "warmup_rounds": benchmark.warmup_rounds,
            "measurement_rounds": benchmark.measurement_rounds,
            "requests_per_mode": benchmark.measurement_rounds * len(benchmark.queries),
            "top_k": benchmark.top_k,
            "snippet_chars": benchmark.snippet_chars,
            "dense_model_name": benchmark.dense_model_name,
            "dense_text_representation": benchmark.dense_text_representation,
            "rrf_k": benchmark.rrf_k,
            "rrf_candidate_depth": benchmark.rrf_candidate_depth,
        },
        "system": system_metadata(),
        "cold_start": {
            "milliseconds": round(cold_start_ms, 3),
            "seconds": round(cold_start_ms / 1000, 3),
            "phases": startup_phases,
            "dominant_resource_phase": dominant_startup_resource_phase(startup_phases),
        },
        "warm": {
            "measurement_wall_ms": round(measurement_wall_ms, 3),
            "requests": total_requests,
            "aggregate_sequential_requests_per_second": _throughput(
                total_requests,
                measurement_wall_ms,
            ),
            "modes": mode_reports,
        },
        "memory": memory_report(
            before=memory_before,
            after_startup=memory_after_startup,
            after_benchmark=memory_after_benchmark,
        ),
        "artifacts": artifacts,
        "cost": {
            "external_api_calls": 0,
            "hosted_service_cost_usd": 0.0,
            "note": "Runs locally; excludes host electricity and hardware depreciation.",
        },
    }
    write_json(benchmark.output_path, report)
    return report


def startup_phase_report(
    *,
    name: str,
    elapsed_ms: float,
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, Any]:
    before_rss = int(before["rss_bytes"])
    after_rss = int(after["rss_bytes"])
    before_peak = int(before["peak_rss_bytes"])
    after_peak = max(int(after["peak_rss_bytes"]), after_rss)
    return {
        "name": name,
        "milliseconds": round(elapsed_ms, 3),
        "rss_before": _byte_measure(before_rss),
        "rss_after": _byte_measure(after_rss),
        "retained_rss_delta": _byte_measure(after_rss - before_rss),
        "peak_rss_after": _byte_measure(after_peak),
        "peak_rss_delta": _byte_measure(max(0, after_peak - before_peak)),
    }


def validate_startup_phases(phases: Sequence[Mapping[str, Any]]) -> None:
    names = tuple(str(phase["name"]) for phase in phases)
    expected = ("api_import", *STARTUP_RESOURCE_PHASES)
    if names != expected:
        raise RuntimeError(
            "Serving startup phases did not match the expected order: "
            + ", ".join(expected)
        )


def dominant_startup_resource_phase(
    phases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resources = [
        phase for phase in phases if str(phase["name"]) in STARTUP_RESOURCE_PHASES
    ]
    if not resources:
        raise ValueError("No serving startup resource phases were recorded")
    dominant = max(
        resources,
        key=lambda phase: int(cast(Mapping[str, Any], phase["retained_rss_delta"])["bytes"]),
    )
    positive_total = sum(
        max(
            0,
            int(cast(Mapping[str, Any], phase["retained_rss_delta"])["bytes"]),
        )
        for phase in resources
    )
    dominant_bytes = int(
        cast(Mapping[str, Any], dominant["retained_rss_delta"])["bytes"]
    )
    share = 0.0 if positive_total == 0 else max(0, dominant_bytes) / positive_total
    return {
        "name": str(dominant["name"]),
        "retained_rss_delta": dominant["retained_rss_delta"],
        "share_of_positive_resource_delta": round(share, 4),
    }


def summarize_mode_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero serving measurements")
    handler_values = [float(row["handler_wall_ms"]) for row in rows]
    stage_reports = {
        stage: latency_summary([float(row["latency_ms"][stage]) for row in rows])
        for stage in STAGE_NAMES
    }
    result_counts = [int(row["result_count"]) for row in rows]
    rss_values = [int(row["rss_bytes"]) for row in rows]
    return {
        "requests": len(rows),
        "handler_latency_ms": latency_summary(handler_values),
        "stage_latency_ms": stage_reports,
        "sequential_requests_per_second": _throughput(len(rows), sum(handler_values)),
        "result_count": {
            "minimum": min(result_counts),
            "maximum": max(result_counts),
        },
        "sampled_process_rss": {
            "minimum": _byte_measure(min(rss_values)),
            "mean": _byte_measure(round(sum(rss_values) / len(rss_values))),
            "maximum": _byte_measure(max(rss_values)),
        },
    }


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize zero latency values")
    return {
        "minimum": round(min(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "maximum": round(max(values), 3),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for zero values")
    if not 0 <= quantile <= 1:
        raise ValueError("Percentile quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def process_memory() -> dict[str, int]:
    return {
        "rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def memory_report(
    *,
    before: Mapping[str, int],
    after_startup: Mapping[str, int],
    after_benchmark: Mapping[str, int],
) -> dict[str, Any]:
    before_rss = int(before["rss_bytes"])
    startup_rss = int(after_startup["rss_bytes"])
    benchmark_rss = int(after_benchmark["rss_bytes"])
    peak_rss = max(
        before_rss,
        startup_rss,
        benchmark_rss,
        int(before["peak_rss_bytes"]),
        int(after_startup["peak_rss_bytes"]),
        int(after_benchmark["peak_rss_bytes"]),
    )
    return {
        "before_startup": _byte_measure(before_rss),
        "after_startup": _byte_measure(startup_rss),
        "startup_rss_delta": _byte_measure(startup_rss - before_rss),
        "after_benchmark": _byte_measure(benchmark_rss),
        "peak": _byte_measure(peak_rss),
    }


def serving_artifact_sizes(benchmark: ServingBenchmark) -> dict[str, Any]:
    files = {
        "corpus": _file_measure(benchmark.corpus_path, benchmark.project_root),
        "sqlite_fts_index": _file_measure(
            benchmark.sqlite_fts_index_path,
            benchmark.project_root,
        ),
        "dense_index": _file_measure(benchmark.dense_index_path, benchmark.project_root),
    }
    model_cache_path = huggingface_model_cache_path(benchmark.dense_model_name)
    model_cache = _directory_measure(model_cache_path)
    total_file_bytes = sum(int(record["bytes"]) for record in files.values())
    return {
        "files": files,
        "corpus_and_indexes_total": _byte_measure(total_file_bytes),
        "dense_model_cache": model_cache,
    }


def huggingface_model_cache_path(model_name: str) -> Path:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path.resolve()
    hub_root_value = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_root_value:
        hub_root = Path(hub_root_value).expanduser()
    else:
        hf_home = Path(
            os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        ).expanduser()
        hub_root = hf_home / "hub"
    cache_name = "models--" + model_name.replace("/", "--")
    return (hub_root / cache_name).resolve()


def system_metadata() -> dict[str, Any]:
    packages = {}
    for package in (
        "clinical-trial-retrieval-and-matching",
        "numpy",
        "torch",
        "sentence-transformers",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
    }


@contextmanager
def temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _validate_artifacts_exist(benchmark: ServingBenchmark) -> None:
    missing = [
        path
        for path in (
            benchmark.corpus_path,
            benchmark.sqlite_fts_index_path,
            benchmark.dense_index_path,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Serving benchmark artifact(s) not found: "
            + ", ".join(str(path) for path in missing)
        )


def _elapsed_ms(start: float, clock: Callable[[], float]) -> float:
    return (clock() - start) * 1000


def _throughput(requests: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return round(requests / (elapsed_ms / 1000), 3)


def _current_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return _peak_rss_bytes()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _file_measure(path: Path, project_root: Path) -> dict[str, Any]:
    try:
        label = path.relative_to(project_root).as_posix()
    except ValueError:
        label = str(path)
    return {"path": label, **_byte_measure(path.stat().st_size)}


def _directory_measure(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, **_byte_measure(0)}
    seen_files: set[tuple[int, int]] = set()
    total_bytes = 0
    file_count = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_files:
            continue
        seen_files.add(identity)
        total_bytes += stat.st_size
        file_count += 1
    return {
        "path": str(path),
        "exists": True,
        "files": file_count,
        **_byte_measure(total_bytes),
    }


def _byte_measure(value: int) -> dict[str, int | float]:
    return {"bytes": value, "mib": round(value / (1024 * 1024), 3)}


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _required_string(payload: Mapping[str, Any], key: str, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _string_list(payload: Mapping[str, Any], key: str, prefix: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{prefix}.{key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _positive_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{prefix}{key} must be a positive integer")
    return value


def _non_negative_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{prefix}{key} must be a non-negative integer")
    return value


def _project_path(
    project_root: Path,
    payload: Mapping[str, Any],
    key: str,
    prefix: str,
) -> Path:
    value = _required_string(payload, key, f"{prefix}.")
    resolved = (project_root / value).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{prefix}.{key} cannot resolve outside project_root") from exc
    return resolved


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    label: str,
    allowed: set[str],
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} field(s): {', '.join(unknown)}")
