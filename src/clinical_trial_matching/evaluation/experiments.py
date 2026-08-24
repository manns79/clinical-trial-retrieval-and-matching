from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_trial_matching.retrieval.bm25 import (
    DEFAULT_FIELD_WEIGHTS,
    normalized_field_weights,
)
from clinical_trial_matching.retrieval.dense import ENCODER_BACKENDS, TEXT_REPRESENTATIONS
from clinical_trial_matching.retrieval.sqlite_fts import (
    normalize_sqlite_fts_field_weights,
)

BM25_EXPERIMENT_SCHEMA_VERSION = 1
DENSE_EXPERIMENT_SCHEMA_VERSION = 1
RRF_EXPERIMENT_SCHEMA_VERSION = 1
SQLITE_FTS_EXPERIMENT_SCHEMA_VERSION = 1
CROSS_ENCODER_EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")


@dataclass(frozen=True)
class Bm25Experiment:
    name: str
    description: str
    retriever: str
    field_weights: dict[str, float]
    top_k: int
    trials_path: Path
    topics_path: Path
    qrels_path: Path
    index_path: Path
    run_output_path: Path
    metrics_output_path: Path
    diagnostics_output_path: Path
    config_path: Path
    config_label: str
    config_sha256: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "description": self.description,
            "schema_version": BM25_EXPERIMENT_SCHEMA_VERSION,
            "config_path": self.config_label,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class DenseExperiment:
    name: str
    description: str
    model_name: str
    text_representation: str
    batch_size: int
    device: str
    max_seq_length: int | None
    dynamic_quantization: bool
    encoder_backend: str
    onnx_model_path: Path | None
    top_k: int
    trials_path: Path
    topics_path: Path
    qrels_path: Path
    index_path: Path
    run_output_path: Path
    metrics_output_path: Path
    diagnostics_output_path: Path
    config_path: Path
    config_label: str
    config_sha256: str

    def metadata(self) -> dict[str, str | int | bool]:
        return {
            "name": self.name,
            "description": self.description,
            "schema_version": DENSE_EXPERIMENT_SCHEMA_VERSION,
            "config_path": self.config_label,
            "config_sha256": self.config_sha256,
            "dynamic_quantization": self.dynamic_quantization,
            "encoder_backend": self.encoder_backend,
        }


@dataclass(frozen=True)
class SQLiteFtsExperiment:
    name: str
    description: str
    field_weights: dict[str, float]
    top_k: int
    trials_path: Path
    topics_path: Path
    qrels_path: Path
    index_path: Path
    run_output_path: Path
    metrics_output_path: Path
    diagnostics_output_path: Path
    config_path: Path
    config_label: str
    config_sha256: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "description": self.description,
            "schema_version": SQLITE_FTS_EXPERIMENT_SCHEMA_VERSION,
            "config_path": self.config_label,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class RrfComponent:
    name: str
    run_path: Path
    weight: float


@dataclass(frozen=True)
class RrfExperiment:
    name: str
    description: str
    rrf_k: int
    candidate_depth: int
    top_k: int
    components: tuple[RrfComponent, ...]
    trials_path: Path
    topics_path: Path
    qrels_path: Path
    run_output_path: Path
    metrics_output_path: Path
    diagnostics_output_path: Path
    config_path: Path
    config_label: str
    config_sha256: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "description": self.description,
            "schema_version": RRF_EXPERIMENT_SCHEMA_VERSION,
            "config_path": self.config_label,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class CrossEncoderExperiment:
    name: str
    description: str
    model_name: str
    model_revision: str
    model_file: str
    model_precision: str
    text_representation: str
    batch_size: int
    device: str
    max_length: int
    candidate_depths: tuple[int, ...]
    top_k: int
    corpus_path: Path
    trial_store_path: Path
    topics_path: Path
    qrels_path: Path
    baseline_run_path: Path
    baseline_metrics_path: Path
    model_artifact_path: Path
    serving_config_path: Path
    peak_process_rss_mib: float
    output_directory: Path
    report_output_path: Path
    headroom_output_path: Path
    config_path: Path
    config_label: str
    config_sha256: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "description": self.description,
            "schema_version": CROSS_ENCODER_EXPERIMENT_SCHEMA_VERSION,
            "config_path": self.config_label,
            "config_sha256": self.config_sha256,
        }

    def run_path(self, candidate_depth: int) -> Path:
        return self.output_directory / f"{self.name}_depth_{candidate_depth}.run"

    def metrics_path(self, candidate_depth: int) -> Path:
        return self.output_directory / f"{self.name}_depth_{candidate_depth}_metrics.json"

    def diagnostics_path(self, candidate_depth: int) -> Path:
        return self.output_directory / f"{self.name}_depth_{candidate_depth}_diagnostics.json"


def load_bm25_experiment(path: Path) -> Bm25Experiment:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid experiment JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("BM25 experiment config must contain a JSON object")

    allowed_keys = {
        "schema_version",
        "name",
        "description",
        "project_root",
        "retriever",
        "field_weights",
        "benchmark",
        "artifacts",
    }
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown BM25 experiment config field(s): {', '.join(unknown_keys)}")

    schema_version = payload.get("schema_version")
    if schema_version != BM25_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported BM25 experiment schema_version "
            f"{schema_version!r}; expected {BM25_EXPERIMENT_SCHEMA_VERSION}"
        )

    name = _required_string(payload, "name")
    if EXPERIMENT_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            "Experiment name must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, dots, underscores, or hyphens"
        )
    description = _required_string(payload, "description")

    project_root_value = _required_string(payload, "project_root")
    project_root_path = Path(project_root_value)
    if project_root_path.is_absolute():
        raise ValueError("project_root must be relative to the experiment config")
    project_root = (path.resolve().parent / project_root_path).resolve()

    retriever = _required_string(payload, "retriever")
    if retriever not in {"bm25", "fielded-bm25"}:
        raise ValueError(f"Unsupported BM25 experiment retriever: {retriever!r}")
    field_weights = _field_weights(payload.get("field_weights"), retriever)

    benchmark = _required_mapping(payload, "benchmark")
    _reject_unknown_fields(benchmark, "benchmark", {"trials", "topics", "qrels", "top_k"})
    top_k = benchmark.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("benchmark.top_k must be a positive integer")

    artifacts = _required_mapping(payload, "artifacts")
    _reject_unknown_fields(
        artifacts,
        "artifacts",
        {"index", "run", "metrics", "diagnostics"},
    )

    config_path = path.resolve()
    try:
        config_label = config_path.relative_to(project_root).as_posix()
    except ValueError:
        config_label = config_path.name

    return Bm25Experiment(
        name=name,
        description=description,
        retriever=retriever,
        field_weights=field_weights,
        top_k=top_k,
        trials_path=_project_path(project_root, benchmark, "trials", "benchmark"),
        topics_path=_project_path(project_root, benchmark, "topics", "benchmark"),
        qrels_path=_project_path(project_root, benchmark, "qrels", "benchmark"),
        index_path=_project_path(project_root, artifacts, "index", "artifacts"),
        run_output_path=_project_path(project_root, artifacts, "run", "artifacts"),
        metrics_output_path=_project_path(project_root, artifacts, "metrics", "artifacts"),
        diagnostics_output_path=_project_path(
            project_root, artifacts, "diagnostics", "artifacts"
        ),
        config_path=config_path,
        config_label=config_label,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_dense_experiment(path: Path) -> DenseExperiment:
    raw, payload = _read_experiment_json(path, "Dense")
    allowed_keys = {
        "schema_version",
        "name",
        "description",
        "project_root",
        "model_name",
        "text_representation",
        "batch_size",
        "device",
        "max_seq_length",
        "dynamic_quantization",
        "encoder_backend",
        "onnx_model_path",
        "benchmark",
        "artifacts",
    }
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown dense experiment config field(s): {', '.join(unknown_keys)}")

    schema_version = payload.get("schema_version")
    if schema_version != DENSE_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported dense experiment schema_version "
            f"{schema_version!r}; expected {DENSE_EXPERIMENT_SCHEMA_VERSION}"
        )

    name = _validated_experiment_name(payload)
    description = _required_string(payload, "description")
    project_root = _project_root(path, payload)
    model_name = _required_string(payload, "model_name")
    text_representation = _required_string(payload, "text_representation")
    if text_representation not in TEXT_REPRESENTATIONS:
        raise ValueError(
            f"Unknown dense text representation {text_representation!r}; expected one of "
            + ", ".join(sorted(TEXT_REPRESENTATIONS))
        )
    batch_size = _positive_integer(payload, "batch_size")
    device = _required_string(payload, "device")
    max_seq_length_value = payload.get("max_seq_length")
    if max_seq_length_value is None:
        max_seq_length = None
    elif isinstance(max_seq_length_value, bool) or not isinstance(max_seq_length_value, int):
        raise ValueError("max_seq_length must be a positive integer or null")
    elif max_seq_length_value < 1:
        raise ValueError("max_seq_length must be a positive integer or null")
    else:
        max_seq_length = max_seq_length_value

    benchmark = _required_mapping(payload, "benchmark")
    _reject_unknown_fields(benchmark, "benchmark", {"trials", "topics", "qrels", "top_k"})
    top_k = _positive_integer(benchmark, "top_k", prefix="benchmark.")
    artifacts = _required_mapping(payload, "artifacts")
    _reject_unknown_fields(
        artifacts,
        "artifacts",
        {"index", "run", "metrics", "diagnostics"},
    )

    config_path = path.resolve()
    try:
        config_label = config_path.relative_to(project_root).as_posix()
    except ValueError:
        config_label = config_path.name

    index_path = _project_path(project_root, artifacts, "index", "artifacts")
    if index_path.suffix.lower() not in {".npz", ".mmap"}:
        raise ValueError("artifacts.index must use the .npz or .mmap suffix")

    dynamic_quantization = payload.get("dynamic_quantization", False)
    if not isinstance(dynamic_quantization, bool):
        raise ValueError("dynamic_quantization must be a boolean")
    encoder_backend = payload.get("encoder_backend", "sentence-transformers")
    if not isinstance(encoder_backend, str) or encoder_backend not in ENCODER_BACKENDS:
        raise ValueError(
            "encoder_backend must be one of: " + ", ".join(sorted(ENCODER_BACKENDS))
        )
    onnx_model_path = (
        _project_path(project_root, payload, "onnx_model_path", "config")
        if "onnx_model_path" in payload
        else None
    )
    if encoder_backend == "onnxruntime" and onnx_model_path is None:
        raise ValueError("onnx_model_path is required for the onnxruntime encoder backend")
    if encoder_backend != "onnxruntime" and onnx_model_path is not None:
        raise ValueError("onnx_model_path is only valid for the onnxruntime encoder backend")
    if dynamic_quantization and encoder_backend != "sentence-transformers":
        raise ValueError(
            "dynamic_quantization is only valid for the sentence-transformers backend"
        )

    return DenseExperiment(
        name=name,
        description=description,
        model_name=model_name,
        text_representation=text_representation,
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
        dynamic_quantization=dynamic_quantization,
        encoder_backend=encoder_backend,
        onnx_model_path=onnx_model_path,
        top_k=top_k,
        trials_path=_project_path(project_root, benchmark, "trials", "benchmark"),
        topics_path=_project_path(project_root, benchmark, "topics", "benchmark"),
        qrels_path=_project_path(project_root, benchmark, "qrels", "benchmark"),
        index_path=index_path,
        run_output_path=_project_path(project_root, artifacts, "run", "artifacts"),
        metrics_output_path=_project_path(project_root, artifacts, "metrics", "artifacts"),
        diagnostics_output_path=_project_path(
            project_root, artifacts, "diagnostics", "artifacts"
        ),
        config_path=config_path,
        config_label=config_label,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_sqlite_fts_experiment(path: Path) -> SQLiteFtsExperiment:
    raw, payload = _read_experiment_json(path, "SQLite FTS5")
    _reject_unknown_fields(
        payload,
        "SQLite FTS5 experiment config",
        {
            "schema_version",
            "name",
            "description",
            "project_root",
            "field_weights",
            "benchmark",
            "artifacts",
        },
    )
    if payload.get("schema_version") != SQLITE_FTS_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported SQLite FTS5 experiment schema_version "
            f"{payload.get('schema_version')!r}; expected "
            f"{SQLITE_FTS_EXPERIMENT_SCHEMA_VERSION}"
        )
    name = _validated_experiment_name(payload)
    description = _required_string(payload, "description")
    project_root = _project_root(path, payload)
    raw_weights = payload.get("field_weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("field_weights must be a JSON object")
    field_weights = normalize_sqlite_fts_field_weights(raw_weights)
    benchmark = _required_mapping(payload, "benchmark")
    _reject_unknown_fields(benchmark, "benchmark", {"trials", "topics", "qrels", "top_k"})
    artifacts = _required_mapping(payload, "artifacts")
    _reject_unknown_fields(
        artifacts,
        "artifacts",
        {"index", "run", "metrics", "diagnostics"},
    )
    config_path = path.resolve()
    try:
        config_label = config_path.relative_to(project_root).as_posix()
    except ValueError:
        config_label = config_path.name
    index_path = _project_path(project_root, artifacts, "index", "artifacts")
    if index_path.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        raise ValueError("artifacts.index must use a SQLite file suffix")
    return SQLiteFtsExperiment(
        name=name,
        description=description,
        field_weights=field_weights,
        top_k=_positive_integer(benchmark, "top_k", prefix="benchmark."),
        trials_path=_project_path(project_root, benchmark, "trials", "benchmark"),
        topics_path=_project_path(project_root, benchmark, "topics", "benchmark"),
        qrels_path=_project_path(project_root, benchmark, "qrels", "benchmark"),
        index_path=index_path,
        run_output_path=_project_path(project_root, artifacts, "run", "artifacts"),
        metrics_output_path=_project_path(project_root, artifacts, "metrics", "artifacts"),
        diagnostics_output_path=_project_path(
            project_root, artifacts, "diagnostics", "artifacts"
        ),
        config_path=config_path,
        config_label=config_label,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_cross_encoder_experiment(path: Path) -> CrossEncoderExperiment:
    raw, payload = _read_experiment_json(path, "Cross-encoder")
    _reject_unknown_fields(
        payload,
        "cross-encoder experiment config",
        {
            "schema_version",
            "name",
            "description",
            "project_root",
            "model",
            "reranking",
            "benchmark",
            "serving_headroom",
            "artifacts",
        },
    )
    if payload.get("schema_version") != CROSS_ENCODER_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported cross-encoder experiment schema_version")
    name = _validated_experiment_name(payload)
    description = _required_string(payload, "description")
    project_root = _project_root(path, payload)
    model = _required_mapping(payload, "model")
    _reject_unknown_fields(
        model,
        "model",
        {
            "name",
            "revision",
            "artifact",
            "file",
            "precision",
            "device",
            "max_length",
            "batch_size",
        },
    )
    model_name = _required_string(model, "name")
    model_revision = _required_string(model, "revision")
    model_file = model.get("file", "onnx/model.onnx")
    if (
        not isinstance(model_file, str)
        or not model_file.strip()
        or Path(model_file).is_absolute()
        or ".." in Path(model_file).parts
    ):
        raise ValueError("model.file must be a safe relative path")
    model_file = Path(model_file).as_posix()
    model_precision = model.get("precision", "fp32")
    if model_precision not in {"fp32", "int8"}:
        raise ValueError("model.precision must be fp32 or int8")
    device = _required_string(model, "device")
    if device != "cpu":
        raise ValueError("The local ONNX cross-encoder experiment supports only CPU")
    max_length = _positive_integer(model, "max_length", prefix="model.")
    batch_size = _positive_integer(model, "batch_size", prefix="model.")
    reranking = _required_mapping(payload, "reranking")
    _reject_unknown_fields(
        reranking,
        "reranking",
        {"candidate_depths", "text_representation"},
    )
    text_representation = _required_string(reranking, "text_representation")
    if text_representation not in TEXT_REPRESENTATIONS:
        raise ValueError(f"Unknown cross-encoder text representation: {text_representation}")
    raw_depths = reranking.get("candidate_depths")
    if not isinstance(raw_depths, list) or not raw_depths:
        raise ValueError("reranking.candidate_depths must be a non-empty integer list")
    if any(
        isinstance(depth, bool) or not isinstance(depth, int) or depth < 1
        for depth in raw_depths
    ):
        raise ValueError("reranking.candidate_depths must contain positive integers")
    candidate_depths = tuple(raw_depths)
    if tuple(sorted(set(candidate_depths))) != candidate_depths:
        raise ValueError("reranking.candidate_depths must be unique and ascending")

    benchmark = _required_mapping(payload, "benchmark")
    _reject_unknown_fields(
        benchmark,
        "benchmark",
        {
            "corpus",
            "trial_store",
            "topics",
            "qrels",
            "baseline_run",
            "baseline_metrics",
            "top_k",
        },
    )
    top_k = _positive_integer(benchmark, "top_k", prefix="benchmark.")
    if candidate_depths[-1] > top_k:
        raise ValueError("Cross-encoder candidate depths cannot exceed benchmark.top_k")
    serving = _required_mapping(payload, "serving_headroom")
    _reject_unknown_fields(serving, "serving_headroom", {"config", "peak_process_rss_mib"})
    peak_process_rss_mib = serving.get("peak_process_rss_mib")
    if (
        isinstance(peak_process_rss_mib, bool)
        or not isinstance(peak_process_rss_mib, (int, float))
        or peak_process_rss_mib <= 0
    ):
        raise ValueError("serving_headroom.peak_process_rss_mib must be positive")
    artifacts = _required_mapping(payload, "artifacts")
    _reject_unknown_fields(
        artifacts,
        "artifacts",
        {"output_directory", "report", "headroom_report"},
    )
    config_path = path.resolve()
    try:
        config_label = config_path.relative_to(project_root).as_posix()
    except ValueError:
        config_label = config_path.name
    return CrossEncoderExperiment(
        name=name,
        description=description,
        model_name=model_name,
        model_revision=model_revision,
        model_file=model_file,
        model_precision=model_precision,
        text_representation=text_representation,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        candidate_depths=candidate_depths,
        top_k=top_k,
        corpus_path=_project_path(project_root, benchmark, "corpus", "benchmark"),
        trial_store_path=_project_path(project_root, benchmark, "trial_store", "benchmark"),
        topics_path=_project_path(project_root, benchmark, "topics", "benchmark"),
        qrels_path=_project_path(project_root, benchmark, "qrels", "benchmark"),
        baseline_run_path=_project_path(
            project_root, benchmark, "baseline_run", "benchmark"
        ),
        baseline_metrics_path=_project_path(
            project_root, benchmark, "baseline_metrics", "benchmark"
        ),
        model_artifact_path=_project_path(project_root, model, "artifact", "model"),
        serving_config_path=_project_path(project_root, serving, "config", "serving_headroom"),
        peak_process_rss_mib=float(peak_process_rss_mib),
        output_directory=_project_path(
            project_root, artifacts, "output_directory", "artifacts"
        ),
        report_output_path=_project_path(project_root, artifacts, "report", "artifacts"),
        headroom_output_path=_project_path(
            project_root, artifacts, "headroom_report", "artifacts"
        ),
        config_path=config_path,
        config_label=config_label,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_rrf_experiment(path: Path) -> RrfExperiment:
    raw, payload = _read_experiment_json(path, "RRF")
    allowed_keys = {
        "schema_version",
        "name",
        "description",
        "project_root",
        "rrf_k",
        "candidate_depth",
        "components",
        "benchmark",
        "artifacts",
    }
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown RRF experiment config field(s): {', '.join(unknown_keys)}")
    if payload.get("schema_version") != RRF_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported RRF experiment schema_version "
            f"{payload.get('schema_version')!r}; expected {RRF_EXPERIMENT_SCHEMA_VERSION}"
        )

    name = _validated_experiment_name(payload)
    description = _required_string(payload, "description")
    project_root = _project_root(path, payload)
    rrf_k = _positive_integer(payload, "rrf_k")
    candidate_depth = _positive_integer(payload, "candidate_depth")
    raw_components = payload.get("components")
    if not isinstance(raw_components, list) or len(raw_components) < 2:
        raise ValueError("components must contain at least two RRF component objects")
    components: list[RrfComponent] = []
    for index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, dict):
            raise ValueError(f"components[{index}] must be a JSON object")
        _reject_unknown_fields(raw_component, f"components[{index}]", {"name", "run", "weight"})
        component_name = _required_string(raw_component, "name")
        weight = raw_component.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"components[{index}].weight must be positive")
        components.append(
            RrfComponent(
                name=component_name,
                run_path=_project_path(project_root, raw_component, "run", f"components[{index}]"),
                weight=float(weight),
            )
        )
    if len({component.name for component in components}) != len(components):
        raise ValueError("RRF component names must be unique")

    benchmark = _required_mapping(payload, "benchmark")
    _reject_unknown_fields(benchmark, "benchmark", {"trials", "topics", "qrels", "top_k"})
    artifacts = _required_mapping(payload, "artifacts")
    _reject_unknown_fields(artifacts, "artifacts", {"run", "metrics", "diagnostics"})
    config_path = path.resolve()
    try:
        config_label = config_path.relative_to(project_root).as_posix()
    except ValueError:
        config_label = config_path.name

    return RrfExperiment(
        name=name,
        description=description,
        rrf_k=rrf_k,
        candidate_depth=candidate_depth,
        top_k=_positive_integer(benchmark, "top_k", prefix="benchmark."),
        components=tuple(components),
        trials_path=_project_path(project_root, benchmark, "trials", "benchmark"),
        topics_path=_project_path(project_root, benchmark, "topics", "benchmark"),
        qrels_path=_project_path(project_root, benchmark, "qrels", "benchmark"),
        run_output_path=_project_path(project_root, artifacts, "run", "artifacts"),
        metrics_output_path=_project_path(project_root, artifacts, "metrics", "artifacts"),
        diagnostics_output_path=_project_path(
            project_root, artifacts, "diagnostics", "artifacts"
        ),
        config_path=config_path,
        config_label=config_label,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _field_weights(value: Any, retriever: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("field_weights must be a JSON object")

    weights: dict[str, float] = {}
    for field_name, weight in value.items():
        if not isinstance(field_name, str):
            raise ValueError("field_weights keys must be strings")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(f"Field weight {field_name!r} must be numeric")
        weights[field_name] = float(weight)

    if retriever == "bm25":
        if weights:
            raise ValueError("Plain bm25 experiments must use an empty field_weights object")
        return {}

    missing_fields = sorted(set(DEFAULT_FIELD_WEIGHTS) - set(weights))
    if missing_fields:
        raise ValueError(
            "Fielded BM25 experiment configs must pin every field weight; missing: "
            + ", ".join(missing_fields)
        )
    return normalized_field_weights(weights)


def _read_experiment_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid experiment JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} experiment config must contain a JSON object")
    return raw, payload


def _validated_experiment_name(payload: dict[str, Any]) -> str:
    name = _required_string(payload, "name")
    if EXPERIMENT_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            "Experiment name must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, dots, underscores, or hyphens"
        )
    return name


def _project_root(path: Path, payload: dict[str, Any]) -> Path:
    project_root_value = _required_string(payload, "project_root")
    project_root_path = Path(project_root_value)
    if project_root_path.is_absolute():
        raise ValueError("project_root must be relative to the experiment config")
    return (path.resolve().parent / project_root_path).resolve()


def _positive_integer(
    payload: dict[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{prefix}{key} must be a positive integer")
    return value


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _reject_unknown_fields(payload: dict[str, Any], label: str, allowed: set[str]) -> None:
    unknown_fields = sorted(set(payload) - allowed)
    if unknown_fields:
        raise ValueError(f"Unknown {label} field(s): {', '.join(unknown_fields)}")


def _project_path(
    project_root: Path,
    payload: dict[str, Any],
    key: str,
    label: str,
) -> Path:
    value = _required_string(payload, key)
    relative_path = Path(value)
    if relative_path.is_absolute():
        raise ValueError(f"{label}.{key} must be relative to project_root")
    resolved = (project_root / relative_path).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"{label}.{key} cannot resolve outside project_root")
    return resolved
