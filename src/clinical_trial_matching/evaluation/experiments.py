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

BM25_EXPERIMENT_SCHEMA_VERSION = 1
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
