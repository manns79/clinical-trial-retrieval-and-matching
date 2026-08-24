from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_trial_matching.io import read_json, write_json

CROSS_ENCODER_COMPARISON_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CrossEncoderReportSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class CrossEncoderComparison:
    name: str
    description: str
    candidate_depth: int
    baseline: CrossEncoderReportSpec
    candidates: tuple[CrossEncoderReportSpec, ...]
    eligible_ndcg_tolerance: float
    broad_ndcg_tolerance: float
    hybrid_p95_ms: float
    reranked_p95_budget_ms: float
    report_path: Path
    table_path: Path
    config_label: str
    config_sha256: str


def load_cross_encoder_comparison(path: Path) -> CrossEncoderComparison:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid cross-encoder comparison JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Cross-encoder comparison config must be a JSON object")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "name",
            "description",
            "project_root",
            "candidate_depth",
            "baseline",
            "candidates",
            "quality_gate",
            "latency_budget",
            "artifacts",
        },
        "comparison config",
    )
    if payload.get("schema_version") != CROSS_ENCODER_COMPARISON_SCHEMA_VERSION:
        raise ValueError("Unsupported cross-encoder comparison schema_version")
    project_root_value = _required_string(payload, "project_root")
    project_root = (path.parent / project_root_value).resolve()
    candidate_depth = _positive_integer(payload, "candidate_depth")
    baseline = _report_spec(payload.get("baseline"), project_root, "baseline")
    candidates_value = payload.get("candidates")
    if not isinstance(candidates_value, list) or not candidates_value:
        raise ValueError("candidates must be a non-empty list")
    candidates = tuple(
        _report_spec(value, project_root, f"candidates[{index}]")
        for index, value in enumerate(candidates_value)
    )
    labels = [baseline.label, *(candidate.label for candidate in candidates)]
    if len(set(labels)) != len(labels):
        raise ValueError("Cross-encoder comparison labels must be unique")

    quality = _required_mapping(payload, "quality_gate")
    _reject_unknown(
        quality,
        {"eligible_ndcg_at_10_tolerance", "broad_ndcg_at_10_tolerance"},
        "quality_gate",
    )
    eligible_tolerance = _nonnegative_number(
        quality, "eligible_ndcg_at_10_tolerance", "quality_gate"
    )
    broad_tolerance = _nonnegative_number(
        quality, "broad_ndcg_at_10_tolerance", "quality_gate"
    )
    latency = _required_mapping(payload, "latency_budget")
    _reject_unknown(
        latency,
        {"hybrid_p95_ms", "reranked_mode_p95_ms"},
        "latency_budget",
    )
    hybrid_p95_ms = _positive_number(latency, "hybrid_p95_ms", "latency_budget")
    reranked_p95_ms = _positive_number(
        latency, "reranked_mode_p95_ms", "latency_budget"
    )
    if reranked_p95_ms <= hybrid_p95_ms:
        raise ValueError("Reranked-mode p95 budget must exceed the hybrid p95 allowance")
    artifacts = _required_mapping(payload, "artifacts")
    _reject_unknown(artifacts, {"report", "table"}, "artifacts")
    config_path = path.resolve()
    try:
        config_label = config_path.relative_to(project_root).as_posix()
    except ValueError:
        config_label = config_path.name
    return CrossEncoderComparison(
        name=_required_string(payload, "name"),
        description=_required_string(payload, "description"),
        candidate_depth=candidate_depth,
        baseline=baseline,
        candidates=candidates,
        eligible_ndcg_tolerance=eligible_tolerance,
        broad_ndcg_tolerance=broad_tolerance,
        hybrid_p95_ms=hybrid_p95_ms,
        reranked_p95_budget_ms=reranked_p95_ms,
        report_path=_project_path(project_root, artifacts, "report", "artifacts"),
        table_path=_project_path(project_root, artifacts, "table", "artifacts"),
        config_label=config_label,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_cross_encoder_comparison(
    comparison: CrossEncoderComparison,
) -> dict[str, Any]:
    baseline_report = _profile_report(
        comparison.baseline,
        candidate_depth=comparison.candidate_depth,
    )
    baseline_eligible_gain = baseline_report["eligible_ndcg_at_10_gain"]
    baseline_broad_gain = baseline_report["broad_ndcg_at_10_gain"]
    rows = []
    for spec in (comparison.baseline, *comparison.candidates):
        row = _profile_report(spec, candidate_depth=comparison.candidate_depth)
        row["eligible_gain_change_vs_reference"] = round(
            row["eligible_ndcg_at_10_gain"] - baseline_eligible_gain,
            6,
        )
        row["broad_gain_change_vs_reference"] = round(
            row["broad_ndcg_at_10_gain"] - baseline_broad_gain,
            6,
        )
        row["estimated_reranked_p95_ms"] = round(
            comparison.hybrid_p95_ms + row["reranker_p95_ms"],
            3,
        )
        row["quality_gate_passed"] = (
            row["eligible_ndcg_at_10_gain"] + comparison.eligible_ndcg_tolerance
            >= baseline_eligible_gain
            and row["broad_ndcg_at_10_gain"] + comparison.broad_ndcg_tolerance
            >= baseline_broad_gain
        )
        row["latency_gate_passed"] = (
            row["estimated_reranked_p95_ms"] <= comparison.reranked_p95_budget_ms
        )
        row["adoption_gate_passed"] = (
            row["quality_gate_passed"] and row["latency_gate_passed"]
        )
        rows.append(row)
    passing_candidates = [
        row["label"]
        for row in rows[1:]
        if row["adoption_gate_passed"]
    ]
    report = {
        "schema_version": CROSS_ENCODER_COMPARISON_SCHEMA_VERSION,
        "comparison": {
            "name": comparison.name,
            "description": comparison.description,
            "config_path": comparison.config_label,
            "config_sha256": comparison.config_sha256,
            "scope": "Development topics only; holdout topics are not read.",
            "candidate_depth": comparison.candidate_depth,
            "baseline_label": comparison.baseline.label,
        },
        "gates": {
            "quality": {
                "eligible_ndcg_at_10_reference_gain": baseline_eligible_gain,
                "eligible_ndcg_at_10_tolerance": comparison.eligible_ndcg_tolerance,
                "broad_ndcg_at_10_reference_gain": baseline_broad_gain,
                "broad_ndcg_at_10_tolerance": comparison.broad_ndcg_tolerance,
            },
            "latency": {
                "hybrid_p95_allowance_ms": comparison.hybrid_p95_ms,
                "reranked_mode_p95_budget_ms": comparison.reranked_p95_budget_ms,
                "incremental_reranker_p95_budget_ms": round(
                    comparison.reranked_p95_budget_ms - comparison.hybrid_p95_ms,
                    3,
                ),
            },
        },
        "rows": rows,
        "passing_candidates": passing_candidates,
        "serving_candidate_selected": passing_candidates[0] if passing_candidates else None,
        "cost": {"hosted_service_cost_usd": 0.0, "external_api_calls": 0},
    }
    write_json(comparison.report_path, report)
    _write_markdown(comparison.table_path, report)
    return report


def _profile_report(spec: CrossEncoderReportSpec, *, candidate_depth: int) -> dict[str, Any]:
    report = read_json(spec.path)
    if not isinstance(report, dict):
        raise ValueError(f"Cross-encoder report must be a JSON object: {spec.path}")
    depth = report.get("depths", {}).get(str(candidate_depth))
    if not isinstance(depth, dict):
        raise ValueError(f"Report does not contain candidate depth {candidate_depth}: {spec.path}")
    model = report.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Report does not contain model metadata: {spec.path}")
    deltas = depth["metric_deltas"]["metrics"]
    latency = depth["latency_ms_per_topic"]["total"]
    return {
        "label": spec.label,
        "report": str(spec.path),
        "precision": str(model.get("precision", "fp32")),
        "max_length": int(model["max_length"]),
        "text_representation": str(model["text_representation"]),
        "artifact_mib": round(float(model["artifact_bytes"]) / (1024 * 1024), 3),
        "eligible_ndcg_at_10_gain": round(
            float(deltas["eligible_only"]["ndcg_at_10"]), 6
        ),
        "broad_ndcg_at_10_gain": round(
            float(deltas["excluded_or_eligible"]["ndcg_at_10"]), 6
        ),
        "eligible_mrr_gain": round(float(deltas["eligible_only"]["mrr"]), 6),
        "reranker_mean_ms": round(float(latency["mean"]), 3),
        "reranker_p50_ms": round(float(latency["p50"]), 3),
        "reranker_p95_ms": round(float(latency["p95"]), 3),
        "process_peak_rss_mib": round(float(report["process_peak_rss_mib"]), 3),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    columns = (
        "label",
        "precision",
        "max_length",
        "text_representation",
        "eligible_ndcg_at_10_gain",
        "broad_ndcg_at_10_gain",
        "reranker_p95_ms",
        "estimated_reranked_p95_ms",
        "artifact_mib",
        "process_peak_rss_mib",
        "quality_gate_passed",
        "latency_gate_passed",
        "adoption_gate_passed",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in report["rows"]:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_spec(value: Any, project_root: Path, prefix: str) -> CrossEncoderReportSpec:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must be an object")
    _reject_unknown(value, {"label", "report"}, prefix)
    return CrossEncoderReportSpec(
        label=_required_string(value, "label", prefix),
        path=_project_path(project_root, value, "report", prefix),
    )


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        label = f"{prefix}.{key}" if prefix else key
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_number(payload: dict[str, Any], key: str, prefix: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{prefix}.{key} must be positive")
    return float(value)


def _nonnegative_number(payload: dict[str, Any], key: str, prefix: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{prefix}.{key} must be nonnegative")
    return float(value)


def _project_path(
    project_root: Path,
    payload: dict[str, Any],
    key: str,
    prefix: str,
) -> Path:
    value = _required_string(payload, key, prefix)
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _reject_unknown(payload: dict[str, Any], allowed: set[str], prefix: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {prefix} field(s): {', '.join(unknown)}")
