from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_trial_matching.io import read_json, write_json

DEFAULT_METRIC_COLUMNS = (
    "precision_at_10",
    "recall_at_100",
    "mrr",
    "ndcg_at_10",
    "ndcg_at_100",
)
DEFAULT_METADATA_COLUMNS = (
    "label",
    "run_name",
    "retriever",
    "view",
    "top_k",
    "topics",
    "trials",
    "run_rows",
)


@dataclass(frozen=True)
class MetricsSpec:
    label: str | None
    path: Path


def parse_metrics_spec(value: str) -> MetricsSpec:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid metrics spec {value!r}; label cannot be empty")
        if not path.strip():
            raise ValueError(f"Invalid metrics spec {value!r}; path cannot be empty")
        return MetricsSpec(label=label, path=Path(path))
    return MetricsSpec(label=None, path=Path(value))


def build_metrics_comparison(
    specs: list[MetricsSpec],
    *,
    views: list[str] | None = None,
) -> dict[str, Any]:
    if not specs:
        raise ValueError("At least one metrics file is required")
    selected_views = set(views or [])
    rows: list[dict[str, Any]] = []
    for spec in specs:
        report = read_json(spec.path)
        if not isinstance(report, dict):
            raise ValueError(f"Metrics report was not a JSON object: {spec.path}")
        rows.extend(_rows_for_report(report, spec=spec, selected_views=selected_views))
    return {
        "reports": len(specs),
        "rows": rows,
        "columns": list(DEFAULT_METADATA_COLUMNS + DEFAULT_METRIC_COLUMNS),
    }


def write_metrics_comparison(path: Path, comparison: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        write_json(path, comparison)
    elif output_format == "csv":
        _write_csv(path, comparison["rows"])
    elif output_format == "markdown":
        _write_markdown(path, comparison["rows"])
    else:
        raise ValueError(f"Unsupported comparison output format: {output_format}")


def infer_comparison_format(path: Path, output_format: str | None) -> str:
    if output_format:
        return output_format
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "markdown"


def _rows_for_report(
    report: dict[str, Any],
    *,
    spec: MetricsSpec,
    selected_views: set[str],
) -> list[dict[str, Any]]:
    run_name = str(report.get("run_name", spec.path.stem))
    label = spec.label or run_name
    views = _metric_views(report)
    if selected_views:
        views = {name: metrics for name, metrics in views.items() if name in selected_views}
    rows: list[dict[str, Any]] = []
    for view_name, metrics in views.items():
        rows.append(
            {
                "label": label,
                "run_name": run_name,
                "retriever": str(report.get("retriever", "")),
                "view": view_name,
                "top_k": report.get("top_k", ""),
                "topics": report.get("topics", ""),
                "trials": report.get("trials", ""),
                "run_rows": report.get("run_rows", ""),
                **{metric: _metric_value(metrics, metric) for metric in DEFAULT_METRIC_COLUMNS},
            }
        )
    return rows


def _metric_views(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = report.get("metrics", {})
    views: dict[str, dict[str, Any]] = {}
    if isinstance(metrics, dict) and all(isinstance(value, dict) for value in metrics.values()):
        views.update({str(name): value for name, value in metrics.items()})
    elif isinstance(metrics, dict):
        views["metrics"] = metrics
    graded_metrics = report.get("graded_metrics")
    if isinstance(graded_metrics, dict):
        views["graded"] = graded_metrics
    if not views:
        raise ValueError(f"Metrics report does not contain recognized metrics: {report.get('run_name', '')}")
    return views


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float | str:
    value = metrics.get(metric_name, "")
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEFAULT_METADATA_COLUMNS + DEFAULT_METRIC_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(DEFAULT_METADATA_COLUMNS + DEFAULT_METRIC_COLUMNS)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_value(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
