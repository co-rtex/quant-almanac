"""JSON views over registry rows used by the governance API and console.

These functions only reshape persisted records. They never fill in missing history:
absent metadata is surfaced as ``None`` so callers can render it as unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from quant_signal.governance.statistics import hanley_mcneil_interval
from quant_signal.storage.models import (
    BacktestRun,
    DatasetVersion,
    IngestionRun,
    ModelEvaluation,
    ModelVersion,
    QualityReportRecord,
    ShapRun,
)


def iso(value: date | datetime | None) -> str | None:
    """Return an ISO-8601 string or ``None``."""

    return value.isoformat() if value is not None else None


def section(payload: Mapping[str, Any] | None, key: str) -> dict[str, Any] | None:
    """Return a nested mapping when present, else ``None`` (never an invented default)."""

    if not payload:
        return None
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else None


def float_or_none(value: object) -> float | None:
    """Return a float for numeric values, else ``None``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def dataset_json(dataset: DatasetVersion) -> dict[str, Any]:
    """Serialize a dataset manifest."""

    metadata = dataset.metadata_json or {}
    lineage = section(metadata, "lineage")
    source_bars = section(lineage, "source_bars")
    return {
        "dataset_version_id": dataset.id,
        "as_of_date": iso(dataset.as_of_date),
        "feature_set_version": dataset.feature_set_version,
        "horizons": list(dataset.horizons),
        "symbols": list(dataset.symbols),
        "row_count": dataset.row_count,
        "feature_columns": list(dataset.feature_columns),
        "label_columns": list(dataset.label_columns),
        "artifact_path": dataset.artifact_path,
        "artifact_sha256": dataset.artifact_hash,
        "date_range": section(metadata, "date_range"),
        "benchmark_symbol": metadata.get("benchmark_symbol"),
        "raw_row_count": metadata.get("raw_row_count"),
        "lineage_recorded": lineage is not None,
        "source_bars": source_bars,
        "code": section(lineage, "code"),
        "runtime": section(lineage, "runtime"),
        "quality": section(metadata, "quality"),
        "created_at": iso(dataset.created_at),
    }


def model_json(model: ModelVersion) -> dict[str, Any]:
    """Serialize a model registry row."""

    metadata = model.metadata_json or {}
    return {
        "model_version_id": model.id,
        "dataset_version_id": model.dataset_version_id,
        "horizon_days": model.horizon_days,
        "model_family": model.model_family,
        "target_column": model.target_column,
        "status": model.status,
        "champion_rank": model.champion_rank,
        "artifact_path": model.artifact_path,
        "artifact_sha256": model.artifact_hash,
        "feature_columns": list(model.feature_columns),
        "windows": {
            "train": {"start": iso(model.train_start_date), "end": iso(model.train_end_date)},
            "validation": {
                "start": iso(model.validation_start_date),
                "end": iso(model.validation_end_date),
            },
            "test": {"start": iso(model.test_start_date), "end": iso(model.test_end_date)},
        },
        "trained_at": metadata.get("trained_at"),
        "training_config": section(metadata, "training_config"),
        "governance": section(metadata, "governance"),
        "provenance": section(metadata, "provenance"),
        "created_at": iso(model.created_at),
    }


def ingestion_run_json(run: IngestionRun) -> dict[str, Any]:
    """Serialize an ingestion run with its provider diagnostics."""

    metadata = run.metadata_json or {}
    provider = section(metadata, "provider") or {}
    provider_metadata = section(provider, "metadata") or {}
    fetch = section(metadata, "provider_fetch") or {}
    return {
        "ingestion_run_id": run.id,
        "provider": run.provider,
        "status": run.status,
        "start_date": iso(run.start_date),
        "end_date": iso(run.end_date),
        "requested_symbols": list(run.requested_symbols),
        "records_written": run.records_written,
        "started_at": iso(run.started_at),
        "completed_at": iso(run.completed_at),
        "synthetic": bool(provider_metadata.get("synthetic")) or run.provider == "synthetic",
        "provider_metadata": provider_metadata,
        "missing_symbols": fetch.get("missing_symbols"),
        "warnings": fetch.get("warnings") or [],
    }


def quality_report_json(
    report: QualityReportRecord, include_context: bool = True
) -> dict[str, Any]:
    """Serialize a persisted quality report."""

    payload: dict[str, Any] = {
        "quality_report_id": report.id,
        "stage": report.stage,
        "status": report.status,
        "policy_version": report.policy_version,
        "dataset_version_id": report.dataset_version_id,
        "model_version_id": report.model_version_id,
        "horizon_days": report.horizon_days,
        "check_count": report.check_count,
        "failure_count": report.failure_count,
        "warning_count": report.warning_count,
        "summary": report.summary,
        "checks": list(report.checks_json),
        "request": dict(report.request_json),
        "created_at": iso(report.created_at),
    }
    if include_context:
        payload["context"] = dict(report.context_json)
    return payload


def evaluation_json(evaluation: ModelEvaluation) -> dict[str, Any]:
    """Serialize an evaluation split with an approximate AUC interval."""

    metrics = evaluation.metrics_json or {}
    sample_count = metrics.get("sample_count")
    interval = hanley_mcneil_interval(
        evaluation.roc_auc,
        int(sample_count) if isinstance(sample_count, (int, float)) else None,
        float_or_none(metrics.get("positive_rate")),
    )
    return {
        "split_name": evaluation.split_name,
        "roc_auc": evaluation.roc_auc,
        "pr_auc": evaluation.pr_auc,
        "brier_score": evaluation.brier_score,
        "calibration_error": evaluation.calibration_error,
        "positive_rate": float_or_none(metrics.get("positive_rate")),
        "sample_count": int(sample_count) if isinstance(sample_count, (int, float)) else None,
        "calibration_bins": list(evaluation.calibration_bins),
        "roc_auc_interval": interval.to_json() if interval else None,
    }


def backtest_json(run: BacktestRun) -> dict[str, Any]:
    """Serialize a backtest run's assumptions, headline results, and lineage."""

    summary = run.summary_json or {}
    metadata = run.metadata_json or {}
    return {
        "backtest_run_id": run.id,
        "status": run.status,
        "horizon_days": run.horizon_days,
        "top_n": run.top_n,
        "min_training_days": run.min_training_days,
        "artifact_path": run.artifact_path,
        "artifact_sha256": run.artifact_hash,
        "detail_artifact_path": metadata.get("detail_artifact_path"),
        "detail_artifact_sha256": metadata.get("detail_artifact_hash"),
        "execution_assumptions": section(metadata, "execution_assumptions"),
        "walk_forward_guard": section(metadata, "walk_forward_guard"),
        "lineage": section(metadata, "lineage"),
        "signal_count": metadata.get("signal_count"),
        "benchmark_symbol": metadata.get("benchmark_symbol"),
        "analytics_version": metadata.get("backtest_analytics_version"),
        "metrics": {
            key: summary.get(key)
            for key in (
                "cumulative_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "hit_rate",
                "total_cost_drag",
            )
        },
        "benchmark_metrics": section(summary, "benchmark_metrics"),
        "active_metrics": section(summary, "active_metrics"),
        "turnover_metrics": section(summary, "turnover_metrics"),
        "created_at": iso(run.created_at),
    }


def shap_json(run: ShapRun) -> dict[str, Any]:
    """Serialize a SHAP run."""

    summary = run.summary_json or {}
    importance = summary.get("global_importance")
    local = summary.get("local_explanations")
    return {
        "shap_run_id": run.id,
        "sample_size": run.sample_size,
        "artifact_path": run.artifact_path,
        "artifact_sha256": run.artifact_hash,
        "global_importance": list(importance) if isinstance(importance, list) else [],
        "local_explanations": list(local) if isinstance(local, list) else [],
        "methodology": section(summary, "methodology"),
        "lineage": section(summary, "lineage"),
        "created_at": iso(run.created_at),
    }
