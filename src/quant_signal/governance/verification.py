"""Verification checks and lineage-stage summaries for registered models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import joblib
import pandas as pd

from quant_signal.core.provenance import provenance_snapshot, runtime_environment
from quant_signal.features.pipeline import fingerprint_source_bars
from quant_signal.governance.serialization import iso
from quant_signal.quality import (
    CheckResult,
    CheckStatus,
    QualityReport,
    check_artifact_integrity,
    worst_status,
)
from quant_signal.storage.models import (
    BacktestRun,
    ModelEvaluation,
    ModelVersion,
    QualityReportRecord,
    ShapRun,
)


def quality_status(dataset_view: dict[str, Any] | None) -> str | None:
    quality = dataset_view.get("quality") if dataset_view else None
    if isinstance(quality, dict) and isinstance(quality.get("status"), str):
        return str(quality["status"])
    return None


def ingestion_run_ids(dataset_view: dict[str, Any]) -> list[str]:
    source_bars = dataset_view.get("source_bars")
    if not isinstance(source_bars, dict):
        return []
    run_ids = source_bars.get("ingestion_run_ids")
    return [str(run_id) for run_id in run_ids] if isinstance(run_ids, list) else []


def is_synthetic_dataset(dataset_view: dict[str, Any] | None) -> bool:
    if not dataset_view:
        return False
    source_bars = dataset_view.get("source_bars")
    sources = source_bars.get("sources") if isinstance(source_bars, dict) else None
    return isinstance(sources, list) and "synthetic" in sources


def candidate_json(model: ModelVersion, evaluations: Sequence[ModelEvaluation]) -> dict[str, Any]:
    by_split = {evaluation.split_name: evaluation for evaluation in evaluations}
    validation = by_split.get("validation")
    test = by_split.get("test")
    return {
        "model_version_id": model.id,
        "model_family": model.model_family,
        "champion_rank": model.champion_rank,
        "validation_pr_auc": validation.pr_auc if validation else None,
        "validation_brier_score": validation.brier_score if validation else None,
        "validation_roc_auc": validation.roc_auc if validation else None,
        "test_roc_auc": test.roc_auc if test else None,
    }


def bundle_consistency_check(model: ModelVersion, integrity: CheckResult) -> CheckResult:
    if integrity.status != "pass":
        return CheckResult(
            check_id="model.bundle_consistency",
            category="lineage",
            status="warn" if integrity.status == "warn" else "fail",
            message="Bundle contents not inspected because artifact integrity did not pass.",
        )
    bundle = joblib.load(model.artifact_path)
    expected = {
        "dataset_version_id": model.dataset_version_id,
        "horizon_days": model.horizon_days,
        "model_family": model.model_family,
        "target_column": model.target_column,
        "feature_columns": list(model.feature_columns),
    }
    actual = {key: getattr(bundle, key, None) for key in expected}
    mismatched = [key for key in expected if expected[key] != actual[key]]
    return CheckResult(
        check_id="model.bundle_consistency",
        category="lineage",
        status="fail" if mismatched else "pass",
        message=(
            f"Serialized bundle disagrees with the registry on: {', '.join(mismatched)}."
            if mismatched
            else "Serialized bundle names the same dataset, horizon, family, target, and "
            "features as the registry."
        ),
        observed={"mismatched_fields": mismatched},
    )


def source_fingerprint_check(
    dataset_view: dict[str, Any],
    current_bars: pd.DataFrame | None,
) -> CheckResult:
    source_bars = dataset_view.get("source_bars")
    recorded = source_bars.get("fingerprint_sha256") if isinstance(source_bars, dict) else None
    if not recorded:
        return CheckResult(
            check_id="lineage.source_bar_fingerprint",
            category="lineage",
            status="warn",
            message="Unavailable: the dataset predates source-bar fingerprinting.",
            observed={"recorded_fingerprint": None},
        )
    if current_bars is None or current_bars.empty:
        return CheckResult(
            check_id="lineage.source_bar_fingerprint",
            category="lineage",
            status="warn",
            message="The source bars this dataset was built from are no longer in the database.",
            observed={"recorded_fingerprint": recorded, "current_fingerprint": None},
        )
    current = fingerprint_source_bars(current_bars)
    matches = current == recorded
    return CheckResult(
        check_id="lineage.source_bar_fingerprint",
        category="lineage",
        status="pass" if matches else "warn",
        message=(
            "Persisted source bars still match the fingerprint recorded at build time."
            if matches
            else "Source bars changed after the build (re-ingestion or backfill). The dataset "
            "artifact is intact, but rebuilding it from the database would differ."
        ),
        observed={
            "recorded_fingerprint": recorded,
            "current_fingerprint": current,
            "current_row_count": int(len(current_bars)),
            "recorded_row_count": source_bars.get("row_count")
            if isinstance(source_bars, dict)
            else None,
        },
    )


def ingestion_lineage_check(run_ids: Sequence[str], runs: Sequence[Any]) -> CheckResult:
    if not run_ids:
        return CheckResult(
            check_id="lineage.ingestion_runs",
            category="lineage",
            status="warn",
            message="Unavailable: no ingestion runs were recorded for this dataset.",
        )
    found = {run.id: run.status for run in runs}
    missing = [run_id for run_id in run_ids if run_id not in found]
    not_completed = [run_id for run_id, status in found.items() if status != "completed"]
    status: CheckStatus = "warn" if (missing or not_completed) else "pass"
    return CheckResult(
        check_id="lineage.ingestion_runs",
        category="lineage",
        status=status,
        message=(
            f"{len(missing)} recorded ingestion runs are missing and {len(not_completed)} "
            "did not complete."
            if status == "warn"
            else f"All {len(run_ids)} recorded ingestion runs exist and completed."
        ),
        observed={"recorded": list(run_ids), "missing": missing, "not_completed": not_completed},
    )


def report_gate_check(
    check_id: str,
    label: str,
    reports: Sequence[QualityReportRecord],
) -> CheckResult:
    if not reports:
        return CheckResult(
            check_id=check_id,
            category="governance",
            status="warn",
            message=f"Unavailable: no {label} quality report was recorded.",
        )
    report = reports[0]
    return CheckResult(
        check_id=check_id,
        category="governance",
        status=worst_status([report.status]),
        message=f"{label.capitalize()} gate: {report.summary}.",
        observed={"quality_report_id": report.id, "status": report.status},
    )


def evaluation_presence_check(evaluations: Sequence[ModelEvaluation]) -> CheckResult:
    splits = sorted({evaluation.split_name for evaluation in evaluations})
    missing = [name for name in ("validation", "test") if name not in splits]
    return CheckResult(
        check_id="evaluation.splits_recorded",
        category="evaluation",
        status="fail" if missing else "pass",
        message=(
            f"Missing persisted evaluations for: {', '.join(missing)}."
            if missing
            else "Validation and test evaluations are persisted."
        ),
        observed={"splits": splits},
    )


def backtest_integrity_check(backtests: Sequence[BacktestRun]) -> CheckResult:
    results: list[dict[str, object]] = []
    for run in backtests:
        metadata = run.metadata_json or {}
        primary = check_artifact_integrity(run.artifact_path, run.artifact_hash, "backtest")
        detail = check_artifact_integrity(
            cast(str | None, metadata.get("detail_artifact_path")),
            cast(str | None, metadata.get("detail_artifact_hash")),
            "backtest detail",
        )
        results.append(
            {"backtest_run_id": run.id, "primary": primary.status, "detail": detail.status}
        )
    status = worst_status(
        [str(row["primary"]) for row in results] + [str(row["detail"]) for row in results]
    )
    return CheckResult(
        check_id="artifact.backtest_integrity",
        category="artifact",
        status=status,
        message=(
            f"All {len(backtests)} backtest runs match their registered artifact hashes."
            if status == "pass"
            else "At least one backtest artifact is missing or modified."
        ),
        observed={"runs": results},
    )


def shap_integrity_check(shap_runs: Sequence[ShapRun]) -> CheckResult:
    results = [
        {
            "shap_run_id": run.id,
            "status": check_artifact_integrity(
                run.artifact_path, run.artifact_hash, "explainability"
            ).status,
        }
        for run in shap_runs
    ]
    status = worst_status([str(row["status"]) for row in results])
    return CheckResult(
        check_id="artifact.explainability_integrity",
        category="artifact",
        status=status,
        message=(
            f"All {len(shap_runs)} SHAP artifacts match their registered hashes."
            if status == "pass"
            else "At least one SHAP artifact is missing or modified (legacy runs shared one "
            "path per model and could be overwritten)."
        ),
        observed={"runs": results},
    )


def recorded_provenance(model: ModelVersion) -> dict[str, Any] | None:
    provenance = (model.metadata_json or {}).get("provenance")
    return provenance if isinstance(provenance, dict) else None


def code_revision_check(model: ModelVersion) -> CheckResult:
    provenance = recorded_provenance(model)
    code = provenance.get("code") if provenance else None
    commit = code.get("commit") if isinstance(code, dict) else None
    dirty = code.get("dirty") if isinstance(code, dict) else None
    if not commit:
        return CheckResult(
            check_id="provenance.code_revision",
            category="reproducibility",
            status="warn",
            message="Unavailable: no source revision was recorded when the model was trained.",
        )
    return CheckResult(
        check_id="provenance.code_revision",
        category="reproducibility",
        status="warn" if dirty else "pass",
        message=(
            f"Trained from commit {str(commit)[:12]} with uncommitted changes in the tree."
            if dirty
            else f"Trained from commit {str(commit)[:12]}."
        ),
        observed={"commit": commit, "dirty": dirty, "source": code.get("source") if code else None},
    )


def runtime_match_check(
    model: ModelVersion,
    check_id: str = "provenance.runtime_versions",
) -> CheckResult:
    provenance = recorded_provenance(model)
    recorded = provenance.get("runtime") if provenance else None
    if not isinstance(recorded, dict):
        return CheckResult(
            check_id=check_id,
            category="reproducibility",
            status="warn",
            message="Unavailable: library versions were not recorded at training time.",
        )
    current = runtime_environment()
    recorded_packages = recorded.get("packages") or {}
    current_packages = cast(dict[str, object], current.get("packages") or {})
    differences = {
        name: {"recorded": recorded_packages.get(name), "current": current_packages.get(name)}
        for name in sorted(set(recorded_packages) | set(current_packages))
        if recorded_packages.get(name) != current_packages.get(name)
    }
    if recorded.get("python") != current.get("python"):
        differences["python"] = {"recorded": recorded.get("python"), "current": current["python"]}
    return CheckResult(
        check_id=check_id,
        category="reproducibility",
        status="warn" if differences else "pass",
        message=(
            f"{len(differences)} runtime versions differ from training time."
            if differences
            else "Python and tracked library versions match the training environment."
        ),
        observed={"differences": differences, "recorded": recorded},
    )


def reproduction_status_check(reports: Sequence[QualityReportRecord]) -> CheckResult:
    if not reports:
        return CheckResult(
            check_id="reproduction.latest",
            category="reproducibility",
            status="warn",
            message="Not yet reproduced. Run `quant-signal-pipeline reproduce-model`.",
        )
    report = reports[0]
    return CheckResult(
        check_id="reproduction.latest",
        category="reproducibility",
        status=worst_status([report.status]),
        message=f"Latest reproduction ({iso(report.created_at)}): {report.summary}.",
        observed={"quality_report_id": report.id, "status": report.status},
    )


def status_of(checks: Sequence[dict[str, Any]], *check_ids: str) -> str:
    statuses = [str(check["status"]) for check in checks if check.get("check_id") in check_ids]
    return worst_status(statuses) if statuses else "unavailable"


def lineage_stages(lineage: dict[str, Any], verification: QualityReport) -> list[dict[str, Any]]:
    """Summarize each hop of the lineage chain with an evidence-backed status."""

    checks = verification.checks_json()
    dataset = lineage["dataset"]
    model = lineage["model"]
    runs = lineage["ingestion_runs"]
    test = next((row for row in lineage["evaluations"] if row["split_name"] == "test"), None)
    source_bars = dataset.get("source_bars") or {}
    providers = sorted({run["provider"] for run in runs}) or list(source_bars.get("sources") or [])
    date_range = dataset.get("date_range") or {}
    backtests = lineage["backtests"]
    explanations = lineage["explanations"]
    dataset_report = lineage["dataset_quality_report"]
    training_report = lineage["training_quality_report"]
    # Every hop combines its build-time verdict with the live checks that cover it, so a
    # tampered artifact or drifted source can never sit next to a green hop.
    check_ids = {str(check["check_id"]) for check in checks}
    ingestion_status = status_of(checks, "lineage.ingestion_runs", "lineage.source_bar_fingerprint")
    source_status = (
        "unavailable"
        if not providers
        else worst_status(["warn" if lineage["synthetic"] else "pass", ingestion_status])
        if ingestion_status != "unavailable"
        else ("warn" if lineage["synthetic"] else "unavailable")
    )
    dataset_integrity = status_of(checks, "artifact.dataset_integrity")
    dataset_status: str
    if dataset_report:
        dataset_status = worst_status([dataset_report["status"], dataset_integrity])
    else:
        dataset_status = "fail" if dataset_integrity == "fail" else "unavailable"
    training_status = (
        worst_status([training_report["status"], status_of(checks, "quality.training_gate")])
        if training_report
        else "unavailable"
    )
    # The evaluation hop reflects record integrity, a recomputed embargo, and what the
    # evidence supports: a test AUC whose interval reaches or falls below 0.5 is flagged.
    evaluation_status = status_of(
        checks, "evaluation.splits_recorded", "evaluation.split_boundaries"
    )
    if "evaluation.split_boundaries" not in check_ids:
        evaluation_status = worst_status([evaluation_status, "warn"])
    interval = test.get("roc_auc_interval") if test else None
    evaluation_detail = f"n = {test['sample_count']}" if test else ""
    if test and not interval:
        evaluation_status = worst_status([evaluation_status, "warn"])
        evaluation_detail += " · no interval"
    if interval and (interval["includes_chance"] or interval["upper_95"] < 0.5):
        evaluation_status = worst_status([evaluation_status, "warn"])
        evaluation_detail += (
            " · below chance" if interval["upper_95"] < 0.5 else " · not better than chance"
        )
    return [
        {
            "key": "source",
            "title": "Source",
            "status": source_status,
            "headline": ", ".join(providers) if providers else "Unavailable",
            "detail": "Synthetic fixture data" if lineage["synthetic"] else "Market data provider",
        },
        {
            "key": "ingestion",
            "title": "Ingestion",
            "status": ingestion_status,
            "headline": f"{len(runs)} run{'s' if len(runs) != 1 else ''}"
            if runs
            else "Unavailable",
            "detail": f"{source_bars.get('row_count', '—')} source bars, fingerprinted"
            if source_bars
            else "Not recorded",
        },
        {
            "key": "dataset",
            "title": "Dataset",
            "status": dataset_status,
            "headline": f"{dataset['row_count']:,} rows",
            "detail": f"{date_range.get('start', '?')} → {date_range.get('end', '?')}",
        },
        {
            "key": "training_gate",
            "title": "Training gate",
            "status": training_status,
            "headline": f"{model['horizon_days']}d horizon",
            "detail": "Split + embargo checks" if training_report else "Not recorded",
        },
        {
            "key": "model",
            "title": "Model",
            "status": status_of(checks, "artifact.model_integrity", "model.bundle_consistency"),
            "headline": model["model_family"].replace("_", " "),
            "detail": f"Champion rank {model['champion_rank']}"
            if model["champion_rank"]
            else "Candidate",
        },
        {
            "key": "evaluation",
            "title": "Evaluation",
            "status": evaluation_status,
            "headline": f"Test ROC-AUC {test['roc_auc']:.3f}"
            if test and test["roc_auc"] is not None
            else "Unavailable",
            "detail": evaluation_detail,
        },
        {
            "key": "backtest",
            "title": "Backtest",
            "status": status_of(checks, "artifact.backtest_integrity")
            if backtests
            else "unavailable",
            "headline": f"{len(backtests)} run{'s' if len(backtests) != 1 else ''}"
            if backtests
            else "None recorded",
            "detail": "Walk-forward, monthly refit" if backtests else "",
        },
        {
            "key": "explanation",
            "title": "Explanation",
            "status": status_of(checks, "artifact.explainability_integrity")
            if explanations
            else "unavailable",
            "headline": f"{len(explanations)} SHAP run{'s' if len(explanations) != 1 else ''}"
            if explanations
            else "None recorded",
            "detail": "Test-window attributions" if explanations else "",
        },
    ]


def current_provenance() -> dict[str, object]:
    """Expose the current runtime provenance (used by the console footer)."""

    return provenance_snapshot()
