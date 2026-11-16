"""Model governance: lineage assembly, live integrity verification, and reproduction."""

from __future__ import annotations

import logging
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from quant_signal.core.config import Settings, get_settings
from quant_signal.evaluation.metrics import compute_classification_metrics
from quant_signal.governance.limitations import derive_limitations
from quant_signal.governance.serialization import (
    backtest_json,
    dataset_json,
    evaluation_json,
    ingestion_run_json,
    iso,
    model_json,
    quality_report_json,
    section,
    shap_json,
)
from quant_signal.governance.verification import (
    backtest_integrity_check,
    bundle_consistency_check,
    candidate_json,
    code_revision_check,
    evaluation_presence_check,
    ingestion_lineage_check,
    ingestion_run_ids,
    is_synthetic_dataset,
    lineage_stages,
    quality_status,
    report_gate_check,
    reproduction_status_check,
    runtime_match_check,
    shap_integrity_check,
    source_fingerprint_check,
)
from quant_signal.quality import (
    STAGE_DATASET_BUILD,
    STAGE_REPRODUCTION,
    STAGE_TRAINING,
    CheckResult,
    CheckStatus,
    DataQualityPolicy,
    QualityReport,
    check_artifact_integrity,
    worst_status,
)
from quant_signal.quality.persistence import persist_quality_report
from quant_signal.quality.training import run_split_checks
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import (
    DatasetVersion,
    ModelEvaluation,
    ModelVersion,
    QualityReportRecord,
)
from quant_signal.storage.repositories import StorageRepository
from quant_signal.training.service import TrainingService

logger = logging.getLogger(__name__)

STAGE_VERIFICATION = "verification"
PREDICTION_EXACT_TOLERANCE = 1e-9
PREDICTION_NUMERICAL_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-9
REPRODUCED_METRICS = ("roc_auc", "pr_auc", "brier_score", "calibration_error")


class GovernanceNotFoundError(LookupError):
    """Raised when a requested registry record does not exist."""


class GovernanceService:
    """Read-side governance over the existing registry plus explicit reproduction runs."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
        quality_policy: DataQualityPolicy | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url
        self.quality_policy = quality_policy or DataQualityPolicy()

    # ------------------------------------------------------------------ listings

    def list_models(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return registry models with stored (not recomputed) governance status."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            models = repository.list_model_versions(limit=limit)
            datasets = {
                dataset.id: dataset for dataset in repository.list_dataset_versions(limit=10_000)
            }
        rows: list[dict[str, Any]] = []
        for model in models:
            dataset = datasets.get(model.dataset_version_id)
            metadata = model.metadata_json or {}
            governance = metadata.get("governance")
            test_metrics = metadata.get("test_metrics")
            dataset_view = dataset_json(dataset) if dataset else None
            rows.append(
                {
                    "model_version_id": model.id,
                    "dataset_version_id": model.dataset_version_id,
                    "horizon_days": model.horizon_days,
                    "model_family": model.model_family,
                    "champion_rank": model.champion_rank,
                    "created_at": iso(model.created_at),
                    "dataset_as_of_date": dataset_view["as_of_date"] if dataset_view else None,
                    "dataset_symbols": dataset_view["symbols"] if dataset_view else [],
                    "dataset_quality_status": quality_status(dataset_view),
                    "training_quality_status": (
                        governance.get("training_quality_status")
                        if isinstance(governance, dict)
                        else None
                    ),
                    "test_roc_auc": (
                        test_metrics.get("roc_auc") if isinstance(test_metrics, dict) else None
                    ),
                    "test_pr_auc": (
                        test_metrics.get("pr_auc") if isinstance(test_metrics, dict) else None
                    ),
                    "synthetic": is_synthetic_dataset(dataset_view),
                }
            )
        return rows

    def list_quality_reports(
        self,
        stage: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return persisted quality reports (including blocked builds), newest first."""

        with session_scope(self.database_url) as session:
            reports = StorageRepository(session).list_quality_reports(
                stage=stage,
                status=status,
                limit=limit,
            )
        return [quality_report_json(report, include_context=False) for report in reports]

    def get_quality_report(self, quality_report_id: str) -> dict[str, Any]:
        """Return one persisted quality report with its full context."""

        with session_scope(self.database_url) as session:
            try:
                report = StorageRepository(session).get_quality_report(quality_report_id)
            except LookupError as exc:
                raise GovernanceNotFoundError(str(exc)) from exc
        return quality_report_json(report)

    def get_dataset_quality(self, dataset_version_id: str) -> dict[str, Any]:
        """Return a dataset manifest with every quality report that references it."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            dataset = self._get_dataset(repository, dataset_version_id)
            reports = repository.list_quality_reports(dataset_version_id=dataset_version_id)
        return {
            "dataset": dataset_json(dataset),
            "quality_reports": [quality_report_json(report) for report in reports],
        }

    # ------------------------------------------------------------------ lineage

    def model_lineage(self, model_version_id: str) -> dict[str, Any]:
        """Assemble the persisted lineage chain for one model version."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            model = self._get_model(repository, model_version_id)
            dataset = self._get_dataset(repository, model.dataset_version_id)
            dataset_view = dataset_json(dataset)
            run_ids = ingestion_run_ids(dataset_view)
            ingestion_runs = repository.list_ingestion_runs(run_ids) if run_ids else []
            evaluations = repository.list_model_evaluations(model.id)
            backtests = repository.list_backtest_runs(model.id)
            shap_runs = repository.list_shap_runs(model.id)
            signals = repository.signal_snapshot_stats(model.id)
            dataset_reports = repository.list_quality_reports(
                stage=STAGE_DATASET_BUILD,
                dataset_version_id=dataset.id,
                limit=1,
            )
            training_report = self._training_report(repository, model)
            reproduction_reports = repository.list_quality_reports(
                stage=STAGE_REPRODUCTION,
                model_version_id=model.id,
                limit=5,
            )
            siblings = [
                sibling
                for sibling in repository.list_model_versions(dataset_version_id=dataset.id)
                if sibling.horizon_days == model.horizon_days
            ]
            sibling_metrics = {
                sibling.id: repository.list_model_evaluations(sibling.id) for sibling in siblings
            }

        model_view = model_json(model)
        found_run_ids = {run.id for run in ingestion_runs}
        lineage: dict[str, Any] = {
            "model": model_view,
            "dataset": dataset_view,
            "ingestion_runs": [ingestion_run_json(run) for run in ingestion_runs],
            "missing_ingestion_run_ids": [
                run_id for run_id in run_ids if run_id not in found_run_ids
            ],
            "dataset_quality_report": (
                quality_report_json(dataset_reports[0]) if dataset_reports else None
            ),
            "training_quality_report": (
                quality_report_json(training_report) if training_report else None
            ),
            "reproduction_reports": [
                quality_report_json(report) for report in reproduction_reports
            ],
            "evaluations": [evaluation_json(evaluation) for evaluation in evaluations],
            "candidates": [
                candidate_json(sibling, sibling_metrics.get(sibling.id, []))
                for sibling in sorted(siblings, key=lambda row: (row.champion_rank or 99, row.id))
            ],
            "backtests": [backtest_json(run) for run in backtests],
            "explanations": [shap_json(run) for run in shap_runs],
            "signals": {
                "snapshot_count": signals.snapshot_count,
                "first_as_of_date": iso(signals.first_as_of_date),
                "last_as_of_date": iso(signals.last_as_of_date),
            },
            "synthetic": is_synthetic_dataset(dataset_view)
            or any(run.provider == "synthetic" for run in ingestion_runs),
        }
        return lineage

    def model_evidence(self, model_version_id: str) -> dict[str, Any]:
        """Return lineage plus live verification, stage statuses, and limitations."""

        lineage = self.model_lineage(model_version_id)
        verification = self.verify_model(model_version_id)
        lineage["verification"] = {
            **verification.summary_json(),
            "checks": verification.checks_json(),
        }
        lineage["stages"] = lineage_stages(lineage, verification)
        lineage["limitations"] = derive_limitations(lineage, verification)
        return lineage

    # ------------------------------------------------------------------ verification

    def verify_model(self, model_version_id: str) -> QualityReport:
        """Re-verify every persisted artifact and lineage pointer for a model (read-only)."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            model = self._get_model(repository, model_version_id)
            dataset = self._get_dataset(repository, model.dataset_version_id)
            evaluations = repository.list_model_evaluations(model.id)
            backtests = repository.list_backtest_runs(model.id)
            shap_runs = repository.list_shap_runs(model.id)
            dataset_reports = repository.list_quality_reports(
                stage=STAGE_DATASET_BUILD,
                dataset_version_id=dataset.id,
                limit=1,
            )
            training_report = self._training_report(repository, model)
            reproduction_reports = repository.list_quality_reports(
                stage=STAGE_REPRODUCTION,
                model_version_id=model.id,
                limit=1,
            )
            dataset_view = dataset_json(dataset)
            run_ids = ingestion_run_ids(dataset_view)
            ingestion_runs = repository.list_ingestion_runs(run_ids) if run_ids else []
            current_bars = self._current_source_bars(repository, dataset_view, dataset)

        dataset_integrity = check_artifact_integrity(
            dataset.artifact_path,
            dataset.artifact_hash,
            "dataset",
            check_id="artifact.dataset_integrity",
        )
        model_integrity = check_artifact_integrity(
            model.artifact_path,
            model.artifact_hash,
            "model",
            check_id="artifact.model_integrity",
        )
        checks: list[CheckResult] = [dataset_integrity, model_integrity]
        checks.append(bundle_consistency_check(model, model_integrity))
        checks.append(source_fingerprint_check(dataset_view, current_bars))
        checks.append(ingestion_lineage_check(run_ids, ingestion_runs))
        checks.append(report_gate_check("quality.dataset_gate", "dataset build", dataset_reports))
        checks.append(
            report_gate_check(
                "quality.training_gate",
                "training",
                [training_report] if training_report else [],
            )
        )
        checks.append(evaluation_presence_check(evaluations))
        if dataset_integrity.status == "pass":
            checks.append(self._split_boundary_check(model, dataset))
        if backtests:
            checks.append(backtest_integrity_check(backtests))
        if shap_runs:
            checks.append(shap_integrity_check(shap_runs))
        checks.append(code_revision_check(model))
        checks.append(runtime_match_check(model))
        checks.append(reproduction_status_check(reproduction_reports))
        return QualityReport(stage=STAGE_VERIFICATION, checks=tuple(checks))

    def _split_boundary_check(self, model: ModelVersion, dataset: DatasetVersion) -> CheckResult:
        """Recompute the label-horizon embargo from the artifact and persisted windows."""

        frame = pd.read_parquet(dataset.artifact_path)
        frame["date"] = pd.to_datetime(frame["date"])
        labeled_dates = pd.DatetimeIndex(
            sorted(frame.loc[frame[model.target_column].notna(), "date"].unique())
        )
        windows = {
            "train": (model.train_start_date, model.train_end_date),
            "validation": (model.validation_start_date, model.validation_end_date),
            "test": (model.test_start_date, model.test_end_date),
        }
        if any(start is None or end is None for start, end in windows.values()):
            return CheckResult(
                check_id="evaluation.split_boundaries",
                category="leakage",
                status="warn",
                message="Split windows were not recorded; the embargo cannot be re-verified.",
            )
        split_dates = {
            name: list(
                labeled_dates[
                    (labeled_dates >= pd.Timestamp(start)) & (labeled_dates <= pd.Timestamp(end))
                ]
            )
            for name, (start, end) in windows.items()
        }
        results = run_split_checks(
            frame,
            split_dates,
            horizon=model.horizon_days,
            feature_columns=list(model.feature_columns),
            target_column=model.target_column,
            policy=self.quality_policy,
        )
        embargo = next(
            result for result in results if result.check_id == "training.label_horizon_embargo"
        )
        ordering = next(
            result for result in results if result.check_id == "training.split_ordering"
        )
        status = worst_status([embargo.status, ordering.status])
        return CheckResult(
            check_id="evaluation.split_boundaries",
            category="leakage",
            status=status,
            message=(
                "Recomputed from the dataset artifact: " + embargo.message
                if status != "fail"
                else f"{ordering.message} {embargo.message}"
            ),
            observed={**embargo.observed, "windows": ordering.observed.get("windows")},
        )

    # ------------------------------------------------------------------ reproduction

    def reproduce_model(self, model_version_id: str) -> tuple[QualityReport, str]:
        """Refit a registered model from its dataset artifact and compare the outputs.

        The persisted report (stage ``reproduction``) records whether the split windows,
        hyperparameters, test predictions, and test metrics were reproduced.
        """

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            model = self._get_model(repository, model_version_id)
            dataset = self._get_dataset(repository, model.dataset_version_id)
            test_evaluation = next(
                (
                    evaluation
                    for evaluation in repository.list_model_evaluations(model.id)
                    if evaluation.split_name == "test"
                ),
                None,
            )

        checks: list[CheckResult] = [
            check_artifact_integrity(
                dataset.artifact_path,
                dataset.artifact_hash,
                "dataset",
                check_id="reproduction.dataset_integrity",
            ),
            check_artifact_integrity(
                model.artifact_path,
                model.artifact_hash,
                "model",
                check_id="reproduction.model_integrity",
            ),
        ]
        notes: list[str] = []
        if all(check.status == "pass" for check in checks):
            checks.extend(self._reproduction_checks(model, dataset, test_evaluation))
        else:
            notes.append("Refit skipped: registered artifacts failed integrity verification.")

        report = QualityReport(stage=STAGE_REPRODUCTION, checks=tuple(checks), notes=tuple(notes))
        with session_scope(self.database_url) as session:
            record = persist_quality_report(
                StorageRepository(session),
                report,
                self.quality_policy,
                request_json={"model_version_id": model.id},
                context_json={"recorded_provenance": (model.metadata_json or {}).get("provenance")},
                dataset_version_id=dataset.id,
                model_version_id=model.id,
                horizon_days=model.horizon_days,
            )
            report_id = record.id
        logger.info("Reproduction of model %s finished: %s", model.id, report.status)
        return report, report_id

    def _reproduction_checks(
        self,
        model: ModelVersion,
        dataset: DatasetVersion,
        test_evaluation: ModelEvaluation | None,
    ) -> list[CheckResult]:
        training = TrainingService(settings=self.settings, database_url=self.database_url)
        bundle = joblib.load(model.artifact_path)
        frame = pd.read_parquet(dataset.artifact_path)
        frame["date"] = pd.to_datetime(frame["date"])
        training_frame = frame.dropna(subset=[model.target_column]).copy()
        results: list[CheckResult] = []

        training_config = section(model.metadata_json, "training_config") or {}
        recorded_split = training_config.get("split")
        if isinstance(recorded_split, dict) and "minimum_train_dates" in recorded_split:
            training.settings = self.settings.model_copy(
                update={"min_training_days": int(recorded_split["minimum_train_dates"])}
            )
        try:
            split = training.build_split(training_frame, model.horizon_days)
        except ValueError as exc:
            return [
                CheckResult(
                    check_id="reproduction.split_windows",
                    category="reproducibility",
                    status="fail",
                    message=f"The split could not be rebuilt: {exc}",
                )
            ]

        rebuilt = {
            "train": (split.train_dates[0], split.train_dates[-1]),
            "validation": (split.validation_dates[0], split.validation_dates[-1]),
            "test": (split.test_dates[0], split.test_dates[-1]),
        }
        recorded = {
            "train": (model.train_start_date, model.train_end_date),
            "validation": (model.validation_start_date, model.validation_end_date),
            "test": (model.test_start_date, model.test_end_date),
        }
        mismatched = [
            name
            for name, (start, end) in rebuilt.items()
            if (start.date(), end.date()) != recorded[name]
        ]
        results.append(
            CheckResult(
                check_id="reproduction.split_windows",
                category="reproducibility",
                status="fail" if mismatched else "pass",
                message=(
                    f"Rebuilt split windows differ for: {', '.join(mismatched)}."
                    if mismatched
                    else "Rebuilt train/validation/test windows match the registry."
                ),
                observed={
                    "rebuilt": {
                        name: [start.date().isoformat(), end.date().isoformat()]
                        for name, (start, end) in rebuilt.items()
                    },
                    "recorded": {
                        name: [iso(start), iso(end)] for name, (start, end) in recorded.items()
                    },
                },
            )
        )

        current_pipeline = training.candidate_pipeline_description(model.model_family)
        recorded_pipeline = training_config.get("pipeline_steps")
        if recorded_pipeline is None:
            results.append(
                CheckResult(
                    check_id="reproduction.hyperparameters",
                    category="reproducibility",
                    status="warn",
                    message="Hyperparameters were not recorded at training time (legacy model); "
                    "the refit uses the current code's configuration.",
                    observed={"current": current_pipeline},
                )
            )
        else:
            same = recorded_pipeline == current_pipeline
            results.append(
                CheckResult(
                    check_id="reproduction.hyperparameters",
                    category="reproducibility",
                    status="pass" if same else "fail",
                    message=(
                        "Current code builds the same pipeline and hyperparameters as recorded."
                        if same
                        else "Current code's pipeline or hyperparameters differ from the "
                        "recorded training configuration."
                    ),
                    observed={"recorded": recorded_pipeline, "current": current_pipeline},
                )
            )

        refit = training.refit_model_family(
            training_frame=training_frame,
            dataset_version_id=dataset.id,
            feature_columns=list(model.feature_columns),
            target_column=model.target_column,
            horizon=model.horizon_days,
            model_family=model.model_family,
            split=split,
        )
        test_frame = training_frame[split.mask(training_frame, "test")]
        refit_probabilities = refit.predict_positive_proba(test_frame)
        registered_probabilities = bundle.predict_positive_proba(test_frame)
        max_difference = float(np.max(np.abs(refit_probabilities - registered_probabilities)))
        prediction_status: CheckStatus = (
            "pass"
            if max_difference <= PREDICTION_EXACT_TOLERANCE
            else "warn"
            if max_difference <= PREDICTION_NUMERICAL_TOLERANCE
            else "fail"
        )
        results.append(
            CheckResult(
                check_id="reproduction.prediction_equivalence",
                category="reproducibility",
                status=prediction_status,
                message=(
                    f"Refit and registered model agree on all {len(test_frame)} test rows "
                    f"(max |Δp| = {max_difference:.2e})."
                    if prediction_status != "fail"
                    else f"Refit predictions diverge from the registered model "
                    f"(max |Δp| = {max_difference:.2e})."
                ),
                observed={"test_rows": int(len(test_frame)), "max_abs_difference": max_difference},
                threshold={
                    "exact_tolerance": PREDICTION_EXACT_TOLERANCE,
                    "numerical_tolerance": PREDICTION_NUMERICAL_TOLERANCE,
                },
            )
        )

        y_test = test_frame[model.target_column].astype(int).to_numpy()
        reproduced = compute_classification_metrics(y_test, refit_probabilities)
        if test_evaluation is None:
            results.append(
                CheckResult(
                    check_id="reproduction.test_metrics",
                    category="reproducibility",
                    status="fail",
                    message="No persisted test evaluation exists to compare against.",
                )
            )
        else:
            persisted = test_evaluation.metrics_json or {}
            differences: dict[str, float | None] = {}
            for metric in REPRODUCED_METRICS:
                left = persisted.get(metric)
                right = reproduced.get(metric)
                if left is None or right is None:
                    differences[metric] = None if left == right else float("inf")
                else:
                    differences[metric] = abs(float(cast(float, left)) - float(right))
            failing = [
                name
                for name, value in differences.items()
                if value is not None and value > METRIC_TOLERANCE
            ]
            results.append(
                CheckResult(
                    check_id="reproduction.test_metrics",
                    category="reproducibility",
                    status="fail" if failing else "pass",
                    message=(
                        f"Reproduced test metrics differ: {', '.join(failing)}."
                        if failing
                        else "Reproduced test ROC-AUC, PR-AUC, Brier score, and calibration "
                        "error match the persisted evaluation."
                    ),
                    observed={
                        "persisted": {name: persisted.get(name) for name in REPRODUCED_METRICS},
                        "reproduced": {name: reproduced.get(name) for name in REPRODUCED_METRICS},
                        "abs_difference": {
                            name: (str(value) if value == float("inf") else value)
                            for name, value in differences.items()
                        },
                    },
                    threshold={"absolute_tolerance": METRIC_TOLERANCE},
                )
            )
        results.append(runtime_match_check(model, check_id="reproduction.runtime_versions"))
        return results

    # ------------------------------------------------------------------ helpers

    def _get_model(self, repository: StorageRepository, model_version_id: str) -> ModelVersion:
        try:
            return repository.get_model_version(model_version_id)
        except ValueError as exc:
            raise GovernanceNotFoundError(str(exc)) from exc

    def _get_dataset(
        self,
        repository: StorageRepository,
        dataset_version_id: str,
    ) -> DatasetVersion:
        try:
            return repository.get_dataset_version(dataset_version_id)
        except ValueError as exc:
            raise GovernanceNotFoundError(str(exc)) from exc

    def _training_report(
        self,
        repository: StorageRepository,
        model: ModelVersion,
    ) -> QualityReportRecord | None:
        governance = (model.metadata_json or {}).get("governance")
        report_id = (
            governance.get("training_quality_report_id") if isinstance(governance, dict) else None
        )
        if not isinstance(report_id, str):
            return None
        try:
            report = repository.get_quality_report(report_id)
        except LookupError:
            return None
        return report if report.stage == STAGE_TRAINING else None

    def _current_source_bars(
        self,
        repository: StorageRepository,
        dataset_view: dict[str, Any],
        dataset: DatasetVersion,
    ) -> pd.DataFrame | None:
        source_bars = dataset_view.get("source_bars")
        if not isinstance(source_bars, dict) or not source_bars.get("symbols"):
            return None
        return repository.load_daily_bars_frame(
            [str(symbol) for symbol in source_bars["symbols"]],
            end_date=dataset.as_of_date,
        )
