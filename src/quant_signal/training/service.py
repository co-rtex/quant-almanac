"""Training, calibration, evaluation, and registry persistence."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_signal.core.config import Settings, get_settings
from quant_signal.core.hashing import sha256_file, sha256_json
from quant_signal.core.provenance import provenance_snapshot
from quant_signal.evaluation.metrics import compute_classification_metrics
from quant_signal.evaluation.reporting import rank_candidate_metrics
from quant_signal.features.splits import TemporalSplit, build_temporal_split
from quant_signal.quality import (
    STAGE_DATASET_BUILD,
    STAGE_TRAINING,
    CheckResult,
    CheckStatus,
    DataQualityGateError,
    DataQualityPolicy,
    QualityReport,
    check_artifact_integrity,
    require_artifact_integrity,
    worst_status,
)
from quant_signal.quality.persistence import persist_quality_report
from quant_signal.quality.training import run_split_checks
from quant_signal.serving.service import rank_signal_frame
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import DatasetVersion, ModelVersion, QualityReportRecord
from quant_signal.storage.repositories import (
    EvaluationRecord,
    ModelArtifactRecord,
    SignalSnapshotRecord,
    StorageRepository,
)
from quant_signal.training.artifacts import ModelArtifactBundle, ProbabilityCalibrator

logger = logging.getLogger(__name__)

SPLIT_VALIDATION_FRACTION = 0.2
SPLIT_TEST_FRACTION = 0.2
CHAMPION_SELECTION_RULE = "validation PR-AUC (desc), then Brier score (asc), then ROC-AUC (desc)"


def describe_pipeline(estimator: Pipeline) -> list[dict[str, object]]:
    """Return JSON-safe step classes and scalar hyperparameters for a pipeline."""

    steps: list[dict[str, object]] = []
    for name, step in estimator.steps:
        params = {
            # Non-finite floats (e.g. SimpleImputer's ``missing_values=nan``) are stored
            # as strings: NaN is not valid JSON (PostgreSQL rejects it) and NaN != NaN
            # would make recorded-vs-current comparisons fail spuriously.
            key: value if not isinstance(value, float) or math.isfinite(value) else str(value)
            for key, value in sorted(step.get_params(deep=False).items())
            if isinstance(value, (bool, int, float, str)) or value is None
        }
        steps.append({"name": name, "class": type(step).__name__, "params": params})
    return steps


@dataclass
class CandidateTrainingResult:
    """In-memory training result for a candidate model family."""

    model_family: str
    bundle: ModelArtifactBundle
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    artifact_path: Path
    artifact_hash: str


@dataclass(frozen=True)
class SignalSnapshotRefreshResult:
    """Summary of one explicit signal snapshot refresh operation."""

    model_version_id: str
    dataset_version_id: str
    horizon_days: int
    model_family: str
    champion_rank: int | None
    snapshots_written: int
    signal_dates: int
    first_as_of_date: date | None
    latest_as_of_date: date | None


class TrainingService:
    """Train calibrated model families and persist registry outputs."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
        quality_policy: DataQualityPolicy | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url
        self.quality_policy = quality_policy or DataQualityPolicy()

    def train(
        self,
        dataset_version_id: str,
        horizons: Sequence[int] | None = None,
    ) -> list[ModelVersion]:
        """Train and persist candidate models for the requested horizons."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            dataset_version = repository.get_dataset_version(dataset_version_id)
            dataset_reports = repository.list_quality_reports(
                stage=STAGE_DATASET_BUILD,
                dataset_version_id=dataset_version_id,
                limit=1,
            )

        requested_horizons = list(horizons or dataset_version.horizons)
        unknown_horizons = sorted(set(requested_horizons) - set(dataset_version.horizons))
        if unknown_horizons:
            raise ValueError(
                f"Dataset {dataset_version.id} has no labels for horizons {unknown_horizons}; "
                f"available horizons: {sorted(dataset_version.horizons)}."
            )
        dataset_quality_report = dataset_reports[0] if dataset_reports else None
        pre_training_checks = [
            check_artifact_integrity(
                dataset_version.artifact_path,
                dataset_version.artifact_hash,
                "dataset",
                check_id="training.dataset_artifact_integrity",
            ),
            self._dataset_quality_gate_check(dataset_quality_report),
        ]
        if any(check.status == "fail" for check in pre_training_checks):
            self._persist_training_report(
                QualityReport(stage=STAGE_TRAINING, checks=tuple(pre_training_checks)),
                dataset_version=dataset_version,
                horizon=None,
                requested_horizons=requested_horizons,
                raise_on_failure=True,
            )

        dataset_frame = pd.read_parquet(dataset_version.artifact_path)
        dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])
        persisted_models: list[ModelVersion] = []
        feature_columns = list(dataset_version.feature_columns)

        # Gate every requested horizon before fitting anything, so a failing (or
        # unsplittable) later horizon cannot leave earlier horizons' models behind.
        gated: list[tuple[int, pd.DataFrame, TemporalSplit, QualityReport]] = []
        for horizon in requested_horizons:
            target_column = f"target_up_{horizon}d"
            training_frame = dataset_frame.dropna(subset=[target_column]).copy()
            split = self.build_split(training_frame, horizon)
            gated.append(
                (
                    horizon,
                    training_frame,
                    split,
                    QualityReport(
                        stage=STAGE_TRAINING,
                        checks=(
                            *pre_training_checks,
                            *run_split_checks(
                                dataset_frame,
                                {
                                    "train": split.train_dates,
                                    "validation": split.validation_dates,
                                    "test": split.test_dates,
                                },
                                horizon=horizon,
                                feature_columns=feature_columns,
                                target_column=target_column,
                                policy=self.quality_policy,
                            ),
                        ),
                    ),
                )
            )
        report_ids = {
            horizon: self._persist_training_report(
                report,
                dataset_version=dataset_version,
                horizon=horizon,
                requested_horizons=requested_horizons,
                raise_on_failure=False,
            )
            for horizon, _, _, report in gated
        }
        for horizon, _, _, report in gated:
            if not report.passed:
                logger.warning(
                    "Training blocked for dataset %s horizon %s by report %s",
                    dataset_version.id,
                    horizon,
                    report_ids[horizon],
                )
                raise DataQualityGateError(report, report_ids[horizon])

        for horizon, training_frame, split, training_report in gated:
            target_column = f"target_up_{horizon}d"
            governance_json: dict[str, object] = {
                "training_quality_report_id": report_ids[horizon],
                "training_quality_status": training_report.status,
                "dataset_quality_report_id": (
                    dataset_quality_report.id if dataset_quality_report else None
                ),
                "dataset_quality_status": (
                    dataset_quality_report.status if dataset_quality_report else None
                ),
                "dataset_artifact_sha256": dataset_version.artifact_hash,
                "quality_policy_version": self.quality_policy.version,
            }
            candidate_results = self._train_candidates(
                dataset_frame=training_frame,
                dataset_version_id=dataset_version.id,
                feature_columns=feature_columns,
                target_column=target_column,
                horizon=horizon,
                split=split,
                governance_json=governance_json,
            )

            ranked_candidates = rank_candidate_metrics(
                [
                    {
                        "model_family": result.model_family,
                        "validation_metrics": result.validation_metrics,
                    }
                    for result in candidate_results
                ]
            )
            ranking_map = {
                record["model_family"]: index
                for index, record in enumerate(ranked_candidates, start=1)
            }

            with session_scope(self.database_url) as session:
                repository = StorageRepository(session)
                for result in candidate_results:
                    champion_rank = ranking_map[result.model_family]
                    split_summary = result.bundle.split_summary
                    train_summary = split_summary["train"]
                    validation_summary = split_summary["validation"]
                    test_summary = split_summary["test"]
                    model_version = repository.create_model_version(
                        ModelArtifactRecord(
                            dataset_version_id=dataset_version.id,
                            horizon_days=horizon,
                            model_family=result.model_family,
                            target_column=target_column,
                            artifact_path=str(result.artifact_path),
                            artifact_hash=result.artifact_hash,
                            feature_columns=feature_columns,
                            champion_rank=champion_rank,
                            train_start_date=cast(date | None, train_summary["start_date"]),
                            train_end_date=cast(date | None, train_summary["end_date"]),
                            validation_start_date=cast(
                                date | None,
                                validation_summary["start_date"],
                            ),
                            validation_end_date=cast(date | None, validation_summary["end_date"]),
                            test_start_date=cast(date | None, test_summary["start_date"]),
                            test_end_date=cast(date | None, test_summary["end_date"]),
                            metadata_json=result.bundle.metadata_json,
                        )
                    )
                    repository.create_model_evaluation(
                        EvaluationRecord(
                            model_version_id=model_version.id,
                            split_name="validation",
                            metrics_json=result.validation_metrics,
                            calibration_bins=result.validation_metrics["calibration_bins"],
                        )
                    )
                    repository.create_model_evaluation(
                        EvaluationRecord(
                            model_version_id=model_version.id,
                            split_name="test",
                            metrics_json=result.test_metrics,
                            calibration_bins=result.test_metrics["calibration_bins"],
                        )
                    )

                    if champion_rank == 1:
                        self._replace_signal_snapshots(
                            repository=repository,
                            model_version=model_version,
                            bundle=result.bundle,
                            dataset_frame=dataset_frame,
                        )

                    persisted_models.append(model_version)

        return persisted_models

    def refresh_signal_snapshots(
        self,
        model_version_id: str,
    ) -> SignalSnapshotRefreshResult:
        """Refresh persisted ranked signal snapshots for one explicit model version."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            model_version = repository.get_model_version(model_version_id)
            dataset_version = repository.get_dataset_version(model_version.dataset_version_id)

        require_artifact_integrity(
            model_version.artifact_path,
            model_version.artifact_hash,
            "model",
        )
        require_artifact_integrity(
            dataset_version.artifact_path,
            dataset_version.artifact_hash,
            "dataset",
        )
        bundle = joblib.load(model_version.artifact_path)
        dataset_frame = pd.read_parquet(dataset_version.artifact_path)
        dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            return self._replace_signal_snapshots(
                repository=repository,
                model_version=model_version,
                bundle=bundle,
                dataset_frame=dataset_frame,
            )

    def build_split(self, training_frame: pd.DataFrame, horizon: int) -> TemporalSplit:
        """Return the embargoed temporal holdout split used for registry training."""

        return build_temporal_split(
            training_frame,
            validation_fraction=SPLIT_VALIDATION_FRACTION,
            test_fraction=SPLIT_TEST_FRACTION,
            embargo_days=horizon,
            minimum_train_dates=self.settings.min_training_days,
        )

    def refit_model_family(
        self,
        training_frame: pd.DataFrame,
        dataset_version_id: str,
        feature_columns: list[str],
        target_column: str,
        horizon: int,
        model_family: str,
        split: TemporalSplit,
    ) -> ModelArtifactBundle:
        """Refit one calibrated family exactly as ``train`` does (used for reproduction)."""

        return self._fit_model_family(
            dataset_frame=training_frame,
            dataset_version_id=dataset_version_id,
            feature_columns=feature_columns,
            target_column=target_column,
            horizon=horizon,
            model_family=model_family,
            split=split,
        )

    def candidate_pipeline_description(self, model_family: str) -> list[dict[str, object]]:
        """Describe the current code's (unfitted) pipeline for one model family."""

        estimators = self._candidate_estimators()
        if model_family not in estimators:
            raise LookupError(f"Unknown model family: {model_family}")
        return describe_pipeline(estimators[model_family])

    def _train_candidates(
        self,
        dataset_frame: pd.DataFrame,
        dataset_version_id: str,
        feature_columns: list[str],
        target_column: str,
        horizon: int,
        split: TemporalSplit,
        governance_json: dict[str, object] | None = None,
    ) -> list[CandidateTrainingResult]:
        """Train all baseline candidate model families for a horizon."""

        train_frame = dataset_frame[split.mask(dataset_frame, "train")].copy()
        validation_frame = dataset_frame[split.mask(dataset_frame, "validation")].copy()
        test_frame = dataset_frame[split.mask(dataset_frame, "test")].copy()

        x_train = train_frame[feature_columns]
        x_validation = validation_frame[feature_columns]
        x_test = test_frame[feature_columns]
        y_train = train_frame[target_column].astype(int).to_numpy()
        y_validation = validation_frame[target_column].astype(int).to_numpy()
        y_test = test_frame[target_column].astype(int).to_numpy()

        self._ensure_class_diversity(y_train, split_name="train", horizon=horizon)
        self._ensure_class_diversity(y_validation, split_name="validation", horizon=horizon)
        self._ensure_class_diversity(y_test, split_name="test", horizon=horizon)

        split_summary = self._build_split_summary(train_frame, validation_frame, test_frame)
        results: list[CandidateTrainingResult] = []

        for model_family, estimator in self._candidate_estimators().items():
            fitted_estimator = estimator.fit(x_train, y_train)
            validation_raw = fitted_estimator.predict_proba(x_validation)[:, 1]
            calibrator = ProbabilityCalibrator.fit(validation_raw, y_validation)
            validation_probabilities = calibrator.predict(validation_raw)
            test_probabilities = calibrator.predict(fitted_estimator.predict_proba(x_test)[:, 1])

            validation_metrics = compute_classification_metrics(
                y_validation,
                validation_probabilities,
            )
            test_metrics = compute_classification_metrics(y_test, test_probabilities)

            bundle = ModelArtifactBundle(
                model_family=model_family,
                feature_columns=feature_columns,
                target_column=target_column,
                horizon_days=horizon,
                dataset_version_id=dataset_version_id,
                estimator=fitted_estimator,
                calibrator=calibrator,
                split_summary=split_summary,
                metadata_json={
                    "trained_at": datetime.now(tz=UTC).isoformat(),
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "training_config": self._training_config(
                        fitted_estimator,
                        calibrator,
                        horizon,
                        split,
                    ),
                    "governance": dict(governance_json or {}),
                    "provenance": provenance_snapshot(),
                },
            )
            artifact_path, artifact_hash = self._write_model_artifact(
                bundle,
                dataset_version_id,
                horizon,
                model_family,
                split_summary,
            )

            results.append(
                CandidateTrainingResult(
                    model_family=model_family,
                    bundle=bundle,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    artifact_path=artifact_path,
                    artifact_hash=artifact_hash,
                )
            )

        return results

    def _build_split_for_backtest(
        self,
        history_frame: pd.DataFrame,
        embargo_days: int,
    ) -> TemporalSplit:
        """Build a temporal split for a walk-forward retraining window."""

        return build_temporal_split(
            history_frame,
            embargo_days=embargo_days,
            minimum_train_dates=self.settings.min_training_days,
        )

    def _fit_model_family(
        self,
        dataset_frame: pd.DataFrame,
        dataset_version_id: str,
        feature_columns: list[str],
        target_column: str,
        horizon: int,
        model_family: str,
        split: TemporalSplit,
    ) -> ModelArtifactBundle:
        """Fit a single calibrated model family and return its bundle."""

        train_frame = dataset_frame[split.mask(dataset_frame, "train")].copy()
        validation_frame = dataset_frame[split.mask(dataset_frame, "validation")].copy()
        x_train = train_frame[feature_columns]
        x_validation = validation_frame[feature_columns]
        y_train = train_frame[target_column].astype(int).to_numpy()
        y_validation = validation_frame[target_column].astype(int).to_numpy()

        self._ensure_class_diversity(y_train, split_name="train", horizon=horizon)
        self._ensure_class_diversity(y_validation, split_name="validation", horizon=horizon)

        estimator = self._candidate_estimators()[model_family]
        fitted_estimator = estimator.fit(x_train, y_train)
        validation_raw = fitted_estimator.predict_proba(x_validation)[:, 1]
        calibrator = ProbabilityCalibrator.fit(validation_raw, y_validation)

        return ModelArtifactBundle(
            model_family=model_family,
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon,
            dataset_version_id=dataset_version_id,
            estimator=fitted_estimator,
            calibrator=calibrator,
            split_summary=self._build_split_summary(
                train_frame,
                validation_frame,
                validation_frame,
            ),
            metadata_json={"trained_for": "backtest"},
        )

    def _build_signal_snapshots(
        self,
        model_version_id: str,
        bundle: ModelArtifactBundle,
        dataset_frame: pd.DataFrame,
    ) -> list[SignalSnapshotRecord]:
        """Score and rank the full dataset for API-ready signal snapshots."""

        scorable = dataset_frame.copy()
        feature_coverage = scorable[bundle.feature_columns].notna().sum(axis=1)
        scorable = scorable[feature_coverage >= max(5, len(bundle.feature_columns) // 2)].copy()
        scorable["score"] = bundle.predict_positive_proba(scorable)
        ranked = rank_signal_frame(scorable[["date", "symbol", "score"]])

        return [
            SignalSnapshotRecord(
                model_version_id=model_version_id,
                horizon_days=bundle.horizon_days,
                as_of_date=row.date.date(),
                symbol=str(row.symbol),
                score=float(row.score),
                rank=int(row.rank),
                metadata_json={"model_family": bundle.model_family},
            )
            for row in ranked.itertuples(index=False)
        ]

    def _replace_signal_snapshots(
        self,
        repository: StorageRepository,
        model_version: ModelVersion,
        bundle: ModelArtifactBundle,
        dataset_frame: pd.DataFrame,
    ) -> SignalSnapshotRefreshResult:
        """Replace one model version's persisted signal snapshots safely."""

        snapshots = self._build_signal_snapshots(
            model_version_id=model_version.id,
            bundle=bundle,
            dataset_frame=dataset_frame,
        )
        if not snapshots:
            raise ValueError(
                f"No signal snapshots were generated for model version {model_version.id}."
            )

        repository.replace_signal_snapshots(model_version.id, snapshots)
        signal_dates = sorted({snapshot.as_of_date for snapshot in snapshots})

        return SignalSnapshotRefreshResult(
            model_version_id=model_version.id,
            dataset_version_id=model_version.dataset_version_id,
            horizon_days=model_version.horizon_days,
            model_family=model_version.model_family,
            champion_rank=model_version.champion_rank,
            snapshots_written=len(snapshots),
            signal_dates=len(signal_dates),
            first_as_of_date=signal_dates[0] if signal_dates else None,
            latest_as_of_date=signal_dates[-1] if signal_dates else None,
        )

    def _candidate_estimators(self) -> dict[str, Pipeline]:
        """Return the baseline candidate model families."""

        return {
            "logistic_regression": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "hist_gradient_boosting": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=250,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }

    def _ensure_class_diversity(
        self,
        targets: Any,
        split_name: str,
        horizon: int,
    ) -> None:
        """Raise a clear error when a split lacks both target classes."""

        unique_classes = set(targets.tolist())
        if len(unique_classes) < 2:
            raise ValueError(
                f"Horizon {horizon}d has only one target class in the {split_name} split."
            )

    def _build_artifact_path(
        self,
        dataset_version_id: str,
        horizon: int,
        model_family: str,
        split_summary: dict[str, dict[str, Any]],
        artifact_hash: str,
    ) -> Path:
        """Build a content-addressed artifact path for a model candidate.

        The configuration key keeps paths readable; the content digest guarantees that
        retraining never overwrites an artifact referenced by an earlier registry row.
        """

        artifact_key = sha256_json(
            {
                "dataset_version_id": dataset_version_id,
                "horizon": horizon,
                "model_family": model_family,
                "split_summary": split_summary,
            }
        )[:12]
        filename = (
            f"model_{dataset_version_id[:8]}_{horizon}d_{model_family}_{artifact_key}"
            f"_{artifact_hash[:12]}.joblib"
        )
        return Path(self.settings.artifact_root) / "models" / filename

    def _write_model_artifact(
        self,
        bundle: ModelArtifactBundle,
        dataset_version_id: str,
        horizon: int,
        model_family: str,
        split_summary: dict[str, dict[str, Any]],
    ) -> tuple[Path, str]:
        """Serialize a bundle to an immutable, content-addressed path."""

        directory = Path(self.settings.artifact_root) / "models"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".model_{uuid4().hex}.joblib.tmp"
        joblib.dump(bundle, temporary)
        artifact_hash = sha256_file(temporary)
        artifact_path = self._build_artifact_path(
            dataset_version_id,
            horizon,
            model_family,
            split_summary,
            artifact_hash,
        )
        os.replace(temporary, artifact_path)
        return artifact_path, artifact_hash

    def _training_config(
        self,
        estimator: Pipeline,
        calibrator: ProbabilityCalibrator,
        horizon: int,
        split: TemporalSplit,
    ) -> dict[str, object]:
        """Return the hyperparameters and evaluation protocol needed to reproduce a fit."""

        return {
            "pipeline_steps": describe_pipeline(estimator),
            "calibration": {"method": calibrator.strategy, "fit_on": "validation"},
            "split": {
                "method": "embargoed_temporal_holdout",
                "validation_fraction": SPLIT_VALIDATION_FRACTION,
                "test_fraction": SPLIT_TEST_FRACTION,
                "embargo_days": split.embargo_days,
                "minimum_train_dates": self.settings.min_training_days,
                "train_dates": len(split.train_dates),
                "validation_dates": len(split.validation_dates),
                "test_dates": len(split.test_dates),
            },
            "horizon_days": horizon,
            "champion_selection": CHAMPION_SELECTION_RULE,
            "test_split_usage": "reporting only; never used for fitting, calibration, "
            "or champion selection",
        }

    def _dataset_quality_gate_check(
        self,
        dataset_report: QualityReportRecord | None,
    ) -> CheckResult:
        """Translate the dataset build verdict into a training gate check."""

        if dataset_report is None:
            return CheckResult(
                check_id="training.dataset_quality_gate",
                category="governance",
                status="warn",
                message=(
                    "No dataset quality report exists (dataset predates quality gates); "
                    "training proceeds with unverified input quality."
                ),
                observed={"dataset_quality_report_id": None},
            )
        status: CheckStatus = worst_status([dataset_report.status])
        return CheckResult(
            check_id="training.dataset_quality_gate",
            category="governance",
            status=status,
            message=f"Dataset quality report verdict: {dataset_report.summary}.",
            observed={
                "dataset_quality_report_id": dataset_report.id,
                "dataset_quality_status": dataset_report.status,
            },
        )

    def _persist_training_report(
        self,
        report: QualityReport,
        dataset_version: DatasetVersion,
        horizon: int | None,
        requested_horizons: Sequence[int],
        raise_on_failure: bool,
    ) -> str:
        """Persist a training gate report and optionally raise when it failed."""

        with session_scope(self.database_url) as session:
            record = persist_quality_report(
                StorageRepository(session),
                report,
                self.quality_policy,
                request_json={
                    "dataset_version_id": dataset_version.id,
                    "horizon_days": horizon,
                    "requested_horizons": list(requested_horizons),
                },
                dataset_version_id=dataset_version.id,
                horizon_days=horizon,
            )
            report_id = record.id
        if raise_on_failure and not report.passed:
            logger.warning(
                "Training blocked for dataset %s horizon %s by report %s",
                dataset_version.id,
                horizon,
                report_id,
            )
            raise DataQualityGateError(report, report_id)
        return report_id

    def _build_split_summary(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """Serialize split boundaries for artifact metadata."""

        return {
            "train": self._frame_summary(train_frame),
            "validation": self._frame_summary(validation_frame),
            "test": self._frame_summary(test_frame),
        }

    def _frame_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Return a date-window summary for a split frame."""

        return {
            "start_date": frame["date"].min().date(),
            "end_date": frame["date"].max().date(),
            "row_count": int(len(frame)),
        }
