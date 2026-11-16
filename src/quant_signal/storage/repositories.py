"""Repository helpers for common persistence workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from quant_signal.storage.models import (
    BacktestRun,
    DailyBar,
    DatasetVersion,
    IngestionRun,
    ModelEvaluation,
    ModelVersion,
    QualityReportRecord,
    ShapRun,
    SignalSnapshot,
    Symbol,
)


@dataclass(frozen=True)
class DailyBarRecord:
    """Normalized daily bar payload for persistence."""

    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    source: str


@dataclass(frozen=True)
class DatasetArtifactRecord:
    """Dataset artifact metadata to persist."""

    as_of_date: date
    feature_set_version: str
    horizons: list[int]
    symbols: list[str]
    row_count: int
    artifact_path: str
    artifact_hash: str
    feature_columns: list[str]
    label_columns: list[str]
    metadata_json: dict[str, object]


@dataclass(frozen=True)
class ModelArtifactRecord:
    """Model artifact metadata to persist."""

    dataset_version_id: str
    horizon_days: int
    model_family: str
    target_column: str
    artifact_path: str
    artifact_hash: str
    feature_columns: list[str]
    champion_rank: int | None
    train_start_date: date | None
    train_end_date: date | None
    validation_start_date: date | None
    validation_end_date: date | None
    test_start_date: date | None
    test_end_date: date | None
    metadata_json: dict[str, object]


@dataclass(frozen=True)
class EvaluationRecord:
    """Model evaluation payload to persist."""

    model_version_id: str
    split_name: str
    metrics_json: dict[str, object]
    calibration_bins: list[dict[str, float]]


@dataclass(frozen=True)
class SignalSnapshotRecord:
    """Ranked signal snapshot payload."""

    model_version_id: str
    horizon_days: int
    as_of_date: date
    symbol: str
    score: float
    rank: int
    metadata_json: dict[str, object]


@dataclass(frozen=True)
class BacktestRunRecord:
    """Backtest result payload."""

    model_version_id: str
    horizon_days: int
    top_n: int
    min_training_days: int
    status: str
    artifact_path: str
    artifact_hash: str
    summary_json: dict[str, object]
    regime_summary_json: dict[str, dict[str, object]]
    metadata_json: dict[str, object]


@dataclass(frozen=True)
class ShapRunRecord:
    """SHAP run payload."""

    model_version_id: str
    sample_size: int
    artifact_path: str
    artifact_hash: str
    summary_json: dict[str, object]


@dataclass(frozen=True)
class QualityReportPayload:
    """Quality or governance gate evidence to persist."""

    stage: str
    status: str
    policy_version: str
    check_count: int
    failure_count: int
    warning_count: int
    summary: str
    checks_json: list[dict[str, object]]
    request_json: dict[str, object]
    context_json: dict[str, object]
    dataset_version_id: str | None = None
    model_version_id: str | None = None
    horizon_days: int | None = None


@dataclass(frozen=True)
class BarProvenanceRow:
    """Aggregated provenance for the persisted bars feeding a dataset build."""

    symbol: str
    source: str
    ingestion_run_id: str | None
    row_count: int
    first_date: date
    last_date: date


@dataclass(frozen=True)
class SignalSnapshotStats:
    """Aggregate statistics for one model version's persisted signal snapshots."""

    snapshot_count: int
    first_as_of_date: date | None
    last_as_of_date: date | None


class StorageRepository:
    """Repository façade over common SQLAlchemy operations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_symbols(
        self,
        symbols: Sequence[str],
        benchmark_symbol: str | None = None,
    ) -> dict[str, Symbol]:
        """Insert missing symbols and return a symbol map keyed by ticker."""

        normalized = sorted({symbol.upper() for symbol in symbols})
        if not normalized:
            return {}

        existing = self.session.execute(
            select(Symbol).where(Symbol.symbol.in_(normalized))
        ).scalars()
        symbol_map = {symbol.symbol: symbol for symbol in existing}

        for ticker in normalized:
            if ticker in symbol_map:
                symbol_map[ticker].is_benchmark = ticker == benchmark_symbol
                continue
            symbol = Symbol(symbol=ticker, is_benchmark=ticker == benchmark_symbol)
            self.session.add(symbol)
            symbol_map[ticker] = symbol

        self.session.flush()
        return symbol_map

    def create_ingestion_run(
        self,
        provider: str,
        start_date: date,
        end_date: date,
        requested_symbols: Sequence[str],
        metadata_json: dict[str, object] | None = None,
    ) -> IngestionRun:
        """Create a pending ingestion run."""

        run = IngestionRun(
            provider=provider,
            status="pending",
            start_date=start_date,
            end_date=end_date,
            requested_symbols=list(requested_symbols),
            metadata_json=metadata_json or {},
        )
        self.session.add(run)
        self.session.flush()
        return run

    def finalize_ingestion_run(
        self,
        run_id: str,
        status: str,
        records_written: int,
        metadata_json: dict[str, object] | None = None,
    ) -> IngestionRun:
        """Update the terminal state for an ingestion run."""

        run = self.session.get(IngestionRun, run_id)
        if run is None:
            raise ValueError(f"Unknown ingestion run: {run_id}")

        run.status = status
        run.records_written = records_written
        if metadata_json:
            run.metadata_json = {**run.metadata_json, **metadata_json}
        from quant_signal.core.time import utc_now

        run.completed_at = utc_now()
        self.session.flush()
        return run

    def upsert_daily_bars(
        self,
        records: Sequence[DailyBarRecord],
        ingestion_run_id: str | None = None,
    ) -> int:
        """Insert or update normalized daily bars."""

        if not records:
            return 0

        symbols = sorted({record.symbol.upper() for record in records})
        symbol_map = self.upsert_symbols(symbols)
        min_date = min(record.trade_date for record in records)
        max_date = max(record.trade_date for record in records)

        existing_rows = self.session.execute(
            select(DailyBar, Symbol.symbol)
            .join(Symbol, DailyBar.symbol_id == Symbol.id)
            .where(Symbol.symbol.in_(symbols))
            .where(DailyBar.trade_date >= min_date)
            .where(DailyBar.trade_date <= max_date)
        ).all()
        existing_map = {
            (symbol, row.trade_date, row.source): row
            for row, symbol in existing_rows
        }

        for record in records:
            key = (record.symbol.upper(), record.trade_date, record.source)
            current = existing_map.get(key)
            if current is None:
                current = DailyBar(
                    symbol_id=symbol_map[record.symbol.upper()].id,
                    ingestion_run_id=ingestion_run_id,
                    trade_date=record.trade_date,
                    open=record.open,
                    high=record.high,
                    low=record.low,
                    close=record.close,
                    adjusted_close=record.adjusted_close,
                    volume=record.volume,
                    source=record.source,
                )
                self.session.add(current)
                continue

            current.ingestion_run_id = ingestion_run_id
            current.open = record.open
            current.high = record.high
            current.low = record.low
            current.close = record.close
            current.adjusted_close = record.adjusted_close
            current.volume = record.volume

        self.session.flush()
        return len(records)

    def load_daily_bars_frame(
        self,
        symbols: Sequence[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Load normalized daily bars into a pandas DataFrame."""

        normalized = sorted({symbol.upper() for symbol in symbols})
        if not normalized:
            return pd.DataFrame()

        statement = (
            select(
                Symbol.symbol.label("symbol"),
                DailyBar.trade_date.label("date"),
                DailyBar.open,
                DailyBar.high,
                DailyBar.low,
                DailyBar.close,
                DailyBar.adjusted_close,
                DailyBar.volume,
                DailyBar.source,
            )
            .join(Symbol, DailyBar.symbol_id == Symbol.id)
            .where(Symbol.symbol.in_(normalized))
        )
        if start_date is not None:
            statement = statement.where(DailyBar.trade_date >= start_date)
        if end_date is not None:
            statement = statement.where(DailyBar.trade_date <= end_date)

        rows = self.session.execute(
            statement.order_by(DailyBar.trade_date, Symbol.symbol)
        ).mappings()
        return pd.DataFrame(rows)

    def create_dataset_version(self, record: DatasetArtifactRecord) -> DatasetVersion:
        """Persist a dataset artifact manifest."""

        dataset_version = DatasetVersion(
            as_of_date=record.as_of_date,
            feature_set_version=record.feature_set_version,
            horizons=record.horizons,
            symbols=record.symbols,
            row_count=record.row_count,
            artifact_path=record.artifact_path,
            artifact_hash=record.artifact_hash,
            feature_columns=record.feature_columns,
            label_columns=record.label_columns,
            metadata_json=record.metadata_json,
        )
        self.session.add(dataset_version)
        self.session.flush()
        return dataset_version

    def get_dataset_version(self, dataset_version_id: str) -> DatasetVersion:
        """Return a dataset version by id."""

        dataset_version = self.session.get(DatasetVersion, dataset_version_id)
        if dataset_version is None:
            raise ValueError(f"Unknown dataset version: {dataset_version_id}")
        return dataset_version

    def create_model_version(self, record: ModelArtifactRecord) -> ModelVersion:
        """Persist model artifact metadata."""

        model_version = ModelVersion(
            dataset_version_id=record.dataset_version_id,
            horizon_days=record.horizon_days,
            model_family=record.model_family,
            target_column=record.target_column,
            artifact_path=record.artifact_path,
            artifact_hash=record.artifact_hash,
            feature_columns=record.feature_columns,
            champion_rank=record.champion_rank,
            train_start_date=record.train_start_date,
            train_end_date=record.train_end_date,
            validation_start_date=record.validation_start_date,
            validation_end_date=record.validation_end_date,
            test_start_date=record.test_start_date,
            test_end_date=record.test_end_date,
            metadata_json=record.metadata_json,
        )
        self.session.add(model_version)
        self.session.flush()
        return model_version

    def create_model_evaluation(self, record: EvaluationRecord) -> ModelEvaluation:
        """Persist evaluation metrics for a model split."""

        evaluation = ModelEvaluation(
            model_version_id=record.model_version_id,
            split_name=record.split_name,
            roc_auc=record.metrics_json.get("roc_auc"),
            pr_auc=record.metrics_json.get("pr_auc"),
            brier_score=record.metrics_json.get("brier_score"),
            calibration_error=record.metrics_json.get("calibration_error"),
            metrics_json=record.metrics_json,
            calibration_bins=record.calibration_bins,
        )
        self.session.add(evaluation)
        self.session.flush()
        return evaluation

    def replace_signal_snapshots(
        self,
        model_version_id: str,
        records: Sequence[SignalSnapshotRecord],
    ) -> None:
        """Replace persisted signal snapshots for a model version."""

        self.session.execute(
            delete(SignalSnapshot).where(SignalSnapshot.model_version_id == model_version_id)
        )
        for record in records:
            self.session.add(
                SignalSnapshot(
                    model_version_id=record.model_version_id,
                    horizon_days=record.horizon_days,
                    as_of_date=record.as_of_date,
                    symbol=record.symbol,
                    score=record.score,
                    rank=record.rank,
                    metadata_json=record.metadata_json,
                )
            )
        self.session.flush()

    def get_model_version(self, model_version_id: str) -> ModelVersion:
        """Return a model version by id."""

        model_version = self.session.get(ModelVersion, model_version_id)
        if model_version is None:
            raise ValueError(f"Unknown model version: {model_version_id}")
        return model_version

    def list_model_evaluations(self, model_version_id: str) -> list[ModelEvaluation]:
        """Return evaluations for a model version ordered by split name."""

        return list(
            self.session.execute(
                select(ModelEvaluation)
                .where(ModelEvaluation.model_version_id == model_version_id)
                .order_by(ModelEvaluation.split_name)
            ).scalars()
        )

    def list_champion_models(self, horizon_days: int | None = None) -> list[ModelVersion]:
        """Return champion model versions ordered by newest first."""

        statement = select(ModelVersion).where(ModelVersion.champion_rank == 1)
        if horizon_days is not None:
            statement = statement.where(ModelVersion.horizon_days == horizon_days)
        statement = statement.order_by(ModelVersion.created_at.desc())
        return list(self.session.execute(statement).scalars())

    def get_ranked_signal_snapshots(
        self,
        as_of_date: date,
        horizon_days: int,
        limit: int,
        model_version_id: str | None = None,
    ) -> list[SignalSnapshot]:
        """Return ranked persisted signal snapshots."""

        target_model_version_id = model_version_id
        if target_model_version_id is None:
            champion_models = self.list_champion_models(horizon_days=horizon_days)
            if not champion_models:
                return []
            target_model_version_id = champion_models[0].id

        statement = (
            select(SignalSnapshot)
            .where(SignalSnapshot.model_version_id == target_model_version_id)
            .where(SignalSnapshot.horizon_days == horizon_days)
            .where(SignalSnapshot.as_of_date == as_of_date)
            .order_by(SignalSnapshot.rank)
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars())

    def create_backtest_run(self, record: BacktestRunRecord) -> BacktestRun:
        """Persist a completed backtest run."""

        backtest_run = BacktestRun(
            model_version_id=record.model_version_id,
            horizon_days=record.horizon_days,
            top_n=record.top_n,
            min_training_days=record.min_training_days,
            status=record.status,
            artifact_path=record.artifact_path,
            artifact_hash=record.artifact_hash,
            summary_json=record.summary_json,
            regime_summary_json=record.regime_summary_json,
            metadata_json=record.metadata_json,
        )
        self.session.add(backtest_run)
        self.session.flush()
        return backtest_run

    def create_shap_run(self, record: ShapRunRecord) -> ShapRun:
        """Persist a SHAP explainability run."""

        shap_run = ShapRun(
            model_version_id=record.model_version_id,
            sample_size=record.sample_size,
            artifact_path=record.artifact_path,
            artifact_hash=record.artifact_hash,
            summary_json=record.summary_json,
        )
        self.session.add(shap_run)
        self.session.flush()
        return shap_run

    def create_quality_report(self, record: QualityReportPayload) -> QualityReportRecord:
        """Persist one append-only quality report."""

        report = QualityReportRecord(
            stage=record.stage,
            status=record.status,
            policy_version=record.policy_version,
            dataset_version_id=record.dataset_version_id,
            model_version_id=record.model_version_id,
            horizon_days=record.horizon_days,
            check_count=record.check_count,
            failure_count=record.failure_count,
            warning_count=record.warning_count,
            summary=record.summary,
            checks_json=record.checks_json,
            request_json=record.request_json,
            context_json=record.context_json,
        )
        self.session.add(report)
        self.session.flush()
        return report

    def get_quality_report(self, quality_report_id: str) -> QualityReportRecord:
        """Return a quality report by id."""

        report = self.session.get(QualityReportRecord, quality_report_id)
        if report is None:
            raise LookupError(f"Unknown quality report: {quality_report_id}")
        return report

    def list_quality_reports(
        self,
        stage: str | None = None,
        status: str | None = None,
        dataset_version_id: str | None = None,
        model_version_id: str | None = None,
        limit: int = 100,
    ) -> list[QualityReportRecord]:
        """Return quality reports, newest first, with optional filters."""

        statement = select(QualityReportRecord)
        if stage is not None:
            statement = statement.where(QualityReportRecord.stage == stage)
        if status is not None:
            statement = statement.where(QualityReportRecord.status == status)
        if dataset_version_id is not None:
            statement = statement.where(
                QualityReportRecord.dataset_version_id == dataset_version_id
            )
        if model_version_id is not None:
            statement = statement.where(QualityReportRecord.model_version_id == model_version_id)
        statement = statement.order_by(
            QualityReportRecord.created_at.desc(),
            QualityReportRecord.id,
        ).limit(limit)
        return list(self.session.execute(statement).scalars())

    def summarize_bar_provenance(
        self,
        symbols: Sequence[str],
        end_date: date | None = None,
    ) -> list[BarProvenanceRow]:
        """Group the bars a dataset build would read by symbol, source, and ingestion run."""

        normalized = sorted({symbol.upper() for symbol in symbols})
        if not normalized:
            return []
        statement = (
            select(
                Symbol.symbol,
                DailyBar.source,
                DailyBar.ingestion_run_id,
                func.count(DailyBar.id),
                func.min(DailyBar.trade_date),
                func.max(DailyBar.trade_date),
            )
            .join(Symbol, DailyBar.symbol_id == Symbol.id)
            .where(Symbol.symbol.in_(normalized))
            .group_by(Symbol.symbol, DailyBar.source, DailyBar.ingestion_run_id)
            .order_by(Symbol.symbol, DailyBar.source)
        )
        if end_date is not None:
            statement = statement.where(DailyBar.trade_date <= end_date)
        return [
            BarProvenanceRow(
                symbol=str(row[0]),
                source=str(row[1]),
                ingestion_run_id=row[2],
                row_count=int(row[3]),
                first_date=row[4],
                last_date=row[5],
            )
            for row in self.session.execute(statement).all()
        ]

    def list_ingestion_runs(self, ingestion_run_ids: Sequence[str]) -> list[IngestionRun]:
        """Return ingestion runs for the given ids, oldest first."""

        if not ingestion_run_ids:
            return []
        statement = (
            select(IngestionRun)
            .where(IngestionRun.id.in_(list(ingestion_run_ids)))
            .order_by(IngestionRun.started_at)
        )
        return list(self.session.execute(statement).scalars())

    def list_dataset_versions(self, limit: int = 100) -> list[DatasetVersion]:
        """Return dataset versions, newest first."""

        statement = (
            select(DatasetVersion)
            .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id)
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars())

    def list_model_versions(
        self,
        limit: int = 200,
        dataset_version_id: str | None = None,
    ) -> list[ModelVersion]:
        """Return model versions, newest first."""

        statement = select(ModelVersion)
        if dataset_version_id is not None:
            statement = statement.where(ModelVersion.dataset_version_id == dataset_version_id)
        statement = statement.order_by(
            ModelVersion.created_at.desc(),
            ModelVersion.horizon_days,
            ModelVersion.champion_rank,
            ModelVersion.id,
        ).limit(limit)
        return list(self.session.execute(statement).scalars())

    def list_backtest_runs(self, model_version_id: str) -> list[BacktestRun]:
        """Return backtest runs for a model version, newest first."""

        statement = (
            select(BacktestRun)
            .where(BacktestRun.model_version_id == model_version_id)
            .order_by(BacktestRun.created_at.desc(), BacktestRun.id)
        )
        return list(self.session.execute(statement).scalars())

    def list_shap_runs(self, model_version_id: str) -> list[ShapRun]:
        """Return SHAP runs for a model version, newest first."""

        statement = (
            select(ShapRun)
            .where(ShapRun.model_version_id == model_version_id)
            .order_by(ShapRun.created_at.desc(), ShapRun.id)
        )
        return list(self.session.execute(statement).scalars())

    def signal_snapshot_stats(self, model_version_id: str) -> SignalSnapshotStats:
        """Return snapshot counts and the published date range for a model version."""

        row = self.session.execute(
            select(
                func.count(SignalSnapshot.id),
                func.min(SignalSnapshot.as_of_date),
                func.max(SignalSnapshot.as_of_date),
            ).where(SignalSnapshot.model_version_id == model_version_id)
        ).one()
        return SignalSnapshotStats(
            snapshot_count=int(row[0]),
            first_as_of_date=row[1],
            last_as_of_date=row[2],
        )
