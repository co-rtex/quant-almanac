"""Shared builders for governance integration tests (SYNTHETIC data only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quant_signal.core.config import Settings
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.ingestion.service import IngestionService
from quant_signal.ingestion.synthetic import SyntheticMarketDataProvider
from quant_signal.storage.db import create_all_tables, session_scope
from quant_signal.storage.models import DatasetVersion, ModelVersion
from quant_signal.storage.repositories import StorageRepository
from quant_signal.training.service import TrainingService

START = date(2022, 1, 3)
END = date(2023, 3, 31)
SYMBOLS = ["SYNA", "SYNB", "SYNC"]


@dataclass
class GovernedEnvironment:
    """A migrated database with ingested bars, a dataset, and trained models."""

    settings: Settings
    dataset: DatasetVersion
    models: list[ModelVersion]

    def champion(self, horizon: int = 5) -> ModelVersion:
        return next(
            model
            for model in self.models
            if model.horizon_days == horizon and model.champion_rank == 1
        )


def governed_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=f"sqlite+pysqlite:///{tmp_path / 'governance.sqlite3'}",
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SYNIDX",
        universe_symbols=[*SYMBOLS, "SYNIDX"],
        default_horizons=[1, 5],
        min_training_days=80,
        top_n_signals=1,
        market_data_provider="synthetic",
    )


def ingest(settings: Settings, symbols: list[str] | None = None) -> None:
    create_all_tables(settings.database_url)
    IngestionService(
        provider=SyntheticMarketDataProvider(),
        settings=settings,
        sleep_fn=lambda _: None,
    ).ingest_daily_bars(symbols or SYMBOLS, START, END)


def build_environment(tmp_path: Path, train: bool = True) -> GovernedEnvironment:
    settings = governed_settings(tmp_path)
    ingest(settings)
    dataset = FeaturePipeline(settings=settings).build_dataset(END, SYMBOLS)
    models = TrainingService(settings=settings).train(dataset.id) if train else []
    return GovernedEnvironment(settings=settings, dataset=dataset, models=models)


def reload_model(settings: Settings, model_version_id: str) -> ModelVersion:
    with session_scope(settings.database_url) as session:
        return StorageRepository(session).get_model_version(model_version_id)
