"""Alembic migration smoke tests."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


@pytest.fixture(autouse=True)
def isolate_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """alembic/env.py prefers DATABASE_URL (set in CI); pin migrations to the test DB."""

    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_alembic_upgrade_head(tmp_path: Path) -> None:
    """Applying the head migration should create the core schema."""

    database_path = tmp_path / "migrations.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"

    project_root = Path(__file__).resolve().parents[2]
    alembic_config = Config(str(project_root / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(project_root / "alembic"))
    alembic_config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(alembic_config, "head")

    inspector = inspect(create_engine(database_url))
    assert {
        "symbols",
        "ingestion_runs",
        "daily_bars",
        "dataset_versions",
        "model_versions",
        "model_evaluations",
        "backtest_runs",
        "shap_runs",
        "signal_snapshots",
        "quality_reports",
    }.issubset(set(inspector.get_table_names()))
    columns = {column["name"] for column in inspector.get_columns("quality_reports")}
    assert {"stage", "status", "dataset_version_id", "model_version_id", "checks_json"} <= columns
    indexes = {index["name"] for index in inspector.get_indexes("quality_reports")}
    assert "ix_quality_reports_dataset_version_id" in indexes


def test_quality_reports_migration_downgrades_cleanly(tmp_path: Path) -> None:
    """The governance migration must be reversible without touching the v1 schema."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'downgrade.sqlite3'}"
    project_root = Path(__file__).resolve().parents[2]
    alembic_config = Config(str(project_root / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(project_root / "alembic"))
    alembic_config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "20260316_0001")
    tables = set(inspect(create_engine(database_url)).get_table_names())
    assert "quality_reports" not in tables
    assert "model_versions" in tables
    command.upgrade(alembic_config, "head")
    assert "quality_reports" in set(inspect(create_engine(database_url)).get_table_names())
