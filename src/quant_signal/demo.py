"""Reproducible end-to-end governance demonstration on SYNTHETIC data.

``quant-almanac-demo run`` drives the real pipeline CLI (the same commands an operator
or scheduler would call) against a local SQLite database and artifact directory:

1. migrate a fresh database with Alembic,
2. ingest synthetic bars (including a deliberately defective ticker),
3. build a dataset that passes the quality gate,
4. attempt a build that includes the defective ticker and confirm it is blocked,
5. train calibrated candidates, backtest and explain the 5-day champion,
6. reproduce the champion by refitting from its dataset artifact,
7. verify every artifact hash and lineage pointer.

Every step's JSON output is written to ``<workdir>/demo_summary.json`` and the run
fails loudly if any expected outcome (including the expected *failure*) is not met.
``quant-almanac-demo serve`` then starts the read-only console on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config

from quant_signal.cli.pipeline import main as pipeline_main
from quant_signal.core.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = Path("demo")
DEMO_START = "2021-01-04"
DEMO_END = "2024-06-28"
BENCHMARK = "SYNIDX"
MODELING_SYMBOLS = ("SYNA", "SYNB", "SYNC", "SYND", "SYNE")
DEFECTIVE_SYMBOL = "BADQ"


class DemoError(RuntimeError):
    """Raised when the demonstration does not produce the expected evidence."""


@dataclass
class DemoResult:
    """Identifiers and outputs produced by one demo run."""

    workdir: Path
    database_url: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    ids: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return the persisted demo summary."""

        return {
            "workdir": str(self.workdir),
            "database_url": self.database_url,
            "data": "SYNTHETIC — seeded random walk; not market data",
            "ids": self.ids,
            "steps": self.steps,
        }


def demo_environment(workdir: Path, database_url: str | None = None) -> dict[str, str]:
    """Return the environment variables that configure the demo deployment.

    The default database is a SQLite file in ``workdir``; pass ``database_url`` to run
    the same demonstration against PostgreSQL (CI does this).
    """

    resolved = workdir.resolve()
    return {
        "APP_ENV": "demo",
        "DATABASE_URL": database_url
        or f"sqlite+pysqlite:///{resolved / 'quant_almanac_demo.sqlite3'}",
        "ARTIFACT_ROOT": str(resolved / "artifacts"),
        "MARKET_DATA_PROVIDER": "synthetic",
        "BENCHMARK_SYMBOL": BENCHMARK,
        "UNIVERSE_SYMBOLS": ",".join([*MODELING_SYMBOLS, BENCHMARK]),
        "DEFAULT_HORIZONS": "1,5,20",
        "MIN_TRAINING_DAYS": "252",
        "TOP_N_SIGNALS": "2",
        "BACKTEST_TRANSACTION_COST_BPS": "5",
        "BACKTEST_SLIPPAGE_BPS": "2",
        "LOG_LEVEL": "WARNING",
    }


def demo_settings(workdir: Path, database_url: str | None = None) -> Settings:
    """Build explicit settings for the demo (independent of any local .env file)."""

    env = demo_environment(workdir, database_url)
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_env=env["APP_ENV"],
        database_url=env["DATABASE_URL"],
        artifact_root=Path(env["ARTIFACT_ROOT"]),
        market_data_provider=env["MARKET_DATA_PROVIDER"],
        benchmark_symbol=env["BENCHMARK_SYMBOL"],
        universe_symbols=env["UNIVERSE_SYMBOLS"].split(","),
        default_horizons=[int(value) for value in env["DEFAULT_HORIZONS"].split(",")],
        min_training_days=int(env["MIN_TRAINING_DAYS"]),
        top_n_signals=int(env["TOP_N_SIGNALS"]),
        backtest_transaction_cost_bps=float(env["BACKTEST_TRANSACTION_COST_BPS"]),
        backtest_slippage_bps=float(env["BACKTEST_SLIPPAGE_BPS"]),
        log_level=env["LOG_LEVEL"],
    )


def migrate(database_url: str) -> None:
    """Apply every Alembic migration to the demo database."""

    # No ini file: alembic's logging config would otherwise reconfigure the root logger.
    config = Config()
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def _cli(
    result: DemoResult,
    settings: Settings,
    name: str,
    argv: Sequence[str],
    expect_exit: int = 0,
    echo: bool = True,
) -> dict[str, Any]:
    stdout, stderr = StringIO(), StringIO()
    exit_code = pipeline_main(list(argv), settings=settings, stdout=stdout, stderr=stderr)
    raw = stdout.getvalue().strip() or stderr.getvalue().strip()
    payload: dict[str, Any] = json.loads(raw.splitlines()[-1]) if raw else {}
    result.steps.append(
        {
            "step": name,
            "command": "quant-signal-pipeline " + " ".join(argv),
            "exit_code": exit_code,
            "output": payload,
        }
    )
    if echo:
        verdict = payload.get("status", "")
        print(
            f"  [{'ok' if exit_code == expect_exit else 'UNEXPECTED'}] {name}: "
            f"exit {exit_code} {verdict}"
        )
    if exit_code != expect_exit:
        raise DemoError(
            f"Step '{name}' exited {exit_code} (expected {expect_exit}): {json.dumps(payload)}"
        )
    return payload


def run_demo(
    workdir: Path = DEFAULT_WORKDIR,
    reset: bool = True,
    echo: bool = True,
    database_url: str | None = None,
) -> DemoResult:
    """Execute the full demonstration and return its evidence.

    With an external ``database_url`` the database must be empty (``reset`` only
    clears the local workdir); the demo never drops data it did not create.
    """

    if reset and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    settings = demo_settings(workdir, database_url)
    result = DemoResult(workdir=workdir, database_url=settings.database_url)

    if echo:
        print(f"Quant Almanac governance demo (SYNTHETIC data) → {workdir.resolve()}")
    migrate(settings.database_url)
    result.steps.append({"step": "migrate", "command": "alembic upgrade head", "exit_code": 0})

    _cli(
        result,
        settings,
        "ingest synthetic bars",
        [
            "ingest",
            "--start-date",
            DEMO_START,
            "--end-date",
            DEMO_END,
            "--symbols",
            *MODELING_SYMBOLS,
            DEFECTIVE_SYMBOL,
        ],
        echo=echo,
    )

    dataset = _cli(
        result,
        settings,
        "build dataset (expected: passes gate)",
        ["build-dataset", "--as-of-date", DEMO_END, "--symbols", *MODELING_SYMBOLS],
        echo=echo,
    )
    result.ids["dataset_version_id"] = str(dataset["dataset_version_id"])
    result.ids["dataset_quality_report_id"] = str(dataset["quality"]["report_id"])
    if dataset["quality"]["status"] == "fail":
        raise DemoError("The clean dataset unexpectedly failed its quality gate.")

    blocked = _cli(
        result,
        settings,
        "build dataset with defective ticker (expected: blocked)",
        [
            "build-dataset",
            "--as-of-date",
            DEMO_END,
            "--symbols",
            MODELING_SYMBOLS[0],
            DEFECTIVE_SYMBOL,
        ],
        expect_exit=1,
        echo=echo,
    )
    if blocked.get("error_type") != "DataQualityGateError":
        raise DemoError(f"Expected a quality-gate failure, got: {blocked}")
    result.ids["blocked_quality_report_id"] = str(blocked["quality_report_id"])

    trained = _cli(
        result,
        settings,
        "train calibrated candidates (1d, 5d, 20d)",
        ["train", "--dataset-version-id", result.ids["dataset_version_id"]],
        echo=echo,
    )
    champion = next(
        str(model["model_version_id"])
        for model in trained["champion_models"]
        if model["horizon_days"] == 5
    )
    result.ids["champion_5d_model_version_id"] = champion

    _cli(
        result,
        settings,
        "walk-forward backtest of 5d champion",
        ["backtest", "--model-version-id", champion],
        echo=echo,
    )
    _cli(
        result,
        settings,
        "SHAP explanations for 5d champion",
        ["explain", "--model-version-id", champion, "--sample-size", "64"],
        echo=echo,
    )
    reproduced = _cli(
        result,
        settings,
        "reproduce 5d champion from artifacts",
        ["reproduce-model", "--model-version-id", champion],
        echo=echo,
    )
    if reproduced.get("status") == "fail":
        raise DemoError(f"Reproduction failed: {reproduced}")
    result.ids["reproduction_report_id"] = str(reproduced["quality_report_id"])

    verified = _cli(
        result,
        settings,
        "verify 5d champion lineage and artifacts",
        ["verify-model", "--model-version-id", champion],
        echo=echo,
    )
    if verified.get("status") == "fail":
        raise DemoError(f"Verification failed: {verified}")

    summary_path = workdir / "demo_summary.json"
    summary_path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True), "utf-8")
    if echo:
        print(f"\nEvidence summary: {summary_path}")
        print("Start the console:  quant-almanac-demo serve")
        print(f"Champion page:      http://127.0.0.1:8000/console/models/{champion}")
        print(
            "Blocked build:      http://127.0.0.1:8000/console/quality-reports/"
            f"{result.ids['blocked_quality_report_id']}"
        )
    return result


def serve(workdir: Path, host: str, port: int) -> None:
    """Serve the read-only console against the demo database."""

    import uvicorn

    from quant_signal.core.config import get_settings
    from quant_signal.storage.db import clear_engine_cache

    if not (workdir / "quant_almanac_demo.sqlite3").exists():
        raise SystemExit(f"No demo database in {workdir}. Run `quant-almanac-demo run` first.")
    os.environ.update(demo_environment(workdir))
    get_settings.cache_clear()
    clear_engine_cache()
    uvicorn.run("quant_signal.api.app:create_app", factory=True, host=host, port=port)


def build_parser() -> argparse.ArgumentParser:
    """Build the demo command parser."""

    parser = argparse.ArgumentParser(
        prog="quant-almanac-demo",
        description="Reproducible governance demo on synthetic data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run the full pipeline demonstration.")
    run_parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    run_parser.add_argument(
        "--database-url",
        help="Run against this (empty) database instead of a local SQLite file.",
    )
    run_parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep an existing workdir instead of starting from scratch.",
    )
    serve_parser = sub.add_parser("serve", help="Serve the console for the demo database.")
    serve_parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Defaults to loopback: the console has no authentication.",
    )
    serve_parser.add_argument("--port", type=int, default=8000)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point."""

    args = build_parser().parse_args(argv)
    if args.command == "run":
        try:
            run_demo(args.workdir, reset=not args.keep, database_url=args.database_url)
        except DemoError as exc:
            print(f"DEMO FAILED: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return
    serve(args.workdir, args.host, args.port)


if __name__ == "__main__":
    run()
