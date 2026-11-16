"""CLI tests for governance commands and quality-gate error payloads."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from governance_fixtures import END, build_environment, governed_settings, ingest

from quant_signal.cli.pipeline import main


def run_cli(argv: list[str], settings: object) -> tuple[int, dict[str, object], str]:
    stdout, stderr = StringIO(), StringIO()
    exit_code = main(argv, settings=settings, stdout=stdout, stderr=stderr)  # type: ignore[arg-type]
    raw = stdout.getvalue().strip() or stderr.getvalue().strip()
    return exit_code, json.loads(raw.splitlines()[-1]), stderr.getvalue()


def test_blocked_build_exits_nonzero_with_evidence_id(tmp_path: Path) -> None:
    settings = governed_settings(tmp_path)
    ingest(settings, ["SYNA", "BADQ"])
    exit_code, payload, stderr = run_cli(
        ["build-dataset", "--as-of-date", END.isoformat(), "--symbols", "SYNA", "BADQ"],
        settings,
    )
    assert exit_code == 1
    assert payload["error_type"] == "DataQualityGateError"
    assert payload["quality"]["status"] == "fail"  # type: ignore[index]
    assert "bars.ohlc_consistency" in payload["quality"]["failed_checks"]  # type: ignore[index]

    listed_code, listed, _ = run_cli(["quality-reports", "--status", "fail"], settings)
    assert listed_code == 0
    reports = listed["reports"]
    assert [report["quality_report_id"] for report in reports] == [  # type: ignore[union-attr]
        payload["quality_report_id"]
    ]
    shown_code, shown, _ = run_cli(
        ["quality-reports", "--report-id", str(payload["quality_report_id"])], settings
    )
    assert shown_code == 0
    assert shown["report"]["status"] == "fail"  # type: ignore[index]


def test_build_dataset_summary_includes_quality_verdict(tmp_path: Path) -> None:
    settings = governed_settings(tmp_path)
    ingest(settings)
    exit_code, payload, _ = run_cli(
        ["build-dataset", "--as-of-date", END.isoformat(), "--symbols", "SYNA", "SYNB"],
        settings,
    )
    assert exit_code == 0
    assert payload["quality"]["status"] == "pass"  # type: ignore[index]


def test_verify_and_reproduce_commands(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()

    code, verified, _ = run_cli(["verify-model", "--model-version-id", champion.id], env.settings)
    assert code == 0
    assert verified["failure_count"] == 0
    assert verified["stage"] == "verification"

    code, reproduced, _ = run_cli(
        ["reproduce-model", "--model-version-id", champion.id], env.settings
    )
    assert code == 0
    assert reproduced["status"] == "pass"
    assert reproduced["quality_report_id"]

    with Path(champion.artifact_path).open("ab") as handle:
        handle.write(b"tamper")
    code, failed, _ = run_cli(["verify-model", "--model-version-id", champion.id], env.settings)
    assert code == 1
    assert "artifact.model_integrity" in failed["failed_checks"]  # type: ignore[operator]

    code, error, _ = run_cli(["verify-model", "--model-version-id", "missing"], env.settings)
    assert code == 1
    assert error["error_type"] == "GovernanceNotFoundError"
