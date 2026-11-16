"""Unit tests for governance statistics, artifact integrity, reports, and views."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pytest

from quant_signal.console.views import calibration_chart, split_timeline
from quant_signal.core.hashing import sha256_file
from quant_signal.governance.statistics import hanley_mcneil_interval
from quant_signal.ingestion.synthetic import SyntheticMarketDataProvider
from quant_signal.quality import (
    ArtifactIntegrityError,
    CheckResult,
    QualityReport,
    check_artifact_integrity,
    require_artifact_integrity,
    worst_status,
)


def test_hanley_mcneil_matches_closed_form_at_chance() -> None:
    interval = hanley_mcneil_interval(0.5, 100, 0.5)
    assert interval is not None
    expected = math.sqrt((0.25 + 49 * (1 / 3 - 0.25) * 2) / 2500)
    assert interval.standard_error == pytest.approx(expected)
    assert interval.includes_chance


def test_hanley_mcneil_handles_missing_and_degenerate_inputs() -> None:
    assert hanley_mcneil_interval(None, 100, 0.5) is None
    assert hanley_mcneil_interval(0.7, 100, 0.0) is None
    strong = hanley_mcneil_interval(0.8, 2000, 0.5)
    assert strong is not None and not strong.includes_chance


def test_artifact_integrity_states(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"evidence")
    digest = sha256_file(artifact)
    assert check_artifact_integrity(artifact, digest, "dataset").status == "pass"
    assert check_artifact_integrity(artifact, None, "dataset").status == "warn"
    assert check_artifact_integrity(tmp_path / "missing", digest, "dataset").status == "fail"
    artifact.write_bytes(b"tampered")
    result = check_artifact_integrity(artifact, digest, "dataset")
    assert result.status == "fail"
    with pytest.raises(ArtifactIntegrityError):
        require_artifact_integrity(artifact, digest, "dataset")


def test_report_status_is_worst_and_round_trips() -> None:
    checks = (
        CheckResult("a", "x", "pass", "ok"),
        CheckResult("b", "x", "warn", "hmm", observed={"n": 1}),
    )
    report = QualityReport(stage="training", checks=checks)
    assert report.status == "warn" and report.passed
    assert worst_status(["pass", "fail", "warn"]) == "fail"
    assert CheckResult.from_json(checks[1].to_json()) == checks[1]
    failed = report.extend([CheckResult("c", "x", "fail", "bad")])
    assert failed.summary_json()["failed_checks"] == ["c"]


def test_synthetic_provider_is_deterministic_and_window_invariant() -> None:
    provider = SyntheticMarketDataProvider()
    full = provider.fetch_daily_bars(["SYNA"], date(2022, 1, 3), date(2022, 6, 30))
    window = provider.fetch_daily_bars(["SYNA"], date(2022, 3, 1), date(2022, 6, 30))
    assert [bar for bar in full.bars if bar.trade_date >= date(2022, 3, 1)] == window.bars
    assert full.provider_metadata["synthetic"] is True
    assert all(bar.trade_date.weekday() < 5 for bar in full.bars)
    assert all(bar.low <= min(bar.open, bar.close) for bar in full.bars)


def test_synthetic_provider_injects_documented_defects_only_for_bad_tickers() -> None:
    provider = SyntheticMarketDataProvider()
    result = provider.fetch_daily_bars(["BADX", "SYNA"], date(2022, 1, 3), date(2023, 12, 29))
    bad = {bar.trade_date: bar for bar in result.bars if bar.symbol == "BADX"}
    assert bad[date(2023, 3, 15)].high < bad[date(2023, 3, 15)].low
    assert bad[date(2023, 6, 14)].close == 0.0
    assert date(2022, 8, 1) not in bad
    good = {bar.trade_date for bar in result.bars if bar.symbol == "SYNA"}
    assert date(2022, 8, 1) in good


def test_chart_geometry_helpers() -> None:
    chart = calibration_chart(
        [{"mean_prediction": 0.5, "observed_rate": 0.5, "count": 3, "lower": 0.4, "upper": 0.5}]
    )
    assert chart["points"][0]["count"] == 3
    timeline = split_timeline(
        {"split": {"train_dates": 60, "validation_dates": 20, "test_dates": 20, "embargo_days": 5}}
    )
    assert timeline["available"]
    assert [segment["kind"] for segment in timeline["segments"]] == [
        "train",
        "embargo",
        "validation",
        "embargo",
        "test",
    ]
    assert [item["kind"] for item in timeline["legend"]] == [
        "train",
        "embargo",
        "validation",
        "test",
    ]
    assert split_timeline(None) == {"available": False}


def test_code_revision_counts_untracked_files_and_rejects_foreign_repos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_signal.core import provenance

    package_dir = Path(provenance.__file__).resolve().parents[1]
    repo_root = str(package_dir.parents[1])
    answers = {"rev-parse --show-toplevel": repo_root, "rev-parse HEAD": "abc123"}

    def fake_git(args: list[str], cwd: Path) -> str | None:
        key = " ".join(args)
        if key.startswith("status"):
            assert "--untracked-files=no" not in args
            return "?? src/quant_signal/new_module.py"
        return answers.get(key)

    monkeypatch.delenv(provenance.CODE_REVISION_ENV, raising=False)
    monkeypatch.setattr(provenance, "_git", fake_git)
    assert provenance.code_revision() == {"commit": "abc123", "dirty": True, "source": "git"}

    answers["rev-parse --show-toplevel"] = "/somewhere/else"
    assert provenance.code_revision()["source"] == "unavailable"

    monkeypatch.setenv(provenance.CODE_REVISION_ENV, "release-1.2")
    assert provenance.code_revision()["commit"] == "release-1.2"
