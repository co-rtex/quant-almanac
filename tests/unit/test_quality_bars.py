"""Unit tests for source-bar quality checks."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_signal.quality.bars import run_bar_checks
from quant_signal.quality.policy import DataQualityPolicy
from quant_signal.quality.report import QualityReport

AS_OF = date(2024, 3, 29)


def make_bars(symbols: tuple[str, ...] = ("AAA", "SPY"), periods: int = 60) -> pd.DataFrame:
    rows = []
    for offset, symbol in enumerate(symbols):
        for index, timestamp in enumerate(pd.bdate_range("2024-01-02", periods=periods)):
            close = 100.0 + offset * 10 + index * 0.1 + (index % 3) * 0.05
            rows.append(
                {
                    "symbol": symbol,
                    "date": timestamp.date(),
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000_000 + index,
                    "source": "fixture",
                }
            )
    return pd.DataFrame(rows)


def run(bars: pd.DataFrame, policy: DataQualityPolicy | None = None) -> QualityReport:
    return QualityReport(
        stage="dataset_build",
        checks=tuple(run_bar_checks(bars, ["AAA"], "SPY", AS_OF, policy or DataQualityPolicy())),
    )


def status(report: QualityReport, check_id: str) -> str:
    return next(check.status for check in report.checks if check.check_id == check_id)


def test_clean_bars_pass_every_check() -> None:
    report = run(make_bars())
    assert report.status == "pass", report.summary_json()
    assert len(report.checks) == 13


def test_missing_columns_fail_schema_and_short_circuit() -> None:
    report = run(make_bars().drop(columns=["volume"]))
    assert [check.check_id for check in report.checks] == ["bars.schema"]
    assert report.status == "fail"


def test_duplicate_keys_from_multiple_sources_fail() -> None:
    bars = make_bars()
    duplicate = bars.iloc[[5]].assign(source="other-provider")
    report = run(pd.concat([bars, duplicate], ignore_index=True))
    assert status(report, "bars.duplicate_keys") == "fail"
    check = next(check for check in report.checks if check.check_id == "bars.duplicate_keys")
    assert check.observed["sources"] == ["fixture", "other-provider"]


def test_weekend_and_future_dates_fail() -> None:
    bars = make_bars()
    bars.loc[0, "date"] = date(2024, 1, 6)
    bars.loc[1, "date"] = date(2024, 4, 2)
    report = run(bars)
    check = next(check for check in report.checks if check.check_id == "bars.session_dates")
    assert check.status == "fail"
    assert check.observed["weekend_rows"] == 1
    assert check.observed["after_as_of_rows"] == 1


@pytest.mark.parametrize("column", ["open", "close", "adjusted_close"])
def test_non_positive_prices_fail(column: str) -> None:
    bars = make_bars()
    bars.loc[10, column] = 0.0
    assert status(run(bars), "bars.positive_prices") == "fail"


def test_inverted_high_low_fails_ohlc_consistency() -> None:
    bars = make_bars()
    bars.loc[3, ["high", "low"]] = bars.loc[3, ["low", "high"]].to_numpy()
    assert status(run(bars), "bars.ohlc_consistency") == "fail"


def test_zero_volume_is_a_warning_but_negative_volume_fails() -> None:
    bars = make_bars()
    bars.loc[4, "volume"] = 0
    assert status(run(bars), "bars.volume") == "warn"
    bars.loc[5, "volume"] = -1
    assert status(run(bars), "bars.volume") == "fail"


def test_missing_requested_symbol_fails_coverage() -> None:
    bars = make_bars(symbols=("SPY",))
    assert status(run(bars), "bars.symbol_coverage") == "fail"


def test_small_calendar_gap_warns_and_large_gap_fails() -> None:
    bars = make_bars()
    aaa_dates = bars.loc[bars["symbol"] == "AAA", "date"].tolist()
    small_gap = bars[~((bars["symbol"] == "AAA") & (bars["date"] == aaa_dates[20]))]
    assert status(run(small_gap), "bars.calendar_alignment") == "warn"
    large_gap = bars[~((bars["symbol"] == "AAA") & bars["date"].isin(aaa_dates[10:20]))]
    assert status(run(large_gap), "bars.calendar_alignment") == "fail"


def test_late_listing_is_not_penalized_as_a_gap() -> None:
    bars = make_bars()
    first_dates = bars.loc[bars["symbol"] == "AAA", "date"].tolist()[:30]
    ipo = bars[~((bars["symbol"] == "AAA") & bars["date"].isin(first_dates))]
    assert status(run(ipo), "bars.calendar_alignment") == "pass"


def test_symbol_ending_early_warns() -> None:
    bars = make_bars()
    last_dates = bars.loc[bars["symbol"] == "AAA", "date"].tolist()[-10:]
    stale = bars[~((bars["symbol"] == "AAA") & bars["date"].isin(last_dates))]
    assert status(run(stale), "bars.trailing_coverage") == "warn"


def test_extreme_returns_and_stale_runs_warn_without_failing() -> None:
    bars = make_bars()
    index = bars.index[(bars["symbol"] == "AAA")][30]
    bars.loc[index, ["close", "adjusted_close", "high"]] = [200.0, 200.0, 200.5]
    report = run(bars)
    assert status(report, "bars.extreme_returns") == "warn"
    stale = make_bars()
    rows = stale.index[stale["symbol"] == "AAA"][10:16]
    stale.loc[rows, ["open", "high", "low", "close", "adjusted_close"]] = [
        101.0,
        101.5,
        100.5,
        101.0,
        101.0,
    ]
    stale_report = run(stale)
    assert status(stale_report, "bars.stale_prices") == "warn"
    assert stale_report.passed


def test_benchmark_calendar_tolerates_holidays_but_fails_large_gaps() -> None:
    bars = make_bars(periods=120)
    spy_dates = bars.loc[bars["symbol"] == "SPY", "date"].tolist()
    holidays = bars[~((bars["symbol"] == "SPY") & bars["date"].isin(spy_dates[30:31]))]
    assert status(run(holidays), "bars.benchmark_calendar") == "pass"
    gapped = bars[~((bars["symbol"] == "SPY") & bars["date"].isin(spy_dates[::4]))]
    report = run(gapped)
    assert status(report, "bars.benchmark_calendar") == "fail"
    long_run = bars[~((bars["symbol"] == "SPY") & bars["date"].isin(spy_dates[40:46]))]
    assert status(run(long_run), "bars.benchmark_calendar") == "fail"
