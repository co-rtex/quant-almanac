"""Source-bar quality checks for daily equity OHLCV data."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from quant_signal.quality.policy import DataQualityPolicy
from quant_signal.quality.report import CheckResult, CheckStatus

REQUIRED_BAR_COLUMNS = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)
PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")


def examples(
    frame: pd.DataFrame,
    columns: Sequence[str],
    limit: int = 5,
) -> list[dict[str, object]]:
    """Return a small, JSON-safe sample of offending rows."""

    rows: list[dict[str, object]] = []
    for record in frame.loc[:, list(columns)].head(limit).to_dict(orient="records"):
        rows.append({str(key): _json_scalar(value) for key, value in record.items()})
    return rows


def _json_scalar(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def run_bar_checks(
    bars: pd.DataFrame,
    modeling_symbols: Sequence[str],
    benchmark_symbol: str,
    as_of_date: date,
    policy: DataQualityPolicy,
) -> list[CheckResult]:
    """Run all source-bar checks and return their results in a stable order."""

    schema = check_bar_schema(bars)
    if schema.status == "fail":
        return [schema]

    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()

    return [
        schema,
        check_required_values(frame),
        check_duplicate_bar_keys(frame),
        check_session_dates(frame, as_of_date),
        check_positive_prices(frame),
        check_ohlc_consistency(frame, policy),
        check_volume(frame),
        check_symbol_coverage(frame, modeling_symbols, benchmark_symbol),
        check_calendar_alignment(frame, modeling_symbols, benchmark_symbol, policy),
        check_trailing_coverage(frame, modeling_symbols, benchmark_symbol, policy),
        check_extreme_returns(frame, policy),
        check_stale_prices(frame, policy),
    ]


def check_bar_schema(bars: pd.DataFrame) -> CheckResult:
    """Required OHLCV columns must be present."""

    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in bars.columns]
    return CheckResult(
        check_id="bars.schema",
        category="schema",
        status="fail" if missing else "pass",
        message=(
            f"Missing required bar columns: {', '.join(missing)}."
            if missing
            else "All required OHLCV columns are present."
        ),
        observed={"columns": sorted(str(column) for column in bars.columns), "missing": missing},
        threshold={"required": list(REQUIRED_BAR_COLUMNS)},
    )


def check_required_values(frame: pd.DataFrame) -> CheckResult:
    """Required fields must be populated with finite values."""

    numeric = frame[[*PRICE_COLUMNS, "volume"]].apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric.isna() | ~np.isfinite(numeric.astype(float))
    invalid_mask = invalid_mask.any(axis=1) | frame["symbol"].isna() | frame["date"].isna()
    offending = frame[invalid_mask]
    return CheckResult(
        check_id="bars.required_values",
        category="integrity",
        status="fail" if len(offending) else "pass",
        message=(
            f"{len(offending)} bars have missing or non-finite required values."
            if len(offending)
            else "Every bar has finite OHLCV values, a symbol, and a date."
        ),
        observed={"invalid_rows": int(len(offending)), "total_rows": int(len(frame))},
        examples=examples(offending, ["symbol", "date", *PRICE_COLUMNS, "volume"]),
    )


def check_duplicate_bar_keys(frame: pd.DataFrame) -> CheckResult:
    """Each (symbol, session) must appear once across every ingestion source."""

    duplicated = frame[frame.duplicated(subset=["symbol", "date"], keep=False)]
    sources = sorted({str(source) for source in frame.get("source", pd.Series(dtype=str))})
    return CheckResult(
        check_id="bars.duplicate_keys",
        category="integrity",
        status="fail" if len(duplicated) else "pass",
        message=(
            f"{len(duplicated)} bars share a (symbol, date) key; features would mix sources."
            if len(duplicated)
            else "Every (symbol, date) key is unique."
        ),
        observed={"duplicate_rows": int(len(duplicated)), "sources": sources},
        examples=examples(
            duplicated.sort_values(["symbol", "date"]),
            [column for column in ("symbol", "date", "source", "close") if column in frame],
        ),
    )


def check_session_dates(frame: pd.DataFrame, as_of_date: date) -> CheckResult:
    """US equity daily bars must fall on weekdays and not after the as-of date."""

    weekend_mask = frame["date"].dt.dayofweek >= 5
    future_mask = frame["date"] > pd.Timestamp(as_of_date)
    weekend = frame[weekend_mask]
    future = frame[future_mask]
    offending = frame[weekend_mask | future_mask]
    return CheckResult(
        check_id="bars.session_dates",
        category="timestamps",
        status="fail" if len(offending) else "pass",
        message=(
            f"{len(weekend)} weekend bars and {len(future)} bars after the as-of date."
            if len(offending)
            else "All bar timestamps are weekday sessions on or before the as-of date."
        ),
        observed={
            "weekend_rows": int(len(weekend)),
            "after_as_of_rows": int(len(future)),
            "first_date": _json_scalar(frame["date"].min()) if len(frame) else None,
            "last_date": _json_scalar(frame["date"].max()) if len(frame) else None,
        },
        threshold={"as_of_date": as_of_date.isoformat()},
        examples=examples(offending, ["symbol", "date"]),
    )


def check_positive_prices(frame: pd.DataFrame) -> CheckResult:
    """Prices must be strictly positive."""

    mask = (frame[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    offending = frame[mask]
    return CheckResult(
        check_id="bars.positive_prices",
        category="sanity",
        status="fail" if len(offending) else "pass",
        message=(
            f"{len(offending)} bars contain zero or negative prices."
            if len(offending)
            else "All prices are strictly positive."
        ),
        observed={"invalid_rows": int(len(offending))},
        examples=examples(offending, ["symbol", "date", *PRICE_COLUMNS]),
    )


def check_ohlc_consistency(frame: pd.DataFrame, policy: DataQualityPolicy) -> CheckResult:
    """High must bound open/close/low from above and low must bound them from below."""

    tolerance = policy.ohlc_relative_tolerance
    upper = frame[["open", "close", "low"]].max(axis=1)
    lower = frame[["open", "close", "high"]].min(axis=1)
    mask = (frame["high"] < upper * (1 - tolerance)) | (frame["low"] > lower * (1 + tolerance))
    offending = frame[mask]
    return CheckResult(
        check_id="bars.ohlc_consistency",
        category="sanity",
        status="fail" if len(offending) else "pass",
        message=(
            f"{len(offending)} bars violate low <= open/close <= high."
            if len(offending)
            else "Every bar satisfies low <= open/close <= high."
        ),
        observed={"invalid_rows": int(len(offending))},
        threshold={"relative_tolerance": tolerance},
        examples=examples(offending, ["symbol", "date", "open", "high", "low", "close"]),
    )


def check_volume(frame: pd.DataFrame) -> CheckResult:
    """Negative volume is invalid; zero volume is a legitimate halt and only warns."""

    negative = frame[frame["volume"] < 0]
    zero = frame[frame["volume"] == 0]
    status: CheckStatus = "fail" if len(negative) else "warn" if len(zero) else "pass"
    if len(negative):
        message = f"{len(negative)} bars report negative volume."
    elif len(zero):
        message = (
            f"{len(zero)} bars report zero volume (allowed: trading halts are legitimate, "
            "but volume-change features are undefined on those sessions)."
        )
    else:
        message = "All bars report positive volume."
    return CheckResult(
        check_id="bars.volume",
        category="sanity",
        status=status,
        message=message,
        observed={"negative_rows": int(len(negative)), "zero_rows": int(len(zero))},
        examples=examples(frame[frame["volume"] <= 0], ["symbol", "date", "volume"]),
    )


def check_symbol_coverage(
    frame: pd.DataFrame,
    modeling_symbols: Sequence[str],
    benchmark_symbol: str,
) -> CheckResult:
    """Every requested symbol and the benchmark must have persisted bars."""

    present = set(frame["symbol"].unique())
    expected = sorted({*modeling_symbols, benchmark_symbol})
    missing = [symbol for symbol in expected if symbol not in present]
    per_symbol = (
        frame.groupby("symbol")["date"].agg(["count", "min", "max"]).sort_index()
        if len(frame)
        else pd.DataFrame(columns=["count", "min", "max"])
    )
    coverage = {
        str(symbol): {
            "rows": int(row["count"]),
            "first_date": _json_scalar(row["min"]),
            "last_date": _json_scalar(row["max"]),
        }
        for symbol, row in per_symbol.iterrows()
    }
    return CheckResult(
        check_id="bars.symbol_coverage",
        category="coverage",
        status="fail" if missing else "pass",
        message=(
            f"No persisted bars for requested symbols: {', '.join(missing)}."
            if missing
            else f"All {len(expected)} requested symbols (including the benchmark) have bars."
        ),
        observed={"missing_symbols": missing, "per_symbol": coverage},
        threshold={"expected_symbols": expected},
    )


def check_calendar_alignment(
    frame: pd.DataFrame,
    modeling_symbols: Sequence[str],
    benchmark_symbol: str,
    policy: DataQualityPolicy,
) -> CheckResult:
    """Symbols must not skip benchmark sessions inside their own trading range.

    Gaps silently stretch return windows (``pct_change(5)`` would span more than five
    sessions). Measuring only inside each symbol's first/last bar means IPOs and
    delistings are not penalized.
    """

    benchmark_dates = pd.DatetimeIndex(
        sorted(frame.loc[frame["symbol"] == benchmark_symbol, "date"].unique())
    )
    per_symbol: dict[str, dict[str, object]] = {}
    worst_fraction = 0.0
    total_missing = 0
    off_calendar_total = 0
    examples_rows: list[dict[str, object]] = []
    if len(benchmark_dates):
        for symbol in sorted(set(modeling_symbols)):
            symbol_dates = pd.DatetimeIndex(frame.loc[frame["symbol"] == symbol, "date"].unique())
            if symbol_dates.empty:
                continue
            window = benchmark_dates[
                (benchmark_dates >= symbol_dates.min()) & (benchmark_dates <= symbol_dates.max())
            ]
            missing = window.difference(symbol_dates)
            off_calendar = symbol_dates.difference(benchmark_dates)
            fraction = float(len(missing) / len(window)) if len(window) else 0.0
            worst_fraction = max(worst_fraction, fraction)
            total_missing += len(missing)
            off_calendar_total += len(off_calendar)
            per_symbol[symbol] = {
                "expected_sessions": int(len(window)),
                "missing_sessions": int(len(missing)),
                "missing_fraction": round(fraction, 6),
                "off_benchmark_calendar": int(len(off_calendar)),
            }
            for timestamp in list(missing[:2]):
                examples_rows.append({"symbol": symbol, "missing_date": _json_scalar(timestamp)})

    status: CheckStatus
    if worst_fraction > policy.max_calendar_gap_fraction:
        status = "fail"
        message = (
            f"A symbol is missing {worst_fraction:.1%} of benchmark sessions in its active range "
            f"(limit {policy.max_calendar_gap_fraction:.1%})."
        )
    elif total_missing or off_calendar_total:
        status = "warn"
        message = (
            f"{total_missing} benchmark sessions missing and {off_calendar_total} off-calendar "
            "bars across symbols (within tolerance)."
        )
    else:
        status = "pass"
        message = "Every symbol trades on every benchmark session inside its active range."
    return CheckResult(
        check_id="bars.calendar_alignment",
        category="coverage",
        status=status,
        message=message,
        observed={"worst_missing_fraction": round(worst_fraction, 6), "per_symbol": per_symbol},
        threshold={"max_calendar_gap_fraction": policy.max_calendar_gap_fraction},
        examples=examples_rows,
    )


def check_trailing_coverage(
    frame: pd.DataFrame,
    modeling_symbols: Sequence[str],
    benchmark_symbol: str,
    policy: DataQualityPolicy,
) -> CheckResult:
    """Warn when a symbol's history ends well before the benchmark's (stale feed)."""

    benchmark_dates = pd.DatetimeIndex(
        sorted(frame.loc[frame["symbol"] == benchmark_symbol, "date"].unique())
    )
    lagging: dict[str, int] = {}
    if len(benchmark_dates):
        for symbol in sorted(set(modeling_symbols)):
            symbol_dates = frame.loc[frame["symbol"] == symbol, "date"]
            if symbol_dates.empty:
                continue
            lag = int((benchmark_dates > symbol_dates.max()).sum())
            if lag > policy.max_trailing_lag_sessions:
                lagging[symbol] = lag
    return CheckResult(
        check_id="bars.trailing_coverage",
        category="coverage",
        status="warn" if lagging else "pass",
        message=(
            f"{len(lagging)} symbols stop trading more than "
            f"{policy.max_trailing_lag_sessions} sessions before the benchmark."
            if lagging
            else "No symbol history ends materially before the benchmark history."
        ),
        observed={"lagging_sessions_by_symbol": lagging},
        threshold={"max_trailing_lag_sessions": policy.max_trailing_lag_sessions},
    )


def check_extreme_returns(frame: pd.DataFrame, policy: DataQualityPolicy) -> CheckResult:
    """Flag extreme adjusted daily moves for review without rejecting them."""

    ordered = frame.sort_values(["symbol", "date"]).copy()
    ordered["adjusted_return"] = ordered.groupby("symbol")["adjusted_close"].pct_change()
    offending = ordered[ordered["adjusted_return"].abs() > policy.extreme_return_threshold]
    return CheckResult(
        check_id="bars.extreme_returns",
        category="sanity",
        status="warn" if len(offending) else "pass",
        message=(
            f"{len(offending)} adjusted daily moves exceed "
            f"{policy.extreme_return_threshold:.0%} (allowed; review for bad ticks)."
            if len(offending)
            else f"No adjusted daily move exceeds {policy.extreme_return_threshold:.0%}."
        ),
        observed={"extreme_rows": int(len(offending))},
        threshold={"extreme_return_threshold": policy.extreme_return_threshold},
        examples=examples(offending, ["symbol", "date", "adjusted_close", "adjusted_return"]),
    )


def check_stale_prices(frame: pd.DataFrame, policy: DataQualityPolicy) -> CheckResult:
    """Warn on long runs of identical adjusted closes (a common stale-feed symptom)."""

    longest: dict[str, int] = {}
    for symbol, group in frame.sort_values(["symbol", "date"]).groupby("symbol"):
        closes = group["adjusted_close"].to_numpy()
        if len(closes) == 0:
            continue
        run_ids = np.concatenate([[0], np.cumsum(closes[1:] != closes[:-1])])
        run_length = int(np.bincount(run_ids).max())
        if run_length >= policy.stale_price_run_length:
            longest[str(symbol)] = run_length
    return CheckResult(
        check_id="bars.stale_prices",
        category="sanity",
        status="warn" if longest else "pass",
        message=(
            f"{len(longest)} symbols repeat the same adjusted close for "
            f"{policy.stale_price_run_length}+ consecutive sessions."
            if longest
            else "No stale runs of identical adjusted closes."
        ),
        observed={"longest_run_by_symbol": longest},
        threshold={"stale_price_run_length": policy.stale_price_run_length},
    )
