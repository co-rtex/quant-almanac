"""Versioned thresholds for data quality and training gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

DATA_QUALITY_POLICY_VERSION = "dq_v1"


@dataclass(frozen=True)
class DataQualityPolicy:
    """Explicit, persisted thresholds used by every quality check.

    The policy is serialized into each quality report so a reviewer can see which
    thresholds produced a verdict, and so a later policy change never silently
    re-interprets historical evidence.
    """

    version: str = DATA_QUALITY_POLICY_VERSION
    # Relative tolerance used when comparing high/low against open/close.
    ohlc_relative_tolerance: float = 1e-6
    # Fraction of benchmark sessions a symbol may miss (inside its own active range)
    # before the build is blocked. Any missing session produces a warning.
    max_calendar_gap_fraction: float = 0.05
    # The benchmark defines the trading calendar, so its own sessions are checked
    # against weekdays. Exchange holidays remove roughly 3.5-4% of weekdays a year, so
    # up to 4.5% missing passes; up to 6.5% (or a run of up to 4 consecutive missing
    # weekdays, e.g. the 2001 four-day closure) warns; more fails.
    benchmark_missing_weekday_pass: float = 0.045
    benchmark_missing_weekday_fail: float = 0.065
    benchmark_max_consecutive_missing_pass: int = 2
    benchmark_max_consecutive_missing_fail: int = 4
    # Sessions a symbol's last bar may lag the benchmark's last bar before warning.
    max_trailing_lag_sessions: int = 5
    # Absolute adjusted daily return considered extreme. Extreme moves are warnings:
    # earnings gaps, halts, and corporate events are legitimate in equity data.
    extreme_return_threshold: float = 0.40
    # Consecutive identical adjusted closes treated as a stale-feed warning.
    stale_price_run_length: int = 5
    # Rows per symbol excluded from missingness checks. 50 rows covers the longest
    # lookback in ``ohlcv_v1`` (the benchmark's 50-session moving average).
    feature_warmup_rows: int = 50
    # Post-warmup null fraction (per feature) that produces a warning.
    max_post_warmup_null_fraction: float = 0.05
    # Absolute Pearson correlation between a feature and a same-row forward return
    # that is treated as target leakage.
    leakage_correlation_threshold: float = 0.98
    # Date quantiles at which point-in-time feature recomputation is performed.
    point_in_time_cutoff_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    # Absolute tolerance for point-in-time feature equality.
    point_in_time_tolerance: float = 1e-9
    # Minimum rows per training split before a warning is raised.
    min_split_rows_warning: int = 30

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the policy."""

        payload = asdict(self)
        payload["point_in_time_cutoff_quantiles"] = list(self.point_in_time_cutoff_quantiles)
        return payload
