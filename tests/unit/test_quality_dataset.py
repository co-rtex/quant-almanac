"""Unit tests for dataset-level label, leakage, and point-in-time checks."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant_signal.features.engineering import FEATURE_COLUMNS, build_feature_frame
from quant_signal.features.labels import add_forward_return_targets
from quant_signal.quality.dataset import check_point_in_time_features, run_dataset_checks
from quant_signal.quality.policy import DataQualityPolicy

HORIZONS = [1, 5]
POLICY = DataQualityPolicy()


def make_bars(periods: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for symbol, scale in (("AAA", 0.012), ("BBB", 0.018), ("SPY", 0.009)):
        closes = 100 * np.exp(np.cumsum(rng.normal(0.0002, scale, periods)))
        for index, timestamp in enumerate(pd.bdate_range("2023-01-02", periods=periods)):
            rows.append(
                {
                    "symbol": symbol,
                    "date": timestamp.date(),
                    "open": closes[index],
                    "high": closes[index] * 1.01,
                    "low": closes[index] * 0.99,
                    "close": closes[index],
                    "adjusted_close": closes[index],
                    "volume": int(1_000_000 + rng.integers(0, 10_000)),
                    "source": "fixture",
                }
            )
    return pd.DataFrame(rows)


def make_dataset(bars: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_frame(bars, "SPY")
    dataset = features[features["symbol"].isin(["AAA", "BBB"])].copy()
    return add_forward_return_targets(dataset, HORIZONS)


def by_id(checks: list, check_id: str):  # type: ignore[no-untyped-def]
    return next(check for check in checks if check.check_id == check_id)


def test_real_feature_pipeline_passes_dataset_checks() -> None:
    bars = make_bars()
    checks = run_dataset_checks(
        make_dataset(bars), FEATURE_COLUMNS, HORIZONS, date(2023, 12, 31), POLICY
    )
    assert {check.status for check in checks} == {"pass"}, [c.to_json() for c in checks]


def test_point_in_time_check_passes_for_backward_looking_features() -> None:
    bars = make_bars()
    result = check_point_in_time_features(
        bars,
        build_feature_frame(bars, "SPY"),
        FEATURE_COLUMNS,
        ["AAA", "BBB"],
        "SPY",
        build_feature_frame,
        POLICY,
    )
    assert result.status == "pass"
    assert result.observed["compared_cells"] > 0


def test_point_in_time_check_detects_look_ahead_feature() -> None:
    def leaky_builder(frame: pd.DataFrame, benchmark: str) -> pd.DataFrame:
        built = build_feature_frame(frame, benchmark)
        # Deliberate bug: a "momentum" that peeks one session into the future.
        built["momentum_5d"] = built.groupby("symbol")["adjusted_close"].shift(-1)
        return built

    bars = make_bars()
    result = check_point_in_time_features(
        bars,
        leaky_builder(bars, "SPY"),
        FEATURE_COLUMNS,
        ["AAA", "BBB"],
        "SPY",
        leaky_builder,
        POLICY,
    )
    assert result.status == "fail"
    assert set(result.observed["mismatched_cells_by_feature"]) == {"momentum_5d"}
    assert result.examples[0]["feature"] == "momentum_5d"


def test_misaligned_labels_fail_label_alignment() -> None:
    dataset = make_dataset(make_bars())
    # Off-by-one horizon bug: 5d labels computed with a 4-session shift.
    shifted = add_forward_return_targets(dataset.drop(columns=["forward_return_5d"]), [4])
    dataset["forward_return_5d"] = shifted["forward_return_4d"]
    check = by_id(
        run_dataset_checks(dataset, FEATURE_COLUMNS, HORIZONS, date(2023, 12, 31), POLICY),
        "dataset.label_alignment",
    )
    assert check.status == "fail"
    assert "5d" in check.observed["forward_return_mismatches"]


def test_infinite_features_fail_and_duplicate_rows_fail() -> None:
    dataset = make_dataset(make_bars())
    dataset.loc[dataset.index[50], "volume_change_1d"] = np.inf
    dataset = pd.concat([dataset, dataset.iloc[[60]]])
    checks = run_dataset_checks(dataset, FEATURE_COLUMNS, HORIZONS, date(2023, 12, 31), POLICY)
    assert by_id(checks, "dataset.non_finite_features").status == "fail"
    assert by_id(checks, "dataset.unique_keys").status == "fail"


def test_label_columns_used_as_features_fail_schema() -> None:
    dataset = make_dataset(make_bars())
    checks = run_dataset_checks(
        dataset, [*FEATURE_COLUMNS, "forward_return_5d"], HORIZONS, date(2023, 12, 31), POLICY
    )
    assert checks[0].check_id == "dataset.schema"
    assert checks[0].status == "fail"


def test_near_perfect_feature_label_correlation_is_flagged_as_warning() -> None:
    dataset = make_dataset(make_bars())
    dataset["momentum_5d"] = dataset["forward_return_1d"] * 3.0
    check = by_id(
        run_dataset_checks(dataset, FEATURE_COLUMNS, HORIZONS, date(2023, 12, 31), POLICY),
        "dataset.feature_target_leakage",
    )
    assert check.status == "warn"
    assert check.observed["suspicious_pairs"][0]["feature"] == "momentum_5d"


def test_dataset_after_as_of_date_fails() -> None:
    dataset = make_dataset(make_bars())
    check = by_id(
        run_dataset_checks(dataset, FEATURE_COLUMNS, HORIZONS, date(2023, 3, 1), POLICY),
        "dataset.temporal_bounds",
    )
    assert check.status == "fail"


def test_label_span_warns_when_a_symbol_gap_stretches_the_horizon() -> None:
    bars = make_bars()
    calendar = sorted(pd.to_datetime(bars.loc[bars["symbol"] == "SPY", "date"]).unique())
    aaa = bars.index[bars["symbol"] == "AAA"]
    gapped = bars.drop(aaa[60:63])
    checks = run_dataset_checks(
        make_dataset(gapped), FEATURE_COLUMNS, HORIZONS, date(2023, 12, 31), POLICY, calendar
    )
    span = by_id(checks, "dataset.label_calendar_span")
    assert span.status == "warn"
    assert span.observed["labels_spanning_more_than_h_sessions"]["5d"] > 0
    clean = run_dataset_checks(
        make_dataset(bars), FEATURE_COLUMNS, HORIZONS, date(2023, 12, 31), POLICY, calendar
    )
    assert by_id(clean, "dataset.label_calendar_span").status == "pass"


def test_zero_volume_session_does_not_produce_infinite_features() -> None:
    bars = make_bars()
    bars.loc[bars.index[bars["symbol"] == "AAA"][70], "volume"] = 0
    checks = run_dataset_checks(
        make_dataset(bars), FEATURE_COLUMNS, HORIZONS, date(2023, 12, 31), POLICY
    )
    assert by_id(checks, "dataset.non_finite_features").status == "pass"
