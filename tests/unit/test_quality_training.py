"""Unit tests for training split and label-horizon embargo gates."""

from __future__ import annotations

import pandas as pd

from quant_signal.quality.policy import DataQualityPolicy
from quant_signal.quality.training import label_realization_dates, run_split_checks


def make_frame(periods: int = 100, symbols: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        for index, timestamp in enumerate(pd.bdate_range("2024-01-01", periods=periods)):
            rows.append(
                {
                    "symbol": symbol,
                    "date": timestamp,
                    "feature": float(index),
                    "target_up_5d": float(index % 2) if index < periods - 5 else None,
                }
            )
    return pd.DataFrame(rows)


def split(frame: pd.DataFrame, embargo: int) -> dict[str, list[pd.Timestamp]]:
    dates = sorted(frame.loc[frame["target_up_5d"].notna(), "date"].unique())
    train_end = 50
    validation_start = train_end + embargo
    validation_end = validation_start + 15
    test_start = validation_end + embargo
    return {
        "train": dates[:train_end],
        "validation": dates[validation_start:validation_end],
        "test": dates[test_start : test_start + 15],
    }


def checks_for(frame: pd.DataFrame, embargo: int) -> dict[str, str]:
    results = run_split_checks(
        frame, split(frame, embargo), 5, ["feature"], "target_up_5d", DataQualityPolicy()
    )
    return {result.check_id: result.status for result in results}


def test_label_realization_uses_each_symbols_own_calendar() -> None:
    frame = make_frame(periods=12)
    gapped = frame.drop(frame.index[(frame["symbol"] == "BBB")][3:5])
    realized = label_realization_dates(gapped, 5)
    first_aaa = realized[(gapped["symbol"] == "AAA")].iloc[0]
    first_bbb = realized[(gapped["symbol"] == "BBB")].iloc[0]
    assert first_aaa == pd.Timestamp("2024-01-08")
    assert first_bbb == pd.Timestamp("2024-01-10")


def test_embargo_equal_to_horizon_passes() -> None:
    statuses = checks_for(make_frame(), embargo=5)
    assert statuses["training.label_horizon_embargo"] == "pass"
    assert statuses["training.split_ordering"] == "pass"


def test_embargo_shorter_than_horizon_fails() -> None:
    statuses = checks_for(make_frame(), embargo=4)
    assert statuses["training.label_horizon_embargo"] == "fail"


def test_single_class_split_fails_and_missing_feature_fails() -> None:
    frame = make_frame()
    frame["target_up_5d"] = frame["target_up_5d"].where(frame["target_up_5d"].isna(), 1.0)
    statuses = checks_for(frame, embargo=5)
    assert statuses["training.split_class_coverage"] == "fail"
    results = run_split_checks(
        make_frame(), split(make_frame(), 5), 5, ["missing"], "target_up_5d", DataQualityPolicy()
    )
    assert results[0].status == "fail"


def test_overlapping_windows_fail_ordering() -> None:
    frame = make_frame()
    windows = split(frame, 5)
    windows["validation"] = windows["train"][-5:]
    results = run_split_checks(frame, windows, 5, ["feature"], "target_up_5d", DataQualityPolicy())
    statuses = {result.check_id: result.status for result in results}
    assert statuses["training.split_ordering"] == "fail"
