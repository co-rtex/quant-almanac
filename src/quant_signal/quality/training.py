"""Training-time gates: split boundaries, label-horizon embargo, and class coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant_signal.quality.policy import DataQualityPolicy
from quant_signal.quality.report import CheckResult, CheckStatus

SPLIT_NAMES = ("train", "validation", "test")


def label_realization_dates(dataset: pd.DataFrame, horizon: int) -> pd.Series:
    """Return, per row, the session on which that row's h-day label becomes known.

    Labels are built per symbol with ``shift(-h)``, so the realization date is the
    symbol's own session ``h`` rows later. Using each symbol's calendar (instead of
    a global one) keeps the result exact when a symbol has gaps.
    """

    ordered = dataset[["symbol", "date"]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values(["symbol", "date"])
    realized = ordered.groupby("symbol", sort=False)["date"].shift(-horizon)
    return realized.reindex(dataset.index)


def run_split_checks(
    dataset: pd.DataFrame,
    split_dates: Mapping[str, Sequence[pd.Timestamp]],
    horizon: int,
    feature_columns: Sequence[str],
    target_column: str,
    policy: DataQualityPolicy,
) -> list[CheckResult]:
    """Check one horizon's temporal split before any model is fit.

    ``dataset`` is the full labeled dataset artifact (not only the training rows), so
    label realization dates can be resolved even when they fall in an embargo gap.
    """

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["label_realized_date"] = label_realization_dates(frame, horizon)
    labeled = frame[frame[target_column].notna()]
    partitions = {
        name: labeled[labeled["date"].isin(pd.DatetimeIndex(split_dates.get(name, ())))]
        for name in SPLIT_NAMES
    }
    return [
        check_feature_availability(frame, feature_columns, target_column),
        check_split_ordering(partitions),
        check_label_horizon_embargo(partitions, horizon),
        check_split_class_coverage(partitions, target_column, policy),
    ]


def check_feature_availability(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> CheckResult:
    """The registered feature and target columns must exist in the artifact."""

    missing = [column for column in [*feature_columns, target_column] if column not in frame]
    return CheckResult(
        check_id="training.feature_availability",
        category="schema",
        status="fail" if missing else "pass",
        message=(
            f"Dataset artifact is missing columns: {', '.join(missing)}."
            if missing
            else f"All {len(feature_columns)} features and the target are present."
        ),
        observed={"missing_columns": missing, "target_column": target_column},
    )


def _window(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "start_date": None, "end_date": None}
    return {
        "rows": int(len(frame)),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
    }


def check_split_ordering(partitions: dict[str, pd.DataFrame]) -> CheckResult:
    """Train, validation, and test windows must be non-empty, ordered, and disjoint."""

    windows = {name: _window(frame) for name, frame in partitions.items()}
    empty = [name for name, frame in partitions.items() if frame.empty]
    problems: list[str] = []
    if not empty:
        if partitions["train"]["date"].max() >= partitions["validation"]["date"].min():
            problems.append("train overlaps or follows validation")
        if partitions["validation"]["date"].max() >= partitions["test"]["date"].min():
            problems.append("validation overlaps or follows test")
    status: CheckStatus = "fail" if (empty or problems) else "pass"
    if empty:
        message = f"Empty split(s): {', '.join(empty)}."
    elif problems:
        message = "; ".join(problems).capitalize() + "."
    else:
        message = "Train < validation < test with no shared dates."
    return CheckResult(
        check_id="training.split_ordering",
        category="splits",
        status=status,
        message=message,
        observed={"windows": windows, "problems": problems},
    )


def check_label_horizon_embargo(partitions: dict[str, pd.DataFrame], horizon: int) -> CheckResult:
    """No training (or validation) label may be realized on or after the next split starts.

    This is the exact leakage condition: a train row at date t uses the close at its
    t+h session, so every such realization must precede the first validation date
    (and likewise validation realizations must precede the first test date).
    """

    observed: dict[str, object] = {"horizon_days": horizon}
    problems: list[str] = []
    for current, following in (("train", "validation"), ("validation", "test")):
        current_frame = partitions[current]
        following_frame = partitions[following]
        if current_frame.empty or following_frame.empty:
            continue
        unknown = int(current_frame["label_realized_date"].isna().sum())
        last_realized = current_frame["label_realized_date"].max()
        boundary = following_frame["date"].min()
        observed[f"{current}_last_label_realized"] = (
            None if pd.isna(last_realized) else pd.Timestamp(last_realized).date().isoformat()
        )
        observed[f"{following}_start"] = pd.Timestamp(boundary).date().isoformat()
        if unknown:
            problems.append(f"{unknown} {current} labels have no realization session")
        elif pd.Timestamp(last_realized) >= pd.Timestamp(boundary):
            problems.append(
                f"{current} labels realize on {pd.Timestamp(last_realized).date()} "
                f"but {following} starts {pd.Timestamp(boundary).date()}"
            )
    return CheckResult(
        check_id="training.label_horizon_embargo",
        category="leakage",
        status="fail" if problems else "pass",
        message=(
            "; ".join(problems) + "."
            if problems
            else f"Every {horizon}d train/validation label is realized before the next "
            "split begins."
        ),
        observed={**observed, "problems": problems},
    )


def check_split_class_coverage(
    partitions: dict[str, pd.DataFrame],
    target_column: str,
    policy: DataQualityPolicy,
) -> CheckResult:
    """Each split needs both target classes; small splits produce a warning."""

    per_split: dict[str, dict[str, object]] = {}
    single_class: list[str] = []
    small: list[str] = []
    for name, frame in partitions.items():
        target = frame[target_column].dropna()
        classes = sorted({int(value) for value in target.unique()})
        per_split[name] = {
            "rows": int(len(target)),
            "positive_rate": round(float(target.mean()), 6) if len(target) else None,
            "classes": classes,
        }
        if len(classes) < 2:
            single_class.append(name)
        if len(target) < policy.min_split_rows_warning:
            small.append(name)
    status: CheckStatus = "fail" if single_class else "warn" if small else "pass"
    if single_class:
        message = f"Split(s) with a single target class: {', '.join(single_class)}."
    elif small:
        message = (
            f"Split(s) below {policy.min_split_rows_warning} rows: {', '.join(small)}; "
            "metrics will be noisy."
        )
    else:
        message = "Every split contains both target classes."
    return CheckResult(
        check_id="training.split_class_coverage",
        category="labels",
        status=status,
        message=message,
        observed={"per_split": per_split},
        threshold={"min_split_rows_warning": policy.min_split_rows_warning},
    )
