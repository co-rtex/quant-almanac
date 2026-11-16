"""Model-ready dataset checks: labels, leakage, missingness, and point-in-time safety."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import pandas as pd

from quant_signal.quality.bars import examples
from quant_signal.quality.policy import DataQualityPolicy
from quant_signal.quality.report import CheckResult, CheckStatus

FeatureBuilder = Callable[[pd.DataFrame, str], pd.DataFrame]


def run_dataset_checks(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    horizons: Sequence[int],
    as_of_date: date,
    policy: DataQualityPolicy,
) -> list[CheckResult]:
    """Run label, leakage, and missingness checks on a materialized dataset frame."""

    schema = check_dataset_schema(dataset, feature_columns, horizons)
    if schema.status == "fail":
        return [schema]

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    return [
        schema,
        check_dataset_keys(frame),
        check_dataset_temporal_bounds(frame, as_of_date),
        check_non_finite_features(frame, feature_columns),
        check_feature_missingness(frame, feature_columns, policy),
        check_label_alignment(frame, horizons),
        check_label_availability(frame, horizons),
        check_feature_target_leakage(frame, feature_columns, horizons, policy),
    ]


def _label_columns(horizons: Sequence[int]) -> list[str]:
    return [
        column
        for horizon in horizons
        for column in (f"target_up_{horizon}d", f"forward_return_{horizon}d")
    ]


def check_dataset_schema(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    horizons: Sequence[int],
) -> CheckResult:
    """Features, labels, and keys must exist, and features must not include labels."""

    required = ["date", "symbol", "adjusted_close", *feature_columns, *_label_columns(horizons)]
    missing = [column for column in required if column not in dataset.columns]
    overlap = sorted(set(feature_columns) & set(_label_columns(horizons)))
    forward_named = sorted(
        column for column in feature_columns if "forward" in column or "target" in column
    )
    status: CheckStatus = "fail" if (missing or overlap or forward_named) else "pass"
    if missing:
        message = f"Dataset is missing required columns: {', '.join(missing)}."
    elif overlap or forward_named:
        message = "Feature columns include label or forward-looking columns."
    else:
        message = "Feature, label, and key columns are present and disjoint."
    return CheckResult(
        check_id="dataset.schema",
        category="schema",
        status=status,
        message=message,
        observed={
            "missing_columns": missing,
            "feature_label_overlap": overlap,
            "forward_named_features": forward_named,
            "feature_count": len(feature_columns),
        },
    )


def check_dataset_keys(frame: pd.DataFrame) -> CheckResult:
    """Each (date, symbol) observation must be unique."""

    duplicated = frame[frame.duplicated(subset=["date", "symbol"], keep=False)]
    return CheckResult(
        check_id="dataset.unique_keys",
        category="integrity",
        status="fail" if len(duplicated) else "pass",
        message=(
            f"{len(duplicated)} dataset rows share a (date, symbol) key."
            if len(duplicated)
            else "Every (date, symbol) observation is unique."
        ),
        observed={"duplicate_rows": int(len(duplicated)), "row_count": int(len(frame))},
        examples=examples(duplicated, ["date", "symbol"]),
    )


def check_dataset_temporal_bounds(frame: pd.DataFrame, as_of_date: date) -> CheckResult:
    """The dataset must be non-empty and end on or before the as-of date."""

    if frame.empty:
        return CheckResult(
            check_id="dataset.temporal_bounds",
            category="timestamps",
            status="fail",
            message="The dataset contains no observations.",
            observed={"row_count": 0},
        )
    last_date = frame["date"].max()
    after = frame[frame["date"] > pd.Timestamp(as_of_date)]
    return CheckResult(
        check_id="dataset.temporal_bounds",
        category="timestamps",
        status="fail" if len(after) else "pass",
        message=(
            f"{len(after)} observations are dated after the as-of date."
            if len(after)
            else "All observations fall on or before the as-of date."
        ),
        observed={
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": last_date.date().isoformat(),
            "unique_dates": int(frame["date"].nunique()),
            "symbols": sorted(str(symbol) for symbol in frame["symbol"].unique()),
        },
        threshold={"as_of_date": as_of_date.isoformat()},
    )


def check_non_finite_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> CheckResult:
    """Infinite feature values break imputation and model fitting."""

    values = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    infinite = np.isinf(values.to_numpy(dtype=float))
    counts = {
        column: int(count)
        for column, count in zip(feature_columns, infinite.sum(axis=0), strict=True)
        if count
    }
    offending = frame[infinite.any(axis=1)]
    return CheckResult(
        check_id="dataset.non_finite_features",
        category="sanity",
        status="fail" if counts else "pass",
        message=(
            f"{int(sum(counts.values()))} infinite feature values across {len(counts)} features."
            if counts
            else "No feature contains infinite values."
        ),
        observed={"infinite_values_by_feature": counts},
        examples=examples(offending, ["date", "symbol"]),
    )


def check_feature_missingness(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    policy: DataQualityPolicy,
) -> CheckResult:
    """After each symbol's warmup window, features should be populated."""

    position = frame.groupby("symbol").cumcount()
    post_warmup = frame[position >= policy.feature_warmup_rows]
    if post_warmup.empty:
        return CheckResult(
            check_id="dataset.feature_missingness",
            category="completeness",
            status="warn",
            message="No observations remain after the feature warmup window.",
            observed={"post_warmup_rows": 0},
            threshold={"feature_warmup_rows": policy.feature_warmup_rows},
        )
    null_fraction = post_warmup[list(feature_columns)].isna().mean()
    high = {
        str(column): round(float(value), 6)
        for column, value in null_fraction.items()
        if value > policy.max_post_warmup_null_fraction
    }
    warmup_rows = frame[position < policy.feature_warmup_rows]
    warmup_nulls = int(warmup_rows[list(feature_columns)].isna().sum().sum())
    return CheckResult(
        check_id="dataset.feature_missingness",
        category="completeness",
        status="warn" if high else "pass",
        message=(
            f"{len(high)} features exceed the post-warmup null limit."
            if high
            else "Features are fully populated after the warmup window."
        ),
        observed={
            "post_warmup_rows": int(len(post_warmup)),
            "max_post_warmup_null_fraction": round(float(null_fraction.max()), 6),
            "features_over_limit": high,
            "warmup_null_cells": warmup_nulls,
        },
        threshold={
            "feature_warmup_rows": policy.feature_warmup_rows,
            "max_post_warmup_null_fraction": policy.max_post_warmup_null_fraction,
        },
    )


def check_label_alignment(frame: pd.DataFrame, horizons: Sequence[int]) -> CheckResult:
    """Labels must be exactly the h-session-forward return of the same symbol.

    Recomputing forward returns from the persisted adjusted closes proves that labels
    are aligned to t+h (not t+h-1 or t+h+1), that no label exists for sessions whose
    outcome is after the dataset end, and that binary targets agree with returns.
    """

    mismatches: dict[str, int] = {}
    unlabeled_tail_violations: dict[str, int] = {}
    target_inconsistency: dict[str, int] = {}
    grouped = frame.groupby("symbol", sort=False)["adjusted_close"]
    example_rows: list[dict[str, object]] = []
    for horizon in horizons:
        stored = frame[f"forward_return_{horizon}d"].astype(float)
        expected = (grouped.shift(-horizon) / frame["adjusted_close"]) - 1.0
        null_mismatch = stored.isna() != expected.isna()
        both = stored.notna() & expected.notna()
        value_mismatch = both & ~np.isclose(stored, expected, rtol=1e-9, atol=1e-12)
        bad = null_mismatch | value_mismatch
        if bad.any():
            mismatches[f"{horizon}d"] = int(bad.sum())
            example_rows.extend(examples(frame[bad], ["date", "symbol"], limit=2))

        tail_position = frame.groupby("symbol").cumcount(ascending=False)
        tail_labeled = (tail_position < horizon) & stored.notna()
        if tail_labeled.any():
            unlabeled_tail_violations[f"{horizon}d"] = int(tail_labeled.sum())

        target = frame[f"target_up_{horizon}d"]
        invalid_values = target.notna() & ~target.isin([0.0, 1.0])
        null_disagree = target.isna() != stored.isna()
        direction = target.notna() & stored.notna() & (target != (stored > 0).astype(float))
        inconsistent = invalid_values | null_disagree | direction
        if inconsistent.any():
            target_inconsistency[f"{horizon}d"] = int(inconsistent.sum())

    failed = bool(mismatches or unlabeled_tail_violations or target_inconsistency)
    return CheckResult(
        check_id="dataset.label_alignment",
        category="leakage",
        status="fail" if failed else "pass",
        message=(
            "Labels are not aligned to the declared forward horizons."
            if failed
            else "Every label equals the recomputed h-session forward return; "
            "no label extends beyond the available history."
        ),
        observed={
            "forward_return_mismatches": mismatches,
            "labeled_rows_in_final_h_sessions": unlabeled_tail_violations,
            "target_inconsistencies": target_inconsistency,
            "horizons": list(horizons),
        },
        examples=example_rows,
    )


def check_label_availability(frame: pd.DataFrame, horizons: Sequence[int]) -> CheckResult:
    """Each horizon needs labeled rows with both outcome classes."""

    per_horizon: dict[str, dict[str, object]] = {}
    status: CheckStatus = "pass"
    for horizon in horizons:
        target = frame[f"target_up_{horizon}d"].dropna()
        classes = sorted({int(value) for value in target.unique()})
        per_horizon[f"{horizon}d"] = {
            "labeled_rows": int(len(target)),
            "positive_rate": round(float(target.mean()), 6) if len(target) else None,
            "classes": classes,
        }
        if target.empty:
            status = "fail"
        elif len(classes) < 2 and status == "pass":
            status = "warn"
    return CheckResult(
        check_id="dataset.label_availability",
        category="labels",
        status=status,
        message=(
            "At least one horizon has no labeled observations."
            if status == "fail"
            else "At least one horizon has a single outcome class."
            if status == "warn"
            else "Every horizon has labeled observations with both outcome classes."
        ),
        observed={"per_horizon": per_horizon},
    )


def check_feature_target_leakage(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    horizons: Sequence[int],
    policy: DataQualityPolicy,
) -> CheckResult:
    """Warn when a feature is almost perfectly correlated with a same-row forward return."""

    suspicious: list[dict[str, object]] = []
    max_abs = 0.0
    for horizon in horizons:
        forward = frame[f"forward_return_{horizon}d"].astype(float)
        for column in feature_columns:
            feature = pd.to_numeric(frame[column], errors="coerce").astype(float)
            mask = feature.notna() & forward.notna() & np.isfinite(feature)
            if int(mask.sum()) < 10:
                continue
            x = feature[mask]
            y = forward[mask]
            if float(x.std()) == 0.0 or float(y.std()) == 0.0:
                continue
            correlation = float(np.corrcoef(x, y)[0, 1])
            max_abs = max(max_abs, abs(correlation))
            if abs(correlation) >= policy.leakage_correlation_threshold:
                suspicious.append(
                    {
                        "feature": column,
                        "horizon": f"{horizon}d",
                        "correlation": round(correlation, 6),
                    }
                )
    # A near-perfect correlation is strong evidence of leakage on market data, but it is
    # a heuristic: a deterministic (e.g. synthetic periodic) series can be genuinely
    # predictable from its past. The blocking look-ahead test is the point-in-time
    # recomputation check; this one warns so a reviewer inspects the named pairs.
    return CheckResult(
        check_id="dataset.feature_target_leakage",
        category="leakage",
        status="warn" if suspicious else "pass",
        message=(
            f"{len(suspicious)} feature/horizon pairs are near-perfectly correlated with "
            "same-row forward returns (heuristic; review for leakage)."
            if suspicious
            else "No feature is near-perfectly correlated with a same-row forward return."
        ),
        observed={"max_abs_correlation": round(max_abs, 6), "suspicious_pairs": suspicious},
        threshold={"leakage_correlation_threshold": policy.leakage_correlation_threshold},
        examples=suspicious,
    )


def check_point_in_time_features(
    bars: pd.DataFrame,
    full_features: pd.DataFrame,
    feature_columns: Sequence[str],
    modeling_symbols: Sequence[str],
    benchmark_symbol: str,
    feature_builder: FeatureBuilder,
    policy: DataQualityPolicy,
) -> CheckResult:
    """Recompute features on truncated history and require identical values.

    A feature that uses information after date ``t`` changes when bars after ``t`` are
    removed. Recomputing at several cutoffs and comparing every feature cell dated on
    or before the cutoff directly tests for look-ahead bias in the feature code.
    """

    full = full_features.copy()
    full["date"] = pd.to_datetime(full["date"])
    full = full[full["symbol"].isin(modeling_symbols)]
    unique_dates = pd.DatetimeIndex(sorted(full["date"].unique()))
    if len(unique_dates) < 2:
        return CheckResult(
            check_id="dataset.point_in_time_features",
            category="leakage",
            status="warn",
            message="Too few dates to run point-in-time recomputation.",
            observed={"unique_dates": int(len(unique_dates))},
        )

    bar_dates = pd.to_datetime(bars["date"])
    cutoffs: list[pd.Timestamp] = []
    for quantile in policy.point_in_time_cutoff_quantiles:
        index = min(len(unique_dates) - 2, max(0, int(round(quantile * (len(unique_dates) - 1)))))
        cutoff = pd.Timestamp(unique_dates[index])
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)

    mismatched_by_feature: dict[str, int] = {}
    compared_cells = 0
    example_rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        truncated = feature_builder(bars[bar_dates <= cutoff].copy(), benchmark_symbol)
        truncated["date"] = pd.to_datetime(truncated["date"])
        truncated = truncated[truncated["symbol"].isin(modeling_symbols)]
        left = full[full["date"] <= cutoff].set_index(["symbol", "date"]).sort_index()
        right = truncated.set_index(["symbol", "date"]).sort_index()
        right = right.reindex(left.index)
        for column in feature_columns:
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            equal = np.isclose(a, b, rtol=0.0, atol=policy.point_in_time_tolerance, equal_nan=True)
            compared_cells += int(len(a))
            mismatched = int((~equal).sum())
            if mismatched:
                mismatched_by_feature[column] = mismatched_by_feature.get(column, 0) + mismatched
                first = int(np.argmax(~equal))
                symbol, timestamp = left.index[first]
                example_rows.append(
                    {
                        "feature": column,
                        "cutoff": cutoff.date().isoformat(),
                        "symbol": str(symbol),
                        "date": pd.Timestamp(timestamp).date().isoformat(),
                        "full_history_value": _finite_or_str(a[first]),
                        "truncated_history_value": _finite_or_str(b[first]),
                    }
                )

    return CheckResult(
        check_id="dataset.point_in_time_features",
        category="leakage",
        status="fail" if mismatched_by_feature else "pass",
        message=(
            f"{len(mismatched_by_feature)} features change when future bars are removed "
            "(look-ahead bias)."
            if mismatched_by_feature
            else "Every feature value is reproduced exactly from history truncated at "
            f"{len(cutoffs)} cutoffs (no look-ahead)."
        ),
        observed={
            "cutoffs": [cutoff.date().isoformat() for cutoff in cutoffs],
            "compared_cells": compared_cells,
            "mismatched_cells_by_feature": mismatched_by_feature,
        },
        threshold={"absolute_tolerance": policy.point_in_time_tolerance},
        examples=example_rows,
    )


def _finite_or_str(value: float) -> object:
    return float(value) if np.isfinite(value) else str(value)
