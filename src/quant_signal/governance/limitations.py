"""Evidence-derived limitations for a model version.

Every limitation is computed from persisted records or from the documented behavior
of the code that produced them. Nothing here is a hard-coded claim about results.
"""

from __future__ import annotations

from typing import Any

from quant_signal.quality.report import QualityReport

SMALL_TEST_SAMPLE = 250


def _limitation(severity: str, source: str, text: str) -> dict[str, str]:
    return {"severity": severity, "source": source, "text": text}


def derive_limitations(
    lineage: dict[str, Any], verification: QualityReport
) -> list[dict[str, str]]:
    """Return ordered limitations (``high`` → ``medium`` → ``info``)."""

    limitations: list[dict[str, str]] = []
    dataset = lineage["dataset"]
    model = lineage["model"]

    if lineage.get("synthetic"):
        limitations.append(
            _limitation(
                "high",
                "data source",
                "Trained and evaluated on SYNTHETIC random-walk data. Metrics and backtests "
                "demonstrate the pipeline only and say nothing about real securities.",
            )
        )

    test = next((row for row in lineage["evaluations"] if row["split_name"] == "test"), None)
    if test is None:
        limitations.append(_limitation("high", "evaluation", "No test-split evaluation exists."))
    else:
        interval = test.get("roc_auc_interval")
        if interval and interval["upper_95"] < 0.5:
            limitations.append(
                _limitation(
                    "high",
                    "evaluation",
                    f"Test ROC-AUC {interval['auc']:.3f} is below random ranking (approx. 95% CI "
                    f"{interval['lower_95']:.3f}–{interval['upper_95']:.3f}): the "
                    "validation-selected model does not generalize to the test period.",
                )
            )
        elif interval and interval.get("includes_chance"):
            limitations.append(
                _limitation(
                    "high",
                    "evaluation",
                    f"Test ROC-AUC {interval['auc']:.3f} is not distinguishable from random "
                    f"ranking (approx. 95% CI {interval['lower_95']:.3f}–"
                    f"{interval['upper_95']:.3f}).",
                )
            )
        sample_count = test.get("sample_count")
        if isinstance(sample_count, int) and sample_count < SMALL_TEST_SAMPLE:
            limitations.append(
                _limitation(
                    "medium",
                    "evaluation",
                    f"The test split has only {sample_count} observations; metrics are noisy.",
                )
            )
        if model["horizon_days"] > 1 and interval:
            limitations.append(
                _limitation(
                    "medium",
                    "evaluation",
                    f"{model['horizon_days']}-day labels overlap across consecutive sessions, so "
                    "observations are serially correlated and the AUC interval (which assumes "
                    "independence) understates uncertainty.",
                )
            )

    config = model.get("training_config") or {}
    calibration = config.get("calibration") if isinstance(config, dict) else None
    if isinstance(calibration, dict) and calibration.get("method") == "identity":
        limitations.append(
            _limitation(
                "medium",
                "calibration",
                "Probability calibration fell back to identity (validation split too small "
                "or single-class), so scores are uncalibrated.",
            )
        )

    symbols = dataset.get("symbols") or []
    if len(symbols) < 3:
        limitations.append(
            _limitation(
                "medium",
                "universe",
                f"The modeling universe has {len(symbols)} symbol(s); cross-sectional "
                "ranking and top-N backtests are close to degenerate.",
            )
        )

    for report_key, label in (
        ("dataset_quality_report", "dataset quality"),
        ("training_quality_report", "training gate"),
    ):
        report = lineage.get(report_key)
        if report:
            warnings = [
                check["check_id"] for check in report["checks"] if check["status"] == "warn"
            ]
            if warnings:
                limitations.append(
                    _limitation(
                        "medium",
                        label,
                        f"Passed with warnings: {', '.join(warnings)}.",
                    )
                )

    unavailable = [
        check.check_id
        for check in verification.checks
        if check.status == "warn" and check.message.startswith("Unavailable")
    ]
    if unavailable:
        limitations.append(
            _limitation(
                "medium",
                "lineage",
                "Incomplete provenance (record predates governance metadata or was not "
                f"captured): {', '.join(unavailable)}.",
            )
        )
    for check in verification.failures:
        limitations.append(
            _limitation("high", "verification", f"{check.check_id}: {check.message}")
        )

    backtests = lineage.get("backtests") or []
    if backtests:
        latest = backtests[0]
        assumptions = latest.get("execution_assumptions") or {}
        if not assumptions.get("total_cost_rate_per_side"):
            limitations.append(
                _limitation(
                    "medium",
                    "backtest",
                    "The latest backtest assumes zero transaction costs and slippage.",
                )
            )

    limitations.extend(
        [
            _limitation(
                "info",
                "methodology",
                "Champion selection uses one embargoed temporal holdout (validation); there "
                "is no cross-validation or hyperparameter search.",
            ),
            _limitation(
                "info",
                "methodology",
                "The universe is fixed when the dataset is built, so delisted securities are "
                "absent (survivorship bias) unless they were requested explicitly.",
            ),
            _limitation(
                "info",
                "methodology",
                "Backtests are long-only, equal-weight, monthly-refit simulations on daily "
                "closes; they are research evidence, not investment performance.",
            ),
        ]
    )
    order = {"high": 0, "medium": 1, "info": 2}
    return sorted(limitations, key=lambda row: order.get(row["severity"], 3))
