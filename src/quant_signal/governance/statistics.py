"""Small, dependency-free statistics used to qualify persisted evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AucInterval:
    """Approximate confidence interval for ROC-AUC."""

    auc: float
    standard_error: float
    lower: float
    upper: float
    positives: int
    negatives: int

    @property
    def includes_chance(self) -> bool:
        """Return True when 0.5 (a random ranking) lies inside the interval."""

        return self.lower <= 0.5 <= self.upper

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "auc": self.auc,
            "standard_error": self.standard_error,
            "lower_95": self.lower,
            "upper_95": self.upper,
            "positives": self.positives,
            "negatives": self.negatives,
            "includes_chance": self.includes_chance,
            "method": "Hanley-McNeil (1982), assumes independent observations",
        }


def hanley_mcneil_interval(
    auc: float | None,
    sample_count: int | None,
    positive_rate: float | None,
    z: float = 1.959964,
) -> AucInterval | None:
    """Return a Hanley-McNeil 95% interval from the summary statistics already persisted.

    Only AUC, sample count, and positive rate are stored for each split, which is
    exactly what the Hanley-McNeil approximation needs. Overlapping multi-day labels
    and cross-sectional correlation violate its independence assumption, so the true
    interval is wider; callers must present it as a lower bound on uncertainty.
    """

    if auc is None or sample_count is None or positive_rate is None:
        return None
    positives = int(round(sample_count * positive_rate))
    negatives = int(sample_count - positives)
    if positives < 1 or negatives < 1:
        return None
    q1 = auc / (2.0 - auc)
    q2 = (2.0 * auc * auc) / (1.0 + auc)
    variance = (
        auc * (1.0 - auc) + (positives - 1) * (q1 - auc * auc) + (negatives - 1) * (q2 - auc * auc)
    ) / (positives * negatives)
    standard_error = math.sqrt(max(variance, 0.0))
    return AucInterval(
        auc=auc,
        standard_error=standard_error,
        lower=max(0.0, auc - z * standard_error),
        upper=min(1.0, auc + z * standard_error),
        positives=positives,
        negatives=negatives,
    )
