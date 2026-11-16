"""Typed check results and reports for data quality and governance gates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

CheckStatus = Literal["pass", "warn", "fail"]

STATUS_ORDER: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}
MAX_EXAMPLES = 5


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one deterministic check, with the evidence that produced it."""

    check_id: str
    category: str
    status: CheckStatus
    message: str
    observed: dict[str, object] = field(default_factory=dict)
    threshold: dict[str, object] = field(default_factory=dict)
    examples: list[dict[str, object]] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "observed": self.observed,
            "threshold": self.threshold,
            "examples": self.examples[:MAX_EXAMPLES],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> CheckResult:
        """Rebuild a check result from persisted JSON."""

        status = str(payload.get("status", "fail"))
        if status not in STATUS_ORDER:
            status = "fail"
        return cls(
            check_id=str(payload.get("check_id", "unknown")),
            category=str(payload.get("category", "unknown")),
            status=cast(CheckStatus, status),
            message=str(payload.get("message", "")),
            observed=_as_dict(payload.get("observed")),
            threshold=_as_dict(payload.get("threshold")),
            examples=_as_list_of_dicts(payload.get("examples")),
        )


def worst_status(statuses: Iterable[str]) -> CheckStatus:
    """Return the most severe status in a collection (``pass`` when empty)."""

    worst: CheckStatus = "pass"
    for status in statuses:
        if STATUS_ORDER.get(status, 2) > STATUS_ORDER[worst]:
            worst = cast(CheckStatus, status if status in STATUS_ORDER else "fail")
    return worst


@dataclass(frozen=True)
class QualityReport:
    """Ordered collection of check results for one pipeline stage."""

    stage: str
    checks: tuple[CheckResult, ...]
    notes: tuple[str, ...] = ()

    @property
    def status(self) -> CheckStatus:
        """Return the aggregate status (the worst individual status)."""

        return worst_status(check.status for check in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        """Return failed checks."""

        return [check for check in self.checks if check.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        """Return checks that produced warnings."""

        return [check for check in self.checks if check.status == "warn"]

    @property
    def passed(self) -> bool:
        """Return True when no check failed (warnings do not block)."""

        return not self.failures

    def extend(self, checks: Sequence[CheckResult], notes: Sequence[str] = ()) -> QualityReport:
        """Return a new report with additional checks and notes appended."""

        return QualityReport(
            stage=self.stage,
            checks=(*self.checks, *checks),
            notes=(*self.notes, *notes),
        )

    def checks_json(self) -> list[dict[str, object]]:
        """Return the persisted representation of the check list."""

        return [check.to_json() for check in self.checks]

    def summary_json(self) -> dict[str, object]:
        """Return compact counts suitable for CLI and API summaries."""

        return {
            "stage": self.stage,
            "status": self.status,
            "check_count": len(self.checks),
            "failure_count": len(self.failures),
            "warning_count": len(self.warnings),
            "failed_checks": [check.check_id for check in self.failures],
            "warning_checks": [check.check_id for check in self.warnings],
        }


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _as_list_of_dicts(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [_as_dict(item) for item in value if isinstance(item, Mapping)]
    return []
