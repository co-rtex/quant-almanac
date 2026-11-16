"""Helpers that turn in-memory quality reports into persisted evidence rows."""

from __future__ import annotations

from quant_signal.core.provenance import provenance_snapshot
from quant_signal.quality.policy import DataQualityPolicy
from quant_signal.quality.report import QualityReport
from quant_signal.storage.models import QualityReportRecord
from quant_signal.storage.repositories import QualityReportPayload, StorageRepository


def summarize_report(report: QualityReport) -> str:
    """Return a one-line, human-readable verdict for a report."""

    verdict = (
        f"{report.status.upper()}: {len(report.checks)} checks, "
        f"{len(report.failures)} failed, {len(report.warnings)} warnings"
    )
    if report.failures:
        verdict += " (failed: " + ", ".join(check.check_id for check in report.failures) + ")"
    return verdict


def persist_quality_report(
    repository: StorageRepository,
    report: QualityReport,
    policy: DataQualityPolicy,
    request_json: dict[str, object],
    context_json: dict[str, object] | None = None,
    dataset_version_id: str | None = None,
    model_version_id: str | None = None,
    horizon_days: int | None = None,
) -> QualityReportRecord:
    """Persist a report together with the policy and provenance that produced it."""

    return repository.create_quality_report(
        QualityReportPayload(
            stage=report.stage,
            status=report.status,
            policy_version=policy.version,
            check_count=len(report.checks),
            failure_count=len(report.failures),
            warning_count=len(report.warnings),
            summary=summarize_report(report),
            checks_json=report.checks_json(),
            request_json=request_json,
            context_json={
                **(context_json or {}),
                "policy": policy.to_json(),
                "notes": list(report.notes),
                "provenance": provenance_snapshot(),
            },
            dataset_version_id=dataset_version_id,
            model_version_id=model_version_id,
            horizon_days=horizon_days,
        )
    )
