"""Data quality checks, reports, and gates."""

from quant_signal.quality.artifacts import (
    ArtifactIntegrityError,
    check_artifact_integrity,
    require_artifact_integrity,
)
from quant_signal.quality.policy import DATA_QUALITY_POLICY_VERSION, DataQualityPolicy
from quant_signal.quality.report import CheckResult, CheckStatus, QualityReport, worst_status

STAGE_DATASET_BUILD = "dataset_build"
STAGE_TRAINING = "training"
STAGE_REPRODUCTION = "reproduction"


class DataQualityGateError(RuntimeError):
    """Raised when a blocking quality check fails; the report has already been persisted."""

    def __init__(self, report: QualityReport, quality_report_id: str) -> None:
        failed = ", ".join(check.check_id for check in report.failures)
        super().__init__(
            f"{report.stage} quality gate failed ({failed}); "
            f"evidence persisted as quality report {quality_report_id}."
        )
        self.report = report
        self.quality_report_id = quality_report_id


__all__ = [
    "DATA_QUALITY_POLICY_VERSION",
    "STAGE_DATASET_BUILD",
    "STAGE_REPRODUCTION",
    "STAGE_TRAINING",
    "ArtifactIntegrityError",
    "CheckResult",
    "CheckStatus",
    "DataQualityGateError",
    "DataQualityPolicy",
    "QualityReport",
    "check_artifact_integrity",
    "require_artifact_integrity",
    "worst_status",
]
