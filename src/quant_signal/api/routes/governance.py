"""Read-only governance endpoints: quality evidence, lineage, and live verification."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from quant_signal.api.deps import get_governance_service
from quant_signal.governance import GovernanceNotFoundError, GovernanceService

router = APIRouter(prefix="/v1/governance", tags=["governance"])

StageFilter = Literal["dataset_build", "training", "reproduction"]
StatusFilter = Literal["pass", "warn", "fail"]


class CheckResultResponse(BaseModel):
    """One deterministic check with its evidence."""

    check_id: str
    category: str
    status: str
    message: str
    observed: dict[str, Any]
    threshold: dict[str, Any]
    examples: list[dict[str, Any]]


class VerificationResponse(BaseModel):
    """Live re-verification of a model's artifacts, lineage, and gates."""

    model_version_id: str
    stage: str
    status: str
    check_count: int
    failure_count: int
    warning_count: int
    failed_checks: list[str]
    warning_checks: list[str]
    checks: list[CheckResultResponse]


class ModelGovernanceSummary(BaseModel):
    """Stored governance status for one registry model."""

    model_version_id: str
    dataset_version_id: str
    horizon_days: int
    model_family: str
    champion_rank: int | None
    created_at: str | None
    dataset_as_of_date: str | None
    dataset_symbols: list[str]
    dataset_quality_status: str | None
    training_quality_status: str | None
    test_roc_auc: float | None
    test_pr_auc: float | None
    synthetic: bool


def _not_found(exc: GovernanceNotFoundError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/models", response_model=list[ModelGovernanceSummary])
def list_models(
    limit: int = Query(default=100, ge=1, le=500),
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> list[dict[str, Any]]:
    """List registry models with their persisted quality and gate status."""

    return service.list_models(limit=limit)


@router.get("/models/{model_version_id}/lineage")
def get_model_lineage(
    model_version_id: str,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> dict[str, Any]:
    """Return the persisted chain: ingestion → dataset → gates → model → evaluation → runs."""

    try:
        return service.model_lineage(model_version_id)
    except GovernanceNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/models/{model_version_id}/verification", response_model=VerificationResponse)
def verify_model(
    model_version_id: str,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> dict[str, Any]:
    """Re-hash artifacts and re-check lineage pointers and gates (read-only, not persisted)."""

    try:
        report = service.verify_model(model_version_id)
    except GovernanceNotFoundError as exc:
        raise _not_found(exc) from exc
    return {
        "model_version_id": model_version_id,
        **report.summary_json(),
        "checks": report.checks_json(),
    }


@router.get("/models/{model_version_id}/evidence")
def get_model_evidence(
    model_version_id: str,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> dict[str, Any]:
    """Return lineage plus live verification, stage statuses, and derived limitations."""

    try:
        return service.model_evidence(model_version_id)
    except GovernanceNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/quality-reports")
def list_quality_reports(
    stage: StageFilter | None = None,
    report_status: StatusFilter | None = Query(default=None, alias="status"),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=500),
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> list[dict[str, Any]]:
    """List persisted quality reports, including builds the gate blocked."""

    return service.list_quality_reports(stage=stage, status=report_status, limit=limit)


@router.get("/quality-reports/{quality_report_id}")
def get_quality_report(
    quality_report_id: str,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> dict[str, Any]:
    """Return one quality report with every check, threshold, and example."""

    try:
        return service.get_quality_report(quality_report_id)
    except GovernanceNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/datasets/{dataset_version_id}/quality")
def get_dataset_quality(
    dataset_version_id: str,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> dict[str, Any]:
    """Return a dataset manifest and every quality report that references it."""

    try:
        return service.get_dataset_quality(dataset_version_id)
    except GovernanceNotFoundError as exc:
        raise _not_found(exc) from exc
