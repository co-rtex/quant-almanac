"""Server-rendered, read-only operator console for model trust and lineage evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from quant_signal.api.deps import get_governance_service
from quant_signal.console import views
from quant_signal.governance import GovernanceNotFoundError, GovernanceService

CONSOLE_DIR = Path(__file__).resolve().parents[2] / "console"
STATIC_DIR = CONSOLE_DIR / "static"

templates = Jinja2Templates(directory=str(CONSOLE_DIR / "templates"))
templates.env.filters.update(
    {
        "num": views.fmt_num,
        "pct": views.fmt_pct,
        "short": views.short_id,
        "pretty": views.pretty_json,
        "status_icon": views.status_icon,
        "status_label": views.status_label,
        "status_class": views.status_class,
    }
)

router = APIRouter(tags=["console"], include_in_schema=False)


def _render(
    request: Request,
    template: str,
    context: dict[str, Any],
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(request, template, context, status_code=status_code)


@router.get("/")
def root() -> RedirectResponse:
    """Send browsers to the console."""

    return RedirectResponse(url="/console", status_code=307)


@router.get("/console", response_class=HTMLResponse)
def console_index(
    request: Request,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> HTMLResponse:
    """List models and recent quality evidence, including blocked dataset builds."""

    models = service.list_models()
    reports = service.list_quality_reports(limit=50)
    blocked_count = service.count_quality_reports(stage="dataset_build", status="fail")
    return _render(
        request,
        "index.html",
        {
            "models": models,
            "reports": reports,
            "blocked_count": blocked_count,
            "dataset_count": len({model["dataset_version_id"] for model in models}),
            "synthetic": any(model["synthetic"] for model in models),
        },
    )


@router.get("/console/models/{model_version_id}", response_class=HTMLResponse)
def console_model(
    model_version_id: str,
    request: Request,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> HTMLResponse:
    """Render the full evidence page for one model version."""

    try:
        evidence = service.model_evidence(model_version_id)
    except GovernanceNotFoundError:
        return _render(
            request,
            "not_found.html",
            {"kind": "Model version", "identifier": model_version_id},
            status_code=404,
        )
    test = next((row for row in evidence["evaluations"] if row["split_name"] == "test"), None)
    latest_shap = evidence["explanations"][0] if evidence["explanations"] else None
    return _render(
        request,
        "model.html",
        {
            "e": evidence,
            "synthetic": evidence["synthetic"],
            "test": test,
            "calibration": views.calibration_chart(test["calibration_bins"]) if test else None,
            "timeline": views.split_timeline(evidence["model"]["training_config"]),
            "importance": (
                views.importance_chart(latest_shap["global_importance"]) if latest_shap else None
            ),
            "latest_shap": latest_shap,
            "latest_backtest": evidence["backtests"][0] if evidence["backtests"] else None,
        },
    )


@router.get("/console/quality-reports/{quality_report_id}", response_class=HTMLResponse)
def console_quality_report(
    quality_report_id: str,
    request: Request,
    service: GovernanceService = Depends(get_governance_service),  # noqa: B008
) -> HTMLResponse:
    """Render one quality report, including blocked builds that have no dataset."""

    try:
        report = service.get_quality_report(quality_report_id)
    except GovernanceNotFoundError:
        return _render(
            request,
            "not_found.html",
            {"kind": "Quality report", "identifier": quality_report_id},
            status_code=404,
        )
    lineage = (report.get("context") or {}).get("lineage") or {}
    source_bars = lineage.get("source_bars") or {}
    categories: dict[str, list[dict[str, Any]]] = {}
    for check in report["checks"]:
        categories.setdefault(str(check.get("category", "other")), []).append(check)
    return _render(
        request,
        "report.html",
        {
            "report": report,
            "categories": categories,
            "source_bars": source_bars,
            "synthetic": "synthetic" in (source_bars.get("sources") or []),
        },
    )
