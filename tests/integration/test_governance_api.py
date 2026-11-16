"""API and console tests for governance evidence (read-only)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from governance_fixtures import END, START, GovernedEnvironment, build_environment

from quant_signal.api.app import create_app
from quant_signal.api.deps import get_governance_service, get_signal_service
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.governance import GovernanceService
from quant_signal.ingestion.service import IngestionService
from quant_signal.ingestion.synthetic import SyntheticMarketDataProvider
from quant_signal.quality import DataQualityGateError
from quant_signal.serving.service import SignalService
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import ModelVersion, QualityReportRecord


@pytest.fixture(scope="module")
def governed(tmp_path_factory: pytest.TempPathFactory) -> tuple[GovernedEnvironment, str]:
    env = build_environment(tmp_path_factory.mktemp("governance-api"))
    IngestionService(
        provider=SyntheticMarketDataProvider(),
        settings=env.settings,
        sleep_fn=lambda _: None,
    ).ingest_daily_bars(["BADQ"], START, END)
    with pytest.raises(DataQualityGateError) as raised:
        FeaturePipeline(settings=env.settings).build_dataset(END, ["SYNA", "BADQ"])
    return env, raised.value.quality_report_id


@pytest.fixture()
def client(governed: tuple[GovernedEnvironment, str]) -> TestClient:
    env, _ = governed
    app = create_app()
    app.dependency_overrides[get_governance_service] = lambda: GovernanceService(
        settings=env.settings
    )
    app.dependency_overrides[get_signal_service] = lambda: SignalService(settings=env.settings)
    return TestClient(app)


def test_governance_json_endpoints(
    client: TestClient,
    governed: tuple[GovernedEnvironment, str],
) -> None:
    env, blocked_id = governed
    champion = env.champion()

    models = client.get("/v1/governance/models")
    assert models.status_code == 200
    assert {row["model_version_id"] for row in models.json()} == {m.id for m in env.models}

    lineage = client.get(f"/v1/governance/models/{champion.id}/lineage").json()
    assert lineage["dataset"]["dataset_version_id"] == env.dataset.id
    assert lineage["training_quality_report"]["status"] == "pass"
    assert [row["split_name"] for row in lineage["evaluations"]] == ["test", "validation"]

    verification = client.get(f"/v1/governance/models/{champion.id}/verification")
    assert verification.status_code == 200
    body = verification.json()
    assert body["failure_count"] == 0
    assert {check["check_id"] for check in body["checks"]} >= {
        "artifact.dataset_integrity",
        "artifact.model_integrity",
        "lineage.source_bar_fingerprint",
    }

    evidence = client.get(f"/v1/governance/models/{champion.id}/evidence").json()
    assert evidence["limitations"]
    assert len(evidence["stages"]) == 8

    blocked = client.get("/v1/governance/quality-reports", params={"status": "fail"}).json()
    assert [report["quality_report_id"] for report in blocked] == [blocked_id]
    detail = client.get(f"/v1/governance/quality-reports/{blocked_id}").json()
    assert detail["dataset_version_id"] is None
    assert detail["failure_count"] >= 1

    dataset = client.get(f"/v1/governance/datasets/{env.dataset.id}/quality").json()
    assert {report["stage"] for report in dataset["quality_reports"]} == {
        "dataset_build",
        "training",
    }


def test_governance_endpoints_validate_input_and_return_404(client: TestClient) -> None:
    assert client.get("/v1/governance/models/missing/lineage").status_code == 404
    assert client.get("/v1/governance/models/missing/verification").status_code == 404
    assert client.get("/v1/governance/quality-reports/missing").status_code == 404
    assert client.get("/v1/governance/datasets/missing/quality").status_code == 404
    assert client.get("/v1/governance/quality-reports", params={"status": "bad"}).status_code == 422
    assert client.get("/v1/models/missing").status_code == 404


def test_console_pages_render_persisted_evidence(
    client: TestClient,
    governed: tuple[GovernedEnvironment, str],
) -> None:
    env, blocked_id = governed
    champion = env.champion()

    index = client.get("/console")
    assert index.status_code == 200
    assert "SYNTHETIC DATA" in index.text
    assert 'data-testid="blocked-count">1<' in index.text
    assert champion.id in index.text
    assert "default-src 'none'" in index.headers["content-security-policy"]
    assert index.headers["x-frame-options"] == "DENY"

    page = client.get(f"/console/models/{champion.id}")
    assert page.status_code == 200
    for marker in (
        'data-testid="lineage-stepper"',
        'data-check-id="artifact.model_integrity"',
        'data-testid="limitations"',
        "Test-split reliability",
        env.dataset.artifact_hash,
    ):
        assert marker in page.text
    assert "<script" not in page.text

    report = client.get(f"/console/quality-reports/{blocked_id}")
    assert report.status_code == 200
    assert 'data-testid="blocked-notice"' in report.text
    assert 'data-check-id="bars.ohlc_consistency" data-status="fail"' in report.text

    assert client.get("/console/static/console.css").status_code == 200
    missing = client.get("/console/models/does-not-exist")
    assert missing.status_code == 404
    assert "Model version not found" in missing.text
    assert client.get("/", follow_redirects=False).headers["location"] == "/console"


def test_console_escapes_untrusted_report_content(
    client: TestClient,
    governed: tuple[GovernedEnvironment, str],
) -> None:
    env, blocked_id = governed

    with session_scope(env.settings.database_url) as session:
        report = session.get(QualityReportRecord, blocked_id)
        assert report is not None
        original = report.summary
        report.summary = "<script>alert(1)</script>"
    try:
        page = client.get(f"/console/quality-reports/{blocked_id}")
        assert "<script>alert(1)</script>" not in page.text
        assert "&lt;script&gt;" in page.text
    finally:
        with session_scope(env.settings.database_url) as session:
            restored = session.get(QualityReportRecord, blocked_id)
            assert restored is not None
            restored.summary = original


def test_console_renders_partially_recorded_models(
    client: TestClient,
    governed: tuple[GovernedEnvironment, str],
) -> None:
    env, _ = governed
    candidate = next(model for model in env.models if model.champion_rank != 1)
    with session_scope(env.settings.database_url) as session:
        model = session.get(ModelVersion, candidate.id)
        assert model is not None
        original = dict(model.metadata_json)
        model.metadata_json = {
            **original,
            "training_config": {"pipeline_steps": [{"name": "model"}]},
            "provenance": {"runtime": {}},
        }
    try:
        page = client.get(f"/console/models/{candidate.id}")
        assert page.status_code == 200
        assert "Unavailable" in page.text
        with session_scope(env.settings.database_url) as session:
            model = session.get(ModelVersion, candidate.id)
            assert model is not None
            model.metadata_json = {}
        assert client.get(f"/console/models/{candidate.id}").status_code == 200
    finally:
        with session_scope(env.settings.database_url) as session:
            model = session.get(ModelVersion, candidate.id)
            assert model is not None
            model.metadata_json = original
