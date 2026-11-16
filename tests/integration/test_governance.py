"""Integration tests for quality gates, lineage capture, verification, and reproduction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from governance_fixtures import END, build_environment, governed_settings, ingest, reload_model
from sqlalchemy import func, select

from quant_signal.backtesting.service import BacktestService
from quant_signal.explainability.service import ExplainabilityService
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.governance import GovernanceNotFoundError, GovernanceService
from quant_signal.quality import ArtifactIntegrityError, DataQualityGateError
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import (
    DailyBar,
    DatasetVersion,
    ModelVersion,
    QualityReportRecord,
    Symbol,
)
from quant_signal.storage.repositories import StorageRepository
from quant_signal.training.service import TrainingService


def test_passing_build_persists_linked_report_and_lineage(tmp_path: Path) -> None:
    env = build_environment(tmp_path, train=False)
    metadata = env.dataset.metadata_json
    quality = metadata["quality"]
    lineage = metadata["lineage"]

    assert quality["status"] == "pass"
    assert lineage["source_bars"]["sources"] == ["synthetic"]
    assert len(lineage["source_bars"]["ingestion_run_ids"]) == 1
    assert len(lineage["source_bars"]["fingerprint_sha256"]) == 64
    assert lineage["code"]["source"] in {"git", "environment", "unavailable"}
    assert env.dataset.artifact_hash[:12] in env.dataset.artifact_path

    with session_scope(env.settings.database_url) as session:
        report = StorageRepository(session).get_quality_report(quality["report_id"])
        assert report.dataset_version_id == env.dataset.id
        assert report.stage == "dataset_build"
        check_ids = {check["check_id"] for check in report.checks_json}
    assert {"bars.ohlc_consistency", "dataset.point_in_time_features"} <= check_ids
    assert report.context_json["policy"]["version"] == "dq_v1"


def test_rebuild_never_overwrites_an_artifact_referenced_by_an_older_manifest(
    tmp_path: Path,
) -> None:
    env = build_environment(tmp_path, train=False)
    original_path = Path(env.dataset.artifact_path)
    original_bytes = original_path.read_bytes()
    with session_scope(env.settings.database_url) as session:
        bar = session.scalars(
            select(DailyBar).join(Symbol).where(Symbol.symbol == "SYNA").limit(1)
        ).one()
        bar.volume = bar.volume + 1_000
    rebuilt = FeaturePipeline(settings=env.settings).build_dataset(END, ["SYNA", "SYNB", "SYNC"])

    assert rebuilt.artifact_path != env.dataset.artifact_path
    assert original_path.read_bytes() == original_bytes
    assert (
        rebuilt.metadata_json["lineage"]["source_bars"]["fingerprint_sha256"]
        != (env.dataset.metadata_json["lineage"]["source_bars"]["fingerprint_sha256"])
    )


def test_defective_source_bars_block_the_build_and_persist_evidence(tmp_path: Path) -> None:
    settings = governed_settings(tmp_path)
    ingest(settings, ["SYNA", "BADQ"])

    with pytest.raises(DataQualityGateError) as raised:
        FeaturePipeline(settings=settings).build_dataset(END, ["SYNA", "BADQ"])

    failed = {check.check_id for check in raised.value.report.failures}
    # The window ends before the zero-close defect date, so two of three defects apply.
    assert failed == {"bars.ohlc_consistency", "bars.calendar_alignment"}
    with session_scope(settings.database_url) as session:
        assert session.scalar(select(func.count(DatasetVersion.id))) == 0
        report = session.get(QualityReportRecord, raised.value.quality_report_id)
        assert report is not None
        assert report.status == "fail"
        assert report.dataset_version_id is None
        assert report.request_json["modeling_symbols"] == ["BADQ", "SYNA"]
        assert report.context_json["notes"]
    assert not (settings.artifact_root / "datasets").exists() or not any(
        (settings.artifact_root / "datasets").glob("*.parquet")
    )


def test_training_persists_gate_reports_and_reproducible_configuration(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    governance = champion.metadata_json["governance"]
    config = champion.metadata_json["training_config"]

    assert governance["training_quality_status"] == "pass"
    assert governance["dataset_artifact_sha256"] == env.dataset.artifact_hash
    assert config["split"]["embargo_days"] == 5
    assert config["pipeline_steps"][-1]["name"] == "model"
    assert champion.metadata_json["provenance"]["runtime"]["packages"]["scikit-learn"]
    with session_scope(env.settings.database_url) as session:
        reports = StorageRepository(session).list_quality_reports(
            stage="training",
            dataset_version_id=env.dataset.id,
        )
    assert sorted(report.horizon_days or 0 for report in reports) == [1, 5]
    embargo = next(
        check
        for check in reports[0].checks_json
        if check["check_id"] == "training.label_horizon_embargo"
    )
    assert embargo["status"] == "pass"


def test_training_refuses_a_tampered_dataset_artifact(tmp_path: Path) -> None:
    env = build_environment(tmp_path, train=False)
    with Path(env.dataset.artifact_path).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(DataQualityGateError) as raised:
        TrainingService(settings=env.settings).train(env.dataset.id)

    assert [check.check_id for check in raised.value.report.failures] == [
        "training.dataset_artifact_integrity"
    ]
    with session_scope(env.settings.database_url) as session:
        assert session.scalar(select(func.count(ModelVersion.id))) == 0


def test_training_refuses_a_dataset_whose_build_report_failed(tmp_path: Path) -> None:
    env = build_environment(tmp_path, train=False)
    with session_scope(env.settings.database_url) as session:
        report = session.get(QualityReportRecord, env.dataset.metadata_json["quality"]["report_id"])
        assert report is not None
        report.status = "fail"

    with pytest.raises(DataQualityGateError):
        TrainingService(settings=env.settings).train(env.dataset.id)


def test_legacy_dataset_without_report_trains_with_a_recorded_warning(tmp_path: Path) -> None:
    env = build_environment(tmp_path, train=False)
    with session_scope(env.settings.database_url) as session:
        for report in session.scalars(select(QualityReportRecord)):
            session.delete(report)
        dataset = session.get(DatasetVersion, env.dataset.id)
        assert dataset is not None
        dataset.metadata_json = {
            key: value
            for key, value in dataset.metadata_json.items()
            if key not in {"lineage", "quality"}
        }

    models = TrainingService(settings=env.settings).train(env.dataset.id, horizons=[5])
    assert models[0].metadata_json["governance"]["training_quality_status"] == "warn"

    evidence = GovernanceService(settings=env.settings).model_evidence(models[0].id)
    assert evidence["dataset"]["lineage_recorded"] is False
    assert evidence["ingestion_runs"] == []
    stages = {stage["key"]: stage["status"] for stage in evidence["stages"]}
    assert stages["dataset"] == "unavailable"
    unavailable = [
        check
        for check in evidence["verification"]["checks"]
        if check["message"].startswith("Unavailable")
    ]
    assert {check["check_id"] for check in unavailable} >= {
        "lineage.source_bar_fingerprint",
        "lineage.ingestion_runs",
        "quality.dataset_gate",
    }


def test_verification_passes_then_detects_tampering_and_source_drift(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    service = GovernanceService(settings=env.settings)

    baseline = {check.check_id: check.status for check in service.verify_model(champion.id).checks}
    assert baseline["artifact.dataset_integrity"] == "pass"
    assert baseline["artifact.model_integrity"] == "pass"
    assert baseline["model.bundle_consistency"] == "pass"
    assert baseline["lineage.source_bar_fingerprint"] == "pass"
    assert baseline["lineage.ingestion_runs"] == "pass"
    assert baseline["evaluation.split_boundaries"] == "pass"

    with session_scope(env.settings.database_url) as session:
        bar = session.scalars(
            select(DailyBar).join(Symbol).where(Symbol.symbol == "SYNA").limit(1)
        ).one()
        bar.volume = bar.volume + 1
    drift = {check.check_id: check.status for check in service.verify_model(champion.id).checks}
    assert drift["lineage.source_bar_fingerprint"] == "warn"
    assert drift["artifact.dataset_integrity"] == "pass"

    with Path(champion.artifact_path).open("ab") as handle:
        handle.write(b"tamper")
    tampered = service.verify_model(champion.id)
    statuses = {check.check_id: check.status for check in tampered.checks}
    assert statuses["artifact.model_integrity"] == "fail"
    assert tampered.status == "fail"
    with pytest.raises(ArtifactIntegrityError):
        TrainingService(settings=env.settings).refresh_signal_snapshots(champion.id)


def test_bundle_registry_mismatch_is_detected(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    bundle = joblib.load(champion.artifact_path)
    bundle.horizon_days = 1
    joblib.dump(bundle, champion.artifact_path)
    with session_scope(env.settings.database_url) as session:
        from quant_signal.core.hashing import sha256_file

        model = session.get(ModelVersion, champion.id)
        assert model is not None
        model.artifact_hash = sha256_file(Path(champion.artifact_path))

    report = GovernanceService(settings=env.settings).verify_model(champion.id)
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses["model.bundle_consistency"] == "fail"


def test_reproduction_refits_exactly_and_is_persisted(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    service = GovernanceService(settings=env.settings)

    report, report_id = service.reproduce_model(champion.id)
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses == {
        "reproduction.dataset_integrity": "pass",
        "reproduction.model_integrity": "pass",
        "reproduction.split_windows": "pass",
        "reproduction.hyperparameters": "pass",
        "reproduction.prediction_equivalence": "pass",
        "reproduction.test_metrics": "pass",
        "reproduction.runtime_versions": "pass",
    }
    verification = {check.check_id: check for check in service.verify_model(champion.id).checks}
    assert verification["reproduction.latest"].observed["quality_report_id"] == report_id


def test_reproduction_detects_changed_hyperparameters(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    with session_scope(env.settings.database_url) as session:
        model = session.get(ModelVersion, champion.id)
        assert model is not None
        metadata = dict(model.metadata_json)
        config = dict(metadata["training_config"])
        steps = [dict(step) for step in config["pipeline_steps"]]
        steps[-1] = {**steps[-1], "params": {**steps[-1]["params"], "random_state": 7}}
        config["pipeline_steps"] = steps
        metadata["training_config"] = config
        model.metadata_json = metadata

    report, _ = GovernanceService(settings=env.settings).reproduce_model(champion.id)
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses["reproduction.hyperparameters"] == "fail"
    assert report.status == "fail"


def test_backtest_and_shap_record_lineage_and_never_overwrite(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    backtest = BacktestService(settings=env.settings).run(champion.id, top_n=1)
    guard = backtest.metadata_json["walk_forward_guard"]
    assert guard["status"] == "pass"
    assert guard["windows_trained"] > 0
    assert guard["min_sessions_between_last_label_and_scoring"] >= champion.horizon_days
    assert backtest.metadata_json["lineage"]["dataset_artifact_sha256"] == env.dataset.artifact_hash

    explain = ExplainabilityService(settings=env.settings)
    first = explain.generate(champion.id, sample_size=8, top_signals=2)
    second = explain.generate(champion.id, sample_size=12, top_signals=2)
    assert first.artifact_path != second.artifact_path
    assert first.summary_json["methodology"]["evaluation_window"]["split"] == "test"

    evidence = GovernanceService(settings=env.settings).model_evidence(champion.id)
    statuses = {check["check_id"]: check["status"] for check in evidence["verification"]["checks"]}
    assert statuses["artifact.backtest_integrity"] == "pass"
    assert statuses["artifact.explainability_integrity"] == "pass"
    assert len(evidence["explanations"]) == 2
    assert evidence["synthetic"] is True
    assert evidence["limitations"][0]["source"] == "data source"
    stage_keys = [stage["key"] for stage in evidence["stages"]]
    assert stage_keys == [
        "source",
        "ingestion",
        "dataset",
        "training_gate",
        "model",
        "evaluation",
        "backtest",
        "explanation",
    ]


def test_unknown_identifiers_raise_not_found(tmp_path: Path) -> None:
    env = build_environment(tmp_path, train=False)
    service = GovernanceService(settings=env.settings)
    with pytest.raises(GovernanceNotFoundError):
        service.model_lineage("missing")
    with pytest.raises(GovernanceNotFoundError):
        service.get_quality_report("missing")
    with pytest.raises(GovernanceNotFoundError):
        service.get_dataset_quality("missing")


def test_model_listing_reports_stored_statuses(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    rows = GovernanceService(settings=env.settings).list_models()
    assert len(rows) == len(env.models)
    assert {row["dataset_quality_status"] for row in rows} == {"pass"}
    assert {row["training_quality_status"] for row in rows} == {"pass"}
    assert all(row["synthetic"] for row in rows)
    reloaded = reload_model(env.settings, rows[0]["model_version_id"])
    assert reloaded.id == rows[0]["model_version_id"]


def test_zero_volume_halt_warns_but_does_not_block_the_build(tmp_path: Path) -> None:
    settings = governed_settings(tmp_path)
    ingest(settings)
    with session_scope(settings.database_url) as session:
        bars = session.scalars(
            select(DailyBar)
            .join(Symbol)
            .where(Symbol.symbol == "SYNA")
            .order_by(DailyBar.trade_date)
        ).all()
        bars[100].volume = 0
    dataset = FeaturePipeline(settings=settings).build_dataset(END, ["SYNA", "SYNB", "SYNC"])
    with session_scope(settings.database_url) as session:
        report = StorageRepository(session).get_quality_report(
            dataset.metadata_json["quality"]["report_id"]
        )
        statuses = {check["check_id"]: check["status"] for check in report.checks_json}
    assert statuses["bars.volume"] == "warn"
    assert statuses["dataset.non_finite_features"] == "pass"
    assert dataset.metadata_json["quality"]["status"] == "warn"


def test_lineage_stages_reflect_live_tampering_and_drift(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    service = GovernanceService(settings=env.settings)

    def stages() -> dict[str, str]:
        evidence = service.model_evidence(champion.id)
        return {stage["key"]: stage["status"] for stage in evidence["stages"]}

    assert stages()["ingestion"] == "pass"
    with session_scope(env.settings.database_url) as session:
        bar = session.scalars(
            select(DailyBar).join(Symbol).where(Symbol.symbol == "SYNB").limit(1)
        ).one()
        bar.volume = bar.volume + 7
    assert stages()["ingestion"] == "warn"

    with Path(env.dataset.artifact_path).open("ab") as handle:
        handle.write(b"tamper")
    after = stages()
    assert after["dataset"] == "fail"
    assert after["evaluation"] in {"warn", "fail"}


def test_training_gates_every_horizon_before_fitting_any_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = build_environment(tmp_path, train=False)
    with pytest.raises(ValueError, match="no labels for horizons"):
        TrainingService(settings=env.settings).train(env.dataset.id, horizons=[5, 250])

    from quant_signal.quality.report import CheckResult
    from quant_signal.training import service as training_module

    real_checks = training_module.run_split_checks

    def failing_for_5d(*args: object, **kwargs: object) -> list[CheckResult]:
        results = real_checks(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("horizon") == 5:
            results.append(CheckResult("training.synthetic_failure", "test", "fail", "forced"))
        return results

    monkeypatch.setattr(training_module, "run_split_checks", failing_for_5d)
    with pytest.raises(DataQualityGateError):
        TrainingService(settings=env.settings).train(env.dataset.id, horizons=[1, 5])
    with session_scope(env.settings.database_url) as session:
        assert session.scalar(select(func.count(ModelVersion.id))) == 0
        reports = StorageRepository(session).list_quality_reports(stage="training")
    assert sorted((report.horizon_days, report.status) for report in reports) == [
        (1, "pass"),
        (5, "fail"),
    ]


def test_backtest_artifacts_are_content_addressed(tmp_path: Path) -> None:
    env = build_environment(tmp_path)
    champion = env.champion()
    service = BacktestService(settings=env.settings)
    first = service.run(champion.id, top_n=1)
    second = service.run(champion.id, top_n=1)
    for run in (first, second):
        assert run.artifact_hash[:12] in run.artifact_path
        assert (
            run.metadata_json["detail_artifact_hash"][:12]
            in run.metadata_json["detail_artifact_path"]
        )
    statuses = {
        check.check_id: check.status
        for check in GovernanceService(settings=env.settings).verify_model(champion.id).checks
    }
    assert statuses["artifact.backtest_integrity"] == "pass"
