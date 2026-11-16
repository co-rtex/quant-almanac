# Changelog

## Unreleased

### Added — Data Quality, Governance & Lineage

- `quant_signal.quality`: versioned `DataQualityPolicy` (`dq_v1`) and 23 dataset-build checks (13 source-bar, 10 dataset), including point-in-time feature recomputation and forward-return label alignment
- Blocking dataset quality gate; blocked builds persist evidence and write no artifact or manifest
- Training gate: dataset artifact integrity, dataset verdict, split ordering, realized-date label-horizon embargo, class coverage
- `quality_reports` table (migration `20261116_0002`) for `dataset_build`, `training`, and `reproduction` evidence
- Lineage capture: source-bar fingerprint, per-symbol/source/ingestion-run provenance, hyperparameters, split configuration, code revision, and runtime versions
- Backtest walk-forward guard and artifact-integrity checks for backtests, SHAP, and signal publication
- `GovernanceService`: lineage, live verification, refit-based reproduction, Hanley–McNeil AUC intervals, evidence-derived limitations
- CLI commands `verify-model`, `reproduce-model`, `quality-reports`; gate errors include `quality_report_id`
- `/v1/governance/*` read-only JSON API and the server-rendered Model Trust & Lineage Console (`/console`) with a strict CSP
- Deterministic `synthetic` market data provider with documented defect injection, and `quant-almanac-demo` end-to-end demonstration
- Unit, integration, migration round-trip, and Playwright browser tests; CI steps for PostgreSQL migration round-trip, demo, and browser tests

### Fixed

- Dataset, model, backtest, and SHAP artifacts could be overwritten on rebuild/retrain/re-run, invalidating hashes recorded by earlier registry rows; they are now content-addressed
- Zero-volume sessions produced infinite volume features; undefined ratios are now missing values
- `train` with a horizon the dataset lacks raised a raw `KeyError`; it now raises a clear `ValueError` before any work
- `GET /v1/models/{id}` returned 500 for unknown ids; it now returns 404
- Migration tests honored the CI `DATABASE_URL` and inspected an empty SQLite file, which had failed every CI run; they now pin their own database
- NaN hyperparameters are stored as strings so model metadata is valid JSON on PostgreSQL

### Added

- Standalone repository bootstrap for the quant forecasting platform
- Project packaging, developer workflow, CI scaffold, and environment configuration
- Architecture notes, execution plan, and delivery log
- FastAPI health endpoints and basic runtime configuration
- SQLAlchemy models, database helpers, Alembic environment, and the initial persistence schema
- Readiness checks backed by real database connectivity
- Provider abstraction, ingestion service, and persisted normalized daily bar workflow
- Feature engineering, forward-return label generation, temporal split utilities, and dataset artifact materialization
- Baseline model training, Platt-style calibration, evaluation metrics, and champion selection rules
- Persisted model registry records, signal snapshots, and read-only FastAPI endpoints for signals and model metadata
- Walk-forward backtesting with benchmark regime summaries and persisted backtest artifacts
- Configurable transaction cost and slippage modeling for walk-forward backtests
- Benchmark-relative backtest analytics with richer momentum and drawdown regime context
- Turnover-aware backtest reporting with companion detail artifacts
- Attribution-ready backtest summaries built from benchmark-relative detail contributions
- Regime-aware attribution summaries for primary regimes and grouped regime dimensions
- Provider fetch envelopes, config-driven provider selection, and richer ingestion run metadata
- Provider failure classification and retry metadata contracts for ingestion runs
- Deterministic retry handling for transient ingestion failures
- Shared pipeline CLI foundation with a scheduler-friendly `ingest` command
- Scheduler-friendly `build-dataset` pipeline command with machine-readable dataset manifest output
- Scheduler-friendly `train` pipeline command with machine-readable model summaries and champion model IDs
- Scheduler-friendly `backtest` pipeline command with machine-readable run summaries, artifact references, and resolved execution assumptions
- Scheduler-friendly `explain` pipeline command with machine-readable SHAP run summaries and artifact references
- Public, idempotent signal snapshot refresh interface on `TrainingService`
- Scheduler-friendly `publish-signals` pipeline command with machine-readable snapshot publication summaries
- SHAP explainability workflow with persisted global and local explanation artifacts tied to model versions

### Validated

- `make lint`
- `make typecheck`
- `make test`
- `make db-up`
- `make migrate`
- synthetic ingestion and dataset materialization integration test
- training-to-API integration test covering model persistence and signal serving
- backtesting and explainability integration test covering persisted artifacts
- cost-aware backtesting unit and integration coverage
- benchmark-relative analytics and richer regime coverage for backtests
- turnover analytics unit and integration coverage for backtests
- attribution-ready detail summary coverage for backtests
- regime-aware attribution summary coverage for backtests
- provider metadata/configuration coverage for ingestion
- provider failure classification coverage for ingestion
- deterministic retry execution coverage for ingestion
- pipeline CLI integration coverage for command execution and failure handling
- pipeline CLI dataset command coverage for success and empty-history failures
- pipeline CLI training command coverage for success and unknown-dataset failures
- pipeline CLI backtest command coverage for success and unknown-model failures
- pipeline CLI explain command coverage for success and unknown-model failures
- pipeline CLI publish-signals coverage for success, idempotent reruns, and unknown-model failures

### Next

- Authentication before any shared console deployment; statistically defensible drift reporting
