# Architecture

## Design Goals

- Reproducibility over speed
- Explicit boundaries between HTTP, orchestration, domain logic, and persistence
- Local developer ergonomics with containerized infrastructure and artifact versioning
- Boring, reviewable solutions that can expand without large rewrites

## System Boundaries

- `quant_signal.api`: FastAPI application, route handlers, request and response schemas
- `quant_signal.cli`: thin command wrappers for scheduler-friendly service execution
- `quant_signal.core`: configuration, hashing, logging, shared constants
- `quant_signal.storage`: SQLAlchemy models, database sessions, repository helpers
- `quant_signal.ingestion`: provider contract, development adapter, ingestion orchestration
- `quant_signal.features`: feature engineering, label generation, dataset versioning, split logic
- `quant_signal.training`: model training, calibration, evaluation orchestration, artifact persistence
- `quant_signal.evaluation`: metrics and reporting utilities
- `quant_signal.backtesting`: walk-forward simulation and regime analysis
- `quant_signal.explainability`: SHAP generation tied to model versions
- `quant_signal.serving`: ranked signal retrieval and model metadata read services
- `quant_signal.quality`: versioned data-quality policy, typed check results, bar/dataset/training gates, artifact integrity
- `quant_signal.governance`: lineage assembly, live verification, refit-based reproduction, evidence-derived limitations
- `quant_signal.console`: server-rendered, read-only Model Trust & Lineage Console (Jinja2 templates, CSS, view helpers)

## Persistence Model

- PostgreSQL stores normalized raw bars, ingestion runs, dataset manifests, model registry records, evaluations, backtests, explainability run metadata, and signal snapshots.
- Versioned Parquet artifacts under `artifacts/` store model-ready feature datasets and serialized model bundles.
- Hashes and metadata in PostgreSQL point to the exact artifact path used by downstream stages.
- Dataset, model, backtest, and SHAP artifacts are content-addressed (the SHA-256 prefix is part of the filename), so a rebuild or retrain never overwrites an artifact that an older registry row references.
- `quality_reports` is an append-only evidence table for `dataset_build`, `training`, and `reproduction` gates. It is the only durable record of builds the gate blocked.
- The v1 schema uses logically grouped tables in the default database schema to keep local testing portable; service boundaries do not depend on physical schema names, so a later Postgres schema split remains reversible.

## Data Flow

1. Ingestion fetches bars from a provider contract, retries transient provider failures with deterministic backoff, records request/provider/retry/provider-fetch metadata at the run level, and stores normalized OHLCV rows with source and ingestion metadata.
2. Feature generation queries persisted bars, joins benchmark context, creates leakage-safe features and labels, and writes a versioned dataset artifact.
3. Training loads a dataset artifact, applies time-aware splits, fits candidate models, calibrates probabilities, evaluates reliability, and registers the champion model per horizon.
4. Backtesting retrains monthly in walk-forward mode, scores daily candidates, simulates overlapping horizon sleeves, applies configurable transaction costs and slippage, and records gross/net performance summaries, benchmark-relative analytics, richer regime slices, turnover-aware detail artifacts, attribution-ready summaries, and regime-aware attribution summaries.
5. Explainability loads model bundles and evaluation slices, produces SHAP summaries, and stores explainability artifacts.
6. FastAPI serves health, model metadata, and ranked historical signal snapshots from persisted records.
7. The pipeline CLI initializes settings and logging once, dispatches explicit subcommands, and returns machine-readable summaries without owning business logic, retry policy, or workflow state.

## Contract Points

- `MarketDataProvider.fetch_daily_bars(symbols, start_date, end_date)` returns a `ProviderFetchResult` envelope with normalized daily bars, returned-symbol coverage, provider metadata, and warnings.
- Provider adapters raise `ProviderTransientError` or `ProviderPermanentError` so ingestion orchestration stays vendor-agnostic.
- Deterministic retry execution is owned by the ingestion layer and limited to provider fetches; database writes and run finalization remain single-attempt.
- `FeaturePipeline.build_dataset(as_of_date, symbols, feature_set_version)` owns feature and label materialization plus dataset manifest creation.
- `TrainingService.train(dataset_version_id, horizons)` owns candidate fitting, probability calibration, evaluation persistence, champion selection, and signal snapshot refresh.
- `TrainingService.refresh_signal_snapshots(model_version_id)` owns explicit, idempotent signal snapshot replacement for one persisted model version.
- `BacktestService.run(model_version_id, top_n)` retrains the registered model family in a monthly walk-forward loop and persists summary artifacts.
- `BacktestService.run(model_version_id, top_n, execution_assumptions)` supports reproducible cost-aware reruns without changing the database schema.
- `label_regimes(benchmark_frame)` preserves the primary trend/volatility regime while adding benchmark momentum and drawdown context for grouped analytics.
- Backtest runs now emit a primary daily artifact plus a companion detail artifact referenced from `metadata_json`, which keeps composition-level reporting schema-light and reversible.
- Attribution-ready reporting is derived from the detail artifact, which carries benchmark-relative contribution fields and lifecycle flags without changing the backtest database schema.
- Regime-aware attribution reporting stays daily-grain: primary regime summaries live in `regime_summary_json`, while dimension-based attribution summaries live under `summary_json`.
- `ExplainabilityService.generate(model_version_id, sample_size, top_signals)` binds SHAP outputs to a specific registered model artifact and evaluation window.
- `SignalService.get_ranked_signals(as_of_date, horizon, limit)` is the read-side contract used by the API layer.
- Ingestion run metadata stays schema-light and nested inside `metadata_json`: request parameters, provider configuration, attempt-level retry metadata, provider fetch diagnostics, and persistence counts are persisted without widening the database schema.
- `quant-signal-pipeline ingest`, `quant-signal-pipeline build-dataset`, `quant-signal-pipeline train`, `quant-signal-pipeline backtest`, `quant-signal-pipeline explain`, and `quant-signal-pipeline publish-signals` map directly to `IngestionService.ingest_daily_bars(...)`, `FeaturePipeline.build_dataset(...)`, `TrainingService.train(...)`, `BacktestService.run(...)`, `ExplainabilityService.generate(...)`, and `TrainingService.refresh_signal_snapshots(...)`, keeping scheduling concerns outside the domain service layer.

## Initial Tradeoffs

- Use sync SQLAlchemy sessions for simplicity and testability.
- Keep training request handling out of the API; API remains read-only.
- Use a free adapter first, but hide it behind a provider interface to avoid leaking vendor assumptions.
- Keep provider selection config-driven so scheduled entrypoints can become thin orchestration later.
- Store wide feature matrices in Parquet artifacts rather than a mutable wide SQL table.
- Keep backtesting assumptions explicit and simple: monthly retraining, equal-weight long-only sleeves, benchmark-derived regime labels, configurable per-side transaction cost/slippage assumptions, benchmark-relative analytics, and companion detail artifacts for turnover-aware, attribution-ready, and regime-aware reporting.

## Data Quality, Governance & Lineage (Phase 16)

### Decisions

- **Gate at consumption, not ingestion.** Raw bars are persisted as fetched; the dataset build is where bars become model inputs, so it runs the blocking checks. A failed build writes no artifact and no manifest, only a `quality_reports` row with the refused request, so nothing downstream can train on it.
- **One new table, no new sources of truth.** Lineage pointers live in the existing `metadata_json` columns (the repository's schema-light convention). `quality_reports` exists because blocked builds have no other home. Reports are immutable; the dataset manifest stores only a pointer plus the verdict.
- **Fail vs. warn is explicit and policy-versioned.** Checks that protect training or evaluation correctness fail (duplicate keys, OHLC violations, non-positive prices, calendar gaps above 5%, a benchmark calendar with more gaps than holidays, labels that differ from recomputation, look-ahead features, embargo violations, artifact tampering). Legitimate-but-notable market conditions warn (zero-volume halts, extreme moves, stale runs, IPO/delisting ranges are excluded from gap math). `DataQualityPolicy` is serialized into every report.
- **Prove look-ahead safety by recomputation.** `dataset.point_in_time_features` rebuilds every feature from history truncated at several cutoffs and requires identical values. The correlation screen is kept only as a warning because deterministic series can be legitimately predictable.
- **Embargo checks use realized label dates.** A row's label is realized on its symbol's own session h rows later, so the check is exact even with symbol-specific gaps, rather than inferring safety from the embargo length.
- **Gate all horizons before fitting.** `train` evaluates and persists every requested horizon's gate report before fitting any model, so a failing horizon cannot leave a partial model set behind.
- **Lineage stages combine verdicts.** Each console hop takes the worst of its build-time report and the live checks covering it.
- **Verification is live and read-only; reproduction is explicit and persisted.** Verification (hashing, bundle comparison, fingerprinting) is cheap enough to compute on each console view and never writes. Reproduction refits a model and is persisted as evidence.
- **Never invent history.** Legacy rows without lineage render as "Unavailable"; no back-filling from current state (e.g. `daily_bars.ingestion_run_id` is last-writer-wins, so it is not trusted retroactively).
- **Server-rendered console.** FastAPI + Jinja2 with no JavaScript keeps the stack single-language, testable with `TestClient`, and compatible with a strict CSP. Chart geometry is computed in Python view helpers.

### Contract Points

- `FeaturePipeline.build_dataset(...)` raises `DataQualityGateError(report, quality_report_id)` when a blocking check fails.
- `TrainingService.train(...)` raises `DataQualityGateError` on a tampered dataset artifact, a failed dataset report, or a failed split/embargo check; each horizon persists a `training` report referenced from `metadata_json.governance`.
- `BacktestService`, `ExplainabilityService`, and `TrainingService.refresh_signal_snapshots` raise `ArtifactIntegrityError` on modified artifacts.
- `GovernanceService.model_lineage / verify_model / reproduce_model / model_evidence` back the CLI (`verify-model`, `reproduce-model`, `quality-reports`), `/v1/governance/*`, and `/console`.

### Tradeoffs

- Content-addressed artifacts use more disk on repeated builds; correctness of historical manifests is worth it.
- Live verification re-hashes artifacts per page view; fine at research scale, and would move behind a cache or background job at larger scale.
- The console has no authentication and binds to loopback in the demo; a shared deployment needs auth in front of it.
