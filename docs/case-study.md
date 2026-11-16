# Case study: making a forecasting pipeline auditable

**Project:** Quant Almanac, a daily US equities forecasting pipeline (Python 3.14, FastAPI, SQLAlchemy/Alembic, PostgreSQL, Parquet, scikit-learn, SHAP).
**Scope of this change:** data-quality gates, experiment lineage, artifact verification, reproduction, and an operator console. No new forecasting models were added.

## Problem

The pipeline already ingested bars, built versioned Parquet datasets, trained calibrated candidates, ran walk-forward backtests, and produced SHAP summaries. Each registry row recorded an artifact hash. An audit showed that the hashes did not make results trustworthy:

| Gap | Consequence |
| --- | --- |
| No validation of source bars | Duplicate (symbol, date) rows from two providers, inverted high/low bars, zero prices, or missing sessions flowed straight into return-window features |
| Artifact paths keyed only by (feature set, date), (dataset, horizon, family), or model id | Rebuilding, retraining, or re-explaining **overwrote** the file an older row pointed to. The recorded hash went stale, and `train` could read different data than its manifest described |
| No dataset → ingestion link; no hyperparameters, library versions, or code revision on models | Provenance could not be stated or checked |
| No checks for label alignment or embargo sufficiency | Leakage-safety rested on reading the code |
| CI red on every run | `alembic/env.py` preferred the CI `DATABASE_URL`, so the migration test migrated PostgreSQL and then inspected an empty SQLite file |

## Approach

### 1. Gate at the point of consumption

The dataset build reads persisted bars and is where they become model inputs, so the blocking checks run there. The build produces one `QualityReport` of typed `CheckResult`s. Each result carries a status, message, observed values, thresholds, and offending rows. On any **fail**, the build persists the report and raises `DataQualityGateError`. It writes no Parquet file and no manifest. On pass or warn, the report is linked to the new `dataset_versions` row.

Severity follows from the domain:

- **Fail** for anything that corrupts features or labels: duplicate keys across providers, OHLC violations, non-positive prices, weekend or future timestamps, more than 5% of benchmark sessions missing inside a symbol's own trading range, a benchmark calendar with more gaps than holidays explain, infinite features, labels that differ from a recomputation, and look-ahead features.
- **Warn** for conditions that are legitimate in equity data but worth a reviewer's attention: zero-volume halts, moves above 40%, frozen prices, and a symbol whose history ends early. Calendar gaps are measured only between a symbol's first and last bar, so IPOs and delistings are not rejected.

All thresholds live in a versioned `DataQualityPolicy`, and each report stores the policy that produced it. A later policy change therefore never reinterprets old evidence.

### 2. Test for leakage instead of asserting it

- **Point-in-time recomputation.** For three date cutoffs, features are rebuilt from bars truncated at the cutoff, and every feature cell on or before it must match the full-history value to within 1e-9. A unit test shows the check catching a deliberately leaky `shift(-1)` feature and naming the feature, date, and both values.
- **Label alignment.** Every `forward_return_{h}d` is recomputed from the artifact's adjusted closes. Labels produced by different code (e.g. off by one), edited after the build, present in a symbol's final h rows, or disagreeing with their target fail the build. A separate check warns when a symbol gap makes an "h-day" label span more than h benchmark sessions. Recomputation alone cannot detect that, because it repeats the same per-symbol shift.
- **Realized-date embargo.** For each training row, the session on which its label becomes known is the symbol's own session h rows later. The training gate requires that every train label is realized before validation starts and every validation label before test starts. This is the exact leakage condition, and it stays correct when symbols have gaps.
- **Walk-forward guard.** Every monthly refit in a backtest checks that its latest realized label precedes the first scored session, or the backtest aborts. The guard also exposed a pre-existing design cost: each refit leaves its newest ~20% of history unused (the split's test partition plus embargoes). In the demo, the newest label behind a scored month was at least 91 sessions old across 21 refits. This is reported as a staleness limitation, not as a safety margin.

The correlation screen (|corr| ≥ 0.98 between a feature and a same-row forward return) produced a false positive on an existing deterministic test fixture whose prices are genuinely predictable. It was downgraded to a warning, and the recomputation test remains the proof.

### 3. Lineage without a second source of truth

Lineage pointers go into the existing `metadata_json` columns, following the repository's schema-light convention:

- Datasets record a canonical SHA-256 fingerprint of every bar read, per-symbol × source × ingestion-run counts, the run ids, the feature and label definitions, the git commit and dirty flag, and library versions.
- Models record pipeline steps and scalar hyperparameters, split method and session counts, embargo, calibration, the champion rule, gate report ids, and the dataset hash at training time.
- Backtests and SHAP runs record the hashes of their inputs.

Only one table was added. `quality_reports` exists because a blocked build has no dataset row to attach evidence to. Reports are append-only.

Dataset, model, backtest, and SHAP artifacts are now **content-addressed**: they are written to a temporary file, hashed, and then atomically renamed to a path containing the hash prefix. This fixes the overwrite defect at its source.

Historical rows are never back-filled. For example, `daily_bars.ingestion_run_id` is last-writer-wins, so it is not trusted retroactively. Missing evidence renders as "Unavailable".

### 4. Verify and reproduce

- **Live verification** runs on every console view and in `verify-model`. It re-hashes all artifacts, reopens the model bundle and compares its dataset, horizon, family, target, and features with the registry, re-fingerprints the current source bars to detect re-ingestion drift, recomputes the embargo from the dataset artifact, and checks gate reports and provenance. It is read-only.
- **Reproduction** (`reproduce-model`) rebuilds the split from the immutable dataset artifact, checks that current code produces the recorded hyperparameters, refits, and compares test predictions and metrics. The result is persisted. In the demo, the refit matched the registered 5d champion exactly: max |Δp| = 0 on 905 test rows, and identical ROC-AUC, PR-AUC, Brier score, and calibration error.

Building reproduction surfaced a real portability bug. `SimpleImputer(missing_values=nan)` serialized a bare `NaN` into JSON. PostgreSQL rejects that, and `NaN != NaN` made the hyperparameter comparison fail. Non-finite parameters are now stored as strings.

### 5. Make the evidence readable

A server-rendered FastAPI and Jinja2 console was chosen over a SPA. It keeps one language and one deployment, it is testable with `TestClient`, and it allows a strict CSP (`default-src 'none'`, no scripts). The model page shows:

- an eight-hop lineage stepper (source → ingestion → dataset → training gate → model → evaluation → backtest → explanation) with evidence-backed statuses;
- live verification, with passing checks collapsed so failures and warnings stand out;
- the dataset manifest and ingestion runs;
- a split timeline;
- a per-split metrics table with Hanley–McNeil AUC intervals, a reliability diagram with a table view, and a candidate comparison;
- backtest assumptions with the walk-forward guard;
- SHAP importance;
- reproduction evidence.

Limitations are **derived** from evidence: synthetic data, an AUC interval at or below chance, small test samples, overlapping labels, identity calibration, gate warnings, and missing provenance. The chart palette was run through a CVD and contrast validator in light and dark modes, and status chips always pair color with an icon and a label.

## Demonstration

`quant-almanac-demo run` migrates a fresh SQLite database and drives the real CLI on seeded synthetic data. It asserts each expected outcome, including the expected rejection:

| Step | Outcome |
| --- | --- |
| Ingest 5 synthetic tickers + `BADQ` (2021-01-04 → 2024-06-28) | 6,283 bars |
| Build clean dataset | 4,550 rows; 23 checks pass |
| Build with `BADQ` | **Blocked**: `bars.positive_prices`, `bars.ohlc_consistency`, `bars.calendar_alignment` (exactly the three injected defects); no artifact written |
| Train 1d/5d/20d | 6 models, 3 training-gate reports pass |
| Backtest + SHAP (5d champion) | Walk-forward guard pass; content-addressed artifacts |
| Reproduce | 7/7 checks pass, max \|Δp\| = 0 |
| Verify | 14 checks, 0 failures |

The demo is honest about results. On a random walk, the validation-selected 5d champion scores a test ROC-AUC of 0.401 (approx. 95% CI 0.363–0.439). The console flags this as below chance and marks the evaluation stage as a warning instead of a pass.

## Independent review

After implementation, a separate reviewer audited the change without seeing my conclusions. It reported 13 findings. The substantiated ones were all fixed, each with a regression test:

- **A single zero-volume session blocked the whole build.** `bars.volume` warned as designed, but `pct_change` of volume produced `inf`, which `dataset.non_finite_features` then failed. Undefined volume ratios now become missing values, which the model pipelines impute. Those rows could never have been trained on before, so no trainable dataset changes.
- **Lineage stepper statuses ignored live checks.** A tampered dataset artifact still showed a green Dataset hop. Every hop now combines its build-time verdict with the live checks that cover it.
- **`label_alignment` overstated its guarantee across symbol gaps.** Its claim was narrowed, and `dataset.label_calendar_span` was added.
- **The benchmark's own calendar was never validated.** `bars.benchmark_calendar` now checks it.
- **The feature warmup was set to 21 rows, but the longest lookback is 50.** Fixed, and the `ohlcv_v1` placeholder quirk is surfaced as a limitation.
- **The git dirty flag ignored untracked modules and was cached per process.** Fixed, and git is trusted only when the package sits in the reported work tree.
- **The console returned 500 on partially recorded models.** The templates are now defensive.
- **Backtest artifacts were not content-addressed.** They now are.
- **A gate failure on a later horizon could leave earlier horizons' models persisted.** All gates now run first.
- **The blocked-build count covered only the latest 50 reports.** It is now a real count.
- **The CI demo ran only on SQLite.** It now runs on PostgreSQL.

## Testing

- **Unit:** every bar, dataset, and training check on valid inputs and on a targeted defect, including a leaky feature caught by point-in-time recomputation, an off-by-one label, a too-short embargo, and per-symbol label realization with gaps. Also covered: AUC interval math, artifact integrity states, synthetic provider determinism and defects, and chart geometry.
- **Integration (SQLite):** a zero-volume halt warns without blocking; stepper statuses turn warn or fail after source drift or artifact tampering; all horizons are gated before any fit; backtest artifacts are content-addressed; a passing build links its report and lineage; a rebuild never overwrites an older artifact; a defective build is blocked and leaves evidence; training refuses a tampered artifact or a failed dataset report; a legacy dataset trains with a recorded warning and renders as unavailable; verification passes and then detects source drift, model tampering, and a bundle/registry mismatch; reproduction passes and detects changed hyperparameters; the backtest guard and SHAP non-overwrite hold.
- **API, console, CLI:** 200 and 404 paths, 422 validation, CSP headers, HTML escaping of persisted text, and non-zero exit codes on failing verdicts.
- **Migrations:** upgrade, downgrade, and upgrade in tests (SQLite) and by hand on PostgreSQL 16, plus the full governed flow on PostgreSQL.
- **Browser (Playwright/Chromium):** index → model → expand collapsed evidence → blocked report, with no console errors and no horizontal scroll at 390 px.

## Tradeoffs and what's next

- Content addressing costs disk on repeated builds, which is worth it for immutable manifests.
- Live verification re-hashes on every view. That is fine at research scale and would move behind a cache at larger scale.
- The console is unauthenticated. It should get auth before any shared deployment.
- Next, only where statistically defensible: drift reports between dataset versions, and persisting walk-forward predictions so backtest metrics can be bootstrapped.
