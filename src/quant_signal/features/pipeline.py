"""Feature dataset materialization pipeline with a blocking data-quality gate."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_signal.core.config import Settings, get_settings
from quant_signal.core.hashing import sha256_file, sha256_frame
from quant_signal.core.provenance import provenance_snapshot
from quant_signal.features.engineering import FEATURE_COLUMNS, build_feature_frame
from quant_signal.features.labels import (
    add_forward_return_targets,
    forward_return_columns,
    target_columns,
)
from quant_signal.quality import (
    STAGE_DATASET_BUILD,
    DataQualityGateError,
    DataQualityPolicy,
    QualityReport,
)
from quant_signal.quality.bars import run_bar_checks
from quant_signal.quality.dataset import check_point_in_time_features, run_dataset_checks
from quant_signal.quality.persistence import persist_quality_report
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import DatasetVersion
from quant_signal.storage.repositories import (
    BarProvenanceRow,
    DatasetArtifactRecord,
    StorageRepository,
)

logger = logging.getLogger(__name__)

SOURCE_BAR_FINGERPRINT_COLUMNS = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "source",
)


def fingerprint_source_bars(bars: pd.DataFrame) -> str:
    """Return the canonical SHA-256 fingerprint of the bars feeding a dataset build."""

    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return sha256_frame(
        frame,
        SOURCE_BAR_FINGERPRINT_COLUMNS,
        sort_by=("symbol", "date", "source"),
    )


class FeaturePipeline:
    """Build versioned, model-ready datasets from persisted market bars."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
        quality_policy: DataQualityPolicy | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url
        self.quality_policy = quality_policy or DataQualityPolicy()

    def build_dataset(
        self,
        as_of_date: date,
        symbols: Sequence[str] | None = None,
        feature_set_version: str = "ohlcv_v1",
    ) -> DatasetVersion:
        """Validate source bars, materialize features/labels, and register the dataset.

        Every build persists a quality report. When a blocking check fails, no artifact
        or dataset manifest is written and ``DataQualityGateError`` is raised.
        """

        benchmark_symbol = self.settings.benchmark_symbol.upper()
        requested = [symbol.upper() for symbol in (symbols or self.settings.universe_symbols)]
        modeling_symbols = sorted({symbol for symbol in requested if symbol != benchmark_symbol})
        fetch_symbols = sorted({*modeling_symbols, benchmark_symbol})
        horizons = list(self.settings.default_horizons)
        policy = self.quality_policy

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            bars = repository.load_daily_bars_frame(fetch_symbols, end_date=as_of_date)
            provenance_rows = repository.summarize_bar_provenance(
                fetch_symbols,
                end_date=as_of_date,
            )

        if bars.empty:
            raise ValueError("No persisted bars were found for the requested dataset build.")

        request_json: dict[str, object] = {
            "as_of_date": as_of_date.isoformat(),
            "requested_symbols": requested,
            "modeling_symbols": modeling_symbols,
            "benchmark_symbol": benchmark_symbol,
            "feature_set_version": feature_set_version,
            "horizons": horizons,
        }
        lineage = self._build_lineage(bars, provenance_rows, feature_set_version)

        report = QualityReport(
            stage=STAGE_DATASET_BUILD,
            checks=tuple(
                run_bar_checks(bars, modeling_symbols, benchmark_symbol, as_of_date, policy)
            ),
        )
        dataset_frame: pd.DataFrame | None = None
        if report.passed:
            feature_frame = build_feature_frame(bars, benchmark_symbol=benchmark_symbol)
            dataset_frame = feature_frame[feature_frame["symbol"].isin(modeling_symbols)].copy()
            dataset_frame = add_forward_return_targets(dataset_frame, horizons)
            dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])
            dataset_frame = dataset_frame.sort_values(["date", "symbol"]).reset_index(drop=True)
            report = report.extend(
                [
                    *run_dataset_checks(
                        dataset_frame,
                        FEATURE_COLUMNS,
                        horizons,
                        as_of_date,
                        policy,
                        calendar=list(
                            pd.to_datetime(
                                bars.loc[bars["symbol"] == benchmark_symbol, "date"]
                            ).unique()
                        ),
                    ),
                    check_point_in_time_features(
                        bars=bars,
                        full_features=feature_frame,
                        feature_columns=FEATURE_COLUMNS,
                        modeling_symbols=modeling_symbols,
                        benchmark_symbol=benchmark_symbol,
                        feature_builder=build_feature_frame,
                        policy=policy,
                    ),
                ]
            )
        else:
            report = report.extend(
                [],
                notes=[
                    "Dataset-level checks were not run: source-bar checks failed, so "
                    "features derived from those bars would not be meaningful."
                ],
            )

        if dataset_frame is None or not report.passed:
            with session_scope(self.database_url) as session:
                record = persist_quality_report(
                    StorageRepository(session),
                    report,
                    policy,
                    request_json=request_json,
                    context_json={"lineage": lineage},
                )
                report_id = record.id
            logger.warning(
                "Dataset build blocked by quality gate report=%s failed=%s",
                report_id,
                [check.check_id for check in report.failures],
            )
            raise DataQualityGateError(report, report_id)

        label_columns = target_columns(horizons) + forward_return_columns(horizons)
        artifact_path, artifact_hash = self._write_artifact(
            dataset_frame,
            as_of_date,
            feature_set_version,
        )

        metadata_json: dict[str, object] = {
            "date_range": {
                "start": str(dataset_frame["date"].min().date()),
                "end": str(dataset_frame["date"].max().date()),
            },
            "benchmark_symbol": benchmark_symbol,
            "raw_row_count": int(len(bars)),
            "lineage": lineage,
        }

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            dataset_version = repository.create_dataset_version(
                DatasetArtifactRecord(
                    as_of_date=as_of_date,
                    feature_set_version=feature_set_version,
                    horizons=horizons,
                    symbols=modeling_symbols,
                    row_count=int(len(dataset_frame)),
                    artifact_path=str(artifact_path),
                    artifact_hash=artifact_hash,
                    feature_columns=FEATURE_COLUMNS,
                    label_columns=label_columns,
                    metadata_json=metadata_json,
                )
            )
            record = persist_quality_report(
                repository,
                report,
                policy,
                request_json=request_json,
                context_json={"lineage": lineage},
                dataset_version_id=dataset_version.id,
            )
            dataset_version.metadata_json = {
                **metadata_json,
                "quality": {
                    "report_id": record.id,
                    "status": report.status,
                    "policy_version": policy.version,
                },
            }
            session.flush()
            logger.info(
                "Dataset %s registered with quality status %s (report %s)",
                dataset_version.id,
                report.status,
                record.id,
            )
            return dataset_version

    def _build_lineage(
        self,
        bars: pd.DataFrame,
        provenance_rows: Sequence[BarProvenanceRow],
        feature_set_version: str,
    ) -> dict[str, object]:
        """Describe exactly which persisted bars and code produced this dataset."""

        dates = pd.to_datetime(bars["date"])
        ingestion_run_ids = sorted(
            {row.ingestion_run_id for row in provenance_rows if row.ingestion_run_id}
        )
        return {
            "source_bars": {
                "fingerprint_sha256": fingerprint_source_bars(bars),
                "fingerprint_columns": list(SOURCE_BAR_FINGERPRINT_COLUMNS),
                "row_count": int(len(bars)),
                "symbols": sorted(str(symbol) for symbol in bars["symbol"].unique()),
                "first_date": dates.min().date().isoformat(),
                "last_date": dates.max().date().isoformat(),
                "sources": sorted({row.source for row in provenance_rows}),
                "ingestion_run_ids": ingestion_run_ids,
                "rows_without_ingestion_run": int(
                    sum(row.row_count for row in provenance_rows if not row.ingestion_run_id)
                ),
                "by_symbol": [
                    {
                        "symbol": row.symbol,
                        "source": row.source,
                        "ingestion_run_id": row.ingestion_run_id,
                        "rows": row.row_count,
                        "first_date": row.first_date.isoformat(),
                        "last_date": row.last_date.isoformat(),
                    }
                    for row in provenance_rows
                ],
            },
            "feature_set_version": feature_set_version,
            "feature_builder": "quant_signal.features.engineering.build_feature_frame",
            "label_definition": "per-symbol adjusted-close return h sessions ahead; "
            "target_up = forward_return > 0",
            **provenance_snapshot(),
        }

    def _write_artifact(
        self,
        dataset_frame: pd.DataFrame,
        as_of_date: date,
        feature_set_version: str,
    ) -> tuple[Path, str]:
        """Write the dataset to a content-addressed path so manifests never go stale.

        Earlier versions reused one path per (feature set, as-of date), so a rebuild
        silently overwrote the artifact referenced by older manifests and broke their
        recorded SHA-256. Embedding the digest in the filename makes every registered
        artifact immutable.
        """

        directory = Path(self.settings.artifact_root) / "datasets"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".dataset_{uuid4().hex}.parquet.tmp"
        dataset_frame.to_parquet(temporary, index=False)
        artifact_hash = sha256_file(temporary)
        artifact_path = self._build_artifact_path(as_of_date, feature_set_version, artifact_hash)
        os.replace(temporary, artifact_path)
        return artifact_path, artifact_hash

    def _build_artifact_path(
        self,
        as_of_date: date,
        feature_set_version: str,
        artifact_hash: str,
    ) -> Path:
        """Build a content-addressed dataset artifact path."""

        filename = f"dataset_{feature_set_version}_{as_of_date:%Y%m%d}_{artifact_hash[:12]}.parquet"
        return Path(self.settings.artifact_root) / "datasets" / filename
