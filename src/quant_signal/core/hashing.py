"""Hashing helpers for artifacts and metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 hex digest for the given bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a SHA-256 hex digest for a file."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_json(payload: Any) -> str:
    """Return a stable SHA-256 digest for JSON-serializable content."""

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256_bytes(serialized)


def sha256_frame(frame: pd.DataFrame, columns: Sequence[str], sort_by: Sequence[str]) -> str:
    """Return a stable digest of selected DataFrame columns in a canonical row order.

    Dates are serialized as ISO strings and floats with round-trip precision, so the
    digest only changes when the underlying values change.
    """

    canonical = frame.loc[:, list(columns)].copy()
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].dt.strftime("%Y-%m-%d")
        elif canonical[column].dtype == object:
            canonical[column] = canonical[column].map(
                lambda value: value.isoformat() if hasattr(value, "isoformat") else value
            )
    canonical = canonical.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    return sha256_bytes(canonical.to_csv(index=False, lineterminator="\n").encode("utf-8"))
