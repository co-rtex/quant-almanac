"""Artifact integrity checks shared by every pipeline stage."""

from __future__ import annotations

import os
from pathlib import Path

from quant_signal.core.hashing import sha256_file
from quant_signal.quality.report import CheckResult


class ArtifactIntegrityError(RuntimeError):
    """Raised when a persisted artifact is missing or no longer matches its registry hash."""

    def __init__(self, result: CheckResult) -> None:
        super().__init__(result.message)
        self.result = result


def check_artifact_integrity(
    artifact_path: str | os.PathLike[str] | None,
    expected_hash: str | None,
    artifact_kind: str,
    check_id: str | None = None,
) -> CheckResult:
    """Compare an artifact's SHA-256 digest with the hash recorded in the registry."""

    resolved_id = check_id or f"artifact.{artifact_kind}_integrity"
    observed: dict[str, object] = {
        "artifact_path": str(artifact_path) if artifact_path else None,
        "expected_sha256": expected_hash,
    }
    if not artifact_path or not expected_hash:
        return CheckResult(
            check_id=resolved_id,
            category="artifact",
            status="warn",
            message=f"No {artifact_kind} artifact hash was recorded; integrity is unverifiable.",
            observed=observed,
        )
    path = Path(artifact_path)
    if not path.is_file():
        return CheckResult(
            check_id=resolved_id,
            category="artifact",
            status="fail",
            message=f"The {artifact_kind} artifact is missing from artifact storage.",
            observed=observed,
        )
    actual = sha256_file(path)
    observed["actual_sha256"] = actual
    matches = actual == expected_hash
    return CheckResult(
        check_id=resolved_id,
        category="artifact",
        status="pass" if matches else "fail",
        message=(
            f"The {artifact_kind} artifact matches its registered SHA-256."
            if matches
            else f"The {artifact_kind} artifact was modified after registration (SHA-256 mismatch)."
        ),
        observed=observed,
    )


def require_artifact_integrity(
    artifact_path: str | os.PathLike[str] | None,
    expected_hash: str | None,
    artifact_kind: str,
) -> CheckResult:
    """Return the passing integrity result or raise ``ArtifactIntegrityError``."""

    result = check_artifact_integrity(artifact_path, expected_hash, artifact_kind)
    if result.status == "fail":
        raise ArtifactIntegrityError(result)
    return result
