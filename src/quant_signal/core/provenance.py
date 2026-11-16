"""Runtime and code provenance captured alongside every governed artifact."""

from __future__ import annotations

import os
import platform
import subprocess
from functools import cache
from importlib import metadata
from pathlib import Path

TRACKED_PACKAGES = ("numpy", "pandas", "pyarrow", "scikit-learn", "shap", "joblib")
CODE_REVISION_ENV = "QUANT_ALMANAC_CODE_REVISION"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def code_revision() -> dict[str, object]:
    """Return the source revision, or an explicit ``unavailable`` marker.

    ``QUANT_ALMANAC_CODE_REVISION`` takes precedence so packaged deployments without a
    ``.git`` directory can still record a revision. Git is only trusted when this
    package lives at ``<repo>/src/quant_signal`` of the reported work tree, so an
    unrelated enclosing repository is never recorded. The ``dirty`` flag counts
    modified *and* untracked (non-ignored) files, because a new, untracked module can
    change behavior. The revision is read on every call (not cached) so long-running
    processes record the code that is actually on disk when an artifact is produced.
    """

    override = os.getenv(CODE_REVISION_ENV)
    if override:
        return {"commit": override.strip(), "dirty": None, "source": "environment"}

    package_dir = Path(__file__).resolve().parents[1]
    toplevel = _git(["rev-parse", "--show-toplevel"], package_dir)
    if not toplevel or (Path(toplevel) / "src" / "quant_signal").resolve() != package_dir:
        return {"commit": None, "dirty": None, "source": "unavailable"}
    commit = _git(["rev-parse", "HEAD"], package_dir)
    if not commit:
        return {"commit": None, "dirty": None, "source": "unavailable"}
    status = _git(["status", "--porcelain"], package_dir)
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "source": "git",
    }


@cache
def runtime_environment() -> dict[str, object]:
    """Return the interpreter and key library versions used to produce artifacts."""

    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.system().lower(),
        "packages": packages,
    }


def provenance_snapshot() -> dict[str, object]:
    """Return the combined code and runtime provenance block."""

    return {"code": code_revision(), "runtime": runtime_environment()}
