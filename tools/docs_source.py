"""Identify the imported source checkout for published documentation."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceIdentity:
    """A display label and a revision whose source lines can be linked safely."""

    label: str
    revision: str | None


def source_identity(package_file: str | Path, release: str) -> SourceIdentity:
    """Use the imported package's clean Git checkout, never an inferred release."""
    package_file = Path(package_file).resolve()
    root = package_file.parents[2]
    if package_file.parent != root / "src/asymmetry" or shutil.which("git") is None:
        return SourceIdentity(f"{release} (source revision unavailable)", None)
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return SourceIdentity(f"{release} (source revision unavailable)", None)
    repository, revision = result.stdout.splitlines()
    if Path(repository).resolve() != root:
        return SourceIdentity(f"{release} (source revision unavailable)", None)
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout:
        return SourceIdentity(f"{release} (modified checkout {revision[:12]})", None)
    tag = subprocess.run(
        ["git", "-C", str(root), "tag", "--points-at", revision],
        capture_output=True,
        text=True,
        check=True,
    )
    label = (
        release
        if f"v{release}" in tag.stdout.splitlines()
        else f"{release} (development {revision[:12]})"
    )
    return SourceIdentity(label, revision)
