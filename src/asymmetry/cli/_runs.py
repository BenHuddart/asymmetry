"""Turning the command line's run vocabulary into files on disk.

``--runs 17294-17296,17300`` is CLI syntax, so it is parsed here rather than
in the core façade; resolving a run number to a file is
:func:`asymmetry.core.io.scan_run_files`'s job and this module only indexes
its result.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from asymmetry.cli._output import UserError

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.workflow.workdir import WorkDir


def parse_run_spec(spec: str) -> list[int]:
    """Expand ``"A-B,C,D-E"`` into an ascending, duplicate-free list of run numbers.

    Whitespace around the separators is allowed. Raises :class:`UserError`
    naming the offending token for anything else — an empty spec, a
    non-numeric token, or a range whose start is above its end.
    """
    runs: set[int] = set()
    tokens = [token.strip() for token in str(spec).split(",")]
    if not any(tokens):
        raise UserError(f"Run spec {spec!r} is empty; expected something like '100-104,110'.")
    for token in tokens:
        if not token:
            raise UserError(f"Run spec {spec!r} has an empty entry between commas.")
        if "-" in token:
            first_text, _, last_text = token.partition("-")
            first, last = _run_number(first_text, spec), _run_number(last_text, spec)
            if first > last:
                raise UserError(f"Run range {token!r} starts at {first}, after its end {last}.")
            runs.update(range(first, last + 1))
        else:
            runs.add(_run_number(token, spec))
    return sorted(runs)


def _run_number(text: str, spec: str) -> int:
    stripped = text.strip()
    if not stripped.isdigit():
        raise UserError(f"Run spec {spec!r} contains {stripped!r}, which is not a run number.")
    return int(stripped)


def run_files(folder: str | Path) -> dict[int, tuple[str, Path]]:
    """Map every run number in *folder* to its ``(prefix, path)``.

    Raises :class:`UserError` when *folder* is not a directory, so a mistyped
    path is a one-line message rather than a traceback.
    """
    from asymmetry.core.io import scan_run_files

    folder = Path(folder)
    if not folder.is_dir():
        raise UserError(f"{folder} does not exist or is not a directory.")
    found = scan_run_files(folder)
    return {run_number: (prefix, path) for prefix, run_number, path in found.entries}


def resolve_runs(folder: str | Path, spec: str) -> list[tuple[int, str, Path]]:
    """The ``(run_number, prefix, path)`` triples in *folder* that *spec* names.

    Run numbers the spec names but the folder does not hold are skipped —
    a scan with gaps is normal. Raises :class:`UserError` only when *no*
    named run exists, naming the range the folder does hold.
    """
    available = run_files(folder)
    wanted = parse_run_spec(spec)
    resolved = [(run, available[run][0], available[run][1]) for run in wanted if run in available]
    if not resolved:
        raise UserError(
            f"No run files in {folder} match {spec!r} "
            f"(the folder holds {_range_text(sorted(available))})."
        )
    return resolved


def resolve_run(folder: str | Path, run_number: int) -> Path:
    """The file for one run in *folder*; :class:`UserError` when it is not there."""
    available = run_files(folder)
    if run_number not in available:
        raise UserError(
            f"Run {run_number} is not in {folder} (the folder holds "
            f"{_range_text(sorted(available))})."
        )
    return available[run_number][1]


def reduced_datasets(workdir: WorkDir, runs: list[int]) -> dict[int, MuonDataset]:
    """The stored reduced spectra for *runs*, keyed by run number.

    The screening and fitting commands read their data from the work directory
    rather than the raw files, so every one of them agrees on the reduction
    that produced it. Raises :class:`UserError` naming the runs that have not
    been reduced, because that is a step the user has to run first.
    """
    stored = set(workdir.reduced_runs())
    missing = [run for run in runs if run not in stored]
    if missing:
        raise UserError(
            f"Run(s) {', '.join(str(run) for run in missing)} have not been reduced into "
            f"{workdir.root}; run 'asymmetry reduce' on them first."
        )
    return {run: workdir.reduced(run) for run in runs}


def _range_text(runs: list[int]) -> str:
    if not runs:
        return "no runs"
    if len(runs) == 1:
        return f"run {runs[0]}"
    return f"runs {runs[0]}-{runs[-1]}"


__all__ = [
    "parse_run_spec",
    "reduced_datasets",
    "resolve_run",
    "resolve_runs",
    "run_files",
]
