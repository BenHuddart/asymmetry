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
    path is a one-line message rather than a traceback, and when two files
    carry the same run number under different prefixes — the work directory is
    keyed on the run number alone, so there is no prefix to disambiguate with
    and the folder has to be split.
    """
    from asymmetry.core.io import scan_run_files

    folder = Path(folder)
    if not folder.is_dir():
        raise UserError(f"{folder} does not exist or is not a directory.")
    found = scan_run_files(folder)
    files: dict[int, tuple[str, Path]] = {}
    for prefix, run_number, path in found.entries:
        if run_number in files:
            raise UserError(_duplicate_run_message(folder, run_number, files[run_number], path))
        files[run_number] = (prefix, path)
    return files


def _duplicate_run_message(
    folder: Path, run_number: int, first: tuple[str, Path], second: Path
) -> str:
    """The message for two files in *folder* sharing one run number."""
    names = sorted([first[1].name, second.name])
    return (
        f"Run {run_number} is in {folder} twice: {names[0]} and {names[1]}. "
        "Every command keys its work directory on the run number alone, so there is "
        "no prefix to tell the two apart; split the folder so each prefix "
        "(instrument) has a directory of its own."
    )


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
    try:
        return {run: workdir.reduced(run) for run in runs}
    except KeyError as exc:
        raise UserError(exc.args[0]) from None


def window_note(workdir: WorkDir, runs: list[int]) -> str | None:
    """A note naming runs whose stored reduction was cut to a time window, or ``None``.

    ``reduce --tmin/--tmax`` trims what every later command sees, not just the
    plot, so a window chosen to zoom on early precession silently starves the
    wizard and the fits of the rest of the record.
    """
    windowed = sorted(
        (run, workdir.entry(run).settings)
        for run in runs
        if workdir.entry(run).settings.t_min is not None
        or workdir.entry(run).settings.t_max is not None
    )
    if not windowed:
        return None
    spans = {
        f"{'start' if s.t_min is None else s.t_min}-{'end' if s.t_max is None else s.t_max} µs"
        for _, s in windowed
    }
    return (
        f"NOTE: run(s) {', '.join(str(run) for run, _ in windowed)} were reduced to a time "
        f"window ({', '.join(sorted(spans))}), so this sees only that part of the record. "
        f"Reduce again without --tmin/--tmax to use it all; --plot-tmax zooms a plot alone."
    )


def _range_text(runs: list[int]) -> str:
    if not runs:
        return "no runs"
    if len(runs) == 1:
        return f"run {runs[0]}"
    return f"runs {runs[0]}-{runs[-1]}"


__all__ = [
    "window_note",
    "parse_run_spec",
    "reduced_datasets",
    "resolve_run",
    "resolve_runs",
    "run_files",
]
