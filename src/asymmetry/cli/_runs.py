"""Turning the command line's run vocabulary into files on disk.

``--runs 17294-17296,17300`` is CLI syntax, so it is parsed here rather than
in the core façade; resolving a run number to a file is
:func:`asymmetry.core.io.scan_run_files`'s job and this module only indexes
its result, narrowed to the one instrument a
:class:`~asymmetry.core.workflow.workdir.RunSelection` names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from asymmetry.cli._output import UserError

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.workflow.workdir import RunSelection, WorkDir


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


def _instrument(text: str) -> str:
    """``--instrument`` as the instrument name the core compares prefixes against."""
    from asymmetry.core.workflow.workdir import instrument_name

    return instrument_name(text)


def add_instrument_argument(parser: argparse.ArgumentParser) -> None:
    """Declare ``--instrument``, the file prefix a folder of two instruments needs."""
    parser.add_argument(
        "--instrument",
        type=_instrument,
        default=None,
        metavar="NAME",
        help=(
            "Use only this instrument's runs — the file prefix, in any case (EMU selects "
            "EMU… and emu…); needed where two instruments in the folder share run numbers"
        ),
    )


def run_clashes(entries: list[tuple[str, int, Path]]) -> dict[int, list[Path]]:
    """The run numbers naming more than one file among *entries*, with those files."""
    files: dict[int, list[Path]] = {}
    for _prefix, run_number, path in entries:
        files.setdefault(run_number, []).append(path)
    return {run: paths for run, paths in files.items() if len(paths) > 1}


def run_files(selection: RunSelection) -> dict[int, tuple[str, Path]]:
    """Map every selected run number to its ``(prefix, path)``.

    This is the one place the instrument filter is applied; every resolver
    goes through it. Raises :class:`UserError` when the folder is not a
    directory, when the instrument names no file in it, and when two files
    carry one run number — the work directory is keyed on the run number
    alone, so the message says whether ``--instrument`` separates them or the
    folder has to be split.
    """
    from asymmetry.core.workflow.workdir import describe_instruments, instrument_name

    if not selection.folder.is_dir():
        raise UserError(f"{selection.folder} does not exist or is not a directory.")
    try:
        found = selection.scan()
    except ValueError as exc:
        raise UserError(str(exc)) from None
    clashes = run_clashes(found.entries)
    if clashes:
        run_number = min(clashes)
        names = " and ".join(sorted(path.name for path in clashes[run_number]))
        more = f" (and {len(clashes) - 1} more run numbers)" if len(clashes) > 1 else ""
        clashing = sorted(
            {instrument_name(prefix) for prefix, run, _path in found.entries if run in clashes}
        )
        if len(clashing) > 1:
            raise UserError(
                f"Run numbers in {selection.folder} collide between instruments "
                f"{' and '.join(clashing)}: run {run_number} is {names}{more}. The work "
                "directory is keyed on the run number alone, so name the instrument: "
                f"--instrument {' or --instrument '.join(clashing)} (the file prefix, in any "
                f"case). The folder holds {describe_instruments(found.entries)}."
            )
        raise UserError(
            f"Run {run_number} is in {selection} twice: {names}{more}. Both are instrument "
            f"{clashing[0]}, so --instrument cannot tell them apart; move one of them out "
            "of the folder."
        )
    return {run_number: (prefix, path) for prefix, run_number, path in found.entries}


def resolve_runs(selection: RunSelection, spec: str) -> list[tuple[int, str, Path]]:
    """The ``(run_number, prefix, path)`` triples among *selection* that *spec* names.

    Run numbers the spec names but the folder does not hold are skipped —
    a scan with gaps is normal. Raises :class:`UserError` only when *no*
    named run exists, naming the range the folder does hold.
    """
    available = run_files(selection)
    wanted = parse_run_spec(spec)
    resolved = [(run, available[run][0], available[run][1]) for run in wanted if run in available]
    if not resolved:
        raise UserError(
            f"No run files in {selection} match {spec!r} "
            f"(the folder holds {range_text(sorted(available))})."
        )
    return resolved


def resolve_run(selection: RunSelection, run_number: int) -> Path:
    """The file for one selected run; :class:`UserError` when it is not there."""
    available = run_files(selection)
    if run_number not in available:
        raise UserError(
            f"Run {run_number} is not in {selection} (the folder holds "
            f"{range_text(sorted(available))})."
        )
    return available[run_number][1]


def reduced_datasets(workdir: WorkDir, runs: list[int]) -> dict[int, MuonDataset]:
    """The stored reduced spectra for *runs*, keyed by run number.

    The screening and fitting commands read their data from the work directory
    rather than the raw files, so every one of them agrees on the reduction
    that produced it — and each of them names a co-add's members, here, on
    stderr. Raises :class:`UserError` naming the runs that have not
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
        datasets = {run: workdir.reduced(run) for run in runs}
        members = {run: workdir.entry(run).members for run in runs}
    except KeyError as exc:
        raise UserError(exc.args[0]) from None
    # stderr, so a --json payload on stdout stays one JSON document.
    for run in runs:
        if members[run]:
            print(f"asymmetry: note: {coadd_note(run, members[run])}", file=sys.stderr)
    return datasets


def coadd_note(run_number: int, members: list[int]) -> str:
    """The line naming the runs a stored co-add summed."""
    return f"Run {run_number} is co-added from {range_text(members)}."


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


def range_text(runs: list[int]) -> str:
    """``"runs 3678-3682"``, ``"runs 101, 103-105"`` or ``"run 7"`` for ascending *runs*."""
    if not runs:
        return "no runs"
    if len(runs) == 1:
        return f"run {runs[0]}"
    spans: list[list[int]] = []
    for run in runs:
        if spans and run == spans[-1][-1] + 1:
            spans[-1].append(run)
        else:
            spans.append([run])
    return "runs " + ", ".join(
        str(span[0]) if len(span) == 1 else f"{span[0]}-{span[-1]}" for span in spans
    )


__all__ = [
    "add_instrument_argument",
    "coadd_note",
    "range_text",
    "run_clashes",
    "window_note",
    "parse_run_spec",
    "reduced_datasets",
    "resolve_run",
    "resolve_runs",
    "run_files",
]
