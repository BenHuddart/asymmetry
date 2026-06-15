"""Resolve a contiguous run series to its files, bypassing the Open dialog.

The native Open dialog's File-name field truncates a long quoted list (~15
names / ~256 chars), so loading a contiguous run series (e.g. BiSCCO 1276–1289)
had to be split into batches by hand. :func:`resolve_run_range` expands a
folder plus an inclusive first/last run number into the sorted set of existing
files, so the GUI can load the whole series in one shot.

This is a pure, GUI-free filesystem helper: it only inspects names and existence
in one folder; it never opens or parses the data files.
"""

from __future__ import annotations

import re
from pathlib import Path

from asymmetry.core.io.base import LoaderRegistry

#: Filename = leading (non-digit-terminated) prefix + trailing run-number digits.
#: ``MUSR00001276`` → prefix ``"MUSR"``, run number ``1276`` (padding-agnostic).
_NAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")


def _parse_stem(stem: str) -> tuple[str, int] | None:
    """Split a filename stem into ``(prefix, run_number)``, or ``None``."""
    match = _NAME_RE.match(stem)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("num"))


def _allowed_extensions(ext: str | None) -> set[str]:
    """Extensions to scan: the given one, else every loader-registered one."""
    if ext is None:
        return {e.lower() for e in LoaderRegistry.supported_extensions()}
    return {f".{ext.lstrip('.').lower()}"}


def scan_run_files(
    folder: str | Path,
    *,
    ext: str | None = None,
) -> list[tuple[str, int, Path]]:
    """Parse every run file in ``folder`` into ``(prefix, run_number, path)``.

    Non-recursive; only files whose extension a loader claims (or ``ext`` when
    given) are considered, so sidecar logs are ignored. The list is sorted by
    run number. Useful for prefilling a run-range dialog (prefix + min/max run)
    from a chosen folder. Raises :class:`ValueError` if ``folder`` is not a
    directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Run-range folder does not exist or is not a directory: {folder}")
    allowed = _allowed_extensions(ext)
    found: list[tuple[str, int, Path]] = []
    for entry in folder.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in allowed:
            continue
        parsed = _parse_stem(entry.stem)
        if parsed is None:
            continue
        name_prefix, run_number = parsed
        found.append((name_prefix, run_number, entry))
    found.sort(key=lambda item: item[1])
    return found


def resolve_run_range(
    folder: str | Path,
    first: int,
    last: int,
    *,
    prefix: str | None = None,
    ext: str | None = None,
) -> list[Path]:
    """Expand ``folder`` + inclusive ``[first, last]`` run range to sorted files.

    Scans ``folder`` (non-recursively) for files whose extension a loader
    claims — so sidecar logs (``.txt``/``.mon``) are ignored — parses the
    trailing digit group of each name as its run number, and returns the
    existing files whose run number falls in ``[first, last]``, sorted by run
    number. The match is padding-agnostic: ``MUSR00001276`` resolves to run
    ``1276`` regardless of zero-pad width.

    Parameters
    ----------
    folder : str or Path
        Directory holding the run files. Must exist and be a directory.
    first, last : int
        Inclusive run-number bounds. ``first`` must not exceed ``last``.
    prefix : str, optional
        Instrument/file prefix (e.g. ``"MUSR"``), matched case-insensitively
        against the leading text of each name. When *None*, the prefix is
        auto-detected from the files that fall in the range: exactly one
        prefix must be present, else a :class:`ValueError` lists the
        candidates so the caller can disambiguate.
    ext : str, optional
        Restrict the scan to a single extension (e.g. ``"nxs"`` or ``".nxs"``).
        When *None*, every loader-registered extension is considered.

    Returns
    -------
    list[pathlib.Path]
        Existing files in ``[first, last]``, sorted ascending by run number.
        **Gaps are skipped silently** — missing run numbers are simply absent,
        so a non-contiguous corpus yields the runs that do exist. An empty list
        means the folder held no matching runs in range (not an error).

    Raises
    ------
    ValueError
        If ``folder`` does not exist or is not a directory, if ``first > last``,
        or if ``prefix`` is *None* and the in-range files carry more than one
        distinct prefix.
    """
    if first > last:
        raise ValueError(f"Run-range start {first} is after end {last}")

    wanted_prefix = prefix.lower() if prefix is not None else None

    candidates = [
        (name_prefix, run_number, path)
        for name_prefix, run_number, path in scan_run_files(folder, ext=ext)
        if first <= run_number <= last
        and (wanted_prefix is None or name_prefix.lower() == wanted_prefix)
    ]

    if wanted_prefix is None:
        distinct = sorted({name_prefix for name_prefix, _, _ in candidates})
        if len(distinct) > 1:
            shown = ", ".join(repr(p) for p in distinct)
            raise ValueError(
                f"Multiple run prefixes found in range [{first}, {last}]: {shown}. "
                "Pass prefix= to choose one."
            )

    return [path for _, _, path in candidates]
