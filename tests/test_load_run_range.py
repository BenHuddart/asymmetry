"""RED target for branch ``fix/load-series-ergonomics`` (Phase 2, finding #6).

Round-2 GUI testing: the native Open dialog's File-name field truncates a long
quoted list (~15 names / ~256 chars), so loading a contiguous run series (e.g.
BiSCCO 1276–1289, ALC-TCNQ 19489–19519) had to be split into batches by hand.

Desired behaviour: a "load run range" path that takes a folder + first/last run
number (and the run prefix) and resolves the contiguous set of existing files,
bypassing the dialog's length limit. Core resolution should be a pure, testable
helper (GUI-agnostic); the GUI adds a "Load run range…" entry that calls it.

This is design-led (helper name/location + the GUI entry are the implementer's
call). The test pins the *behaviour* of the core resolver. xfail(strict) until it
exists; corpus-conditional so CI stays green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _corpus_root() -> Path | None:
    for cand in (
        os.environ.get("WIMDA_CORPUS_ROOT"),
        r"C:\Users\benhu\Source\wimda-corpus",
        str(Path.home() / "Documents" / "WiMDA muon school"),
    ):
        if cand and Path(cand).exists():
            return Path(cand)
    return None


_BISCCO = None
_root = _corpus_root()
if _root is not None:
    hits = sorted((_root / "Superconductivity").rglob("MUSR00001276.nxs"))
    hits = [h for h in hits if "hdf5" in str(h).lower()]
    if hits:
        _BISCCO = hits[0].parent  # the Data_hdf5 folder


@pytest.mark.skipif(_BISCCO is None, reason="WiMDA corpus BiSCCO runs not present")
def test_resolve_run_range_expands_contiguous_series() -> None:
    from asymmetry.core.io import resolve_run_range

    files = resolve_run_range(_BISCCO, 1276, 1289, prefix="MUSR")
    # 1276–1289 inclusive = 14 contiguous runs, sorted by run number, all existing.
    assert len(files) == 14
    assert all(Path(f).exists() for f in files)
    assert Path(files[0]).name.endswith("1276.nxs")
    assert Path(files[-1]).name.endswith("1289.nxs")


# ── corpus-free unit coverage (runs on CI) ───────────────────────────────────


def _touch(folder: Path, name: str) -> Path:
    path = folder / name
    path.write_bytes(b"")
    return path


def test_resolve_run_range_is_inclusive_sorted_and_padding_agnostic(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    for run in (8, 9, 10, 11, 12):
        _touch(tmp_path, f"MUSR{run:08d}.nxs")

    files = resolve_run_range(tmp_path, 9, 11, prefix="MUSR")

    assert [Path(f).name for f in files] == [
        "MUSR00000009.nxs",
        "MUSR00000010.nxs",
        "MUSR00000011.nxs",
    ]


def test_resolve_run_range_skips_gaps_silently(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    for run in (100, 102, 105):  # 101, 103, 104 missing
        _touch(tmp_path, f"MUSR{run:08d}.nxs")

    files = resolve_run_range(tmp_path, 100, 105, prefix="MUSR")

    assert [Path(f).name for f in files] == [
        "MUSR00000100.nxs",
        "MUSR00000102.nxs",
        "MUSR00000105.nxs",
    ]


def test_resolve_run_range_auto_detects_single_prefix(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    for run in (5, 6, 7):
        _touch(tmp_path, f"EMU{run:08d}.nxs")

    files = resolve_run_range(tmp_path, 5, 7)  # no prefix given

    assert len(files) == 3
    assert all(Path(f).name.startswith("EMU") for f in files)


def test_resolve_run_range_ignores_nonloader_extensions(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    _touch(tmp_path, "MUSR00000020.nxs")
    _touch(tmp_path, "MUSR00000021.nxs")
    _touch(tmp_path, "MUSR00000020.txt")  # sidecar log — must be ignored
    _touch(tmp_path, "MUSR00000022.log")

    files = resolve_run_range(tmp_path, 20, 22, prefix="MUSR")

    assert [Path(f).suffix for f in files] == [".nxs", ".nxs"]


def test_resolve_run_range_can_restrict_to_one_extension(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    _touch(tmp_path, "MUSR00000030.nxs")
    _touch(tmp_path, "MUSR00000031.bin")

    files = resolve_run_range(tmp_path, 30, 31, prefix="MUSR", ext="nxs")

    assert [Path(f).name for f in files] == ["MUSR00000030.nxs"]


def test_resolve_run_range_raises_on_mixed_prefixes_without_prefix(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    _touch(tmp_path, "MUSR00000040.nxs")
    _touch(tmp_path, "EMU00000041.nxs")

    with pytest.raises(ValueError, match="Multiple run prefixes"):
        resolve_run_range(tmp_path, 40, 41)


def test_resolve_run_range_empty_when_nothing_in_range(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    _touch(tmp_path, "MUSR00000050.nxs")

    assert resolve_run_range(tmp_path, 60, 70, prefix="MUSR") == []


def test_resolve_run_range_raises_on_bad_inputs(tmp_path: Path) -> None:
    from asymmetry.core.io import resolve_run_range

    with pytest.raises(ValueError, match="does not exist or is not a directory"):
        resolve_run_range(tmp_path / "nope", 1, 5)

    with pytest.raises(ValueError, match="is after end"):
        resolve_run_range(tmp_path, 9, 1, prefix="MUSR")
