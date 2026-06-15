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
@pytest.mark.xfail(reason="fix/load-series-ergonomics not yet implemented", strict=True)
def test_resolve_run_range_expands_contiguous_series() -> None:
    # Implementer: provide a pure resolver (adjust import/signature to taste).
    from asymmetry.core.io import resolve_run_range  # type: ignore[attr-defined]

    files = resolve_run_range(_BISCCO, 1276, 1289, prefix="MUSR")
    # 1276–1289 inclusive = 14 contiguous runs, sorted by run number, all existing.
    assert len(files) == 14
    assert all(Path(f).exists() for f in files)
    assert Path(files[0]).name.endswith("1276.nxs")
    assert Path(files[-1]).name.endswith("1289.nxs")
