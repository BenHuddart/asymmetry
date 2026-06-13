"""GUI-level tests for Data Browser run arithmetic (co-add + reference subtract).

Covers the histogram-backed combined row, the reference-subtraction action,
"Separate Combined" restoration, and sign-aware rebuild. Numerical correctness
of the kernel itself lives in tests/test_combine.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.panels.data_browser import DataBrowserPanel

pytestmark = pytest.mark.usefixtures("qapp")


def _dataset(
    rn: int, *, frames: float = 1000.0, expected: float = 300.0, seed: int = 0
) -> MuonDataset:
    rng = np.random.default_rng(seed)
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 12,
        "last_good_bin": 90,
        "good_frames": frames,
        "t0_bin": 10,
    }
    hs = [
        Histogram(
            counts=rng.poisson(np.full(100, expected * (1.0 if d == 0 else 0.97))).astype(float),
            bin_width=0.016,
            t0_bin=10,
            good_bin_start=10,
            good_bin_end=90,
        )
        for d in range(2)
    ]
    run = Run(
        run_number=rn,
        histograms=hs,
        metadata={"run_number": rn, "title": "S", "temperature": 5.0, "field": 100.0},
        grouping=grouping,
        source_file=f"/tmp/run_{rn}.nxs",
    )
    return MuonDataset(
        time=np.arange(100.0),
        asymmetry=np.zeros(100),
        error=np.ones(100),
        metadata=dict(run.metadata),
        run=run,
    )


def test_coadd_row_is_histogram_backed():
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(401, frames=1000, seed=1))
    panel.add_dataset(_dataset(402, frames=1000, seed=2))
    crn = panel.add_combined_dataset([401, 402], sign=1)
    assert crn is not None
    combined = panel.get_dataset(crn)
    # The correctness fix: a combined row now carries real summed histograms.
    assert combined.run is not None and combined.run.histograms
    assert combined.run.grouping["good_frames"] == pytest.approx(2000.0)
    assert panel._combined_run_display(crn) == "401 + 402"


def test_subtract_reference_row():
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(301, frames=1000, seed=1))
    panel.add_dataset(_dataset(302, frames=2000, seed=2))
    crn = panel.add_combined_dataset([301, 302], sign=-1)
    assert crn is not None
    assert panel._combined_signs[crn] == -1
    combined = panel.get_dataset(crn)
    assert combined.run is not None and combined.run.histograms
    assert combined.metadata["combination"]["method"] == "subtract_reference"
    # Reference scaled by sample/reference good frames = 1000/2000.
    assert combined.metadata["combination"]["reference_scale"] == pytest.approx(0.5)
    assert panel._combined_run_display(crn) == "301 − 302"


def test_subtract_action_uses_picker(monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(301, frames=1000, seed=1))
    panel.add_dataset(_dataset(302, frames=1000, seed=2))
    # Drive the picker to pick run 302 without showing a dialog.
    monkeypatch.setattr(panel, "_prompt_reference_run", lambda sample, cands: 302)
    panel._subtract_reference_run(301)
    combined = next(rn for rn in panel._combined_datasets if panel._combined_signs.get(rn) == -1)
    assert panel._combined_datasets[combined] == [301, 302]
    assert set(panel._get_selected_run_numbers()) == {combined}
    # Sources are hidden under the combined row.
    assert 301 not in panel._datasets
    assert 302 not in panel._datasets


def test_subtract_action_no_candidates_is_safe(monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(301, frames=1000, seed=1))
    # No second run -> the action informs and does nothing.
    monkeypatch.setattr(
        "asymmetry.gui.panels.data_browser.QMessageBox.information", lambda *a, **k: None
    )
    panel._subtract_reference_run(301)
    assert not panel._combined_datasets


def test_separate_restores_subtraction_sources(monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(301, frames=1000, seed=1))
    panel.add_dataset(_dataset(302, frames=1000, seed=2))
    monkeypatch.setattr(panel, "_prompt_reference_run", lambda sample, cands: 302)
    panel._subtract_reference_run(301)
    combined = next(iter(panel._combined_datasets))
    panel._separate_combined()
    assert 301 in panel._datasets
    assert 302 in panel._datasets
    assert set(panel._get_selected_run_numbers()) == {301, 302}
    assert combined not in panel._combined_datasets
    assert combined not in panel._combined_signs


def test_rebuild_combined_dataset_preserves_sign():
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(301, frames=1000, seed=1))
    panel.add_dataset(_dataset(302, frames=2000, seed=2))
    crn = panel.add_combined_dataset([301, 302], sign=-1)
    rebuilt = panel.rebuild_combined_dataset(crn)
    assert rebuilt is not None
    assert rebuilt.metadata["combination"]["method"] == "subtract_reference"
    assert rebuilt.metadata["combination"]["reference_scale"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Symmetric N-run signed co-subtract
# ---------------------------------------------------------------------------


def test_signed_subtract_action_combines_three_runs(monkeypatch):
    panel = DataBrowserPanel()
    for rn in (501, 502, 503):
        panel.add_dataset(_dataset(rn, frames=1000, seed=rn))
    panel.select_runs({501, 502, 503})
    # Drive the dialog to pick 501 as the sample (positive) run.
    monkeypatch.setattr(panel, "_prompt_signed_subtract", lambda runs: [501, 502, 503])
    panel._signed_subtract_selected()

    combined = next(iter(panel._combined_datasets))
    assert panel._combined_datasets[combined] == [501, 502, 503]
    assert panel._combined_signs[combined] == -1
    assert panel._combined_methods[combined] == "subtract_signed"
    ds = panel.get_dataset(combined)
    assert ds.metadata["combination"]["method"] == "subtract_signed"
    assert "reference_run_number" not in ds.metadata["combination"]
    assert panel._combined_run_display(combined) == "501 − 502 − 503"
    # Sources are hidden under the combined row.
    for rn in (501, 502, 503):
        assert rn not in panel._datasets


def test_signed_subtract_rebuild_preserves_method(monkeypatch):
    panel = DataBrowserPanel()
    for rn in (501, 502, 503):
        panel.add_dataset(_dataset(rn, frames=1000, seed=rn))
    panel.select_runs({501, 502, 503})
    monkeypatch.setattr(panel, "_prompt_signed_subtract", lambda runs: [501, 502, 503])
    panel._signed_subtract_selected()
    combined = next(iter(panel._combined_datasets))
    rebuilt = panel.rebuild_combined_dataset(combined)
    assert rebuilt is not None
    assert rebuilt.metadata["combination"]["method"] == "subtract_signed"


def test_signed_subtract_separate_restores_all_sources(monkeypatch):
    panel = DataBrowserPanel()
    for rn in (501, 502, 503):
        panel.add_dataset(_dataset(rn, frames=1000, seed=rn))
    panel.select_runs({501, 502, 503})
    monkeypatch.setattr(panel, "_prompt_signed_subtract", lambda runs: [501, 502, 503])
    panel._signed_subtract_selected()
    combined = next(iter(panel._combined_datasets))
    panel._separate_combined()
    for rn in (501, 502, 503):
        assert rn in panel._datasets
    assert combined not in panel._combined_methods
    assert combined not in panel._combined_signs


def test_refit_coadded_emits_selected_runs():
    panel = DataBrowserPanel()
    for rn in (601, 602, 603):
        panel.add_dataset(_dataset(rn, frames=1000, seed=rn))
    panel.select_runs({601, 602, 603})
    emitted: list[list[int]] = []
    panel.refit_coadded_requested.connect(emitted.append)
    panel._emit_refit_coadded()
    assert emitted == [[601, 602, 603]]


def test_refit_coadded_ignores_combined_rows(monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(601, frames=1000, seed=1))
    panel.add_dataset(_dataset(602, frames=1000, seed=2))
    crn = panel.add_combined_dataset([601, 602], sign=1)
    panel.select_runs({crn})
    emitted: list[list[int]] = []
    panel.refit_coadded_requested.connect(emitted.append)
    panel._emit_refit_coadded()
    assert emitted == []  # a single combined row is not a ≥2 regular selection


def test_signed_subtract_restores_via_add_combined_dataset():
    """Mirrors the .asymp restore path (operation='subtract_signed')."""
    panel = DataBrowserPanel()
    for rn in (501, 502, 503):
        panel.add_dataset(_dataset(rn, frames=1000, seed=rn))
    crn = panel.add_combined_dataset([501, 502, 503], sign=-1, operation="subtract_signed")
    assert crn is not None
    assert panel._combined_methods[crn] == "subtract_signed"
    assert panel.get_dataset(crn).metadata["combination"]["method"] == "subtract_signed"
