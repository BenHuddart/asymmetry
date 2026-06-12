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
