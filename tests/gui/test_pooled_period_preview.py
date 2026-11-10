"""Pooled preview and the rotating-frame availability boundary in actual widgets."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.rotating_frame import Provenance, RotatingFrame
from tests.core.vector_synthetic import synthetic_vector_run
from tests.gui.test_grouping_preview_pane import _histogram_dataset, _reduce_preview
from tests.gui.test_rotating_frame_projection import _rotate, _show
from tests.gui.test_rotating_frame_projection import mainwindow as mainwindow

pytestmark = pytest.mark.gui


def test_pooled_preview_has_one_ratio(qapp):
    from asymmetry.core.io.periods import combine_mapped_periods

    sources = [
        _histogram_dataset(forward=np.full(16, f), backward=np.full(16, b))
        for f, b in [(60, 40), (90, 10)]
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "green"})
    mapped.run.grouping["period_mode"] = "green_plus_red"
    result = _reduce_preview(mapped, run=mapped.run)
    np.testing.assert_allclose(result.asymmetry, 50)


def test_unequal_pooled_frame_explains_unavailability_and_lab_remains(mainwindow):
    def polarisation(t):
        return np.array([np.cos(t), np.sin(t), np.ones_like(t)])

    run = synthetic_vector_run([polarisation, polarisation], n_bins=64, seed=None)
    run.grouping["period_mode"] = "green_plus_red"
    dataset = MuonDataset(np.zeros(1), np.zeros(1), np.ones(1), run=run)
    frame = RotatingFrame.typed_frequency(1, 2).with_baseline(1, Provenance.TYPED, 1, 2)
    mainwindow._project_model.rotating_frames[run.run_number] = frame
    panel = _show(mainwindow, dataset)
    _rotate(panel)
    assert "Pooled G+R rotation unavailable" in panel._ax.texts[0].get_text()
    assert mainwindow._frame_snapshot(dataset) is None
    assert panel.frame_bar._edits["baseline_x"].text() == ""
    assert "must agree" in panel.frame_bar._edits["baseline_x"].toolTip()
    panel._projection_bar._lab_btn.click()
    assert mainwindow._build_vector_axis_datasets([dataset], ["P_x"])["P_x"]
    equal = frame.with_baseline(0, Provenance.TYPED, 1, 2)
    mainwindow._project_model.rotating_frames[run.run_number] = equal
    _rotate(panel)
    assert mainwindow._frame_snapshot(dataset) is not None
    assert list(panel._subplot_axes_by_polarization) == ["P′_x", "P′_y", "P_z"]
