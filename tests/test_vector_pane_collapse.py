"""RED target for branch ``fix/vector-pane-persist`` (Round-3 GUI bug B4).

Round-3 GUI finding (``_findings/windows-gui/Round3_progress.md``): after enabling
the HiFi Transverse **"all"** projection (the dual-pane Left-Right / Top-Bottom
vector view), then switching to **non-vector** data (new project + EMU 2-group
runs, or any single-projection grouping), the plot **stays split into two stacked
panes**. The projection chip bar disappears (so there's no control to pick a single
axis), ``View → Reset layout`` does not help, and the only recovery is toggling a
data-view sub-tab — non-discoverable.

Root cause: ``PlotPanel.set_projections([])`` (the ``len(specs) < 2`` branch) resets
the projection *state* (``_current_polarization_axis = None``,
``_vector_subplot_datasets = {}``, chip bar cleared) but never triggers a **replot**,
unlike the ``>= 2`` branch which calls ``_apply_limits()``. So the canvas keeps the
stale dual-pane subplots until some unrelated replot fires.

Contract: clearing the projections while a vector (multi-pane) view was active must
collapse the figure back to a single pane. RED today (the figure keeps both axes).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.plot_panel import PlotPanel


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app) -> PlotPanel:
    widget = PlotPanel()
    yield widget
    widget.close()
    widget.deleteLater()


def _dataset(run_number: int = 12345) -> MuonDataset:
    t = np.linspace(0, 10, 100)
    a = 0.2 * np.exp(-0.5 * t)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": run_number})


def test_clearing_active_vector_projection_collapses_to_single_pane(panel):
    ds = _dataset()

    # Baseline: a normal single-dataset plot is one Axes.
    panel.plot_dataset(ds)
    assert len(panel._figure.axes) == 1

    # Simulate an active vector "ALL" dual-pane view (what the HiFi Transverse
    # "all" projection produces): two stacked subplots + the vector state set.
    panel._current_polarization_axis = "ALL"
    panel._vector_subplot_datasets = {"P_x": [ds], "P_y": [ds]}
    panel._figure.clf()
    panel._figure.add_subplot(2, 1, 1)
    panel._figure.add_subplot(2, 1, 2)
    assert len(panel._figure.axes) == 2, "precondition: dual-pane staged"

    # Switching to non-vector data clears the projections (fewer than two specs).
    panel.set_projections([])

    # State is reset today...
    assert panel._current_polarization_axis is None
    assert panel._vector_subplot_datasets == {}

    # ...but the canvas must also collapse back to a single pane. BUG B4: it does
    # not — set_projections([]) clears state without replotting, so both axes
    # linger until an unrelated replot fires.
    assert len(panel._figure.axes) == 1, (
        "dual-pane persisted after clearing the vector projection: "
        "set_projections([]) cleared the vector state but did not replot to a "
        "single pane (no _redraw_current_view / _apply_limits in the clear branch)"
    )
