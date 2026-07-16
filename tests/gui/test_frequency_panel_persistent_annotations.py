"""Pinned frequency-panel annotations/labels survive a replot (show-time render).

The frequency ``PlotPanel`` clears its axis and recomputes the x-axis label on
every render, so markers/labels drawn straight onto the axis were wiped by the
show-time replot (benzene corpus sweep). ``set_custom_x_axis_label`` and
``add_persistent_frequency_marker`` pin decorations that are recreated on every
render instead.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.gui.panels.plot_panel import PlotPanel  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _frequency_dataset(run_number: int = 1, center: float = 3.2) -> MuonDataset:
    freq = np.linspace(1.0, 5.0, 401)
    values = 0.4 + 7.0 * np.exp(-4.0 * np.log(2.0) * ((freq - center) / 0.35) ** 2)
    return MuonDataset(
        time=freq,
        asymmetry=values,
        error=np.full_like(freq, 0.05),
        metadata={"run_number": run_number, "plot_domain": "frequency", "field": 200.0},
    )


def _vertical_marker_count(panel: PlotPanel) -> int:
    """Number of full-height dashed marker lines currently on the axis."""
    from matplotlib.lines import Line2D

    count = 0
    for line in panel._ax.get_lines() + list(getattr(panel._ax, "artists", [])):
        if isinstance(line, Line2D):
            ydata = list(line.get_ydata())
            if ydata == [0.0, 1.0]:
                count += 1
    return count


def test_custom_x_axis_label_survives_replot(app: QApplication) -> None:
    panel = PlotPanel(domain="frequency")
    panel.plot_dataset(_frequency_dataset())

    panel.set_custom_x_axis_label("Hyperfine coupling A_µ (MHz)")
    assert panel._ax.get_xlabel() == "Hyperfine coupling A_µ (MHz)"

    # A fresh render (the show-time replot recomputes the label) must not wipe it.
    panel.plot_dataset(_frequency_dataset())
    assert panel._ax.get_xlabel() == "Hyperfine coupling A_µ (MHz)"

    # Clearing the override restores the computed label.
    panel.set_custom_x_axis_label(None)
    assert panel._ax.get_xlabel() != "Hyperfine coupling A_µ (MHz)"
    panel.deleteLater()


def test_persistent_frequency_marker_survives_replot(app: QApplication) -> None:
    panel = PlotPanel(domain="frequency")
    panel.plot_dataset(_frequency_dataset())
    baseline = _vertical_marker_count(panel)

    panel.add_persistent_frequency_marker(3.2, label="ν₁")
    after_add = _vertical_marker_count(panel)
    assert after_add == baseline + 1

    # The marker is recreated by the next render rather than wiped by it.
    panel.plot_dataset(_frequency_dataset())
    assert _vertical_marker_count(panel) == baseline + 1

    panel.clear_persistent_frequency_markers()
    panel.plot_dataset(_frequency_dataset())
    assert _vertical_marker_count(panel) == baseline
    panel.deleteLater()
