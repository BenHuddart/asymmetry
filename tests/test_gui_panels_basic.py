"""Focused tests for lightweight GUI panels."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.fourier_panel import FourierPanel
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.panels.plot_panel import PlotPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_log_panel_appends_message(qapp: QApplication) -> None:
    panel = LogPanel()
    panel.log("hello world")
    text = panel._text.toPlainText()
    assert "hello world" in text


def test_fourier_panel_defaults(qapp: QApplication) -> None:
    panel = FourierPanel()
    assert panel._window_combo.currentText() == "none"
    assert panel._padding_spin.value() == 1
    assert panel._display_combo.currentText() == "Real"
    assert panel._fft_btn.text() == "Compute FFT"


def test_plot_panel_basic_plot_fit_clear_flow(qapp: QApplication) -> None:
    panel = PlotPanel()
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    t = np.linspace(0.0, 5.0, 50)
    ds = MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.5 * t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 7},
    )

    panel.plot_dataset(ds)
    panel.plot_fit(t, 0.2 * np.exp(-0.4 * t), label="fit")
    panel.clear_fit()
    panel.set_global_fits({7: (t, 0.2 * np.exp(-0.3 * t), "global")})
    panel.clear()

    assert panel._current_dataset is None
    assert panel._fit_curve is None
    assert panel._fit_curves == {}
