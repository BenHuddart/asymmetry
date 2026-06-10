"""GUI-level coverage for the Phase 1 MaxEnt reconstruction overlay."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QWidget  # type: ignore

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.maxent import (
    MaxEntConfig,
    build_maxent_input,
    reconstruct_group_signals,
    run_cycles,
)
from asymmetry.core.representation import build_maxent_reconstruction_datasets
from asymmetry.gui.panels.maxent_panel import MaxEntPanel
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.panels.plot_workspace_panel import PlotWorkspacePanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fb_run() -> Run:
    rng = np.random.default_rng(11)
    bin_width = 0.04
    n = 256
    time = np.arange(n, dtype=float) * bin_width
    histograms = []
    for phase in (0.0, 180.0):
        signal = 1.0 + 0.20 * np.cos(2.0 * np.pi * 1.5 * time + np.deg2rad(phase))
        counts = 3000.0 * np.exp(-time / 2.1969811) * signal
        histograms.append(
            Histogram(
                counts=rng.poisson(np.clip(counts, 1.0, None)).astype(float),
                bin_width=bin_width,
                t0_bin=0,
            )
        )
    return Run(
        run_number=7,
        histograms=histograms,
        metadata={"field": 110.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


def _reconstruction_datasets():
    run = _fb_run()
    config = MaxEntConfig(
        n_spectrum_points=128,
        f_min_mhz=0.5,
        f_max_mhz=3.0,
        auto_window=False,
        outer_cycles=4,
        inner_iterations=4,
        fit_phases=False,
        group_phase_degrees={1: 0.0, 2: 180.0},
    )
    maxent_input = build_maxent_input(run, config)
    result = run_cycles(maxent_input, config)
    recon = reconstruct_group_signals(maxent_input, result.state)
    return build_maxent_reconstruction_datasets(recon, run)


def test_plot_panel_renders_reconstruction_with_residual_strips(qapp: QApplication) -> None:
    panel = PlotPanel()
    try:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        datasets = _reconstruction_datasets()
        panel.plot_maxent_reconstruction(datasets)
        # One main axis + one residual strip per group.
        assert len(panel._figure.axes) == 2 * len(datasets)
    finally:
        panel.close()
        panel.deleteLater()


def test_maxent_panel_reconstruction_toggle_round_trips_and_signals(qapp: QApplication) -> None:
    panel = MaxEntPanel()
    try:
        assert panel.get_state()["show_reconstruction"] is False
        assert panel.show_reconstruction_enabled() is False

        received: list[bool] = []
        panel.reconstruction_toggled.connect(received.append)
        panel._show_reconstruction_check.setChecked(True)
        assert received == [True]
        assert panel.get_state()["show_reconstruction"] is True

        # restore_state and set_show_reconstruction must not re-emit the signal.
        received.clear()
        panel.restore_state({"show_reconstruction": False})
        assert panel.show_reconstruction_enabled() is False
        panel.set_show_reconstruction(True)
        assert panel.show_reconstruction_enabled() is True
        assert received == []
    finally:
        panel.close()
        panel.deleteLater()


def test_plot_workspace_exposes_reconstruction_as_time_view(qapp: QApplication) -> None:
    workspace = PlotWorkspacePanel(time_panel=QWidget(), frequency_panel=QWidget())
    try:
        assert "reconstruction" in workspace._VIEW_TOKENS
        workspace.set_available_views(["fb_asymmetry", "reconstruction", "maxent"])
        assert workspace.is_view_enabled("reconstruction")
        workspace.set_active_view("reconstruction")
        assert workspace.active_view() == "reconstruction"
        # It is a time-domain view (renders on the time panel), not a freq view.
        assert workspace.active_domain() == "time"
    finally:
        workspace.close()
        workspace.deleteLater()
