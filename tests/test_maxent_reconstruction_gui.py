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


def test_plot_panel_renders_combined_reconstruction_on_one_axis(qapp: QApplication) -> None:
    panel = PlotPanel()
    try:
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")
        datasets = _reconstruction_datasets()
        assert len(datasets) >= 2  # the combined layout only earns its keep here
        panel.plot_maxent_reconstruction(datasets, combined=True)
        # Combined: exactly one main axis + one shared residuals strip, whatever
        # the group count — distinct from the per-group stack (2·n axes).
        assert len(panel._figure.axes) == 2
    finally:
        panel.close()
        panel.deleteLater()


def test_maxent_panel_combined_reconstruction_toggle_round_trips_and_signals(
    qapp: QApplication,
) -> None:
    panel = MaxEntPanel()
    try:
        assert panel.reconstruction_combined() is False
        assert panel.get_state()["reconstruction_combined"] is False

        received: list[bool] = []
        panel.reconstruction_layout_changed.connect(received.append)
        panel._combine_reconstruction_check.setChecked(True)
        assert received == [True]
        assert panel.reconstruction_combined() is True
        assert panel.get_state()["reconstruction_combined"] is True

        # restore_state must not re-emit; a missing key defaults OFF.
        received.clear()
        panel.restore_state({"reconstruction_combined": False})
        assert panel.reconstruction_combined() is False
        panel.restore_state({})
        assert panel.reconstruction_combined() is False
        assert received == []
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
        # A state dict missing the key must default OFF (matches init/to_dict),
        # not silently flip the overlay on.
        panel.restore_state({})
        assert panel.show_reconstruction_enabled() is False
        panel.set_show_reconstruction(True)
        assert panel.show_reconstruction_enabled() is True
        assert received == []
    finally:
        panel.close()
        panel.deleteLater()


def test_maxent_panel_pulse_and_exclusion_controls_round_trip(qapp: QApplication) -> None:
    panel = MaxEntPanel()
    try:
        panel.restore_state(
            {
                "pulse_mode": "double",
                "pulse_half_width_us": 0.08,
                "pulse_separation_us": 0.324,
                "exclude_t_min_us": 1.5,
                "exclude_t_max_us": 2.5,
            }
        )
        state = panel.get_state()
        assert state["pulse_mode"] == "double"
        assert state["pulse_half_width_us"] == pytest.approx(0.08)
        assert state["pulse_separation_us"] == pytest.approx(0.324)
        assert state["exclude_t_min_us"] == pytest.approx(1.5)
        assert state["exclude_t_max_us"] == pytest.approx(2.5)
        # These flow straight into a MaxEntConfig.
        config = panel.maxent_config(cycles=1)
        assert config.pulse_mode == "double"
        assert config.exclude_t_min_us == pytest.approx(1.5)
    finally:
        panel.close()
        panel.deleteLater()


def test_maxent_panel_mode_specbg_and_phase_exchange(qapp: QApplication) -> None:
    panel = MaxEntPanel()
    try:
        # Mode + SpecBG round-trip; SpecBG only enabled in ZF/LF mode.
        assert panel.mode() == "general"
        assert panel._specbg_group.isEnabled() is False
        panel.restore_state(
            {
                "mode": "zf_lf",
                "specbg_enabled": True,
                "specbg_gaussian_width_mhz": 0.2,
                "specbg_lorentzian_fraction": 0.3,
            }
        )
        assert panel.mode() == "zf_lf"
        assert panel._specbg_group.isEnabled() is True
        state = panel.get_state()
        assert state["mode"] == "zf_lf"
        assert state["specbg_enabled"] is True
        assert state["specbg_gaussian_width_mhz"] == pytest.approx(0.2)
        assert state["specbg_lorentzian_fraction"] == pytest.approx(0.3)

        # apply_phase_table updates only matching rows.
        panel.set_group_definitions({1: "F", 2: "B"}, {1: 0.0, 2: 0.0}, {1: True, 2: True})
        updated = panel.apply_phase_table({1: 12.5, 2: 192.5})
        assert updated == 2
        assert panel.group_phase_table()[1] == pytest.approx(12.5)
        assert panel.group_phase_table()[2] == pytest.approx(192.5)

        # The exchange/export actions exist as signals the main window wires.
        for name in (
            "use_fitted_phases_requested",
            "send_phases_to_fit_requested",
            "fit_deadtime_requested",
            "apply_deadtime_requested",
            "export_spectrum_requested",
            "export_log_requested",
        ):
            assert hasattr(panel, name)
    finally:
        panel.close()
        panel.deleteLater()


def test_frequency_plot_panel_offers_tesla_axis(qapp: QApplication) -> None:
    panel = PlotPanel(domain="frequency")
    try:
        if not getattr(panel, "_has_mpl", False) or not hasattr(panel, "_frequency_x_unit_combo"):
            pytest.skip("frequency plot panel unavailable")
        units = [
            panel._frequency_x_unit_combo.itemData(i)
            for i in range(panel._frequency_x_unit_combo.count())
        ]
        assert units == ["frequency_mhz", "field_gauss", "field_tesla"]
        # In Tesla mode a 135.538817 MHz line maps to 1 T (γ_μ/2π).
        panel._current_frequency_x_unit = "field_tesla"
        converted = panel._convert_frequency_axis_for_display(np.array([135.538817]))
        assert float(converted[0]) == pytest.approx(1.0)
        assert panel._display_x_label() == "Field (T)"
    finally:
        panel.close()
        panel.deleteLater()


def test_plot_workspace_exposes_reconstruction_as_time_view(qapp: QApplication) -> None:
    workspace = PlotWorkspacePanel(time_panel=QWidget(), frequency_panel=QWidget())
    try:
        assert "reconstruction" in workspace._VIEW_TOKENS
        workspace.set_available_views(["fb_asymmetry", "groups", "reconstruction", "maxent"])
        assert workspace.is_view_enabled("reconstruction")
        workspace.set_active_view("groups")
        workspace.set_active_view("reconstruction")
        assert workspace.active_view() == "reconstruction"
        # It is a time-domain view (renders on the time panel), not a freq view.
        assert workspace.active_domain() == "time"
        # The diagnostic overlay must NOT become the time-view fallback: a switch
        # back to the time domain lands on real data (the last primary view).
        workspace.set_active_view("frequency")
        workspace.set_active_domain("time")
        assert workspace.active_view() == "groups"
    finally:
        workspace.close()
        workspace.deleteLater()


def test_multi_group_fit_window_exposes_phase_exchange_methods(qapp: QApplication) -> None:
    # Regression: the MaxEnt phase-exchange handlers target
    # self._multi_group_fit_window, so both methods must live there (they are
    # defined on the grouped tabs, not on FitPanel).
    from asymmetry.gui.windows.multi_group_fit_window import MultiGroupFitWindow

    window = MultiGroupFitWindow(None)
    try:
        assert hasattr(window, "grouped_simulate_seed_for_run")
        assert hasattr(window, "update_grouped_phase_seed")
        # No grouped fit cached → update returns False, not an exception.
        assert window.update_grouped_phase_seed(999, {1: 0.0}) is False
    finally:
        window.close()
        window.deleteLater()
