"""Focused tests for lightweight GUI panels."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QHeaderView

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.fourier_panel import FourierPanel
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.panels.maxent_panel import MaxEntPanel
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
    assert not hasattr(panel, "_source_combo")
    assert not hasattr(panel, "_group_output_combo")
    assert not hasattr(panel, "_center_range_check")
    assert panel._filter_none_radio.isChecked() is True
    assert panel._filter_start_edit.text() == "0.0"
    assert panel._filter_time_constant_edit.text() == "1.5"
    assert panel._filter_start_edit.isEnabled() is False
    assert panel._filter_time_constant_edit.isEnabled() is False
    assert panel._padding_spin.value() == 1
    assert float(panel._phase_spin.text()) == pytest.approx(0.0)
    assert float(panel._t0_offset_spin.text()) == pytest.approx(0.0)
    assert panel._current_display_mode() == "(Power)^1/2"
    assert panel._phase_mode_info_btn.text() == "Info"
    assert panel._phase_spin.isEnabled() is False
    assert panel._t0_offset_spin.isEnabled() is False
    assert panel._auto_phase_btn.isEnabled() is False
    assert panel._subtract_average_signal_check.isChecked() is True
    assert panel._subtract_average_signal_check.text() == "Subtract average signal"
    assert panel._auto_method_combo.currentText() == "Peak"
    assert panel._auto_phase_btn.text() == "Fill Phase Estimates"
    assert panel._estimate_average_error_check.text() == "Average errors"
    assert panel._estimate_average_error_check.toolTip() == "Estimate errors for averaged spectra."
    assert "#1f4d8a" in panel._phase_opt_real_radio.styleSheet()
    assert (
        panel._phase_opt_real_radio.minimumHeight()
        >= panel._phase_opt_real_radio.sizeHint().height()
    )
    assert panel._phase_table.horizontalHeaderItem(0).text() == "Include"
    assert (
        panel._phase_table.horizontalHeader().sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    )
    assert panel._fft_btn.text() == "Compute FFT"


def test_maxent_panel_defaults_and_group_state(qapp: QApplication) -> None:
    panel = MaxEntPanel()
    panel.set_group_definitions({2: "Right", 1: "Left"}, {1: 12.0}, {2: False})

    state = panel.get_state()

    assert panel._points_spin.value() == 1024
    assert state["default_level"] == pytest.approx(0.01)
    assert state["auto_window"] is True
    assert state["window_half_width_gauss"] == pytest.approx(300.0)
    assert state["t_min_us"] is None
    assert state["t_max_us"] is None
    assert state["time_binning_factor"] == 1
    assert panel.selected_group_ids() == [1]
    assert panel.group_phase_table()[1] == pytest.approx(12.0)
    assert panel.group_enabled_table() == {1: True, 2: False}
    assert panel._cycle_one_btn.text() == "+1"
    assert panel._converge_btn.text() == "Converge"

    config = panel.maxent_config(cycles=5)

    assert config.outer_cycles == 5
    assert config.selected_group_ids == [1]
    assert config.time_binning_factor == 1


def test_fourier_panel_group_order_matches_workflow(qapp: QApplication) -> None:
    panel = FourierPanel()

    titles = [group.title() for group in panel.findChildren(QGroupBox) if group.title()]

    assert titles[:4] == ["FFT Phase Mode", "Apodisation", "Groups", "Phase"]


def test_fourier_panel_apodisation_radios_enable_text_fields(qapp: QApplication) -> None:
    panel = FourierPanel()

    panel._filter_gaussian_radio.setChecked(True)

    assert panel._filter_start_edit.isEnabled() is True
    assert panel._filter_time_constant_edit.isEnabled() is True

    panel._filter_none_radio.setChecked(True)

    assert panel._filter_start_edit.isEnabled() is False
    assert panel._filter_time_constant_edit.isEnabled() is False


def test_fourier_panel_group_table_defaults_to_all_groups_enabled(qapp: QApplication) -> None:
    panel = FourierPanel()
    panel.set_group_definitions({1: "Left", 2: "Right"}, {1: 12.0, 2: -4.0})

    assert panel.selected_group_ids() == [1, 2]
    assert panel.group_enabled_table() == {1: True, 2: True}
    assert panel._phase_table.isEnabled() is True
    assert not (panel._phase_table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable)

    panel._use_phase_table_check.setChecked(True)
    panel._phase_mode_radio.setChecked(True)
    assert panel._phase_table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable

    panel._phase_table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)

    assert panel.selected_group_ids() == [1]


def test_fourier_panel_phase_mode_controls_follow_selected_mode(qapp: QApplication) -> None:
    panel = FourierPanel()
    panel.set_group_definitions({1: "Left", 2: "Right"}, {1: 12.0, 2: -4.0})

    panel._cos_radio.setChecked(True)
    assert float(panel._phase_spin.text()) == pytest.approx(0.0)
    assert panel._auto_method_combo.isEnabled() is False
    assert panel._auto_phase_btn.isEnabled() is False

    panel._sin_radio.setChecked(True)
    assert float(panel._phase_spin.text()) == pytest.approx(90.0)
    assert panel._t0_offset_spin.isEnabled() is False

    panel._phase_mode_radio.setChecked(True)
    assert panel._phase_spin.isEnabled() is True
    assert panel._t0_offset_spin.isEnabled() is True
    assert panel._auto_method_combo.isEnabled() is True
    assert panel._auto_phase_btn.isEnabled() is True
    assert "#1f4d8a" in panel._phase_spin.styleSheet()

    panel._use_phase_table_check.setChecked(True)
    assert "#67676b" in panel._phase_spin.styleSheet()
    assert panel._phase_table.item(0, 2).foreground().color().name() == "#1f4d8a"


def test_fourier_panel_auto_filled_group_phases_turn_green(qapp: QApplication) -> None:
    panel = FourierPanel()
    panel.set_group_definitions({1: "Left", 2: "Right"}, {1: 12.0, 2: -4.0})
    panel._phase_mode_radio.setChecked(True)
    panel._use_phase_table_check.setChecked(True)

    panel.set_group_phases({1: 5.0, 2: -3.0}, auto_filled=True)

    assert panel._phase_table.item(0, 2).foreground().color().name() == "#2a7a3f"
    assert panel._phase_table.item(1, 2).foreground().color().name() == "#2a7a3f"


def test_plot_panel_frequency_dataset_skips_rebin_and_uses_metadata_labels(
    qapp: QApplication,
) -> None:
    panel = PlotPanel()
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    dataset = MuonDataset(
        time=np.array([0.0, 0.5, 1.0]),
        asymmetry=np.array([1.0, 2.0, 1.5]),
        error=np.zeros(3, dtype=float),
        metadata={
            "run_number": 9,
            "plot_domain": "frequency",
            "x_label": "Frequency (MHz)",
            "y_label": "FFT Magnitude (a.u.)",
        },
    )

    panel.set_bunch_factor(4, emit_signal=False)
    assert panel.get_analysis_dataset(dataset) is dataset

    panel.plot_dataset(dataset)

    assert panel._ax.get_xlabel() == "Frequency (MHz)"
    assert panel._ax.get_ylabel() == "FFT Magnitude (a.u.)"


def test_frequency_plot_panel_switches_between_mhz_and_field_units(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = PlotPanel(domain="frequency")
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    dataset = MuonDataset(
        time=np.array([0.0, 1.0, 2.0]),
        asymmetry=np.array([1.0, 2.0, 1.5]),
        error=np.zeros(3, dtype=float),
        metadata={
            "run_number": 10,
            "plot_domain": "frequency",
            "y_label": "FFT Magnitude (a.u.)",
        },
    )

    panel.plot_dataset(dataset)
    panel.set_view_limits(0.0, 2.0, 0.0, 3.0)

    assert panel._ax.get_xlabel() == "Frequency (MHz)"
    assert panel._x_unit_label.text() == "MHz"

    draw_calls: list[int] = []
    monkeypatch.setattr(panel._canvas, "draw", lambda: draw_calls.append(1))

    panel._frequency_x_unit_combo.setCurrentText("Field (G)")

    assert panel._ax.get_xlabel() == "Field (G)"
    assert panel._x_unit_label.text() == "G"
    assert len(draw_calls) == 1
    x_min, x_max, _y_min, _y_max = panel.get_view_limits()
    assert x_min == pytest.approx(0.0)
    assert x_max == pytest.approx(2.0 / (135.538817 * 1.0e-4), abs=1e-3)


def test_fourier_panel_correlation_mode_reveals_controls(qapp: QApplication) -> None:
    panel = FourierPanel()
    # Controls are inert until the specialist mode is selected.
    assert panel._correlation_field_edit.isEnabled() is False
    assert panel._correlation_order_spin.isEnabled() is False

    panel._correlation_radio.setChecked(True)
    assert panel._current_display_mode() == "Correlation (radical)"
    assert panel._correlation_field_edit.isEnabled() is True
    assert panel._correlation_order_spin.isEnabled() is True

    state = panel.get_state()
    assert state["display"] == "Correlation (radical)"
    assert state["correlation_order"] == 2  # WiMDA default
    assert state["correlation_reference_field_gauss"] is None  # blank → auto


def test_fourier_panel_correlation_state_roundtrips(qapp: QApplication) -> None:
    panel = FourierPanel()
    panel._correlation_radio.setChecked(True)
    panel._correlation_field_edit.setText("2900")
    panel._correlation_order_spin.setValue(3)
    state = panel.get_state()
    assert state["correlation_reference_field_gauss"] == pytest.approx(2900.0)

    restored = FourierPanel()
    restored.restore_state(state)
    assert restored._current_display_mode() == "Correlation (radical)"
    assert restored._correlation_order_spin.value() == 3
    assert float(restored._correlation_field_edit.text()) == pytest.approx(2900.0)
    assert restored._correlation_field_edit.isEnabled() is True


def test_frequency_plot_panel_correlation_axis_locks_unit_selector(
    qapp: QApplication,
) -> None:
    panel = PlotPanel(domain="frequency")
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    correlation = MuonDataset(
        time=np.array([100.0, 200.0, 514.4, 700.0]),
        asymmetry=np.array([0.1, 0.2, 1.0, 0.1]),
        error=np.zeros(4, dtype=float),
        metadata={
            "run_number": 11,
            "plot_domain": "frequency",
            "x_label": "Muon hyperfine coupling Aμ (MHz)",
            "y_label": "Radical correlation (a.u.)",
            "correlation_axis": True,
        },
    )
    panel.plot_dataset(correlation)
    assert panel._ax.get_xlabel() == "Muon hyperfine coupling Aμ (MHz)"
    # The A_µ axis is a coupling, not γ_µ·B — the field-unit selector is locked.
    assert panel._frequency_x_unit_combo.isEnabled() is False
    assert panel._frequency_axis_is_correlation is True

    # Switching back to an ordinary frequency spectrum re-enables the selector.
    ordinary = MuonDataset(
        time=np.array([0.0, 1.0, 2.0]),
        asymmetry=np.array([1.0, 2.0, 1.5]),
        error=np.zeros(3, dtype=float),
        metadata={"run_number": 12, "plot_domain": "frequency", "y_label": "FFT Magnitude (a.u.)"},
    )
    panel.plot_dataset(ordinary)
    assert panel._frequency_axis_is_correlation is False
    assert panel._frequency_x_unit_combo.isEnabled() is True
    assert panel._ax.get_xlabel() == "Frequency (MHz)"


def test_frequency_plot_panel_can_show_relative_field_axis(qapp: QApplication) -> None:
    panel = PlotPanel(domain="frequency")
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    dataset = MuonDataset(
        time=np.array([1.0, 1.5, 2.0]),
        asymmetry=np.array([1.0, 2.0, 1.5]),
        error=np.zeros(3, dtype=float),
        metadata={
            "run_number": 11,
            "plot_domain": "frequency",
            "y_label": "FFT Magnitude (a.u.)",
            "field": 100.0,
        },
    )

    panel.plot_dataset(dataset)
    abs_x_min, abs_x_max, _abs_y_min, _abs_y_max = panel.get_view_limits()
    abs_axis_min, abs_axis_max = panel._ax.get_xlim()
    panel.set_frequency_axis_relative_to_reference(True)

    assert panel._ax.get_xlabel() == "Frequency (MHz)"
    x_min, x_max, _y_min, _y_max = panel.get_view_limits()
    center = 100.0 * 135.538817 * 1.0e-4
    assert x_min == pytest.approx(abs_x_min - center, abs=1e-3)
    assert x_max == pytest.approx(abs_x_max - center, abs=1e-3)
    assert panel._ax.get_xlim()[0] == pytest.approx(abs_axis_min, abs=1e-3)
    assert panel._ax.get_xlim()[1] == pytest.approx(abs_axis_max, abs=1e-3)


def test_frequency_plot_panel_relative_limits_drive_absolute_axis(qapp: QApplication) -> None:
    panel = PlotPanel(domain="frequency")
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    dataset = MuonDataset(
        time=np.array([1.0, 1.5, 2.0]),
        asymmetry=np.array([1.0, 2.0, 1.5]),
        error=np.zeros(3, dtype=float),
        metadata={
            "run_number": 12,
            "plot_domain": "frequency",
            "y_label": "FFT Magnitude (a.u.)",
            "field": 100.0,
        },
    )

    panel.plot_dataset(dataset)
    panel.set_frequency_axis_relative_to_reference(True)
    panel.set_view_limits(-0.5, 0.25, 0.0, 3.0)

    center = 100.0 * 135.538817 * 1.0e-4
    x_min, x_max, _y_min, _y_max = panel.get_view_limits()

    assert x_min == pytest.approx(-0.5, abs=1e-6)
    assert x_max == pytest.approx(0.25, abs=1e-6)
    assert panel._ax.get_xlim()[0] == pytest.approx(center - 0.5, abs=1e-3)
    assert panel._ax.get_xlim()[1] == pytest.approx(center + 0.25, abs=1e-3)


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


def test_plot_panel_set_global_fits_preserves_multi_dataset_view(qapp: QApplication) -> None:
    panel = PlotPanel()
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    t = np.linspace(0.0, 5.0, 50)
    ds1 = MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.5 * t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 7},
    )
    ds2 = MuonDataset(
        time=t,
        asymmetry=0.15 * np.exp(-0.3 * t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 8},
    )

    panel.plot_datasets([ds1, ds2])
    panel.set_global_fits(
        {
            7: (t, 0.2 * np.exp(-0.4 * t), "global"),
            8: (t, 0.15 * np.exp(-0.2 * t), "global"),
        }
    )

    assert len(panel._current_datasets) == 2
    assert panel._current_datasets[0] is ds1
    assert panel._current_datasets[1] is ds2


def test_plot_panel_set_global_fits_preserves_other_group_fits(qapp: QApplication) -> None:
    """Running a global fit for one group must not remove fit curves from others."""
    panel = PlotPanel()
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    t = np.linspace(0.0, 5.0, 50)

    # First global fit (group A, runs 1 & 2)
    panel.set_global_fits(
        {
            1: (t, 0.2 * np.exp(-0.5 * t), "Global Fit", []),
            2: (t, 0.15 * np.exp(-0.3 * t), "Global Fit", []),
        }
    )
    assert 1 in panel._fit_curves
    assert 2 in panel._fit_curves

    # Second global fit (group B, runs 3 & 4)
    panel.set_global_fits(
        {
            3: (t, 0.25 * np.exp(-0.4 * t), "Global Fit", []),
            4: (t, 0.10 * np.exp(-0.6 * t), "Global Fit", []),
        }
    )

    # Both groups' fit curves should be present
    assert 1 in panel._fit_curves, "Group A fit for run 1 was incorrectly removed"
    assert 2 in panel._fit_curves, "Group A fit for run 2 was incorrectly removed"
    assert 3 in panel._fit_curves
    assert 4 in panel._fit_curves


def test_plot_panel_can_render_grouped_time_domain_subplots(qapp: QApplication) -> None:
    panel = PlotPanel()
    if not getattr(panel, "_has_mpl", False):
        pytest.skip("matplotlib backend not available in this environment")

    t = np.linspace(0.0, 5.0, 50)
    ds1 = MuonDataset(
        time=t,
        asymmetry=100.0 * np.exp(-0.2 * t),
        error=np.full_like(t, 2.0),
        metadata={
            "run_number": -42001,
            "run_label": "Forward",
            "y_label": "Lifetime-corrected counts",
        },
    )
    ds2 = MuonDataset(
        time=t,
        asymmetry=90.0 * np.exp(-0.2 * t),
        error=np.full_like(t, 2.0),
        metadata={
            "run_number": -42002,
            "run_label": "Backward",
            "y_label": "Lifetime-corrected counts",
        },
    )

    panel.plot_grouped_time_domain_subplots([ds1, ds2])

    assert len(panel._current_datasets) == 2
    assert len(panel._subplot_axes_by_polarization) == 2
    assert panel._current_dataset is ds2
