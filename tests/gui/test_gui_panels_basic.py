"""Focused tests for lightweight GUI panels."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)

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
    # P3-2: apodisation value fields stay editable (with a clarifying tooltip)
    # even when the mode is "None"; a greyed field reads as broken.
    assert panel._filter_start_edit.isEnabled() is True
    assert panel._filter_time_constant_edit.isEnabled() is True
    assert "Lorentzian" in panel._filter_start_edit.toolTip()
    assert "Lorentzian" in panel._filter_time_constant_edit.toolTip()
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
    assert panel._auto_phase_btn.text() == "Fill phases"
    assert panel._estimate_average_error_check.text() == "Average errors"
    assert panel._estimate_average_error_check.toolTip() == "Estimate errors for averaged spectra."
    assert "#1f4d8a" in panel._phase_opt_real_radio.styleSheet()
    assert (
        panel._phase_opt_real_radio.minimumHeight()
        >= panel._phase_opt_real_radio.sizeHint().height()
    )
    assert panel._phase_table.horizontalHeaderItem(0).text() == "✓"
    assert (
        panel._phase_table.horizontalHeader().sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    )
    assert panel._fft_btn.text() == "Compute FFT"


def test_fourier_compute_button_pinned_outside_scroll(qapp: QApplication) -> None:
    """Compute FFT must live in the pinned footer, not the scrolling content.

    Regression guard for the "bury-the-button" fix: the primary action sat at
    the bottom of the ~9-section scroll content and was unreachable at the
    default window size. It now lives in a footer added to the panel's top-level
    layout, so it stays visible at any scroll position.
    """
    panel = FourierPanel()
    scroll = panel.findChild(QScrollArea)
    footer = panel._action_footer
    assert scroll is not None
    assert footer is not None
    assert panel._fft_btn in footer.findChildren(QPushButton)
    assert panel._fft_btn not in scroll.widget().findChildren(QPushButton)


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


def test_maxent_cycle_controls_pinned_outside_scroll(qapp: QApplication) -> None:
    """The cycle/Converge grid must live in the pinned footer, not mid-scroll.

    Regression guard for the "bury-the-button" fix: the cycle controls sat
    mid-scroll with six more sections below them, buried whichever way the user
    scrolled. They now live in a footer added to the panel's top-level layout.
    """
    panel = MaxEntPanel()
    scroll = panel.findChild(QScrollArea)
    footer = panel.findChild(QWidget, "maxentActionFooter")
    assert scroll is not None
    assert footer is not None
    assert panel._converge_btn in footer.findChildren(QPushButton)
    assert panel._converge_btn not in scroll.widget().findChildren(QPushButton)


def test_maxent_footer_buttons_present_and_enable_state_tracks_busy(qapp: QApplication) -> None:
    """All run-control buttons live in the pinned footer and flip with set_busy.

    Idle: the cycle steppers, Converge, Restart, and Apply-to-selection are
    enabled and Cancel is disabled. Running: the reverse, so a user cannot
    stack a second run and Cancel is the only reachable action.
    """
    panel = MaxEntPanel()
    footer = panel.findChild(QWidget, "maxentActionFooter")
    assert footer is not None
    for button in (
        panel._cycle_one_btn,
        panel._cycle_five_btn,
        panel._cycle_twentyfive_btn,
        panel._converge_btn,
        panel._restart_btn,
        panel._cancel_btn,
        panel._apply_to_selection_btn,
    ):
        assert button in footer.findChildren(QPushButton)

    # Idle (constructor default): run buttons enabled, Cancel disabled.
    assert panel._cycle_one_btn.isEnabled() is True
    assert panel._converge_btn.isEnabled() is True
    assert panel._restart_btn.isEnabled() is True
    assert panel._apply_to_selection_btn.isEnabled() is True
    assert panel._cancel_btn.isEnabled() is False

    panel.set_busy(True)
    assert panel._cycle_one_btn.isEnabled() is False
    assert panel._cycle_five_btn.isEnabled() is False
    assert panel._cycle_twentyfive_btn.isEnabled() is False
    assert panel._converge_btn.isEnabled() is False
    assert panel._restart_btn.isEnabled() is False
    assert panel._apply_to_selection_btn.isEnabled() is False
    assert panel._cancel_btn.isEnabled() is True

    panel.set_busy(False)
    assert panel._cycle_one_btn.isEnabled() is True
    assert panel._converge_btn.isEnabled() is True
    assert panel._cancel_btn.isEnabled() is False


def test_maxent_progress_uses_determinate_footer_bar(qapp: QApplication) -> None:
    """set_progress drives the footer's determinate 0..total bar (per-cycle count)."""
    panel = MaxEntPanel()
    panel.set_busy(True)
    panel.set_progress(3, 10, "Cycle 3 of 10")
    footer = panel._action_footer
    assert footer._progress_bar.minimum() == 0
    assert footer._progress_bar.maximum() == 10
    assert footer._progress_bar.value() == 3
    assert "Cycle 3 of 10" in footer._progress_label.text()
    panel.set_busy(False)
    # Idle resets the bar back to indeterminate for the next run.
    assert footer._progress_bar.maximum() == 0


def test_maxent_panel_sections_collapse_persist_via_isolated_settings(qapp: QApplication) -> None:
    """Collapsed/expanded state for the collapsible sections persists across
    panel instances via QSettings, isolated from the app's real settings scope
    (mirrors the PanelSection persistence pattern)."""
    settings = QSettings("AsymmetryTest", "maxent_panel_sections_test")
    settings.clear()
    try:
        panel = MaxEntPanel()
        # Patch the already-constructed sections' settings scope for this test
        # (MaxEntPanel does not take a settings override, so drive the
        # PanelSection objects directly — same isolation guarantee).
        pulse_section = panel._pulse_section
        pulse_section._settings = settings
        pulse_section._settings_key = "maxent/sections/pulse_shape"

        assert not pulse_section.isExpanded()
        pulse_section.setExpanded(True)
        assert settings.value("maxent/sections/pulse_shape", type=bool) is True

        # A fresh PanelSection reading the same key comes up expanded.
        from asymmetry.gui.widgets.panel_section import PanelSection

        reloaded = PanelSection(
            "Pulse shape",
            collapsible=True,
            settings_key="maxent/sections/pulse_shape",
            settings=settings,
        )
        assert reloaded.isExpanded()
    finally:
        settings.clear()


def test_maxent_panel_specbg_section_title_suffix_reflects_enabled(qapp: QApplication) -> None:
    """The Zero-frequency background section shows an 'on' summary chip when
    its enable checkbox is checked, and clears it when unchecked — the
    "meaningful collapsed state gets a summary" requirement."""
    panel = MaxEntPanel()
    assert panel._specbg_group._suffix_label.isHidden()
    panel._specbg_enabled_check.setChecked(True)
    assert not panel._specbg_group._suffix_label.isHidden()
    assert "on" in panel._specbg_group._suffix_label.text().lower()
    panel._specbg_enabled_check.setChecked(False)
    assert panel._specbg_group._suffix_label.isHidden()


def test_maxent_panel_settings_round_trip_unchanged_after_restructure(qapp: QApplication) -> None:
    """A full get_state -> restore_state round trip is unaffected by the
    section-based restructure (serialization keys/shape are unchanged)."""
    panel = MaxEntPanel()
    panel.set_group_definitions({1: "F", 2: "B"})
    panel.restore_state(
        {
            "n_spectrum_points": 2048,
            "default_level": 0.02,
            "auto_window": False,
            "window_half_width_gauss": 150.0,
            "f_min_mhz": 0.1,
            "f_max_mhz": 5.0,
            "t_min_us": 0.05,
            "t_max_us": 8.0,
            "time_binning_factor": 2,
            "pulse_mode": "single",
            "pulse_half_width_us": 0.09,
            "pulse_separation_us": 0.4,
            "mode": "zf_lf",
            "specbg_enabled": True,
            "specbg_gaussian_width_mhz": 0.3,
            "specbg_lorentzian_width_mhz": 0.4,
            "specbg_lorentzian_fraction": 0.6,
            "inner_iterations": 20,
            "chi2_target_over_n": 1.2,
            "fit_phases": False,
            "fit_amplitudes": False,
            "fit_backgrounds": False,
            "fit_constant_background": False,
            "use_deadtime_correction": False,
            "show_reconstruction": True,
            "reconstruction_combined": True,
            "group_enabled_table": {1: True, 2: False},
            "group_phase_degrees": {1: 10.0, 2: 190.0},
        }
    )
    state = panel.get_state()
    assert state["n_spectrum_points"] == 2048
    assert state["default_level"] == pytest.approx(0.02)
    assert state["auto_window"] is False
    assert state["window_half_width_gauss"] == pytest.approx(150.0)
    assert state["f_min_mhz"] == pytest.approx(0.1)
    assert state["f_max_mhz"] == pytest.approx(5.0)
    assert state["t_min_us"] == pytest.approx(0.05)
    assert state["t_max_us"] == pytest.approx(8.0)
    assert state["time_binning_factor"] == 2
    assert state["pulse_mode"] == "single"
    assert state["pulse_half_width_us"] == pytest.approx(0.09)
    assert state["pulse_separation_us"] == pytest.approx(0.4)
    assert state["mode"] == "zf_lf"
    assert state["specbg_enabled"] is True
    assert state["specbg_gaussian_width_mhz"] == pytest.approx(0.3)
    assert state["specbg_lorentzian_width_mhz"] == pytest.approx(0.4)
    assert state["specbg_lorentzian_fraction"] == pytest.approx(0.6)
    assert state["inner_iterations"] == 20
    assert state["chi2_target_over_n"] == pytest.approx(1.2)
    assert state["fit_phases"] is False
    assert state["fit_amplitudes"] is False
    assert state["fit_backgrounds"] is False
    assert state["fit_constant_background"] is False
    assert state["use_deadtime_correction"] is False
    assert state["show_reconstruction"] is True
    assert state["reconstruction_combined"] is True
    assert state["group_enabled_table"] == {1: True, 2: False}
    assert state["group_phase_degrees"][1] == pytest.approx(10.0)
    assert state["group_phase_degrees"][2] == pytest.approx(190.0)

    config = panel.maxent_config(cycles=7)
    assert config.mode == "zf_lf"
    assert config.n_spectrum_points == 2048


def test_fourier_panel_group_order_matches_workflow(qapp: QApplication) -> None:
    """The usage-tier restructure orders sections always-visible → conditional →
    collapsed: FFT Phase Mode, Apodisation, Groups, FFT settings, then the
    conditional Phase section, then the collapsed stack."""
    from asymmetry.gui.styles.widgets import SECTION_HEADER_OBJECT_NAME
    from asymmetry.gui.widgets.panel_section import PanelSection

    panel = FourierPanel()

    # Flat section headers (make_section) carry the shared header objectName; the
    # nested "Advanced / experimental" disclosure header is excluded so only the
    # top-level workflow sections remain, in layout order.
    advanced_labels = set(panel._advanced_modes_group.findChildren(QLabel))
    titles = [
        label.text()
        for label in panel.findChildren(QLabel)
        if label.objectName() == SECTION_HEADER_OBJECT_NAME and label not in advanced_labels
    ]
    # make_section uppercases; assert the always-visible tier then the Phase gate.
    assert titles[:5] == ["FFT PHASE MODE", "APODISATION", "GROUPS", "FFT SETTINGS", "PHASE"]

    # The four collapsed sections follow, each a collapsible PanelSection.
    collapsed_titles = [
        section.title()
        for section in panel.findChildren(PanelSection)
        if section is not panel._advanced_modes_group
    ]
    assert collapsed_titles == [
        "Spectral moments",
        "Conditioning",
        "Diamagnetic correction",
        "Frequency exclusions",
    ]


def test_fourier_panel_advanced_modes_collapsed_by_default(qapp: QApplication) -> None:
    """P2-1: the three niche phase modes live behind a collapsed disclosure."""
    panel = FourierPanel()

    group = panel._advanced_modes_group
    assert group.title() == "Advanced"
    # Collapsed by default — only the routine modes show up front.
    assert group.isExpanded() is False
    # The three niche radios are parented into the disclosure, not the main column.
    for radio in (panel._phase_opt_real_radio, panel._burg_radio, panel._correlation_radio):
        assert radio in group.findChildren(type(radio))
    # Labels are short (qualifiers moved to tooltips) so nothing clips at 236px.
    assert panel._burg_radio.text() == "Resolution (Burg)"
    assert panel._correlation_radio.text() == "Correlation (radical)"
    assert "diagnostic" in panel._burg_radio.toolTip().lower()
    assert "specialist" in panel._correlation_radio.toolTip().lower()
    # Radio exclusivity still spans both the main column and the disclosure.
    panel._burg_radio.setChecked(True)
    assert panel._power_sqrt_radio.isChecked() is False
    assert panel._current_display_mode() == "Resolution (Burg)"


def test_fourier_panel_restoring_advanced_mode_expands_disclosure(qapp: QApplication) -> None:
    """Restoring a project saved in an advanced mode reveals the disclosure."""
    panel = FourierPanel()
    assert panel._advanced_modes_group.isExpanded() is False

    panel._set_display_mode("phaseOptReal")

    assert panel._phase_opt_real_radio.isChecked() is True
    assert panel._advanced_modes_group.isExpanded() is True


def test_fourier_panel_apodisation_fields_stay_enabled(qapp: QApplication) -> None:
    """P3-2: the apodisation value fields are editable regardless of mode."""
    panel = FourierPanel()

    assert panel._filter_start_edit.isEnabled() is True
    assert panel._filter_time_constant_edit.isEnabled() is True

    panel._filter_gaussian_radio.setChecked(True)

    assert panel._filter_start_edit.isEnabled() is True
    assert panel._filter_time_constant_edit.isEnabled() is True

    panel._filter_none_radio.setChecked(True)

    assert panel._filter_start_edit.isEnabled() is True
    assert panel._filter_time_constant_edit.isEnabled() is True


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
    ordinary = MuonDataset(
        time=np.array([0.0, 1.0, 2.0]),
        asymmetry=np.array([1.0, 2.0, 1.5]),
        error=np.zeros(3, dtype=float),
        metadata={"run_number": 12, "plot_domain": "frequency", "y_label": "FFT Magnitude (a.u.)"},
    )

    # The user is viewing a normal FFT in Field (G) with the relative axis on.
    panel.plot_dataset(ordinary)
    panel._frequency_x_unit_combo.setCurrentText("Field (G)")
    panel._frequency_axis_relative_check.setChecked(True)
    assert panel._current_frequency_x_unit == "field_gauss"
    assert panel._frequency_axis_relative_to_reference is True

    # Switching to a correlation spectrum locks the axis to MHz and disables the
    # field-unit selector, the relative checkbox, and the reference spin.
    panel.plot_dataset(correlation)
    assert panel._ax.get_xlabel() == "Muon hyperfine coupling Aμ (MHz)"
    assert panel._frequency_axis_is_correlation is True
    assert panel._frequency_x_unit_combo.isEnabled() is False
    assert panel._frequency_axis_relative_check.isEnabled() is False
    # The relative-axis checkbox must reflect the forced-off backing flag.
    assert panel._frequency_axis_relative_check.isChecked() is False
    assert panel._frequency_reference_spin.isEnabled() is False

    # Switching back restores the user's Field (G) + relative selection exactly.
    panel.plot_dataset(ordinary)
    assert panel._frequency_axis_is_correlation is False
    assert panel._frequency_x_unit_combo.isEnabled() is True
    assert panel._current_frequency_x_unit == "field_gauss"
    assert panel._ax.get_xlabel() == "Field (G)"
    assert panel._frequency_axis_relative_to_reference is True
    assert panel._frequency_axis_relative_check.isChecked() is True

    # A project saved in this (restored) state persists the user's unit, not MHz.
    assert panel.get_state()["frequency_x_unit"] == "field_gauss"


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
