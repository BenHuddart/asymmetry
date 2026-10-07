"""The rotating frame as a projection of vector runs in the main window (synthetic data).

Covers docs/plans/rotating-frame-projection.md Phase 2: the Lab | Rotating
switch, the frame bar's writes into ``ProjectModel.rotating_frames``, the
rotated subplots (rotated before bunching, D8), Auto-detect's review (D6), and
the filtered RRF bar staying out of vector mode (D2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.rebin import rebin
from asymmetry.core.transform.rotating_frame import (
    Provenance,
    RotatingFrame,
    rotate_transverse,
)
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.widgets.rotating_frame_review import FrameReviewDialog
from tests._qt_helpers import wait_for
from tests.core.vector_synthetic import synthetic_vector_run

pytestmark = [pytest.mark.gui]

#: Invented drive: ν_RF (MHz), nutation rate (rad/µs) and RF phase at t0 (°).
NU, W1, PHI = 1.49, 2.0 * math.pi * 0.12, 35.0


def _nutation(t: np.ndarray) -> np.ndarray:
    """(P_x, P_y, P_z) of a nutation about B₁ ∥ x seen in the lab, sense −1."""
    lab = 1j * np.sin(W1 * t) * np.exp(-1j * (2.0 * math.pi * NU * t + math.radians(PHI)))
    return np.stack([lab.real, lab.imag, np.cos(W1 * t)])


def _vector(run_number: int) -> MuonDataset:
    run = synthetic_vector_run(
        [_nutation], amplitudes=(0.1, 0.1, 0.2), rate=4.0e5, n_bins=600, run_number=run_number
    )
    return MuonDataset(
        time=np.zeros(1), asymmetry=np.zeros(1), error=np.ones(1), metadata={}, run=run
    )


def _single_pair(run_number: int) -> MuonDataset:
    t = np.arange(64) * 0.016
    histograms = [Histogram(counts=1000.0 * np.exp(-t / 2.2), bin_width=0.016) for _ in range(2)]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 63,
        "deadtime_correction": False,
    }
    run = Run(run_number=run_number, histograms=histograms, metadata={}, grouping=grouping)
    return MuonDataset(
        time=t, asymmetry=np.zeros_like(t), error=np.full_like(t, 0.01), metadata={}, run=run
    )


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    QSettings().setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    window.resize(1600, 1000)
    return window


def _show(mainwindow: MainWindow, *datasets: MuonDataset) -> PlotPanel:
    with mainwindow._data_browser.batch_updates():
        for dataset in datasets:
            mainwindow._data_browser.add_dataset(dataset)
    panel = mainwindow._plot_panel
    panel.set_overlay_enabled(True, emit_signal=True)
    mainwindow._data_browser.select_runs([ds.run_number for ds in datasets])
    bar = panel._projection_bar
    if bar.frame_available():
        bar.set_selected(["P_x", "P_y", "P_z"])
        bar.selection_changed.emit(bar.selected_labels())
    if not panel._auto_x_btn.isChecked():
        panel._auto_x_btn.click()
    return panel


def _rotate(panel: PlotPanel) -> None:
    panel._projection_bar._rotating_btn.click()


def _type(panel: PlotPanel, name: str, text: str) -> None:
    edit = panel.frame_bar._edits[name]
    edit.setText(text)
    edit.setModified(True)
    edit.editingFinished.emit()


def _frame(**values) -> RotatingFrame:
    return RotatingFrame.typed_frequency(NU, 1).with_values(Provenance.TYPED, **values)


def test_the_switch_is_offered_only_for_groupings_with_the_transverse_pair(mainwindow):
    vector, plain = _vector(901), _single_pair(902)
    panel = _show(mainwindow, vector)
    assert panel._projection_bar.frame_available()
    assert not panel._projection_bar._frame_switch.isHidden()
    _rotate(panel)
    assert [chip.text() for chip in panel._projection_bar._chips.values()] == [
        "P′_x",
        "P′_y",
        "P_z",
    ]
    mainwindow._data_browser.add_dataset(plain)
    mainwindow._data_browser.select_runs([902])
    assert not panel.frame_rotating()
    assert panel.frame_bar.isHidden()


def test_the_first_switch_asks_for_frequency_and_typing_it_creates_the_frames(mainwindow):
    panel = _show(mainwindow, _vector(903), _vector(904))
    _rotate(panel)
    assert not panel.frame_bar.isHidden()
    assert [text.get_text() for text in panel._ax.texts] == [
        "Enter ν_RF to show the rotating frame."
    ]
    assert mainwindow._project_model.rotating_frames == {}

    _type(panel, "frequency_mhz", "1.49")

    frames = mainwindow._project_model.rotating_frames
    assert frames == {
        903: RotatingFrame.typed_frequency(1.49, 1),
        904: RotatingFrame.typed_frequency(1.49, 1),
    }
    assert list(panel._subplot_axes_by_polarization) == ["P′_x", "P′_y", "P_z"]
    assert mainwindow._dirty


def test_rotated_subplots_are_the_exact_rotation_of_each_run(mainwindow):
    dataset = _vector(905)
    mainwindow._project_model.rotating_frames[905] = _frame(rf_phase_deg=PHI)
    panel = _show(mainwindow, dataset)
    _rotate(panel)

    built = mainwindow._build_vector_axis_datasets([dataset], ["P_x", "P_y"])
    expected = rotate_transverse(
        built["P_x"][0], built["P_y"][0], _frame(rf_phase_deg=PHI), weights=(1.0,)
    )
    shown = panel._vector_subplot_datasets
    for label, rotated in zip(("P′_x", "P′_y"), expected, strict=True):
        np.testing.assert_allclose(shown[label][0].asymmetry, rotated.asymmetry)
        np.testing.assert_allclose(shown[label][0].error, rotated.error)
    # The nutation leaves +z for +y′ and stays off x′.
    t = shown["P′_y"][0].time
    early = (t > 0.5) & (t < 2.5)
    assert np.mean(shown["P′_y"][0].asymmetry[early]) > 3.0
    assert abs(np.mean(shown["P′_x"][0].asymmetry[early])) < 0.5
    top = panel._subplot_axes_by_polarization["P′_x"]
    assert top.get_title(loc="right") == (
        "Rotating frame · ν = 1.49 MHz · B₁ ∥ x′ · φ_RF = 35.0° · s = −1"
    )
    assert r"\parallel B_1" in top.get_ylabel()


def test_bunching_rebins_the_rotated_values(mainwindow):
    mainwindow._project_model.rotating_frames[906] = _frame(rf_phase_deg=PHI)
    panel = _show(mainwindow, _vector(906))
    _rotate(panel)
    rotated = panel._vector_subplot_datasets["P′_y"][0]
    panel.set_bunch_factor(4)

    bunched = panel.get_analysis_dataset(panel._vector_subplot_datasets["P′_y"][0])
    time, values, errors = rebin(rotated.time, rotated.asymmetry, rotated.error, 4)
    np.testing.assert_allclose(bunched.time, time)
    np.testing.assert_allclose(bunched.asymmetry, values)
    np.testing.assert_allclose(bunched.error, errors)


def _ctrl_click(mainwindow: MainWindow, run_number: int) -> None:
    """Add *run_number* to the browser selection and make it the current row."""
    browser = mainwindow._data_browser
    table = browser._table
    row = next(
        r
        for r in range(table.rowCount())
        if table.item(r, 0).data(browser._GROUP_ROLE) == run_number
    )
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    table.selectionModel().setCurrentIndex(table.model().index(row, 0), flags)


def test_setup_edits_reach_every_run_and_run_edits_only_the_selected_one(mainwindow):
    panel = _show(mainwindow, _vector(907), _vector(908))
    _rotate(panel)
    mainwindow._data_browser.select_runs([907])
    _ctrl_click(mainwindow, 908)
    assert panel.frame_bar.runs() == [907, 908]
    assert panel.frame_bar._run_header.text() == "RUN 908"
    _type(panel, "frequency_mhz", "1.49")

    _type(panel, "rf_phase_deg", "20")
    _type(panel, "baseline_x", "0.4")

    frames = mainwindow._project_model.rotating_frames
    assert frames[907].rf_phase_deg == frames[908].rf_phase_deg == 20.0
    assert frames[908].baselines[0].x == 0.4
    assert frames[908].baselines[0].provenance is Provenance.TYPED
    assert frames[907].baselines[0].x == 0.0
    assert frames[907].baselines[0].provenance is Provenance.DEFAULT

    frames[907] = frames[907].with_values(Provenance.TYPED, rf_phase_deg=50.0)
    mainwindow._render_current_selection_plot()
    phase = panel.frame_bar._edits["rf_phase_deg"]
    assert phase.text() == "" and phase.placeholderText() == "mixed"
    assert "φ_RF mixed" in panel._subplot_axes_by_polarization["P′_x"].get_title(loc="right")


def test_auto_detect_proposes_and_apply_writes_estimates(mainwindow, qapp):
    panel = _show(mainwindow, _vector(909))
    _rotate(panel)
    _type(panel, "frequency_mhz", "1.49")
    _type(panel, "gain", "1.0")  # typed: Auto-detect must not pre-tick its row

    panel.frame_bar._detect_btn.click()
    wait_for(lambda: mainwindow.findChildren(FrameReviewDialog), qapp, timeout_s=20.0)
    dialog = mainwindow.findChild(FrameReviewDialog)
    ticks = {tick.text(): tick.isChecked() for tick, _row in dialog._ticks}
    assert ticks == {"Sense, φ_RF": True, "Run 909 · gain": False, "Run 909": True}
    assert mainwindow._project_model.rotating_frames[909].rf_phase_deg == 0.0

    dialog._apply_btn.click()

    frame = mainwindow._project_model.rotating_frames[909]
    assert frame.rf_phase_deg == pytest.approx(PHI, abs=5.0)
    assert frame.sense == -1
    assert frame.provenance["rf_phase_deg"] is Provenance.ESTIMATED
    assert frame.provenance["gain"] is Provenance.TYPED
    assert frame.baselines[0].provenance is Provenance.ESTIMATED
    assert panel._frame_status_label.text().startswith("✓ Applied 2 estimates · contrast ")


#: Each period's own transverse baselines (%): red, then green (D9).
RED_BASELINES, GREEN_BASELINES = (0.4, -0.3), (0.1, 0.05)


def _with_baselines(polarisation, baselines):
    def shifted(t):
        p = polarisation(t)
        return p + np.array([[baselines[0] / 10.0], [baselines[1] / 10.0], [0.0]])

    return shifted


def _still(t: np.ndarray) -> np.ndarray:
    return np.stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])


def _two_period(run_number: int, period_mode: PeriodMode) -> MuonDataset:
    """RF on (red, nutating) and RF off (green, still), each with its own baselines."""
    run = synthetic_vector_run(
        [_with_baselines(_nutation, RED_BASELINES), _with_baselines(_still, GREEN_BASELINES)],
        amplitudes=(0.1, 0.1, 0.2),
        rate=4.0e5,
        n_bins=600,
        run_number=run_number,
    )
    run.grouping["period_mode"] = str(period_mode)
    return MuonDataset(
        time=np.zeros(1), asymmetry=np.zeros(1), error=np.ones(1), metadata={}, run=run
    )


def test_green_minus_red_removes_the_difference_of_the_period_baselines(mainwindow):
    frame = (
        RotatingFrame.typed_frequency(NU, 2)
        .with_values(Provenance.TYPED, rf_phase_deg=PHI)
        .with_baseline(0, Provenance.TYPED, *RED_BASELINES)
        .with_baseline(1, Provenance.TYPED, *GREEN_BASELINES)
    )
    mainwindow._project_model.rotating_frames[914] = frame
    dataset = _two_period(914, PeriodMode.GREEN_MINUS_RED)
    panel = _show(mainwindow, dataset)
    _rotate(panel)

    built = mainwindow._build_vector_axis_datasets([dataset], ["P_x", "P_y"])
    expected = rotate_transverse(built["P_x"][0], built["P_y"][0], frame, weights=(-1.0, 1.0))
    shown = panel._vector_subplot_datasets
    for label, rotated in zip(("P′_x", "P′_y"), expected, strict=True):
        np.testing.assert_allclose(shown[label][0].asymmetry, rotated.asymmetry)
    # The RF-off period carries no transverse signal: green − red turns the
    # red nutation over, into −y′, rather than φ_RF hiding it.
    t = shown["P′_y"][0].time
    early = (t > 0.5) & (t < 3.0)
    assert np.mean(shown["P′_y"][0].asymmetry[early]) < -1.0
    # A combination's baselines are derived, so the bar shows them read-only.
    b_x = panel.frame_bar._edits["baseline_x"]
    assert not b_x.isEnabled()
    assert float(b_x.text()) == pytest.approx(GREEN_BASELINES[0] - RED_BASELINES[0])


def test_auto_detect_reads_every_period_whatever_the_display_combines(mainwindow, qapp):
    panel = _show(mainwindow, _two_period(915, PeriodMode.GREEN_MINUS_RED))
    _rotate(panel)
    _type(panel, "frequency_mhz", str(NU))

    panel.frame_bar._detect_btn.click()
    wait_for(lambda: mainwindow.findChildren(FrameReviewDialog), qapp, timeout_s=20.0)
    dialog = mainwindow.findChild(FrameReviewDialog)
    assert {tick.text() for tick, _row in dialog._ticks} == {
        "Sense, φ_RF",
        "Run 915 · gain",
        "Run 915 · red",
        "Run 915 · green",
    }
    dialog._apply_btn.click()

    frame = mainwindow._project_model.rotating_frames[915]
    # φ_RF comes from the periods themselves, not from the flipped difference.
    assert abs((frame.rf_phase_deg - PHI + 180.0) % 360.0 - 180.0) < 5.0
    for baseline, truth in zip(frame.baselines, (RED_BASELINES, GREEN_BASELINES), strict=True):
        assert baseline.x == pytest.approx(truth[0], abs=0.05)
        assert baseline.y == pytest.approx(truth[1], abs=0.05)
        assert baseline.provenance is Provenance.ESTIMATED


def test_a_period_split_off_a_many_period_file_gets_a_one_period_frame(mainwindow):
    dataset = _vector(916)
    dataset.run.metadata["period_count"] = 4
    panel = _show(mainwindow, dataset)
    _rotate(panel)
    _type(panel, "frequency_mhz", str(NU))
    assert len(mainwindow._project_model.rotating_frames[916].baselines) == 1
    assert list(panel._subplot_axes_by_polarization) == ["P′_x", "P′_y", "P_z"]


def test_the_filtered_rrf_bar_stays_out_of_vector_mode(mainwindow):
    panel = _show(mainwindow, _vector(910))
    panel.set_rrf_feature_enabled(True)
    assert panel._rrf_controls.isHidden()
    _rotate(panel)
    assert panel._rrf_controls.isHidden() and not panel.frame_bar.isHidden()
    mainwindow._data_browser.add_dataset(_single_pair(911))
    mainwindow._data_browser.select_runs([911])
    assert not panel._rrf_controls.isHidden() and panel.frame_bar.isHidden()


def test_a_transverse_grouping_without_the_pair_keeps_the_filtered_bar(mainwindow):
    # HiFi-style transverse-vector projections are not P_x/P_y: no exact frame is
    # offered, so the filtered rotating frame stays available.
    dataset = _vector(913)
    dataset.run.grouping["projections"] = [
        {"label": "Left-Right", "forward_group": 5, "backward_group": 6},
        {"label": "Top-Bottom", "forward_group": 3, "backward_group": 4},
    ]
    panel = _show(mainwindow, dataset)
    panel.set_rrf_feature_enabled(True)
    assert not panel.offers_rotating_frame()
    assert not panel._rrf_controls.isHidden()


def test_lab_fit_overlays_stay_off_rotated_subplots(mainwindow):
    mainwindow._project_model.rotating_frames[912] = _frame()
    panel = _show(mainwindow, _vector(912))
    curve = (np.linspace(0.0, 5.0, 50), np.ones(50), "Fit")
    panel._fit_curves_by_key[(912, "P_x", "single")] = curve
    panel._fit_curves[912] = curve

    def fit_lines(label: str) -> int:
        axis = panel._subplot_axes_by_polarization[label]
        return sum(line.get_linewidth() == 2 for line in axis.lines)

    panel._redraw_current_view()
    assert fit_lines("P_x") == 1
    _rotate(panel)
    assert fit_lines("P′_x") == 0 and fit_lines("P′_y") == 0


def test_the_frame_choice_round_trips_through_plot_state(mainwindow, qapp):
    panel = _show(mainwindow, _vector(914))
    _rotate(panel)
    panel.frame_bar._unit_group.button(1).click()
    state = panel.get_state()
    assert state["rotating_frame"] == {"rotating": True, "frequency_unit": "gauss"}

    restored = PlotPanel()
    restored.restore_state(state)
    assert restored._projection_bar.rotating_chosen()
    assert restored.frame_bar.frequency_unit().value == "gauss"
    restored.restore_state({key: value for key, value in state.items() if key != "rotating_frame"})
    assert not restored._projection_bar.rotating_chosen()


def test_the_rotating_frame_never_widens_the_plot_panel(mainwindow):
    mainwindow._project_model.rotating_frames[915] = _frame()
    panel = _show(mainwindow, _vector(915))
    lab = panel.minimumSizeHint().width()
    _rotate(panel)
    assert panel.minimumSizeHint().width() == lab
    assert panel.frame_bar.minimumSizeHint().width() <= lab
    panel.set_frame_status("✓ Applied 4 estimates · contrast 18")
    assert panel.minimumSizeHint().width() == lab
