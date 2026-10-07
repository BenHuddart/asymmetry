"""Fitting the rotated projections P′_x and P′_y (synthetic data).

docs/plans/rotating-frame-projection.md D7: a rotated subplot is a Single and
Batch fit target like any projection; the fit records the frame its data were
rotated in and reads stale once the run's frame no longer gives that data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import RepresentationType
from asymmetry.core.transform.projections import reduction_identity
from asymmetry.core.transform.rotating_frame import FrameSnapshot, rotate_transverse
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.plot_panel import SINGLE_FIT_ID, PlotPanel
from tests.gui.test_rotating_frame_projection import (
    PHI,
    W1,
    _ctrl_click,
    _frame,
    _rotate,
    _show,
    _type,
    _vector,
)

pytestmark = [pytest.mark.gui]

_FB = RepresentationType.TIME_FB_ASYMMETRY
#: A nutation on a constant: what P′_y of the synthetic runs shows.
_NUTATION_MODEL = CompositeModel(["Oscillatory", "Constant"], operators=["+"])
_NU1 = W1 / (2.0 * math.pi)


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    QSettings().setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    window.resize(1600, 1000)
    return window


def _rotated_target(mainwindow: MainWindow, *runs: int) -> PlotPanel:
    """Show *runs* (each with a frame) rotated, with P′_y the fit target."""
    for run in runs:
        mainwindow._project_model.rotating_frames[run] = _frame(rf_phase_deg=PHI)
    panel = _show(mainwindow, *[_vector(run) for run in runs])
    _rotate(panel)
    panel.set_fit_target_projection("P′_y")
    panel.set_fit_range(0.05, 9.0)
    return panel


def _seed_nutation(tab) -> None:
    tab._set_composite_model(_NUTATION_MODEL)
    seeds = {"A_1": 8.0, "frequency": 0.11, "phase": -math.pi / 2.0, "A_bg": 0.0}
    tab._param_table.restore_parameters(
        {name: {"name": name, "value": value} for name, value in seeds.items()}
    )


def _fit_single(mainwindow: MainWindow) -> None:
    tab = mainwindow._fit_panel._single_tab
    _seed_nutation(tab)
    tab._run_fit()
    assert tab.wait_for_fit()


def _fitted(mainwindow: MainWindow, run: int, projection: str):
    return mainwindow._project_model.representation(run, _FB).fit_for(projection)


def test_a_rotated_subplot_is_the_fit_target_and_its_data_are_the_drawn_curve(mainwindow):
    panel = _rotated_target(mainwindow, 920)
    panel.set_fit_range(0.2, 6.0)

    assert mainwindow._current_fit_block_state() == (False, "")
    assert mainwindow._current_single_fit_projection() == "P′_y"
    fitted = mainwindow._fit_panel.single_dataset()
    drawn = panel._vector_subplot_datasets["P′_y"][0].time_range(0.2, 6.0)
    np.testing.assert_array_equal(fitted.time, drawn.time)
    np.testing.assert_array_equal(fitted.asymmetry, drawn.asymmetry)
    np.testing.assert_array_equal(fitted.error, drawn.error)
    # … which is the exact rotation of the run's lab pair in its frame.
    dataset = mainwindow._data_browser.get_dataset(920)
    built = mainwindow._build_vector_axis_datasets([dataset], ["P_x", "P_y"])
    _px, py = rotate_transverse(
        built["P_x"][0], built["P_y"][0], _frame(rf_phase_deg=PHI), weights=(1.0,)
    )
    np.testing.assert_allclose(fitted.asymmetry, py.time_range(0.2, 6.0).asymmetry)
    assert fitted.metadata[mw_module.FRAME_METADATA_KEY] == FrameSnapshot(
        _frame(rf_phase_deg=PHI), (1.0,), reduction_identity(dataset.run.grouping)
    )

    panel.set_fit_target_projection("P_z")
    assert mainwindow._current_single_fit_projection() == "P_z"
    assert mw_module.FRAME_METADATA_KEY not in mainwindow._fit_panel.single_dataset().metadata


def test_a_damped_sine_on_p_prime_y_finds_the_nutation_and_records_its_frame(mainwindow):
    panel = _rotated_target(mainwindow, 921)
    _fit_single(mainwindow)

    slot = _fitted(mainwindow, 921, "P′_y")
    assert slot.result["parameters"]["frequency"] == pytest.approx(_NU1, rel=0.02)
    grouping = mainwindow._data_browser.get_dataset(921).run.grouping
    assert slot.frame_snapshot == FrameSnapshot(
        _frame(rf_phase_deg=PHI), (1.0,), reduction_identity(grouping)
    )
    assert _fitted(mainwindow, 921, "P_y").is_empty()
    # The curve overlays P′_y only.
    assert (921, "P′_y", SINGLE_FIT_ID) in panel._fit_curves_by_key

    def fit_lines(label: str) -> int:
        return sum(
            line.get_linewidth() == 2 for line in panel._subplot_axes_by_polarization[label].lines
        )

    panel._redraw_current_view()
    assert (fit_lines("P′_x"), fit_lines("P′_y"), fit_lines("P_z")) == (0, 1, 0)


def test_editing_the_frame_makes_the_fit_stale_and_refitting_refreshes_it(mainwindow):
    panel = _rotated_target(mainwindow, 922)
    _fit_single(mainwindow)
    tab = mainwindow._fit_panel._single_tab
    assert mainwindow._saved_fit_catalogue().open_entry().stale_reason == ""

    _type(panel, "gain", "1.0")  # the same value, now typed: provenance alone
    assert mainwindow._saved_fit_catalogue().open_entry().stale_reason == ""

    _type(panel, "rf_phase_deg", "50")
    assert mainwindow._saved_fit_catalogue().open_entry().stale_reason == mw_module._FRAME_STALE
    assert tab._saved_fit_hint.text() == (
        "Fitted in a different rotating frame — re-run the fit to update it."
    )
    # The Single tab now holds the curve in the new frame.
    snapshot = mainwindow._fit_panel.single_dataset().metadata[mw_module.FRAME_METADATA_KEY]
    assert snapshot.frame.rf_phase_deg == 50.0

    _fit_single(mainwindow)
    assert _fitted(mainwindow, 922, "P′_y").frame_snapshot == snapshot
    assert mainwindow._saved_fit_catalogue().open_entry().stale_reason == ""
    assert tab._saved_fit_hint.isHidden()


def test_a_new_reduction_of_the_pair_makes_the_fit_stale(mainwindow):
    # Regrouping or a new alpha gives other P_x and P_y (review of #357).
    _rotated_target(mainwindow, 929)
    _fit_single(mainwindow)
    assert mainwindow._saved_fit_catalogue().open_entry().stale_reason == ""
    mainwindow._data_browser.get_dataset(929).run.grouping["alpha_x"] = 1.2
    assert mainwindow._saved_fit_catalogue().open_entry().stale_reason == mw_module._FRAME_STALE


def test_lab_and_rotated_fits_live_side_by_side_on_one_run(mainwindow):
    panel = _rotated_target(mainwindow, 923)
    _fit_single(mainwindow)
    panel._projection_bar._lab_btn.click()
    panel.set_fit_target_projection("P_y")
    assert mainwindow._current_single_fit_projection() == "P_y"
    _fit_single(mainwindow)

    lab, rotated = _fitted(mainwindow, 923, "P_y"), _fitted(mainwindow, 923, "P′_y")
    assert lab.frame_snapshot is None and rotated.frame_snapshot is not None
    assert lab.fit_id != rotated.fit_id
    assert {(923, "P_y", SINGLE_FIT_ID), (923, "P′_y", SINGLE_FIT_ID)} <= set(
        panel._fit_curves_by_key
    )


def _batch_fit(mainwindow: MainWindow) -> str:
    """Batch-fit the Batch tab's members, seeded from a single fit sent across."""
    _fit_single(mainwindow)
    assert mainwindow._fit_panel.send_single_model_to_batch()
    tab = mainwindow._fit_panel._global_tab
    tab._run_global_fit()
    assert tab.wait_for_fit()
    return mainwindow._project_model.active_series_id(_FB)


def test_a_rotated_series_records_each_members_frame_and_goes_stale_with_one(mainwindow, qapp):
    panel = _rotated_target(mainwindow, 924, 925)
    pool = mainwindow._fit_panel.batch_datasets()
    assert [ds.metadata["projection"] for ds in pool] == ["P′_y", "P′_y"]

    series = mainwindow._project_model.batch(_batch_fit(mainwindow))
    assert series.projection == "P′_y"
    assert set(series.member_frames) == {924, 925}
    for run in (924, 925):
        assert series.results_by_run[run]["parameters"]["frequency"] == pytest.approx(
            _NU1, rel=0.03
        )
    assert mainwindow._series_stale_reason(series, None) == ""

    mainwindow._data_browser.select_runs([924])
    _ctrl_click(mainwindow, 925)
    _type(panel, "gain", "0.9")  # a run field: run 925 only
    assert mainwindow._project_model.rotating_frames[924].gain == 1.0
    assert mainwindow._series_stale_reason(series, None) == mw_module._FRAME_STALE
    pills = mainwindow._fit_parameters_panel._stale_series_reasons
    assert pills[series.batch_id] == mw_module._FRAME_STALE

    # The open series' members were re-rotated, so re-running it refreshes it.
    assert mainwindow._fit_panel.open_series_id() == series.batch_id
    tab = mainwindow._fit_panel._global_tab
    tab._run_global_fit()
    assert tab.wait_for_fit()
    assert mainwindow._project_model.active_series_id(_FB) == series.batch_id
    assert series.member_frames[925].frame.gain == 0.9
    assert mainwindow._series_stale_reason(series, None) == ""


def test_a_lab_frame_series_records_its_projection(mainwindow):
    # The same recipe on another lab subplot is another series (review of #357).
    panel = _show(mainwindow, _vector(932), _vector(933))
    panel.set_fit_target_projection("P_z")
    panel.set_fit_range(0.05, 9.0)
    series = mainwindow._project_model.batch(_batch_fit(mainwindow))
    assert series.projection == "P_z"
    assert series.member_frames == {}


def test_a_batch_draft_follows_the_fit_target_into_the_rotating_frame(mainwindow):
    for run in (930, 931):
        mainwindow._project_model.rotating_frames[run] = _frame(rf_phase_deg=PHI)
    panel = _show(mainwindow, _vector(930), _vector(931))
    panel.set_fit_range(0.05, 9.0)
    panel.set_fit_target_projection("P_z")
    browser = mainwindow._data_browser.get_dataset(930).time_range(0.05, 9.0)
    pool = mainwindow._fit_panel.batch_datasets()
    np.testing.assert_array_equal(pool[0].asymmetry, browser.asymmetry)

    _rotate(panel)
    panel.set_fit_target_projection("P′_x")
    pool = mainwindow._fit_panel.batch_datasets()
    assert [ds.metadata["projection"] for ds in pool] == ["P′_x", "P′_x"]
    drawn = panel._vector_subplot_datasets["P′_x"][1].time_range(0.05, 9.0)
    np.testing.assert_array_equal(pool[1].asymmetry, drawn.asymmetry)


def test_a_run_without_a_frame_cannot_join_a_rotated_series(mainwindow):
    _rotated_target(mainwindow, 926, 927)
    mainwindow._data_browser.add_dataset(_vector(928))
    runs = [926, 927, 928]

    datasets = mainwindow._batch_datasets_for_runs(runs, None, _FB, "P′_y")
    unavailable = mainwindow._unavailable_members(runs, "P′_y")
    assert [ds.run_number for ds in datasets] == [926, 927]
    assert unavailable == {928: "No rotating frame — set ν_RF in the rotating view"}
    assert mainwindow._unavailable_members(runs, "P_y") == {}

    mainwindow._fit_panel.open_draft(datasets=datasets, unavailable=unavailable)
    members = mainwindow._fit_panel._global_tab._members_list
    rows = [members.item(row) for row in range(members.count())]
    assert [row.text() for row in rows] == ["926", "927", "928"]
    assert not rows[2].flags() & Qt.ItemFlag.ItemIsEnabled
    assert rows[2].toolTip() == unavailable[928]
    assert [ds.run_number for ds in mainwindow._fit_panel.batch_datasets()] == [926, 927]
