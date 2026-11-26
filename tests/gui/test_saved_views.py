"""Drive Saved views through the project toolbar with loaded data."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.project import save_project
from asymmetry.core.project.view_migration import diagnostic_axis_id
from asymmetry.core.simulate import BUILTIN_TEMPLATES, simulate_run
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.shell import ProjectShell
from tests._qt_helpers import wait_for

pytestmark = pytest.mark.gui


@pytest.fixture
def mainwindow(qapp):
    return MainWindow()


@pytest.fixture
def source(tmp_path):
    pytest.importorskip("h5py")
    template = replace(
        BUILTIN_TEMPLATES["ideal_pulsed_fb"],
        n_bins=240,
        bin_width_us=0.02,
        n_detectors=2,
        forward_detectors=(1,),
        backward_detectors=(2,),
        t0_bin=0,
    ).build()
    run = simulate_run(
        template, lambda t: 16 * np.cos(2 * np.pi * t), total_events=3e6, seed=17, run_number=4103
    )
    path = tmp_path / "run.nxs"
    write_nexus_v1(run, path)
    return str(path)


def _hold(panel, axis, limits):
    minimum, maximum = {"x": (panel._x_min, panel._x_max), "y": (panel._y_min, panel._y_max)}[axis]
    minimum.setValue(limits[0])
    maximum.setValue(limits[1])
    maximum.editingFinished.emit()


def _auto(panel, x, y):
    for button, value in [(panel._auto_x_btn, x), (panel._auto_y_btn, y)]:
        if button.isChecked() != value:
            button.click()


def test_view_modes_restore_per_mode_limits_and_bunch(mainwindow: MainWindow) -> None:
    """Visit and revisit presets with two runs of different lengths and amplitudes."""
    panel = mainwindow._plot_panel
    for run, extent, amplitude in [(4101, 2.0, 6.0), (4102, 5.0, 20.0)]:
        time = np.linspace(0.0, extent, 120)
        mainwindow._data_browser.add_dataset(
            MuonDataset(time, amplitude * np.cos(time), np.ones_like(time), {"run_number": run})
        )
    mainwindow._data_browser.select_runs({4101})
    mainwindow._view_bunch_spin.setValue(2)
    _hold(panel, "x", (0.25, 1.75))
    _auto(panel, False, True)
    first_y = panel._ax.get_ylim()

    first, second, third = mainwindow._view_mode_buttons
    second.click()
    assert mainwindow._view_bunch_spin.value() == 2
    assert panel._ax.get_xlim() == pytest.approx((0.25, 1.75))
    assert panel._auto_y_btn.isChecked()

    _auto(panel, True, False)
    _hold(panel, "y", (-4.0, 8.0))
    mainwindow._view_bunch_spin.setValue(3)
    second_x = panel._ax.get_xlim()

    third.click()
    assert mainwindow._view_bunch_spin.value() == 3
    assert panel._auto_x_btn.isChecked()
    assert panel._ax.get_ylim() == pytest.approx((-4.0, 8.0))
    mainwindow._view_bunch_spin.setValue(5)

    second.click()
    assert mainwindow._view_bunch_spin.value() == 3
    first.click()
    assert mainwindow._view_bunch_spin.value() == 2
    assert not panel._auto_x_btn.isChecked()
    assert panel._auto_y_btn.isChecked()

    mainwindow._data_browser.select_runs({4102})
    assert panel._ax.get_xlim() == pytest.approx((0.25, 1.75))
    assert panel._ax.get_ylim() != pytest.approx(first_y)
    third.click()
    assert mainwindow._view_bunch_spin.value() == 5
    assert panel._auto_x_btn.isChecked()
    assert not panel._auto_y_btn.isChecked()
    assert panel._ax.get_xlim()[1] > second_x[1] * 2
    assert panel._ax.get_ylim() == pytest.approx((-4.0, 8.0))


@pytest.mark.real_save_guard
def test_save_and_reopen_active_and_inactive_views(mainwindow, source, qapp, tmp_path, monkeypatch):
    mainwindow._load_files([source])
    panel = mainwindow._plot_panel
    _auto(panel, True, False)
    _hold(panel, "y", (-12.0, 19.0))
    mainwindow._view_bunch_spin.setValue(2)
    mainwindow._view_mode_buttons[2].click()
    _auto(panel, False, True)
    _hold(panel, "x", (0.5, 2.75))
    mainwindow._view_bunch_spin.setValue(4)
    panel._set_fit_range(1.0, 2.0, emit_signal=True, redraw=True)
    path = tmp_path / "views.asymp"
    mainwindow._write_project(str(path))
    wait_for(lambda: not mainwindow._project_save_active, qapp, timeout_s=10)
    assert path.is_file() and not mainwindow._dirty

    reopened = MainWindow()
    reopened._open_project_file(str(path))
    panel = reopened._plot_panel
    assert not reopened._dirty
    assert reopened._view_mode_buttons[2].isChecked()
    assert reopened._view_bunch_spin.value() == 4
    assert panel._ax.get_xlim() == pytest.approx((0.5, 2.75))
    assert not panel._auto_x_btn.isChecked() and panel._auto_y_btn.isChecked()
    assert panel.get_fit_range() == pytest.approx((1.0, 2.0))
    reopened._render_current_selection_plot()
    assert not reopened._dirty

    prompts = []

    def warning(_parent, title, *args):
        prompts.append(title)
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "warning", warning)
    assert reopened._maybe_save("closing")
    assert prompts == []
    reopened._view_mode_buttons[0].click()
    assert reopened._dirty and not reopened._maybe_save("closing")
    assert prompts == ["Unsaved Changes"]
    assert reopened._view_bunch_spin.value() == 2
    assert panel._ax.get_ylim() == pytest.approx((-12.0, 19.0))
    assert panel._auto_x_btn.isChecked() and not panel._auto_y_btn.isChecked()
    assert panel.get_fit_range() == pytest.approx((1.0, 2.0))
    reopened._view_mode_buttons[2].click()
    reopened._view_mode_buttons[1].click()
    assert reopened._view_bunch_spin.value() == 4
    assert panel._ax.get_xlim() == pytest.approx((0.5, 2.75))
    assert panel._auto_y_btn.isChecked()
    reopened._clear_dirty()


def test_editing_presets_marks_project_dirty_and_new_projects_reset(
    mainwindow, source, monkeypatch
):
    mainwindow._load_files([source])
    panel = mainwindow._plot_panel
    for edit in (
        lambda: mainwindow._view_mode_buttons[1].click(),
        lambda: _hold(panel, "x", (0.5, 1.5)),
        panel._auto_y_btn.click,
        lambda: mainwindow._view_bunch_spin.setValue(3),
    ):
        mainwindow._clear_dirty()
        mainwindow._render_current_selection_plot()
        assert not mainwindow._dirty
        edit()
        assert mainwindow._dirty

    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Discard)
    mainwindow._on_new_project()
    assert mainwindow._view_mode_buttons[0].isChecked()
    assert mainwindow._view_bunch_spin.value() == 1
    assert not mainwindow._plot_panel._auto_y_btn.isChecked()
    assert not mainwindow._dirty
    mainwindow._load_files([source])
    mainwindow._view_mode_buttons[1].click()
    assert mainwindow._view_bunch_spin.value() == 1
    assert mainwindow._plot_panel._ax.get_xlim()[1] > 4.0

    shell = ProjectShell()
    first = shell.add_project()
    first._load_files([source])
    first._view_mode_buttons[2].click()
    first._view_bunch_spin.setValue(5)
    first._on_new_project()
    second = shell.current_page()
    assert second is not first
    assert second._view_mode_buttons[0].isChecked()
    assert second._view_bunch_spin.value() == 1
    assert not second._dirty
    assert first._view_mode_buttons[2].isChecked()
    assert first._view_bunch_spin.value() == 5
    first._clear_dirty()


def test_non_fb_views_cannot_recall_presets(mainwindow, source, qapp):
    mainwindow._load_files([source])
    panel = mainwindow._plot_panel
    _hold(panel, "x", (0.5, 2.5))
    _auto(panel, False, True)
    first, second, _ = mainwindow._view_mode_buttons
    second.click()
    mainwindow._view_bunch_spin.setValue(4)
    panel._set_fit_range(0.75, 1.75, emit_signal=True, redraw=True)
    grouping = deepcopy(mainwindow._current_dataset.run.grouping)
    mainwindow._on_compute_fourier()
    wait_for(lambda: not mainwindow._fourier_compute_active, qapp, timeout_s=10)
    assert mainwindow._frequency_plot_panel.has_plot_content()

    views = mainwindow._domain_buttons_by_token
    for control, view, bunch_available in (
        (views["frequency"].click, "frequency", False),
        (views["maxent"].click, "maxent", False),
        (views["integral_scan"].click, "integral_scan", False),
        (views["groups"].click, "groups", True),
        (mainwindow._raw_counts_action.trigger, "raw_counts", True),
    ):
        control()
        assert mainwindow._plot_workspace.active_view() == view
        assert not any(button.isEnabled() for button in mainwindow._view_mode_buttons)
        assert mainwindow._view_bunch_spin.isEnabled() == bunch_available
        mainwindow._clear_dirty()
        plots = (panel, mainwindow._frequency_plot_panel)
        axes = [ax for plot in plots for ax in plot._figure.axes]
        axes.append(mainwindow._alc_scan_view._ax)
        before = [(axis.get_xlim(), axis.get_ylim()) for axis in axes]
        factor = panel.get_bunch_factor()
        first.click()
        mainwindow._on_view_mode_button_clicked(0)
        assert [(axis.get_xlim(), axis.get_ylim()) for axis in axes] == before
        assert panel.get_bunch_factor() == factor
        assert second.isChecked()
        assert not mainwindow._dirty

    mainwindow._view_bunch_spin.setValue(2)
    assert panel.get_bunch_factor() == 2
    views["fb_asymmetry"].click()
    assert second.isChecked() and second.isEnabled()
    assert mainwindow._view_bunch_spin.value() == 4
    assert panel._ax.get_xlim() == pytest.approx((0.5, 2.5))
    assert panel._auto_y_btn.isChecked()
    assert panel.get_fit_range() == pytest.approx((0.75, 1.75))
    assert mainwindow._current_dataset.run.grouping == grouping


def test_legacy_numeric_views_reopen_with_the_active_plot_policy(mainwindow, source, tmp_path):
    mainwindow._load_files([source])
    panel = mainwindow._plot_panel
    _hold(panel, "x", (0.5, 2.5))
    _auto(panel, False, True)
    state = mainwindow.collect_project_state()
    state["schema_version"] = 42
    state["view_modes_state"] = {
        "active_index": 2,
        "modes": [
            {"bunch_factor": 2, "x_min": 0.25, "x_max": 1.5, "y_min": -8, "y_max": 22},
            {"bunch_factor": 1, "x_min": 0, "x_max": 10, "y_min": -30, "y_max": 30},
            {"bunch_factor": 4, "x_min": 0.5, "x_max": 2.5, "y_min": -20, "y_max": 20},
        ],
    }
    path = tmp_path / "legacy.asymp"
    save_project(state, path)
    reopened = MainWindow()
    reopened._open_project_file(str(path))
    panel = reopened._plot_panel
    first, second, third = reopened._view_mode_buttons
    assert third.isChecked()
    assert reopened._view_bunch_spin.value() == 4
    assert panel._ax.get_xlim() == pytest.approx((0.5, 2.5))
    assert panel._auto_y_btn.isChecked()
    second.click()
    assert reopened._view_bunch_spin.value() == 4
    assert panel._auto_y_btn.isChecked()
    first.click()
    assert reopened._view_bunch_spin.value() == 2
    assert panel._ax.get_xlim() == pytest.approx((0.25, 1.5))
    assert panel._ax.get_ylim() == pytest.approx((-8, 22))
    assert not panel._auto_y_btn.isChecked()


def test_saved_views_recall_each_projection_limits(mainwindow):
    from tests.gui.test_mainwindow_additional import _make_vector_dataset

    dataset = _make_vector_dataset(8801)
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({8801})
    panel = mainwindow._plot_panel
    panel._projection_bar._chips["P_x"].click()
    assert panel._subplot_axes_by_polarization.keys() == {"P_x", "P_z"}
    for projection, limits in [("P_x", (-4.0, 12.0)), ("P_z", (-2.0, 18.0))]:
        panel.set_fit_target_projection(projection)
        _hold(panel, "y", limits)
    first, second, _ = mainwindow._view_mode_buttons
    second.click()
    panel.set_fit_target_projection("P_x")
    _hold(panel, "y", (-20.0, 30.0))

    first.click()
    assert panel._subplot_axes_by_polarization["P_x"].get_ylim() == pytest.approx((-4, 12))
    assert panel._subplot_axes_by_polarization["P_z"].get_ylim() == pytest.approx((-2, 18))
    second.click()
    assert panel._subplot_axes_by_polarization["P_x"].get_ylim() == pytest.approx((-20, 30))
    assert panel._subplot_axes_by_polarization["P_z"].get_ylim() == pytest.approx((-2, 18))


def test_save_from_counts_preserves_the_hidden_fb_slot(mainwindow, source, tmp_path):
    mainwindow._load_files([source])
    panel = mainwindow._plot_panel
    mainwindow._view_mode_buttons[2].click()
    mainwindow._view_bunch_spin.setValue(4)
    _hold(panel, "x", (0.5, 2.5))
    _auto(panel, False, True)
    mainwindow._raw_counts_action.trigger()
    mainwindow._view_bunch_spin.setValue(2)
    _hold(panel, "x", (1.25, 2.0))
    path = tmp_path / "counts.asymp"
    save_project(mainwindow.collect_project_state(), path)

    reopened = MainWindow()
    reopened._open_project_file(str(path))
    panel = reopened._plot_panel
    assert reopened._plot_workspace.active_view() == "raw_counts"
    assert reopened._view_bunch_spin.value() == 2
    assert panel._ax.get_xlim() == pytest.approx((1.25, 2.0))
    reopened._domain_buttons_by_token["fb_asymmetry"].click()
    assert reopened._view_mode_buttons[2].isChecked()
    assert reopened._view_bunch_spin.value() == 4
    assert panel._ax.get_xlim() == pytest.approx((0.5, 2.5))
    assert panel._auto_y_btn.isChecked()
    assert not reopened._dirty


def test_saved_auto_y_follows_a_run_without_the_saved_projections(mainwindow):
    from tests.gui.test_mainwindow_additional import _make_vector_dataset

    panel = mainwindow._plot_panel
    vector = _make_vector_dataset(8801)
    time = np.linspace(0.0, 3.0, 120)
    scalar = MuonDataset(time, 14.0 * np.cos(time), np.ones_like(time), {"run_number": 8802})
    for dataset in (scalar, vector):
        mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({8802})
    mainwindow._data_browser.select_runs({8801})
    panel._projection_bar._chips["P_x"].click()
    assert panel._subplot_axes_by_polarization.keys() == {"P_x", "P_z"}
    _auto(panel, False, True)
    first, second, _ = mainwindow._view_mode_buttons
    second.click()

    mainwindow._data_browser.select_runs({8802})
    _hold(panel, "y", (-4.0, 2.0))
    first.click()
    assert panel._auto_y_btn.isChecked()
    assert panel._ax.get_ylim()[1] > 14.0


@pytest.mark.parametrize("view", ["raw_counts", "groups"])
def test_fb_recall_preserves_count_y_limits_without_drawing_outgoing_counts(
    mainwindow, source, monkeypatch, view
):
    mainwindow._load_files([source])
    panel = mainwindow._plot_panel
    first, second, _ = mainwindow._view_mode_buttons
    _auto(panel, False, True)
    second.click()
    mainwindow._view_bunch_spin.setValue(4)
    views = mainwindow._domain_buttons_by_token
    enter_counts = (
        mainwindow._raw_counts_action.trigger if view == "raw_counts" else views["groups"].click
    )
    enter_counts()
    assert mainwindow._plot_workspace.active_view() == view
    count_limits = (850.0, 1500.0)
    count_axis = next(iter(panel._subplot_axes_by_polarization))
    panel.set_fit_target_projection(count_axis)
    _hold(panel, "y", count_limits)
    mainwindow._view_bunch_spin.setValue(2)
    mainwindow._on_view_mode_button_clicked(0)
    assert second.isChecked()
    assert panel._ax.get_ylim() == pytest.approx(count_limits)
    assert panel.get_bunch_factor() == 2

    original_draw = panel.plot_grouped_time_domain_subplots
    count_draws = []

    def record_draw(datasets):
        count_draws.append(mainwindow._plot_workspace.active_view())
        original_draw(datasets)

    monkeypatch.setattr(panel, "plot_grouped_time_domain_subplots", record_draw)
    views["fb_asymmetry"].click()
    assert count_draws == []
    assert panel.get_bunch_factor() == 4
    first.click()
    assert panel._auto_y_btn.isChecked()
    state = mainwindow.collect_project_state()
    saved_axes = state["view_modes_state"]["modes"][1]["axis_limits"]["axes"]
    assert f"y:{count_axis}" not in saved_axes

    enter_counts()
    assert count_draws == [view]
    assert not panel._auto_y_btn.isChecked()
    assert panel._ax.get_ylim() == pytest.approx(count_limits)


def test_fb_recall_preserves_reconstruction_y_limits(mainwindow, source, monkeypatch):
    mainwindow._load_files([source])
    dataset = mainwindow._current_dataset
    monkeypatch.setattr(mainwindow, "_maxent_reconstruction_datasets", lambda _run: [dataset])
    panel = mainwindow._plot_panel
    first, second, _ = mainwindow._view_mode_buttons
    _auto(panel, False, True)
    second.click()
    mainwindow._view_bunch_spin.setValue(4)
    views = mainwindow._domain_buttons_by_token
    views["groups"].click()
    reconstruction = mainwindow._maxent_panel._show_reconstruction_check
    reconstruction.click()
    assert mainwindow._plot_workspace.active_view() == "reconstruction"
    axis_key = next(iter(panel._subplot_axes_by_polarization))
    panel.set_fit_target_projection(axis_key)
    limits = (-7.0, 9.0)
    _hold(panel, "y", limits)
    assert panel._ax.get_ylim() == pytest.approx(limits)
    assert panel.axis_limit_state()["axes"][diagnostic_axis_id(f"y:{axis_key}")]["auto"] is False

    draw = panel.plot_maxent_reconstruction
    calls = []

    def record_draw(*args, **kwargs):
        calls.append(mainwindow._plot_workspace.active_view())
        draw(*args, **kwargs)

    monkeypatch.setattr(panel, "plot_maxent_reconstruction", record_draw)
    views["fb_asymmetry"].click()
    first.click()
    assert panel._auto_y_btn.isChecked()
    assert calls == []
    assert panel.axis_limit_state()["axes"][diagnostic_axis_id(f"y:{axis_key}")]["auto"] is False
    reconstruction.click()
    assert calls == ["reconstruction"]
    assert not panel._auto_y_btn.isChecked()
    assert panel._ax.get_ylim() == pytest.approx(limits)


@pytest.mark.parametrize("label", ["group:North", "-1001"])
def test_saved_views_hold_exact_custom_projection_labels(mainwindow, label):
    from tests.gui.test_mainwindow_additional import _make_vector_dataset

    dataset = _make_vector_dataset(1)
    dataset.run.grouping["projections"][-1]["label"] = label
    dataset.run.grouping["vector_axis"] = label
    mainwindow._data_browser.add_dataset(dataset)
    mainwindow._data_browser.select_runs({1})
    panel = mainwindow._plot_panel
    assert panel.get_current_polarization_axis() == label
    _hold(panel, "y", (-4.0, 13.0))
    first, second, _ = mainwindow._view_mode_buttons
    second.click()
    _hold(panel, "y", (-7.0, 17.0))
    first.click()
    assert panel._ax.get_ylim() == pytest.approx((-4.0, 13.0))

    mainwindow._domain_buttons_by_token["groups"].click()
    count_axis = next(iter(panel._subplot_axes_by_polarization))
    if label == "-1001":
        assert count_axis == label
    panel.set_fit_target_projection(count_axis)
    _hold(panel, "y", (70.0, 120.0))
    mainwindow._domain_buttons_by_token["fb_asymmetry"].click()
    assert panel.get_current_polarization_axis() == label
    assert panel._ax.get_ylim() == pytest.approx((-4.0, 13.0))
    second.click()
    assert panel._ax.get_ylim() == pytest.approx((-7.0, 17.0))
    mainwindow._domain_buttons_by_token["groups"].click()
    assert panel._ax.get_ylim() == pytest.approx((70.0, 120.0))
