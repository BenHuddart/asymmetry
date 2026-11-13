"""Project rejection must precede replacement of unsaved analysis state."""

import json

import pytest
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, save_project
from asymmetry.gui.mainwindow import MainWindow
from tests.gui.test_project_shell import _ds

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("datasets", ["invalid", {}, [{"run_number": 7, "source_file": []}]])
def test_malformed_open_preserves_unsaved_work(qapp, tmp_path, monkeypatch, datasets):
    window = MainWindow()
    dataset = _ds(11)
    window._data_browser.add_dataset(dataset)
    window._mark_dirty()
    model = window._project_model
    profiles = window._grouping_profiles
    window._current_project_path = str(tmp_path / "existing.asymp")
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    path = tmp_path / "malformed.asymp"
    path.write_text(json.dumps({"schema_version": CURRENT_SCHEMA_VERSION, "datasets": datasets}))

    window._open_project_file(str(path))

    assert window._data_browser.get_dataset(11) is dataset
    assert window._project_model is model
    assert window._grouping_profiles is profiles
    assert window._dirty
    assert window._current_project_path == str(tmp_path / "existing.asymp")
    assert len(errors) == 1
    assert "datasets" in errors[0]
    window._clear_dirty()
    window.close()


def test_direct_restore_rejects_before_mutation(qapp, tmp_path):
    window = MainWindow()
    window._data_browser.add_dataset(_ds())
    profiles = window._grouping_profiles
    with pytest.raises(ValueError, match="datasets"):
        window.restore_project_state(
            {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": "bad"},
            str(tmp_path / "bad.asymp"),
        )
    assert window._data_browser.get_dataset(11) is not None
    assert window._grouping_profiles is profiles
    assert window._dirty


@pytest.mark.parametrize("action", ["cancel", "close", "new", "supersede"])
def test_pre_apply_cancel_retains_work_and_discards_late_result(
    qapp, tmp_path, monkeypatch, action
):
    import threading

    from PySide6.QtCore import QTimer

    import asymmetry.gui.utils.project_preparation as module
    from tests._qt_helpers import wait_for

    window = MainWindow()
    dataset = _ds()
    window._data_browser.add_dataset(dataset)
    window._mark_dirty()
    state = window.collect_project_state()
    path = tmp_path / "valid.asymp"
    path.write_text(json.dumps(state))
    release = threading.Event()
    entered = threading.Event()
    real_load = module.load_prepared_project

    def delayed(path, **kwargs):
        entered.set()
        release.wait(5)
        return real_load(path)  # Deliberately finish despite cancellation.

    monkeypatch.setattr(module, "load_prepared_project", delayed)
    beats = []
    visible = []
    timer = QTimer(window)
    timer.setInterval(1)
    timer.timeout.connect(lambda: beats.append(1))
    timer.start()

    def cancel():
        if not entered.is_set():
            QTimer.singleShot(1, window, cancel)
            return
        from asymmetry.gui.utils.project_preparation import ProjectPreparation

        visible.append(window.findChild(ProjectPreparation).isVisible())
        if action == "close":
            assert not window.close()
        elif action == "new":
            window._on_new_project()
        elif action == "supersede":
            window._open_project_file(str(path))
            window._bulk_load_cancel()
        else:
            window._bulk_load_cancel()

    QTimer.singleShot(650, window, cancel)
    window._open_project_file(str(path))
    assert len(beats) > 2
    assert visible == [True]
    assert window._data_browser.get_dataset(11) is dataset
    assert window._dirty
    assert window._current_project_path is None
    assert not window._bulk_load_active
    assert "existing work retained" in window.statusBar().currentMessage()
    release.set()
    wait_for(lambda: window._tasks.active_count == 0, qapp, timeout_s=5)
    assert window._data_browser.get_dataset(11) is dataset
    assert window._current_project_path is None
    timer.stop()


def test_malformed_autosave_retains_work(qapp, tmp_path, monkeypatch):
    from tests.gui.test_project_autosave import _patch_recovery_choice, _write_project_pair

    window = MainWindow()
    dataset = _ds()
    window._data_browser.add_dataset(dataset)
    window._mark_dirty()
    project, autosave = _write_project_pair(window, tmp_path)
    autosave.write_text(json.dumps({"schema_version": CURRENT_SCHEMA_VERSION, "datasets": "bad"}))
    _patch_recovery_choice(monkeypatch, "load")
    errors = []
    import asymmetry.gui.mainwindow as module

    monkeypatch.setattr(
        module.QMessageBox, "critical", lambda *args: errors.append(args[-1]), raising=False
    )
    window._open_project_file(str(project))
    assert window._data_browser.get_dataset(11) is dataset
    assert window._dirty
    assert autosave.exists()
    assert errors and "datasets" in errors[0]


def test_shell_rejection_retains_other_tabs(qapp, tmp_path, monkeypatch):
    from asymmetry.gui.shell import ProjectShell

    shell = ProjectShell()
    page = shell.add_project()
    dataset = _ds()
    page._data_browser.add_dataset(dataset)
    path = tmp_path / "bad.asymp"
    path.write_text(json.dumps({"schema_version": CURRENT_SCHEMA_VERSION, "datasets": "bad"}))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    shell.open_project(str(path))
    assert page._data_browser.get_dataset(11) is dataset
    assert page._dirty
    assert errors and "datasets" in errors[0]


def test_project_model_round_trips_browser_group_kind(qapp):
    from asymmetry.core.representation.project_model import ProjectModel
    from asymmetry.gui.panels.data_browser import DataBrowserPanel

    model = ProjectModel()
    panel = DataBrowserPanel()
    panel.set_project_model(model)
    for number in (1, 2, 3):
        panel.add_dataset(_ds(number))
    user = panel.create_data_group([1, 2], name="User group")
    automatic = panel.create_data_group([2, 3], name="Auto group", kind="auto")
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [{"run_number": number} for number in (1, 2, 3)],
    }
    model.write_to_project_state(state)
    restored = ProjectModel.from_project_state(state)
    assert restored.data_group(user).kind == "user"
    assert restored.data_group(automatic).kind == "auto"
    assert sorted(restored.data_group(automatic).member_run_numbers) == [2, 3]


@pytest.mark.parametrize(
    "path,value",
    [
        (("plot_state", "rotating_frame"), []),
        (("plot_state", "rotating_frame"), None),
        (("plot_state", "projection_memory"), ["bad"]),
        (("plot_state", "projection_memory"), {"x": None}),
        (("plot_state", "fit_curves"), ["bad"]),
        (("plot_state", "fit_curves"), None),
        (("plot_state", "fit_curve"), {"t": [1], "y": []}),
        (("browser_state", "filters"), ["bad"]),
        (("browser_state", "filters"), None),
        (("browser_state", "filters"), {"bad": []}),
        (("browser_state", "filters"), {"1": [{}]}),
        (("browser_state", "sort_column"), "1.5"),
        (("fourier_state", "padding"), ["bad"]),
        (("fourier_state", "auto_phase_method"), []),
        (("fourier_state", "auto_phase_method"), None),
        (("fourier_state", "phase_fit_tolerance_degrees"), None),
        (("fourier_state", "exclusion_ranges"), [[[], 0]]),
        (("fit_parameters_state", "inferred_x_key"), []),
        (("plot_state", "axis_limits"), {}),
        (("plot_state", "axis_limits"), {"axes": None}),
        (("plot_state", "axis_limits"), {"axes": {"x": {}}}),
        (
            ("plot_state", "axis_limits"),
            {"axes": {"x": {"auto": True, "held": [None, 1], "quantity": None}}},
        ),
        (("fit_states", "time", "global_fit_state", "parameters"), [{"name": "A", "value": []}]),
        (
            ("plot_state", "axis_limits"),
            {"axes": {"x": {"auto": False, "held": [float("nan"), 1.0], "quantity": None}}},
        ),
        (
            ("plot_state", "axis_limits"),
            {"axes": {"x": {"auto": False, "held": [float("inf"), 1.0], "quantity": None}}},
        ),
        (
            ("plot_state", "axis_limits"),
            {"axes": {"x": {"auto": False, "held": [-float("inf"), 1.0], "quantity": None}}},
        ),
        (("fit_states", "time", "domain"), "frequency"),
        (("fit_states", "time", "fit_ui_state"), None),
    ],
)
def test_collected_owned_state_rejects_before_clear(qapp, tmp_path, monkeypatch, path, value):
    window = MainWindow()
    dataset = _ds()
    window._data_browser.add_dataset(dataset)
    state = window.collect_project_state()
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    project = tmp_path / "malformed.asymp"
    save_project(state, project)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    window._open_project_file(str(project))
    assert window._data_browser.get_dataset(11) is dataset
    assert window._dirty
    assert window._current_project_path is None
    assert errors and path[0] in errors[0] and path[-1] in errors[0]


def test_global_blank_draft_restores_without_numeric_default(qapp, tmp_path):
    from asymmetry.core.fitting.composite import CompositeModel

    window = MainWindow()
    state = window.collect_project_state()
    state["fit_states"]["time"]["global_fit_state"].update(
        composite_model=CompositeModel.from_expression("Exponential").to_dict(),
        parameters=[{"name": "A_1", "value": ""}],
    )
    window.restore_project_state(state, str(tmp_path / "draft.asymp"))
    table = window._fit_panel._global_tab._param_table
    row = next(i for i in range(table.rowCount()) if table.item(i, 0).data(256) == "A_1")
    assert table.item(row, 1).text() == ""
