"""Actual diagnostic actions, captured fit inputs and worker lifecycle."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.pull_diagnostic import RefitOutcome, make_pull_request, run_pull_request
from asymmetry.core.pull_refit import EngineRefit
from asymmetry.core.simulate import reduce_run_to_dataset, simulate_run
from asymmetry.gui.panels.fit.single_tab import DIAGNOSTIC_ACTION, SingleFitTab
from asymmetry.gui.windows.pull_diagnostic_window import PullDiagnosticWindow
from tests._qt_helpers import wait_for
from tests.core.test_pull_diagnostic import _constant, _small_template

pytestmark = pytest.mark.gui


def _window(refit=None):
    request = make_pull_request(
        _small_template(), _constant, {"a": 20}, total_events=100000, n_seeds=3, seed_start=71
    )
    return PullDiagnosticWindow(
        request=request,
        model=_constant,
        refit=refit
        or (lambda ds, cancel: RefitOutcome(True, {"a": float(np.mean(ds.asymmetry))}, {"a": 1})),
    )


def _drain(window, qapp):
    wait_for(lambda: window._worker is None and window._tasks.active_count == 0, qapp, timeout_s=10)


def test_actual_run_button_worker_heartbeat_and_seed_capture(qapp):
    entered, release = threading.Event(), threading.Event()
    seen = []

    def held(ds, cancel):
        seen.append(threading.current_thread())
        entered.set()
        assert release.wait(5)
        return RefitOutcome(True, {"a": 21}, {"a": 1})

    window = _window(held)
    beats = []
    timer = QTimer()
    timer.timeout.connect(lambda: beats.append(1))
    timer.start(1)
    try:
        window._run_button.click()
        wait_for(entered.is_set, qapp, timeout_s=5)
        wait_for(lambda: len(beats) >= 3, qapp, timeout_s=5)
        assert not window._run_button.isEnabled()
        window._seed_start_spin.setValue(999)
    finally:
        release.set()
        timer.stop()
    _drain(window, qapp)
    assert all(t is not threading.main_thread() for t in seen)
    assert [a.seed for a in window.last_result.attempts] == [71, 72, 73]
    assert "3 usable" in window._verdict_label.text()
    assert window._run_button.isEnabled()


def test_stop_held_numerical_refit_partial_then_recovery(qapp):
    from asymmetry.core.fitting.engine import FitCancelledError

    entered, release = threading.Event(), threading.Event()

    def held(ds, cancel):
        entered.set()
        assert release.wait(5)
        assert cancel()
        raise FitCancelledError("stopped in numerical fit")

    window = _window(held)
    try:
        window._run_button.click()
        wait_for(entered.is_set, qapp, timeout_s=5)
        window._cancel_button.click()
        assert not window._cancel_button.isEnabled()
    finally:
        release.set()
    _drain(window, qapp)
    result = window.last_result
    assert result.stopped and result.n_attempted == 1 and result.n_failed == 0
    assert "partial" in window._verdict_label.text()
    window._refit = lambda ds, cancel: RefitOutcome(True, {"a": 20}, {"a": 1})
    window._run_button.click()
    _drain(window, qapp)
    assert not window.last_result.stopped and window.last_result.n_usable == 3


def test_error_recovery_keeps_previous_result(qapp):
    window = _window()
    window._run_button.click()
    _drain(window, qapp)
    previous = window.last_result

    def broken(ds, cancel):
        raise RuntimeError("coding error is visible")

    window._refit = broken
    window._run_button.click()
    _drain(window, qapp)
    assert window.last_result is previous
    assert "coding error is visible" in window._verdict_label.text()
    window._refit = lambda ds, cancel: None
    window._run_button.click()
    _drain(window, qapp)
    assert window.last_result.n_failed == 3


@pytest.mark.parametrize("terminal", ["finished", "error"])
def test_close_with_queued_terminal_does_not_publish_and_reopens(qapp, terminal):
    release, terminal_sent = threading.Event(), threading.Event()

    def refit(ds, cancel):
        assert release.wait(5)
        if terminal == "error":
            raise RuntimeError("late error")
        return RefitOutcome(True, {"a": 20}, {"a": 1})

    window = _window(refit)
    window._seeds_spin.setValue(1)
    window._run_button.click()
    signal = window._worker.finished if terminal == "finished" else window._worker.error
    signal.connect(lambda *args: terminal_sent.set(), Qt.ConnectionType.DirectConnection)
    release.set()
    # Terminal is emitted while the GUI event loop is held, guaranteeing queueing.
    assert terminal_sent.wait(5)
    window.close()
    qapp.processEvents()
    assert window.last_result is None
    assert window._tasks.active_count == 0
    window._refit = lambda ds, cancel: RefitOutcome(True, {"a": 20}, {"a": 1})
    window.show()
    assert window._run_button.isEnabled()
    window._run_button.click()
    _drain(window, qapp)
    assert window.last_result.n_usable == 1


def test_histogram_keeps_extreme_tail_denominator(qapp):
    outcomes = iter([RefitOutcome(True, {"a": x}, {"a": 1}) for x in (20, 21, 30)])
    window = _window()
    result = run_pull_request(window._request, _constant, lambda ds, cancel: next(outcomes))
    window._on_diagnostic_finished(result)
    ax = window._figure.axes[0]
    assert sum(p.get_height() * p.get_width() for p in ax.patches) == pytest.approx(1)
    assert max(p.get_x() + p.get_width() for p in ax.patches) >= 1
    assert result.parameters["a"].n == 3


def _fitted_tab(qapp, *, fixed=False):
    tab = SingleFitTab()
    run = simulate_run(_small_template(), _constant, {"a": 20}, total_events=2e6, seed=17)
    dataset = reduce_run_to_dataset(run).time_range(0.2, 5)
    tab.set_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Constant"]))
    tab._param_table.item(0, 1).setText("15")
    if fixed:
        from PySide6.QtWidgets import QCheckBox

        tab._param_table.cellWidget(0, 2).findChild(QCheckBox).setChecked(True)
    tab._minos_checkbox.setChecked(True)
    tab._run_fit()
    assert tab.wait_for_fit(10000)
    assert tab._last_fit_result.success
    return tab, dataset


def test_single_action_uses_execution_model_truth_options_support_and_state(qapp):
    import copy

    tab, dataset = _fitted_tab(qapp)
    assert tab._can_run_pull_diagnostic()
    original = copy.deepcopy(tab._last_fit_result.parameters["A_bg"])
    recipe = dataset.effective_recipe
    grouping = copy.deepcopy(dataset.run.grouping)
    original_result = tab._last_fit_result
    history = copy.deepcopy(original_result.parameter_history)
    tab._composite_model = CompositeModel(["Exponential"])
    tab._minos_checkbox.setChecked(False)
    tab._results_card._actions[DIAGNOSTIC_ACTION].click()
    window = tab._pull_diagnostic_window
    tab.set_dataset(None)
    window._seeds_spin.setValue(2)
    window._run_button.click()
    _drain(window, qapp)
    config = window.last_result.request.to_dict()
    assert config["parameters"]["A_bg"] == original.value
    assert config["model"] == CompositeModel(["Constant"]).to_dict()
    assert config["refit"]["options"]["minos"] is True
    assert config["time_range"] == [0.2, 5]
    assert config["source_context"]["effective_recipe"] == recipe.to_dict()
    assert window.last_result.n_attempted == 2
    assert dataset.run.grouping == grouping
    assert original_result.parameter_history == history
    assert tab._pull_fit_input is None
    assert window.last_result.request.to_dict()["refit"]["source_fit_input_attribution"] is not None


def test_fixed_only_fit_has_no_free_parameter_diagnostic(qapp):
    tab, _ = _fitted_tab(qapp, fixed=True)
    assert not tab._can_run_pull_diagnostic()
    assert "Fixed-only" in tab._pull_fit_input.unavailable_reason()


def test_engine_refit_detaches_and_passes_cancel_without_blanket_catch(qapp, monkeypatch):
    from asymmetry.core.fitting.engine import FitCancelledError, FitEngine

    model = CompositeModel(["Constant"])
    params = ParameterSet([Parameter("A_bg", 20)])
    refit = EngineRefit(model, params, minos=True)
    params["A_bg"].value = 99

    def spy(self, ds, fn, parameters, **options):
        assert parameters["A_bg"].value == 20
        assert options["minos"] is True
        assert options["cancel_callback"]()
        raise FitCancelledError("cooperative")

    monkeypatch.setattr(FitEngine, "fit", spy)
    with pytest.raises(FitCancelledError):
        refit(reduce_run_to_dataset(_small_template()), lambda: True)


def test_pre_cancel_actual_button_attempts_zero(qapp, monkeypatch):
    from asymmetry.gui.tasks import TaskWorker

    original = TaskWorker.run

    def pre_cancel(self):
        self.cancel()
        return original(self)

    monkeypatch.setattr(TaskWorker, "run", pre_cancel)
    window = _window()
    window._run_button.click()
    _drain(window, qapp)
    assert window.last_result.stopped and window.last_result.n_attempted == 0


def test_supported_refit_is_owned_and_extreme_finite_histogram_survives(qapp):
    refit = EngineRefit(_constant, ParameterSet([Parameter("a", 18)]))
    window = _window(refit)
    refit.parameters["a"].value = 99
    assert window._refit.parameters["a"].value == 18
    results = iter([-1e308, 1e308])
    config = window._request.to_dict()
    config["n_seeds"] = 2
    from asymmetry.core.pull_request import PullRequest

    result = run_pull_request(
        PullRequest.from_dict(config),
        _constant,
        lambda ds, cancel: RefitOutcome(True, {"a": next(results)}, {"a": 1}),
    )
    window._on_diagnostic_finished(result)
    assert window.last_result.n_usable == 2
    assert "1e+308" in window._figure.axes[0].get_xlabel()


@pytest.mark.parametrize("terminal", ["cancel", "error", "close"])
def test_actual_source_action_preserves_project_history_selection_and_dirty_state(
    qapp, terminal, monkeypatch
):
    import copy
    import json

    from asymmetry.core.fitting.engine import FitCancelledError
    from asymmetry.core.project.json_values import encode_non_finite
    from asymmetry.gui.mainwindow import MainWindow

    host = MainWindow()
    try:
        host._on_fit()
        datasets = []
        for number in (446, 447):
            run = simulate_run(
                _small_template(),
                _constant,
                {"a": 20},
                total_events=1e6,
                seed=number,
                run_number=number,
            )
            ds = reduce_run_to_dataset(run)
            host._data_browser.add_dataset(ds)
            datasets.append(ds)
        host._on_dataset_selected(446)
        tab = host._fit_panel._single_tab
        tab._set_composite_model(CompositeModel(["Constant"]))
        tab._run_fit()
        assert tab.wait_for_fit(10000)
        assert tab._last_fit_result.success
        tab._results_card._actions[DIAGNOSTIC_ACTION].click()
        dialog = tab._pull_diagnostic_window
        entered, release = threading.Event(), threading.Event()

        def held(ds, cancel):
            entered.set()
            assert release.wait(5)
            if terminal == "error":
                raise RuntimeError("injected code error")
            if cancel():
                raise FitCancelledError("cooperative")
            return RefitOutcome(True, {"A_bg": 20}, {"A_bg": 1})

        monkeypatch.setattr(EngineRefit, "__call__", lambda self, ds, cancel: held(ds, cancel))
        representation = host._project_model.representation(446, host._active_representation_type())
        fits = representation.fit_set(None).fits
        assert fits
        history = json.dumps(encode_non_finite([f.to_dict() for f in fits]), sort_keys=True)
        grouping = copy.deepcopy(datasets[0].run.grouping)
        original = tab._last_fit_result
        values = {p.name: p.value for p in original.parameters}
        try:
            dialog._run_button.click()
            wait_for(entered.is_set, qapp, timeout_s=5)
            host._on_dataset_selected(447)
            dirty = (host._dirty, host._dirty_generation)
            selection = host._current_dataset
            if terminal == "cancel":
                dialog._cancel_button.click()
            release.set()
            if terminal == "close":
                dialog.close()
        finally:
            release.set()
        _drain(dialog, qapp)
        assert json.dumps(encode_non_finite([f.to_dict() for f in fits]), sort_keys=True) == history
        assert datasets[0].run.grouping == grouping
        assert {p.name: p.value for p in original.parameters} == values
        assert host._current_dataset is selection
        assert (host._dirty, host._dirty_generation) == dirty
    finally:
        host._dirty = False
        host.close()


def test_captured_model_preparation_runs_on_same_diagnostic_worker(qapp, monkeypatch):
    tab, _ = _fitted_tab(qapp)
    entered, release = threading.Event(), threading.Event()
    original = CompositeModel.function
    seen = []

    def held_model(self, t, **values):
        seen.append(threading.current_thread())
        entered.set()
        assert release.wait(5)
        return original(self, t, **values)

    monkeypatch.setattr(CompositeModel, "function", held_model)
    tab._results_card._actions[DIAGNOSTIC_ACTION].click()
    window = tab._pull_diagnostic_window
    assert not entered.is_set()
    beats = []
    timer = QTimer()
    timer.timeout.connect(lambda: beats.append(1))
    timer.start(1)
    try:
        window._run_button.click()
        wait_for(entered.is_set, qapp, timeout_s=5)
        wait_for(lambda: len(beats) >= 3, qapp, timeout_s=5)
        window._cancel_button.click()
    finally:
        release.set()
        timer.stop()
    _drain(window, qapp)
    assert all(t is not threading.main_thread() for t in seen)
    assert window.last_result.stopped and window.last_result.n_attempted == 0
