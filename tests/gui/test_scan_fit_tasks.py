"""Production scan fit actions keep numerical work off the application thread."""

import numpy as np
import pytest
from PySide6.QtCore import QThread

from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
from asymmetry.gui.mainwindow import MainWindow
from tests.gui.scan_fit_helpers import wait_for_fit
from tests.gui.test_integral_scan_gui import _ds, _set_regions
from tests.gui.test_rf_scan_panel_gui import _rf_datasets

pytestmark = pytest.mark.gui


def prepare_scan(window, kind):
    window._plot_workspace.set_active_view("integral_scan")
    if kind == "rf":
        datasets = _rf_datasets()
        window._alc_fit_panel._rf_difference_check.setChecked(True)
    else:
        datasets = [
            _ds(i + 1, 100 * (1 + a), 100 * (1 - a), float(x))
            for i, x in enumerate(np.linspace(50, 350, 31))
            for a in [0.1 - 0.05 * np.exp(-0.5 * ((x - 200) / 20) ** 2)]
        ]
    window._fit_panel.set_datasets(datasets)
    window._on_scan_requested()
    if kind != "rf":
        _set_regions(window._alc_scan_view, [(50, 125), (275, 350)])
    if kind == "peaks":
        window._on_fit_baseline()
        wait_for_fit(window)
        window._alc_scan_view._add_peak("Gaussian")


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_actual_action_evaluates_off_thread(kind, qapp, monkeypatch):
    window = MainWindow()
    try:
        prepare_scan(window, kind)
        threads = []
        original = ParameterCompositeModel.function

        def evaluate(model, *args, **kwargs):
            threads.append(QThread.currentThread() == qapp.thread())
            return original(model, *args, **kwargs)

        monkeypatch.setattr(ParameterCompositeModel, "function", evaluate)
        getattr(
            window,
            {"baseline": "_on_fit_baseline", "peaks": "_on_fit_peaks", "rf": "_on_fit_rf"}[kind],
        )()
        wait_for_fit(window)
        assert threads and not any(threads)
        assert kind in window._alc_scan_view.fit_history.active
    finally:
        window._clear_dirty()
        window.close()


def wait_until(predicate, message):
    from PySide6.QtCore import QElapsedTimer
    from PySide6.QtTest import QTest

    timer = QElapsedTimer()
    timer.start()
    while not predicate() and timer.elapsed() < 10000:
        QTest.qWait(5)
    assert predicate(), message


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
@pytest.mark.parametrize("action", ["stop", "edit", "rebuild", "project", "supersede", "error"])
def test_internal_solve_heartbeat_cancel_stale_recovery(kind, action, qapp, monkeypatch):
    import threading
    from copy import deepcopy

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QTableWidgetItem

    window = MainWindow()
    entered, release = threading.Event(), threading.Event()
    timer = QTimer()
    ticks = []
    timer.setInterval(2)
    timer.timeout.connect(lambda: ticks.append(1))
    original = ParameterCompositeModel.function
    calls = []

    def held(model, *args, **kwargs):
        value = original(model, *args, **kwargs)
        calls.append(QThread.currentThread() != qapp.thread())
        if not entered.is_set():
            entered.set()
            assert release.wait(15), "Numerical barrier was not released"
            if action == "error":
                raise ValueError("owned numerical failure")
        return value

    buttons = {"baseline": "_fit_baseline_btn", "peaks": "_fit_peaks_btn", "rf": "_fit_rf_btn"}
    try:
        prepare_scan(window, kind)
        view = window._alc_scan_view
        history = view.fit_history
        previous = deepcopy(history.to_dict()["records"])
        window._clear_dirty()
        monkeypatch.setattr(ParameterCompositeModel, "function", held)
        timer.start()
        getattr(view, buttons[kind]).click()
        wait_until(entered.is_set, "Fit never reached actual model evaluation")
        wait_until(lambda: len(ticks) >= 3, "GUI heartbeat stopped inside solver")
        old_job = window._scan_fits.current
        assert calls and all(calls)
        if action == "stop":
            view.fit_controls.button.click()
        elif action == "edit":
            if kind == "baseline":
                view._regions_table.setItem(0, 0, QTableWidgetItem("60"))
            elif kind == "peaks":
                view._peaks_table.setItem(0, 1, QTableWidgetItem("205"))
            else:
                view._rf_a_mu_spin.setValue(510)
        elif action == "rebuild":
            window._on_scan_requested()
        elif action == "project":
            window._on_new_project()
        elif action == "supersede":
            getattr(view, buttons[kind]).click()
            assert window._scan_fits.pending is not None
            assert window._scan_fits.current is old_job
        if action != "error":
            assert old_job.worker.is_cancelled()
        release.set()
        wait_for_fit(window)
        if action == "supersede":
            assert len(history.records) == len(previous) + 1
            assert kind in history.active
        else:
            assert history.to_dict()["records"] == previous
            if action in {"stop", "error"}:
                assert not window._dirty
        if action == "project":
            assert not view.fit_history.records
            assert window._alc_baseline is None
            prepare_scan(window, kind)
        elif action == "rebuild" and kind == "peaks":
            _set_regions(view, [(50, 125), (275, 350)])
            window._on_fit_baseline()
            wait_for_fit(window)
            view._add_peak("Gaussian")
        elif action == "rebuild" and kind == "baseline":
            _set_regions(view, [(50, 125), (275, 350)])
        # Late obsolete progress and every terminal type cannot touch a completed replacement.
        getattr(view, buttons[kind]).click()
        wait_for_fit(window)
        captured = deepcopy(view.fit_history.to_dict())
        controls = (
            view.fit_controls.progress_label.text(),
            view._peaks_results.text(),
            view._rf_results.text(),
        )
        old_job.progress(0, 1, "obsolete progress")
        old_job.finished(view.fit_history.records[-1])
        old_job.error("obsolete error")
        old_job.cancelled()
        assert view.fit_history.to_dict() == captured
        assert controls == (
            view.fit_controls.progress_label.text(),
            view._peaks_results.text(),
            view._rf_results.text(),
        )
        assert kind in view.fit_history.active
    finally:
        release.set()
        timer.stop()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()


@pytest.mark.parametrize("stage", ["rf_seed", "peak_jacobian", "baseline_covariance", "prediction"])
def test_stop_at_dependent_numerical_stage(stage, qapp, monkeypatch):
    import threading

    import asymmetry.core.fitting.field_scan as fs
    import asymmetry.core.fitting.parameter_models as pm
    import asymmetry.gui.scan.fitting as fitting

    kind = "rf" if stage == "rf_seed" else "baseline" if stage == "prediction" else "peaks"
    window = MainWindow()
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    phase = False

    def barrier():
        entered.set()
        assert release.wait(15), "Dependent numerical stage was not released"

    try:
        if stage == "rf_seed":
            import tests.gui.test_rf_scan_panel_gui as rf_fixture
            from asymmetry.core.fitting.muon_proton import rf_resonance_mup

            def narrow_doublet(field):
                return float(
                    rf_resonance_mup(np.array([field]), 515, 124, 218.5, 2, 8, 2, 8, 0.1)[0]
                )

            monkeypatch.setattr(rf_fixture, "_gmr_value", narrow_doublet)
        prepare_scan(window, kind)
        history = window._alc_scan_view.fit_history
        previous = list(history.records)
        if stage == "rf_seed":
            original = pm.rf_transition_freqs

            def detuning(*args, **kwargs):
                value = original(*args, **kwargs)
                if not entered.is_set():
                    barrier()
                return value

            monkeypatch.setattr(pm, "rf_transition_freqs", detuning)
        elif stage == "prediction":
            original = fitting.ScanFitRecord.capture

            def capture(*args, **kwargs):
                value = original(*args, **kwargs)
                barrier()
                return value

            monkeypatch.setattr(fitting.ScanFitRecord, "capture", capture)
        else:
            original_jacobian = fs._model_jacobian
            original_evaluate = ParameterCompositeModel.function

            def jacobian(*args, **kwargs):
                nonlocal phase
                # Peaks have their own Jacobian before the baseline operator's Jacobian.
                model = args[0]
                phase = (
                    ("Linear" in model.component_names)
                    if stage == "baseline_covariance"
                    else ("GaussianLCR" in model.component_names)
                )
                return original_jacobian(*args, **kwargs)

            def evaluate(*args, **kwargs):
                value = original_evaluate(*args, **kwargs)
                if phase and not entered.is_set():
                    barrier()
                return value

            monkeypatch.setattr(fs, "_model_jacobian", jacobian)
            monkeypatch.setattr(ParameterCompositeModel, "function", evaluate)
        original_compute = fitting.ScanFitJob.compute

        def compute(job, worker):
            try:
                return original_compute(job, worker)
            finally:
                exited.set()

        monkeypatch.setattr(fitting.ScanFitJob, "compute", compute)
        getattr(
            window._alc_scan_view,
            {"baseline": "_fit_baseline_btn", "peaks": "_fit_peaks_btn", "rf": "_fit_rf_btn"}[kind],
        ).click()
        wait_until(entered.is_set, "Dependent stage never executed")
        window._alc_scan_view.fit_controls.button.click()
        release.set()
        wait_for_fit(window)
        assert exited.is_set()
        assert history.records == previous
    finally:
        release.set()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()


def test_peak_cannot_bind_to_replaced_baseline(qapp, monkeypatch):
    import threading

    window = MainWindow()
    entered, release = threading.Event(), threading.Event()
    original = ParameterCompositeModel.function

    def held(*args, **kwargs):
        value = original(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(15)
        return value

    try:
        prepare_scan(window, "peaks")
        view = window._alc_scan_view
        old_baseline = view.fit_history.records[0]
        monkeypatch.setattr(ParameterCompositeModel, "function", held)
        view._fit_peaks_btn.click()
        wait_until(entered.is_set, "Peak solver not reached")
        # An explicit refit has the same coordinates but owns a new execution.
        view._fit_baseline_btn.click()
        assert window._scan_fits.pending.kind == "baseline"
        release.set()
        wait_for_fit(window)
        assert len(view.fit_history.records) == 2
        assert view.fit_history.records[0] is old_baseline
        assert view.fit_history.active["baseline"] == 1
        assert "peaks" not in view.fit_history.active
    finally:
        release.set()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()


@pytest.mark.parametrize(
    "change", ["axis", "derivative", "exclude", "selection", "model", "rf_mode", "window", "open"]
)
def test_actual_inputs_cancel_held_fit(change, tmp_path, qapp, monkeypatch):
    import threading
    from copy import deepcopy

    from asymmetry.core.project import save_project

    window = MainWindow()
    entered, release = threading.Event(), threading.Event()
    original = ParameterCompositeModel.function

    def held(*args, **kwargs):
        result = original(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(15)
        return result

    try:
        prepare_scan(window, "baseline")
        path = tmp_path / "captured-scan.asymp"
        save_project(window.collect_project_state(), path)
        view = window._alc_scan_view
        history = view.fit_history
        previous = deepcopy(history.to_dict()["records"])
        monkeypatch.setattr(ParameterCompositeModel, "function", held)
        view._fit_baseline_btn.click()
        wait_until(entered.is_set, "Solver never entered")
        job = window._scan_fits.current
        if change == "axis":
            view._x_combo.setCurrentIndex(2)
        elif change == "derivative":
            view._derivative_check.setChecked(True)
        elif change == "exclude":
            window._on_alc_point_toggled(window._alc_scan_points[0]["run"])
        elif change == "selection":
            window._fit_panel.set_datasets(window._fit_panel.batch_datasets()[:-1])
        elif change == "model":
            view._baseline_model_combo.setCurrentText("Constant")
        elif change == "rf_mode":
            window._alc_fit_panel._rf_difference_check.setChecked(True)
        elif change == "window":
            window._plot_panel.set_fit_range(0.0, 0.015)
        else:
            window._open_project_file(str(path))
        assert job.worker.is_cancelled()
        after_edit = deepcopy(view.fit_history.to_dict())
        release.set()
        wait_for_fit(window)
        assert history.to_dict()["records"] == previous
        assert view.fit_history.to_dict() == after_edit
    finally:
        release.set()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()


@pytest.mark.parametrize("terminal", ["success", "error", "cancel"])
def test_close_joins_numerical_exit_without_delivery(terminal, qapp, monkeypatch):
    import threading

    import asymmetry.gui.scan.fitting as fitting
    from asymmetry.gui.tasks import TaskCancelledError

    window = MainWindow()
    entered, cancelled, exited = threading.Event(), threading.Event(), threading.Event()
    original_evaluate = ParameterCompositeModel.function

    def held(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert cancelled.wait(15), "Close did not cancel its owned worker"
            if terminal == "error":
                raise ValueError("late close error")
            if terminal == "cancel":
                raise TaskCancelledError()
        return result

    original_compute = fitting.ScanFitJob.compute

    def compute(job, worker):
        try:
            return original_compute(job, worker)
        finally:
            exited.set()

    try:
        prepare_scan(window, "baseline")
        monkeypatch.setattr(ParameterCompositeModel, "function", held)
        monkeypatch.setattr(fitting.ScanFitJob, "compute", compute)
        # Signal the numerical barrier from the actual cooperative cancel request.
        original_cancel = fitting.TaskRunner.start

        def start(runner, *args, **kwargs):
            worker = original_cancel(runner, *args, **kwargs)
            cancel = worker.cancel

            def request_cancel():
                cancel()
                cancelled.set()

            worker.cancel = request_cancel
            return worker

        monkeypatch.setattr(fitting.TaskRunner, "start", start)
        window._alc_scan_view._fit_baseline_btn.click()
        wait_until(entered.is_set, "No held numerical stage")
        window._clear_dirty()
        window.close()
        assert cancelled.is_set() and exited.is_set()
        assert window._scan_fits.idle
        assert not window._alc_scan_view.fit_history.records
    finally:
        cancelled.set()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()


def test_queued_old_terminals_preserve_running_replacement(qapp, monkeypatch):
    import threading

    from PySide6.QtCore import QTimer

    window = MainWindow()
    entered_a, entered_b = threading.Event(), threading.Event()
    release_a, release_b, delivered = threading.Event(), threading.Event(), threading.Event()
    original = ParameterCompositeModel.function

    def held(*args, **kwargs):
        value = original(*args, **kwargs)
        if not entered_a.is_set():
            entered_a.set()
            assert release_a.wait(15)
        elif release_a.is_set() and not entered_b.is_set():
            entered_b.set()
            assert release_b.wait(15)
        return value

    try:
        prepare_scan(window, "baseline")
        view = window._alc_scan_view
        monkeypatch.setattr(ParameterCompositeModel, "function", held)
        view._fit_baseline_btn.click()
        wait_until(entered_a.is_set, "First solve not held")
        old = window._scan_fits.current
        view._fit_baseline_btn.click()
        release_a.set()
        wait_until(entered_b.is_set, "Replacement solve not held")
        current = window._scan_fits.current
        label = view.fit_controls.progress_label.text()

        def late():
            old.progress(1, 1, "obsolete progress")
            old.error("obsolete error")
            old.cancelled()
            old.finished(None)
            delivered.set()

        QTimer.singleShot(0, late)
        wait_until(delivered.is_set, "Queued obsolete delivery not exercised")
        assert window._scan_fits.current is current
        assert not current.worker.is_cancelled()
        assert view.fit_controls.progress_label.text() == label
        assert not view.fit_history.records
        release_b.set()
        wait_for_fit(window)
        assert len(view.fit_history.records) == 1
    finally:
        release_a.set()
        release_b.set()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_actual_fit_progress_remains_indeterminate(kind, qapp, monkeypatch):
    import threading

    import asymmetry.gui.scan.fitting as fitting

    window = MainWindow()
    entered, release, progress_delivered = threading.Event(), threading.Event(), threading.Event()
    evaluate = ParameterCompositeModel.function
    progress = fitting.ScanFitJob.progress

    def held(*args, **kwargs):
        value = evaluate(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(15)
        return value

    def deliver(job, *args):
        assert QThread.currentThread() == qapp.thread()
        progress(job, *args)
        progress_delivered.set()

    try:
        prepare_scan(window, kind)
        monkeypatch.setattr(ParameterCompositeModel, "function", held)
        monkeypatch.setattr(fitting.ScanFitJob, "progress", deliver)
        getattr(
            window._alc_scan_view,
            {"baseline": "_fit_baseline_btn", "peaks": "_fit_peaks_btn", "rf": "_fit_rf_btn"}[kind],
        ).click()
        wait_until(entered.is_set, "Solver stage not reached")
        wait_until(progress_delivered.is_set, "Queued worker progress not delivered")
        bar = window._alc_scan_view.fit_controls.progress_bar
        assert (bar.minimum(), bar.maximum()) == (0, 0)
    finally:
        release.set()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_stop_then_indivisible_model_error_reports_stopped(kind, qapp, monkeypatch):
    import threading

    window = MainWindow()
    entered, release = threading.Event(), threading.Event()
    evaluate = ParameterCompositeModel.function

    def held(*args, **kwargs):
        value = evaluate(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(15)
            raise ValueError("Native model failure after Stop")
        return value

    try:
        prepare_scan(window, kind)
        view = window._alc_scan_view
        records = list(view.fit_history.records)
        monkeypatch.setattr(ParameterCompositeModel, "function", held)
        getattr(
            view,
            {"baseline": "_fit_baseline_btn", "peaks": "_fit_peaks_btn", "rf": "_fit_rf_btn"}[kind],
        ).click()
        wait_until(entered.is_set, "Actual model evaluation not held")
        view.fit_controls.button.click()
        release.set()
        wait_for_fit(window)
        assert view.fit_controls.progress_label.text() == "Fit stopped."
        assert view.fit_history.records == records
        assert "Native model failure after Stop" not in window._log_panel.to_plain_text()
    finally:
        release.set()
        window._scan_fits.shutdown()
        window._clear_dirty()
        window.close()
