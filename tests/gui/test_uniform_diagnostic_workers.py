"""Actual wizard preview preparation stays cancellable and off the GUI thread."""

import threading
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QPushButton

import asymmetry.core.fourier.diagnostic as diagnostic
from asymmetry.core.fitting.component_tags import PhysicsClass
from asymmetry.core.fitting.wizard_scope import WizardScope
from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow
from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow
from tests._qt_helpers import wait_for
from tests.core.test_binning_entry_points import _run
from tests.core.test_wizard_preparation import observed

pytestmark = pytest.mark.gui


def _variable_line():
    t = np.arange(200) * 0.1
    y = 0.2 * np.cos(2 * np.pi * 1.3 * t)
    return observed(
        _run(10000 * (1 + y), 10000 * (1 - y), binning_mode="variable", bin0_us=0.14, bin10_us=0.56)
    )


@pytest.mark.parametrize("terminal", ["success", "cancel", "close"])
def test_actual_preview_worker_heartbeat_terminal_and_recovery(qapp, monkeypatch, terminal):
    entered, release = threading.Event(), threading.Event()
    seen = []
    original = diagnostic.prepare_histogram_asymmetry

    def held(*args, **kwargs):
        seen.append(threading.current_thread())
        entered.set()
        assert release.wait(5), "preview preparation was never released"
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "prepare_histogram_asymmetry", held)
    window = FitWizardWindow()
    dataset = _variable_line()
    beats = []
    timer = QTimer()

    def heartbeat():
        if not entered.is_set():
            return
        beats.append(1)
        if len(beats) == 3:
            if terminal == "cancel":
                QApplication.activeModalWidget().findChild(QPushButton).click()
            elif terminal == "close":
                window.close()
            release.set()

    timer.timeout.connect(heartbeat)
    timer.start(1)
    try:
        window.set_analysis_context(dataset)
    finally:
        release.set()
        timer.stop()
        # The dialog's synchronous boundary drains its own TaskRunner in finally.
    assert len(beats) >= 3
    assert seen and all(t is not threading.main_thread() for t in seen)
    assert (window._spectral_preview is not None) is (terminal == "success")
    monkeypatch.setattr(diagnostic, "prepare_histogram_asymmetry", original)
    window.set_analysis_context(dataset)
    assert window._spectral_preview is not None
    np.testing.assert_allclose(window._spectral_preview.observation.time, np.arange(200) * 0.1)
    assert "fixed native diagnostic" in window._fft_ax.get_title()
    window.close()


def test_analysis_reuses_preview_and_rendering_does_no_new_replay(qapp, monkeypatch):
    window = FitWizardWindow()
    window.set_analysis_context(_variable_line())
    view = window._spectral_preview.observation

    def forbidden(*args, **kwargs):
        raise AssertionError("the shared observation was prepared again")

    monkeypatch.setattr(diagnostic, "prepare_histogram_asymmetry", forbidden)
    window._picker.set_scope(WizardScope(physics=frozenset({PhysicsClass.GENERIC_RELAXATION})))
    window._analyze_btn.click()
    try:
        wait_for(lambda: window._tasks.active_count == 0, qapp, timeout_s=30)
        result = window.current_recommendation()
        assert result is not None, window._status_label.text()
        assert (
            result.fingerprint.diagnostic_attribution["source_observation"]
            == view.attribution.to_dict()["source_observation"]
        )
        assert result.assessments
        window._populate_fingerprint_plot()
    finally:
        window.close()


def test_global_setup_lists_scope_without_numerical_work(qapp, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("global setup evaluated spectral observations")

    monkeypatch.setattr(diagnostic, "prepare_histogram_asymmetry", forbidden)
    first = _variable_line()
    second = replace(first, run=replace(first.run, run_number=2))
    window = GlobalFitWizardWindow()
    try:
        window.set_analysis_context([first, second])
        names = window._expectation_parameter_names
        assert names
        # Stretched relaxation is a later-stage member, still available at setup.
        assert "beta" in names
        assert window._tasks.active_count == 0
    finally:
        window.close()


@pytest.mark.parametrize("revision", ["current", "older", None])
def test_actual_project_save_open_resave_retains_diagnostic_history_without_computation(
    qapp, tmp_path, monkeypatch, revision
):
    import asymmetry.core.fourier.fft as fft_module
    from asymmetry.core import computation
    from asymmetry.core.fitting import fit_wizard as single
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.io import load
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from asymmetry.core.project.schema import load_project, save_project
    from asymmetry.gui.mainwindow import MainWindow

    source = tmp_path / "owned.nxs"
    original = _variable_line().run
    original.histograms = [replace(h, counts=np.rint(h.counts)) for h in original.histograms]
    write_nexus_v1(original, source)
    loaded = load(source)
    loaded.run.grouping.update(original.grouping)
    dataset = observed(loaded.run)
    template = single.build_null_baseline_templates()[0]
    recommendation = single.build_fit_wizard_recommendation_for_templates(
        dataset, (template,), variant_budget=1
    )
    first, second = MainWindow(), MainWindow()
    try:
        first._data_browser.add_dataset(dataset)
        first._on_dataset_selected(dataset.run_number)
        tab = first._fit_panel._single_tab
        tab._apply_fit_wizard_assessment(recommendation.assessments[0], recommendation)
        captured_revision = computation.COMPUTATION_REVISION if revision == "current" else revision
        historical = replace(recommendation, computation_revision=captured_revision)
        tab._on_fit_wizard_analysis_cached(
            historical, "captured diagnostic", tab._wizard_context_signature()
        )
        first._record_single_fit_slot(recommendation.assessments[0].fit_result)
        path = tmp_path / "owned.asymp"
        save_project(first.collect_project_state(), path)
        before = load_project(path)

        def forbidden(*args, **kwargs):
            raise AssertionError("loading history performed a new fit or FFT")

        monkeypatch.setattr(FitEngine, "fit", forbidden)
        monkeypatch.setattr(FitEngine, "global_fit", forbidden)
        monkeypatch.setattr(fft_module, "_fft_complex_arrays", forbidden)
        monkeypatch.setattr(diagnostic, "prepare_spectral_preview", forbidden)
        second.restore_project_state(before, str(path))
        save_project(second.collect_project_state(), path)
        after = load_project(path)
        # The actual stored recipe, spectral attribution and historical scores survive.
        import json

        from asymmetry.core.project.json_values import encode_non_finite

        text = json.dumps(encode_non_finite(after), sort_keys=True)
        assert "spectral_residual_attribution" in text
        assert "separate_fixed_count_reduction" in text
        assert "fixed_native_support" in text

        def recommendations(value):
            if isinstance(value, dict):
                if "recommendation" in value and isinstance(value["recommendation"], dict):
                    yield value["recommendation"]
                for child in value.values():
                    yield from recommendations(child)
            elif isinstance(value, list):
                for child in value:
                    yield from recommendations(child)

        saved = list(recommendations(after))
        assert saved
        restored = single.deserialize_fit_wizard_recommendation(saved[0])
        assert restored.computation_revision == captured_revision
        assert (
            restored.fingerprint.diagnostic_attribution
            == historical.fingerprint.diagnostic_attribution
        )
        assert (
            restored.assessments[0].fit_result.chi_squared
            == historical.assessments[0].fit_result.chi_squared
        )
    finally:
        first._dirty = second._dirty = False
        first.close()
        second.close()


@pytest.mark.parametrize(
    "terminal", ["finished_close", "error_close", "error_cancel", "live_error"]
)
def test_actual_queued_preview_terminal_cancellation_and_live_error_recovery(
    qapp, monkeypatch, terminal
):
    from PySide6.QtCore import Qt

    from asymmetry.gui.tasks import TaskRunner

    entered, release, emitted = threading.Event(), threading.Event(), threading.Event()
    original_prepare, original_start = diagnostic.prepare_histogram_asymmetry, TaskRunner.start

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if terminal != "finished_close":
            raise RuntimeError("owned preview error remains visible when current")
        return original_prepare(*args, **kwargs)

    def start(self, *args, **kwargs):
        worker = original_start(self, *args, **kwargs)
        signal = worker.finished if terminal == "finished_close" else worker.error
        signal.connect(lambda *args: emitted.set(), Qt.ConnectionType.DirectConnection)
        return worker

    monkeypatch.setattr(diagnostic, "prepare_histogram_asymmetry", held)
    monkeypatch.setattr(TaskRunner, "start", start)
    window, timer = FitWizardWindow(), QTimer()
    terminal_handled = threading.Event()

    def deliver():
        if not entered.is_set() or terminal_handled.is_set():
            return
        terminal_handled.set()
        release.set()
        assert emitted.wait(5), "actual worker terminal signal was not emitted"
        # Keep the GUI event loop held until the real signal is queued, then cancel/close.
        if terminal.endswith("close"):
            window.close()
        elif terminal == "error_cancel":
            QApplication.activeModalWidget().findChild(QPushButton).click()

    timer.timeout.connect(deliver)
    timer.start(1)
    try:
        if terminal == "live_error":
            with pytest.raises(ValueError, match="owned preview error remains visible"):
                window.set_analysis_context(_variable_line())
        else:
            window.set_analysis_context(_variable_line())
        assert terminal_handled.is_set() and emitted.is_set()
        assert window._spectral_preview is None
    finally:
        release.set()
        timer.stop()
        monkeypatch.setattr(diagnostic, "prepare_histogram_asymmetry", original_prepare)
        monkeypatch.setattr(TaskRunner, "start", original_start)
    window.set_analysis_context(_variable_line())
    assert window._spectral_preview is not None
    window.close()
