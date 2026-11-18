"""Real workspace transitions and failure settlement for moment tasks."""

import threading

import pytest

from asymmetry.core.representation.base import RepresentationType
from tests.gui.test_spectral_moments_gui import (
    _activate_fft_spectrum,
    _await_event,
    _moments_batches,
    _spectrum,
    _wait_moments,
)
from tests.gui.test_spectral_moments_gui import (
    window as window,
)

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("source,target", [("frequency", "maxent"), ("maxent", "frequency")])
@pytest.mark.parametrize("stage", ["preparation", "extraction"])
def test_representation_change_cancels_moment_batch(window, monkeypatch, source, target, stage):
    import asymmetry.gui.moment_tasks as tasks

    monkeypatch.setattr(window, "_dataset_supports_maxent", lambda dataset: True)
    browser = window._data_browser
    datasets = _activate_fft_spectrum(window, [1])
    window._data_browser = browser
    window._current_dataset = datasets[0]
    browser.add_dataset(datasets[0])
    browser.select_runs([1])
    window._frequency_cache(RepresentationType.FREQ_MAXENT)[1] = [_spectrum(1)]
    window._plot_workspace.set_available_views(["fb_asymmetry", "frequency", "maxent"])
    window._plot_workspace.set_active_view(source)
    _wait_moments(window)
    entered, release = threading.Event(), threading.Event()
    continuations = []
    original = tasks.moments_trend_row

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    if stage == "preparation":
        monkeypatch.setattr(
            window,
            "_ensure_frequency_spectra_for_runs_async",
            lambda *args, on_ready: continuations.append(on_ready),
        )
    else:
        monkeypatch.setattr(tasks, "moments_trend_row", gated)
    window._on_moments_send_to_trend(window._active_moments_widget())
    if stage == "extraction":
        _await_event(entered)
    try:
        window._plot_workspace.set_active_view(target)
        assert window._plot_workspace.active_view() == target
        assert window._moment_batch._intent is None
        if stage == "preparation":
            continuations.pop()()
            assert window._moment_batch._worker is None
    finally:
        release.set()
    _wait_moments(window)
    assert _moments_batches(window) == []


def test_current_readout_failure_restores_points_and_can_retry(window, monkeypatch):
    import asymmetry.gui.moment_tasks as tasks

    _activate_fft_spectrum(window, [1])
    original = tasks.spectrum_moments
    notes, full_calls = [], []

    def fail_once(*args, **kwargs):
        if kwargs["uncertainty"] == "bootstrap":
            full_calls.append(None)
            if len(full_calls) == 1:
                raise ValueError("Synthetic uncertainty failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(tasks, "spectrum_moments", fail_once)
    monkeypatch.setattr(window, "_set_fourier_status", notes.append)
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.2)
    window._on_moments_settings_changed(widget)
    points = widget._value_labels["b_ave"].text()
    _wait_moments(window)
    assert widget._points_label.text().startswith("Points:")
    assert widget._value_labels["b_ave"].text() == points
    assert "±" not in points
    assert "Synthetic uncertainty failure" in notes[-1]
    window._on_moments_settings_changed(widget)
    _wait_moments(window)
    assert len(full_calls) == 2
    assert "±" in widget._value_labels["b_ave"].text()


def test_obsolete_readout_error_preserves_new_pending_points_and_status(window, monkeypatch):
    import asymmetry.gui.moment_tasks as tasks

    _activate_fft_spectrum(window, [1])
    entered, release, replacement_entered, replacement_release = (
        threading.Event() for _ in range(4)
    )
    original = tasks.spectrum_moments
    notes = []

    def gated(*args, **kwargs):
        if kwargs["uncertainty"] == "bootstrap":
            if kwargs["cutoff_fraction"] == 0.1:
                entered.set()
                assert release.wait(5)
                raise ValueError("Obsolete uncertainty failure")
            replacement_entered.set()
            assert replacement_release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(tasks, "spectrum_moments", gated)
    monkeypatch.setattr(window, "_set_fourier_status", notes.append)
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.1)
    window._on_moments_settings_changed(widget)
    _await_event(entered)
    try:
        widget.set_cutoff_fraction(0.2)
        window._on_moments_settings_changed(widget)
        points = widget._value_labels["b_ave"].text()
        notes.append("Current request status")
        release.set()
        _await_event(replacement_entered)
        assert widget._points_label.text() == "Calculating approximate errors…"
        assert widget._value_labels["b_ave"].text() == points
        assert notes == ["Current request status"]
    finally:
        release.set()
        replacement_release.set()
    _wait_moments(window)
    assert "±" in widget._value_labels["b_ave"].text()
