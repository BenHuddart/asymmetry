"""GUI: pull-distribution diagnostic window and its fit-panel launch hook."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("iminuit")

from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.pull_diagnostic import ParameterPull, PullDistribution
from asymmetry.core.simulate import (
    build_builtin_template,
    reduce_run_to_dataset,
    simulate_run,
    total_events_of,
)
from asymmetry.gui.windows import pull_diagnostic_window as pull_diagnostic_window_module
from asymmetry.gui.windows.pull_diagnostic_window import PullDiagnosticWindow, make_engine_refit
from tests._qt_helpers import wait_for


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _exp_composite() -> CompositeModel:
    return CompositeModel(["Exponential", "Constant"], operators=["+"])


def test_window_runs_and_centres_pulls(qapp) -> None:
    template = build_builtin_template("ideal_pulsed_fb")
    model = _exp_composite()
    # Generating values on the composite's mangled names (A_1, Lambda, A_bg).
    truth = {"A_1": 20.0, "Lambda": 0.5, "A_bg": 0.0}
    params = ParameterSet(
        [
            Parameter(name="A_1", value=15.0, min=0.0, max=100.0),
            Parameter(name="Lambda", value=1.0, min=0.0, max=10.0),
            Parameter(name="A_bg", value=0.0, min=-10.0, max=10.0, fixed=True),
        ]
    )
    refit = make_engine_refit(model, params, t_min=0.0, t_max=8.0)
    window = PullDiagnosticWindow(
        template=template,
        model=model,
        parameters=truth,
        refit=refit,
        track=["A_1", "Lambda"],
        total_events=20.0e6,
        time_range=(0.0, 8.0),
    )
    result = window.run_diagnostic(40)
    assert result.n_converged >= 30
    for name in ("A_1", "Lambda"):
        pull = result.parameters[name]
        # Loose tolerance for N=40: mean near 0, width near 1.
        assert abs(pull.mean) < 0.5, name
        assert 0.6 < pull.width < 1.6, name
    assert "converged" in window.last_result.verdict()


def _make_window(
    *, track: list[str], parameters: dict[str, float] | None = None
) -> PullDiagnosticWindow:
    template = build_builtin_template("ideal_pulsed_fb")
    model = _exp_composite()
    truth = parameters if parameters is not None else {"A_1": 20.0, "Lambda": 0.5, "A_bg": 0.0}
    params = ParameterSet(
        [
            Parameter(name="A_1", value=15.0, min=0.0, max=100.0),
            Parameter(name="Lambda", value=1.0, min=0.0, max=10.0),
            Parameter(name="A_bg", value=0.0, min=-10.0, max=10.0, fixed=True),
        ]
    )
    refit = make_engine_refit(model, params, t_min=0.0, t_max=8.0)
    return PullDiagnosticWindow(
        template=template,
        model=model,
        parameters=truth,
        refit=refit,
        track=track,
        total_events=20.0e6,
        time_range=(0.0, 8.0),
    )


def test_run_button_runs_off_the_gui_thread_and_populates_results(qapp, monkeypatch) -> None:
    """The GUI-thread invariant: clicking Run must not execute the diagnostic
    on the calling thread — this is the exact anti-pattern (processEvents
    keep-alive) the fix removes. A spy wraps the real
    run_pull_distribution so the test both records which thread ran it and
    exercises the genuine simulate+refit path end to end."""
    seen: dict[str, threading.Thread] = {}
    real_run = pull_diagnostic_window_module.run_pull_distribution

    def _spy(*args, **kwargs):
        seen["thread"] = threading.current_thread()
        return real_run(*args, **kwargs)

    monkeypatch.setattr(pull_diagnostic_window_module, "run_pull_distribution", _spy)

    window = _make_window(track=["A_1", "Lambda"])
    window._seeds_spin.setValue(10)
    window._on_run()
    assert window._running is True
    assert window._run_button.isEnabled() is False
    assert window._cancel_button.isEnabled() is True

    wait_for(lambda: not window._running, qapp, timeout_s=20.0)

    assert seen["thread"] is not threading.main_thread()
    assert window.last_result is not None
    assert window.last_result.n_seeds == 10
    assert window._run_button.isEnabled() is True
    assert window._cancel_button.isEnabled() is False
    assert window._verdict_label.text() != "Running…"


def test_cancel_mid_run_stops_promptly_and_restores_ui(qapp, monkeypatch) -> None:
    """Cancel gives immediate UI feedback (button disabled) before the gated
    worker call ever returns, then the finished result — a normal (shorter)
    PullDistribution, since run_pull_distribution has no cancel exception —
    is reported as cancelled and the Run/Cancel buttons are restored."""
    gate = threading.Event()

    def _fake_run(*_args, **_kwargs):
        gate.wait(timeout=2.0)
        return PullDistribution(
            parameters={
                "A_1": ParameterPull(name="A_1", truth=20.0, pulls=np.array([0.1, -0.2])),
                "Lambda": ParameterPull(name="Lambda", truth=0.5, pulls=np.array([0.05, -0.1])),
            },
            truth={"A_1": 20.0, "Lambda": 0.5},
            n_seeds=2,
            n_converged=2,
            total_events=1.0e6,
        )

    monkeypatch.setattr(pull_diagnostic_window_module, "run_pull_distribution", _fake_run)

    window = _make_window(track=["A_1", "Lambda"])
    try:
        window._on_run()
        wait_for(lambda: window._running, qapp, timeout_s=5.0)
        assert window._cancel_button.isEnabled() is True

        window._on_cancel()
        # Immediate feedback: the button disables before the gated worker
        # call has returned anything at all.
        assert window._cancel_button.isEnabled() is False
        assert window._running is True
    finally:
        gate.set()

    wait_for(lambda: not window._running, qapp, timeout_s=5.0)

    assert window._cancel_requested is True
    assert window._run_button.isEnabled() is True
    assert window._cancel_button.isEnabled() is False
    assert "Cancelled" in window._verdict_label.text()


def test_close_mid_run_shuts_down_without_crashing(qapp, monkeypatch) -> None:
    """Closing the window mid-run must cancel and tear the worker down
    cleanly instead of crashing or leaving a dangling thread — the Close
    button can now stay enabled throughout the run."""

    def _fake_run(*_args, should_continue=None, **_kwargs):
        # Poll cooperatively like the real engine so the bounded shutdown()
        # started from closeEvent/reject() returns promptly.
        for _ in range(500):
            if should_continue is not None and not should_continue():
                break
            time.sleep(0.01)
        return PullDistribution(parameters={}, truth={}, n_seeds=0, n_converged=0, total_events=0.0)

    monkeypatch.setattr(pull_diagnostic_window_module, "run_pull_distribution", _fake_run)

    window = _make_window(track=[])
    assert window._button_box.isEnabled()

    window._on_run()
    wait_for(lambda: window._running, qapp, timeout_s=5.0)

    window.close()  # native close path: closeEvent -> reject() -> shutdown()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert window._tasks.active_count == 0


def test_total_events_of_sums_histograms(qapp) -> None:
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_run(
        template,
        lambda t, a0=10.0: a0 * np.ones_like(t),
        total_events=5.0e6,
        seed=1,
    )
    total = total_events_of(run)
    assert total == sum(float(h.counts.sum()) for h in run.histograms)
    assert total > 0


class TestFitPanelHook:
    def test_button_enables_after_successful_fit_and_opens_window(self, qapp) -> None:
        from asymmetry.gui.panels.fit_panel import SingleFitTab

        tab = SingleFitTab()
        template = build_builtin_template("ideal_pulsed_fb")
        run = simulate_run(
            template,
            lambda t, a0=20.0, rate=0.4: a0 * np.exp(-rate * t),
            {"a0": 20.0, "rate": 0.4},
            total_events=20.0e6,
            seed=2,
        )
        dataset = reduce_run_to_dataset(run)
        tab.set_dataset(dataset)
        # No fit yet → diagnostic disabled.
        assert not tab._pull_diagnostic_btn.isEnabled()
        assert not tab._can_run_pull_diagnostic()

        # Simulate a converged fit having run on this dataset. The fit engine
        # does NOT mutate the input set, so _last_fit_parameters holds the
        # pre-fit START guesses while _last_fit_result.parameters holds the
        # converged values — they must differ here to catch the truth-source bug.
        tab._composite_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
        tab._last_fit_parameters = ParameterSet(
            [
                Parameter(name="A_1", value=10.0, min=0.0, max=100.0),  # start guess
                Parameter(name="Lambda", value=1.5, min=0.0, max=10.0),  # start guess
                Parameter(name="A_bg", value=0.0, min=-10.0, max=10.0, fixed=True),
            ]
        )

        class _Result:
            success = True
            parameters = ParameterSet(
                [
                    Parameter(name="A_1", value=20.0),  # converged
                    Parameter(name="Lambda", value=0.4),  # converged
                    Parameter(name="A_bg", value=0.0),
                ]
            )

        tab._last_fit_result = _Result()
        assert tab._can_run_pull_diagnostic()

        tab._on_pull_diagnostic()
        window = tab._pull_diagnostic_window
        assert isinstance(window, PullDiagnosticWindow)
        # Truth must be the CONVERGED values, not the start guesses.
        assert window._parameters["Lambda"] == pytest.approx(0.4)
        assert window._parameters["A_1"] == pytest.approx(20.0)

    def test_changing_dataset_clears_fit_result(self, qapp) -> None:
        from asymmetry.gui.panels.fit_panel import SingleFitTab

        tab = SingleFitTab()

        class _Result:
            success = True

        tab._last_fit_result = _Result()
        tab._last_fit_parameters = ParameterSet([Parameter(name="x", value=1.0)])
        tab.set_dataset(None)
        assert tab._last_fit_result is None
        assert not tab._pull_diagnostic_btn.isEnabled()
