"""GUI: pull-distribution diagnostic window and its fit-panel launch hook."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("iminuit")

from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.simulate import (
    build_builtin_template,
    reduce_run_to_dataset,
    simulate_run,
    total_events_of,
)
from asymmetry.gui.windows.pull_diagnostic_window import PullDiagnosticWindow, make_engine_refit


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
