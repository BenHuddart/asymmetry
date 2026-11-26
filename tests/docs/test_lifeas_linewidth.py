"""Synthetic controls for the LiFeAs worked example."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.results import FitResult
from asymmetry.core.fitting.sc import GaussianWidth


@pytest.fixture
def lifeas():
    pytest.importorskip("PySide6")
    from docs.screenshots.scenarios.corpus import lifeas_pnictide

    return lifeas_pnictide


def test_lifeas_sums_the_selected_counts_before_forming_the_ratio(lifeas):
    run = Run(
        histograms=[
            Histogram(np.array([400, 50, 50, 90, 1, 400]), 0.1, 1, 1, 4),
            Histogram(np.array([400, 50, 50, 10, 9, 400]), 0.1, 1, 1, 4),
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Up", 2: "Down"},
            "forward_group": 2,
            "backward_group": 1,
        },
    )
    dataset = lifeas.reduce_lifeas_run(run, rebin=2, time_range=(0.2, 0.3))

    forward, backward = 91, 19
    total = forward + backward
    # dA/dF = 2B/(F+B)² and dA/dB = −2F/(F+B)², with Poisson variances F and B.
    variance = ((2 * backward) ** 2 * forward + (2 * forward) ** 2 * backward) / total**4
    np.testing.assert_allclose(dataset.time, [0.25])
    np.testing.assert_allclose(dataset.asymmetry / 100, [72 / 110])
    np.testing.assert_allclose((dataset.error / 100) ** 2, [variance])
    assert ((90 - 10) / (90 + 10) + (1 - 9) / (1 + 9)) / 2 == 0


def _fit(sigma):
    return FitResult(
        success=True,
        objective=Objective.GAUSSIAN,
        parameters=ParameterSet([Parameter("sigma", sigma)]),
    )


_FAILED_FIT = FitResult(success=False, objective=Objective.GAUSSIAN)


@pytest.mark.parametrize(
    ("width", "factor"),
    [(GaussianWidth.GAUSSIAN_COMPONENT, 1), (GaussianWidth.SECOND_MOMENT, np.sqrt(2))],
)
def test_lifeas_resolved_width_uses_the_muon_angular_rate_and_gaussian_convention(
    lifeas, width, factor
):
    # 3–4–5 triangle in µs⁻¹. The CODATA value rounded to 135.539 MHz/T gives
    # 2π × 0.135539 rad/(µs mT), so the second-moment rate becomes a width in mT.
    expected_mt = np.sqrt(2) * 4 / (2 * np.pi * 0.135539)
    assert lifeas.vortex_field_width_mt(
        _fit(5 * factor), _fit(3 * factor), sigma_parameter="sigma", width=width
    ) == pytest.approx(expected_mt, rel=2e-6)


@pytest.mark.parametrize(
    ("signal", "normal"),
    [
        (_fit(1), _fit(2)),
        (_fit(2), _fit(2)),
        (_FAILED_FIT, _fit(2)),
        (_fit(3), _FAILED_FIT),
    ],
)
def test_lifeas_unresolved_or_failed_width_is_unavailable(lifeas, signal, normal):
    assert (
        lifeas.vortex_field_width_mt(
            signal, normal, sigma_parameter="sigma", width=GaussianWidth.GAUSSIAN_COMPONENT
        )
        is None
    )
