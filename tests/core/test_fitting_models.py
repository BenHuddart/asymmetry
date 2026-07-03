"""Tests for built-in fit model functions and registry."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.models import (
    MODELS,
    exponential_relaxation,
    gaussian_relaxation,
    oscillatory,
    static_gkt_zf,
    stretched_exponential,
)


def test_model_functions_return_finite_arrays() -> None:
    t = np.linspace(0.0, 5.0, 50)

    y1 = exponential_relaxation(t, A0=25.0, Lambda=0.5)
    y2 = gaussian_relaxation(t, A0=25.0, sigma=0.3)
    y3 = oscillatory(t, A0=25.0, frequency=1.0, Lambda=0.2)
    y4 = stretched_exponential(t, A0=25.0, Lambda=0.4, beta=0.8)
    y5 = static_gkt_zf(t, A0=25.0, Delta=0.2)

    for y in (y1, y2, y3, y4, y5):
        assert y.shape == t.shape
        assert np.all(np.isfinite(y))


def test_registry_contains_expected_models_and_defaults() -> None:
    expected = {
        "ExponentialRelaxation",
        "GaussianRelaxation",
        "Oscillatory",
        "StretchedExponential",
        "StaticGKT_ZF",
    }
    assert expected.issubset(set(MODELS))

    osc = MODELS["Oscillatory"]
    assert "frequency" in osc.param_names
    assert osc.param_defaults["A0"] == 25.0
