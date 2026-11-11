"""Fixed double-delta request domains and an independent causal count control."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.count_domain import fit_fb_alpha, fit_single_histogram
from asymmetry.core.fitting.parameters import Parameter
from tests.core.test_count_domain_fits import (
    _continuous_single,
    _double_pulse_dataset,
    _fb_double_pulse_dataset,
    _fb_double_pulse_params,
    _tf,
)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("target", ["single", "fb"])
def test_fixed_separation_requires_nonnegative_finite_value(value, target):
    ds = _fb_double_pulse_dataset()
    params = _continuous_single() if target == "single" else _fb_double_pulse_params()
    params.add(Parameter("dpsep", value, fixed=True))
    with pytest.raises(ValueError, match="dpsep.*nonnegative.*finite"):
        if target == "single":
            fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
        else:
            fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")


def test_count_model_matches_independent_causal_birth_sum():
    from asymmetry.core.fitting.count_response import pulse_population
    from asymmetry.core.utils.constants import MUON_LIFETIME_US

    time = np.array([0.0, 0.1, 0.2, 0.5, 1.0])

    def polarization(age):
        return 0.2 * np.exp(-0.7 * age) * np.cos(2 * np.pi * 2.5 * age + 0.3)

    expected = np.zeros_like(time)
    for birth in (-0.162, 0.162):
        age = time - birth
        live = age > 0
        expected[live] += (
            0.5 * np.exp(-age[live] / MUON_LIFETIME_US) * (1 + polarization(age[live]))
        )
    actual = pulse_population(polarization, time, 0.324, 1.0, {})
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)


@pytest.mark.parametrize("target", ["single", "fb"])
@pytest.mark.parametrize("constraint", ["link", "expression"])
def test_fixed_separation_cannot_follow_or_constrain_other_fit_parameters(target, constraint):
    from asymmetry.core.maxent import PulseRequestError

    ds = _fb_double_pulse_dataset()
    params = _continuous_single() if target == "single" else _fb_double_pulse_params()
    separation = Parameter("dpsep", 0.324, fixed=True)
    if constraint == "link":
        params["f"].link_group = 1
        separation.link_group = 1
    else:
        separation.expr = "f"
    params.add(separation)
    with pytest.raises(PulseRequestError, match="dpsep.*fixed.*supplied"):
        if target == "single":
            fit_single_histogram(ds, 1, _tf, params, cost="gaussian")
        else:
            fit_fb_alpha(ds, 1, 2, _tf, params, cost="gaussian")


def test_fixed_separation_singleton_link_is_unused():
    params = _continuous_single()
    params.add(Parameter("dpsep", 0.324, fixed=True, link_group=1))
    result = fit_single_histogram(_double_pulse_dataset(), 1, _tf, params, cost="gaussian")
    assert result.parameters["dpsep"].fixed
    assert result.parameters["dpsep"].value == 0.324
