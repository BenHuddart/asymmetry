"""Analytic checks for the silicon walkthrough's calibration and lifetime."""

import numpy as np
import pytest

from docs.workflows.silicon_lifetime import (
    CALIBRATION_MODEL,
    DENSITY_UNIT_CM3,
    REFERENCE_DENSITY_CM3,
    fit_calibration,
    fit_carrier_decay,
)


def test_lifetime_uses_displayed_calibration_and_shared_covariance():
    beta, exponent, tau = 1.8, 0.74, 7.3
    reference_density = REFERENCE_DENSITY_CM3
    densities = np.geomspace(0.12, 1.0, 10) * reference_density
    calibration_rate = beta * (densities / reference_density) ** exponent
    calibration_error = np.linspace(0.01, 0.02, len(densities))
    delay = np.linspace(0.2, 12, 9)
    initial_density = 0.93 * reference_density
    rate = (
        beta * (initial_density / reference_density) ** exponent * np.exp(-exponent * delay / tau)
    )
    rate_error = np.full_like(rate, 0.02)

    calibration = fit_calibration(densities, calibration_rate, calibration_error)
    decay = fit_carrier_decay(calibration, delay, rate, rate_error, max_delay_us=20)
    params = {p.name: p.value for p in calibration.parameters}
    density_ratio = decay.density * DENSITY_UNIT_CM3 / reference_density

    assert calibration.parameters["c"].fixed
    assert params["c"] == 0
    assert params["a"] == pytest.approx(beta, rel=5e-4)
    assert params["n"] == pytest.approx(exponent, rel=5e-4)
    np.testing.assert_allclose(CALIBRATION_MODEL.function(density_ratio, **params), rate)
    np.testing.assert_allclose(
        decay.density * DENSITY_UNIT_CM3, initial_density * np.exp(-delay / tau), rtol=5e-4
    )
    np.testing.assert_allclose(
        decay.error,
        density_ratio * reference_density / DENSITY_UNIT_CM3 * rate_error / (params["n"] * rate),
    )
    assert decay.fit.parameters["tau"].value == pytest.approx(tau, rel=5e-4)

    # For this exact decay, ln(rate) has slope -n/tau. Varying the calibration
    # prefactor rescales every density equally and leaves the fitted tau alone.
    names, covariance = calibration.covariance
    index = names.index("n")
    expected_error = tau / exponent * np.sqrt(covariance[index][index])
    assert decay.calibration_tau_error == pytest.approx(expected_error, rel=0.03)


def test_missing_rate_error_cannot_be_used_as_a_weight():
    density = np.linspace(0.1, 1.0, 10) * REFERENCE_DENSITY_CM3
    rate = np.linspace(0.1, 1.0, len(density))
    error = np.where(rate < 0.2, np.nan, 0.02)

    with pytest.raises(ValueError, match="positive finite rate standard errors"):
        fit_calibration(density, rate, error)
