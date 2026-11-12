"""Independent scalar-observation response checks for FIR magnitude errors."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.transform.rrf import rrf_demodulate, rrf_demodulate_values
from asymmetry.core.transform.units import PERCENT_PER_FRACTION


@pytest.mark.parametrize("phase", [0.0, 45.0, -37.0, 90.0])
def test_identity_magnitude_noise_is_phase_invariant(phase):
    t = np.arange(101) * 0.005
    curve = rrf_demodulate(
        t,
        np.full(t.size, 10.0),
        np.ones(t.size),
        frequency_mhz=1.0,
        bandwidth_mhz=100.0,
        phase_deg=phase,
    )
    assert curve.filter_taps == 1
    np.testing.assert_allclose(curve.magnitude, 20.0, rtol=1e-14)
    np.testing.assert_allclose(curve.magnitude_error, 2.0, rtol=1e-14)


@pytest.mark.parametrize("bandwidth,frequency,n", [(16.0, 8.0, 121), (None, 20.0, 401)])
@pytest.mark.parametrize("holes", [False, True])
def test_fir_noise_matches_independent_linear_response(bandwidth, frequency, n, holes):
    t = np.arange(n) * 0.01
    a = 12 * np.cos(2 * np.pi * (frequency + 0.3) * t + 0.6)
    e = 1 + 2 * np.exp(-(((t - t[n // 2]) / 0.025) ** 2))
    if holes:
        a[n // 2 - 2 : n // 2 + 1] = np.nan
        e[n // 2 + 2] = np.nan
    options = dict(frequency_mhz=frequency, phase_deg=37.0, bandwidth_mhz=bandwidth)
    curve = rrf_demodulate(t, a, e, **options)
    # Unit impulses measure the value operator without using variance code or
    # its kernel. Keep the same finite mask for every response column.
    missing = ~np.isfinite(a) | ~np.isfinite(e)
    response = []
    for j in range(n):
        impulse = np.zeros(n)
        impulse[j] = 1.0
        impulse[missing] = np.nan
        output = rrf_demodulate_values(t, impulse, **options)
        response.append(output.real + 1j * output.imag)
    linear = np.asarray(response).T
    noise = linear * np.where(missing, 0.0, e)
    expected_re = np.sum(noise.real**2, axis=1)
    expected_im = np.sum(noise.imag**2, axis=1)
    expected_cov = np.sum(noise.real * noise.imag, axis=1)
    np.testing.assert_allclose(curve.real_error**2, expected_re, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(curve.imag_error**2, expected_im, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(curve.real_imag_covariance, expected_cov, atol=2e-14, rtol=2e-12)
    assert np.min(expected_cov) < 0 < np.max(expected_cov)
    expected_values = linear @ np.where(missing, 0.0, a)
    np.testing.assert_allclose(curve.real + 1j * curve.imag, expected_values, atol=1e-13)
    direction = (curve.real - 1j * curve.imag) / curve.magnitude
    expected_error = np.sqrt(np.sum((direction[:, None] * noise).real ** 2, axis=1))
    np.testing.assert_allclose(curve.magnitude_error, expected_error, rtol=2e-12, atol=2e-14)
    assert not curve.valid[0] and not curve.valid[-1]
    assert curve.valid[n // 2 + 10]
    rotated = rrf_demodulate(t, a, e, **dict(options, phase_deg=-68.0))
    np.testing.assert_array_equal(curve.valid, rotated.valid)
    np.testing.assert_allclose(curve.magnitude, rotated.magnitude, atol=1e-13)
    np.testing.assert_allclose(curve.magnitude_error, rotated.magnitude_error, atol=2e-13)
    scaled = rrf_demodulate(t, a * PERCENT_PER_FRACTION, e * PERCENT_PER_FRACTION, **options)
    np.testing.assert_allclose(scaled.magnitude, curve.magnitude * PERCENT_PER_FRACTION, atol=1e-11)
    np.testing.assert_allclose(
        scaled.magnitude_error, curve.magnitude_error * PERCENT_PER_FRACTION, atol=1e-11
    )
    np.testing.assert_allclose(
        scaled.real_imag_covariance, expected_cov * PERCENT_PER_FRACTION**2, atol=2e-10
    )
    overlay = rrf_demodulate_values(t, a, **options)
    for errors in (
        overlay.real_error,
        overlay.imag_error,
        overlay.magnitude_error,
        overlay.real_imag_covariance,
    ):
        np.testing.assert_array_equal(errors, 0.0)


def test_zero_magnitude_fallback_and_invalid_covariance():
    t = np.arange(11) * 0.01
    curve = rrf_demodulate(
        t, np.zeros(t.size), np.ones(t.size), frequency_mhz=1.0, bandwidth_mhz=100.0
    )
    np.testing.assert_allclose(curve.magnitude_error, np.sqrt(2.0))
    nonzero = replace(
        curve,
        real=np.ones(t.size),
        imag=np.ones(t.size),
        real_error=np.ones(t.size),
        imag_error=np.ones(t.size),
    )
    roundoff = replace(nonzero, real_imag_covariance=np.full(t.size, -1.0 - np.finfo(float).eps))
    np.testing.assert_array_equal(roundoff.magnitude_error, 0.0)
    invalid = replace(nonzero, real_imag_covariance=np.full(t.size, -2.0))
    with pytest.raises(RuntimeError, match="magnitude variance"):
        _ = invalid.magnitude_error


def test_near_zero_magnitude_retains_first_order_error():
    t = np.arange(11) * 0.01
    curve = rrf_demodulate(
        t,
        np.full(t.size, 1e-200),
        np.ones(t.size),
        frequency_mhz=1.0,
        bandwidth_mhz=100.0,
        phase_deg=45.0,
    )
    np.testing.assert_allclose(curve.magnitude_error, 2.0, rtol=1e-14)


def test_wimda_comparison_has_zero_cross_covariance():
    t = np.arange(101) * 0.01
    options = dict(frequency_mhz=8.0, phase_deg=37.0, method="wimda")
    curve = rrf_demodulate(t, np.ones(t.size), np.ones(t.size), **options)
    np.testing.assert_array_equal(curve.real_imag_covariance, 0.0)
    np.testing.assert_allclose(curve.magnitude_error[curve.valid], curve.real_error[curve.valid])
    overlay = rrf_demodulate_values(t, np.ones(t.size), **options)
    np.testing.assert_array_equal(overlay.magnitude_error, 0.0)
    np.testing.assert_array_equal(overlay.real_imag_covariance, 0.0)
