"""Core muon Knight-shift conversion (Phase 3a)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.fitting.knight_shift import (
    MUON_LARMOR_MHZ_PER_G,
    KnightShiftUnit,
    clogston_jaccarino_fit,
    concrete_unit,
    knight_shift,
    label_for_unit,
    larmor_frequency_mhz,
    resolve_auto_unit,
    scale_for_unit,
)


def test_larmor_frequency_matches_gamma_times_field():
    # γ_µ ≈ 0.013554 MHz/G; 7000 G → ~94.9 MHz.
    assert larmor_frequency_mhz(7000.0) == pytest.approx(MUON_LARMOR_MHZ_PER_G * 7000.0)
    assert larmor_frequency_mhz(7000.0) == pytest.approx(94.877, abs=1e-3)
    assert larmor_frequency_mhz(0.0) == 0.0


def test_knight_shift_value():
    nu_ref = larmor_frequency_mhz(7000.0)
    nu = nu_ref * 1.0023  # a 2300 ppm shift
    k, _ = knight_shift(nu, nu_ref)
    assert k == pytest.approx(0.0023, rel=1e-9)


def test_applied_field_error_is_sigma_over_ref():
    # Exact reference (γ_µ·B): σ_K = σ_ν / ν_ref.
    nu_ref = larmor_frequency_mhz(7000.0)
    _, sigma_k = knight_shift(nu_ref * 1.001, nu_ref, sigma_nu=0.01)
    assert sigma_k == pytest.approx(0.01 / nu_ref, rel=1e-9)


def test_positive_covariance_reduces_uncertainty():
    # For a measured reference, ν and ν_ref are correlated; a positive covariance
    # shrinks the shift uncertainty (the shared error partly cancels in the ratio).
    common = dict(sigma_nu=0.1, sigma_ref=0.1)
    _, sigma_no_cov = knight_shift(10.0, 10.0, cov=0.0, **common)
    _, sigma_pos_cov = knight_shift(10.0, 10.0, cov=0.005, **common)
    assert sigma_no_cov == pytest.approx(math.sqrt(0.0002))
    assert sigma_pos_cov == pytest.approx(math.sqrt(0.0001))
    assert sigma_pos_cov < sigma_no_cov


def test_genuinely_negative_variance_is_nan():
    # A covariance large enough to drive the variance meaningfully below zero is
    # ill-posed; surface NaN rather than a misleadingly precise zero uncertainty.
    _, sigma_k = knight_shift(10.0, 10.0, sigma_nu=0.1, sigma_ref=0.1, cov=1.0)
    assert math.isnan(sigma_k)


def test_roundoff_near_zero_variance_is_finite():
    # Perfectly correlated equal errors nearly cancel in the ratio; the result is a
    # finite (≈0) uncertainty, never NaN, despite floating-point round-off.
    _, sigma_k = knight_shift(10.0, 10.0, sigma_nu=0.1, sigma_ref=0.1, cov=0.01)
    assert math.isfinite(sigma_k)
    assert sigma_k == pytest.approx(0.0, abs=1e-6)


def test_zero_reference_is_nan():
    k, sigma_k = knight_shift(10.0, 0.0, sigma_nu=0.1)
    assert math.isnan(k)
    assert math.isnan(sigma_k)


def test_auto_unit_picks_ppm_for_small_shifts():
    assert resolve_auto_unit([1e-5, 5e-5, -3e-5]) is KnightShiftUnit.PPM


def test_auto_unit_picks_percent_for_large_shifts():
    assert resolve_auto_unit([1e-5, 0.02, -0.01]) is KnightShiftUnit.PERCENT


def test_auto_unit_empty_defaults_to_ppm():
    assert resolve_auto_unit([]) is KnightShiftUnit.PPM
    assert resolve_auto_unit([float("nan"), float("inf")]) is KnightShiftUnit.PPM


def test_concrete_unit_resolves_auto_but_passes_explicit_through():
    assert concrete_unit(KnightShiftUnit.AUTO, [1e-5]) is KnightShiftUnit.PPM
    assert concrete_unit(KnightShiftUnit.PERCENT, [1e-5]) is KnightShiftUnit.PERCENT


def test_clogston_jaccarino_recovers_slope_and_intercept():
    chi = [1.0, 2.0, 3.0, 4.0, 5.0]
    knight = [2.0 * c + 3.0 for c in chi]  # K = 2χ + 3
    result = clogston_jaccarino_fit(chi, knight)
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(3.0)
    assert result.n == 5


def test_clogston_jaccarino_unweighted_errors_follow_residual_scatter_and_units():
    chi = np.array([1.0, 2.0, 3.0, 4.0])
    knight_fraction = np.array([0.001, 0.0021, 0.0029, 0.0042])
    design = np.column_stack((chi, np.ones_like(chi)))
    results = []
    for knight in (knight_fraction, 1.0e6 * knight_fraction):
        coefficients, sse, rank, _ = np.linalg.lstsq(design, knight, rcond=None)
        covariance = (sse.item() / (chi.size - rank)) * np.linalg.inv(design.T @ design)
        result = clogston_jaccarino_fit(chi, knight)
        values = np.array([result.slope, result.intercept, result.slope_err, result.intercept_err])
        np.testing.assert_allclose(
            values, np.concatenate((coefficients, np.sqrt(covariance.diagonal())))
        )
        results.append(values)

    fraction, ppm = results
    np.testing.assert_allclose(ppm, 1.0e6 * fraction)


def test_clogston_jaccarino_two_unweighted_points_have_unavailable_errors():
    result = clogston_jaccarino_fit([1.0, 2.0, 3.0], [5.0, 7.0, np.nan])
    assert result.n == 2
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(3.0)
    assert math.isnan(result.slope_err)
    assert math.isnan(result.intercept_err)


@pytest.mark.parametrize("n", [2, 4])
def test_clogston_jaccarino_uses_absolute_measurement_errors(n):
    chi = np.array([1.0, 2.0, 3.0, 4.0])[:n]
    knight = np.array([0.001, 0.0021, 0.0029, 0.0042])[:n]
    sigma = np.array([0.0002, 0.0001, 0.0003, 0.00015])[:n]
    weighted_design = np.column_stack((chi, np.ones_like(chi))) / sigma[:, None]
    coefficients = np.linalg.lstsq(weighted_design, knight / sigma, rcond=None)[0]
    covariance = np.linalg.inv(weighted_design.T @ weighted_design)

    result = clogston_jaccarino_fit(
        np.append(chi, [np.nan, 5.0, 6.0, 7.0, 8.0, 9.0]),
        np.append(knight, [0.5, np.inf, 0.5, 0.5, 0.5, 0.5]),
        np.append(sigma, [0.0001, 0.0001, 0.0, -0.0001, np.inf, np.nan]),
    )

    assert result.n == n
    np.testing.assert_allclose([result.slope, result.intercept], coefficients)
    np.testing.assert_allclose(
        [result.slope_err, result.intercept_err], np.sqrt(covariance.diagonal())
    )


def test_clogston_jaccarino_weighting_downweights_outlier():
    chi = [1.0, 2.0, 3.0, 4.0]
    knight = [5.0, 7.0, 9.0, 100.0]  # last point is a wild outlier (true line K=2χ+3)
    sigma = [0.1, 0.1, 0.1, 1e6]  # but with a huge error bar
    result = clogston_jaccarino_fit(chi, knight, sigma)
    assert result.slope == pytest.approx(2.0, abs=1e-3)
    assert result.intercept == pytest.approx(3.0, abs=1e-3)


def test_clogston_jaccarino_drops_non_finite_points():
    chi = [1.0, 2.0, float("nan"), 3.0]
    knight = [5.0, 7.0, 0.0, 9.0]
    result = clogston_jaccarino_fit(chi, knight)
    assert result.n == 3
    assert result.slope == pytest.approx(2.0)


def test_clogston_jaccarino_constant_chi_raises():
    with pytest.raises(ValueError):
        clogston_jaccarino_fit([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])


def test_clogston_jaccarino_too_few_points_raises():
    with pytest.raises(ValueError):
        clogston_jaccarino_fit([1.0], [2.0])


def test_scale_and_label():
    assert scale_for_unit(KnightShiftUnit.PPM) == 1.0e6
    assert scale_for_unit(KnightShiftUnit.PERCENT) == 100.0
    assert scale_for_unit(KnightShiftUnit.FRACTION) == 1.0
    assert label_for_unit(KnightShiftUnit.PPM) == "ppm"
    assert label_for_unit(KnightShiftUnit.PERCENT) == "%"
    assert label_for_unit(KnightShiftUnit.FRACTION) == ""
