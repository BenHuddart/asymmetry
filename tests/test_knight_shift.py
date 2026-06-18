"""Core muon Knight-shift conversion (Phase 3a)."""

from __future__ import annotations

import math

import pytest

from asymmetry.core.fitting.knight_shift import (
    MUON_LARMOR_MHZ_PER_G,
    KnightShiftUnit,
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


def test_scale_and_label():
    assert scale_for_unit(KnightShiftUnit.PPM) == 1.0e6
    assert scale_for_unit(KnightShiftUnit.PERCENT) == 100.0
    assert scale_for_unit(KnightShiftUnit.FRACTION) == 1.0
    assert label_for_unit(KnightShiftUnit.PPM) == "ppm"
    assert label_for_unit(KnightShiftUnit.PERCENT) == "%"
    assert label_for_unit(KnightShiftUnit.FRACTION) == ""
