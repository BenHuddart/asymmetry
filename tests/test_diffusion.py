"""Tests for diffusive LF relaxation model."""

from __future__ import annotations

import time

import numpy as np
import pytest

from asymmetry.core.fitting.diffusion import (
    autocorrelation_nD,
    lambda_diff,
    lambda_total,
    spectral_density,
)
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    component_names_for_x,
)


def test_autocorrelation_shape_and_t0_value() -> None:
    t = np.linspace(0.0, 10.0, 101)
    for n in (1, 2, 3):
        s = autocorrelation_nD(t, D_nD=1.2, D_perp=0.5, n=n)
        assert s.shape == t.shape
        assert np.isclose(s[0], 1.0)
        assert np.all(np.isfinite(s))


def test_autocorrelation_non_negative_and_decaying() -> None:
    t = np.linspace(0.0, 40.0, 300)
    s = autocorrelation_nD(t, D_nD=0.8, D_perp=0.2, n=2)
    assert np.all(s >= 0.0)
    # Numerical noise can produce tiny positive upticks; allow tolerance.
    assert np.all(np.diff(s) <= 1e-12)


def test_autocorrelation_2d_matches_reference_form_when_dperp_zero() -> None:
    t = np.linspace(0.0, 8.0, 200)
    d = 1.1
    expected = np.power(np.exp(-2.0 * d * t) * np.i0(2.0 * d * t), 2)
    got = autocorrelation_nD(t, D_nD=d, D_perp=0.0, n=2)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_autocorrelation_dimension_dependence() -> None:
    t = np.linspace(0.0, 20.0, 150)
    s1 = autocorrelation_nD(t, D_nD=1.0, D_perp=0.0, n=1)
    s2 = autocorrelation_nD(t, D_nD=1.0, D_perp=0.0, n=2)
    s3 = autocorrelation_nD(t, D_nD=1.0, D_perp=0.0, n=3)
    # At long times higher powers decay faster.
    assert s1[-1] > s2[-1] > s3[-1]


def test_autocorrelation_anisotropic_decay_is_faster() -> None:
    t = np.linspace(0.0, 20.0, 150)
    iso = autocorrelation_nD(t, D_nD=1.0, D_perp=0.0, n=2)
    aniso = autocorrelation_nD(t, D_nD=1.0, D_perp=0.8, n=2)
    assert aniso[-1] < iso[-1]


def test_autocorrelation_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n must be one of"):
        autocorrelation_nD(np.array([0.0, 1.0]), D_nD=1.0, n=4)
    with pytest.raises(ValueError, match="D_nD must be >= 0"):
        autocorrelation_nD(np.array([0.0, 1.0]), D_nD=-0.1, n=2)
    with pytest.raises(ValueError, match="D_perp must be >= 0"):
        autocorrelation_nD(np.array([0.0, 1.0]), D_nD=1.0, D_perp=-0.1, n=2)


def test_spectral_density_positive_and_decreasing_with_field_frequency() -> None:
    j1 = spectral_density(omega=20.0, D_nD=2.0, D_perp=0.0, n=2)
    j2 = spectral_density(omega=60.0, D_nD=2.0, D_perp=0.0, n=2)
    j3 = spectral_density(omega=200.0, D_nD=2.0, D_perp=0.0, n=2)
    assert j1 > 0.0
    assert j1 > j2 > j3


def test_spectral_density_is_reproducible() -> None:
    j1 = spectral_density(omega=125.0, D_nD=1.7, D_perp=0.2, n=2)
    j2 = spectral_density(omega=125.0, D_nD=1.7, D_perp=0.2, n=2)
    assert np.isclose(j1, j2, rtol=1e-8, atol=1e-10)


def test_spectral_density_reference_values() -> None:
    # Regression anchors from this implementation's deterministic integration settings.
    j_a = spectral_density(omega=50.0, D_nD=2.0, D_perp=0.0, n=2)
    j_b = spectral_density(omega=120.0, D_nD=2.0, D_perp=0.0, n=2)
    assert np.isclose(j_a, 0.0061272971, rtol=2e-3)
    assert np.isclose(j_b, 0.0010993077, rtol=2e-3)


def test_spectral_density_zero_frequency_returns_infinity() -> None:
    j0 = spectral_density(omega=0.0, D_nD=1.0, D_perp=0.0, n=2)
    assert np.isinf(j0)


def test_lambda_diff_vectorized_and_c_scaling() -> None:
    b = np.array([10.0, 40.0, 100.0, 300.0])
    lam1 = lambda_diff(b, C=0.5, D_nD=2.0, D_perp=0.0, n=2)
    lam2 = lambda_diff(b, C=1.0, D_nD=2.0, D_perp=0.0, n=2)
    assert lam1.shape == b.shape
    np.testing.assert_allclose(lam2, 4.0 * lam1, rtol=1e-8, atol=1e-12)


def test_lambda_diff_decreases_with_field() -> None:
    b = np.array([10.0, 50.0, 100.0, 300.0, 1000.0])
    lam = lambda_diff(b, C=1.0, D_nD=1.5, D_perp=0.0, n=2)
    assert np.all(np.diff(lam) < 0.0)


def test_lambda_total_fast_diffusion_has_no_spurious_high_field_cliff() -> None:
    # Regression guard: high D can trigger numerical artifacts if integration
    # window is forced much larger than correlation times.
    b = np.array([900.0, 930.0, 950.0, 970.0, 980.0, 990.0])
    lam = lambda_total(
        b,
        C=91.688705,
        D_nD=3897.362,
        D_perp=0.0,
        n=2,
        lambda_0D=0.091413373,
    )

    # The curve should decrease smoothly, not collapse abruptly between nearby fields.
    ratios = lam[1:] / lam[:-1]
    assert np.all(ratios > 0.8)


def test_lambda_total_adds_offset() -> None:
    b = np.array([20.0, 80.0, 200.0])
    lam_d = lambda_diff(b, C=1.2, D_nD=1.8, D_perp=0.3, n=2)
    lam_t = lambda_total(b, C=1.2, D_nD=1.8, D_perp=0.3, n=2, lambda_0D=0.07)
    np.testing.assert_allclose(lam_t, lam_d + 0.07, rtol=1e-10, atol=1e-12)


def test_lambda_diff_negative_c_has_same_result() -> None:
    b = np.array([20.0, 60.0, 200.0])
    pos = lambda_diff(b, C=1.1, D_nD=2.5, D_perp=0.0, n=3)
    neg = lambda_diff(b, C=-1.1, D_nD=2.5, D_perp=0.0, n=3)
    np.testing.assert_allclose(pos, neg, rtol=1e-12, atol=1e-14)


def test_lambda_invalid_inputs() -> None:
    b = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match="n must be one of"):
        lambda_diff(b, C=1.0, D_nD=1.0, n=0)
    with pytest.raises(ValueError, match="D_nD must be >= 0"):
        lambda_diff(b, C=1.0, D_nD=-1.0, n=2)


def test_diffusion_components_registered_for_field_scope() -> None:
    names = component_names_for_x("field")
    assert "DiffusionLF_1D" in names
    assert "DiffusionLF_2D" in names
    assert "DiffusionLF_3D" in names

    names_temp = component_names_for_x("temperature")
    names_run = component_names_for_x("run")
    assert "DiffusionLF_2D" not in names_temp
    assert "DiffusionLF_2D" not in names_run


def test_diffusion_component_metadata_and_callable() -> None:
    comp = PARAMETER_MODEL_COMPONENTS["DiffusionLF_2D"]
    assert comp.param_names == ["A", "D_2D", "D_perp"]
    assert comp.param_defaults["D_perp"] == 0.0

    b = np.array([20.0, 50.0, 100.0])
    y = comp.function(b, A=1.0, D_2D=1.0, D_perp=0.0)
    assert y.shape == b.shape
    assert np.all(np.isfinite(y))


def test_smoke_typical_gui_grid_runs_quickly() -> None:
    b = np.linspace(10.0, 5000.0, 100)
    t0 = time.perf_counter()
    y = lambda_total(b, C=0.8, D_nD=2.0, D_perp=0.2, n=2, lambda_0D=0.05)
    elapsed = time.perf_counter() - t0
    assert y.shape == b.shape
    assert np.all(np.isfinite(y))
    # Keep a relaxed bound for CI variance while still guarding regressions.
    assert elapsed < 30.0
