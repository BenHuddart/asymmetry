"""Tests for diffusive LF relaxation model."""

from __future__ import annotations

import time

import numpy as np
import pytest

from asymmetry.core.fitting import diffusion as diffusion_mod
from asymmetry.core.fitting.diffusion import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G as GAMMA_E,
)
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


# ---------------------------------------------------------------------------
# Fast Filon transform: parity with the scalar-quad reference and fall-back.
#
# The vectorized fast path is the default. The scalar cosine-weighted quad is
# kept as `_spectral_density_quad` and is the parity reference here. The gate
# uses a MIXED relative/absolute tolerance: away from the noise floor J agrees
# to ~1e-4 relative, but at large omega J underflows towards a quad noise floor
# (the reference itself scatters to ~1e-7..1e-9, sometimes negative), so there
# we only require agreement to a small fraction of the curve maximum. This is
# the documented, principled exception: the fast path is self-consistent to
# ~1e-4 of the curve max across the whole grid (verified separately below), and
# in several large-omega / multi-scale corners it is in fact MORE accurate than
# the scalar quad, which is a known QUADPACK weakness for oscillatory tails.
# ---------------------------------------------------------------------------

# Regimes where the scalar quad is a trustworthy reference: rate at/above the
# fast-path floor, and not the anisotropic huge-D corner where quad underflows.
_PARITY_REGIMES = [
    (2.0, 0.0, 1),
    (2.0, 0.0, 2),
    (2.0, 0.0, 3),
    (2.0, 0.5, 2),
    (0.5, 0.0, 2),
    (10.0, 0.0, 2),
    (100.0, 0.0, 2),
    (1000.0, 0.0, 2),
    (3897.362, 0.0, 2),
]


def _spectral_density_fast_vec(omega: np.ndarray, D_nD: float, D_perp: float, n: int) -> np.ndarray:
    return np.array([spectral_density(float(w), D_nD=D_nD, D_perp=D_perp, n=n) for w in omega])


def _spectral_density_quad_vec(omega: np.ndarray, D_nD: float, D_perp: float, n: int) -> np.ndarray:
    return np.array(
        [
            diffusion_mod._spectral_density_quad(float(w), D_nD=D_nD, D_perp=D_perp, n=n)
            for w in omega
        ]
    )


@pytest.mark.parametrize(("D_nD", "D_perp", "n"), _PARITY_REGIMES)
def test_fast_path_matches_quad_reference_mixed_tol(D_nD: float, D_perp: float, n: int) -> None:
    # omega spans ~1e-3 .. 1e3 of the characteristic diffusion rate.
    char = max(D_nD, 1e-3)
    omega = char * np.geomspace(1e-3, 1e3, 60)

    j_fast = _spectral_density_fast_vec(omega, D_nD, D_perp, n)
    j_quad = _spectral_density_quad_vec(omega, D_nD, D_perp, n)

    scale = float(np.max(np.abs(j_quad)))
    # Mixed tolerance: 1e-4 relative + an absolute floor at 3e-3 of the curve
    # max to absorb the large-omega quad noise floor.
    tol = 1e-4 * np.abs(j_quad) + 3e-3 * scale
    resid = np.abs(j_fast - j_quad)
    worst = int(np.argmax(resid - tol))
    assert np.all(resid <= tol), (
        f"parity failed at omega={omega[worst]:.3g}: "
        f"fast={j_fast[worst]:.4e} quad={j_quad[worst]:.4e} "
        f"resid={resid[worst]:.3e} tol={tol[worst]:.3e}"
    )


@pytest.mark.parametrize(("D_nD", "D_perp", "n"), _PARITY_REGIMES)
def test_fast_path_is_self_consistent_under_grid_refinement(
    D_nD: float, D_perp: float, n: int
) -> None:
    # Independent accuracy check that does NOT lean on quad: refine the Filon
    # grid ~10x and require the default grid to agree to <=1e-3 of the curve max
    # everywhere (the true fast-path error, free of quad noise).
    char = max(D_nD, 1e-3)
    omega = char * np.geomspace(1e-3, 1e3, 80)

    j_default = _spectral_density_fast_vec(omega, D_nD, D_perp, n)

    original = diffusion_mod._FILON_NPTS
    diffusion_mod._filon_grid.cache_clear()
    try:
        diffusion_mod._FILON_NPTS = 16000
        j_fine = _spectral_density_fast_vec(omega, D_nD, D_perp, n)
    finally:
        diffusion_mod._FILON_NPTS = original
        diffusion_mod._filon_grid.cache_clear()

    scale = float(np.max(np.abs(j_fine)))
    assert scale > 0.0
    assert np.max(np.abs(j_default - j_fine)) <= 1e-3 * scale


def test_lambda_total_curves_match_quad_for_paper_like_params() -> None:
    # Curves for paper-like parameter sets must agree to a tolerance that cannot
    # move a fit minimum: <= a few 1e-4 of the curve maximum. (The residual is
    # dominated by scalar-quad noise, not the fast path.)
    b = np.linspace(10.0, 5000.0, 120)
    param_sets = [
        dict(C=6.0, D_nD=3.0, lambda_0D=0.03, D_perp=0.0, n=2),
        dict(C=0.8, D_nD=2.0, lambda_0D=0.05, D_perp=0.2, n=2),
        dict(C=3.0, D_nD=20.0, lambda_0D=0.02, D_perp=0.0, n=3),
        dict(C=2.0, D_nD=1.5, lambda_0D=0.01, D_perp=0.3, n=1),
    ]
    for params in param_sets:
        lam_fast = lambda_total(b, **params)

        w = np.abs(GAMMA_E * b)
        j_quad = _spectral_density_quad_vec(w, params["D_nD"], params["D_perp"], params["n"])
        lam_quad = (params["C"] ** 2) / 4.0 * j_quad + params["lambda_0D"]

        curve_max = float(np.max(np.abs(lam_quad)))
        assert np.max(np.abs(lam_fast - lam_quad)) <= 5e-4 * curve_max, params


def test_slow_diffusion_falls_back_to_quad(monkeypatch: pytest.MonkeyPatch) -> None:
    # Below the min-rate floor the fast path must not be used; the scalar quad
    # is invoked instead so the result stays correct in that corner.
    calls = {"n": 0}
    real_quad = diffusion_mod._spectral_density_quad

    def counting_quad(omega: float, **kwargs: object) -> float:
        calls["n"] += 1
        return real_quad(omega, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(diffusion_mod, "_spectral_density_quad", counting_quad)

    slow = 0.5 * diffusion_mod._DIFFUSION_MIN_RATE_FLOOR
    b = np.array([50.0, 200.0, 1000.0])
    lam = lambda_diff(b, C=1.0, D_nD=slow, D_perp=0.0, n=2)
    assert np.all(np.isfinite(lam))
    assert calls["n"] == b.size  # every point went through quad


def test_fast_path_used_above_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real_quad = diffusion_mod._spectral_density_quad

    def counting_quad(omega: float, **kwargs: object) -> float:
        calls["n"] += 1
        return real_quad(omega, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(diffusion_mod, "_spectral_density_quad", counting_quad)

    b = np.array([50.0, 200.0, 1000.0])
    lambda_diff(b, C=1.0, D_nD=2.0, D_perp=0.0, n=2)
    assert calls["n"] == 0  # fast path, no quad calls


def test_env_var_forces_quad_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real_quad = diffusion_mod._spectral_density_quad

    def counting_quad(omega: float, **kwargs: object) -> float:
        calls["n"] += 1
        return real_quad(omega, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(diffusion_mod, "_spectral_density_quad", counting_quad)
    monkeypatch.setenv("ASYMMETRY_DIFFUSION_QUAD", "1")

    b = np.array([50.0, 200.0, 1000.0])
    lambda_diff(b, C=1.0, D_nD=2.0, D_perp=0.0, n=2)
    assert calls["n"] == b.size


def test_filon_grid_memo_shared_across_omega_sweep() -> None:
    # One (D_nD, D_perp, n) at fixed t_max builds the S(t) grid once and reuses
    # it across the whole omega sweep and across repeated calls.
    diffusion_mod._filon_grid.cache_clear()
    b = np.linspace(10.0, 5000.0, 96)
    lambda_diff(b, C=1.0, D_nD=2.0, D_perp=0.0, n=2)
    info1 = diffusion_mod._filon_grid.cache_info()
    # Exactly one grid build for the whole sweep.
    assert info1.misses == 1
    assert info1.currsize == 1

    # A second identical sweep is a pure cache hit (no new build).
    lambda_diff(b, C=1.0, D_nD=2.0, D_perp=0.0, n=2)
    info2 = diffusion_mod._filon_grid.cache_info()
    assert info2.misses == 1
    assert info2.hits > info1.hits


def test_filon_grid_memo_is_bounded() -> None:
    assert diffusion_mod._filon_grid.cache_info().maxsize == 64


def test_fast_path_800_point_sweep_is_fast() -> None:
    # Regression guard on the whole point of WP-E: an 800-point curve must not
    # take seconds. (Generous bound for CI variance; locally ~0.1 s.)
    b = np.linspace(10.0, 5000.0, 800)
    t0 = time.perf_counter()
    y = lambda_total(b, C=0.8, D_nD=2.0, D_perp=0.2, n=2, lambda_0D=0.05)
    elapsed = time.perf_counter() - t0
    assert y.shape == b.shape
    assert np.all(np.isfinite(y))
    assert elapsed < 5.0
