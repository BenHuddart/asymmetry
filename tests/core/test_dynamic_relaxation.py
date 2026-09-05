"""Tests for the dynamic / fluctuating-field relaxation functions.

Covers dynamic Gaussian & Lorentzian Kubo-Toyabe (strong collision), the Keren
LF function and the Abragam function: analytic limits, properties, registration
metadata, and a fitting round-trip.  See docs/porting/dynamic-relaxation/.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting import COMPONENTS, MODELS, CompositeModel
from asymmetry.core.fitting.component_docs import get_component_applicability
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import (
    _dynamic_kt_grid,
    _exponential_sum,
    _lorentzian_kt_zf_realisation,
    _lorentzian_lf_lineshape_reference,
    _lorentzian_lf_uniform,
    _strong_collision_modes,
    _strong_collision_solve,
    _strong_collision_solve_reference,
    abragam,
    dynamic_gaussian_kt,
    dynamic_lorentzian_kt,
    keren,
    longitudinal_field_kubo_toyabe,
    static_gkt_zf,
    static_lorentzian_kt_lf,
    static_lorentzian_kt_zf,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

T = np.linspace(0.0, 8.0, 200)
NEW = ["DynamicGaussianKT", "DynamicLorentzianKT", "Keren", "Abragam"]


# --- Dynamic Gaussian KT: analytic limits -------------------------------------
def test_dyn_gaussian_nu0_zf_equals_static() -> None:
    assert np.allclose(
        dynamic_gaussian_kt(T, 1.0, 0.3, 0.0, 0.0), static_gkt_zf(T, 1.0, 0.3), atol=1e-9
    )


def test_dyn_gaussian_nu0_lf_equals_static_lf() -> None:
    assert np.allclose(
        dynamic_gaussian_kt(T, 1.0, 0.3, 0.0, 50.0),
        longitudinal_field_kubo_toyabe(T, 1.0, 0.3, 50.0),
        atol=1e-9,
    )


def test_dyn_gaussian_nu_continuity() -> None:
    # As nu -> 0 the dynamic result is continuous with the static one.
    near = dynamic_gaussian_kt(T, 1.0, 0.3, 1e-6, 0.0)
    assert np.max(np.abs(near - static_gkt_zf(T, 1.0, 0.3))) < 1e-4


def test_dyn_gaussian_fast_fluctuation_grid_independent() -> None:
    # The numerical strong-collision result is grid-independent (matches a finer
    # grid), which is the meaningful correctness check; the asymptotic
    # exp(-2 Delta^2 t / nu) is only an approximation in the intermediate regime.
    delta, nu = 0.3, 5.0
    coarse = dynamic_gaussian_kt(T, 1.0, delta, nu, 0.0)
    n = 8001
    grid = np.linspace(0.0, 8.0, n)
    gs = static_gkt_zf(grid, 1.0, delta)
    fine = np.interp(T, grid, _strong_collision_solve(gs, nu, grid[1] - grid[0]))
    assert np.max(np.abs(coarse - fine)) < 5e-3


def test_dyn_gaussian_motional_narrowing_trend() -> None:
    # Increasing nu removes the zero-field 1/3 dip and slows the long-time decay.
    delta = 0.4
    tail_slow = dynamic_gaussian_kt(np.array([8.0]), 1.0, delta, 0.5, 0.0)[0]
    tail_fast = dynamic_gaussian_kt(np.array([8.0]), 1.0, delta, 10.0, 0.0)[0]
    assert tail_fast > tail_slow  # faster fluctuations -> less relaxation at 8 us


def test_dyn_gaussian_lf_decoupling() -> None:
    # Large longitudinal field decouples: polarization stays near 1.
    g = dynamic_gaussian_kt(T, 1.0, 0.3, 1.0, 5000.0)
    assert np.min(g) > 0.9


def test_dynamic_kt_bounded_at_high_nu() -> None:
    # Regression: the explicit strong-collision solver diverges in the fast-
    # fluctuation regime; high nu must stay finite and within [-A0/2, A0] (the
    # grid-solved cases switch to the analytic motional-narrowing limit there;
    # the zero-field Lorentzian is exact at every rate).
    amp = 25.0
    for fn in (dynamic_gaussian_kt, dynamic_lorentzian_kt):
        for nu in (20.0, 50.0, 100.0, 500.0, 1000.0):
            for b_l in (0.0, 20.0):
                g = fn(T, amp, 0.4, nu, b_l)
                assert np.all(np.isfinite(g)), (fn.__name__, nu, b_l)
                assert g.max() <= amp * 1.01 and g.min() >= -0.5 * amp, (fn.__name__, nu, b_l)


def test_dynamic_kt_continuous_across_fast_switch() -> None:
    # No large discontinuity as nu crosses the solver -> analytic crossover
    # (the zero-field Lorentzian has no crossover at all; it is exact).
    for fn in (dynamic_gaussian_kt, dynamic_lorentzian_kt):
        for b_l in (0.0, 20.0):
            jump = np.max(np.abs(fn(T, 25.0, 0.4, 11.99, b_l) - fn(T, 25.0, 0.4, 12.01, b_l)))
            assert jump < 0.5, (fn.__name__, b_l, jump)  # < 2% of A0=25


# --- Keren / Abragam internal consistency -------------------------------------
def test_keren_zf_equals_abragam_form() -> None:
    delta, nu = 0.3, 2.0
    nt = nu * T
    abform = np.exp(-(2.0 * delta**2 / nu**2) * (np.exp(-nt) - 1.0 + nt))
    assert np.allclose(keren(T, 1.0, delta, nu, 0.0), abform, atol=1e-9)


def test_keren_zf_is_abragam_squared() -> None:
    # Keren ZF carries a factor 2 (two transverse components) vs single Abragam.
    delta, nu = 0.3, 2.0
    assert np.allclose(keren(T, 1.0, delta, nu, 0.0), abragam(T, 1.0, delta, nu) ** 2, atol=1e-9)


def test_abragam_gaussian_limit() -> None:
    sigma = 0.4
    assert np.max(np.abs(abragam(T, 1.0, sigma, 1e-9) - np.exp(-0.5 * sigma**2 * T**2))) < 1e-4


def test_abragam_fast_exponential_limit() -> None:
    sigma, nu = 0.4, 10.0
    assert np.max(np.abs(abragam(T, 1.0, sigma, nu) - np.exp(-(sigma**2 / nu) * T))) < 0.02


def test_keren_zero_field_zero_nu_is_finite_gaussian() -> None:
    # The denom -> 0 guard: nu=0, B_L=0 gives exp(-Delta^2 t^2), finite.
    g = keren(T, 1.0, 0.3, 0.0, 0.0)
    assert np.all(np.isfinite(g)) and np.allclose(g, np.exp(-(0.3**2) * T**2), atol=1e-9)


# --- Lorentzian ---------------------------------------------------------------
def test_dyn_lorentzian_nu0_equals_static() -> None:
    assert np.allclose(
        dynamic_lorentzian_kt(T, 1.0, 0.5, 0.0, 0.0),
        static_lorentzian_kt_zf(T, 1.0, 0.5),
        atol=1e-9,
    )


def test_static_lorentzian_zf_value_at_zero() -> None:
    assert abs(static_lorentzian_kt_zf(np.array([0.0]), 1.0, 0.5)[0] - 1.0) < 1e-12


def test_lorentzian_lf_zero_field_shortcut() -> None:
    # A negligible field (omega0 < 0.05 a_L) is treated as exact zero field.
    assert np.allclose(
        static_lorentzian_kt_lf(T, 1.0, 0.5, 0.1), static_lorentzian_kt_zf(T, 1.0, 0.5), atol=1e-12
    )


def _omega0(b_gauss: float) -> float:
    from asymmetry.core.utils.constants import (
        GAUSS_TO_TESLA,
        MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    )

    return 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * (b_gauss * GAUSS_TO_TESLA)


def test_lorentzian_lf_accuracy_against_high_resolution() -> None:
    # The spectral-density line shape (interpolated from its cached grid) agrees
    # with the original angular-average quadrature at high resolution to that
    # quadrature's own accuracy (~1e-3 at n_w=12000) over 0-16 us in the
    # decoupling regime.
    a = 0.5
    for b in (20.0, 50.0):
        g = static_lorentzian_kt_lf(T, 1.0, a, b)
        ref = _lorentzian_lf_lineshape_reference(a, _omega0(b), T, n_w=12000)
        assert np.max(np.abs(g - ref)) < 2e-3


def test_lorentzian_lf_uniform_is_grid_converged() -> None:
    # The FFT evaluation of the closed-form spectral density is spectrally
    # convergent: refining the time step (which refines the frequency sampling
    # and widens the aliasing period) changes nothing at the 1e-9 level, and
    # G(0) = 1 exactly.  Covers weak, intermediate and strong fields.
    for a, b, h in ((0.3, 20.0, 0.02), (5.0, 50.0, 0.02), (0.05, 5.0, 0.02), (1.0, 300.0, 0.001)):
        n = 801
        coarse = _lorentzian_lf_uniform(a, _omega0(b), h, n)
        fine = _lorentzian_lf_uniform(a, _omega0(b), h / 4.0, 4 * (n - 1) + 1)
        assert coarse[0] == pytest.approx(1.0, abs=1e-12)
        assert np.max(np.abs(coarse - fine[::4])) < 1e-9, (a, b)


def test_lorentzian_lf_uniform_matches_reference_quadrature() -> None:
    # Independent check of the closed-form spectral density against the direct
    # angular-average quadrature, to the latter's own accuracy.
    t = np.linspace(0.0, 8.0, 41)
    for a, b in ((0.3, 20.0), (5.0, 50.0), (0.05, 5.0), (1.0, 100.0)):
        omega0 = _omega0(b)
        grid = _lorentzian_lf_uniform(a, omega0, 0.2, 41)
        ref = _lorentzian_lf_lineshape_reference(a, omega0, t, n_w=12000)
        assert np.max(np.abs(grid - ref)) < 2e-3, (a, b)


def test_lorentzian_lf_high_field_long_window_is_resolved() -> None:
    # Regression: the previous fixed 220-node line-shape grid aliased the Larmor
    # oscillation at high field on a long window (9e-2 error at 1000 G over
    # 32 us); the grid step now follows omega0.
    a, b = 5.0, 1000.0
    t = np.linspace(0.0, 32.0, 17)
    g = static_lorentzian_kt_lf(t, 1.0, a, b)
    ref = _lorentzian_lf_lineshape_reference(a, _omega0(b), t, n_w=12000)
    assert np.max(np.abs(g - ref)) < 1e-3


def test_lorentzian_lf_static_grid_respects_fft_cap() -> None:
    # An extreme B_L * tmax would ask for a Larmor-resolved grid whose spectral
    # FFT exceeds the cap; the cached grid coarsens instead of allocating it,
    # and the sign of omega0 is immaterial.
    from asymmetry.core.fitting.models import _LOR_LF_FFT_CAP, _static_lorentzian_lf_grid

    grid, g = _static_lorentzian_lf_grid(0.5, _omega0(20000.0), 200.0)
    assert 2 * grid.shape[0] <= _LOR_LF_FFT_CAP
    assert grid[-1] >= 200.0 - 1e-9 and np.all(np.isfinite(g))
    assert np.allclose(g, _static_lorentzian_lf_grid(0.5, -_omega0(20000.0), 200.0)[1])
    with pytest.raises(ValueError):
        _lorentzian_lf_uniform(0.5, _omega0(20.0), 1e-4, _LOR_LF_FFT_CAP)


def test_lorentzian_lf_continuous_at_zero_field_shortcut() -> None:
    # Just above the omega0 < 0.05 a_L shortcut the field correction is
    # second order, so the numerical line shape sits within ~1e-3 of the
    # analytic zero-field form it hands over to.
    a = 0.5
    b = 1.0001 * 0.05 * a / _omega0(1.0)
    assert (
        np.max(np.abs(static_lorentzian_kt_lf(T, 1.0, a, b) - static_lorentzian_kt_zf(T, 1.0, a)))
        < 2e-3
    )


def test_lorentzian_lf_decoupling_and_origin() -> None:
    g = static_lorentzian_kt_lf(T, 1.0, 0.5, 5000.0)
    assert abs(g[0] - 1.0) < 1e-6  # G(0) = 1
    assert np.min(g) > 0.99  # large longitudinal field decouples


def test_lorentzian_lf_recovers_with_field() -> None:
    # Long-time polarization rises monotonically as the longitudinal field grows.
    tails = [static_lorentzian_kt_lf(T, 1.0, 0.5, b)[-1] for b in (0.0, 20.0, 50.0, 200.0)]
    assert tails == sorted(tails)


def test_dynamic_lorentzian_lf_consistency() -> None:
    # nu -> 0 with a field uses the numerical static Lorentzian LF directly.
    g0 = dynamic_lorentzian_kt(T, 1.0, 0.5, 0.0, 50.0)
    assert np.allclose(g0, static_lorentzian_kt_lf(T, 1.0, 0.5, 50.0), atol=1e-9)
    gd = dynamic_lorentzian_kt(T, 1.0, 0.5, 1.0, 50.0)
    assert np.all(np.isfinite(gd)) and abs(gd[0] - 1.0) < 1e-6


def test_dynamic_lorentzian_lf_is_the_solver_on_the_spectral_line_shape() -> None:
    # The field path is the Volterra solver fed the spectral-density line shape
    # on its own grid (the line shape itself is checked against the reference
    # quadrature above); a finely resolved solve on that line shape must agree
    # to the solver's O((nu h)^2) error.
    grid = np.linspace(0.0, 8.0, 8001)
    h = grid[1] - grid[0]
    for a, b, nu in ((0.5, 50.0, 1.0), (0.3, 20.0, 0.3), (2.0, 300.0, 5.0)):
        gs = _lorentzian_lf_uniform(a, _omega0(b), h, grid.shape[0])
        ref = np.interp(T, grid, _strong_collision_solve(gs, nu, h))
        tol = 1e-5 + 8.0 * (nu * h) ** 2 + 8.0 * (nu * 0.02) ** 2
        assert np.max(np.abs(dynamic_lorentzian_kt(T, 1.0, a, nu, b) - ref)) < tol, (a, b, nu)


def test_dynamic_kt_grid_resolves_larmor_oscillation() -> None:
    # The Volterra grid step must resolve the static line shape's oscillation
    # at omega0 (<= 0.1 rad per step) as well as the collision rate.
    for kind in ("gaussian", "lorentzian"):
        grid, gd = _dynamic_kt_grid(kind, 1.0, 0.5, 1000.0, 16.0)
        assert (grid[1] - grid[0]) * _omega0(1000.0) <= 0.1 + 1e-9, kind
        assert np.all(np.isfinite(gd)) and abs(gd[0] - 1.0) < 1e-9


# --- Zero-field dynamic Lorentzian KT: exact closed form ----------------------
def _volterra_residual(t: np.ndarray, gd: np.ndarray, gs: np.ndarray, nu: float) -> float:
    # max |G_d - f - nu * (f * G_d)| on a uniform grid, trapezoidal convolution.
    h = t[1] - t[0]
    f = np.exp(-nu * t) * gs
    n = len(t)
    size = 1 << (2 * n).bit_length()
    conv = np.fft.irfft(np.fft.rfft(f, size) * np.fft.rfft(gd, size), size)[:n] * h
    conv -= 0.5 * h * (f * gd[0] + f[0] * gd)
    return float(np.max(np.abs(gd - f - nu * conv)[1:]))


@pytest.mark.parametrize("nu", [0.05, 0.5, 3.0, 11.9])
def test_dynamic_lorentzian_zf_closed_form_matches_solver(nu: float) -> None:
    # The eigenmode closed form reproduces a finely resolved Volterra solve
    # across widths; the residual difference is the trapezoidal solver's own
    # O((nu h)^2) error (~1e-4 at nu = 12 on this grid).
    n = 20001
    grid = np.linspace(0.0, 8.0, n)
    h = grid[1] - grid[0]
    tol = 1e-5 + 8.0 * (nu * h) ** 2
    for a in (0.05, 0.5, 5.0):
        ref = _strong_collision_solve(static_lorentzian_kt_zf(grid, 1.0, a, 0.0), nu, h)
        assert np.max(np.abs(dynamic_lorentzian_kt(grid, 1.0, a, nu, 0.0) - ref)) < tol, a


@pytest.mark.parametrize("nu", [2e-9, 1e-6, 1e-3, 1.0, 12.0, 1e2, 1e4, 1e6])
def test_dynamic_lorentzian_zf_satisfies_volterra_equation_at_every_rate(nu: float) -> None:
    # No grid, no switch: the closed form solves the strong-collision equation
    # itself at any rate, including just above the static cutoff (where the
    # Jordan block of the t e^{-at} term is barely split) and far beyond the
    # old solver's stability limit.  The residual floor is the check's own
    # trapezoidal quadrature error on the fast e^{-nu t} kernel.
    a = 0.5
    t = np.linspace(0.0, min(8.0, 60.0 / max(nu, a)), 20001)
    gd = dynamic_lorentzian_kt(t, 1.0, a, nu, 0.0)
    assert gd[0] == pytest.approx(1.0, abs=1e-12)
    assert _volterra_residual(t, gd, static_lorentzian_kt_zf(t, 1.0, a, 0.0), nu) < 1e-6


def test_dynamic_lorentzian_zf_near_static_limit_is_conditioned() -> None:
    # Just above the nu <= 1e-9 static cutoff the closed form must equal the
    # static function to round-off, not blow up on the near-defective modes.
    static = static_lorentzian_kt_zf(T, 1.0, 0.5)
    assert np.max(np.abs(dynamic_lorentzian_kt(T, 1.0, 0.5, 2e-9, 0.0) - static)) < 1e-7


def test_dynamic_lorentzian_zf_rate_saturates_at_four_thirds_a() -> None:
    # Fast fluctuations of a Lorentzian field do not narrow: the slowest mode's
    # rate tends to 4 a_L / 3, independent of nu.
    a = 0.5
    for nu in (50.0, 500.0, 5000.0):
        lam, _ = _strong_collision_modes(*_lorentzian_kt_zf_realisation(a), nu)
        assert max(lam.real) == pytest.approx(-4.0 * a / 3.0, rel=2e-3), nu


def test_strong_collision_modes_generic_realisation() -> None:
    # The rank-one-feedback construction is generic: a damped cosine
    # e^{-t} cos(2t) as a 2x2 system, dynamicised, must satisfy the Volterra
    # equation and reduce to the static function at nu -> 0.
    a_mat = np.array([[-1.0, -2.0], [2.0, -1.0]])
    b = np.array([1.0, 0.0])
    c = np.array([1.0, 0.0])
    t = np.linspace(0.0, 8.0, 20001)
    lam, w = _strong_collision_modes(a_mat, b, c, 0.0)
    assert np.allclose(_exponential_sum(t, lam, w), np.exp(-t) * np.cos(2.0 * t), atol=1e-12)
    for nu in (0.3, 3.0, 300.0):
        t = np.linspace(0.0, min(8.0, 60.0 / nu), 20001)
        gs = np.exp(-t) * np.cos(2.0 * t)
        lam, w = _strong_collision_modes(a_mat, b, c, nu)
        assert _volterra_residual(t, _exponential_sum(t, lam, w), gs, nu) < 1e-6


# --- General properties -------------------------------------------------------
def test_all_finite_and_origin() -> None:
    for fn, args in (
        (dynamic_gaussian_kt, (0.25, 0.3, 2.0, 10.0)),
        (dynamic_lorentzian_kt, (0.25, 0.4, 2.0, 0.0)),
        (keren, (0.25, 0.3, 2.0, 20.0)),
        (abragam, (0.25, 0.3, 2.0)),
    ):
        y = fn(T, *args)
        assert np.all(np.isfinite(y))
        assert abs(fn(np.array([0.0]), *args)[0] - 0.25) < 1e-9  # G(0) = A0


def test_scalar_input_supported() -> None:
    assert np.ndim(dynamic_gaussian_kt(0.0, 1.0, 0.3, 1.0, 5.0)) == 0


# --- Registration / metadata --------------------------------------------------
def test_registered_in_both_registries() -> None:
    for name in NEW:
        assert name in MODELS
        assert name in COMPONENTS


def test_composites_build_and_evaluate() -> None:
    for name in NEW:
        md = CompositeModel.from_expression(f"{name} + Constant").to_model_definition()
        y = md.function(T, **md.param_defaults)
        assert y.shape == T.shape and np.all(np.isfinite(y))


def test_metadata_units_citation_latex_infohelp() -> None:
    expected_units = {"Delta": "µs⁻¹", "a_L": "µs⁻¹", "sigma": "µs⁻¹", "nu": "MHz", "B_L": "G"}
    for name in NEW:
        comp = COMPONENTS[name]
        # paper citation present in the description
        assert any(y in comp.description for y in ("1979", "1985", "1994", "1961"))
        # clean equation + info-helper note
        assert len(comp.latex_equation) > 10
        assert len(get_component_applicability(name)) > 40
        for p, info in comp.param_info.items():
            if p in expected_units:
                assert info.unit == expected_units[p], (name, p, info.unit)
            assert info.description  # every parameter has a description


def test_all_component_equations_render_with_mathtext() -> None:
    # The component-info dialog renders ``latex_equation`` with matplotlib
    # mathtext (a LaTeX subset).  Guard against unsupported commands
    # (e.g. \tfrac, \big, \lvert) that silently fall back to raw source.
    import io

    from matplotlib.mathtext import math_to_image

    failures = []
    for name, comp in COMPONENTS.items():
        eq = (comp.latex_equation or "").strip()
        if not eq:
            continue
        expr = eq if eq.startswith("$") else f"${eq}$"
        try:
            math_to_image(expr, io.BytesIO(), dpi=120, format="png")
        except Exception as exc:  # noqa: BLE001 - report which component/why
            failures.append(f"{name}: {exc}")
    assert not failures, "latex_equation does not render under mathtext:\n" + "\n".join(failures)


# --- Fitting round-trip -------------------------------------------------------
def test_dynamic_gaussian_kt_round_trip() -> None:
    rng = np.random.default_rng(0)
    delta_true, nu_true, amp_true = 0.37, 1.5, 23.0
    from asymmetry.core.data.dataset import MuonDataset

    t = np.linspace(0.05, 12.0, 400)
    y = dynamic_gaussian_kt(t, amp_true, delta_true, nu_true, 0.0) + rng.normal(0, 0.3, t.size)
    ds = MuonDataset(time=t, asymmetry=y, error=np.full_like(t, 0.3), metadata={"run_number": 1})

    md = CompositeModel.from_expression("DynamicGaussianKT").to_model_definition()
    # Single-component composites uniquify the amplitude name A -> A_1.
    amp = md.param_names[0]
    params = ParameterSet(
        [
            Parameter(amp, value=22.0, min=0.0, max=60.0),
            Parameter("Delta", value=0.3, min=0.01, max=3.0),
            Parameter("nu", value=1.0, min=0.0, max=20.0),
            Parameter("B_L", value=0.0, fixed=True),
        ]
    )
    res = FitEngine().fit(ds, md.function, params, t_min=0.1, t_max=12.0)
    assert res.success
    assert abs(res.parameters["Delta"].value - delta_true) < 0.05
    assert abs(res.parameters["nu"].value - nu_true) < 0.6


# --- Strong-collision solver: fast blocked path vs. scalar reference ----------
def _static_gs(kind: str, grid: np.ndarray, width: float, b_l: float) -> np.ndarray:
    if kind == "gaussian":
        gs = (
            static_gkt_zf(grid, 1.0, width, 0.0)
            if abs(b_l) < 1e-9
            else longitudinal_field_kubo_toyabe(grid, 1.0, width, b_l, 0.0)
        )
    else:
        gs = (
            static_lorentzian_kt_zf(grid, 1.0, width, 0.0)
            if abs(b_l) < 1e-9
            else static_lorentzian_kt_lf(grid, 1.0, width, b_l, 0.0)
        )
    return np.asarray(gs, dtype=float)


@pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("nu", [0.05, 0.5, 3.0, 6.0, 11.9])
def test_strong_collision_fast_matches_reference(kind: str, nu: float) -> None:
    # The fast blocked/FFT solver must reproduce the scalar O(n^2) recursion to
    # machine precision across widths, B_L (incl. zero field), grid sizes (small
    # edge cases, a non-power-of-two, a single-block and a multi-block size) and
    # steps with nu*h straddling the 0.02 the caller targets.
    for width in (0.1, 0.5):
        for b_l in (0.0, 25.0):
            for h in (0.02 / nu, 0.018 / nu):
                for n in (2, 3, 64, 777, 1000):
                    grid = np.linspace(0.0, (n - 1) * h, n)
                    gs = _static_gs(kind, grid, width, b_l)
                    fast = _strong_collision_solve(gs, nu, h)
                    ref = _strong_collision_solve_reference(gs, nu, h)
                    assert np.allclose(fast, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("kind", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("nu", [0.5, 11.9])
def test_strong_collision_fast_matches_reference_large_grid(kind: str, nu: float) -> None:
    # 20001 points is the caller's grid cap and exercises the multi-block FFT
    # cross-coupling path (n >> the 512-point block).
    n, h = 20001, 0.02 / nu
    grid = np.linspace(0.0, (n - 1) * h, n)
    gs = _static_gs(kind, grid, 0.3, 0.0)
    fast = _strong_collision_solve(gs, nu, h)
    ref = _strong_collision_solve_reference(gs, nu, h)
    assert np.allclose(fast, ref, rtol=1e-9, atol=1e-12)


def test_strong_collision_single_point_grid() -> None:
    # Degenerate one-sample grid: G_d(0) = 1 with no recursion.
    assert _strong_collision_solve(np.array([1.0]), 3.0, 0.02) == np.array([1.0])
