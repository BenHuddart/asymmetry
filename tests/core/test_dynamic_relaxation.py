"""Tests for the dynamic / fluctuating-field relaxation functions.

Covers dynamic Gaussian & Lorentzian Kubo-Toyabe (strong collision), the Keren
LF function and the Abragam function: analytic limits, properties, registration
metadata, and a fitting round-trip.  See docs/porting/dynamic-relaxation/.
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting import COMPONENTS, MODELS, CompositeModel
from asymmetry.core.fitting.component_docs import get_component_applicability
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import (
    _strong_collision_solve,
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
    # implementation switches to the analytic motional-narrowing limit there).
    amp = 25.0
    for fn in (dynamic_gaussian_kt, dynamic_lorentzian_kt):
        for nu in (20.0, 50.0, 100.0, 500.0, 1000.0):
            for b_l in (0.0, 20.0):
                g = fn(T, amp, 0.4, nu, b_l)
                assert np.all(np.isfinite(g)), (fn.__name__, nu, b_l)
                assert g.max() <= amp * 1.01 and g.min() >= -0.5 * amp, (fn.__name__, nu, b_l)


def test_dynamic_kt_continuous_across_fast_switch() -> None:
    # No large discontinuity as nu crosses the solver -> analytic crossover.
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


def test_lorentzian_lf_accuracy_against_high_resolution() -> None:
    # The analytic angular-average line shape (default n_w, interpolated) agrees
    # with a high-resolution reference to better than ~0.5% over 0-16 us in the
    # decoupling regime.
    from asymmetry.core.fitting.models import _lorentzian_lf_lineshape
    from asymmetry.core.utils.constants import (
        GAUSS_TO_TESLA,
        MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    )

    a = 0.5
    for b in (20.0, 50.0):
        omega0 = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * (b * GAUSS_TO_TESLA)
        g = static_lorentzian_kt_lf(T, 1.0, a, b)
        ref = _lorentzian_lf_lineshape(a, omega0, T, n_w=12000)
        assert np.max(np.abs(g - ref)) < 5e-3


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
