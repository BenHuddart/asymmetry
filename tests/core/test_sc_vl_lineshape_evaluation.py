"""Equivalence of the fast ``R(t)`` evaluator with the straight-line definition.

:func:`~asymmetry.core.fitting.sc.lineshape.vortex_lattice_relaxation` no longer
forms the ``n_bins × N_t`` complex-exponential matrix: it factors the uniform
histogram bin centres out into a carrier and evaluates the remaining polynomial
either by a chirp z-transform (uniform time axis) or a Horner recurrence (any
axis). Both tiers must reproduce
:func:`~asymmetry.core.fitting.sc.lineshape._reference_relaxation` — the retained
pre-optimisation body — to machine precision, over every axis shape and
degenerate input the public entry point accepts. The physics tests live in
``test_sc_vl_lineshape.py``; nothing here re-tests the field distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.sc.lineshape import (
    _CZT_MIN_BINS,
    _characteristic_function,
    _czt_inner_sum,
    _horner_inner_sum,
    _reference_relaxation,
    vortex_lattice_relaxation,
)

#: (lambda_nm, B0_gauss, Bc2_tesla) — a spread of line widths.
_CASES = [
    (195.0, 400.0, 25.0),
    (240.0, 400.0, 25.0),
    (150.0, 1600.0, 30.0),
    (320.0, 200.0, 12.0),
]

_RNG = np.random.default_rng(11)

#: Time axes: uniform (czt path) with and without a t0 offset, non-uniform
#: (Horner path) both random and log-spaced, plus the short/degenerate shapes.
_AXES = {
    "uniform_0_8_500": np.linspace(0.0, 8.0, 500),
    "uniform_long_8192": np.linspace(0.0, 10.0, 8192),
    "uniform_t0_offset": 0.14 + np.arange(4096) * (10.0 / 4095),
    "uniform_t0_negative": -0.5 + np.arange(2000) * 0.004,
    "uniform_short_64": np.linspace(0.0, 8.0, 64),
    "nonuniform_random": np.sort(_RNG.uniform(0.0, 10.0, 3000)),
    "nonuniform_logspace": np.logspace(-3.0, 1.0, 2048),
    "single_point": np.array([3.3]),
    "constant_axis": np.full(300, 2.0),
    "empty": np.array([]),
}


def _errors(new: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(max relative error on |R|, max absolute error on arg R)."""
    rel_mag = np.abs(np.abs(new) - np.abs(ref)) / np.maximum(np.abs(ref), 1e-300)
    arg = np.abs(np.angle(new * np.conj(ref)))
    return float(rel_mag.max()), float(arg.max())


@pytest.mark.parametrize("powder", [True, False])
@pytest.mark.parametrize(("lam", "B0", "Bc2"), _CASES)
@pytest.mark.parametrize("axis", sorted(_AXES))
def test_matches_reference_over_case_matrix(
    axis: str, lam: float, B0: float, Bc2: float, powder: bool
) -> None:
    """Powder and single crystal, every axis shape: agreement at 1e-10."""
    t = _AXES[axis]
    ref = _reference_relaxation(t, lam, B0, Bc2, powder=powder)
    new = vortex_lattice_relaxation(t, lam, B0, Bc2, powder=powder)
    assert new.shape == ref.shape
    assert new.dtype == ref.dtype == np.complex128
    if ref.size == 0:
        return
    rel_mag, arg_err = _errors(new, ref)
    assert rel_mag < 1.0e-10
    assert arg_err < 1.0e-10


def test_both_tiers_agree_on_a_uniform_axis() -> None:
    """The czt path is exercised for real, and the Horner path on the same axis
    (which the uniform axis would otherwise route past) gives the same answer."""
    from asymmetry.core.fitting.sc.lineshape import _calibrated_field_histogram

    histogram = _calibrated_field_histogram(195.0, 400.0, 25.0, True, 10, 96, 512)
    assert histogram is not None
    centres, weights = histogram
    t = np.linspace(0.0, 10.0, 4096)
    delta = (float(centres[-1]) - float(centres[0])) / (centres.size - 1)

    czt_inner = _czt_inner_sum(weights, t, delta)
    assert czt_inner is not None, "a 4096-point uniform axis must take the czt path"
    horner_inner = _horner_inner_sum(weights, t, delta)
    assert np.abs(czt_inner - horner_inner).max() < 1.0e-11


def test_non_uniform_axis_falls_back_to_horner() -> None:
    """A log-spaced axis has no constant step, so the czt path must decline it."""
    from asymmetry.core.fitting.sc.lineshape import _calibrated_field_histogram

    histogram = _calibrated_field_histogram(195.0, 400.0, 25.0, True, 10, 96, 512)
    assert histogram is not None
    centres, weights = histogram
    delta = (float(centres[-1]) - float(centres[0])) / (centres.size - 1)
    assert _czt_inner_sum(weights, np.logspace(-3.0, 1.0, 1024), delta) is None
    # A single point cannot define a step, and neither can a constant axis.
    assert _czt_inner_sum(weights, np.array([1.0]), delta) is None
    assert _czt_inner_sum(weights, np.full(64, 2.0), delta) is None
    # Below the measured bin-count crossover the recurrence is cheaper.
    coarse = np.full(_CZT_MIN_BINS - 1, 1.0 / (_CZT_MIN_BINS - 1))
    assert _czt_inner_sum(coarse, np.linspace(0.0, 8.0, 512), delta) is None


def test_a_barely_perturbed_axis_still_takes_the_fast_path() -> None:
    """Float round-off in a generated axis must not knock it off the czt path,
    while a deliberate gap (a dropped bin) must."""
    from asymmetry.core.fitting.sc.lineshape import _calibrated_field_histogram

    histogram = _calibrated_field_histogram(195.0, 400.0, 25.0, True, 10, 96, 512)
    assert histogram is not None
    centres, weights = histogram
    delta = (float(centres[-1]) - float(centres[0])) / (centres.size - 1)
    t = np.linspace(0.0, 8.0, 1024)
    assert _czt_inner_sum(weights, t, delta) is not None
    gapped = t.copy()
    gapped[500:] += 0.5 * (t[1] - t[0])
    assert _czt_inner_sum(weights, gapped, delta) is None


def test_scalar_and_sequence_inputs_are_unchanged() -> None:
    """Scalar, list and 0-d inputs keep the reference's shapes and values."""
    for t in (2.0, [1.0, 2.0, 3.0], np.float64(0.0)):
        new = vortex_lattice_relaxation(t, 195.0, 400.0, 25.0)
        ref = _reference_relaxation(t, 195.0, 400.0, 25.0)
        assert new.shape == ref.shape
        assert np.abs(new - ref).max() < 1.0e-13


def test_degenerate_no_lattice_still_returns_unity() -> None:
    """R(t) == 1 wherever there is no vortex lattice — unchanged by the rewrite."""
    t = np.linspace(0.0, 8.0, 512)
    for lam, B0, Bc2 in [(195.0, 400_000.0, 25.0), (-5.0, 400.0, 25.0), (195.0, 400.0, 0.0)]:
        r = vortex_lattice_relaxation(t, lam, B0, Bc2)
        assert r.dtype == np.complex128
        assert np.allclose(r, 1.0)


def test_relaxation_starts_at_exactly_unity() -> None:
    """R(0) = Σ w_k = 1 to round-off on both tiers (weights are normalised)."""
    for n_t in (500, 4096):
        r = vortex_lattice_relaxation(np.linspace(0.0, 10.0, n_t), 195.0, 400.0, 25.0)
        assert abs(r[0] - 1.0) < 1.0e-14


def test_single_bin_histogram_is_a_pure_phase() -> None:
    """The n_bins == 1 corner (no spacing to factor out) still matches the sum."""
    from asymmetry.core.fitting.sc.lineshape import _TWO_PI_GAMMA

    t = np.linspace(0.0, 8.0, 256)
    centres = np.array([1.7e-4])
    weights = np.array([1.0])
    new = _characteristic_function(centres, weights, t)
    ref = weights @ np.exp(1j * _TWO_PI_GAMMA * centres[:, None] * t[None, :])
    assert new.dtype == np.complex128
    assert np.abs(new - ref).max() < 1.0e-13


def test_fitted_lambda_is_identical_through_either_evaluator() -> None:
    """The whole point: a real fit returns the same λ_ab through the fast
    evaluator and through the pre-optimisation definition.

    ``test_sc_vl_lineshape.test_fitted_lambda_unchanged_vs_full_average`` already
    pins the fit against the full real-space average at 0.5 nm; this pins it
    against the *exact same* histogram sum, where any difference can only come
    from the evaluator, so the tolerance is five orders of magnitude tighter.
    """
    from scipy.optimize import curve_fit

    from asymmetry.core.fitting.sc.lineshape import vortex_lattice_powder_component
    from asymmetry.core.utils.constants import (
        GAUSS_TO_TESLA,
        MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    )

    t = np.linspace(0.0, 8.0, 400)
    true_lam, B0, Bc2 = 195.0, 400.0, 25.0
    freq = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * B0
    rng = np.random.default_rng(7)
    nuclear = np.exp(-0.5 * (0.20 * t) ** 2)
    signal = (
        vortex_lattice_powder_component(t, 18.0, B0, 0.3, true_lam, Bc2) * nuclear
        + 3.0 * np.cos(2 * np.pi * freq * t + 0.3)
        + 0.4
    )
    y = signal + rng.normal(0.0, 0.15, t.size)
    err = np.full_like(t, 0.15)

    def make_model(relax):
        def model(tt, amp, lam, phi, sig_n, a_bg, c):
            nu = np.exp(-0.5 * (sig_n * tt) ** 2)
            carrier = np.exp(1j * (2 * np.pi * freq * tt + phi))
            vl = amp * np.real(carrier * relax(tt, lam, B0, Bc2, powder=True)) * nu
            return vl + a_bg * np.cos(2 * np.pi * freq * tt + phi) + c

        return model

    p0 = [15.0, 230.0, 0.0, 0.25, 2.0, 0.0]
    bounds = ([0, 120, -np.pi, 0, 0, -2], [50, 360, np.pi, 1, 20, 2])
    kwargs = {"p0": p0, "sigma": err, "bounds": bounds, "maxfev": 40000}
    fast, _ = curve_fit(make_model(vortex_lattice_relaxation), t, y, **kwargs)
    ref, _ = curve_fit(make_model(_reference_relaxation), t, y, **kwargs)
    # Measured difference is ~3e-8 nm — pure minimiser path sensitivity.
    assert abs(fast[1] - ref[1]) < 1.0e-4
    assert np.abs(fast - ref).max() < 1.0e-4


def test_no_large_intermediate_is_materialised() -> None:
    """The point of the rewrite: peak allocation is O(N_t), not O(n_bins·N_t).

    A 65536-point axis against the default 512-bin histogram would need a 512 MB
    complex matrix under the old formulation; here it must stay comfortably in
    tens of megabytes, so a hard cap catches any reintroduced outer product.
    """
    import tracemalloc

    t = np.linspace(0.0, 10.0, 65536)
    vortex_lattice_relaxation(t, 195.0, 400.0, 25.0)  # warm the histogram cache
    tracemalloc.start()
    try:
        vortex_lattice_relaxation(t, 195.0, 400.0, 25.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # 512 x 65536 complex128 == 512 MB; a handful of N_t-sized complex buffers
    # is ~1 MB each. 64 MB is generous headroom for scipy's FFT workspace.
    assert peak < 64 * 1024**2
