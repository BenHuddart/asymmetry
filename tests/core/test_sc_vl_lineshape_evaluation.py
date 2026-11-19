"""Fast transforms of the same compact moments; physical accuracy is tested separately."""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from asymmetry.core.fitting.sc.lineshape import (
    _CZT_MIN_BINS,
    _TWO_PI_GAMMA,
    _calibrated_field_moments,
    _characteristic_function,
    _czt_inner_sum,
    _horner_inner_sum,
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
    "uniform_reverse_8192": np.linspace(10.0, 0.0, 8192),
    "uniform_t0_offset": 0.14 + np.arange(4096) * (10.0 / 4095),
    "uniform_t0_negative": -0.5 + np.arange(2000) * 0.004,
    "uniform_short_64": np.linspace(0.0, 8.0, 64),
    "nonuniform_random": np.sort(_RNG.uniform(0.0, 10.0, 3000)),
    "nonuniform_logspace": np.logspace(-3.0, 1.0, 2048),
    "single_point": np.array([3.3]),
    "constant_axis": np.full(300, 2.0),
    "empty": np.array([]),
}


def _direct_quadrature(t, lam, field, bc2, *, powder=True):
    """Direct bin exponentials test transform arithmetic, not field-map accuracy."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    data = _calibrated_field_moments(lam, field, bc2, powder, 80, 384, 512)
    if data is None:
        return np.ones(t.shape, dtype=complex)
    centre, delta, moments = data
    fields = centre + delta * np.arange(moments.shape[-1])
    exponentials = np.exp(1j * _TWO_PI_GAMMA * fields[:, None] * t.ravel())
    transformed = moments @ exponentials
    phase = 1j * _TWO_PI_GAMMA * delta * t.ravel()
    result = sum(row * phase**order / factorial(order) for order, row in enumerate(transformed))
    return result.reshape(t.shape)


@pytest.mark.parametrize("powder", [True, False])
@pytest.mark.parametrize(("lam", "B0", "Bc2"), _CASES)
@pytest.mark.parametrize("axis", sorted(_AXES))
def test_matches_reference_over_case_matrix(
    axis: str, lam: float, B0: float, Bc2: float, powder: bool
) -> None:
    """Powder and single crystal, every axis shape: agreement at 1e-10."""
    t = _AXES[axis]
    ref = _direct_quadrature(t, lam, B0, Bc2, powder=powder)
    new = vortex_lattice_relaxation(t, lam, B0, Bc2, powder=powder)
    assert new.shape == ref.shape
    assert new.dtype == ref.dtype == np.complex128
    if ref.size == 0:
        return
    assert np.max(np.abs(new - ref)) < 1e-10


def test_both_tiers_agree_on_a_uniform_axis() -> None:
    """The czt path is exercised for real, and the Horner path on the same axis
    (which the uniform axis would otherwise route past) gives the same answer."""
    quadrature = _calibrated_field_moments(195.0, 400.0, 25.0, True, 80, 384, 512)
    assert quadrature is not None
    _, delta, weights = quadrature
    t = np.linspace(0.0, 10.0, 4096)

    czt_inner = _czt_inner_sum(weights, t, delta)
    assert czt_inner is not None, "a 4096-point uniform axis must take the czt path"
    horner_inner = _horner_inner_sum(weights, t, delta)
    assert np.abs(czt_inner - horner_inner).max() < 1.0e-11


def test_non_uniform_axis_falls_back_to_horner() -> None:
    """A log-spaced axis has no constant step, so the czt path must decline it."""
    quadrature = _calibrated_field_moments(195.0, 400.0, 25.0, True, 80, 384, 512)
    assert quadrature is not None
    _, delta, weights = quadrature
    assert _czt_inner_sum(weights, np.logspace(-3.0, 1.0, 1024), delta) is None
    # A single point cannot define a step, and neither can a constant axis.
    assert _czt_inner_sum(weights, np.array([1.0]), delta) is None
    assert _czt_inner_sum(weights, np.full(64, 2.0), delta) is None
    # Below the measured bin-count crossover the recurrence is cheaper.
    coarse = np.full(_CZT_MIN_BINS - 1, 1.0 / (_CZT_MIN_BINS - 1))
    assert _czt_inner_sum(coarse, np.linspace(0.0, 8.0, 512), delta) is None
    # A column vector has a uniform step but a flat czt result would broadcast
    # against the 2-D carrier into a wrong (N, N); it must take Horner.
    assert _czt_inner_sum(weights, np.linspace(0.0, 8.0, 512).reshape(512, 1), delta) is None


def test_a_column_time_axis_keeps_its_shape() -> None:
    """A column vector uses elementwise recurrence and keeps its own shape."""
    flat = np.linspace(0.0, 8.0, 200)
    column = flat.reshape(200, 1)
    ref = _direct_quadrature(flat, 195.0, 400.0, 25.0)
    new = vortex_lattice_relaxation(column, 195.0, 400.0, 25.0)
    assert new.shape == (200, 1)
    assert np.abs(new - ref.reshape(200, 1)).max() < 1.0e-13


def test_a_barely_perturbed_axis_still_takes_the_fast_path() -> None:
    """Float round-off in a generated axis must not knock it off the czt path,
    while a deliberate gap (a dropped bin) must."""
    quadrature = _calibrated_field_moments(195.0, 400.0, 25.0, True, 80, 384, 512)
    assert quadrature is not None
    _, delta, weights = quadrature
    t = np.linspace(0.0, 8.0, 1024)
    assert _czt_inner_sum(weights, t, delta) is not None
    gapped = t.copy()
    gapped[500:] += 0.5 * (t[1] - t[0])
    assert _czt_inner_sum(weights, gapped, delta) is None


def test_scalar_and_sequence_inputs_are_unchanged() -> None:
    """Scalar, list and 0-d inputs keep the reference's shapes and values."""
    for t in (2.0, [1.0, 2.0, 3.0], np.float64(0.0)):
        new = vortex_lattice_relaxation(t, 195.0, 400.0, 25.0)
        ref = _direct_quadrature(t, 195.0, 400.0, 25.0)
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


def test_single_bin_point_distribution_is_a_pure_phase() -> None:
    """A point mass has unit zeroth moment and vanishing higher local moments."""
    t = np.linspace(0.0, 8.0, 256)
    centre = 1.7e-4
    moments = np.zeros((5, 1))
    moments[0] = 1.0
    actual = _characteristic_function(centre, 1e-5, moments, t)
    expected = np.exp(1j * _TWO_PI_GAMMA * centre * t)
    assert np.max(np.abs(actual - expected)) < 1e-13


def test_no_large_intermediate_is_materialised() -> None:
    """Warm transforms retain bounded time buffers rather than a bin-time matrix."""
    import tracemalloc

    t = np.linspace(0.0, 10.0, 65536)
    vortex_lattice_relaxation(t, 195.0, 400.0, 25.0)  # warm the histogram cache
    tracemalloc.start()
    try:
        vortex_lattice_relaxation(t, 195.0, 400.0, 25.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 64 * 1024**2


def test_changing_parameters_retain_compact_data_without_raw_map_growth():
    """Twenty parameter changes retain compact moments rather than twenty spatial maps."""
    import tracemalloc

    times = np.linspace(0, 8, 129)
    vortex_lattice_relaxation(times, 195, 400, 25)
    _calibrated_field_moments.cache_clear()
    tracemalloc.start()
    try:
        for lam in np.linspace(180, 220, 20):
            vortex_lattice_relaxation(times, lam, 400, 25)
        retained = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert retained < 1024**2
