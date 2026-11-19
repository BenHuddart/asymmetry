"""Independent numerical accuracy of the defined calibrated vortex signal."""

import numpy as np
import pytest

from asymmetry.core.fitting.sc.lineshape import vortex_lattice_relaxation
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MAGNETIC_FLUX_QUANTUM_WB,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)
from tests.core.sc_vl_reference import lattice_average, lattice_offsets


@pytest.mark.parametrize("powder", [False, True])
def test_hardest_nearby_lambda_matches_independent_average(powder):
    """The 195-nm anchor's 0.9-lambda variant through 8 us resolves the core tail."""
    times = np.linspace(0, 8, 129)
    reference = lattice_average(times, 175.5, 400.0, 25.0, powder=powder)
    actual = vortex_lattice_relaxation(times, 175.5, 400.0, 25.0, powder=powder)
    assert np.max(np.abs(actual - reference)) < 1e-6


# YB6 supplies an independently fitted cutoff, so its mapped Bc2 is numerical
# stress rather than reproduction of the paper's separately specified bulk Bc2.
_ANCHORS = [
    ("known195", 195.0, 400.0, 25.0, 8.0),
    ("known240", 240.0, 400.0, 25.0, 8.0),
    ("Re3Ta", 599.0, 300.0, 7.3, 4.0),
    ("YB6", 153.0, 1500.0, MAGNETIC_FLUX_QUANTUM_WB / (2 * np.pi * (25.6e-9) ** 2), 6.0),
]
_VARIATIONS = [
    ("anchor", 1.0, 1.0, 1.0),
    ("lambda0.9", 0.9, 1.0, 1.0),
    ("lambda1.1", 1.1, 1.0, 1.0),
    ("field0.8", 1.0, 0.8, 1.0),
    ("field1.2", 1.0, 1.2, 1.0),
    ("Bc20.8", 1.0, 1.0, 0.8),
    ("Bc21.2", 1.0, 1.0, 1.2),
]
_CASES = [
    pytest.param(lam * lm, field * fm, bc2 * bm, end, id=f"{anchor}-{variation}")
    for anchor, lam, field, bc2, end in _ANCHORS
    for variation, lm, fm, bm in _VARIATIONS
]


@pytest.mark.parametrize("powder", [False, True])
@pytest.mark.parametrize("lam,field,bc2,end", _CASES)
def test_sampled_literature_anchors_and_nearby_variants(lam, field, bc2, end, powder):
    """Independent quadrature covers sampled points, not a continuous parameter box."""
    times = end * np.unique(np.r_[np.linspace(0, 1, 33), 0.901, 0.947, 0.981, 0.997])
    reference = lattice_average(times, lam, field, bc2, powder=powder)
    actual = vortex_lattice_relaxation(times, lam, field, bc2, powder=powder)
    assert np.max(np.abs(actual - reference)) < 1e-6


@pytest.mark.parametrize("powder", [False, True])
def test_hard_variant_time_shapes_and_signed_component_consumers(powder):
    from asymmetry.core.fitting.sc.lineshape import (
        vortex_lattice_component,
        vortex_lattice_powder_component,
    )

    times = np.linspace(0, 8, 129)
    reference = lattice_average(times, 175.5, 400, 25, powder=powder)
    for axis, expected in [
        (times[::-1], reference[::-1]),
        (-times, reference.conj()),
        (times[:, None], reference[:, None]),
        (times[::7], reference[::7]),
    ]:
        actual = vortex_lattice_relaxation(axis, 175.5, 400, 25, powder=powder)
        assert actual.shape == expected.shape
        assert np.max(np.abs(actual - expected)) < 1e-6
    shifted = times[1::7] + 0.017
    expected = lattice_average(shifted, 175.5, 400, 25, powder=powder)
    assert (
        np.max(np.abs(vortex_lattice_relaxation(shifted, 175.5, 400, 25, powder=powder) - expected))
        < 1e-6
    )
    assert abs(vortex_lattice_relaxation(0, 175.5, 400, 25, powder=powder)[0] - 1) < 1e-14
    component = vortex_lattice_powder_component if powder else vortex_lattice_component
    for sign in [-1, 1]:
        envelope = vortex_lattice_relaxation(times, 175.5, sign * 400, 25, powder=powder)
        assert np.max(np.abs(envelope - reference)) < 1e-6
        for phase in [0, np.pi / 4, np.pi / 2]:
            carrier = np.exp(
                1j
                * (
                    2
                    * np.pi
                    * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
                    * GAUSS_TO_TESLA
                    * sign
                    * 400
                    * times
                    + phase
                )
            )
            actual = component(times, 0.2, sign * 400, phase, 175.5, 25)
            assert np.max(np.abs(actual - 0.2 * (carrier * reference).real)) / 0.2 < 1e-6


@pytest.mark.parametrize(
    "case",
    [(175.5, 400, 25, False), (195, 320, 25, False), (195, 400, 30, False), (175.5, 400, 25, True)],
)
def test_hard_variant_reference_refines_reciprocal_and_spatial_axes_separately(case):
    lam, field, bc2, powder = case
    times = np.r_[np.linspace(0, 8, 17), 7.931, 7.997]
    baseline = lattice_average(times, lam, field, bc2, powder=powder, n_g=100, n_grid=512)
    finer_grid = lattice_average(times, lam, field, bc2, powder=powder, n_g=100, n_grid=768)
    finer_sum = lattice_average(times, lam, field, bc2, powder=powder, n_g=120, n_grid=768)
    assert np.max(np.abs(baseline - finer_grid)) < 1e-7
    assert np.max(np.abs(finer_grid - finer_sum)) < 1e-7


@pytest.mark.parametrize("powder", [False, True])
@pytest.mark.parametrize(
    "n_g,n_grid,n_bins",
    [(3, 25, 64), (10, 21, 128), (10, 22, 128), (20, 41, 256), (10, 96, 512), (80, 384, 512)],
)
def test_resolution_overrides_preserve_empirical_grid_calibration(powder, n_g, n_grid, n_bins):
    from asymmetry.core.fitting.sc.lineshape import _field_offsets_calibrated, _lattice_geometry

    expected = lattice_offsets(599, 300, 7.3, powder=powder, n_g=n_g, n_grid=n_grid)
    actual = _field_offsets_calibrated(599, 300, 7.3, powder=powder, n_g=n_g, n_grid=n_grid)
    assert np.max(np.abs(actual - expected)) < 1e-14
    geometry = _lattice_geometry(n_g, n_grid)
    reduced = _field_offsets_calibrated(
        599, 300, 7.3, powder=powder, n_g=n_g, n_grid=n_grid, reduce_orbits=True
    )
    assert np.max(np.abs(reduced - expected[geometry.orbit_indices])) < 1e-14
    assert abs(geometry.orbit_weights @ reduced**2 - np.mean(expected**2)) < 1e-20
    times = np.linspace(0, 0.5, 17)
    reference = lattice_average(times, 599, 300, 7.3, powder=powder, n_g=n_g, n_grid=n_grid)
    result = vortex_lattice_relaxation(
        times, 599, 300, 7.3, powder=powder, n_g=n_g, n_grid=n_grid, n_bins=n_bins
    )
    assert np.max(np.abs(result - reference)) < 1e-7


def test_independent_fit_recovers_lambda_field_and_phase():
    from scipy.optimize import least_squares

    from asymmetry.core.fitting.sc.lineshape import vortex_lattice_component

    times = np.linspace(0, 8, 129)
    reference = lattice_average(times, 195, 400, 25)
    frequency = 2 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
    nuclear = np.exp(-0.5 * (0.2 * times) ** 2)
    data = 0.2 * (np.exp(1j * (frequency * 400 * times + 0.3)) * reference).real * nuclear

    def residual(parameters):
        lam, field, phase = parameters
        return vortex_lattice_component(times, 0.2, field, phase, lam, 25) * nuclear - data

    result = least_squares(
        residual,
        [210, 400.2, 0.25],
        bounds=([100, 395, -0.5], [300, 405, 1]),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    assert result.success
    assert abs(result.x[0] - 195) < 2e-4
    assert abs(result.x[1] - 400) < 1e-4
    assert abs(result.x[2] - 0.3) < 1e-5
    assert np.max(np.abs(result.fun)) / 0.2 < 1e-6


@pytest.mark.parametrize("step", [1e-3, 1e-7])
def test_lambda_finite_difference_matches_independent_average(step):
    times = np.array([0.1, 0.4, 0.8, 1.5, 2.2, 3.5, 4.7, 6.1, 7.5, 8])
    ref = [lattice_average(times, 195 + sign * step, 400, 25) for sign in [-1, 1]]
    actual = [
        vortex_lattice_relaxation(times, 195 + sign * step, 400, 25, powder=False)
        for sign in [-1, 1]
    ]
    expected_derivative = (ref[1] - ref[0]) / (2 * step)
    derivative = (actual[1] - actual[0]) / (2 * step)
    assert (
        np.max(np.abs(derivative - expected_derivative)) / np.max(np.abs(expected_derivative))
        < 1e-5
    )
