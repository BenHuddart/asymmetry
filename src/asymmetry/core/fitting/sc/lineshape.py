r"""Calibrated Gaussian-cutoff London lineshape for an ideal triangular lattice.

The raw reciprocal coefficients are ``exp(-xi² G²/2)/(1 + lambda² G²)``,
with ``xi² = Phi0/(2*pi*Bc2)``. Field deviations are rescaled to the Brandt
second moment; the powder variant substitutes ``3**(1/4) * lambda_ab``.
These are distinct assumptions: matching a second moment does not validate
an entire field distribution or perform a physical powder orientation average.

Midpoint unit-cell quadrature uses a real inverse FFT. Uniform field bins
retain local moments through fourth order, so their characteristic function
includes sub-bin structure. A unit-circle transform handles uniform time axes;
a polynomial recurrence preserves arbitrary time-array shapes.

References
----------
R. Kadono et al., Phys. Rev. B 76, 094501 (2007), Eqs. (5)–(7).
J. A. T. Barker et al., Phys. Rev. B 98, 104506 (2018), Eqs. (32)–(35).
E. H. Brandt, Phys. Rev. B 68, 054506 (2003), Eq. (13) (second moment).
F. L. Pratt et al., Phys. Rev. B 79, 052508 (2009), Eq. (3) (powder width).
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft2

from asymmetry.core.fitting.sc.models import (
    _POWDER_LAMBDA_FACTOR,
    brandt_field_width_sigma,
)
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MAGNETIC_FLUX_QUANTUM_WB,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

ArrayLikeFloat = NDArray[np.float64]
_TWO_PI_GAMMA = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
_DEFAULT_N_G = 80
_DEFAULT_N_GRID = 384
_DEFAULT_N_BINS = 512
_MOMENT_ORDER = 4
_UNIFORM_T_RTOL = 1.0e-9
_CZT_MIN_BINS = 128


class _LatticeGeometry(NamedTuple):
    metric: NDArray[np.int64]
    midpoint_phase: NDArray[np.complex128]
    rows: NDArray[np.int64]
    orbit_indices: NDArray[np.int64]
    orbit_weights: ArrayLikeFloat


@lru_cache(maxsize=1)
def _lattice_geometry(n_g: int, n_grid: int) -> _LatticeGeometry:
    """Parameter-independent half-spectrum and midpoint symmetry orbits."""
    if n_grid < 2 * n_g + 1:
        raise ValueError(
            f"n_grid ({n_grid}) must be >= 2*n_g+1 ({2 * n_g + 1}) to avoid "
            "reciprocal-lattice aliasing in the FFT field-map evaluation."
        )
    m = np.arange(-n_g, n_g + 1)[:, None]
    n = np.arange(n_g + 1)[None, :]
    metric = m * m + n * n - m * n
    midpoint_phase = np.exp(1j * np.pi * (m + n) / n_grid)
    j, k = np.triu_indices(n_grid)
    keep = j + k < n_grid
    j, k = j[keep], k[keep]
    # Exchange (j,k)->(k,j) and inversion ->(N-1-j,N-1-k) preserve the
    # triangular coefficients. Diagonal orbits have two points, general ones four.
    multiplicity = np.where((j == k) | (j + k == n_grid - 1), 2.0, 4.0)
    if n_grid % 2:
        multiplicity[(j == k) & (j + k == n_grid - 1)] = 1.0
    geometry = _LatticeGeometry(
        metric, midpoint_phase, m[:, 0] % n_grid, j * n_grid + k, multiplicity / n_grid**2
    )
    for array in geometry:
        array.setflags(write=False)
    return geometry


def _centered_field_offsets(
    lambda_eff_nm: float,
    B0_tesla: float,
    Bc2_tesla: float,
    n_g: int,
    n_grid: int,
) -> ArrayLikeFloat:
    r"""Fresh full-grid samples of ``B(r)-B0`` in tesla, without a raw-map cache.

    The triangular reciprocal metric is ``G²=8*pi²*B/Phi0/sqrt(3)*(m²+n²-mn)``.
    The midpoint phase shifts the inverse FFT to ``r=((j+1/2)a1+(k+1/2)a2)/N``.
    Conjugate reciprocal coefficients permit a real inverse transform.
    """
    geometry = _lattice_geometry(n_g, n_grid)
    cutoff = 2 * np.pi * B0_tesla / (np.sqrt(3) * Bc2_tesla)
    screening = (
        8
        * np.pi**2
        * B0_tesla
        * (lambda_eff_nm * 1e-9) ** 2
        / (np.sqrt(3) * MAGNETIC_FLUX_QUANTUM_WB)
    )
    coefficients = np.exp(-cutoff * geometry.metric) / (1 + screening * geometry.metric)
    coefficients[n_g, 0] = 0.0  # G=0 is the subtracted uniform mean field.
    spectrum = np.zeros((n_grid, n_grid // 2 + 1), dtype=complex)
    spectrum[geometry.rows, : n_g + 1] = coefficients * geometry.midpoint_phase
    return irfft2(spectrum, s=(n_grid, n_grid)).ravel() * (n_grid**2 * B0_tesla)


def _field_offsets_calibrated(
    lambda_nm: float,
    B0_gauss: float,
    Bc2_tesla: float,
    *,
    powder: bool,
    n_g: int,
    n_grid: int,
    reduce_orbits: bool = False,
) -> ArrayLikeFloat | None:
    """Calibrated full-grid or exact orbit samples, with empirical grid RMS."""
    field_tesla = abs(float(B0_gauss)) * GAUSS_TO_TESLA
    if not 0 < lambda_nm < np.inf or field_tesla <= 0 or Bc2_tesla <= 0 or field_tesla >= Bc2_tesla:
        return None
    effective_lambda = float(lambda_nm) * (_POWDER_LAMBDA_FACTOR if powder else 1.0)
    offsets = _centered_field_offsets(
        effective_lambda, field_tesla, float(Bc2_tesla), int(n_g), int(n_grid)
    )
    if reduce_orbits:
        geometry = _lattice_geometry(n_g, n_grid)
        offsets = offsets[geometry.orbit_indices]
        raw_rms = float(np.sqrt(geometry.orbit_weights @ offsets**2))
    else:
        raw_rms = float(np.sqrt(np.mean(offsets**2)))
    rate = float(brandt_field_width_sigma(B0_gauss, effective_lambda, Bc2_tesla, 0, powder=False))
    if raw_rms <= 0 or rate <= 0:
        return None
    # Empirical RMS also preserves calibration for deliberately coarse grid overrides.
    offsets *= rate / (_TWO_PI_GAMMA * raw_rms)
    return offsets


@lru_cache(maxsize=16)
def _calibrated_field_moments(
    lambda_nm: float,
    B0_gauss: float,
    Bc2_tesla: float,
    powder: bool,
    n_g: int,
    n_grid: int,
    n_bins: int,
) -> tuple[float, float, ArrayLikeFloat] | None:
    """Uniform-bin centres and weighted local moments of the calibrated field."""
    offsets = _field_offsets_calibrated(
        lambda_nm,
        B0_gauss,
        Bc2_tesla,
        powder=powder,
        n_g=n_g,
        n_grid=n_grid,
        reduce_orbits=True,
    )
    if offsets is None:
        return None
    geometry = _lattice_geometry(n_g, n_grid)
    edges = np.histogram_bin_edges(offsets, bins=n_bins)
    delta = float((edges[-1] - edges[0]) / n_bins)
    positions = (offsets - edges[0]) / delta
    indices = np.minimum(positions.astype(int), n_bins - 1)
    relative = positions - indices - 0.5
    moments = np.empty((_MOMENT_ORDER + 1, n_bins))
    power = np.ones(offsets.size)
    for order in range(_MOMENT_ORDER + 1):
        moments[order] = np.bincount(
            indices, weights=geometry.orbit_weights * power, minlength=n_bins
        )
        power *= relative
    moments.setflags(write=False)
    return float(edges[0] + delta / 2), delta, moments


def _horner_inner_sum(
    weights: ArrayLikeFloat, t: ArrayLikeFloat, delta: float
) -> NDArray[np.complex128]:
    """Batched ``sum_k weights[...,k] exp(i*gamma*delta*t*k)`` for any time shape."""
    shape = weights.shape[:-1] + (1,) * t.ndim
    accumulator = np.array(
        np.broadcast_to(weights[..., -1].reshape(shape), weights.shape[:-1] + t.shape),
        dtype=complex,
    )
    z = np.exp(1j * _TWO_PI_GAMMA * delta * t)
    for index in range(weights.shape[-1] - 2, -1, -1):
        accumulator *= z
        accumulator += weights[..., index].reshape(shape)
    return accumulator


def _czt_inner_sum(
    weights: ArrayLikeFloat, t: ArrayLikeFloat, delta: float
) -> NDArray[np.complex128] | None:
    """Batched unit-circle transform, or decline a nonuniform/nonvector time axis."""
    if t.ndim != 1 or t.size < 2 or weights.shape[-1] < _CZT_MIN_BINS:
        return None
    start = float(t[0])
    step = (float(t[-1]) - start) / (t.size - 1)
    if step == 0 or not np.all(np.abs(np.diff(t) - step) <= _UNIFORM_T_RTOL * abs(step)):
        return None
    from scipy.signal import zoom_fft

    phase = _TWO_PI_GAMMA * delta
    return np.asarray(
        zoom_fft(
            weights, [-phase * start, -phase * float(t[-1])], t.size, fs=2 * np.pi, endpoint=True
        ),
        dtype=complex,
    )


def _characteristic_function(
    centre: float, delta: float, moments: ArrayLikeFloat, t: ArrayLikeFloat
) -> NDArray[np.complex128]:
    r"""Fourth-order sub-bin quadrature of ``<exp(i*gamma*offset*t)>``.

    With ``offset=c_k+delta*r``, each moment row is ``sum weights*r**p``.
    The omitted remainder is bounded by ``(gamma*delta*max|t|)**5 E|r|**5/5!``;
    this controls compression only, separately from field-map discretization.
    """
    if t.size == 0:
        return np.zeros(t.shape, dtype=complex)
    inner = _czt_inner_sum(moments, t, delta)
    if inner is None:
        inner = _horner_inner_sum(moments, t, delta)
    phase = 1j * _TWO_PI_GAMMA * delta * t
    result = inner[-1] / factorial(_MOMENT_ORDER)
    for order in range(_MOMENT_ORDER - 1, -1, -1):
        result *= phase
        result += inner[order] / factorial(order)
    return np.exp(1j * _TWO_PI_GAMMA * centre * t) * result


def vortex_lattice_relaxation(
    t_us: ArrayLikeFloat | list[float] | float,
    lambda_nm: float,
    B0_gauss: float,
    Bc2_tesla: float,
    *,
    powder: bool = True,
    n_g: int = _DEFAULT_N_G,
    n_grid: int = _DEFAULT_N_GRID,
    n_bins: int = _DEFAULT_N_BINS,
) -> NDArray[np.complex128]:
    r"""Complex relaxation of the calibrated Gaussian-cutoff London distribution.

    The real signal is ``Re[exp(i*(2*pi*gamma*B0*t+phase))*R(t)]``. The envelope
    uses ``abs(B0)`` and the carrier uses the signed field. Scalar times return
    a length-one array. Other time-array shapes are preserved. A NaN depth gives
    complex NaNs. Positive infinite depth gives unity (zero lattice width).
    Nonpositive depth or Bc2, zero field, and fields at or above Bc2 also give unity.

    Defaults use reciprocal extent 80, a 384² midpoint grid and 512 field bins
    with fourth-order local moments. Explicit resolution overrides retain their
    meanings but do not inherit accuracy demonstrated for the default settings.
    """
    t = np.atleast_1d(np.asarray(t_us, dtype=float))
    if np.isnan(lambda_nm):
        return np.full(t.shape, complex(np.nan, np.nan))
    quadrature = _calibrated_field_moments(
        lambda_nm, B0_gauss, Bc2_tesla, powder, int(n_g), int(n_grid), n_bins
    )
    if quadrature is None:
        return np.ones(t.shape, dtype=complex)
    centre, delta, moments = quadrature
    return _characteristic_function(centre, delta, moments, t)


def vortex_lattice_powder_relaxation(
    t_us: ArrayLikeFloat | list[float] | float,
    lambda_ab_nm: float,
    B0_gauss: float,
    Bc2_tesla: float,
) -> NDArray[np.complex128]:
    """Scalar effective-lambda powder variant of :func:`vortex_lattice_relaxation`."""
    return vortex_lattice_relaxation(t_us, lambda_ab_nm, B0_gauss, Bc2_tesla, powder=True)


def _vortex_lattice_signal(
    t_us: ArrayLikeFloat,
    A: float,
    field: float,
    phase: float,
    lambda_ab: float,
    Bc2: float,
    *,
    powder: bool,
) -> ArrayLikeFloat:
    t = np.asarray(t_us, dtype=float)
    frequency = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * float(field)
    relaxation = vortex_lattice_relaxation(t, lambda_ab, field, Bc2, powder=powder)
    carrier = np.exp(1j * (2 * np.pi * frequency * t + float(phase)))
    return float(A) * np.real(carrier * relaxation)


def vortex_lattice_component(
    t_us: ArrayLikeFloat,
    A: float,
    field: float,
    phase: float,
    lambda_ab: float,
    Bc2: float,
) -> ArrayLikeFloat:
    """Single-crystal signal: field in gauss, phase in radians, lambda in nm, Bc2 in T."""
    return _vortex_lattice_signal(t_us, A, field, phase, lambda_ab, Bc2, powder=False)


def vortex_lattice_powder_component(
    t_us: ArrayLikeFloat,
    A: float,
    field: float,
    phase: float,
    lambda_ab: float,
    Bc2: float,
) -> ArrayLikeFloat:
    """Scalar powder signal with effective lambda = 3^(1/4) * lambda_ab (nm)."""
    return _vortex_lattice_signal(t_us, A, field, phase, lambda_ab, Bc2, powder=True)
