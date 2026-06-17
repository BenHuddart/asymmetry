r"""Vortex-lattice field-distribution lineshape for TF-muSR penetration depth.

The temperature- and field-domain :mod:`~asymmetry.core.fitting.sc.models`
machinery maps a *Gaussian rate* :math:`\sigma_{VL}` to the penetration depth
:math:`\lambda`. That mapping is only as good as the :math:`\sigma_{VL}` you feed
it, and a **single Gaussian fitted to the time domain underestimates**
:math:`\sigma_{VL}` for a vortex lattice: the field distribution :math:`p(B)` of
a flux-line lattice is strongly **non-Gaussian** -- a sharp low-field cutoff at
the saddle point, a van Hove peak below the mean, and a long tail to high field
near the vortex cores (a positively skewed line). A symmetric Gaussian fitted to
the resulting time signal returns a rate that depends on the fit window/binning
rather than the true second moment.

This module provides the **time-domain relaxation of the real** :math:`p(B)`,
so the lineshape itself is fitted instead of a Gaussian proxy. The spatial field
profile is the standard *modified London* model of an ideal triangular flux-line
lattice,

.. math::

    B(\mathbf r) = \bar B \sum_{\mathbf G}
        \frac{e^{-\xi^2 G^2/2}}{1 + \lambda^2 G^2}\, e^{i\mathbf G\cdot\mathbf r},

summed over the reciprocal lattice :math:`\mathbf G` of the triangular FLL, with
core cutoff :math:`\xi=\sqrt{\Phi_0/2\pi B_{c2}}`. The field distribution
:math:`p(B)=\langle\delta(B-B(\mathbf r))\rangle_{\mathbf r}` is sampled on a
real-space grid over one unit cell, and the muon relaxation is its
characteristic function

.. math::

    R(t) = \big\langle e^{i\,2\pi\gamma_\mu (B(\mathbf r)-\bar B)\,t}\big\rangle_{\mathbf r},
    \qquad R(0)=1 .

To stay numerically consistent with the rest of the SC stack, the line's
**second moment is calibrated to** :func:`brandt_field_width_sigma` /
:func:`brandt_field_width_sigma_powder` -- i.e. the *width* (hence the extracted
:math:`\lambda` and :math:`B_{c2}`) is exactly the validated Brandt result, while
the modified-London computation supplies only the **shape** (skew, higher
moments) that a Gaussian lacks. Fitting this lineshape and reading
:math:`\sigma_{VL}\to\lambda` through the existing converters are therefore
guaranteed to agree.

References
----------
E. H. Brandt, *Phys. Rev. B* **68**, 054506 (2003).
J. E. Sonier, J. H. Brewer, R. F. Kiefl, *Rev. Mod. Phys.* **72**, 769 (2000).
F. L. Pratt *et al.*, *Phys. Rev. B* **79**, 052508 (2009) (LiFeAs powder).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.sc.constants import FLUX_QUANTUM_WB
from asymmetry.core.fitting.sc.models import (
    _POWDER_LAMBDA_FACTOR,
    brandt_field_width_sigma,
)
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

ArrayLikeFloat = NDArray[np.float64]

#: rad per (microsecond * tesla): the muon precession phase rate.
_TWO_PI_GAMMA = 2.0 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

#: Default reciprocal-lattice half-range and real-space grid. The *shape*
#: (which is all this computation supplies — the width is calibrated to Brandt)
#: converges by ``n_g≈8``; 10/96 is comfortably converged.
_DEFAULT_N_G = 10
_DEFAULT_N_GRID = 96

#: Number of field bins for the cached ``p(B)`` histogram the relaxation reduces
#: over. The line *width* is calibrated to Brandt, so this only sets how finely
#: the *shape* is resolved when forming ``R(t)``; 512 bins keep the fitted
#: ``λ_ab`` within ~0.1 nm of the full real-space average (see test_sc_vl_lineshape).
_DEFAULT_N_BINS = 512


@lru_cache(maxsize=256)
def _centered_field_offsets(
    lambda_eff_nm: float,
    B0_tesla: float,
    Bc2_tesla: float,
    n_g: int,
    n_grid: int,
) -> ArrayLikeFloat:
    r"""Sample ``B(r) - B0`` (tesla) of the modified-London triangular FLL.

    Cached on rounded float keys; returns the flat array of centered field
    offsets sampled at the ``n_grid × n_grid`` cell-centred real-space points
    over one unit cell (mean ≈ 0).

    The modified-London sum ``B(r)/B0 = Σ_G h_G e^{iG·r}`` is evaluated by a 2D
    inverse FFT rather than an explicit ``O(N_G·N_grid)`` double sum: with the
    real-space points ``r_{jk} = ((j+½)/N)a₁ + ((k+½)/N)a₂`` the phase is
    ``G·r = 2π(mj+nk)/N + π(m+n)/N``, so multiplying ``h_{mn}`` by the half-cell
    phase ``e^{iπ(m+n)/N}`` and inverse-FFTing the reciprocal grid reproduces the
    same field samples exactly (to machine precision) for far less work. The
    returned *set* of values is identical to the direct sum; only the (irrelevant)
    ordering differs, and every consumer reduces over it order-independently.
    """
    if n_grid < 2 * n_g + 1:
        # Each reciprocal mode m,n in [-n_g, n_g] must map to a distinct FFT grid
        # index; otherwise modes alias (the np.ix_ assignment would drop colliders
        # last-wins instead of folding them) and the field map is silently wrong.
        raise ValueError(
            f"n_grid ({n_grid}) must be >= 2*n_g+1 ({2 * n_g + 1}) to avoid "
            "reciprocal-lattice aliasing in the FFT field-map evaluation."
        )
    lam = lambda_eff_nm * 1.0e-9
    xi = np.sqrt(FLUX_QUANTUM_WB / (2.0 * np.pi * Bc2_tesla))
    # Triangular FLL: area per vortex = Phi0/B0 = (sqrt(3)/2) a^2.
    a = np.sqrt(2.0 * FLUX_QUANTUM_WB / (np.sqrt(3.0) * B0_tesla))
    a1 = np.array([a, 0.0])
    a2 = np.array([0.5 * a, np.sqrt(3.0) / 2.0 * a])
    cell = abs(a1[0] * a2[1] - a1[1] * a2[0])
    b1 = 2.0 * np.pi * np.array([a2[1], -a2[0]]) / cell
    b2 = 2.0 * np.pi * np.array([-a1[1], a1[0]]) / cell

    ms = np.arange(-n_g, n_g + 1)
    grid_m, grid_n = np.meshgrid(ms, ms)
    gx = grid_m * b1[0] + grid_n * b2[0]
    gy = grid_m * b1[1] + grid_n * b2[1]
    g2 = gx**2 + gy**2
    h = np.exp(-(xi**2) * g2 / 2.0) / (1.0 + lam**2 * g2)
    # G=0 is the uniform mean field (set explicitly so mean(B(r)/B0) is exactly 1).
    h[n_g, n_g] = 1.0
    # Half-cell phase for cell-centred sampling; place h e^{iπ(m+n)/N} on the
    # N×N reciprocal grid (negative indices wrap) and inverse-FFT to real space.
    h_shifted = h * np.exp(1j * np.pi * (grid_m + grid_n) / n_grid)
    spectrum = np.zeros((n_grid, n_grid), dtype=np.complex128)
    wrapped = ms % n_grid
    spectrum[np.ix_(wrapped, wrapped)] = h_shifted
    profile = np.fft.ifft2(spectrum).real * (n_grid * n_grid)
    return np.ascontiguousarray((profile.ravel() - 1.0) * B0_tesla)


def _field_offsets_calibrated(
    lambda_nm: float,
    B0_gauss: float,
    Bc2_tesla: float,
    *,
    powder: bool,
    n_g: int,
    n_grid: int,
) -> ArrayLikeFloat | None:
    """Centered offsets (tesla) rescaled so the second moment equals the Brandt
    rate. Returns ``None`` for the degenerate (no-lattice) case."""
    B0_tesla = abs(float(B0_gauss)) * GAUSS_TO_TESLA
    lam_eff = float(lambda_nm) * (_POWDER_LAMBDA_FACTOR if powder else 1.0)
    if B0_tesla <= 0.0 or Bc2_tesla <= 0.0 or lambda_nm <= 0.0 or B0_tesla >= Bc2_tesla:
        return None

    offsets = _centered_field_offsets(
        round(lam_eff, 3), round(B0_tesla, 9), round(float(Bc2_tesla), 6), int(n_g), int(n_grid)
    )
    raw_rms = float(np.sqrt(np.mean(offsets**2)))
    # Target rate from the validated Brandt model (powder factor already folded
    # into lam_eff above, so call the single-crystal width with lam_eff).
    target_rate = float(brandt_field_width_sigma(B0_gauss, lam_eff, Bc2_tesla, 0.0, powder=False))
    if raw_rms <= 0.0 or target_rate <= 0.0:
        return None
    target_rms_tesla = target_rate / _TWO_PI_GAMMA
    return offsets * (target_rms_tesla / raw_rms)


@lru_cache(maxsize=256)
def _calibrated_field_histogram(
    lambda_nm: float,
    B0_gauss: float,
    Bc2_tesla: float,
    powder: bool,
    n_g: int,
    n_grid: int,
    n_bins: int,
) -> tuple[ArrayLikeFloat, ArrayLikeFloat] | None:
    r"""Field distribution ``p(B)`` of the calibrated line as ``(centres, weights)``.

    Bins the ``n_grid²`` calibrated field offsets (tesla, mean ≈ 0) into ``n_bins``
    and returns the bin centres and their normalised weights (``Σ weights = 1``).
    Cached on the shape-determining params so a minimiser pays the grid build and
    histogram once per distinct ``(λ, B0, B_c2)`` and every ``R(t)`` evaluation is
    then a cheap ``n_bins``-term characteristic-function sum. Returns ``None`` for
    the degenerate (no-lattice) case. The histogram is used *only* for ``R(t)``;
    second-moment/skew consumers read the full offsets from
    :func:`_field_offsets_calibrated` directly, so the calibrated width is exact.
    """
    offsets = _field_offsets_calibrated(
        lambda_nm, B0_gauss, Bc2_tesla, powder=powder, n_g=n_g, n_grid=n_grid
    )
    if offsets is None:
        return None
    counts, edges = np.histogram(offsets, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    if total <= 0:
        return None
    weights = counts / total
    centres = np.ascontiguousarray(centres)
    weights = np.ascontiguousarray(weights)
    # Cached arrays are shared across calls; freeze them against accidental mutation.
    centres.setflags(write=False)
    weights.setflags(write=False)
    return centres, weights


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
    r"""Complex time-domain relaxation :math:`R(t)` of the VL field distribution.

    The measured polarisation is
    ``P_x(t) = Re[ exp(i(2*pi*gamma*B0*t + phase)) * R(t) ]``. ``R(0) = 1`` and
    ``|R(t)|`` is the depolarisation envelope; ``arg R(t)`` carries the skew of
    the line. For ``B0 >= Bc2`` (or non-physical inputs) there is no lattice and
    ``R(t) = 1``. A scalar ``t`` returns a length-1 array.

    ``R(t)`` is the characteristic function of the field distribution,
    ``Σ_B p(B) e^{i 2π γ B t}``, evaluated over the cached ``n_bins``-bin ``p(B)``
    histogram rather than over every real-space grid point — the per-call cost is
    ``O(n_bins · N_t)`` instead of ``O(n_grid² · N_t)``, with ``n_bins`` chosen so
    the fitted ``λ_ab`` is unchanged within tolerance.
    """
    t = np.atleast_1d(np.asarray(t_us, dtype=float))
    histogram = _calibrated_field_histogram(
        lambda_nm, B0_gauss, Bc2_tesla, powder, n_g, n_grid, n_bins
    )
    if histogram is None:
        return np.ones(t.shape, dtype=np.complex128)
    centres, weights = histogram
    return weights @ np.exp(1j * _TWO_PI_GAMMA * centres[:, None] * t[None, :])


def vortex_lattice_powder_relaxation(
    t_us: ArrayLikeFloat | list[float] | float,
    lambda_ab_nm: float,
    B0_gauss: float,
    Bc2_tesla: float,
) -> NDArray[np.complex128]:
    """Polycrystalline variant of :func:`vortex_lattice_relaxation`."""
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
    freq_mhz = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * float(field)
    r = vortex_lattice_relaxation(t, lambda_ab, field, Bc2, powder=powder)
    carrier = np.exp(1j * (2.0 * np.pi * freq_mhz * t + float(phase)))
    return float(A) * np.real(carrier * r)


def vortex_lattice_component(
    t_us: ArrayLikeFloat,
    A: float,
    field: float,
    phase: float,
    lambda_ab: float,
    Bc2: float,
) -> ArrayLikeFloat:
    r"""Single-crystal vortex-lattice oscillation component ``f(t)``.

    ``A`` asymmetry amplitude, ``field`` in gauss, ``phase`` in radians,
    ``lambda_ab`` (here the single-crystal :math:`\lambda`) in nm, ``Bc2`` in
    tesla. Compose with a Gaussian (nuclear dipolar broadening, multiplied) and a
    plain Oscillatory + Constant (sample-holder background) as needed.
    """
    return _vortex_lattice_signal(t_us, A, field, phase, lambda_ab, Bc2, powder=False)


def vortex_lattice_powder_component(
    t_us: ArrayLikeFloat,
    A: float,
    field: float,
    phase: float,
    lambda_ab: float,
    Bc2: float,
) -> ArrayLikeFloat:
    r"""Polycrystalline vortex-lattice oscillation component ``f(t)``.

    As :func:`vortex_lattice_component` but ``lambda_ab`` is the ab-plane depth
    of a powder; the second moment uses the :math:`3^{1/4}\lambda_{ab}` average
    (Pratt et al. Eq. (3)), consistent with
    :func:`~asymmetry.core.fitting.sc.models.brandt_field_width_sigma_powder`.
    """
    return _vortex_lattice_signal(t_us, A, field, phase, lambda_ab, Bc2, powder=True)
