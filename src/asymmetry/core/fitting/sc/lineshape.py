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
from scipy.signal import czt

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


#: Relative tolerance on the spread of ``np.diff(t)`` for the time axis to count
#: as uniform (and so unlock the czt fast path). A µSR time axis is generated as
#: ``t0 + j·dt`` (or ``np.linspace``), so its diffs agree to a few ulp — 1e-9
#: relative is many orders of magnitude looser than that yet far tighter than any
#: physically deliberate non-uniformity (rebinned/log axes differ by ≫1e-9).
_UNIFORM_T_RTOL = 1.0e-9

#: Bin count at or above which the czt fast path beats the Horner recurrence.
#: The crossover is set by ``n_bins``, not by ``N_t``: Horner pays ``n_bins - 1``
#: Python-level array ops whatever the axis length, so its cost has a large
#: ``n_bins``-proportional floor, while czt pays one ``O((n_bins+N_t)log)`` FFT
#: pair. Measured ratio Horner/czt (>1 means czt wins), this machine, uniform
#: axis, ``python 3.12 / numpy 2.2.6 / scipy 1.17.1``::
#:
#:      N_t ->      2      16     128     512    2048    8192
#:   n_bins=8    0.25    0.25    0.21    0.16    0.13    0.12
#:   n_bins=32   0.80    0.79    0.64    0.43    0.29    0.22
#:   n_bins=128  2.02    2.06    2.23    1.48    0.88    0.77
#:   n_bins=512  3.60    3.70    3.92    5.26    3.46    2.96
#:   n_bins=2048 4.63    4.66    5.09    7.06   11.88   10.49
#:
#: 128 is the smallest tabulated bin count that wins across the axis lengths a
#: µSR fit actually sees (worst case at n_bins=128 is 1.3× *slower* on a
#: 8192-point axis; the shipped default is 512, where czt wins 3–5× everywhere).
_CZT_MIN_BINS = 128


# Measured on the realistic minimiser pattern — λ different on every call, so the
# p(B) histogram lru_cache misses every time, as it does during an iminuit line
# search — powder line at (λ=195 nm, B0=400 G, Bc2=25 T), ms per call, against
# the retained `_reference_relaxation` (the pre-optimisation body):
#
#     axis                     before     after   speedup
#     uniform      N_t=1024    15.09      0.27      56x     (czt)
#     uniform      N_t=4096    56.70      0.87      65x     (czt)
#     uniform      N_t=8192   100.81      1.12      90x     (czt)
#     non-uniform  N_t=1024    15.36      0.94      16x     (Horner)
#     non-uniform  N_t=4096    56.93      1.62      35x     (Horner)
#     non-uniform  N_t=8192   103.81      2.78      37x     (Horner)
#
# End to end a six-parameter curve_fit of the powder component runs 25× (N_t=400)
# to 50× (N_t=8192) faster, at an identical fitted λ_ab (≤4e-7 nm apart).
# The rest of a cache-missing call is now the p(B) rebuild — 0.14 ms for the 96²
# ifft2 field map plus the Brandt calibration, 0.27 ms including the 9216-point
# histogram, against 0.30 ms for the characteristic function itself at N_t=4096.
# Both were negligible before this change and neither is worth optimising now.
def _characteristic_function(
    centres: ArrayLikeFloat,
    weights: ArrayLikeFloat,
    t: ArrayLikeFloat,
) -> NDArray[np.complex128]:
    r"""``Σ_k w_k e^{i 2π γ c_k t}`` without ever forming the ``n_bins × N_t`` matrix.

    The naive evaluation is a complex ``exp`` of an ``n_bins × N_t`` outer product,
    which dominates the cost of *every* model evaluation during a fit (the
    histogram cache keys on ``λ``, so a minimiser's line search misses it on
    essentially every call). Both tiers below exploit the one structural fact the
    naive form throws away: :func:`np.histogram` bin centres are **uniform**,
    ``c_k = c₀ + kΔ``, so

    .. math::

        R(t) = e^{i\gamma c_0 t}\sum_k w_k z^k, \qquad z = e^{i\gamma\Delta t},
        \qquad \gamma \equiv 2\pi\gamma_\mu .

    **Tier 1 — geometric (Horner) recurrence, any time axis.** Evaluate the
    polynomial ``Σ_k w_k z^k`` by Horner's rule in ``z``, so only ``N_t`` complex
    exponentials are needed (for ``z`` itself) and the powers come from
    ``n_bins - 1`` in-place complex multiply-adds over length-``N_t`` arrays. No
    ``n_bins × N_t`` temporary is ever materialised: peak memory drops from
    ``O(n_bins·N_t)`` to ``O(N_t)``. Numerically ``|z| = 1``, so the recurrence is
    neutrally stable — 512 unit-modulus steps accumulate only ~``n_bins·ε`` of
    phase, ~1e-13, confirmed at ≤1e-10 against the reference by
    ``tests/core/test_sc_vl_lineshape_evaluation.py``.

    **Tier 2 — chirp z-transform, uniform time axis (the µSR norm).** With
    ``t_j = t₀ + j·δt`` the inner sum becomes

    .. math::

        S_j = \sum_k w_k A^k W^{jk}, \qquad
        A = e^{i\gamma\Delta t_0}, \quad W = e^{i\gamma\Delta\,\delta t},

    which is exactly a chirp z-transform of the weight vector. SciPy evaluates
    ``X[j] = Σ_k x_k z_j^{-k}`` at ``z_j = a·w^{-j}``, i.e. ``X[j] = Σ_k x_k
    a^{-k} w^{jk}``; matching term by term gives ``w = W = exp(+iγΔ·δt)`` and
    ``a = A⁻¹ = exp(-iγΔ·t₀)`` — the ``t₀`` offset is absorbed entirely into the
    czt starting point ``a``, and the ``e^{iγc₀t}`` carrier is applied afterwards
    using the *caller's* ``t`` (not a reconstructed one), so a t-axis that is
    uniform only to floating-point round-off stays exact in the carrier.
    Note the ``+`` in ``w``: the physics convention here is ``e^{+iγBt}`` while
    SciPy's z-transform convention carries the inverse power, and the two signs
    cancel. Cost is ``O((n_bins + N_t) log(n_bins + N_t))``.

    Tier 2 is used only when ``t`` is a 1-D vector of at least two points that is
    uniform within :data:`_UNIFORM_T_RTOL`, and the histogram has at least
    :data:`_CZT_MIN_BINS` bins; everything else (scalar, empty, log-spaced,
    ragged, non-vector, coarse-binned) takes Tier 1, which is itself ~10–40×
    faster than the exp matrix it replaces.
    """
    n_bins = centres.size
    if t.size == 0:
        return np.zeros(0, dtype=np.complex128)
    if n_bins == 1:
        return np.asarray(
            weights[0] * np.exp(1j * _TWO_PI_GAMMA * centres[0] * t), dtype=np.complex128
        )

    c0 = float(centres[0])
    # Uniform by construction (np.histogram edges are linspace); the endpoint
    # form is the best-conditioned estimate of the common spacing.
    delta = (float(centres[-1]) - c0) / (n_bins - 1)
    carrier = np.exp(1j * _TWO_PI_GAMMA * c0 * t)

    inner = _czt_inner_sum(weights, t, delta)
    if inner is None:
        inner = _horner_inner_sum(weights, t, delta)
    return carrier * inner


def _horner_inner_sum(
    weights: ArrayLikeFloat, t: ArrayLikeFloat, delta: float
) -> NDArray[np.complex128]:
    """``Σ_k w_k z^k`` with ``z = e^{iγΔt}`` by Horner's rule (Tier 1; any ``t``)."""
    z = np.exp(1j * _TWO_PI_GAMMA * delta * t)
    acc = np.full(t.shape, weights[-1], dtype=np.complex128)
    for k in range(weights.size - 2, -1, -1):
        acc *= z
        acc += weights[k]
    return acc


def _czt_inner_sum(
    weights: ArrayLikeFloat, t: ArrayLikeFloat, delta: float
) -> NDArray[np.complex128] | None:
    """``Σ_k w_k z_j^k`` via ``scipy.signal.czt`` (Tier 2), or ``None`` when the
    axis is not uniform or the transform would not pay for itself."""
    n_t = t.size
    # ndim: the transform is over a single axis, and a (N, 1)-shaped ``t`` would
    # otherwise broadcast the flat czt result against the 2-D carrier into a
    # silently wrong (N, N). Anything but a plain vector takes the recurrence,
    # which broadcasts elementwise and so keeps ``t``'s shape whatever it is.
    if t.ndim != 1 or n_t < 2 or weights.size < _CZT_MIN_BINS:
        return None
    t0 = float(t[0])
    step = (float(t[-1]) - t0) / (n_t - 1)
    if step == 0.0 or not np.all(np.abs(np.diff(t) - step) <= _UNIFORM_T_RTOL * abs(step)):
        return None
    phase = _TWO_PI_GAMMA * delta
    return np.asarray(
        czt(weights, n_t, np.exp(1j * phase * step), np.exp(-1j * phase * t0)),
        dtype=np.complex128,
    )


def _reference_relaxation(
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
    """Straight-line ``R(t)`` definition, kept **only** as a test oracle.

    This is the pre-optimisation body of :func:`vortex_lattice_relaxation`: the
    explicit ``n_bins × N_t`` complex-exponential matrix contracted against the
    weights. It is the unambiguous statement of what ``R(t)`` means, so the two
    tiers in :func:`_characteristic_function` are pinned against it rather than
    against each other. Not part of the public API and not used at runtime.
    """
    t = np.atleast_1d(np.asarray(t_us, dtype=float))
    histogram = _calibrated_field_histogram(
        lambda_nm, B0_gauss, Bc2_tesla, powder, n_g, n_grid, n_bins
    )
    if histogram is None:
        return np.ones(t.shape, dtype=np.complex128)
    centres, weights = histogram
    return weights @ np.exp(1j * _TWO_PI_GAMMA * centres[:, None] * t[None, :])


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
    histogram rather than over every real-space grid point — so the sum is over
    ``n_bins`` terms, not ``n_grid²``, with ``n_bins`` chosen so the fitted
    ``λ_ab`` is unchanged within tolerance. The sum itself is evaluated by
    :func:`_characteristic_function`, which uses the uniformity of the histogram
    bin centres to avoid the ``n_bins × N_t`` complex-exponential matrix
    entirely (chirp z-transform on a uniform time axis, geometric recurrence
    otherwise).
    """
    t = np.atleast_1d(np.asarray(t_us, dtype=float))
    histogram = _calibrated_field_histogram(
        lambda_nm, B0_gauss, Bc2_tesla, powder, n_g, n_grid, n_bins
    )
    if histogram is None:
        return np.ones(t.shape, dtype=np.complex128)
    centres, weights = histogram
    return _characteristic_function(centres, weights, t)


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
