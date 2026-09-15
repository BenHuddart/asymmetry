"""Exact muon polarization for the local-field distribution of a helical magnet.

At a muon site in a helical (or cycloidal, or any single-q) magnetic structure
the local field vector traces an ellipse centred on zero,
``B(φ) = B_max cos φ ê₁ + B_min sin φ ê₂``, with the helix phase φ uniformly
distributed. Its magnitude follows

    D(B) = (2/π) B / sqrt((B² − B_min²)(B_max² − B²)),   B_min ≤ B ≤ B_max

(A. Amato *et al.*, Phys. Rev. B 89, 184425 (2014), Eq. 10), whose cosine
transform has no closed form. Amato's shifted-Overhauser approximation (their
Eq. 15, ``OverhauserPowderCutoff``) replaces D by the arcsine density on
``[B_min, B_max]``; the exact density is that arcsine density times the smooth
factor ``h(B) = 2B / sqrt((B + B_min)(B + B_max))``. With the Chebyshev
coefficients c_n of h on ``[B_min, B_max]`` this gives the exact expansion

    ∫ D(B) cos(γ_μ B t + φ) dB = Σ_n c_n J_n(Δω t) cos(ω_av t + φ + nπ/2),

with ``ω_av, Δω = γ_μ (B_max ± B_min)/2``. The c_n depend only on the ratio
``r = B_min/B_max`` and decay geometrically, so the series is summed with Bessel
functions from an upward recurrence where that is stable (Δω t above the
series length) and replaced by a Gauss–Chebyshev quadrature of the same
factorised integrand at earlier times.

Single crystal: with the initial polarization along P̂₀ and ``a = ê₁·P̂₀``,
``c = ê₂·P̂₀``, the non-precessing weight of a field ``b_z²`` depends on |B| only,
``b_z² = c² + (a² − c²) λ(B)`` with ``λ = (1 − B_min²/B²)/(1 − r²)``, so the
crystal line is the same transform with the density weighted by ``1 − b_z²``
and the non-precessing fraction is ``(a² + c² r)/(1 + r)``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.special import j0, j1, struve

#: Target absolute accuracy of the transform (in units of the line amplitude).
_EPS = 1e-12

#: Below this ratio the exact r = 0 transform, J₀ and the Struve function H₀,
#: is closer to the exact line than the quadrature can resolve the lower
#: cut-off: the two differ by ≈ 0.4 r² ω_max t ln(1/(r ω_max t)), under 1e-8 for
#: r = 1e-6 out to ω_max t ≈ 2e4.
_RATIO_ZERO = 1e-6

#: Time points per direct-quadrature block; each block sizes its node count to
#: the largest Δω t it holds.
_BLOCK = 512


def _log_rho(ratio: float, *, weighted: bool) -> float:
    """Log of the Bernstein-ellipse parameter of the expanded weight on [B_min, B_max].

    The nearest singularity of h sits at B = −B_min; the crystal weight h·λ adds
    the pole of 1/B² at B = 0, which is nearer. In the Chebyshev variable those
    are x = −(1 + 3r)/(1 − r) and x = −(1 + r)/(1 − r).
    """
    root = np.sqrt(ratio)
    if weighted:
        return float(2.0 * np.arctanh(root))
    return float(np.arccosh((1.0 + 3.0 * ratio) / (1.0 - ratio)))


def _node_weights(ratio: float, n: int) -> tuple[NDArray[np.float64], ...]:
    """Chebyshev nodes as ``B/B_max``, with h and h·λ at each node."""
    x = np.cos((np.arange(n) + 0.5) * (np.pi / n))
    s = 0.5 * (1.0 + ratio) + 0.5 * (1.0 - ratio) * x
    h = 2.0 * s / np.sqrt((s + ratio) * (s + 1.0))
    # λ = (s² − r²)/(s²(1 − r²)) written without the cancellation near r = 1.
    lam = (1.0 + x) * (s + ratio) / (2.0 * (1.0 + ratio) * s * s)
    return s, h, h * lam


@lru_cache(maxsize=256)
def _series_coefficients(
    ratio: float, weighted: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Chebyshev coefficients of h and of h·λ, truncated for :data:`_EPS`.

    The coefficient tail falls below the tolerance within ``(1 + 0.3 r)`` times
    the geometric bound ``ln(1/ε)/ln ρ`` (calibrated for 1e-5 ≤ r ≤ 0.95).
    """
    bound = np.log(1.0 / _EPS) / _log_rho(ratio, weighted=weighted)
    terms = int(np.ceil((1.0 + 0.3 * ratio) * bound)) + 4
    k = 4 * terms + 32
    _, h, h_lam = _node_weights(ratio, k)
    coeffs = dct(np.stack([h, h_lam]), type=2, axis=1) / k
    coeffs[:, 0] /= 2.0
    return coeffs[0, :terms].copy(), coeffs[1, :terms].copy()


@lru_cache(maxsize=256)
def _quadrature(ratio: float, n: int) -> tuple[NDArray[np.float64], ...]:
    s, h, h_lam = _node_weights(ratio, n)
    return s, h / n, h_lam / n


def _direct_nodes(z: float, ratio: float, weighted: bool) -> int:
    """Nodes resolving both the oscillation (Δω t = z) and the weight's structure.

    The midpoint rule is exact to degree 2N − 1, so the oscillation needs about
    z/2 nodes and the weight a fraction of its geometric bound that grows
    slowly with r (calibrated against converged references for 1e-5 ≤ r ≤ 0.95).
    """
    oscillation = 1.1 * (0.5 * z + 4.5 * max(z, 1.0) ** (1.0 / 3.0) + 2.0)
    fraction = max(0.34, 0.34 + 0.045 * np.log10(1e5 * ratio))
    structure = fraction * np.log(1.0 / _EPS) / _log_rho(ratio, weighted=weighted) + 12.0
    return int(np.ceil(max(oscillation, structure)))


def _series(
    t: NDArray[np.float64],
    omega_max: float,
    ratio: float,
    phase: float,
    coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Σ c_n J_n(Δω t) cos(ω_av t + φ + nπ/2), with J_n by upward recurrence."""
    z = 0.5 * omega_max * (1.0 - ratio) * t
    two_over_z = 2.0 / z
    j_prev = j0(z)
    j_curr = j1(z)
    real = coeffs[0] * j_prev
    imag = coeffs[1] * j_curr
    for n in range(1, coeffs.size - 1):
        j_prev, j_curr = j_curr, (n * two_over_z) * j_curr - j_prev
        k = n + 1
        quarter = k & 3
        if quarter == 0:
            real += coeffs[k] * j_curr
        elif quarter == 1:
            imag += coeffs[k] * j_curr
        elif quarter == 2:
            real -= coeffs[k] * j_curr
        else:
            imag -= coeffs[k] * j_curr
    carrier = 0.5 * omega_max * (1.0 + ratio) * t + phase
    return np.cos(carrier) * real - np.sin(carrier) * imag


def _line(
    t: NDArray,
    frequency: float,
    ratio: float,
    phase: float,
    base: float,
    slope: float,
) -> NDArray[np.float64]:
    """``∫ D(B) [base + slope·λ(B)] cos(γ_μ B t + φ) dB`` for ``B_max = frequency/γ_μ``."""
    times = np.asarray(t, dtype=float)
    omega_max = 2.0 * np.pi * float(frequency)
    # Both closed-form limits keep the exact weight ∫ D (base + slope·λ) dB, with
    # ⟨λ⟩ = 1/(1 + r), so the line and the non-precessing fraction still sum to 1.
    weight = base + slope / (1.0 + ratio)
    if ratio == 1.0:
        # Every muon sees B_max.
        return weight * np.cos(omega_max * times + phase)
    if ratio < _RATIO_ZERO:
        # The shape is the Overhauser density's on [0, B_max].
        arg = omega_max * times
        return weight * (np.cos(phase) * j0(arg) - np.sin(phase) * struve(0, arg))

    weighted = slope != 0.0
    coeffs_h, coeffs_lam = _series_coefficients(ratio, weighted)
    coeffs = base * coeffs_h + slope * coeffs_lam
    flat = times.ravel()
    order = np.argsort(np.abs(flat), kind="stable")
    ordered = flat[order]
    z = 0.5 * omega_max * (1.0 - ratio) * np.abs(ordered)
    # The recurrence is stable once z exceeds the series length; below twice
    # that length the quadrature is also the cheaper of the two.
    split = int(np.searchsorted(z, 2.0 * coeffs.size + 8.0))
    result = np.empty_like(ordered)
    for start in range(0, split, _BLOCK):
        stop = min(start + _BLOCK, split)
        nodes, weight_h, weight_lam = _quadrature(
            ratio, _direct_nodes(z[stop - 1], ratio, weighted)
        )
        cosines = np.cos(np.multiply.outer(ordered[start:stop], omega_max * nodes) + phase)
        result[start:stop] = cosines @ (base * weight_h + slope * weight_lam)
    if split < ordered.size:
        result[split:] = _series(ordered[split:], omega_max, ratio, phase, coeffs)
    out = np.empty_like(result)
    out[order] = result
    return out.reshape(times.shape)


def _canonical(frequency: float, ratio: float) -> tuple[float, float, bool]:
    """Map any ratio onto ``0 ≤ r ≤ 1`` and the upper cut-off.

    The ellipse's semi-axes are ``frequency`` and ``|ratio|·frequency``; a ratio
    above 1 names the *lower* cut-off by ``frequency``, which is the same ellipse
    with the axes relabelled. Returns ``(f_max, r, swapped)``.
    """
    magnitude = abs(float(ratio))
    if magnitude > 1.0:
        return float(frequency) * magnitude, 1.0 / magnitude, True
    return float(frequency), magnitude, False


def helical_line(t: NDArray, frequency: float, ratio: float, phase: float) -> NDArray[np.float64]:
    """Precessing polarization of the helical field distribution, ``∫ D(B) cos(γ_μ B t + φ) dB``.

    ``frequency`` is the upper cut-off ``γ_μ B_max/2π`` in MHz, ``ratio`` is
    ``B_min/B_max`` and ``phase`` is in radians. Equals 1 at ``t = 0`` for zero
    phase; reduces to ``J₀(2π f t)`` at ``ratio = 0`` and ``cos(2π f t + φ)`` at
    ``ratio = 1``.
    """
    f_max, r, _ = _canonical(frequency, ratio)
    return _line(t, f_max, r, float(phase), 1.0, 0.0)


def helical_crystal_line(
    t: NDArray,
    frequency: float,
    ratio: float,
    phase: float,
    theta_h: float,
    phi_h: float,
) -> tuple[float, NDArray[np.float64]]:
    """Non-precessing fraction and precessing polarization for a single crystal.

    ``theta_h`` is the angle (degrees) between the initial muon polarization and
    the normal to the plane the local field rotates in; ``phi_h`` is the angle
    (degrees), within that plane, between the polarization's projection and the
    ``B_max`` axis. Returns ``(W₀, line)`` with ``W₀ = (a² + c² r)/(1 + r)`` the
    fraction along the local field and ``line`` its precessing counterpart, so
    ``W₀ + line(0) = 1`` at zero phase.
    """
    f_max, r, swapped = _canonical(frequency, ratio)
    sin_theta = np.sin(np.radians(theta_h))
    along_max = (sin_theta * np.cos(np.radians(phi_h))) ** 2
    along_min = (sin_theta * np.sin(np.radians(phi_h))) ** 2
    if swapped:
        along_max, along_min = along_min, along_max
    non_precessing = (along_max + along_min * r) / (1.0 + r)
    return float(non_precessing), _line(
        t, f_max, r, float(phase), 1.0 - along_min, along_min - along_max
    )
