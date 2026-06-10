"""ISIS pulse-shape response for the MaxEnt forward model.

At a pulsed muon source the muons arrive spread over a finite pulse (≈ tens of
ns), so a precession signal oscillating fast compared with the pulse width is
averaged across the spread of arrival times and its amplitude is suppressed.
Above ~5 MHz this distorts the recovered spectrum unless the forward model
accounts for it (Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy*,
OUP 2022, §14.2).

The muon arrival-time distribution is the proton-pulse shape convolved with the
pion-decay exponential.  Its Fourier transform multiplies each frequency's
contribution to the modelled time signal.  For a normalised symmetric parabola
of half-width *w* the cosine transform is

    G(ω) = 3 [ sin(x)/x³ − cos(x)/x² ],   x = ω·w,   G(0) = 1,

and the pion lifetime τ_π adds a single-pole low-pass 1/(1 + i ω τ_π).  Writing
the combined response as ``P(ω) = P_cos(ω) − i·P_sin(ω)`` the forward kernel for
one frequency becomes

    P_cos(ω)·cos(2πνt + φ) + P_sin(ω)·sin(2πνt + φ),

i.e. a per-frequency amplitude ``R = √(P_cos² + P_sin²)`` and phase shift
``δ = atan2(P_sin, P_cos)`` applied to the bare cosine kernel — the form the
engine consumes.  A second proton pulse separated by *s* adds a cosine
interference term weighted by ``tanh(s/2τ_µ)`` for the muon-decay depletion of
the later pulse.

This module is the pure pulse mathematics; the engine folds ``(R, δ)`` into its
real cosine/sine forward and adjoint maps.  Frequencies are in MHz and times in
µs, so ``ω = 2πν`` is in rad/µs and ``ωt`` is dimensionless.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.utils.constants import MUON_LIFETIME_US, PION_LIFETIME_US

#: Pulse-mode tokens accepted by :func:`pulse_response`.
PULSE_MODES = ("ignore", "single", "double")

_DC_TOL = 1.0e-8


def pulse_response(
    frequencies_mhz: NDArray[np.float64],
    *,
    half_width_us: float,
    separation_us: float = 0.0,
    n_pulses: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(P_cos, P_sin)`` over *frequencies_mhz* for the pulse response.

    ``n_pulses == 0`` (or a non-positive half-width) means no pulse shaping:
    ``P_cos = 1``, ``P_sin = 0``.  ``n_pulses == 1`` is the single proton pulse;
    ``n_pulses >= 2`` adds the double-pulse interference at *separation_us*.

    At DC (ν = 0) the response is exactly ``(1, 0)``.  The single-pulse case is
    the ``separation_us → 0`` limit of the double-pulse formula.
    """
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64)
    if int(n_pulses) <= 0 or float(half_width_us) <= 0.0:
        return np.ones_like(frequencies), np.zeros_like(frequencies)

    omega = 2.0 * np.pi * frequencies  # rad/µs
    width = float(half_width_us)
    x = omega * width
    # Parabolic proton-pulse cosine transform, with the x→0 limit G(0)=1.
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = 3.0 * (np.sin(x) / x**3 - np.cos(x) / x**2)
    gw = np.where(np.abs(x) < _DC_TOL, 1.0, gw)

    tpion = PION_LIFETIME_US
    amplitude = gw / (1.0 + (omega * tpion) ** 2)

    separation = float(separation_us) if int(n_pulses) >= 2 else 0.0
    tmuon = MUON_LIFETIME_US
    tanh_weight = np.tanh(separation / (2.0 * tmuon)) if tmuon > 0.0 else 0.0
    cos_half = np.cos(omega * separation / 2.0)
    sin_half = np.sin(omega * separation / 2.0)

    p_cos = amplitude * (cos_half - tanh_weight * sin_half * omega * tpion)
    p_sin = amplitude * (tanh_weight * sin_half + cos_half * omega * tpion)
    return p_cos, p_sin


def pulse_amplitude_phase(
    frequencies_mhz: NDArray[np.float64],
    *,
    half_width_us: float,
    separation_us: float = 0.0,
    n_pulses: int = 1,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Return the per-frequency ``(amplitude R, phase shift δ)`` for the kernel.

    ``R = √(P_cos² + P_sin²)`` and ``δ = atan2(P_sin, P_cos)`` so that the
    pulse-shaped kernel is ``R(ν)·cos(2πνt + φ − δ(ν))``.  Returns ``(None,
    None)`` when there is no pulse shaping, which the engine treats as the
    identity (R = 1, δ = 0) on its fast path.
    """
    if int(n_pulses) <= 0 or float(half_width_us) <= 0.0:
        return None, None
    p_cos, p_sin = pulse_response(
        frequencies_mhz,
        half_width_us=half_width_us,
        separation_us=separation_us,
        n_pulses=n_pulses,
    )
    amplitude = np.hypot(p_cos, p_sin)
    phase = np.arctan2(p_sin, p_cos)
    return amplitude, phase
