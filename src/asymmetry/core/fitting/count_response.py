"""Absolute causal count population for fixed instantaneous source births."""

import numpy as np

from asymmetry.core.fitting.time_response import live_source_ages
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def pulse_population(fraction_fn, time, separation_us, sign, physics):
    """Sum surviving incident populations; background and N0 remain outside.

    Unlike a normalised polarisation this remains zero before the first birth
    and underflows physically at late times, without rescaling count N0.
    """
    time = np.asarray(time, dtype=float)
    population = np.zeros_like(time)
    for live, age, _birth, incident in live_source_ages(time, separation_us):
        population[live] += (
            incident
            * np.exp(-age / MUON_LIFETIME_US)
            * (1.0 + sign * np.asarray(fraction_fn(age, **physics), dtype=float))
        )
    return population


def double_pulse_single_model(fraction_fn):
    """Raw single-bank prediction with fixed orientation and one background."""

    def model(t, *, N0, background, amplitude, relative_phase, dpsep, **physics):  # noqa: N803
        return float(N0) * pulse_population(fraction_fn, t, dpsep, amplitude, physics) + float(
            background
        )

    return model


def double_pulse_fb_model(fraction_fn, *, with_beta=False):
    """Raw F/B prediction; detector beta multiplies only backward polarisation."""

    def model(t, *, alpha, N0, background, sign, dpsep, **physics):  # noqa: N803
        beta = float(physics.pop("beta", 1.0)) if with_beta else 1.0
        root_alpha = np.sqrt(abs(float(alpha)))
        scale = root_alpha if sign >= 0.0 else 1.0 / root_alpha
        orientation = sign if sign >= 0.0 else sign * beta
        return float(N0) * scale * pulse_population(
            fraction_fn, t, dpsep, orientation, physics
        ) + float(background)

    return model
