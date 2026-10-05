"""Dense fitted curves for drawing: how many samples, and the curve itself."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import split_parameter_name
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


def fit_curve_sample_count(
    model: CompositeModel,
    param_values: dict[str, float],
    t_min: float,
    t_max: float,
    *,
    base_points: int = 500,
    points_per_cycle: int = 40,
    max_points: int = 20000,
) -> int:
    """Return a dense-enough sample count for plotting oscillatory models."""
    duration = max(float(t_max) - float(t_min), 0.0)
    if duration <= 0.0:
        return base_points

    max_frequency_mhz = 0.0
    for name, value in param_values.items():
        base_name, _index = split_parameter_name(name)
        try:
            numeric_value = abs(float(value))
        except (TypeError, ValueError):
            continue

        if base_name == "frequency":
            max_frequency_mhz = max(max_frequency_mhz, numeric_value)
        elif base_name == "field":
            field_frequency = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * numeric_value
            max_frequency_mhz = max(max_frequency_mhz, field_frequency)
        elif base_name in {"A_hf", "D_mu", "f_dip", "f_quad"}:
            # Hyperfine/dipolar couplings set the oscillation scale of the
            # muonium and spin-J components (lines up to ~A_hf in MHz).
            max_frequency_mhz = max(max_frequency_mhz, numeric_value)

    if max_frequency_mhz <= 0.0:
        return base_points

    cycles = max_frequency_mhz * duration
    required_points = int(np.ceil(cycles * points_per_cycle)) + 1
    return int(max(base_points, min(max_points, required_points)))


def dense_fit_curve(
    model: CompositeModel, param_values: dict[str, float], x_min: float, x_max: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(x, f(x))`` over ``[x_min, x_max]``, sampled densely enough to draw."""
    x = np.linspace(x_min, x_max, fit_curve_sample_count(model, param_values, x_min, x_max))
    return x, np.asarray(model.function(x, **param_values), dtype=float)
