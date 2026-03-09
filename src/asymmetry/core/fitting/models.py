"""Built-in μSR fit functions.

Each model is a callable ``f(t, **params) -> array`` plus metadata describing
its parameters.  Models are collected in the :data:`MODELS` registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ModelDefinition:
    """Descriptor for a built-in fit function."""

    name: str
    description: str
    function: Callable[..., NDArray[np.float64]]
    param_names: list[str]
    param_defaults: dict[str, float]


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def exponential_relaxation(
    t: NDArray, A0: float, Lambda: float, baseline: float = 0.0
) -> NDArray:
    """Simple exponential: A(t) = A0 exp(−Λt) + baseline."""
    # Clamp exponent to prevent overflow; exp(-700) ≈ 0 numerically
    exponent = np.clip(-Lambda * np.abs(t), -700, 0)
    return A0 * np.exp(exponent) + baseline


def gaussian_relaxation(
    t: NDArray, A0: float, sigma: float, baseline: float = 0.0
) -> NDArray:
    """Gaussian relaxation: A(t) = A0 exp(−σ²t²) + baseline."""
    # Clamp exponent to prevent overflow
    exponent = np.clip(-(sigma * t) ** 2, -700, 0)
    return A0 * np.exp(exponent) + baseline


def oscillatory(
    t: NDArray, A0: float, frequency: float, phase: float = 0.0, Lambda: float = 0.0,
    baseline: float = 0.0,
) -> NDArray:
    """Damped oscillation: A0 cos(2πft + φ) exp(−Λt) + baseline."""
    # Clamp damping exponent to prevent overflow
    exponent = np.clip(-Lambda * np.abs(t), -700, 0)
    return A0 * np.cos(2.0 * np.pi * frequency * t + phase) * np.exp(exponent) + baseline


def stretched_exponential(
    t: NDArray, A0: float, Lambda: float, beta: float = 1.0, baseline: float = 0.0,
) -> NDArray:
    """Stretched exponential: A0 exp(−(Λt)^β) + baseline."""
    # Clamp exponent to prevent overflow
    exponent = np.clip(-np.abs(Lambda * t) ** beta, -700, 0)
    return A0 * np.exp(exponent) + baseline


def static_gkt_zf(
    t: NDArray, A0: float, Delta: float, baseline: float = 0.0,
) -> NDArray:
    """Static Gaussian Kubo-Toyabe (zero field).

    GKT(t) = A0 [1/3 + 2/3 (1 − Δ²t²) exp(−Δ²t²/2)] + baseline
    """
    dt2 = (Delta * t) ** 2
    # Clamp exponent to prevent overflow
    exponent = np.clip(-dt2 / 2.0, -700, 0)
    return A0 * (1.0 / 3.0 + 2.0 / 3.0 * (1.0 - dt2) * np.exp(exponent)) + baseline


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS: dict[str, ModelDefinition] = {}


def _register(name: str, desc: str, fn: Callable, params: list[str], defaults: dict) -> None:
    MODELS[name] = ModelDefinition(name, desc, fn, params, defaults)


_register(
    "ExponentialRelaxation",
    "A0 exp(−Λt) + baseline",
    exponential_relaxation,
    ["A0", "Lambda", "baseline"],
    {"A0": 25.0, "Lambda": 0.5, "baseline": 0.0},
)

_register(
    "GaussianRelaxation",
    "A0 exp(−σ²t²) + baseline",
    gaussian_relaxation,
    ["A0", "sigma", "baseline"],
    {"A0": 25.0, "sigma": 0.5, "baseline": 0.0},
)

_register(
    "Oscillatory",
    "A0 cos(2πft + φ) exp(−Λt) + baseline",
    oscillatory,
    ["A0", "frequency", "phase", "Lambda", "baseline"],
    {"A0": 25.0, "frequency": 1.0, "phase": 0.0, "Lambda": 0.0, "baseline": 0.0},
)

_register(
    "StretchedExponential",
    "A0 exp(−(Λt)^β) + baseline",
    stretched_exponential,
    ["A0", "Lambda", "beta", "baseline"],
    {"A0": 25.0, "Lambda": 0.5, "beta": 1.0, "baseline": 0.0},
)

_register(
    "StaticGKT_ZF",
    "Static Gaussian Kubo-Toyabe (zero field)",
    static_gkt_zf,
    ["A0", "Delta", "baseline"],
    {"A0": 25.0, "Delta": 0.5, "baseline": 0.0},
)
