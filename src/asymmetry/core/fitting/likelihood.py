"""Validated observations and likelihood objectives shared by all fit drivers.

Poisson deviance follows Baker & Cousins, Nucl. Instrum. Methods Phys. Res.
221, 437–442 (1984), doi:10.1016/0167-5087(84)90016-4. Gaussian errors are
independent stated standard deviations; their data-only normalization cancels.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Objective(str, Enum):
    GAUSSIAN = "gaussian"
    POISSON = "poisson"


class CountSource(str, Enum):
    RAW = "raw"
    DEADTIME_CORRECTED = "deadtime-corrected"
    LIFETIME_CORRECTED = "lifetime-corrected"


class FitInputError(ValueError):
    """An observation cannot be used under the requested likelihood."""


def _observation_arrays(
    *values: ArrayLike, t_min=None, t_max=None
) -> tuple[NDArray[np.float64], ...]:
    arrays = tuple(np.array(value, dtype=float, copy=True) for value in values)
    if any(array.ndim != 1 for array in arrays):
        raise FitInputError("fit observations must be one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise FitInputError("fit observation arrays must have the same length")
    if arrays[0].size == 0:
        raise FitInputError("fit observations contain no points")
    if t_min is not None or t_max is not None:
        mask = np.ones(arrays[0].shape, dtype=bool)
        if t_min is not None:
            mask &= arrays[0] >= t_min
        if t_max is not None:
            mask &= arrays[0] <= t_max
        arrays = tuple(array[mask] for array in arrays)
        if arrays[0].size == 0:
            raise FitInputError("fit observations contain no points in the fit window")
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _point_label(run_number, point, time):
    run = "unknown" if run_number is None else str(run_number)
    return f"run {run}, point {point} (t = {time:.3f} µs)"


@dataclass(frozen=True)
class GaussianObservations:
    """Owned read-only observations with finite, strictly positive stated errors."""

    time: NDArray[np.float64]
    values: NDArray[np.float64]
    sigma: NDArray[np.float64]
    run_number: int | str | None = None
    t_min: InitVar[float | None] = None
    t_max: InitVar[float | None] = None

    def __post_init__(self, t_min, t_max):
        time, values, sigma = _observation_arrays(
            self.time, self.values, self.sigma, t_min=t_min, t_max=t_max
        )
        for name, array in (("time", time), ("value", values), ("uncertainty", sigma)):
            invalid = ~np.isfinite(array)
            if name == "uncertainty":
                invalid |= array <= 0
            if invalid.any():
                point = int(np.flatnonzero(invalid)[0])
                label = _point_label(self.run_number, point, time[point])
                if name == "uncertainty":
                    raise FitInputError(
                        f"{label}: uncertainty {array[point]:g} is not finite and positive; "
                        "Gaussian fits need finite positive errors"
                    )
                raise FitInputError(f"{label}: {name} {array[point]:g} is not finite")
        for name, array in (("time", time), ("values", values), ("sigma", sigma)):
            object.__setattr__(self, name, array)

    def __reduce__(self):
        return type(self), (self.time, self.values, self.sigma, self.run_number)


@dataclass(frozen=True)
class PoissonCounts:
    """Owned read-only integer counts from an uncorrected acquisition."""

    time: NDArray[np.float64]
    counts: NDArray[np.float64]
    source: CountSource
    run_number: int | str | None = None
    t_min: InitVar[float | None] = None
    t_max: InitVar[float | None] = None

    def __post_init__(self, t_min, t_max):
        source = CountSource(self.source)
        if source is not CountSource.RAW:
            raise FitInputError(
                f"Poisson likelihood needs raw counts; these counts were {source.value}. "
                "Use the DT₀ count-domain model or the Gaussian cost."
            )
        time, counts = _observation_arrays(self.time, self.counts, t_min=t_min, t_max=t_max)
        invalid = (
            ~np.isfinite(time) | ~np.isfinite(counts) | (counts < 0) | (counts != np.floor(counts))
        )
        if invalid.any():
            point = int(np.flatnonzero(invalid)[0])
            label = _point_label(self.run_number, point, time[point])
            raise FitInputError(
                f"{label}: Poisson fits need finite times and finite non-negative integer counts; "
                f"count {counts[point]:g}"
            )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "source", source)

    def __reduce__(self):
        return type(self), (self.time, self.counts, self.source, self.run_number)

    def invalid_model_reason(self, model: NDArray[np.float64]) -> str | None:
        invalid = ~np.isfinite(model) | (model < 0) | ((self.counts > 0) & (model == 0))
        points = np.flatnonzero(invalid)
        if points.size == 0:
            return None
        label = _point_label(self.run_number, int(points[0]), self.time[points[0]])
        return f"{label}: Poisson optimum has invalid expected counts in bins {points.tolist()}"


Observations = GaussianObservations | PoissonCounts


def gaussian_chi2(observed, model, errors) -> float:
    """Σ((n − μ)/σ)² for validated observations with stated independent errors."""
    return float(np.sum(((observed - model) / errors) ** 2))


# The continuation joins far below the smallest positive observation. Its slope
# is negative and curvature positive, so it cannot introduce an exterior minimum.
_POISSON_CONTINUATION_FRACTION = 1e-6


def poisson_deviance(observed: NDArray[np.float64], model: NDArray[np.float64]) -> float:
    """Poisson deviance with a differentiable restoring penalty for trial μ ≤ 0."""
    if not np.all(np.isfinite(model)):
        return float("inf")
    positive = observed > 0
    n, mu = observed[positive], model[positive]
    join = _POISSON_CONTINUATION_FRACTION * n
    safe = np.maximum(mu, join)
    terms = safe - n + n * np.log(n / safe)
    delta = np.minimum(mu - join, 0)
    terms += (1 - n / join) * delta + n * (delta / join) ** 2 / 2
    return 2 * float(np.sum(terms) + np.sum(np.abs(model[~positive])))
