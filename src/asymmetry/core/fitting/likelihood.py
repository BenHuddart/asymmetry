"""Validated observations and likelihood objectives shared by all fit drivers.

Poisson deviance follows Baker & Cousins, Nucl. Instrum. Methods Phys. Res.
221, 437–442 (1984), doi:10.1016/0167-5087(84)90016-4. Gaussian errors are
independent stated standard deviations whose data-only normalisation cancels.
Spectral least squares records the applicable weights and their interpretation.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from asymmetry.core.data.ratio_validity import ObservationReport, RatioReason
from asymmetry.core.data.time_window import time_window_mask
from asymmetry.core.fitting.spectral_weights import (
    SpectralWeighting,
    SpectralWeightPolicy,
    spectral_fit_weights,
)


class Objective(str, Enum):
    GAUSSIAN = "gaussian"
    POISSON = "poisson"


class CountSource(str, Enum):
    RAW = "raw"
    RETAINED_VARIANCE = "retained-variance arithmetic or supplied noise"
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
        mask = time_window_mask(arrays[0], t_min, t_max)
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
    """Owned read-only least-squares observations with positive denominators."""

    time: NDArray[np.float64]
    values: NDArray[np.float64]
    sigma: NDArray[np.float64]
    run_number: int | str | None = None
    t_min: InitVar[float | None] = None
    t_max: InitVar[float | None] = None
    report: ObservationReport | None = None
    spectral_weighting: SpectralWeighting | None = None

    @property
    def stated_sigma(self) -> NDArray[np.float64] | None:
        """The stated bin errors, when the objective used them."""
        return None if self.uses_unit_weights else self.sigma

    @property
    def uses_unit_weights(self) -> bool:
        return (
            self.spectral_weighting is not None
            and self.spectral_weighting.policy is SpectralWeightPolicy.UNIT
        )

    def insufficient_bins_message(self, n_free: int) -> str | None:
        if self.spectral_weighting is not None:
            return self.spectral_weighting.insufficient_bins_message(len(self.time), n_free)
        if self.report is not None and len(self.time) <= n_free:
            return (
                "Insufficient usable Gaussian observations for free parameters. "
                + self.report.summary()
            )
        return None

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
        return type(self), (
            self.time,
            self.values,
            self.sigma,
            self.run_number,
            None,
            None,
            self.report,
            self.spectral_weighting,
        )


@dataclass(frozen=True)
class GaussianSelection:
    """Requested full-grid membership and its optional histogram omission report."""

    mask: NDArray[np.bool_]
    report: ObservationReport | None

    @property
    def used(self) -> int:
        return int(np.count_nonzero(self.mask))


def gaussian_selection(dataset, t_min=None, t_max=None) -> GaussianSelection:
    """Select a requested fit window and known usable histogram observations."""
    mask = dataset.window_mask(t_min, t_max)
    report = None
    if dataset.ratio_validity is not None:
        validity = dataset.ratio_validity.select(mask)
        report = validity.report()
        mask[mask] = validity.reasons == RatioReason.USABLE
    return GaussianSelection(mask, report)


def dataset_observations(dataset, t_min=None, t_max=None):
    """Resolve spectral weights or known histogram support before validation."""
    if dataset.metadata.get("plot_domain") == "frequency":
        mask = gaussian_selection(dataset, t_min, t_max).mask
        time, values = dataset.time[mask], dataset.asymmetry[mask]
        error = np.asarray(dataset.error, dtype=float)
        if dataset.error is None or error.size == 0:
            error = np.zeros_like(dataset.time)
        if error.shape != dataset.time.shape:
            raise FitInputError("spectral errors must be absent or align with the frequency bins")
        error = error[mask]
        weighting, used = spectral_fit_weights(error)
        return GaussianObservations(
            time[used],
            values[used],
            np.ones_like(time) if weighting.policy is SpectralWeightPolicy.UNIT else error[used],
            run_number=dataset.run_number,
            spectral_weighting=weighting,
        )
    if dataset.ratio_validity is None:
        return GaussianObservations(
            dataset.time,
            dataset.asymmetry,
            dataset.error,
            run_number=dataset.run_number,
            t_min=t_min,
            t_max=t_max,
        )
    selection = gaussian_selection(dataset, t_min, t_max)
    if selection.report is not None and not selection.used:
        raise FitInputError(
            f"run {dataset.run_number}: no usable Gaussian observations. "
            + selection.report.summary()
        )
    mask = selection.mask
    return GaussianObservations(
        dataset.time[mask],
        dataset.asymmetry[mask],
        dataset.error[mask],
        run_number=dataset.run_number,
        report=selection.report,
    )


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
        points = np.flatnonzero(poisson_invalid_expected_counts(self.counts, model))
        if points.size == 0:
            return None
        label = _point_label(self.run_number, int(points[0]), self.time[points[0]])
        return f"{label}: Poisson optimum has invalid expected counts in bins {points.tolist()}"


Observations = GaussianObservations | PoissonCounts


def gaussian_chi2(observed, model, errors) -> float:
    """Σ((n − μ)/σ)² for validated observations with stated independent errors."""
    return float(np.sum(((observed - model) / errors) ** 2))


def poisson_invalid_expected_counts(observed, model) -> NDArray[np.bool_]:
    """Only an empty observed bin can have zero expected counts."""
    return ~np.isfinite(model) | (model < 0) | ((observed > 0) & (model == 0))


_LOG_MINIMUM_EXPECTATION = np.log(np.finfo(float).smallest_subnormal)
_INVALID_EXPECTATION_SLOPE = 2e6
_DEVIANCE_SERIES = tuple((-1) ** power / power for power in range(2, 10))


def poisson_deviance(
    observed: NDArray[np.float64], model: NDArray[np.float64], *, restore_invalid: bool = False
) -> float:
    """Baker–Cousins deviance, with optional restoration for finite invalid trials."""
    invalid = poisson_invalid_expected_counts(observed, model)
    positive = observed > 0
    penalty = 0.0
    if invalid.any():
        invalid_model = model[invalid]
        if not restore_invalid or not np.all(np.isfinite(invalid_model)):
            return float("inf")
        occupied = observed[invalid & positive]
        # An occupied bin at zero costs at least its deviance at the smallest
        # positive mean. A steep exterior slope restores finite negative means.
        penalty = np.sum(occupied * (np.log(occupied) - _LOG_MINIMUM_EXPECTATION))
        penalty -= _INVALID_EXPECTATION_SLOPE * np.sum(invalid_model)

    used = positive & ~invalid
    n, mu = observed[used], np.asarray(model[used], dtype=float)
    delta = mu - n
    local = np.abs(delta) <= n / 2
    far = ~local
    n_far = n[far]
    # Separate logarithms avoid overflow in n/μ. Apply the count scale after
    # combining the normalized terms so n·log(n/μ) need not fit on its own.
    delta[far] = n_far * (delta[far] / n_far + (np.log(n_far) - np.log(mu[far])))
    relative = delta[local] / n[local]
    terms = relative - np.log1p(relative)
    # (μ−n)/n − log1p((μ−n)/n) starts at second order near μ=n.
    small = np.abs(relative) < 0.01
    x = relative[small]
    terms[small] = x**2 * np.polynomial.polynomial.polyval(x, _DEVIANCE_SERIES)
    delta[local] = n[local] * terms
    return 2 * float(np.sum(delta) + np.sum(model[~positive & ~invalid]) + penalty)
