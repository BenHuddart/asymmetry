"""Effective least-squares weighting for a selected spectral window."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from asymmetry.core.fitting.engine import _CoupledGlobalProblem


class SpectralWeightPolicy(StrEnum):
    UNIT = "unit"
    DIAGONAL = "diagonal"
    POSITIVE_ONLY = "positive_only"
    UNKNOWN = "unknown"


class SpectralWeightRecord(TypedDict):
    policy: str
    omitted_bin_ranges: list[list[int]]


@dataclass(frozen=True)
class SpectralWeighting:
    """Which bins in the requested window contributed to the fit objective."""

    policy: SpectralWeightPolicy
    # Half-open positions in the window's sampled frequency grid, before omissions.
    omitted_bin_ranges: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> SpectralWeightRecord:
        return {
            "policy": self.policy.value,
            "omitted_bin_ranges": [list(span) for span in self.omitted_bin_ranges],
        }

    def insufficient_bins_message(self, n_used: int, n_free: int) -> str | None:
        if self.policy is SpectralWeightPolicy.POSITIVE_ONLY and n_used < n_free:
            return (
                f"Only {n_used} of the selected spectral bins have positive errors for {n_free} "
                "free parameters. Widen the fit range or fix parameters."
            )
        return None


def spectral_weight_policy(attribution: Mapping | None) -> SpectralWeightPolicy | None:
    weighting = (attribution or {}).get("spectral_weighting")
    return SpectralWeightPolicy(weighting["policy"]) if weighting is not None else None


def spectral_weighting_detail(attribution: Mapping | None) -> str | None:
    """Describe the actual selection and weighting for a result card or report."""
    policy = spectral_weight_policy(attribution)
    if policy is None:
        return None
    if policy is SpectralWeightPolicy.UNKNOWN:
        return (
            "Spectral weighting unknown. Any stored parameter errors and statistics are historical."
        )
    used = int(attribution["n"])
    window = f"{attribution['min']:.6g} to {attribution['max']:.6g} MHz"
    if policy is SpectralWeightPolicy.UNIT:
        return (
            f"Unit weights on {used} spectral bins ({window}). "
            "Parameter uncertainties and a χ² verdict are unavailable."
        )
    ranges = attribution["spectral_weighting"]["omitted_bin_ranges"]
    omitted = sum(stop - start for start, stop in ranges)
    if omitted:
        return (
            f"Diagonal spectral errors on {used} of {used + omitted} bins ({window}). "
            f"Omitted {omitted} bins without positive errors."
        )
    return f"Diagonal spectral errors on all {used} fitted bins ({window})."


def spectral_quality_reason(attribution: Mapping | None) -> str | None:
    policy = spectral_weight_policy(attribution)
    if (attribution or {}).get("coupled_unit_weights"):
        return (
            "Unit weights in a linked member leave parameter uncertainties and a χ² verdict "
            "unavailable for the coupled parameters."
        )
    if policy is SpectralWeightPolicy.UNIT:
        return "Unit weights provide a residual sum of squares without a χ² verdict."
    if policy is SpectralWeightPolicy.UNKNOWN:
        return "Spectral weighting unknown. A χ² verdict is unavailable."
    if policy is not None:
        return "Correlated spectral bins preclude a calibrated χ² band or verdict."
    return None


def spectral_fit_weights(error: np.ndarray) -> tuple[SpectralWeighting, np.ndarray]:
    """Choose unit weights, all stated errors, or only the positive stated errors."""
    positive = np.isfinite(error) & (error > 0)
    if not positive.any():
        return SpectralWeighting(SpectralWeightPolicy.UNIT), np.ones_like(positive)
    if positive.all():
        return SpectralWeighting(SpectralWeightPolicy.DIAGONAL), positive
    edges = np.flatnonzero(np.diff(np.r_[True, positive, True]))
    omitted = tuple((int(start), int(stop)) for start, stop in edges.reshape(-1, 2))
    return SpectralWeighting(SpectralWeightPolicy.POSITIVE_ONLY, omitted), positive


def _connected_fit_members(
    columns_by_run: Mapping[int, tuple[int, ...]], seeds: set[int]
) -> tuple[set[int], set[int]]:
    """Find the free columns and members in components containing any seed member."""
    runs_by_column: dict[int, set[int]] = defaultdict(set)
    for run, member_columns in columns_by_run.items():
        for column in member_columns:
            runs_by_column[column].add(run)
    pending = list(seeds)
    runs = seeds.copy()
    columns: set[int] = set()
    while pending:
        run = pending.pop()
        for column in set(columns_by_run[run]) - columns:
            columns.add(column)
            newly_connected = runs_by_column[column] - runs
            pending.extend(newly_connected)
            runs.update(newly_connected)
    return columns, runs


def coupled_spectral_scope(problem: _CoupledGlobalProblem) -> tuple[set[int], set[int]]:
    """Find the parameter columns and members connected to unit-weight data."""
    from asymmetry.core.fitting.likelihood import GaussianObservations

    runs = {
        dataset.run_number
        for dataset in problem.fitted_datasets
        if isinstance(record := problem.observations[dataset.run_number], GaussianObservations)
        and record.uses_unit_weights
    }
    if not runs:
        return set(), set()
    columns_by_run = {
        dataset.run_number: columns
        for dataset, columns in zip(problem.fitted_datasets, problem.dataset_columns, strict=True)
    }
    return _connected_fit_members(columns_by_run, runs)


def check_coupled_spectral_support(
    datasets: Sequence,
    observations: Mapping,
    columns: Sequence[tuple[int, ...]],
) -> None:
    """Check structural identifiability of the columns affected by partial spectra."""
    from asymmetry.core.fitting.likelihood import FitInputError, GaussianObservations

    partial = {
        dataset.run_number
        for dataset in datasets
        if isinstance(record := observations[dataset.run_number], GaussianObservations)
        and record.spectral_weighting is not None
        and record.spectral_weighting.policy is SpectralWeightPolicy.POSITIVE_ONLY
    }
    if not partial:
        return

    from scipy.sparse.csgraph import structural_rank

    from asymmetry.core.fitting.global_covariance import member_jacobian_sparsity

    columns_by_run = {
        dataset.run_number: member for dataset, member in zip(datasets, columns, strict=True)
    }
    connected_columns, connected_runs = _connected_fit_members(columns_by_run, partial)
    if not connected_columns:
        return
    positions = {column: index for index, column in enumerate(sorted(connected_columns))}
    runs = sorted(connected_runs)
    member_columns = [tuple(positions[column] for column in columns_by_run[run]) for run in runs]
    # Extra identical row patterns cannot match more columns than they touch.
    points = [
        min(len(observations[run].time), len(member))
        for run, member in zip(runs, member_columns, strict=True)
    ]
    supported = structural_rank(member_jacobian_sparsity(points, member_columns, len(positions)))
    if supported < len(positions):
        raise FitInputError(
            f"Retained spectral bins support {supported} of {len(positions)} free parameters "
            "in the coupled fit. Widen the fit range or fix parameters."
        )


def spectral_oversampling_correction(
    *,
    chi_squared: float,
    ndata: int,
    nfree: int,
    uncertainties: dict[str, float],
    covariance: NDArray[np.float64] | None,
    minos_errors: dict[str, tuple[float, float]] | None,
    error_oversampling: float,
) -> tuple[
    float,
    float,
    int,
    dict[str, float],
    NDArray[np.float64] | None,
    dict[str, tuple[float, float]] | None,
    str | None,
]:
    """Apply heuristic scalar padding rescaling without reconstructing covariance."""
    from asymmetry.core.fitting.fit_quality import reduced_statistic

    s = float(error_oversampling)
    sqrt_s = math.sqrt(s)
    ndata_eff = max(int(round(ndata / s)), 1)
    corrected_dof = ndata_eff - nfree
    corrected_chi2 = chi_squared / s
    corrected_reduced_chi2 = reduced_statistic(corrected_chi2, corrected_dof)
    scaled_uncertainties = {name: value * sqrt_s for name, value in uncertainties.items()}
    scaled_covariance = None if covariance is None else np.asarray(covariance, dtype=float) * s
    scaled_minos = None
    if minos_errors:
        scaled_minos = {
            name: (lower * sqrt_s, upper * sqrt_s) for name, (lower, upper) in minos_errors.items()
        }
    message = (
        f"Heuristic zero-padding rescaling: errors ×{sqrt_s:.3g}, χ² ÷{s:g}; "
        "no covariance reconstruction."
    )
    return (
        corrected_chi2,
        corrected_reduced_chi2,
        corrected_dof,
        scaled_uncertainties,
        scaled_covariance,
        scaled_minos,
        message,
    )
