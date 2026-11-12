"""Count-fit result extraction and immutable joint covariance ownership."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from asymmetry.core import computation
from asymmetry.core.fitting.engine import FitResult, _minuit_status_message
from asymmetry.core.fitting.likelihood import GaussianObservations, Objective, PoissonCounts
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info
from asymmetry.core.transform.corrections import CorrectionOutcome
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow


@dataclass(frozen=True)
class CountFitDomain:
    """Resolved bank selection on native timing, before any lifetime correction.

    Raw windows use the cropped alignment coordinate; detector offsets map them
    to acquisition bins. Each output time is the centre of its contributing bins.
    """

    group_id: int
    detector_indices: tuple[int, ...]
    time_us: tuple[float, ...]
    raw_windows: tuple[BinWindow, ...]
    alignment: AlignmentCrop
    bin_width_us: float
    deadtime: CorrectionOutcome
    background_preflight: tuple[tuple[str, CorrectionOutcome], ...]
    #: (flat offset, channel count, source, run, input ordinal, coefficient, data period, DAE period).
    acquired_sources: tuple[
        tuple[int, int, str, int | None, int | None, float, int | None, int | None], ...
    ] = ()


def count_fit_domain(group, dataset) -> CountFitDomain:
    """Retain the selection resolved by the existing grouped-count input owner."""
    metadata = group.metadata
    alignment = dict(metadata["grouped_alignment"])
    del alignment["timing_spread_us"]
    return CountFitDomain(
        int(group.group_id),
        group.detector_indices,
        tuple(group.time),
        tuple(BinWindow(*window) for window in metadata["count_fit_raw_windows"]),
        AlignmentCrop(**alignment),
        metadata["count_bin_width_us"],
        group.deadtime,
        tuple(
            (label, CorrectionOutcome.from_dict(outcome))
            for label, outcome in metadata["count_background_preflight"].items()
        ),
        tuple(tuple(item) for item in metadata["acquired_sources"]),
    )


@dataclass(frozen=True)
class CountFitParameter:
    """Immutable parameter specification/value; count scales are counts per bin.

    Free TF overrides inherit default search bounds at infinite endpoints.
    Fixed overrides retain explicit endpoints without numerical search defaults.
    Finite endpoints and fixed values must respect physical parameter domains;
    fixed values must also lie within explicitly supplied endpoints.
    """

    name: str
    value: float
    min: float = -float("inf")
    max: float = float("inf")
    fixed: bool = False

    @property
    def unit(self) -> str | None:
        count_units = {
            "N0": "counts/bin",
            "background": "counts/bin",
            "background_b": "counts/bin",
            "alpha": "1",
            "beta": "1",
            "stretch": "1",
            "tau": "µs",
            "t0": "µs",
            "dpsep": "µs",
            "DT0": "µs",
            "lambda_base": "µs⁻¹",
            "beta_base": "1",
        }
        return (
            count_units[self.name] if self.name in count_units else get_param_info(self.name).unit
        )


@dataclass(frozen=True)
class JointCountFit:
    """One complete joint solution, including every background and nuisance.

    Covariance order contains unique free parameters only. Uncertainty is
    conditional curvature at unit change in deviance, without residual scaling.
    """

    parameters: tuple[CountFitParameter, ...]
    covariance_parameters: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...] | None
    covariance_accurate: bool
    objective: Objective
    statistic: float
    dof: int
    success: bool
    message: str
    domains: tuple[CountFitDomain, ...] = ()
    minos_intervals: tuple[tuple[str, float, float], ...] = ()

    @property
    def uncertainty_status(self) -> str:
        if self.covariance is None:
            return "unavailable"
        return "available" if self.success and self.covariance_accurate else "unreliable"

    def standard_error(self, name: str) -> float | None:
        if self.uncertainty_status != "available" or name not in self.covariance_parameters:
            return None
        i = self.covariance_parameters.index(name)
        variance = self.covariance[i][i]
        return float(np.sqrt(variance)) if np.isfinite(variance) and variance > 0 else None

    def correlation(self, first: str, second: str) -> float | None:
        errors = (self.standard_error(first), self.standard_error(second))
        if any(error is None for error in errors):
            return None
        i, j = (self.covariance_parameters.index(name) for name in (first, second))
        return float(self.covariance[i][j] / (errors[0] * errors[1]))

    @property
    def statistic_per_dof(self) -> float | None:
        return self.statistic / self.dof if self.dof > 0 else None


def joint_count_fit(result: FitResult, domains: tuple[CountFitDomain, ...]) -> JointCountFit:
    """Freeze the full minimizer solution, never reconstruct it from bank slices."""
    return JointCountFit(
        parameters=tuple(
            CountFitParameter(p.name, p.value, p.min, p.max, p.fixed) for p in result.parameters
        ),
        covariance_parameters=tuple(result.covariance_parameters),
        covariance=tuple(tuple(float(v) for v in row) for row in result.covariance)
        if result.covariance is not None
        else None,
        covariance_accurate=result.covariance_accurate,
        objective=result.objective,
        statistic=result.chi_squared,
        dof=result.dof,
        success=result.success,
        message=result.message,
        domains=domains,
        minos_intervals=tuple(
            (name, *interval) for name, interval in (result.minos_errors or {}).items()
        ),
    )


def count_bank_view(result: FitResult, observations, model_values, *, omit: str, statistic: float):
    """Project a bank view from the complete joint solution for existing consumers."""
    names = [name for name in result.covariance_parameters if name != omit]
    indices = [result.covariance_parameters.index(name) for name in names]
    covariance = (
        result.covariance[np.ix_(indices, indices)] if result.covariance is not None else None
    )
    counts = (
        observations.values
        if isinstance(observations, GaussianObservations)
        else observations.counts
    )
    nfree = result.residuals.size - result.dof - (omit in result.covariance_parameters)
    dof = counts.size - nfree
    return replace(
        result,
        parameters=ParameterSet([p for p in result.parameters if p.name != omit]),
        uncertainties={n: e for n, e in result.uncertainties.items() if n != omit},
        covariance=covariance,
        covariance_parameters=names,
        chi_squared=statistic,
        reduced_chi_squared=statistic / max(dof, 1),
        dof=dof,
        residuals=counts - model_values,
        minos_errors={n: e for n, e in result.minos_errors.items() if n != omit}
        if result.minos_errors
        else None,
        warnings=list(result.warnings),
    )


def _result_from_minuit(
    m,
    params: ParameterSet,
    free_names: list[str],
    *,
    observations: GaussianObservations | PoissonCounts,
    model_values: NDArray[np.float64],
    chi_squared: float | None = None,
    followers: dict[str, str] | None = None,
    success_message: str = "Count fit successful",
    failure_prefix: str = "Count fit failed",
) -> FitResult:
    """Record the complete solution and equality-linked parameter values."""
    link_followers = followers or {}
    # MINOS stores signed interval offsets; the symmetric HESSE errors stay unchanged.
    merrors = getattr(m, "merrors", None)
    result_params = ParameterSet()
    uncertainties: dict[str, float] = {}
    minos_errors: dict[str, tuple[float, float]] = {}
    for p in params:
        source = link_followers.get(p.name, p.name)
        if source in free_names:
            idx = free_names.index(source)
            result_params.add(Parameter(name=p.name, value=m.values[idx], min=p.min, max=p.max))
            if m.errors[idx] is not None:
                uncertainties[p.name] = float(m.errors[idx])
            if merrors is not None:
                try:
                    merror = merrors[source]
                except (KeyError, TypeError):
                    merror = None
                if merror is not None and getattr(merror, "is_valid", False):
                    minos_errors[p.name] = (float(merror.lower), float(merror.upper))
        else:
            result_params.add(
                Parameter(name=p.name, value=p.value, min=p.min, max=p.max, fixed=True)
            )

    covariance = None
    covariance_order = list(free_names)
    if m.valid and getattr(m, "covariance", None) is not None:
        cov = np.asarray(m.covariance, dtype=float)
        covariance = cov.copy()

    counts = (
        observations.values
        if isinstance(observations, GaussianObservations)
        else observations.counts
    )
    objective = (
        Objective.GAUSSIAN if isinstance(observations, GaussianObservations) else Objective.POISSON
    )
    invalid_model = (
        observations.invalid_model_reason(model_values)
        if isinstance(observations, PoissonCounts)
        else None
    )
    ndata = int(np.size(counts))
    nfree = len(free_names)
    residuals = np.asarray(counts, dtype=float) - np.asarray(model_values, dtype=float)
    chi2 = float(m.fval) if chi_squared is None else float(chi_squared)
    return FitResult(
        computation_revision=computation.COMPUTATION_REVISION,
        objective=objective,
        success=bool(m.valid) and invalid_model is None,
        minimum_expected_count=float(np.min(model_values))
        if isinstance(observations, PoissonCounts) and invalid_model is None
        else None,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / max(ndata - nfree, 1),
        parameters=result_params,
        uncertainties=uncertainties,
        covariance=covariance,
        covariance_parameters=covariance_order,
        residuals=residuals,
        message=invalid_model
        or _minuit_status_message(
            m, success_message=success_message, failure_prefix=failure_prefix
        ),
        function_calls=int(getattr(m, "nfcn", 0) or 0),
        dof=ndata - nfree,
        minos_errors=minos_errors or None,
        covariance_accurate=bool(m.accurate),
    )
