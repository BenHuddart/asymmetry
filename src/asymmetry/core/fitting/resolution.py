"""Per-component resolution / identifiability diagnostic for composite fits.

A composite relaxation model will happily converge with one branch railed to a
rate so high that its 1/e time spans a fraction of a time bin — pure early-bin
absorption rather than a measured relaxation — or collapsed so slow that it is
degenerate with a free constant baseline.  Either way an information criterion
still "prefers" the extra component, because the extra branch does buy χ²: it is
acting as a free amplitude adjustment, not as a rate.  This module turns that
judgement into a library primitive.

The rule
--------
For every *relaxation rate* parameter of a fitted composite model (a parameter
carrying the ``µs⁻¹`` unit — the inverse characteristic time of its component's
envelope), the rate must lie inside the window the data can actually resolve:

* **fast edge** — the 1/e time must span at least ``min_bins_per_e_folding``
  time bins.  Faster than that and the component is absorbed by the first few
  bins.
* **slow edge** — the 1/e time must be no longer than the fit window.  Slower
  than that and the component is degenerate with a free constant baseline.

The check is deliberately **two-sided**: a one-sided (fast-edge-only) rule
silently admits the collapsed-to-zero branch, which is the same pathology seen
from the other end.

Judged over the Δχ² neighbourhood, not the point estimate
---------------------------------------------------------
A point-estimate test gives *optimizer-dependent* verdicts.  Where the
likelihood has a flat ridge in a rate — the fit is statistically indifferent
between a resolved rate and a railed one — two optimizers landing in
statistically indistinguishable minima return opposite answers on the same data.
So the verdict is a property of the likelihood, not of the basin an optimizer
happened to find: a component counts as resolved only if the whole
``Δχ² <= delta_chi_squared`` neighbourhood of the rate stays inside the resolved
window.  When the admissible interval *spans* the boundary — resolved at one end,
railed at the other — the honest verdict is :attr:`ResolutionVerdict.UNDETERMINED`,
which is distinct from both "resolved" and "railed".

The neighbourhood comes from one of two sources:

``"profile"`` (default when a dataset is supplied)
    Each boundary is probed directly: pin the rate at the boundary value, re-fit
    every other free parameter, and read off Δχ².  Two warm re-fits per rate
    parameter answer exactly the question the rule asks — *can the admissible
    set reach the boundary?* — without needing to map the interval out.

``solutions``
    A caller-supplied multistart solution list
    (``(chi_squared, {parameter: value})`` pairs).  Every solution within
    ``delta_chi_squared`` of the best contributes its rates, and the verdict is
    taken over their span.  This is the cheap path for callers that already ran
    a multistart and kept the converged seeds.

With neither, the assessment falls back to the point estimate and says so
(:attr:`ComponentResolutionAssessment.neighbourhood_source` is ``"point"``);
that mode is documented as optimizer-dependent and is not the default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

__all__ = [
    "DEFAULT_DELTA_CHI_SQUARED",
    "DEFAULT_MIN_BINS_PER_E_FOLDING",
    "ComponentResolution",
    "ComponentResolutionAssessment",
    "ResolutionVerdict",
    "assess_component_resolution",
    "relaxation_rate_parameter_names",
]

#: Default fast-edge tolerance: a component's 1/e time must span at least this
#: many time bins to count as a *shape* the data resolve rather than an
#: instantaneous amplitude adjustment inside the leading bins.  Deliberately
#: permissive — analyses with a specific binning argument routinely want more
#: (a rebinned corpus may justify tens of bins), so it is a keyword argument.
DEFAULT_MIN_BINS_PER_E_FOLDING = 3.0

#: Default neighbourhood width.  Δχ² <= 1 is the standard 1σ interval for one
#: parameter of interest, i.e. "rates the data cannot distinguish from the best
#: fit at 1σ".
DEFAULT_DELTA_CHI_SQUARED = 1.0

#: Unit strings that mark a parameter as an inverse characteristic time.  These
#: are the components' declared units (``ComponentDefinition.param_info``), so
#: the diagnostic tracks the component registry rather than a private name list.
_RATE_UNITS = frozenset({"µs⁻¹", "us^-1", "μs⁻¹"})

#: Component categories that are not relaxation channels: a background carries
#: no characteristic time, and frequency-domain components are not time-domain
#: envelopes at all.
_NON_RELAXATION_CATEGORIES = frozenset({"Background", "Frequency Domain"})

#: The profile probe pins a rate *just outside* the resolved window rather than
#: exactly on its edge.  A rate sitting exactly on the edge still satisfies the
#: rule ("at least ``min_bins_per_e_folding`` bins"), so probing the edge itself
#: would answer a question one step short of the one being asked; probing 5 %
#: outside asks it directly — *is an unresolved rate admissible here?*
_BOUNDARY_PROBE_MARGIN = 1.05

_EPS = 1e-12


def _effective_delta_chi_squared(
    delta_chi_squared: float,
    fit_result: FitResult,
    n_points: int,
) -> float:
    """Δχ² threshold rescaled when the fit is far better than its quoted errors.

    Δχ² = 1 is the 1σ interval *on the assumption that the quoted errors describe
    the scatter*.  When a fit lands at χ²/ν far below 1 — synthetic data
    evaluated without noise, or a dataset whose errors are overstated — that
    assumption fails, and a unit χ² excursion covers a parameter range the data
    plainly do determine; every component would read as undetermined.  Scaling
    the threshold by the fitted χ²/ν in that regime is the usual error-rescaling
    convention, and it restores the intended meaning: "rates the data cannot
    distinguish from the best fit at the noise level the fit actually sees".

    Only applied when χ²/ν < 1.  A poor fit (χ²/ν > 1) keeps the plain threshold:
    widening the neighbourhood there would make a badly fitting model *easier* to
    call unidentifiable, which is the wrong direction and not what the rule is
    about.
    """
    dof = fit_result.dof
    if dof <= 0:
        free = len(fit_result.parameters.free_parameters)
        dof = int(n_points) - free
    if dof <= 0 or not math.isfinite(fit_result.chi_squared):
        return delta_chi_squared
    reduced = float(fit_result.chi_squared) / float(dof)
    if not math.isfinite(reduced) or reduced >= 1.0:
        return delta_chi_squared
    return delta_chi_squared * max(reduced, 0.0)


class ResolutionVerdict(str, Enum):
    """Per-component verdict of the resolution rule."""

    #: The rate, and its whole Δχ² neighbourhood, lie inside the resolved window.
    RESOLVED = "resolved"
    #: The rate is railed fast: its 1/e time spans fewer than the required bins.
    UNRESOLVED_FAST = "unresolved_fast"
    #: The rate has collapsed: its 1/e time exceeds the fit window, so the
    #: component is degenerate with a constant baseline.
    UNRESOLVED_SLOW = "unresolved_slow"
    #: The point estimate is inside the window but the admissible neighbourhood
    #: reaches outside it — the data do not determine whether this component is
    #: a measured rate.
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class ComponentResolution:
    """The resolution verdict for one relaxation-rate parameter."""

    #: Fitted parameter name (the unique composite name, e.g. ``"Lambda_2"``).
    parameter_name: str
    #: Name of the component the parameter belongs to (e.g. ``"Exponential"``).
    component_name: str
    #: Index of that component within the composite model.
    component_index: int
    #: Fitted rate, µs⁻¹.
    rate: float
    #: 1/e time implied by the rate, µs.
    e_folding_time: float
    #: 1/e time expressed in time bins — the quantity the fast edge tests.
    bins_per_e_folding: float
    verdict: ResolutionVerdict
    #: ``(slow_edge, fast_edge)`` rates bounding the resolved window, µs⁻¹.
    resolved_rate_window: tuple[float, float]
    #: Span of the rate over the Δχ² neighbourhood, µs⁻¹, or ``None`` when the
    #: neighbourhood was not explored (``neighbourhood_source == "point"``).
    admissible_rate_range: tuple[float, float] | None
    #: Number of neighbourhood samples the verdict rests on (probes or
    #: solutions); ``1`` for a point-estimate verdict.
    neighbourhood_size: int
    #: Human-readable statement of what was found.
    detail: str

    @property
    def is_resolved(self) -> bool:
        return self.verdict is ResolutionVerdict.RESOLVED


@dataclass(frozen=True)
class ComponentResolutionAssessment:
    """Resolution verdicts for every relaxation channel of one composite fit."""

    components: tuple[ComponentResolution, ...]
    #: Time-bin width used for the fast edge, µs.
    bin_width: float
    #: ``(t_min, t_max)`` of the window the verdict was formed over, µs.
    fit_window: tuple[float, float]
    min_bins_per_e_folding: float
    delta_chi_squared: float
    #: ``"profile"``, ``"solutions"`` or ``"point"`` — see the module docstring.
    neighbourhood_source: str

    @property
    def is_resolved(self) -> bool:
        """True when every relaxation channel is resolved.

        A model with no relaxation channel at all (a pure background, a pure
        oscillation) is vacuously resolved: the rule has nothing to say about it.
        """
        return all(component.is_resolved for component in self.components)

    @property
    def unresolved(self) -> tuple[ComponentResolution, ...]:
        """The components that failed, in model order."""
        return tuple(component for component in self.components if not component.is_resolved)

    def disqualification_reasons(self) -> tuple[str, ...]:
        """Reasons suitable for a wizard candidate's ``disqualification_reasons``."""
        return tuple(component.detail for component in self.unresolved)


def _clone_parameters(parameters: ParameterSet) -> ParameterSet:
    return ParameterSet(
        [
            Parameter(
                name=parameter.name,
                value=parameter.value,
                min=parameter.min,
                max=parameter.max,
                fixed=parameter.fixed,
                expr=parameter.expr,
            )
            for parameter in parameters
        ]
    )


def _window_and_bin_width(
    time: NDArray[np.float64],
    t_min: float | None,
    t_max: float | None,
) -> tuple[tuple[float, float], float]:
    values = np.asarray(time, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot assess component resolution: the time axis is empty.")
    low = float(values.min() if t_min is None else t_min)
    high = float(values.max() if t_max is None else t_max)
    if high <= low:
        raise ValueError(
            f"Cannot assess component resolution: the fit window ({low}, {high}) is empty."
        )
    inside = values[(values >= low) & (values <= high)]
    if inside.size >= 2:
        diffs = np.diff(np.sort(inside))
        diffs = diffs[diffs > 0.0]
    else:
        diffs = np.asarray([], dtype=float)
    if diffs.size:
        bin_width = float(np.median(diffs))
    else:
        # Single retained sample: the window itself is the only length scale we
        # have, so use it rather than inventing a bin.
        bin_width = high - low
    return (low, high), bin_width


def _rate_parameters(
    model: CompositeModel,
    parameters: ParameterSet | None,
) -> list[tuple[int, str, str]]:
    """Return ``(component_index, component_name, fitted_parameter_name)`` triples.

    One entry per relaxation-rate parameter of the model.  ``parameters`` filters
    to the ones actually present in a fitted set; ``None`` reports the model's
    full complement, which is what a caller sizing a search budget wants.
    """
    found: list[tuple[int, str, str]] = []
    mappings = model.parameter_mapping()
    for index, (component, mapping) in enumerate(zip(model.components, mappings, strict=True)):
        if component.category in _NON_RELAXATION_CATEGORIES:
            continue
        for local_name, fitted_name in mapping.items():
            info = component.param_info.get(local_name)
            unit = getattr(info, "unit", None)
            if unit not in _RATE_UNITS:
                continue
            if parameters is not None and fitted_name not in parameters:
                continue
            found.append((index, component.name, fitted_name))
    return found


def relaxation_rate_parameter_names(
    model: CompositeModel,
    parameters: ParameterSet | None = None,
) -> tuple[str, ...]:
    """Fitted parameter names that the resolution rule judges, in model order.

    Callers that need to know *whether* a candidate has a composite relaxation
    structure worth judging — the fit wizard gates its per-candidate probe on
    having at least two, and sizes its seed ladder by the same count — can ask
    without running the assessment.  With ``parameters=None`` the model's full
    complement is reported; otherwise the answer is filtered to the names that
    fitted set actually carries.
    """
    return tuple(name for _index, _component, name in _rate_parameters(model, parameters))


def _probe_boundary(
    *,
    engine: FitEngine,
    dataset: MuonDataset,
    model: CompositeModel,
    parameters: ParameterSet,
    parameter_name: str,
    boundary: float,
    t_min: float | None,
    t_max: float | None,
) -> float:
    """χ² of the best fit with ``parameter_name`` pinned at ``boundary``.

    Returns ``inf`` when the constrained fit fails, so a failed probe reads as
    "the neighbourhood does not reach the boundary" rather than silently
    flipping a verdict.
    """
    probe = _clone_parameters(parameters)
    pinned = probe[parameter_name]
    pinned.value = float(boundary)
    pinned.fixed = True
    try:
        result = engine.fit(dataset, model.function, probe, t_min=t_min, t_max=t_max)
    except Exception:  # pragma: no cover - engine/backend failure is not a verdict
        return float("inf")
    if result is None or not result.success or not math.isfinite(result.chi_squared):
        return float("inf")
    return float(result.chi_squared)


def _reachable(parameter: Parameter, boundary: float) -> bool:
    """Can the fit put ``parameter`` at ``boundary`` without leaving its bounds?"""
    if np.isfinite(parameter.min) and boundary < parameter.min:
        return False
    return not (np.isfinite(parameter.max) and boundary > parameter.max)


def _verdict_for(
    *,
    rate: float,
    slow_edge: float,
    fast_edge: float,
    span: tuple[float, float] | None,
) -> ResolutionVerdict:
    if not math.isfinite(rate) or rate <= 0.0:
        return ResolutionVerdict.UNRESOLVED_SLOW
    if rate > fast_edge:
        return ResolutionVerdict.UNRESOLVED_FAST
    if rate < slow_edge:
        return ResolutionVerdict.UNRESOLVED_SLOW
    if span is not None and (span[0] < slow_edge or span[1] > fast_edge):
        return ResolutionVerdict.UNDETERMINED
    return ResolutionVerdict.RESOLVED


def _detail_for(
    *,
    verdict: ResolutionVerdict,
    parameter_name: str,
    rate: float,
    bins: float,
    min_bins: float,
    window: tuple[float, float],
    span: tuple[float, float] | None,
) -> str:
    duration = window[1] - window[0]
    if verdict is ResolutionVerdict.UNRESOLVED_FAST:
        return (
            f"{parameter_name} is not resolved at this binning: the fitted rate "
            f"{rate:.4g} µs⁻¹ has a 1/e time spanning {bins:.2g} time bins "
            f"(< {min_bins:g} required), so the component absorbs the leading bins "
            f"rather than measuring a rate"
        )
    if verdict is ResolutionVerdict.UNRESOLVED_SLOW:
        return (
            f"{parameter_name} is not resolved in this fit window: the fitted rate "
            f"{rate:.4g} µs⁻¹ has a 1/e time longer than the {duration:.4g} µs window, "
            f"so the component is degenerate with a constant baseline"
        )
    if verdict is ResolutionVerdict.UNDETERMINED:
        low, high = span if span is not None else (float("nan"), float("nan"))
        return (
            f"{parameter_name} is undetermined: rates from {low:.4g} to {high:.4g} µs⁻¹ "
            f"are statistically indistinguishable here, spanning both resolved and "
            f"unresolved values, so the data do not determine this component"
        )
    return (
        f"{parameter_name} = {rate:.4g} µs⁻¹ is resolved "
        f"({bins:.3g} bins per 1/e time, within the {duration:.4g} µs window)"
    )


def _span_from_solutions(
    solutions: Sequence[tuple[float, Mapping[str, float]]],
    parameter_name: str,
    delta_chi_squared: float,
) -> tuple[tuple[float, float] | None, int]:
    finite = [
        (float(chi2), values)
        for chi2, values in solutions
        if math.isfinite(float(chi2)) and parameter_name in values
    ]
    if not finite:
        return None, 0
    best = min(chi2 for chi2, _ in finite)
    near = [
        float(values[parameter_name]) for chi2, values in finite if chi2 - best <= delta_chi_squared
    ]
    near = [rate for rate in near if math.isfinite(rate)]
    if not near:
        return None, 0
    return (min(near), max(near)), len(near)


def assess_component_resolution(
    fit_result: FitResult,
    model: CompositeModel,
    dataset: MuonDataset | None = None,
    *,
    time: NDArray[np.float64] | Sequence[float] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    min_bins_per_e_folding: float = DEFAULT_MIN_BINS_PER_E_FOLDING,
    delta_chi_squared: float = DEFAULT_DELTA_CHI_SQUARED,
    solutions: Sequence[tuple[float, Mapping[str, float]]] | None = None,
    probe_neighbourhood: bool | None = None,
    fit_engine: FitEngine | None = None,
) -> ComponentResolutionAssessment:
    """Judge whether each relaxation channel of a composite fit is resolved.

    Parameters
    ----------
    fit_result
        The converged fit whose components are to be judged.
    model
        The composite model that was fitted — its component registry supplies
        which parameters are relaxation rates.
    dataset
        The fitted data.  Supplies the time binning and the fit window, and is
        what the profile probe re-fits against.  May be omitted only when
        ``time`` is given *and* the neighbourhood comes from ``solutions`` (or
        is waived), since there is nothing to re-fit without it.
    time
        Explicit time axis, when no dataset is available.
    t_min, t_max
        Fit window, if narrower than the data.  Defaults to the data extent.
    min_bins_per_e_folding
        Fast-edge tolerance in time bins.  See
        :data:`DEFAULT_MIN_BINS_PER_E_FOLDING`.
    delta_chi_squared
        Width of the admissible neighbourhood.  See
        :data:`DEFAULT_DELTA_CHI_SQUARED`.
    solutions
        Optional multistart solutions as ``(chi_squared, {parameter: value})``.
        When given, the neighbourhood is taken from these rather than probed.
    probe_neighbourhood
        Force the profile probe on (``True``) or off (``False``).  The default
        (``None``) probes whenever a dataset is available and no ``solutions``
        were supplied.
    fit_engine
        Engine used for the probe re-fits.  A fresh :class:`FitEngine` by default.

    Returns
    -------
    ComponentResolutionAssessment
        One :class:`ComponentResolution` per relaxation-rate parameter, in model
        order.  A model with no relaxation channel yields an empty, vacuously
        resolved assessment.
    """
    if dataset is None and time is None:
        raise ValueError(
            "assess_component_resolution needs either a dataset or an explicit time axis."
        )
    axis = np.asarray(dataset.time if dataset is not None else time, dtype=float)
    window, bin_width = _window_and_bin_width(axis, t_min, t_max)

    min_bins = float(max(min_bins_per_e_folding, _EPS))
    fast_edge = 1.0 / max(min_bins * bin_width, _EPS)
    slow_edge = 1.0 / max(window[1] - window[0], _EPS)

    parameters = fit_result.parameters
    rate_parameters = _rate_parameters(model, parameters)

    use_probe = probe_neighbourhood
    if use_probe is None:
        use_probe = solutions is None and dataset is not None and fit_result.success
    if use_probe and dataset is None:
        raise ValueError("Probing the Δχ² neighbourhood requires the fitted dataset.")

    if solutions is not None:
        source = "solutions"
    elif use_probe:
        source = "profile"
    else:
        source = "point"

    engine = fit_engine or FitEngine()
    base_chi2 = float(fit_result.chi_squared)
    effective_delta = _effective_delta_chi_squared(
        float(delta_chi_squared), fit_result, int(axis.size)
    )

    components: list[ComponentResolution] = []
    for component_index, component_name, parameter_name in rate_parameters:
        parameter = parameters[parameter_name]
        rate = float(parameter.value)
        e_folding = 1.0 / rate if rate > 0.0 and math.isfinite(rate) else float("inf")
        bins = e_folding / bin_width if math.isfinite(e_folding) else float("inf")

        span: tuple[float, float] | None = None
        neighbourhood_size = 1
        if source == "solutions":
            span, neighbourhood_size = _span_from_solutions(
                solutions or (), parameter_name, effective_delta
            )
            if span is not None:
                span = (min(span[0], rate), max(span[1], rate))
            neighbourhood_size = max(neighbourhood_size, 1)
        elif source == "profile" and parameter.fixed:
            # A rate held fixed by the caller is a modelling decision, not a
            # fitted quantity; probing it would re-fit a parameter the caller
            # pinned. Judge the point estimate and say the neighbourhood is
            # trivial.
            span = (rate, rate)
        elif source == "profile":
            low, high = rate, rate
            probes = 1
            for boundary in (
                slow_edge / _BOUNDARY_PROBE_MARGIN,
                fast_edge * _BOUNDARY_PROBE_MARGIN,
            ):
                if not _reachable(parameter, boundary):
                    continue
                chi2 = _probe_boundary(
                    engine=engine,
                    dataset=dataset,  # type: ignore[arg-type]
                    model=model,
                    parameters=parameters,
                    parameter_name=parameter_name,
                    boundary=boundary,
                    t_min=t_min,
                    t_max=t_max,
                )
                probes += 1
                if chi2 - base_chi2 <= effective_delta:
                    low = min(low, boundary)
                    high = max(high, boundary)
            span = (low, high)
            neighbourhood_size = probes

        verdict = _verdict_for(rate=rate, slow_edge=slow_edge, fast_edge=fast_edge, span=span)
        components.append(
            ComponentResolution(
                parameter_name=parameter_name,
                component_name=component_name,
                component_index=component_index,
                rate=rate,
                e_folding_time=e_folding,
                bins_per_e_folding=bins,
                verdict=verdict,
                resolved_rate_window=(slow_edge, fast_edge),
                admissible_rate_range=span,
                neighbourhood_size=neighbourhood_size,
                detail=_detail_for(
                    verdict=verdict,
                    parameter_name=parameter_name,
                    rate=rate,
                    bins=bins,
                    min_bins=min_bins,
                    window=window,
                    span=span,
                ),
            )
        )

    return ComponentResolutionAssessment(
        components=tuple(components),
        bin_width=bin_width,
        fit_window=window,
        min_bins_per_e_folding=min_bins,
        delta_chi_squared=float(effective_delta),
        neighbourhood_source=source,
    )
