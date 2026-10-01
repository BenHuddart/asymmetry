"""The sharing ladder: one template's sharing patterns, each a real coupled fit.

Starting from the all-local fits, the ladder shares the background, then the
amplitudes, then one further parameter at a time, and keeps a parameter shared
only while the fit stays adequate. Every rung is returned with its cost and the
trend quality of what it leaves local, so a caller can show the whole climb and
pre-select the rung that trends best. Design, evidence and the rejected
alternatives: ``docs/plans/global-wizard-trend-objective.md`` (D3, D7–D9, D12).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import ParameterKind
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
from asymmetry.core.fitting.parameter_carry import ComponentParameter
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.fitting.trend_quality import (
    CandidateTrend,
    PassDiagnostic,
    TracePoint,
    pass_diagnostic,
    trace_quality,
)

__all__ = [
    "ADEQUACY_SIGMA",
    "RUNG_MAX_CALLS",
    "LadderRung",
    "SharingLadder",
    "climb_sharing_ladder",
]

#: A rung is adequate while χ²ᵣ rises by no more than this many of its own
#: standard deviations, √(2/ν), for the series and for every run (plan D9, D18).
ADEQUACY_SIGMA = 2.0
#: Offending runs this many in a row are a block, not isolated anomalies.
_BLOCK_LENGTH = 3
#: At most max(this many, this share of the series) runs may be exempted.
_EXEMPT_MIN_RUNS = 2
_EXEMPT_SERIES_SHARE = 0.1
#: Residual evaluations one rung's solve may spend. A pattern the data reject
#: (a rate shared across decades) crawls for thousands; one they accept is
#: there in a few hundred.
RUNG_MAX_CALLS = 1000

#: The order in which rung 3 tries one further parameter (plan D8): what a
#: series is least expected to move first, rates and frequencies last. A
#: fraction weight is absent: it is shared only through its group total.
_FURTHER_KIND_ORDER: dict[ParameterKind, int] = {
    ParameterKind.PHASE: 0,
    ParameterKind.SHAPE: 1,
    ParameterKind.STATIC_WIDTH: 2,
    ParameterKind.GEOMETRY: 3,
    ParameterKind.FIELD: 3,
    ParameterKind.RATE: 4,
    ParameterKind.FREQUENCY: 4,
}
#: The record's scale and baseline: degenerate with each other when the
#: relaxation is slow, so their all-local values are not a seed worth keeping.
_SCALE_KINDS = frozenset({ParameterKind.AMPLITUDE, ParameterKind.BACKGROUND})
_TRENDING_KINDS = frozenset({ParameterKind.RATE, ParameterKind.FREQUENCY})
#: Group key of the runs that share an amplitude other runs are exempt from.
_SHARING_RUNS = "sharing"


@dataclass(frozen=True)
class LadderRung:
    """One sharing pattern of one template, fitted over the series."""

    #: Parameters with one value for the series, in the order they were shared.
    shared: tuple[str, ...]
    #: Runs that keep their own value of the shared amplitudes (plan D12).
    exempt_runs: tuple[int, ...]
    #: Per-run results in axis order; the all-local fits on the first rung.
    results_by_run: Mapping[int, FitResult]
    #: Fitted columns over the whole series: a shared parameter counts once.
    free_parameter_count: int
    #: Rise of the series χ²ᵣ over the all-local fits, in units of √(2/ν).
    series_cost: float
    #: The same for each run, exempt runs included.
    run_costs: Mapping[int, float]
    #: Trend quality of every parameter that is still local.
    trend: CandidateTrend
    #: Shared rates and frequencies whose own component's amplitude is local (plan D3).
    hard_to_justify: tuple[str, ...]
    #: A block of runs at an end of the series through which the amplitude
    #: cannot be shared — possible missing asymmetry. Empty when there is none.
    amplitude_unshareable_runs: tuple[int, ...]
    #: Local parameters whose acquisition passes disagree, with the passes (plan D15).
    pass_disagreements: Mapping[str, PassDiagnostic]

    @property
    def converged(self) -> bool:
        return all(result.success for result in self.results_by_run.values())

    @property
    def offending_runs(self) -> tuple[int, ...]:
        """Non-exempt runs whose own cost is over the tolerance, in axis order."""
        return tuple(
            run
            for run, cost in self.run_costs.items()
            if cost > ADEQUACY_SIGMA and run not in self.exempt_runs
        )

    @property
    def adequate(self) -> bool:
        return self.converged and self.series_cost <= ADEQUACY_SIGMA and not self.offending_runs


@dataclass(frozen=True)
class SharingLadder:
    """Every rung climbed for one template, the all-local rung first."""

    rungs: tuple[LadderRung, ...]

    @property
    def preselected(self) -> LadderRung:
        """The adequate rung that trends best (plan D9).

        Ties go to the rung that is not hard to justify, then to the one with
        fewer free parameters. The all-local rung is always adequate.
        """
        return max(
            (rung for rung in self.rungs if rung.adequate),
            key=lambda rung: (
                rung.trend.ordering_key,
                not rung.hard_to_justify,
                -rung.free_parameter_count,
            ),
        )


def climb_sharing_ladder(
    datasets: Sequence[MuonDataset],
    model: CompositeModel,
    *,
    all_local_results: Mapping[int, FitResult],
    base_by_run: Mapping[int, ParameterSet],
    axis_values: Sequence[float],
    cancel_callback: Callable[[], bool],
    max_further_parameters: int | None = None,
    max_calls: int = RUNG_MAX_CALLS,
) -> SharingLadder:
    """Climb the sharing ladder for one template over one series (or one phase of it).

    ``axis_values`` pairs with ``datasets``; the series is taken in axis order,
    ties by run number. ``all_local_results`` are the converged independent
    per-run fits and ``base_by_run`` each run's limits and fixed flags. Only a
    parameter free on every run can be shared; one fixed on some runs stays
    local.

    The rungs, each continuing from the last adequate one: every background
    parameter; every amplitude; then one further parameter at a time in kind
    order, at most ``max_further_parameters`` of them (``None``: all). A rung
    that is not adequate, or whose fit does not converge within ``max_calls``
    residual evaluations, is returned and its parameters are released again.

    Seeds. A shared parameter starts at the series median of its all-local
    values and a local one at its run's own all-local value, except that once
    any amplitude or background is shared every amplitude and background starts
    at its median: their all-local values sit in the degenerate basin the
    sharing is there to leave.

    Cost. ``FitResult.dof`` of a coupled fit's run is that run's points minus
    every free parameter of the model, shared or not — the same ν_r as the
    all-local fit, so the per-run cost is ``(χ²_r − χ²_r,local) / √(2 ν_r)``.
    Those ν_r do not sum to the series': ν = Σ ν_r,local + Σ over shared
    parameters of (runs sharing it − 1), each χ²ᵣ is taken over its own ν, and
    the difference is measured in √(2/ν) of the rung.

    Exemptions apply on the rung that shares the amplitudes. Isolated offending
    runs keep their own amplitudes and the rung is refitted once; later rungs
    inherit them. Offenders that are not isolated fail the rung, and a block of
    them reaching one end of the series (not both: then no run is left to share
    with) is recorded as the runs the amplitude cannot be shared through. A
    shared amplitude is not a local parameter, exempt runs or not, so it has no
    trace and no trend quality.

    Raises :class:`FitCancelledError` when ``cancel_callback`` returns true
    before a rung, or inside its fit.
    """
    ordered = sorted(
        zip(axis_values, datasets, strict=True),
        key=lambda pair: (pair[0], pair[1].run_number),
    )
    series = [dataset for _x, dataset in ordered]
    axis = {int(dataset.run_number): float(x) for x, dataset in ordered}
    runs = tuple(axis)
    failed = [run for run in runs if not all_local_results[run].success]
    if failed:
        raise ValueError(f"the ladder starts from converged all-local fits; runs {failed} are not")

    names = model.param_names
    kinds = model.parameter_kinds()
    identities = model.parameter_identities()
    free_runs = {name: sum(not base_by_run[run][name].fixed for run in runs) for name in names}
    medians = {
        name: float(np.median([all_local_results[run].parameters[name].value for run in runs]))
        for name in names
    }
    local_chi2 = sum(all_local_results[run].chi_squared for run in runs)
    local_dof = sum(all_local_results[run].dof for run in runs)

    def rung(
        shared: tuple[str, ...], exempt: tuple[int, ...], results: Mapping[int, FitResult]
    ) -> LadderRung:
        columns_saved = sum(
            len(runs) - 1 - (len(exempt) if kinds[name] is ParameterKind.AMPLITUDE else 0)
            for name in shared
        )
        dof = local_dof + columns_saved
        chi2 = sum(results[run].chi_squared for run in runs)
        local_names = [name for name in names if free_runs[name] and name not in shared]
        traces = {
            name: [
                TracePoint(
                    x=axis[run],
                    run=run,
                    value=results[run].parameters[name].value,
                    error=results[run].uncertainties.get(name),
                )
                for run in runs
            ]
            for name in local_names
        }
        diagnostics = {name: pass_diagnostic(points) for name, points in traces.items()}
        local_amplitude_components = {
            identities[name].component
            for name in local_names
            if kinds[name] is ParameterKind.AMPLITUDE
            and isinstance(identities[name], ComponentParameter)
        }
        return LadderRung(
            shared=shared,
            exempt_runs=exempt,
            results_by_run={run: results[run] for run in runs},
            free_parameter_count=sum(free_runs.values()) - columns_saved,
            series_cost=(chi2 / dof - local_chi2 / local_dof) / math.sqrt(2.0 / dof),
            run_costs={
                run: (
                    results[run].chi_squared / results[run].dof
                    - all_local_results[run].chi_squared / all_local_results[run].dof
                )
                / math.sqrt(2.0 / results[run].dof)
                for run in runs
            },
            trend=CandidateTrend({name: trace_quality(points) for name, points in traces.items()}),
            hard_to_justify=tuple(
                name
                for name in shared
                if kinds[name] in _TRENDING_KINDS
                and identities[name].component in local_amplitude_components
            ),
            amplitude_unshareable_runs=(),
            pass_disagreements={
                name: diagnostic
                for name, diagnostic in diagnostics.items()
                if diagnostic.passes_disagree
            },
        )

    def fit(shared: tuple[str, ...], exempt: tuple[int, ...]) -> LadderRung:
        if cancel_callback():
            raise FitCancelledError("Sharing ladder cancelled.")
        at_median = set(shared)
        if any(kinds[name] in _SCALE_KINDS for name in shared):
            at_median.update(name for name in names if kinds[name] in _SCALE_KINDS)
        initial = {
            run: ParameterSet(
                [
                    parameter
                    if parameter.fixed
                    else replace(
                        parameter,
                        value=medians[parameter.name]
                        if parameter.name in at_median
                        else all_local_results[run].parameters[parameter.name].value,
                    )
                    for parameter in base_by_run[run]
                ]
            )
            for run in runs
        }
        # With exempt runs a shared amplitude is an engine local whose value
        # the remaining runs hold in common.
        grouped = [name for name in shared if exempt and kinds[name] is ParameterKind.AMPLITUDE]
        results, _shared_values = FitEngine().global_fit(
            series,
            model.function,
            [name for name in shared if name not in grouped],
            [name for name in names if name not in shared or name in grouped],
            initial,
            strategy="least_squares",
            max_calls=max_calls,
            cancel_callback=cancel_callback,
            local_param_groups={
                name: {run: _SHARING_RUNS for run in runs if run not in exempt} for name in grouped
            },
        )
        return rung(shared, exempt, results)

    shareable = [name for name in names if free_runs[name] == len(runs)]
    further = sorted(
        (name for name in shareable if kinds[name] in _FURTHER_KIND_ORDER),
        key=lambda name: _FURTHER_KIND_ORDER[kinds[name]],
    )
    additions = [
        tuple(name for name in shareable if kinds[name] is ParameterKind.BACKGROUND),
        tuple(name for name in shareable if kinds[name] is ParameterKind.AMPLITUDE),
        *((name,) for name in further[:max_further_parameters]),
    ]

    climbed = [rung((), (), all_local_results)]
    foothold = climbed[0]
    for addition in filter(None, additions):
        shared = foothold.shared + addition
        step = fit(shared, foothold.exempt_runs)
        offenders = step.offending_runs
        if kinds[addition[0]] is ParameterKind.AMPLITUDE and step.converged and offenders:
            # Offenders' positions in axis order, split wherever they stop being neighbours.
            positions = [runs.index(run) for run in offenders]
            blocks = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
            if max(map(len, blocks)) < _BLOCK_LENGTH and len(offenders) <= max(
                _EXEMPT_MIN_RUNS, int(_EXEMPT_SERIES_SHARE * len(runs))
            ):
                step = fit(shared, offenders)
            else:
                step = replace(
                    step,
                    amplitude_unshareable_runs=tuple(
                        runs[position]
                        for block in blocks
                        if len(block) >= _BLOCK_LENGTH
                        and (block[0] == 0) != (block[-1] == len(runs) - 1)
                        for position in block
                    ),
                )
        climbed.append(step)
        if step.adequate:
            foothold = step
    return SharingLadder(rungs=tuple(climbed))
