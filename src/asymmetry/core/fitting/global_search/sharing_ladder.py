"""The sharing ladder: one template's sharing patterns, each a real coupled fit.

Starting from the all-local fits, the ladder shares the background, then the
amplitudes, then one further parameter at a time, and keeps a parameter shared
only while the fit stays adequate. Every rung is returned with its cost and the
trend quality of what it leaves local, so a caller can show the whole climb and
pre-select the rung that trends best. Design, evidence and the rejected
alternatives: ``docs/plans/global-wizard-trend-objective.md`` (D3, D7–D9,
D11–D12, D16).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import ParameterKind
from asymmetry.core.fitting.composite import (
    CompositeModel,
    ExprLeaf,
    ExprProduct,
    iter_nodes,
    leaf_indices,
)
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
from asymmetry.core.fitting.fraction_form import signal_fraction_form
from asymmetry.core.fitting.parameter_carry import (
    FractionWeight,
    GroupAmplitude,
    carry_parameter_set,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
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
    "RungVerdict",
    "SharingLadder",
    "climb_sharing_ladder",
]

#: A rung is adequate while χ²ᵣ rises by no more than this many of its own
#: standard deviations, √(2/ν), for the series and for every run (plan D9, D18).
ADEQUACY_SIGMA = 2.0
#: Departing runs this many in a row are a block, not isolated anomalies.
_BLOCK_LENGTH = 3
#: At most max(this many, this share of the series) runs may be exempted.
_EXEMPT_MIN_RUNS = 2
_EXEMPT_SERIES_SHARE = 0.1
#: A run's amplitude is anomalous this many scatters from the series median,
#: the scatter being the series' robust one combined with the run's own error.
_ANOMALY_SIGMA = 4.0
#: Standard deviations per median absolute deviation, for a normal scatter.
_MAD_TO_SIGMA = 1.4826
#: A signal amplitude this many of its errors on the far side of zero from its
#: run's total is a real sign, and the series then has no fractions.
_OPPOSING_AMPLITUDE_SIGMA = 2.0
#: Residual evaluations one rung's solve may spend. A pattern the data reject
#: (a rate shared across decades) crawls for thousands; one they accept is
#: there in a few hundred.
RUNG_MAX_CALLS = 1000

#: The order in which the last stage tries one further parameter (plan D8):
#: what a series is least expected to move first, rates and frequencies last.
#: A fraction weight is absent: it is shared only when every amplitude is.
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
_SCALE_KINDS = frozenset(
    {ParameterKind.AMPLITUDE, ParameterKind.FRACTION, ParameterKind.BACKGROUND}
)
_TRENDING_KINDS = frozenset({ParameterKind.RATE, ParameterKind.FREQUENCY})
#: Group key of the runs that share an amplitude other runs are exempt from.
_SHARING_RUNS = "sharing"


@dataclass(frozen=True)
class RungVerdict:
    """What one sharing pattern costs and how what it leaves local trends."""

    #: Runs that keep their own value of the shared amplitudes (plan D12).
    exempt_runs: tuple[int, ...]
    #: Rise of the series χ²ᵣ over the all-local fits, in units of √(2/ν).
    series_cost: float
    #: The same for each run, exempt runs included.
    run_costs: Mapping[int, float]
    #: Trend quality of every parameter that is still local.
    trend: CandidateTrend
    #: Shared rates and frequencies whose own component's amplitude is local (plan D3).
    hard_to_justify: tuple[str, ...]
    #: Local parameters whose acquisition passes disagree, with the passes (plan D15).
    pass_disagreements: Mapping[str, PassDiagnostic]

    @property
    def offending_runs(self) -> tuple[int, ...]:
        """Non-exempt runs whose own cost is over the tolerance, in axis order."""
        return tuple(
            run
            for run, cost in self.run_costs.items()
            if cost > ADEQUACY_SIGMA and run not in self.exempt_runs
        )

    @property
    def within_tolerance(self) -> bool:
        """The series and every non-exempt run cost no more than the tolerance (plan D9)."""
        return self.series_cost <= ADEQUACY_SIGMA and not self.offending_runs


@dataclass(frozen=True)
class LadderRung(RungVerdict):
    """One sharing pattern of one template, fitted over the series."""

    #: The template in the form this rung was fitted in: as given, or with its
    #: signal terms under one total (plan D11). Every name below is this model's.
    model: CompositeModel
    #: Parameters with one value for the series, in the order they were shared.
    shared: tuple[str, ...]
    #: Per-run results in axis order; the all-local fits on the first rung.
    results_by_run: Mapping[int, FitResult]
    #: Fitted columns over the whole series: a shared parameter counts once.
    free_parameter_count: int
    #: A block of runs at one end of the series whose amplitude departs from the
    #: rest — possible missing asymmetry. Empty when the rung is adequate, and
    #: when the amplitude departs at both ends or only inside the series.
    amplitude_unshareable_runs: tuple[int, ...]

    @property
    def converged(self) -> bool:
        return all(result.success for result in self.results_by_run.values())

    @property
    def adequate(self) -> bool:
        return self.converged and self.within_tolerance


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

    @property
    def amplitude_unshareable_runs(self) -> tuple[int, ...]:
        """The end block the climb could not share the amplitudes through, or empty.

        Only the rungs that share the amplitudes can name one; when the shared
        total and the amplitudes as given both do, it is the first one's.
        """
        return next(
            (
                rung.amplitude_unshareable_runs
                for rung in self.rungs
                if rung.amplitude_unshareable_runs
            ),
            (),
        )


@dataclass(frozen=True)
class _Form:
    """One way of writing the template, with the series' all-local fits expressed in it."""

    model: CompositeModel
    #: Name in the template as given → the parameter here that shares it. In the
    #: shared-total form every signal amplitude maps to the total.
    names: Mapping[str, str]
    #: Each run's limits and fixed flags.
    base_by_run: Mapping[int, ParameterSet]
    #: Each run's all-local values.
    local_values: Mapping[int, Mapping[str, float]]


@dataclass(frozen=True)
class _Step:
    """Parameters to share next, named as in the template as given."""

    addition: tuple[str, ...]
    #: The form to fit in; ``None`` is the form of the rung the step continues from.
    form: _Form | None = None


@dataclass(frozen=True)
class _Foothold:
    """An adequate rung, with what it shares named as in the template as given."""

    rung: LadderRung
    form: _Form
    shared: tuple[str, ...]


def _shared_total_forms(
    model: CompositeModel,
    runs: Sequence[int],
    all_local_results: Mapping[int, FitResult],
    base_by_run: Mapping[int, ParameterSet],
) -> tuple[_Form, ...]:
    """The template with its signal terms under one total, when the series has fractions.

    Empty when the model has no such form (:func:`signal_fraction_form`), or when
    this series cannot be written in it. A fraction weight lives in [0, 1] and a
    shared total has one sign, so every run's total must have the same sign and
    no signal amplitude may oppose its run's total by more than
    ``_OPPOSING_AMPLITUDE_SIGMA`` of its own error: a line whose sign the
    amplitude carries has no fraction. An amplitude that opposes by less is a
    term that has vanished, and starts at fraction zero. The signal amplitudes
    must be free on every run, so both forms fit the same number of parameters.

    The all-local fits are not repeated in this form: their values are mapped
    into it. A total is limited to the sum of its amplitudes' limits.
    """
    form = signal_fraction_form(model)
    if form is None:
        return ()
    signs: set[float] = set()
    for run in runs:
        result = all_local_results[run]
        total = sum(result.parameters[name].value for name in form.amplitudes)
        signs.add(float(np.sign(total)))
        if any(
            base_by_run[run][name].fixed
            or (
                result.parameters[name].value * total < 0.0
                and abs(result.parameters[name].value)
                > _OPPOSING_AMPLITUDE_SIGMA * result.uncertainties.get(name, 0.0)
            )
            for name in form.amplitudes
        ):
            return ()
    if signs not in ({1.0}, {-1.0}):
        return ()

    local_values = {
        run: form.grouped.normalized_parameter_values(
            form.grouped_values(
                {name: all_local_results[run].parameters[name].value for name in model.param_names}
            )
        )
        for run in runs
    }
    base: dict[int, ParameterSet] = {}
    for run in runs:
        limits = [base_by_run[run][name] for name in form.amplitudes]
        parameters = {
            form.total: Parameter(
                form.total,
                local_values[run][form.total],
                min=sum(limit.min for limit in limits),
                max=sum(limit.max for limit in limits),
            ),
            **{
                name: Parameter(name, local_values[run][name], min=0.0, max=1.0)
                for name in form.fractions
            },
            **{
                parameter.name: parameter
                for parameter in carry_parameter_set(
                    model, form.grouped, range(len(model.component_names)), base_by_run[run]
                )
            },
        }
        base[run] = ParameterSet([parameters[name] for name in form.grouped.param_names])
    return (
        _Form(
            model=form.grouped,
            names={
                **{given: grouped for grouped, given in form.carried.items()},
                **dict.fromkeys(form.amplitudes, form.total),
            },
            base_by_run=base,
            local_values=local_values,
        ),
    )


def _components_with_local_amplitude(model: CompositeModel, local: Sequence[str]) -> set[int]:
    """Components whose asymmetry a local amplitude, total or fraction sets (plan D3).

    An amplitude scales its own component and the components multiplied onto
    it; a group's total and each of its fractions scale every term of the group.
    """
    identities = model.parameter_identities()
    # Per product: every component under it, and the ones that are its own factors.
    products = [
        (
            set(leaf_indices(node)),
            {factor.index for factor in node.factors if isinstance(factor, ExprLeaf)},
        )
        for node in iter_nodes(model.expression_tree())
        if isinstance(node, ExprProduct)
    ]
    groups = [
        identity.components
        for identity in identities.values()
        if isinstance(identity, GroupAmplitude)
    ]
    kinds = model.parameter_kinds()
    scaled: set[int] = set()
    for name in local:
        identity = identities[name]
        if isinstance(identity, GroupAmplitude):
            scaled |= identity.components
        elif isinstance(identity, FractionWeight):
            scaled |= next(group for group in groups if identity.term_start in group)
        elif kinds[name] is ParameterKind.AMPLITUDE:
            scaled.add(identity.component)
    return scaled.union(*(factors for leaves, factors in products if leaves & scaled))


def _blocks(positions: Sequence[int]) -> list[np.ndarray]:
    """Split ascending positions in the series wherever they stop being neighbours."""
    return np.split(np.asarray(positions, dtype=int), np.flatnonzero(np.diff(positions) != 1) + 1)


def _isolated(positions: Sequence[int], series_length: int) -> bool:
    """Whether runs at these positions may all be exempted (plan D12): few, and no block."""
    return max(map(len, _blocks(positions))) < _BLOCK_LENGTH and len(positions) <= max(
        _EXEMPT_MIN_RUNS, int(_EXEMPT_SERIES_SHARE * series_length)
    )


def _end_block(positions: Sequence[int], series_length: int) -> list[int]:
    """The block among ``positions`` that reaches one end of the series, and only one.

    Empty when neither end is among them, and when both are: an amplitude that
    drifts across the whole series departs at both ends, and that is a trend,
    not asymmetry missing from a stretch of runs.
    """
    at_start = 0 in positions
    if at_start == (series_length - 1 in positions):
        return []
    block = _blocks(positions)[0 if at_start else -1]
    return [int(position) for position in block] if len(block) >= _BLOCK_LENGTH else []


def _departing(
    runs: Sequence[int],
    results: Mapping[int, FitResult],
    amplitude_groups: Sequence[Sequence[str]],
) -> list[int]:
    """Positions of the runs whose amplitude stands apart from the series'.

    Each group of amplitudes is about to take one value for its sum. A run
    departs when that sum lies more than ``_ANOMALY_SIGMA`` scatters from the
    series median, the scatter combining the series' median absolute deviation
    with the run's own error on the sum. A run whose fit gave no covariance for
    the amplitudes cannot be judged and does not depart.
    """
    departing: set[int] = set()
    for names in amplitude_groups:
        values = np.array(
            [sum(results[run].parameters[name].value for name in names) for run in runs]
        )
        centre = float(np.median(values))
        scatter = _MAD_TO_SIGMA * float(np.median(np.abs(values - centre)))
        for position, run in enumerate(runs):
            covered = results[run].covariance_parameters
            if results[run].covariance is None or not set(names) <= set(covered):
                continue
            columns = [covered.index(name) for name in names]
            # Rounding can take the variance of a sum just below zero.
            variance = max(float(results[run].covariance[np.ix_(columns, columns)].sum()), 0.0)
            if abs(values[position] - centre) > _ANOMALY_SIGMA * math.hypot(
                scatter, math.sqrt(variance)
            ):
                departing.add(position)
    return sorted(departing)


@dataclass(frozen=True)
class _Series:
    """The series in axis order with its all-local fits: what every rung is fitted and scored on."""

    datasets: Sequence[MuonDataset]
    #: Axis position by run number, in axis order.
    axis: Mapping[int, float]
    all_local_results: Mapping[int, FitResult]
    cancel_callback: Callable[[], bool]
    max_calls: int

    def rung(
        self,
        form: _Form,
        shared: tuple[str, ...],
        exempt: tuple[int, ...],
        results: Mapping[int, FitResult],
    ) -> LadderRung:
        """Score one pattern's results against the all-local fits."""
        runs = tuple(self.axis)
        names = form.model.param_names
        kinds = form.model.parameter_kinds()
        identities = form.model.parameter_identities()
        free_runs = {
            name: sum(not form.base_by_run[run][name].fixed for run in runs) for name in names
        }
        local_chi2 = sum(self.all_local_results[run].chi_squared for run in runs)
        local_dof = sum(self.all_local_results[run].dof for run in runs)
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
                    x=self.axis[run],
                    run=run,
                    value=results[run].parameters[name].value,
                    error=results[run].uncertainties.get(name),
                )
                for run in runs
            ]
            for name in local_names
        }
        diagnostics = {name: pass_diagnostic(points) for name, points in traces.items()}
        local_amplitude_components = _components_with_local_amplitude(form.model, local_names)
        return LadderRung(
            model=form.model,
            shared=shared,
            exempt_runs=exempt,
            results_by_run={run: results[run] for run in runs},
            free_parameter_count=sum(free_runs.values()) - columns_saved,
            series_cost=(chi2 / dof - local_chi2 / local_dof) / math.sqrt(2.0 / dof),
            run_costs={
                run: (
                    results[run].chi_squared / results[run].dof
                    - self.all_local_results[run].chi_squared / self.all_local_results[run].dof
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

    def fit(self, form: _Form, shared: tuple[str, ...], exempt: tuple[int, ...]) -> LadderRung:
        """Fit one pattern over the series, seeded as the ladder's docstring says."""
        if self.cancel_callback():
            raise FitCancelledError("Sharing ladder cancelled.")
        runs = tuple(self.axis)
        names = form.model.param_names
        kinds = form.model.parameter_kinds()
        at_median = set(shared)
        if any(kinds[name] in _SCALE_KINDS for name in shared):
            at_median.update(name for name in names if kinds[name] in _SCALE_KINDS)
        medians = {
            name: float(np.median([form.local_values[run][name] for run in runs]))
            for name in at_median
        }
        initial = {
            run: ParameterSet(
                [
                    parameter
                    if parameter.fixed
                    else replace(
                        parameter,
                        value=medians[parameter.name]
                        if parameter.name in at_median
                        else form.local_values[run][parameter.name],
                    )
                    for parameter in form.base_by_run[run]
                ]
            )
            for run in runs
        }
        # With exempt runs a shared amplitude is an engine local whose value
        # the remaining runs hold in common.
        grouped = [name for name in shared if exempt and kinds[name] is ParameterKind.AMPLITUDE]
        results, _shared_values = FitEngine().global_fit(
            list(self.datasets),
            form.model.function,
            [name for name in shared if name not in grouped],
            [name for name in names if name not in shared or name in grouped],
            initial,
            strategy="least_squares",
            max_calls=self.max_calls,
            cancel_callback=self.cancel_callback,
            local_param_groups={
                name: {run: _SHARING_RUNS for run in runs if run not in exempt} for name in grouped
            },
        )
        return self.rung(form, shared, exempt, results)

    def share_amplitudes(
        self,
        form: _Form,
        shared: tuple[str, ...],
        amplitude_groups: Sequence[Sequence[str]],
        below: LadderRung,
    ) -> LadderRung:
        """Fit the rung that shares the amplitudes, exempting isolated anomalous runs (plan D12).

        ``below`` is the rung being climbed from, in the template as given, and
        ``amplitude_groups`` its amplitudes that are each about to share their
        sum. Runs whose amplitude departs in ``below`` are exempt from the
        first fit when they are isolated; runs that still offend are exempted
        in a second and last fit when, with the first, they are isolated too.
        A rung that stays inadequate names the end block of whichever located
        the departure: the runs departing in ``below`` when they were too many
        to exempt, else the exempt and offending runs of the fit.
        """
        runs = tuple(self.axis)
        departing = _departing(runs, below.results_by_run, amplitude_groups)
        exempted = departing if _isolated(departing, len(runs)) else []
        rung = self.fit(form, shared, tuple(runs[position] for position in exempted))
        located = sorted({*exempted, *(runs.index(run) for run in rung.offending_runs)})
        if rung.converged and rung.offending_runs and _isolated(located, len(runs)):
            rung = self.fit(form, shared, tuple(runs[position] for position in located))
        if not rung.converged or rung.adequate:
            return rung
        return replace(
            rung,
            amplitude_unshareable_runs=tuple(
                runs[position]
                for position in _end_block(
                    located if exempted == departing else departing, len(runs)
                )
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
    climb_further: Callable[[], bool] = lambda: True,
    max_calls: int = RUNG_MAX_CALLS,
) -> SharingLadder:
    """Climb the sharing ladder for one template over one series (or one phase of it).

    ``axis_values`` pairs with ``datasets``; the series is taken in axis order,
    ties by run number. ``all_local_results`` are the converged independent
    per-run fits and ``base_by_run`` each run's limits and fixed flags. Only a
    parameter free on every run can be shared; one fixed on some runs stays
    local.

    The rungs, in three stages. Each stage continues from the last adequate
    rung of the stage before, and a rung that is not adequate, or whose fit
    does not converge within ``max_calls`` residual evaluations, is returned
    and its parameters are released again.

    1. Every background parameter.
    2. The amplitudes. When the series can be written with its signal terms
       under one total (:func:`_shared_total_forms`), first the total alone,
       each term's fraction staying local; then every amplitude, in the
       template as given. Both continue from stage 1 and the later adequate
       one is the foothold, so the rungs above a shared total are fitted in the
       fraction form (see :attr:`LadderRung.model`).
    3. One further parameter at a time in kind order, for as long as
       ``climb_further`` — asked before each of these rungs — returns true
       (the caller's budget, plan D8). A fraction is never one of them.

    Seeds. A shared parameter starts at the series median of its all-local
    values and a local one at its run's own all-local value, except that once
    any amplitude or background is shared every amplitude, fraction and
    background starts at its median: their all-local values sit in the
    degenerate basin the sharing is there to leave.

    Cost. ``FitResult.dof`` of a coupled fit's run is that run's points minus
    every free parameter of the model, shared or not — the same ν_r as the
    all-local fit in either form, a total and n − 1 fractions standing for n
    amplitudes — so the per-run cost is ``(χ²_r − χ²_r,local) / √(2 ν_r)``.
    Those ν_r do not sum to the series': ν = Σ ν_r,local + Σ over shared
    parameters of (runs sharing it − 1), each χ²ᵣ is taken over its own ν, and
    the difference is measured in √(2/ν) of the rung.

    Exemptions apply on the stage-2 rungs (:meth:`_Series.share_amplitudes`) and
    stage 3 inherits its foothold's. A shared amplitude or total is not a local
    parameter, exempt runs or not, so it has no trace and no trend quality.

    Raises :class:`FitCancelledError` when ``cancel_callback`` returns true
    before a rung, or inside its fit.
    """
    ordered = sorted(
        zip(axis_values, datasets, strict=True),
        key=lambda pair: (pair[0], pair[1].run_number),
    )
    axis = {int(dataset.run_number): float(x) for x, dataset in ordered}
    runs = tuple(axis)
    failed = [
        run for run in runs if not (all_local_results[run].success and all_local_results[run].dof)
    ]
    if failed:
        raise ValueError(
            "the ladder starts from converged all-local fits that carry their degrees of "
            f"freedom; runs {failed} do not"
        )
    series = _Series(
        datasets=[dataset for _x, dataset in ordered],
        axis=axis,
        all_local_results=all_local_results,
        cancel_callback=cancel_callback,
        max_calls=max_calls,
    )

    names = model.param_names
    kinds = model.parameter_kinds()
    as_given = _Form(
        model=model,
        names={name: name for name in names},
        base_by_run=base_by_run,
        local_values={
            run: {name: all_local_results[run].parameters[name].value for name in names}
            for run in runs
        },
    )
    shareable = [name for name in names if not any(base_by_run[run][name].fixed for run in runs)]
    amplitudes = tuple(name for name in shareable if kinds[name] is ParameterKind.AMPLITUDE)
    further = sorted(
        (name for name in shareable if kinds[name] in _FURTHER_KIND_ORDER),
        key=lambda name: _FURTHER_KIND_ORDER[kinds[name]],
    )
    stages: list[list[_Step]] = [
        [_Step(tuple(name for name in shareable if kinds[name] is ParameterKind.BACKGROUND))],
        [
            _Step(amplitudes, form)
            for form in (
                *_shared_total_forms(model, runs, all_local_results, base_by_run),
                as_given,
            )
        ],
        *([_Step((name,))] for name in further),
    ]

    climbed = [series.rung(as_given, (), (), all_local_results)]
    foothold = _Foothold(climbed[0], as_given, ())
    for index, stage in enumerate(stages):
        if index >= 2 and not climb_further():
            break
        below = foothold
        for step in stage:
            if not step.addition:
                continue
            form = step.form or below.form
            given = below.shared + step.addition
            shared = tuple(dict.fromkeys(form.names[name] for name in given))
            if step.addition is amplitudes:
                sums: dict[str, list[str]] = {}
                for name in amplitudes:
                    sums.setdefault(form.names[name], []).append(name)
                rung = series.share_amplitudes(form, shared, list(sums.values()), below.rung)
            else:
                rung = series.fit(form, shared, below.rung.exempt_runs)
            climbed.append(rung)
            if rung.adequate:
                foothold = _Foothold(rung, form, given)
    return SharingLadder(rungs=tuple(climbed))
