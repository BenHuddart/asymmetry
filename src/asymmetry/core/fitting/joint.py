"""Joint fitting: several series, different models, shared parameters.

A sample measured across a phase transition needs two fit functions — an
oscillating model in the magnetically ordered phase, a relaxing one in the
paramagnetic phase — and some parameters are the same physical quantity in both:
the initial asymmetry, the background, alpha, a phase. Fitting the two series
separately and hoping the shared quantity comes out the same throws away the
constraint *and* reports uncertainties that pretend the two measurements never
met. A joint fit puts both series in one cost function with one column for the
shared quantity, so its value is estimated from all the data and its σ
propagates into every other parameter's.

This module is the user-facing seam over :meth:`FitEngine.joint_fit`, the way
:mod:`asymmetry.core.fitting.asymmetry_global` sits over
:meth:`FitEngine.global_fit`. It owns three things the engine deliberately does
not: the three-scope vocabulary (Local per run, **Global** per series, **Shared**
across series), the validation that turns a malformed shared table into a clear
error instead of a silently wrong fit, and
:func:`suggest_shared_parameters`, which proposes the default shared table from
the member models themselves.

Everything here raises rather than guards. A run in two series, a shared
parameter that one series pins everywhere, a shared row with only one member —
each is a question only the caller can answer, and quietly dropping it would
produce a fit that *looks* joint and is not.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import (
    CoupledBlock,
    FitEngine,
    FitResult,
    SharedColumn,
)
from asymmetry.core.fitting.parameter_carry import ComponentParameter
from asymmetry.core.fitting.parameters import (
    ParameterSet,
    get_param_info,
    split_parameter_name,
)

#: The series-level role a parameter must carry to be shareable (D4). Local and
#: Fixed rows belong to one series by definition; only a Global row already
#: denotes "one value for this whole series", which is what a shared column
#: widens to "one value across these series".
GLOBAL_ROLE = "global"


@dataclass(frozen=True)
class JointSeriesProblem:
    """One series' contribution to a joint fit: its data, model and partition.

    Mirrors what the Batch tab already records for a series — the model, the
    member runs, the fit range, the per-run parameter table with its
    global/local/fixed roles — so a joint fit composes series the user has
    already made rather than asking them to reassign runs to functions.
    """

    #: The caller's identifier for this series (a batch id, a label, an index).
    key: Hashable
    #: This series' datasets. Run numbers must be unique across the whole joint fit.
    datasets: list[MuonDataset]
    #: ``f(t, **params) -> array`` for *this* series; other series may differ.
    model_fn: Callable[..., NDArray]
    #: Names with one fitted value for the whole series (the Global role).
    global_params: list[str]
    #: Names fitted independently per run (the Local role).
    local_params: list[str]
    #: ``{run number: ParameterSet}`` — seeds, bounds and pins, per run.
    initial_params: dict[int, ParameterSet]
    t_min: float | None = None
    t_max: float | None = None
    #: WiMDA-style equality link groups: ``{local name: {run: group key}}``.
    #: Runs sharing a key share one fitted value, exactly as in ``global_fit``.
    local_param_groups: dict[str, dict[int, Hashable]] | None = None


@dataclass(frozen=True)
class SharedParameter:
    """One quantity held equal across two or more series.

    ``members`` maps a series key to the parameter name *that series* spells the
    quantity with, because two models may perfectly well call the same physics
    ``A_bg`` and ``baseline``. The seed and bounds belong to the shared row, not
    to either series: the shared value is a new fitted quantity (D6 seeds it from
    the first contributing series, but that is the caller's choice to make).
    """

    name: str
    members: dict[Hashable, str]
    value: float
    min: float = -math.inf
    max: float = math.inf


@dataclass
class JointFitResult:
    """Result bundle for a joint fit.

    ``series_results`` is what every existing consumer reads: each run's
    :class:`FitResult` carries the shared value *and its uncertainty* under that
    series' own parameter name, so a trend plot, an export or a results table
    needs to know nothing about joint fitting to render a jointly fitted series.
    The shared table itself — values, σ, and the covariance block that says how
    the shared quantities trade off against each other — is reported separately
    for the surfaces that do.
    """

    success: bool
    #: The shared quantities' fitted values, under the shared names.
    shared_parameters: ParameterSet
    #: 1σ on each shared quantity, ``{shared name: sigma}``.
    shared_uncertainties: dict[str, float] = field(default_factory=dict)
    #: Covariance over the shared columns, in the order they were passed.
    shared_covariance: NDArray | None = None
    #: ``{series key: {run number: FitResult}}``.
    series_results: dict[Hashable, dict[int, FitResult]] = field(default_factory=dict)
    #: ``{series key: that series' fitted globals}``, shared values included
    #: under the series' own (local) names.
    series_global_parameters: dict[Hashable, ParameterSet] = field(default_factory=dict)
    #: Per-series χ²ᵣ, each computed as if that series were fitted alone (a
    #: shared column counts against every series that contributes to it), so it
    #: reads as the familiar per-series goodness of fit.
    series_reduced_chi_squared: dict[Hashable, float] = field(default_factory=dict)
    #: Σ χ² over every run of every series.
    chi_squared: float = 0.0
    #: ΣN − (number of fitted columns), a shared column counted once.
    dof: int = 0
    reduced_chi_squared: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class SharedSuggestion:
    """A proposed row of the shared table, with the evidence for it.

    ``tier`` is ``"exact"`` when the same parameter name means the same thing in
    every model beyond doubt (ticked by default) and ``"candidate"`` when the
    match is a reasonable inference from the models' structure (offered
    unticked). Anything more ambiguous than that is not proposed at all.
    """

    name: str
    members: dict[Hashable, str]
    tier: Literal["exact", "candidate"]
    rationale: str


def fit_joint(
    problems: Sequence[JointSeriesProblem],
    shared: Sequence[SharedParameter],
    *,
    strategy: str = "least_squares",
    method: str = "migrad",
    max_calls: int = 10000,
    minos: bool = False,
    fit_engine: FitEngine | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> JointFitResult:
    """Fit several series at once, holding the named parameters equal across them.

    One cost function spans every run of every series. Each series keeps its own
    model, fit range, seeds, bounds and pins; the entries of ``shared`` are the
    only thing that couples them. A series that *pins* a shared member on one of
    its runs uses its own pinned value there — a file-pinned or user-pinned
    quantity is a measurement, and the constraint must not overwrite it.

    Parameters
    ----------
    problems
        Two or more series. Run numbers must be unique across the whole joint
        fit: the same data in the cost twice would double-count it.
    shared
        The shared table. Each entry names two or more series' parameters that
        are the same quantity. A member must be a **Global** parameter of its
        series (``global_params``) and must be free on at least one of that
        series' runs — a name that series pins everywhere has no column to share.
    strategy
        ``"least_squares"`` (default, bounded sparse trust region) or
        ``"joint"`` (one Minuit over the whole vector). ``"profiled"`` raises:
        it profiles one set of globals over datasets that share a model, which a
        joint fit does not have.
    minos
        Asymmetric MINOS intervals; ``"joint"`` strategy only.
    fit_engine
        Optional :class:`FitEngine` to reuse; a fresh one is created otherwise.
    cancel_callback
        Cooperative cancel probe polled inside the cost function; a truthy return
        raises :class:`~asymmetry.core.fitting.engine.FitCancelledError`, and the
        run records nothing.

    Raises
    ------
    ValueError
        Fewer than two series; duplicate series keys; a run number in two series;
        a shared row with fewer than two members, an unknown series key, a member
        that is not one of that series' Global parameters, or a member that
        series pins on every run; the same parameter shared twice; an unknown
        ``strategy``.
    NotImplementedError
        ``strategy="profiled"``, or an affine parameter tie anywhere (ties are
        honoured only by the single-run path).
    """

    problems = list(problems)
    shared = list(shared)

    if len(problems) < 2:
        raise ValueError(
            "A joint fit needs at least two series; with one series use "
            "FitEngine.global_fit (the Batch tab's Global role)."
        )

    by_key: dict[Hashable, JointSeriesProblem] = {}
    for problem in problems:
        if problem.key in by_key:
            raise ValueError(f"Duplicate series key in a joint fit: {problem.key!r}")
        by_key[problem.key] = problem

    _validate_runs_do_not_overlap(problems)
    _validate_shared_table(by_key, shared)

    blocks = [_block_for(problem) for problem in problems]
    shared_columns = [
        SharedColumn(
            name=entry.name,
            members=dict(entry.members),
            value=entry.value,
            min=entry.min,
            max=entry.max,
        )
        for entry in shared
    ]

    engine = fit_engine or FitEngine()
    solution = engine.joint_fit(
        blocks,
        shared_columns,
        strategy=strategy,
        method=method,
        max_calls=max_calls,
        minos=minos,
        cancel_callback=cancel_callback,
    )

    return JointFitResult(
        success=solution.success,
        shared_parameters=solution.shared_parameters,
        shared_uncertainties=solution.shared_uncertainties,
        shared_covariance=solution.shared_covariance,
        series_results=solution.series_results,
        series_global_parameters=solution.series_global_parameters,
        series_reduced_chi_squared={
            key: chi2 / max(solution.series_dof[key], 1)
            for key, chi2 in solution.series_chi_squared.items()
        },
        chi_squared=solution.chi_squared,
        dof=solution.dof,
        reduced_chi_squared=solution.chi_squared / max(solution.dof, 1),
        message=solution.message,
    )


def _validate_runs_do_not_overlap(problems: Sequence[JointSeriesProblem]) -> None:
    """Raise unless every run number appears in exactly one series (D3)."""
    owner: dict[int, Hashable] = {}
    for problem in problems:
        for ds in problem.datasets:
            run = int(ds.run_number)
            if run in owner:
                raise ValueError(
                    f"Run {run} appears in both series {owner[run]!r} and {problem.key!r}; "
                    "a joint fit may not weigh the same data twice."
                )
            owner[run] = problem.key
            if run not in problem.initial_params:
                raise KeyError(f"Series {problem.key!r} has no initial parameter set for run {run}")


def _validate_shared_table(
    by_key: Mapping[Hashable, JointSeriesProblem],
    shared: Sequence[SharedParameter],
) -> None:
    """Raise on any shared row that cannot mean what it says."""
    seen_names: set[str] = set()
    claimed: dict[tuple[Hashable, str], str] = {}
    for entry in shared:
        if entry.name in seen_names:
            raise ValueError(f"Duplicate shared parameter name {entry.name!r}")
        seen_names.add(entry.name)
        if len(entry.members) < 2:
            raise ValueError(
                f"Shared parameter {entry.name!r} has {len(entry.members)} member(s); "
                "a shared parameter must be contributed by at least two series."
            )
        for key, pname in entry.members.items():
            problem = by_key.get(key)
            if problem is None:
                raise ValueError(
                    f"Shared parameter {entry.name!r} names series {key!r}, which is not "
                    "one of the joint fit's series."
                )
            if pname not in problem.global_params:
                raise ValueError(
                    f"Shared parameter {entry.name!r} maps series {key!r} to {pname!r}, "
                    "which is not one of that series' Global parameters; only a Global "
                    "row can be shared across series."
                )
            if all(problem.initial_params[ds.run_number][pname].fixed for ds in problem.datasets):
                raise ValueError(
                    f"Shared parameter {entry.name!r} maps series {key!r} to {pname!r}, "
                    "which that series pins on every run; a pinned value has no column "
                    "to share. Free it in at least one run, or drop the series from "
                    "this shared parameter."
                )
            previous = claimed.get((key, pname))
            if previous is not None:
                raise ValueError(
                    f"Series {key!r} maps {pname!r} to both shared parameters "
                    f"{previous!r} and {entry.name!r}; a parameter can only be one "
                    "shared quantity."
                )
            claimed[(key, pname)] = entry.name


def _block_for(problem: JointSeriesProblem) -> CoupledBlock:
    """Translate a series problem into the engine's block."""

    groups = problem.local_param_groups

    def local_group_key(pname: str, run_number: int) -> Hashable:
        """The sharing key for a local param on a run (run number by default)."""
        if groups and pname in groups:
            return groups[pname].get(run_number, run_number)
        return run_number

    # "Free" means free on at least one run: a parameter pinned from metadata on
    # one run (a longitudinal field at 0 on a zero-field run) is still the
    # series' free Global parameter on the others.
    free_global_params = [
        pname
        for pname in problem.global_params
        if any(not problem.initial_params[ds.run_number][pname].fixed for ds in problem.datasets)
    ]
    return CoupledBlock(
        key=problem.key,
        datasets=list(problem.datasets),
        model_fn=problem.model_fn,
        global_params=list(problem.global_params),
        local_params=list(problem.local_params),
        initial_params=problem.initial_params,
        free_global_params=free_global_params,
        local_group_key=local_group_key,
        t_min=problem.t_min,
        t_max=problem.t_max,
    )


# --------------------------------------------------------------------------- #
# Autodetection of the default shared table (D7)
# --------------------------------------------------------------------------- #


def suggest_shared_parameters(
    models: Sequence[CompositeModel],
    roles: Sequence[Mapping[str, str]],
) -> list[SharedSuggestion]:
    """Propose the default shared table from the member models and their roles.

    Two tiers, and nothing else:

    * ``"exact"`` — the *same full parameter name*, with the same unit, occurring
      exactly once in each model and carrying the Global role in each series.
      There is no room for doubt about what such a row means, so the caller ticks
      these by default.
    * ``"candidate"`` — the same **base** name at a different component index
      (``A_1`` here, ``A_2`` there), or the same **component type** where each
      model has exactly one instance of it; again unit-equal and Global
      everywhere. A reasonable inference, offered unticked.

    Anything ambiguous is not proposed. A name that occurs twice in one model, a
    component type one model instantiates twice, a unit that differs between
    models, a parameter that is Local or Fixed in any series — each is left for
    the user, because guessing wrong here silently constrains two quantities that
    are not the same measurement.

    The one caveat the docs must repeat: "initial asymmetry" is shareable only
    where the model exposes it as *one* parameter (a fraction-group model with an
    overall scale). A sum of component amplitudes is a constraint, not a shared
    parameter, and is out of scope.
    """

    if len(models) != len(roles):
        raise ValueError(
            f"suggest_shared_parameters got {len(models)} models and {len(roles)} role maps; "
            "there must be one role map per model."
        )
    if len(models) < 2:
        return []

    keys = list(range(len(models)))
    suggestions: list[SharedSuggestion] = []
    proposed: set[tuple[tuple[Hashable, str], ...]] = set()

    def _propose(
        name: str,
        members: dict[Hashable, str],
        tier: Literal["exact", "candidate"],
        rationale: str,
    ) -> None:
        signature = tuple(sorted(members.items(), key=lambda item: str(item[0])))
        if signature in proposed:
            return
        proposed.add(signature)
        suggestions.append(
            SharedSuggestion(name=name, members=members, tier=tier, rationale=rationale)
        )

    # Tier one: the same full name everywhere.
    for pname in models[0].param_names:
        if not all(model.param_names.count(pname) == 1 for model in models):
            continue
        if not all(_is_global(roles[index], pname) for index in keys):
            continue
        if not _units_agree([pname] * len(models)):
            continue
        # A name is only as meaningful as the component that owns it: the
        # composite disambiguates amplitudes by term index, so ``Gaussian +
        # Constant`` and ``Exponential + Constant`` both spell their first
        # amplitude ``A_1`` while meaning different things. Same name on
        # different component types is a reasonable inference, not a certainty.
        component_types = _component_types(models, pname)
        if len(component_types) > 1:
            _propose(
                pname,
                {index: pname for index in keys},
                "candidate",
                f"{pname} appears once in every model, but on different components "
                f"({', '.join(sorted(component_types))}).",
            )
            continue
        _propose(
            pname,
            {index: pname for index in keys},
            "exact",
            f"{pname} appears once in every model with the same unit and the Global role.",
        )

    # Tier two, rule one: the same base name at a different component index.
    for base in _ordered_bases(models):
        members: dict[Hashable, str] = {}
        for index, model in enumerate(models):
            matches = [
                pname
                for pname in model.param_names
                if split_parameter_name(pname)[0] == base and _is_global(roles[index], pname)
            ]
            if len(matches) != 1:
                break
            members[index] = matches[0]
        else:
            if _units_agree(list(members.values())):
                _propose(
                    base,
                    members,
                    "candidate",
                    f"Each model has exactly one Global {base} parameter "
                    f"({', '.join(sorted(set(members.values())))}), same unit.",
                )

    # Tier two, rule two: the same component type, instantiated once in each model.
    for component_name, local_name in _ordered_component_parameters(models):
        members = {}
        for index, model in enumerate(models):
            pname = _sole_component_parameter(model, component_name, local_name)
            if pname is None or not _is_global(roles[index], pname):
                break
            members[index] = pname
        else:
            if _units_agree(list(members.values())):
                _propose(
                    split_parameter_name(members[0])[0],
                    members,
                    "candidate",
                    f"Every model has exactly one {component_name} component, and its "
                    f"{local_name} is Global in each.",
                )

    return suggestions


def _is_global(roles: Mapping[str, str], pname: str) -> bool:
    return roles.get(pname) == GLOBAL_ROLE


def _units_agree(names: Sequence[str]) -> bool:
    """True when every name carries the same unit (``None`` counts as a unit)."""
    units = {get_param_info(name).unit for name in names}
    return len(units) == 1


def _component_types(models: Sequence[CompositeModel], pname: str) -> set[str]:
    """The component types that own ``pname`` across ``models``.

    Only a :class:`ComponentParameter` names a component; a group amplitude or a
    fraction weight belongs to a term structure rather than a component type and
    contributes nothing here, so a name owned by such identities everywhere
    yields an empty set and stays in the exact tier.
    """
    types: set[str] = set()
    for model in models:
        identity = model.parameter_identities()[pname]
        if isinstance(identity, ComponentParameter):
            types.add(model.component_names[identity.component])
    return types


def _ordered_bases(models: Sequence[CompositeModel]) -> list[str]:
    """Base names of the first model's parameters, in order, deduplicated."""
    seen: list[str] = []
    for pname in models[0].param_names:
        base = split_parameter_name(pname)[0]
        if base not in seen:
            seen.append(base)
    return seen


def _ordered_component_parameters(
    models: Sequence[CompositeModel],
) -> list[tuple[str, str]]:
    """``(component type, local parameter)`` pairs of the first model, in order."""
    pairs: list[tuple[str, str]] = []
    for identity in models[0].parameter_identities().values():
        if not isinstance(identity, ComponentParameter):
            continue
        pair = (models[0].component_names[identity.component], identity.local_name)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _sole_component_parameter(
    model: CompositeModel, component_name: str, local_name: str
) -> str | None:
    """The fitted name of ``local_name`` on ``component_name``, if unambiguous.

    ``None`` when the model has no such component, has more than one of them (the
    match would be a coin flip), or suppresses that parameter (an amplitude the
    product-scale policy folded into another factor is not a parameter at all).
    """
    indices = [index for index, name in enumerate(model.component_names) if name == component_name]
    if len(indices) != 1:
        return None
    for pname, identity in model.parameter_identities().items():
        if (
            isinstance(identity, ComponentParameter)
            and identity.component == indices[0]
            and identity.local_name == local_name
        ):
            return pname
    return None


__all__ = [
    "JointFitResult",
    "JointSeriesProblem",
    "SharedParameter",
    "SharedSuggestion",
    "fit_joint",
    "suggest_shared_parameters",
]
