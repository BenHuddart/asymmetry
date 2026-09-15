"""Fit one recipe across a scan, and reduce the result to a trend table.

:func:`fit_series` is the scripted form of the desktop Batch Fit tab's
block-separable path: it wraps
:func:`~asymmetry.core.fitting.series.fit_asymmetry_series` with
``seeding="auto"``, so a scan with a usable order key is *chained* — each run
warm-starts from the previous good run, and a run that lands on the spurious
near-transition branch (amplitude collapsed, frequency off the trend) is
reseeded from the trend and refitted.

Where the chain starts
----------------------

A chain is only as good as the run it starts from, and the coldest run of a
scan is rarely the best one: a recipe screened at 300 K, walked down to 100 K
one run at a time, arrives at each run from a neighbour that was itself fitted
from a poor start. ``start_run`` names the run to chain *outward* from, so the
recipe's own values seed the run they were measured on and every other run
warm-starts from a neighbour nearer to it.

That is the recommended way to use this: screen the run with the clearest
structure, then start the series there.

The split is done here rather than in
:func:`~asymmetry.core.fitting.series.fit_asymmetry_series`, which needs no
change: the batch is block-separable, so two chains over disjoint halves of a
scan are as valid as one over the whole of it. The scan is cut at the start run
into a descending branch (the start run, then every run below it) and an
ascending branch (the start run, then every run above it); each is chained in
its own direction and the per-run results are merged back into scan order. A
branch holding only the start run is not fitted, so starting at the first run
of a scan is one chain over the whole scan — exactly what omitting
``start_run`` does.

What ``global_params`` means here
--------------------------------

``fit_asymmetry_series`` fits a **block-separable** batch: one independent
minimisation per run. It therefore cannot fit a parameter jointly across runs —
it echoes the named parameters back unchanged as the "fitted globals", which is
only meaningful if they are held. So a parameter named global here is **pinned
at its recipe value in every run and not fitted**; everything else is local and
free per run. (The desktop application routes a batch with a *free* global to
``FitEngine.global_fit`` instead; a scripted simultaneous fit is not part of
this façade.) :func:`fit_series` enforces the pinning itself, so a global
parameter cannot silently end up fitted per run.

Run-bound values are re-seeded per run
--------------------------------------

A recipe carries one set of starting values, but some of those values describe
*the run they were seeded from* rather than the physics being fitted: an
applied field (``field``, ``B_L``) and a frequency-domain spectral peak.
:func:`~asymmetry.core.fitting.seeding.seed_parameters` marks exactly those
with :attr:`~asymmetry.core.fitting.seeding.Seed.run_bound`, and this function
re-seeds each of them from *each run's own record* before fitting it — so a
scan whose field changes run to run starts every fit at that run's field
instead of at the field of whichever run the recipe came from.

Two things are left alone. A parameter in :attr:`~asymmetry.core.workflow.recipe.FitRecipe.pinned`
— pinned by ``--fix``, by ``--global``, or by a hand edit — keeps the value a
person chose, since that value was not measured off a run. And a parameter the
*model or the wizard* holds fixed is still re-seeded and still held: a pinned
``B_L`` stays pinned, at each run's own field, which is the only reading of
"held" that means anything across a scan.

Nothing is dropped
------------------

A run whose fit failed, pinned a bound, was reseeded or carries a poor χ²
verdict keeps its row in the trend table and is flagged. Excluding a point is
the analyst's call, never this function's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fitting.seeding import SeedContext, seed_parameters
from asymmetry.core.fitting.series import fit_asymmetry_series
from asymmetry.core.fitting.series_seeding import resolve_series_params
from asymmetry.core.workflow.recipe import FitRecipe

#: Quantities a series may be ordered along. ``"run"`` needs no metadata; the
#: other two are read from each run's recorded scan metadata.
ORDER_KEYS = ("temperature", "field", "run")


def order_values(datasets_by_run: Mapping[int, MuonDataset], order_key: str) -> dict[int, float]:
    """The scan coordinate of every run, for ordering and for the trend x-axis.

    Raises :class:`ValueError` naming the runs that do not record *order_key* —
    a scan cannot be chained along a quantity half its members lack, and
    silently falling back to run number would mislabel the whole trend.
    """
    if order_key not in ORDER_KEYS:
        raise ValueError(
            f"Unknown order key {order_key!r}; expected one of {', '.join(ORDER_KEYS)}."
        )
    if order_key == "run":
        return {run: float(run) for run in datasets_by_run}

    values: dict[int, float] = {}
    missing: list[int] = []
    for run, dataset in datasets_by_run.items():
        value = dataset.metadata.get(order_key)
        if value is None:
            missing.append(int(run))
        else:
            values[int(run)] = float(value)
    if missing:
        raise ValueError(
            f"Run(s) {', '.join(str(run) for run in sorted(missing))} record no {order_key}; "
            f"order the series by a quantity every run has."
        )
    return values


@dataclass(frozen=True)
class TrendTable:
    """The per-run trend: one row per run, ordered along the scan."""

    order_key: str
    columns: list[str]
    rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "order_key": self.order_key,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
        }

    def to_csv(self) -> str:
        """The table as CSV: one header line, then one line per run."""
        lines = [",".join(self.columns)]
        for row in self.rows:
            lines.append(",".join(_csv_cell(row[column]) for column in self.columns))
        return "\n".join(lines) + "\n"


def _csv_cell(value: Any) -> str:
    """Render one trend cell for CSV (a flag list joins on ``;``)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return str(value)


@dataclass(frozen=True)
class SeriesBranch:
    """One chain of a series: the runs it covered and how it was seeded.

    A series started at its first run has one branch; one started in the middle
    has two, and they can resolve their seeding differently — a branch with too
    few members to chain falls back to independent seeds while the other one
    chains (see
    :func:`~asymmetry.core.fitting.series_seeding.recommend_series_seeding`).
    That is why the seeding is reported per branch rather than once.

    ``seeding_reason`` quotes the branch's *chaining* coordinate, which on a
    descending branch is the scan coordinate negated (that is what makes the
    chain run downward); ``runs`` names the members it actually covered.
    """

    direction: str
    #: The branch's runs in chaining order — starting at the series' start run.
    runs: list[int]
    seeding_used: str
    seeding_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "direction": self.direction,
            "runs": list(self.runs),
            "seeding_used": self.seeding_used,
            "seeding_reason": self.seeding_reason,
        }


@dataclass(frozen=True)
class SeriesOutcome:
    """Everything :func:`fit_series` produced for one scan."""

    name: str
    order_key: str
    expression: str
    global_params: list[str]
    free_params: list[str]
    #: The run the chain started from, and ``None`` when it started at the
    #: first run in scan order.
    start_run: int | None
    #: One entry per chain — one branch, or two when the series started in the
    #: middle of the scan.
    branches: list[SeriesBranch]
    #: Runs reseeded mid-chain, in scan order.
    reseeded_runs: list[int]
    #: One entry per run, in scan order: ``fit_result_summary`` output plus
    #: ``run``, ``x``, ``reseeded`` and the series' own ``member_quality``.
    results: list[dict[str, Any]]
    trend: TrendTable

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "name": self.name,
            "order_key": self.order_key,
            "expression": self.expression,
            "global_params": list(self.global_params),
            "free_params": list(self.free_params),
            "start_run": self.start_run,
            "branches": [branch.to_dict() for branch in self.branches],
            "reseeded_runs": list(self.reseeded_runs),
            "results": [dict(entry) for entry in self.results],
            "trend": self.trend.to_dict(),
        }


def _run_parameter_set(
    recipe: FitRecipe,
    model: CompositeModel,
    dataset: MuonDataset,
    *,
    global_params: Sequence[str],
) -> ParameterSet:
    """One run's starting parameters.

    Two departures from the recipe as written, both in the module docstring:
    every global parameter is held (a block-separable batch cannot fit a shared
    parameter, so naming one global means pinning it), and every *run-bound*
    value that nobody pinned is re-seeded from this run's own record.
    """
    parameters = recipe.parameter_set()
    kept = set(recipe.pinned) | set(global_params)
    seeds = seed_parameters(model, SeedContext(dataset=dataset, field_gauss=dataset.field))
    for name, seed in seeds.items():
        if seed.run_bound and name not in kept:
            parameters[name].value = seed.value
    for name in global_params:
        parameters[name].fixed = True
    return parameters


def _prepared(dataset: MuonDataset, recipe: FitRecipe) -> MuonDataset:
    """The record a recipe is fitted against (its rebin applied)."""
    return dataset if recipe.rebin <= 1 else dataset.rebin(recipe.rebin)


def fit_one(dataset: MuonDataset, recipe: FitRecipe) -> dict[str, Any]:
    """Fit one run with *recipe* and summarise the result.

    Starts from the recipe exactly as written — no run-bound re-seeding. That
    only makes sense across a scan, where the recipe came from a run other than
    the one being fitted; a single fit the caller aimed at one run should do
    what the recipe says.

    Returns :func:`~asymmetry.core.fitting.result_summary.fit_result_summary`
    output — values, uncertainties, χ², the quality verdict, ``params_at_bound``
    and the advisory ``quality_flags`` — with the run number and the recipe's
    free parameters alongside.
    """
    record = _prepared(dataset, recipe)
    result = FitEngine().fit(
        record,
        recipe.model().function,
        recipe.parameter_set(),
        t_min=recipe.t_min,
        t_max=recipe.t_max,
    )
    return {
        "run": int(dataset.run_number),
        "free_params": recipe.free_parameter_names(),
        **fit_result_summary(result),
    }


def _branches(runs: Sequence[int], start_run: int | None) -> list[tuple[str, list[int]]]:
    """Split *runs* (in scan order) into the chains to fit, in fitting order.

    Descending first, so that merging the branches in this order leaves the
    *ascending* branch's fit of the start run standing — the two fit it from
    identical values, and keeping one of them by a fixed rule is what makes the
    merge deterministic. A branch that would hold only the start run is
    dropped: there is nothing to chain, and the other branch fits that run.
    """
    if start_run is None:
        return [("ascending", list(runs))]
    if start_run not in runs:
        raise ValueError(
            f"Start run {start_run} is not in this series "
            f"(it holds {', '.join(str(run) for run in runs)})."
        )
    index = list(runs).index(start_run)
    descending = list(runs[index::-1])
    ascending = list(runs[index:])
    chains: list[tuple[str, list[int]]] = []
    if len(descending) > 1:
        chains.append(("descending", descending))
    if len(ascending) > 1 or not chains:
        chains.append(("ascending", ascending))
    return chains


def fit_series(
    datasets_by_run: Mapping[int, MuonDataset],
    recipe: FitRecipe,
    *,
    order_key: str = "run",
    global_params: Iterable[str] = (),
    start_run: int | None = None,
    name: str,
) -> SeriesOutcome:
    """Fit *recipe* across every run, chained outward from *start_run*.

    ``start_run`` is the run the chain starts at — the one the recipe was
    screened on, normally. Runs below it are chained downward and runs above it
    upward, so every fit warm-starts from a neighbour nearer the run the recipe
    describes. ``None`` starts at the first run in scan order, which is one
    chain over the whole scan. The start run itself is fitted by both branches
    from identical values and the ascending branch's result is the one kept.

    ``global_params`` names the parameters held identical across the scan (see
    the module docstring — they are pinned, not jointly fitted). Each run's
    run-bound values are re-seeded from its own record first, as that section
    describes. Raises :class:`ValueError` for an unknown order key, a run that
    does not record it, or a *start_run* outside the series, and
    :class:`KeyError` for a global parameter the recipe does not carry.
    """
    global_params = list(global_params)
    unknown = sorted(set(global_params) - set(recipe.parameter_names))
    if unknown:
        raise KeyError(
            f"{', '.join(unknown)} is not a parameter of {recipe.expression!r} "
            f"(it has {', '.join(recipe.parameter_names)})."
        )

    order = order_values(datasets_by_run, order_key)
    runs = sorted(datasets_by_run, key=lambda run: (order[run], run))
    model = recipe.model()
    local_params = [
        parameter for parameter in recipe.parameter_names if parameter not in global_params
    ]
    amplitude_param, frequency_param = resolve_series_params(model.param_names)

    records = {run: _prepared(datasets_by_run[run], recipe) for run in runs}
    free_params = [
        parameter.name
        for parameter in _run_parameter_set(
            recipe, model, records[runs[0]], global_params=global_params
        ).free_parameters
    ]

    fitted: dict[int, Any] = {}
    quality_by_run: dict[int, Any] = {}
    reseeded: set[int] = set()
    branches: list[SeriesBranch] = []
    for direction, chain in _branches(runs, start_run):
        # The chaining coordinate runs forward along the branch, which for the
        # descending one means the scan coordinate negated: that is the whole
        # mechanism by which fit_asymmetry_series walks a scan downward.
        sign = -1.0 if direction == "descending" else 1.0
        outcome = fit_asymmetry_series(
            [records[run] for run in chain],
            model.function,
            global_params,
            local_params,
            # A fresh set per branch: the start run is in both, and both must
            # fit it from the recipe's values rather than from each other's.
            {
                run: _run_parameter_set(recipe, model, records[run], global_params=global_params)
                for run in chain
            },
            t_min=recipe.t_min,
            t_max=recipe.t_max,
            seeding="auto",
            order_key={run: sign * order[run] for run in chain},
            amplitude_param=amplitude_param,
            frequency_param=frequency_param,
        )
        fitted.update(outcome.results)
        quality_by_run.update(outcome.member_quality)
        reseeded.update(outcome.reseeded_runs)
        branches.append(
            SeriesBranch(
                direction=direction,
                runs=list(chain),
                seeding_used=outcome.seeding_used,
                seeding_reason=outcome.seeding_reason,
            )
        )

    results: list[dict[str, Any]] = []
    for run in runs:
        quality = quality_by_run[run]
        # The series' flag set is the trend-aware one (it alone can see the
        # scan), so the summary is asked for those flags rather than
        # re-deriving a narrower set from the result on its own.
        summary = fit_result_summary(fitted[run], extra_flags=tuple(sorted(quality.quality_flags)))
        results.append(
            {
                "run": int(run),
                "x": order[run],
                "reseeded": run in reseeded,
                "member_quality": quality.to_payload(),
                **summary,
            }
        )

    return SeriesOutcome(
        name=name,
        order_key=order_key,
        expression=recipe.expression,
        global_params=global_params,
        free_params=free_params,
        start_run=start_run,
        branches=branches,
        # In scan order, which for a single ascending chain is the order the
        # chain hit them.
        reseeded_runs=[run for run in runs if run in reseeded],
        results=results,
        trend=build_trend_table(results, free_params, order_key),
    )


def build_trend_table(
    results: Sequence[Mapping[str, Any]],
    free_params: Sequence[str],
    order_key: str,
) -> TrendTable:
    """The trend table for a series: run, scan coordinate, each free parameter, flags.

    Every run that was fitted has a row, flagged or not — see the module
    docstring.
    """
    columns = ["run", "x"]
    for name in free_params:
        columns.extend([name, f"{name}_err"])
    columns.append("flags")

    rows: list[dict[str, Any]] = []
    for entry in results:
        row: dict[str, Any] = {"run": entry["run"], "x": entry["x"]}
        for name in free_params:
            row[name] = entry["parameters"].get(name)
            row[f"{name}_err"] = entry["uncertainties"].get(name)
        row["flags"] = list(entry["quality_flags"])
        rows.append(row)
    return TrendTable(order_key=order_key, columns=columns, rows=rows)


__all__ = [
    "ORDER_KEYS",
    "SeriesBranch",
    "SeriesOutcome",
    "TrendTable",
    "build_trend_table",
    "fit_one",
    "fit_series",
    "order_values",
]
