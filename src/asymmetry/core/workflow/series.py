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

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import AsymmetryScaleWarning, FitEngine
from asymmetry.core.fitting.models import LINEAR_PARAM_ROLE_NAMES
from asymmetry.core.fitting.parameters import ParameterSet, split_parameter_name
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fitting.seeding import SeedContext, seed_parameters
from asymmetry.core.fitting.series import fit_asymmetry_series
from asymmetry.core.fitting.series_seeding import resolve_series_params
from asymmetry.core.workflow.recipe import FitRecipe

#: Quantities a series may be ordered along by reading each run's metadata.
#: ``"run"`` needs none; ``"temperature"`` is the setpoint and
#: ``"sample_temperature_logged"`` the measured sample temperature, which can sit
#: several kelvin away from it. Any other quantity (a concentration, a foil
#: count, a magnet current) is supplied per run by the analyst through
#: :func:`supplied_axis`.
ORDER_KEYS = ("temperature", "sample_temperature_logged", "field", "run")


#: A fit whose amplitudes (backgrounds included) add up to more than this many
#: times the largest early-time asymmetry the record holds is describing a
#: signal the data do not contain — typically two amplitudes cancelling.
AMPLITUDE_EXCESS_FACTOR = 1.5
AMPLITUDE_EXCEEDS_DATA = "amplitude_exceeds_data"


def amplitude_exceeds_data(dataset: MuonDataset, parameters: Mapping[str, float]) -> bool:
    """Whether a fit's amplitudes add up to more than the record can hold (see above).

    The record's scale is the 95th percentile of ``|A|`` over its first tenth —
    robust to an oscillation, whose early mean can sit near zero.
    """
    import numpy as np

    asymmetry = np.abs(np.asarray(dataset.asymmetry, dtype=float))
    early = asymmetry[: max(5, asymmetry.size // 10)]
    scale = max(float(np.percentile(early, 95)), 1.0)
    total = sum(
        abs(value)
        for name, value in parameters.items()
        if split_parameter_name(name)[0] in LINEAR_PARAM_ROLE_NAMES
    )
    return total > AMPLITUDE_EXCESS_FACTOR * scale


FREQUENCY_UNRESOLVED = "frequency_unresolved"


def frequency_unresolved(
    dataset: MuonDataset, parameters: Mapping[str, float], free: Sequence[str]
) -> bool:
    """Whether a free frequency completes too few cycles in the informative window.

    The wizard's own :data:`MIN_CYCLES_IN_EFFECTIVE_WINDOW` rule: below it a
    fitted "frequency" is a relaxation in disguise — the spurious branch a weak
    line fitted without its relaxing background falls onto.
    """
    from asymmetry.core.fitting.fit_wizard import (
        MIN_CYCLES_IN_EFFECTIVE_WINDOW,
        effective_window_duration,
    )

    window = effective_window_duration(dataset)
    return any(
        abs(parameters[name]) * window < MIN_CYCLES_IN_EFFECTIVE_WINDOW
        for name in free
        if split_parameter_name(name)[0] == "frequency"
    )


#: The two relaxation envelopes a single-envelope recipe is weighed between on
#: every run: a static distribution of fields dephases as a Gaussian, one
#: fluctuating faster than its width (motional narrowing) as an exponential.
RIVAL_ENVELOPES = {"Gaussian": "Exponential", "Exponential": "Gaussian"}

#: The χ² margin by which one envelope must beat the other to be named — the
#: two have the same parameter count, so this is also the AIC difference.
ENVELOPE_MARGIN = 10.0


def rival_envelope_model(model: CompositeModel) -> tuple[CompositeModel, dict[str, str]] | None:
    """*model* with its one relaxation envelope swapped, and the parameter renames.

    ``None`` unless the model carries exactly one Gaussian or Exponential — with
    two, which one is "the" envelope is not the model's to say.
    """
    names = model.component_names
    envelopes = [index for index, name in enumerate(names) if name in RIVAL_ENVELOPES]
    if len(envelopes) != 1:
        return None
    index = envelopes[0]
    payload = model.to_dict()
    payload["component_names"] = [
        *names[:index],
        RIVAL_ENVELOPES[names[index]],
        *names[index + 1 :],
    ]
    rival = CompositeModel.from_dict(payload)
    return rival, dict(zip(model.param_names, rival.param_names, strict=True))


def envelope_preference(
    record: MuonDataset,
    recipe: FitRecipe,
    rival: tuple[CompositeModel, dict[str, str]],
    fitted: Mapping[str, float],
    chi_squared: float,
) -> dict[str, Any]:
    """Refit one run with the rival envelope, started from its fitted values.

    Returns the preferred envelope's name (``"either"`` inside
    :data:`ENVELOPE_MARGIN`) and ``delta_chi2`` = χ²(rival) − χ²(recipe); both
    ``None`` when the rival fit failed, since a failed fit weighs nothing.
    """
    model, renames = rival
    parameters = ParameterSet(
        [
            replace(entry, name=renames[entry.name], value=fitted[entry.name]).to_parameter()
            for entry in recipe.parameters
        ]
    )
    # The start is a converged fit, not a seed, so the seed-scale guard's
    # premise does not hold: a small fitted amplitude is a result.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AsymmetryScaleWarning)
        result = FitEngine().fit(
            record, model.function, parameters, t_min=recipe.t_min, t_max=recipe.t_max
        )
    if not result.success:
        return {"preferred": None, "delta_chi2": None}
    own = next(name for name in recipe.model().component_names if name in RIVAL_ENVELOPES)
    delta = float(result.chi_squared) - float(chi_squared)
    if delta >= ENVELOPE_MARGIN:
        preferred = own
    elif delta <= -ENVELOPE_MARGIN:
        preferred = RIVAL_ENVELOPES[own]
    else:
        preferred = "either"
    return {"preferred": preferred, "delta_chi2": delta}


def envelope_change(trend: TrendTable) -> str | None:
    """A note naming where the preferred envelope changes along the scan, or ``None``."""
    if "envelope" not in trend.columns:
        return None
    decided = [row for row in trend.rows if row["envelope"] in RIVAL_ENVELOPES]
    if len({row["envelope"] for row in decided}) < 2:
        return None
    blocks: list[tuple[str, list[dict[str, Any]]]] = []
    for row in decided:
        if blocks and blocks[-1][0] == row["envelope"]:
            blocks[-1][1].append(row)
        else:
            blocks.append((row["envelope"], [row]))
    described = "; ".join(
        f"{shape} on {', '.join(str(row['run']) for row in rows)} ({trend.order_key} {_span(rows)})"
        for shape, rows in blocks
    )
    return (
        f"NOTE: the relaxation shape changes along this scan — {described} (the envelope "
        f"column; runs marked 'either' fit both alike). A Gaussian (a static spread of fields) "
        f"turning exponential as the fluctuations outrun it is motional narrowing: report the "
        f"shape against {trend.order_key}, not only the rate."
    )


def _span(rows: Sequence[Mapping[str, Any]]) -> str:
    values = [row["x"] for row in rows]
    return f"{min(values):g}" if len(values) == 1 else f"{min(values):g}–{max(values):g}"


@dataclass(frozen=True)
class ScanAxis:
    """The coordinate a series is ordered and trended along: one value per run."""

    name: str
    values: dict[int, float]


def scan_axis(datasets_by_run: Mapping[int, MuonDataset], order_key: str) -> ScanAxis:
    """The axis *order_key* names, read from every run's recorded metadata.

    Raises :class:`ValueError` naming the runs that do not record *order_key* —
    a scan cannot be chained along a quantity half its members lack, and
    silently falling back to run number would mislabel the whole trend.
    """
    if order_key not in ORDER_KEYS:
        raise ValueError(
            f"{order_key!r} is not recorded in the files (they record "
            f"{', '.join(ORDER_KEYS)}); supply its value for every run instead."
        )
    if order_key == "run":
        return ScanAxis(order_key, {int(run): float(run) for run in datasets_by_run})

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
    return ScanAxis(order_key, values)


def supplied_axis(name: str, values: Mapping[int, float], runs: Iterable[int]) -> ScanAxis:
    """An axis the analyst supplies, with exactly one value for each of *runs*.

    *name* labels the trend; it may not be one of :data:`ORDER_KEYS`, whose
    values come from the files. Raises :class:`ValueError` naming the runs with
    no value, and the values given for runs outside the series.
    """
    if name in ORDER_KEYS:
        raise ValueError(f"{name!r} is read from the files; order by it without supplying values.")
    runs = {int(run) for run in runs}
    given = {int(run) for run in values}
    if runs - given:
        raise ValueError(
            f"No {name} value for run(s) {', '.join(str(run) for run in sorted(runs - given))}."
        )
    if given - runs:
        raise ValueError(
            f"{name} values given for run(s) "
            f"{', '.join(str(run) for run in sorted(given - runs))}, which are not in the series."
        )
    return ScanAxis(name, {int(run): float(value) for run, value in values.items()})


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


def _workflow_flags(
    record: MuonDataset, summary: Mapping[str, Any], free: Sequence[str]
) -> list[str]:
    """The engine's quality flags plus the workflow's own checks against the record.

    Not member-quality flags (that vocabulary is the engine's): a fit whose
    amplitudes the record cannot hold, or whose frequency it cannot resolve.
    """
    flags = set(summary["quality_flags"])
    if amplitude_exceeds_data(record, summary["parameters"]):
        flags.add(AMPLITUDE_EXCEEDS_DATA)
    if frequency_unresolved(record, summary["parameters"], free):
        flags.add(FREQUENCY_UNRESOLVED)
    return sorted(flags)


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
    summary = fit_result_summary(result)
    summary["quality_flags"] = _workflow_flags(record, summary, recipe.free_parameter_names())
    return {
        "run": int(dataset.run_number),
        "free_params": recipe.free_parameter_names(),
        **summary,
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
    axis: ScanAxis,
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
    describes. *axis* orders the scan and is the trend's x; it carries a value
    for every run. Raises :class:`ValueError` for a *start_run* outside the
    series, and :class:`KeyError` for a global parameter the recipe does not
    carry.
    """
    global_params = list(global_params)
    unknown = sorted(set(global_params) - set(recipe.parameter_names))
    if unknown:
        raise KeyError(
            f"{', '.join(unknown)} is not a parameter of {recipe.expression!r} "
            f"(it has {', '.join(recipe.parameter_names)})."
        )

    order = axis.values
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

    rival = rival_envelope_model(model)
    results: list[dict[str, Any]] = []
    for run in runs:
        quality = quality_by_run[run]
        # The series' flag set is the trend-aware one (it alone can see the
        # scan), so the summary is asked for those flags rather than
        # re-deriving a narrower set from the result on its own.
        summary = fit_result_summary(fitted[run], extra_flags=tuple(sorted(quality.quality_flags)))
        summary["quality_flags"] = _workflow_flags(records[run], summary, free_params)
        results.append(
            {
                "run": int(run),
                "x": order[run],
                "reseeded": run in reseeded,
                "member_quality": quality.to_payload(),
                **summary,
                **(
                    {}
                    if rival is None
                    else {"envelope": {"preferred": None, "delta_chi2": None}}
                    if not summary["success"]
                    else {
                        "envelope": envelope_preference(
                            records[run],
                            recipe,
                            rival,
                            summary["parameters"],
                            summary["chi_squared"],
                        )
                    }
                ),
            }
        )

    return SeriesOutcome(
        name=name,
        order_key=axis.name,
        expression=recipe.expression,
        global_params=global_params,
        free_params=free_params,
        start_run=start_run,
        branches=branches,
        # In scan order, which for a single ascending chain is the order the
        # chain hit them.
        reseeded_runs=[run for run in runs if run in reseeded],
        results=results,
        trend=build_trend_table(
            results,
            free_params,
            axis.name,
            survey_lines=(
                {run: survey_line(datasets_by_run[run]) for run in runs}
                if any(name.startswith("frequency") for name in free_params)
                else None
            ),
        ),
    )


def survey_line(dataset: MuonDataset) -> float | None:
    """The line the folder's survey measured in this run (MHz), or ``None``.

    A reduced run carries its survey row as metadata; a line is recorded there
    when the survey found precession (``larmor`` or ``other``).
    """
    if dataset.metadata.get("precession") not in ("larmor", "other"):
        return None
    return float(dataset.metadata["precession_frequency_mhz"])


def build_trend_table(
    results: Sequence[Mapping[str, Any]],
    free_params: Sequence[str],
    order_key: str,
    survey_lines: Mapping[int, float | None] | None = None,
) -> TrendTable:
    """The trend table for a series: run, scan coordinate, each free parameter, flags.

    Every run that was fitted has a row, flagged or not — see the module
    docstring. With *survey_lines*, a ``survey_line_mhz`` column sets each
    run's surveyed line beside its fitted frequencies, so a fit that has
    drifted off the measured line, or found one where the survey saw none,
    shows in the table.
    """
    columns = ["run", "x"]
    for name in free_params:
        columns.extend([name, f"{name}_err"])
    if survey_lines is not None:
        columns.append("survey_line_mhz")
    compared = all("envelope" in entry for entry in results)
    if compared:
        columns.extend(["envelope", "envelope_dchi2"])
    columns.append("flags")

    rows: list[dict[str, Any]] = []
    for entry in results:
        row: dict[str, Any] = {"run": entry["run"], "x": entry["x"]}
        for name in free_params:
            row[name] = entry["parameters"].get(name)
            row[f"{name}_err"] = entry["uncertainties"].get(name)
        if survey_lines is not None:
            row["survey_line_mhz"] = survey_lines[entry["run"]]
        if compared:
            row["envelope"] = entry["envelope"]["preferred"]
            row["envelope_dchi2"] = entry["envelope"]["delta_chi2"]
        row["flags"] = list(entry["quality_flags"])
        rows.append(row)
    return TrendTable(order_key=order_key, columns=columns, rows=rows)


__all__ = [
    "AMPLITUDE_EXCEEDS_DATA",
    "AMPLITUDE_EXCESS_FACTOR",
    "ENVELOPE_MARGIN",
    "FREQUENCY_UNRESOLVED",
    "ORDER_KEYS",
    "RIVAL_ENVELOPES",
    "ScanAxis",
    "SeriesBranch",
    "SeriesOutcome",
    "TrendTable",
    "amplitude_exceeds_data",
    "build_trend_table",
    "envelope_change",
    "envelope_preference",
    "fit_one",
    "fit_series",
    "frequency_unresolved",
    "rival_envelope_model",
    "scan_axis",
    "supplied_axis",
    "survey_line",
]
