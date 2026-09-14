"""Fit one recipe across a scan, and reduce the result to a trend table.

:func:`fit_series` is the scripted form of the desktop Batch Fit tab's
block-separable path: it wraps
:func:`~asymmetry.core.fitting.series.fit_asymmetry_series` with
``seeding="auto"``, so a scan with a usable order key is *chained* — each run
warm-starts from the previous good run, and a run that lands on the spurious
near-transition branch (amplitude collapsed, frequency off the trend) is
reseeded from the trend and refitted.

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
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.fitting.result_summary import fit_result_summary
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
class SeriesOutcome:
    """Everything :func:`fit_series` produced for one scan."""

    name: str
    order_key: str
    expression: str
    global_params: list[str]
    free_params: list[str]
    seeding_used: str
    seeding_reason: str
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
            "seeding_used": self.seeding_used,
            "seeding_reason": self.seeding_reason,
            "reseeded_runs": list(self.reseeded_runs),
            "results": [dict(entry) for entry in self.results],
            "trend": self.trend.to_dict(),
        }


def _pinned_parameter_set(recipe: FitRecipe, global_params: Sequence[str]) -> ParameterSet:
    """One run's starting parameters, with every global parameter held.

    See the module docstring: a block-separable batch cannot fit a shared
    parameter, so naming one global means pinning it.
    """
    parameters = recipe.parameter_set()
    for name in global_params:
        parameters[name].fixed = True
    return parameters


def _prepared(dataset: MuonDataset, recipe: FitRecipe) -> MuonDataset:
    """The record a recipe is fitted against (its rebin applied)."""
    return dataset if recipe.rebin <= 1 else dataset.rebin(recipe.rebin)


def fit_one(dataset: MuonDataset, recipe: FitRecipe) -> dict[str, Any]:
    """Fit one run with *recipe* and summarise the result.

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


def fit_series(
    datasets_by_run: Mapping[int, MuonDataset],
    recipe: FitRecipe,
    *,
    order_key: str = "run",
    global_params: Iterable[str] = (),
    name: str,
) -> SeriesOutcome:
    """Fit *recipe* across every run, chained along *order_key*.

    ``global_params`` names the parameters held identical across the scan (see
    the module docstring — they are pinned, not jointly fitted). Raises
    :class:`ValueError` for an unknown order key or a run that does not record
    it, and :class:`KeyError` for a global parameter the recipe does not carry.
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

    initial_params = {run: _pinned_parameter_set(recipe, global_params) for run in runs}
    free_params = [parameter.name for parameter in initial_params[runs[0]].free_parameters]

    outcome = fit_asymmetry_series(
        [_prepared(datasets_by_run[run], recipe) for run in runs],
        model.function,
        global_params,
        local_params,
        initial_params,
        t_min=recipe.t_min,
        t_max=recipe.t_max,
        seeding="auto",
        order_key=order,
        amplitude_param=amplitude_param,
        frequency_param=frequency_param,
    )

    reseeded = set(outcome.reseeded_runs)
    results: list[dict[str, Any]] = []
    for run in runs:
        quality = outcome.member_quality[run]
        # The series' flag set is the trend-aware one (it alone can see the
        # scan), so the summary is asked for those flags rather than
        # re-deriving a narrower set from the result on its own.
        summary = fit_result_summary(
            outcome.results[run], extra_flags=tuple(sorted(quality.quality_flags))
        )
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
        seeding_used=outcome.seeding_used,
        seeding_reason=outcome.seeding_reason,
        reseeded_runs=list(outcome.reseeded_runs),
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
    "SeriesOutcome",
    "TrendTable",
    "build_trend_table",
    "fit_one",
    "fit_series",
    "order_values",
]
