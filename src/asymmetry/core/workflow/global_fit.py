"""True simultaneous fitting for a group of reduced runs, or for a batch of groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fitting.seeding import SeedContext, seed_parameters
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.series import ScanAxis, TrendTable, build_trend_table


@dataclass(frozen=True)
class GlobalFitOutcome:
    expression: str
    shared_params: list[str]
    local_params: list[str]
    field_params: list[str]
    strategy: str
    shared: dict[str, float]
    shared_uncertainties: dict[str, float]
    #: The run-local parameters that were fitted — what the trend tabulates.
    free_params: list[str]
    #: One entry per run, ordered along the axis, each carrying its ``x``.
    results: list[dict[str, Any]]
    trend: TrendTable

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "order_key": self.trend.order_key,
            "shared_params": list(self.shared_params),
            "local_params": list(self.local_params),
            "field_params": list(self.field_params),
            "strategy": self.strategy,
            "shared": dict(self.shared),
            "shared_uncertainties": dict(self.shared_uncertainties),
            "free_params": list(self.free_params),
            "results": [dict(result) for result in self.results],
            "trend": self.trend.to_dict(),
        }


def fit_global(
    datasets_by_run: Mapping[int, MuonDataset],
    recipe: FitRecipe,
    *,
    shared_params: Sequence[str],
    axis: ScanAxis[int],
    field_params: Sequence[str] = (),
    strategy: str = "joint",
) -> GlobalFitOutcome:
    """Fit one coupled objective with shared and per-run parameters.

    ``field_params`` are re-seeded from each run's recorded field and held for
    that run.  This is the key contract for LF decoupling triplets: the applied
    field differs, while amplitudes and dynamic parameters are fitted jointly.
    The runs are reported in *axis* order, and the free run-local parameters
    are tabulated against it as the fit's trend.
    """
    if len(datasets_by_run) < 2:
        raise ValueError("A simultaneous fit needs at least two reduced runs.")
    shared = list(dict.fromkeys(shared_params))
    field = list(dict.fromkeys(field_params))
    known = set(recipe.parameter_names)
    unknown = sorted((set(shared) | set(field)) - known)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {', '.join(unknown)}; {recipe.expression!r} has "
            f"{', '.join(recipe.parameter_names)}."
        )
    overlap = sorted(set(shared) & set(field))
    if overlap:
        raise ValueError(f"Field parameter(s) {', '.join(overlap)} cannot also be shared.")
    if not shared:
        raise ValueError("Name at least one fitted shared parameter.")

    model = recipe.model()
    runs = sorted(datasets_by_run, key=lambda run: (axis.values[run], run))
    datasets = [
        datasets_by_run[run] if recipe.rebin <= 1 else datasets_by_run[run].rebin(recipe.rebin)
        for run in runs
    ]
    initial = {}
    for dataset in datasets:
        parameters = recipe.parameter_set()
        seeds = seed_parameters(
            model,
            SeedContext(dataset=dataset, field_gauss=dataset.field),
        )
        for name, seed in seeds.items():
            if seed.run_bound and name not in recipe.pinned:
                parameters[name].value = seed.value
        for name in field:
            if dataset.field is None:
                raise ValueError(
                    f"Run {dataset.run_number} records no field for --field-param {name}."
                )
            parameters[name].value = float(dataset.field)
            parameters[name].fixed = True
        initial[int(dataset.run_number)] = parameters

    local = [name for name in recipe.parameter_names if name not in shared]
    results_by_run, fitted_shared = FitEngine().global_fit(
        datasets,
        model.function,
        shared,
        local,
        initial,
        t_min=recipe.t_min,
        t_max=recipe.t_max,
        strategy=strategy,
    )
    results = [
        {"run": run, "x": axis.values[run], **fit_result_summary(results_by_run[run])}
        for run in runs
    ]
    free_params = [
        name for name in recipe.free_parameter_names() if name not in shared and name not in field
    ]
    shared_values = {parameter.name: float(parameter.value) for parameter in fitted_shared}
    shared_uncertainties: dict[str, float] = {}
    for name in shared_values:
        for result in results:
            value = result["uncertainties"].get(name)
            if value is not None:
                shared_uncertainties[name] = float(value)
                break
    return GlobalFitOutcome(
        expression=recipe.expression,
        shared_params=shared,
        local_params=local,
        field_params=field,
        strategy=strategy,
        shared=shared_values,
        shared_uncertainties=shared_uncertainties,
        free_params=free_params,
        results=results,
        trend=build_trend_table(
            {str(result["run"]): result for result in results}, free_params, axis.name
        ),
    )


@dataclass(frozen=True)
class BatchOutcome:
    """One simultaneous fit per group, and the groups' shared parameters as a trend."""

    expression: str
    #: The shared parameters that were fitted — what the batch trend tabulates.
    free_params: list[str]
    #: Each group's fit, keyed by its member name, in batch-axis order.
    groups: dict[str, GlobalFitOutcome]
    trend: TrendTable

    def to_dict(self) -> dict[str, Any]:
        """Serialize the batch's own trend; each group serializes as its own fit."""
        return {
            "expression": self.expression,
            "order_key": self.trend.order_key,
            "free_params": list(self.free_params),
            "trend": self.trend.to_dict(),
        }


def fit_global_batch(
    groups: Mapping[str, Mapping[int, MuonDataset]],
    recipe: FitRecipe,
    *,
    shared_params: Sequence[str],
    axis: ScanAxis[int],
    batch_axis: ScanAxis[str],
    field_params: Sequence[str] = (),
    strategy: str = "joint",
) -> BatchOutcome:
    """:func:`fit_global` on every group, then its shared parameters against *batch_axis*.

    *groups* maps each member name to its runs; *axis* orders the runs within a
    group and *batch_axis* the groups. A row of the batch trend is one group:
    its fitted shared parameters, their errors, and every flag its runs raised.
    Raises :class:`ValueError` naming a group of fewer than two runs, as well
    as whatever :func:`fit_global` refuses.
    """
    small = [name for name, runs in groups.items() if len(runs) < 2]
    if small:
        raise ValueError(
            f"A simultaneous fit needs at least two runs per group; too few in {', '.join(small)}."
        )
    names = sorted(groups, key=lambda name: (batch_axis.values[name], name))
    outcomes = {
        name: fit_global(
            groups[name],
            recipe,
            shared_params=shared_params,
            axis=axis,
            field_params=field_params,
            strategy=strategy,
        )
        for name in names
    }
    free_params = [name for name in recipe.free_parameter_names() if name in shared_params]
    entries = {
        name: {
            "x": batch_axis.values[name],
            "parameters": outcome.shared,
            "uncertainties": outcome.shared_uncertainties,
            "quality_flags": sorted(
                {flag for result in outcome.results for flag in result["quality_flags"]}
            ),
        }
        for name, outcome in outcomes.items()
    }
    return BatchOutcome(
        expression=recipe.expression,
        free_params=free_params,
        groups=outcomes,
        trend=build_trend_table(entries, free_params, batch_axis.name),
    )


__all__ = ["BatchOutcome", "GlobalFitOutcome", "fit_global", "fit_global_batch"]
