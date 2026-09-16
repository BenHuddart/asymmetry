"""True simultaneous fitting for a group of reduced runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fitting.seeding import SeedContext, seed_parameters
from asymmetry.core.workflow.recipe import FitRecipe


@dataclass(frozen=True)
class GlobalFitOutcome:
    expression: str
    shared_params: list[str]
    local_params: list[str]
    field_params: list[str]
    strategy: str
    shared: dict[str, float]
    shared_uncertainties: dict[str, float]
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "shared_params": list(self.shared_params),
            "local_params": list(self.local_params),
            "field_params": list(self.field_params),
            "strategy": self.strategy,
            "shared": dict(self.shared),
            "shared_uncertainties": dict(self.shared_uncertainties),
            "results": [dict(result) for result in self.results],
        }


def fit_global(
    datasets_by_run: Mapping[int, MuonDataset],
    recipe: FitRecipe,
    *,
    shared_params: Sequence[str],
    field_params: Sequence[str] = (),
    strategy: str = "joint",
) -> GlobalFitOutcome:
    """Fit one coupled objective with shared and per-run parameters.

    ``field_params`` are re-seeded from each run's recorded field and held for
    that run.  This is the key contract for LF decoupling triplets: the applied
    field differs, while amplitudes and dynamic parameters are fitted jointly.
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
    runs = sorted(datasets_by_run)
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
    results = [{"run": run, **fit_result_summary(results_by_run[run])} for run in runs]
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
        results=results,
    )


__all__ = ["GlobalFitOutcome", "fit_global"]
