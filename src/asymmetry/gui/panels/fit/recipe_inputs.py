"""One series' recipe → the engine inputs a coupled fit takes.

A :class:`~asymmetry.core.representation.series.FitSeries` recipe already says
everything a coupled fit needs to know about that series: which parameter is
Global, Local, Fixed or File, each one's seed and limits, and the fit window
(D2 of ``docs/plans/series-workflow.md``). Turning that into the engine's
argument shape — the two name lists plus one :class:`ParameterSet` per run —
used to live inline in ``GlobalFitTab._run_global_fit``, which is fine while
the Batch tab is the only thing that launches a coupled fit. A joint fit runs
several series at once (``docs/plans/joint-fit.md`` D6) and must resolve each
member's recipe *exactly* as the Batch tab resolves it when running that series
alone, or the same series would fit differently inside and outside the joint
fit. So the step lives here, once, and both callers go through it.

The one thing this module does **not** do is per-run seed precedence: the Batch
tab's inherited single-fit seeds and its Initial-values dialog are live form
state, not recipe, and D6 excludes them from a joint run. The Batch tab passes
them in as *values_by_run*; the joint window passes nothing and gets the recipe
values.

Qt-free itself; the file-pinned lookup and the bounds-string split are imported
from the fit tabs' shared base so there is one spelling of each.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
    split_parameter_name,
)
from asymmetry.core.representation.series import normalise_recipe
from asymmetry.gui.panels.fit.tab_base import (
    _get_file_value_for_parameter,
    _split_bounds_text,
)

__all__ = ["RecipeEngineInputs", "build_recipe_engine_inputs"]


@dataclass(frozen=True)
class RecipeEngineInputs:
    """What a coupled fit engine takes for one series.

    ``global_params``/``local_params`` are the engine's partition; Fixed and
    File rows appear in neither, because a pinned parameter has no column —
    it reaches the engine only as a ``fixed=True`` entry of every run's
    :class:`ParameterSet`. The two pinned lists are carried anyway so a caller
    can tell *why* a name is absent (the Batch tab reports the free-Global
    count to decide between the chained and the tied path).
    """

    global_params: list[str]
    local_params: list[str]
    fixed_params: list[str]
    file_params: list[str]
    #: ``{run number: ParameterSet}`` — seeds, limits and pins, per run.
    initial_params: dict[int, ParameterSet]
    t_min: float | None
    t_max: float | None


def _bound(text: str, unbounded: float) -> float:
    """One limit as a float; an unparseable or blank limit means "no bound".

    Mirrors ``tab_base._read_bounds_cells``: a half-typed or empty limit is a
    real "unbounded on this side" case, not a reason to refuse the fit.
    """
    try:
        return float(text)
    except ValueError:
        return unbounded


def build_recipe_engine_inputs(
    recipe: Mapping[str, Any],
    model: CompositeModel,
    datasets: Sequence[MuonDataset],
    *,
    values_by_run: Mapping[int, Mapping[str, float]] | None = None,
) -> RecipeEngineInputs:
    """Resolve *recipe* against *model* and *datasets* into engine inputs.

    Every model parameter gets a row from the recipe (a recipe is recorded from
    the same table that built the model, so a missing row is a corrupt pairing
    and raises rather than silently fitting a default). A File row is pinned to
    each dataset's own metadata value, so the same recipe gives run 1001 its
    field and run 1002 its own.

    *values_by_run* overrides the recipe's seed for the runs and names it
    carries; anything it omits keeps the recipe value. A File row's pinned
    value still wins over both — a measured quantity is not a seed.
    """
    recipe = normalise_recipe(dict(recipe))
    rows = {str(row["name"]): row for row in recipe["parameters"]}

    global_params: list[str] = []
    local_params: list[str] = []
    fixed_params: list[str] = []
    file_params: list[str] = []
    for row in recipe["parameters"]:
        name = str(row["name"])
        role = str(row["type"])
        if role == "Global":
            global_params.append(name)
        elif role == "Local":
            local_params.append(name)
        elif role == "File":
            file_params.append(name)
        else:  # Fixed — and anything unrecognised, as the Batch tab reads it
            fixed_params.append(name)
    pinned = set(fixed_params) | set(file_params)
    file_set = set(file_params)

    overrides = values_by_run or {}
    initial_params: dict[int, ParameterSet] = {}
    for dataset in datasets:
        run_number = int(dataset.run_number)
        run_values = overrides.get(run_number, {})
        params = ParameterSet()
        for pname in model.param_names:
            row = rows[pname]
            minimum, maximum = _split_bounds_text(row["bounds"])
            value = float(run_values.get(pname, row["value"]))
            if pname in file_set:
                base_name, _index = split_parameter_name(pname)
                file_value = _get_file_value_for_parameter(dataset, base_name)
                if file_value is not None:
                    value = file_value
            params.add(
                Parameter(
                    name=pname,
                    value=value,
                    min=_bound(minimum, -float("inf")),
                    max=_bound(maximum, float("inf")),
                    fixed=pname in pinned,
                )
            )
        initial_params[run_number] = params

    window = recipe["fit_range"]
    return RecipeEngineInputs(
        global_params=global_params,
        local_params=local_params,
        fixed_params=fixed_params,
        file_params=file_params,
        initial_params=initial_params,
        t_min=window["min"],
        t_max=window["max"],
    )
