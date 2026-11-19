"""The persisted series recipe resolves identically for Batch and joint fits."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.recipe_inputs import build_recipe_engine_inputs


def test_recipe_roles_bounds_and_file_precedence():
    model = CompositeModel.from_expression("Oscillatory + Constant")
    names = model.param_names
    roles = ["Global", "Local", "Fixed", "File"]
    recipe = {
        "parameters": [
            {"name": name, "value": i + 1.0, "type": role, "bounds": bounds}
            for i, (name, role, bounds) in enumerate(
                zip(names, roles, ["0, 9", "bad, 8", "-2", ""])
            )
        ],
        "fit_range": {"min": 0.2, "max": 6.0},
    }
    datasets = [
        MuonDataset(
            time=np.arange(4.0),
            asymmetry=np.zeros(4),
            error=np.ones(4),
            run=Run(run_number=n, histograms=[]),
        )
        for n in (11, 12)
    ]
    inputs = build_recipe_engine_inputs(
        recipe, model, datasets, values_by_run={11: {names[0]: 7.0}}
    )
    assert inputs.global_params == [names[0]]
    assert inputs.local_params == [names[1]]
    assert inputs.fixed_params == [names[2]]
    assert inputs.file_params == [names[3]]
    assert inputs.initial_params[11][names[0]].value == 7.0
    assert inputs.initial_params[12][names[0]].value == 1.0
    assert inputs.initial_params[11][names[1]].min == -float("inf")
    assert inputs.initial_params[11][names[1]].max == 8.0
    assert inputs.initial_params[11][names[2]].max == float("inf")
    assert inputs.initial_params[11][names[2]].fixed
    assert inputs.initial_params[11][names[3]].fixed
    assert (inputs.t_min, inputs.t_max) == (0.2, 6.0)


def test_file_field_wins_over_explicit_values():
    model = CompositeModel.from_expression("OscillatoryField")
    recipe = {
        "parameters": [
            {"name": name, "value": 1.0, "type": "File" if name == "field" else "Fixed"}
            for name in model.param_names
        ]
    }
    dataset = MuonDataset(
        time=np.arange(4.0),
        asymmetry=np.zeros(4),
        error=np.ones(4),
        metadata={"field": 75.0},
        run=Run(run_number=3, histograms=[], metadata={"field": 90.0}),
    )
    inputs = build_recipe_engine_inputs(
        recipe, model, [dataset], values_by_run={3: {"field": 30.0}}
    )
    assert inputs.initial_params[3]["field"].value == 90.0
    assert inputs.initial_params[3]["field"].fixed


def test_missing_model_row_is_rejected():
    dataset = MuonDataset(time=np.arange(4.0), asymmetry=np.zeros(4), error=np.ones(4))
    with pytest.raises(KeyError):
        build_recipe_engine_inputs({}, CompositeModel.from_expression("Exponential"), [dataset])
