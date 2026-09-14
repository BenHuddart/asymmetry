"""Tests for :mod:`asymmetry.core.workflow.recipe`."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import CandidateAssessment, CandidateTemplate
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.seeding import record_scale_estimate
from asymmetry.core.workflow.recipe import FitRecipe, RecipeParameter
from asymmetry.core.workflow.workdir import WorkDir

_EXPRESSION = "Exponential + Constant"


def _assessment() -> CandidateAssessment:
    """A wizard assessment whose fit pinned one parameter and bounded another."""
    model = CompositeModel.from_expression(_EXPRESSION)
    parameters = ParameterSet(
        [
            Parameter(name="A_1", value=18.5, min=0.0, max=50.0),
            Parameter(name="Lambda", value=0.31, min=0.0, max=math.inf),
            Parameter(name="A_bg", value=0.0, min=-5.0, max=5.0, fixed=True),
        ]
    )
    empty = np.array([], dtype=np.float64)
    return CandidateAssessment(
        template=CandidateTemplate(
            key="exp_constant",
            title="Exponential + Constant",
            category="General",
            rationale="baseline",
            model=model,
        ),
        fit_result=FitResult(success=True, parameters=parameters),
        aic=10.0,
        aicc=10.5,
        bic=12.0,
        selected_score=10.5,
        residual_rms=0.1,
        runs_z_score=0.2,
        max_abs_autocorrelation=0.05,
        residual_fft_peak_snr=1.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
    )


# -- construction -----------------------------------------------------------


def test_from_expression_seeds_every_parameter_of_the_model() -> None:
    recipe = FitRecipe.from_expression(_EXPRESSION)
    assert recipe.expression == _EXPRESSION
    assert recipe.parameter_names == recipe.model().param_names
    assert recipe.source == {"user": True}
    # The relaxation rate carries the registry's zero floor, so a fit cannot
    # wander to a negative rate from a recipe nobody edited.
    rate = next(p for p in recipe.parameters if p.name == "Lambda")
    assert rate.min == 0.0


def test_from_expression_with_a_dataset_seeds_the_records_own_scale(
    reduced_workdir: WorkDir, first_scan_run: int
) -> None:
    dataset = reduced_workdir.reduced(first_scan_run)
    recipe = FitRecipe.from_expression(_EXPRESSION, dataset=dataset)

    # The amplitude and background start where the record's own scale says, not
    # at the component's static default — the same reading every fit surface
    # seeds from.
    amplitude, tail = record_scale_estimate(dataset.time, dataset.asymmetry)
    by_name = {parameter.name: parameter for parameter in recipe.parameters}
    assert by_name["A_1"].value == pytest.approx(amplitude)
    assert by_name["A_bg"].value == pytest.approx(tail)

    without = {p.name: p.value for p in FitRecipe.from_expression(_EXPRESSION).parameters}
    assert without["A_1"] != pytest.approx(amplitude)


def test_from_assessment_keeps_the_fitted_values_bounds_and_fixed_flags() -> None:
    recipe = FitRecipe.from_assessment(_assessment(), run_number=102)

    assert recipe.source == {"wizard_run": 102, "template_key": "exp_constant"}
    by_name = {parameter.name: parameter for parameter in recipe.parameters}
    assert by_name["A_1"].value == pytest.approx(18.5)
    assert by_name["A_1"].max == pytest.approx(50.0)
    assert by_name["Lambda"].max == math.inf
    assert by_name["A_bg"].fixed is True
    assert by_name["A_bg"].min == pytest.approx(-5.0)
    assert recipe.free_parameter_names() == ["A_1", "Lambda"]


def test_a_recipe_rejects_a_rebin_below_one_and_an_inverted_window() -> None:
    recipe = FitRecipe.from_expression(_EXPRESSION)
    with pytest.raises(ValueError):
        FitRecipe.from_expression(_EXPRESSION, rebin=0)
    with pytest.raises(ValueError):
        recipe.with_window(t_min=8.0, t_max=1.0)


# -- serialisation ----------------------------------------------------------


def test_a_recipe_round_trips_through_its_dict() -> None:
    recipe = FitRecipe.from_assessment(_assessment(), run_number=102).with_window(
        t_min=0.2, t_max=12.0
    )
    restored = FitRecipe.from_dict(recipe.to_dict())

    assert restored == recipe
    assert restored.model().param_names == recipe.model().param_names


def test_a_recipe_writes_standard_json_with_no_infinity_token(tmp_path: Path) -> None:
    # An infinite bound is JSON ``null``: Python would write ``Infinity``, which
    # only Python reads back, and a recipe is meant to be read by other tools.
    recipe = FitRecipe.from_assessment(_assessment(), run_number=102)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")

    assert "Infinity" not in path.read_text(encoding="utf-8")
    stored = json.loads(path.read_text(encoding="utf-8"))
    rate = next(entry for entry in stored["parameters"] if entry["name"] == "Lambda")
    assert rate["max"] is None
    assert FitRecipe.from_dict(stored).parameters[1].max == math.inf


def test_parameter_set_is_a_fresh_object_every_call() -> None:
    recipe = FitRecipe.from_expression(_EXPRESSION)
    first, second = recipe.parameter_set(), recipe.parameter_set()
    first["Lambda"].value = 99.0

    assert second["Lambda"].value != 99.0
    assert recipe.parameters[1].value != 99.0


# -- editing ----------------------------------------------------------------


def test_with_overrides_pins_and_releases_named_parameters() -> None:
    recipe = FitRecipe.from_assessment(_assessment(), run_number=102)

    edited = recipe.with_overrides(fix={"Lambda": 0.25}, free=["A_bg"])

    by_name = {parameter.name: parameter for parameter in edited.parameters}
    assert by_name["Lambda"] == RecipeParameter("Lambda", 0.25, 0.0, math.inf, fixed=True)
    assert by_name["A_bg"].fixed is False
    assert edited.free_parameter_names() == ["A_1", "A_bg"]
    # The original is untouched — a recipe is a value, not a mutable form.
    assert recipe.free_parameter_names() == ["A_1", "Lambda"]


def test_with_overrides_rejects_a_name_the_model_does_not_have() -> None:
    recipe = FitRecipe.from_expression(_EXPRESSION)
    with pytest.raises(KeyError, match="Nope"):
        recipe.with_overrides(fix={"Nope": 1.0})
    with pytest.raises(KeyError, match="Nope"):
        recipe.with_overrides(free=["Nope"])
