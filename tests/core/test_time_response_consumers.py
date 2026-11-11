"""Authored response survives the model owners used by fits and replay."""

import json
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    build_candidate_templates,
    build_fit_wizard_recommendation_for_templates,
    build_null_baseline_templates,
    build_wizard_families,
    deserialize_fit_wizard_recommendation,
    fingerprint_spectrum,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_fit_wizard import build_global_fit_wizard_candidate_portfolio
from asymmetry.core.fitting.time_response import TimeResponse
from asymmetry.core.representation.series import FitSeries
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.series import fit_one


def causal_dataset(run=1):
    """Independent count-ratio exponential on a midpoint observation clock."""
    time = np.linspace(0, 4, 120)
    f, b = np.zeros_like(time), np.zeros_like(time)
    for birth in [-0.162, 0.162]:
        live = time >= birth
        age = time[live] - birth
        weight = np.exp(-age / MUON_LIFETIME_US)
        p = 0.19 * np.exp(-0.7 * age) + 0.015
        f[live] += weight * (1 + p)
        b[live] += weight * (1 - p)
    return MuonDataset(time, 100 * (f - b) / (f + b), np.full_like(time, 0.1), {"run_number": run})


def test_recipe_json_and_actual_refit():
    recipe = FitRecipe.from_expression("Exponential + Constant")
    model = recipe.model().with_time_response(TimeResponse("double_delta", 0.324))
    recipe = replace(recipe, model_payload=model.to_dict()).with_overrides(
        initial={"A_1": 18, "Lambda": 0.6, "A_bg": 1}
    )
    restored = FitRecipe.from_dict(json.loads(json.dumps(recipe.to_dict())))
    summary = fit_one(causal_dataset(), restored)
    assert summary["success"]
    assert summary["parameters"]["Lambda"] == pytest.approx(0.7, abs=1e-4)
    assert restored.model().time_response == model.time_response
    off = FitRecipe.from_expression("Exponential + Constant")
    assert "time_response" not in off.model_payload


def test_series_identity_distinguishes_consumed_response_only():
    model = CompositeModel(["Exponential"])
    series = FitSeries("a", "time_fb_asymmetry", canonical_model=model.to_dict())
    identity = series.recipe_identity()
    series.canonical_model = {
        **model.to_dict(),
        "time_response": {"mode": "off", "separation_us": "unused"},
    }
    assert series.recipe_identity() == identity
    series.canonical_model = model.with_time_response(TimeResponse("double_delta", 0.324)).to_dict()
    assert series.recipe_identity() != identity
    assert FitSeries.from_dict(series.to_dict()).recipe_identity() == series.recipe_identity()


def test_all_new_wizard_families_and_nulls_inherit_fixed_response():
    data = causal_dataset()
    model = CompositeModel(["Exponential", "Constant"]).with_time_response(
        TimeResponse("double_delta", 0.324)
    )
    fingerprint = fingerprint_spectrum(data)
    templates = list(build_candidate_templates(fingerprint, model))
    for family in build_wizard_families(fingerprint, model):
        templates.extend([family.stage1_rep, *family.stage1_extras, *family.stage2_members])
    templates.extend(build_null_baseline_templates(model.time_response))
    assert len(templates) > 10
    assert all(t.model.time_response == model.time_response for t in templates)
    portfolio = build_global_fit_wizard_candidate_portfolio([data, causal_dataset(2)], model)
    assert all(t.model.time_response == model.time_response for t in portfolio.templates)


def test_wizard_dense_curves_cache_and_recipe_keep_fitted_response():
    data = causal_dataset()
    model = CompositeModel(["Exponential", "Constant"]).with_time_response(
        TimeResponse("double_delta", 0.324)
    )
    template = CandidateTemplate(
        key="pulse_exp", title="Exponential", category="General", rationale="", model=model
    )
    report = build_fit_wizard_recommendation_for_templates(data, [template], variant_budget=1)
    restored = deserialize_fit_wizard_recommendation(
        json.loads(json.dumps(serialize_fit_wizard_recommendation(report)))
    )
    assessment = restored.assessments[0]
    assert assessment.fit_result.success
    assert assessment.template.model.time_response == model.time_response
    parameters = {p.name: p.value for p in assessment.fit_result.parameters}
    np.testing.assert_allclose(
        assessment.fitted_curve, model.function(assessment.fitted_time, **parameters)
    )
    recipe = FitRecipe.from_assessment(assessment, run_number=1, seed_field=None)
    assert recipe.model().time_response == model.time_response
    assert fit_one(data, recipe)["parameters"]["Lambda"] == pytest.approx(0.7, abs=1e-4)


def test_joint_fit_uses_each_serialized_series_response():
    from asymmetry.core.fitting.joint import JointSeriesProblem, SharedParameter, fit_joint
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(
        ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
    )
    restored = CompositeModel.from_dict(json.loads(json.dumps(model.to_dict())))
    problems = []
    for key, run in [("a", 1), ("b", 2)]:
        problems.append(
            JointSeriesProblem(
                key=key,
                datasets=[causal_dataset(run)],
                model_fn=restored.function,
                global_params=["Lambda"],
                local_params=["A_1", "A_bg"],
                initial_params={
                    run: ParameterSet(
                        [
                            Parameter("Lambda", 0.6, min=0),
                            Parameter("A_1", 18),
                            Parameter("A_bg", 1),
                        ]
                    )
                },
            )
        )
    result = fit_joint(
        problems, [SharedParameter("rate", {"a": "Lambda", "b": "Lambda"}, 0.6, min=0)]
    )
    assert result.success
    assert result.shared_parameters["rate"].value == pytest.approx(0.7, abs=1e-4)
    assert all(
        np.max(np.abs(r.residuals)) < 1e-3
        for results in result.series_results.values()
        for r in results.values()
    )


def test_wizard_build_identity_tracks_only_consumed_response():
    from asymmetry.core.fitting.fit_wizard import single_fit_build_signature

    off = single_fit_build_signature(None)
    assert single_fit_build_signature(None, time_response=TimeResponse("off", float("nan"))) == off
    active = single_fit_build_signature(None, time_response=TimeResponse("double_delta", 0.324))
    assert active != off
    assert active != single_fit_build_signature(
        None, time_response=TimeResponse("double_delta", 0.4)
    )
