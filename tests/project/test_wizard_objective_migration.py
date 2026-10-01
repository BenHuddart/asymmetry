"""Tests for the v23->v24 global wizard migration (docs/plans/global-wizard-trend-objective.md).

A v23 project stores Global Fit Wizard recommendations with no objective, no
ladder rung on a candidate, and ``total_variation``/``roughness`` on every
parameter recommendation. v24 marks each one statistical, gives every candidate
``rung: null`` and drops the two trace diagnostics, so the runtime reads one
shape strictly.
"""

from __future__ import annotations

import json

import numpy as np

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    GlobalParameterRecommendation,
    deserialize_global_fit_wizard_recommendation,
    serialize_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.trend_objective import (
    CandidateRung,
    SelectionObjective,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.trend_quality import CandidateTrend, PassDiagnostic, TraceQuality
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, migrate_to_current, validate

_RUNS = (11, 12, 13)


def _assessment(
    model: CompositeModel, shared: tuple[str, ...], **extra
) -> GlobalCandidateAssessment:
    local = tuple(name for name in model.param_names if name not in shared)
    return GlobalCandidateAssessment(
        template=CandidateTemplate(
            key="two_lines", title="Two lines", category="General", rationale="", model=model
        ),
        fit_results_by_run={
            run: FitResult(
                success=True,
                chi_squared=10.0,
                reduced_chi_squared=1.0,
                parameters=ParameterSet(
                    [Parameter(name, 1.0 + 0.1 * index) for name in model.param_names]
                ),
                uncertainties=dict.fromkeys(model.param_names, 0.05),
            )
            for index, run in enumerate(_RUNS)
        },
        global_parameters=ParameterSet([Parameter(name, 1.0) for name in shared]),
        global_param_names=shared,
        local_param_names=local,
        fixed_param_names=(),
        run_diagnostics=(),
        series_warnings=(),
        aic=1.0,
        aicc=2.0,
        bic=3.0,
        selected_score=2.0,
        fitted_curves_by_run={run: (np.arange(3.0), np.ones(3)) for run in _RUNS},
        component_curves_by_run={},
        assessment_key=f"two_lines|g={','.join(shared)}",
        **{"parameter_recommendations": (), **extra},
    )


def _recommendation(
    *assessments: GlobalCandidateAssessment, **extra
) -> GlobalFitWizardRecommendation:
    return GlobalFitWizardRecommendation(
        series_axis_key="temperature",
        series_axis_label="Temperature (K)",
        mixed_axes_warning=None,
        fingerprints_by_run={},
        dataset_order=_RUNS,
        templates=tuple(assessment.template for assessment in assessments),
        assessments=assessments,
        metric=SelectionMetric.AICC,
        recommended_key=assessments[0].selection_key,
        comparable_keys=(),
        summary="",
        **extra,
    )


def _as_v23(payload: dict) -> dict:
    """The payload as schema v23 wrote it."""

    def assessment(entry: dict) -> dict:
        legacy = {key: value for key, value in entry.items() if key != "rung"}
        legacy["parameter_recommendations"] = [
            {**recommendation, "total_variation": 0.4, "roughness": 0.2}
            for recommendation in entry["parameter_recommendations"]
        ]
        return legacy

    legacy = {key: value for key, value in payload.items() if key != "objective"}
    legacy["assessments"] = [assessment(entry) for entry in payload["assessments"]]
    legacy["phase_assessments"] = [
        {**entry, "assessment": assessment(entry["assessment"])}
        for entry in payload["phase_assessments"]
    ]
    return legacy


def test_v23_recommendation_migrates_to_a_statistical_one_without_rungs():
    model = CompositeModel.from_expression("Exponential + Constant")
    candidate = _assessment(
        model,
        ("A_1", "A_bg"),
        parameter_recommendations=(
            GlobalParameterRecommendation(
                name="Lambda",
                recommended_role="Local",
                global_score=5.0,
                local_score=1.0,
                score_delta=4.0,
                rationale="varies",
            ),
        ),
    )
    stored = _recommendation(
        candidate,
        phase_assessments={(1, 0): candidate},
        objective=SelectionObjective.STATISTICAL,
    )
    current = json.loads(json.dumps(serialize_global_fit_wizard_recommendation(stored)))
    project = {
        "schema_version": 23,
        "datasets": [],
        "multi_group_fit_state": {
            "batch": {
                "wizard_state_by_run_set": [
                    {
                        "signature": {"run_numbers": list(_RUNS)},
                        "recommendation": _as_v23(current),
                        "log_text": "",
                    }
                ]
            }
        },
    }

    migrated = migrate_to_current(project)
    validate(migrated)

    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION == 24
    cache = migrated["multi_group_fit_state"]["batch"]["wizard_state_by_run_set"][0]
    assert cache["recommendation"] == current
    restored = deserialize_global_fit_wizard_recommendation(cache["recommendation"])
    assert restored.objective is SelectionObjective.STATISTICAL
    assert restored.assessments[0].rung is None
    assert restored.phase_assessments[(1, 0)].rung is None
    assert restored.assessments[0].parameter_recommendations == candidate.parameter_recommendations
    # Re-migrating a current project is a no-op.
    assert migrate_to_current(migrated) == migrated


def test_trend_recommendation_round_trips_with_its_rungs():
    """A fraction-form rung keeps its own model, and an exempt run its exemption."""
    grouped = CompositeModel.from_expression("(Exponential + Gaussian){frac} + Constant")
    total = next(name for name in grouped.param_names if name.startswith("A"))
    local = tuple(name for name in grouped.param_names if name != total)
    rung = CandidateRung(
        exempt_runs=(12,),
        series_cost=0.7,
        run_costs={11: 0.1, 12: 3.5, 13: -0.2},
        trend=CandidateTrend(
            {name: TraceQuality(determined=1.0, signal=0.8, zigzag=0.25) for name in local}
        ),
        hard_to_justify=(local[0],),
        pass_disagreements={
            local[0]: PassDiagnostic(
                zigzag_axis_order=0.5, zigzag_run_order=0.0, passes=((11, 13), (12,))
            )
        },
        preselected=True,
        amplitude_unshareable_runs=(11,),
        all_local_chi2r=1.02,
    )
    candidate = _assessment(grouped, (total,), rung=rung)
    stored = _recommendation(candidate, objective=SelectionObjective.TREND)

    restored = deserialize_global_fit_wizard_recommendation(
        json.loads(json.dumps(serialize_global_fit_wizard_recommendation(stored, compact=True)))
    )

    assert restored.objective is SelectionObjective.TREND
    (assessment,) = restored.assessments
    assert assessment.rung == rung
    assert assessment.template.model.param_names == grouped.param_names
    assert assessment.template.model.fraction_groups == grouped.fraction_groups
    assert assessment.global_param_names == (total,)
    assert assessment.exemptions == {total: (12,)}
    assert assessment.parameter_count == candidate.parameter_count == 1 + 1 + len(local) * 3
