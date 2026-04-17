"""Tests for staged global-search helpers."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import build_global_fit_wizard_recommendation
from asymmetry.core.fitting.global_search import (
    RelaxedFitProblem,
    RelaxedFitResult,
    SciPyRelaxedOptimizer,
    compile_legacy_structure,
    compile_structure_to_legacy_roles,
)
from asymmetry.core.fitting.global_search.heuristics import (
    allows_rate_first_localization,
    is_background_parameter,
    localisation_threshold_scale,
)
from asymmetry.core.fitting.global_search.moves import generate_search_moves
from asymmetry.core.fitting.global_search.proposal import extract_discrete_candidates
from asymmetry.core.fitting.global_search.types import DiscreteCandidate, SearchMoveType
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _dataset_for(
    run_number: int,
    *,
    field: float,
    model: CompositeModel,
    params: dict[str, float],
) -> MuonDataset:
    time = np.linspace(0.0, 8.0, 120)
    asymmetry = model.function(time, **params)
    error = np.full_like(time, 0.01)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": run_number,
            "run_label": str(run_number),
            "field": field,
            "temperature": 5.0,
        },
    )


def _template() -> CandidateTemplate:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    return CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )


def _base_parameter_sets(values_by_run: dict[int, float]) -> dict[int, ParameterSet]:
    base: dict[int, ParameterSet] = {}
    for run_number, lambda_value in values_by_run.items():
        base[run_number] = ParameterSet(
            [
                Parameter("A_1", value=0.2, min=0.0, max=1.0),
                Parameter("Lambda", value=lambda_value, min=0.0, max=2.0),
                Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
            ]
        )
    return base


def test_compile_legacy_structure_roundtrips_roles() -> None:
    structure = compile_legacy_structure(
        _template(),
        current_parameter_types={"A_1": "Global", "Lambda": "Local", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )

    global_names, local_names, fixed_names = compile_structure_to_legacy_roles(structure)

    assert global_names == ("A_1",)
    assert local_names == ("Lambda",)
    assert fixed_names == ("A_bg",)
    assert structure.signature() == structure.signature()


def test_compile_legacy_structure_can_treat_nonfixed_roles_as_hints() -> None:
    structure = compile_legacy_structure(
        _template(),
        current_parameter_types={"A_1": "Global", "Lambda": "Local", "A_bg": "Local"},
        current_values={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
        treat_nonfixed_roles_as_hints=True,
    )

    tie_modes = {
        tie.parameter_name: tie.mode.value
        for tie in structure.parameter_ties
    }

    assert tie_modes["A_1"] == "relaxed_shared"
    assert tie_modes["Lambda"] == "relaxed_shared"
    assert tie_modes["A_bg"] == "relaxed_shared"


def test_relaxed_optimizer_keeps_shared_parameter_nearly_shared() -> None:
    template = _template()
    structure = compile_legacy_structure(
        template,
        current_parameter_types={"A_1": "Global", "Lambda": "Global", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )
    base_by_run = _base_parameter_sets({101: 0.35, 102: 0.35, 103: 0.35, 104: 0.35})
    datasets = [
        _dataset_for(
            run_number=run_number,
            field=10.0 * index,
            model=template.model,
            params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        )
        for index, run_number in enumerate(sorted(base_by_run), start=1)
    ]
    result = SciPyRelaxedOptimizer().solve(
        RelaxedFitProblem(
            structure=structure,
            datasets=tuple(datasets),
            initial_params_by_run=base_by_run,
            penalty_weight=4.0,
        )
    )

    max_deviation = max(
        abs(run_values["Lambda"])
        for run_values in result.deviations_by_run.values()
    )
    assert result.success is True
    assert max_deviation < 1e-2


def test_relaxed_optimizer_continuation_freezes_nearly_shared_parameter() -> None:
    template = _template()
    lambda_values = {101: 0.35, 102: 0.351, 103: 0.349, 104: 0.35}
    structure = compile_legacy_structure(
        template,
        current_parameter_types={"A_1": "Global", "Lambda": "Local", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
        treat_nonfixed_roles_as_hints=True,
    )
    base_by_run = _base_parameter_sets(lambda_values)
    datasets = [
        _dataset_for(
            run_number=run_number,
            field=10.0 * index,
            model=template.model,
            params={"A_1": 0.2, "Lambda": lambda_value, "A_bg": 0.01},
        )
        for index, (run_number, lambda_value) in enumerate(sorted(lambda_values.items()), start=1)
    ]

    result = SciPyRelaxedOptimizer().solve(
        RelaxedFitProblem(
            structure=structure,
            datasets=tuple(datasets),
            initial_params_by_run=base_by_run,
            penalty_weight=4.0,
            penalty_schedule=(2.0, 4.0, 7.0),
            active_set_threshold=0.01,
        )
    )

    assert result.success is True
    assert "Lambda" in result.frozen_shared_names
    assert result.continuation_penalties == (2.0, 4.0, 7.0)
    assert len(result.continuation_objectives) == 3


def test_relaxed_to_discrete_extraction_localizes_varying_lambda() -> None:
    template = _template()
    structure = compile_legacy_structure(
        template,
        current_parameter_types={"A_1": "Global", "Lambda": "Global", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )
    lambda_values = {201: 0.15, 202: 0.25, 203: 0.55, 204: 0.90}
    base_by_run = _base_parameter_sets(lambda_values)
    datasets = [
        _dataset_for(
            run_number=run_number,
            field=20.0 * index,
            model=template.model,
            params={"A_1": 0.2, "Lambda": lambda_value, "A_bg": 0.01},
        )
        for index, (run_number, lambda_value) in enumerate(sorted(lambda_values.items()), start=1)
    ]
    result = SciPyRelaxedOptimizer().solve(
        RelaxedFitProblem(
            structure=structure,
            datasets=tuple(datasets),
            initial_params_by_run=base_by_run,
            penalty_weight=4.0,
        )
    )

    candidates = extract_discrete_candidates(result, structure)

    assert candidates
    global_names, local_names, fixed_names = compile_structure_to_legacy_roles(
        candidates[0].structure
    )
    assert "Lambda" in local_names
    assert "A_1" in global_names
    assert "A_bg" in fixed_names


def test_staged_strategy_preserves_public_recommendation_shape() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    lambdas = [0.15, 0.25, 0.55, 0.9]
    datasets = [
        _dataset_for(
            run_number=300 + idx,
            field=50.0 * idx,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        metric=SelectionMetric.BIC,
        search_strategy="staged_v1",
    )

    assessment = recommendation.recommended_assessment

    assert recommendation.recommended_key is not None
    assert assessment is not None
    assert "Lambda" in assessment.local_param_names
    assert isinstance(assessment.parameter_recommendations, tuple)


def test_staged_v2_strategy_preserves_public_recommendation_shape() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    lambdas = [0.15, 0.25, 0.55, 0.9]
    datasets = [
        _dataset_for(
            run_number=400 + idx,
            field=75.0 * idx,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        metric=SelectionMetric.BIC,
        search_strategy="staged_v2",
    )

    assessment = recommendation.recommended_assessment

    assert recommendation.recommended_key is not None
    assert assessment is not None
    assert "Lambda" in assessment.local_param_names
    assert isinstance(assessment.parameter_recommendations, tuple)


def test_move_generation_only_splits_ambiguous_shared_parameters() -> None:
    structure = compile_legacy_structure(
        _template(),
        current_parameter_types={"A_1": "Global", "Lambda": "Global", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )
    candidate = DiscreteCandidate(
        structure=structure,
        source_relaxed_signature=(),
        initial_params_by_run=_base_parameter_sets({101: 0.35, 102: 0.40}),
        ambiguous_param_names=("Lambda",),
    )

    moves = generate_search_moves(candidate, full_structure=structure)

    split_targets = {
        move.target for move in moves if move.move_type == SearchMoveType.SPLIT_TO_LOCAL
    }
    assert split_targets == {"Lambda"}


def test_extract_discrete_candidates_preserves_relaxed_role_metadata() -> None:
    template = _template()
    structure = compile_legacy_structure(
        template,
        current_parameter_types={"A_1": "Global", "Lambda": "Global", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )
    lambda_values = {201: 0.15, 202: 0.25, 203: 0.55, 204: 0.90}
    base_by_run = _base_parameter_sets(lambda_values)
    datasets = [
        _dataset_for(
            run_number=run_number,
            field=20.0 * index,
            model=template.model,
            params={"A_1": 0.2, "Lambda": lambda_value, "A_bg": 0.01},
        )
        for index, (run_number, lambda_value) in enumerate(sorted(lambda_values.items()), start=1)
    ]
    result = SciPyRelaxedOptimizer().solve(
        RelaxedFitProblem(
            structure=structure,
            datasets=tuple(datasets),
            initial_params_by_run=base_by_run,
            penalty_weight=4.0,
        )
    )

    candidate = extract_discrete_candidates(result, structure)[0]

    assert "Lambda" in candidate.relaxed_local_names


def test_staged_heuristics_bias_amplitudes_toward_shared_roles() -> None:
    assert localisation_threshold_scale("Lambda") == 1.0
    assert localisation_threshold_scale("sigma") == 1.0
    assert localisation_threshold_scale("A_1") > localisation_threshold_scale("Lambda")
    assert localisation_threshold_scale("A_bg") > localisation_threshold_scale("A_1")
    assert allows_rate_first_localization("Lambda") is True
    assert allows_rate_first_localization("sigma") is True
    assert allows_rate_first_localization("A_1") is False
    assert allows_rate_first_localization("A_bg") is False
    assert is_background_parameter("A_bg") is True


def test_move_generation_does_not_split_ambiguous_amplitudes_by_default() -> None:
    structure = compile_legacy_structure(
        _template(),
        current_parameter_types={"A_1": "Global", "Lambda": "Global", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )
    candidate = DiscreteCandidate(
        structure=structure,
        source_relaxed_signature=(),
        initial_params_by_run=_base_parameter_sets({101: 0.35, 102: 0.40}),
        ambiguous_param_names=("A_1", "Lambda"),
    )

    moves = generate_search_moves(candidate, full_structure=structure)

    split_targets = {
        move.target for move in moves if move.move_type == SearchMoveType.SPLIT_TO_LOCAL
    }
    assert split_targets == {"Lambda"}


def test_move_generation_only_merges_ambiguous_local_parameters() -> None:
    structure = compile_legacy_structure(
        _template(),
        current_parameter_types={"A_1": "Global", "Lambda": "Local", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )
    candidate = DiscreteCandidate(
        structure=structure,
        source_relaxed_signature=(),
        initial_params_by_run=_base_parameter_sets({101: 0.35, 102: 0.40}),
        ambiguous_param_names=(),
    )

    moves = generate_search_moves(candidate, full_structure=structure)

    merge_targets = {
        move.target for move in moves if move.move_type == SearchMoveType.MERGE_TO_SHARED
    }
    assert merge_targets == set()

    ambiguous_candidate = DiscreteCandidate(
        structure=structure,
        source_relaxed_signature=(),
        initial_params_by_run=_base_parameter_sets({101: 0.35, 102: 0.40}),
        ambiguous_param_names=("Lambda",),
    )

    ambiguous_moves = generate_search_moves(ambiguous_candidate, full_structure=structure)

    ambiguous_merge_targets = {
        move.target for move in ambiguous_moves if move.move_type == SearchMoveType.MERGE_TO_SHARED
    }
    assert ambiguous_merge_targets == {"Lambda"}


def test_rate_first_extraction_keeps_amplitudes_and_background_shared() -> None:
    structure = compile_legacy_structure(
        _template(),
        current_parameter_types={},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
    )
    result = RelaxedFitResult(
        success=True,
        objective_value=1.0,
        base_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        local_values_by_run={
            1: {"A_1": 0.35, "Lambda": 0.20, "A_bg": 0.03},
            2: {"A_1": 0.10, "Lambda": 0.70, "A_bg": -0.04},
        },
        deviations_by_run={
            1: {"A_1": 0.15, "Lambda": -0.15, "A_bg": 0.02},
            2: {"A_1": -0.10, "Lambda": 0.35, "A_bg": -0.05},
        },
        activity_weights={},
        seed_params_by_run={
            1: ParameterSet(
                [
                    Parameter("A_1", value=0.35, min=0.0, max=1.0),
                    Parameter("Lambda", value=0.20, min=0.0, max=2.0),
                    Parameter("A_bg", value=0.03, min=-0.2, max=0.2),
                ]
            ),
            2: ParameterSet(
                [
                    Parameter("A_1", value=0.10, min=0.0, max=1.0),
                    Parameter("Lambda", value=0.70, min=0.0, max=2.0),
                    Parameter("A_bg", value=-0.04, min=-0.2, max=0.2),
                ]
            ),
        },
    )

    candidate = extract_discrete_candidates(result, structure)[0]
    global_names, local_names, _fixed_names = compile_structure_to_legacy_roles(candidate.structure)

    assert "Lambda" in local_names
    assert "A_1" in global_names
    assert "A_bg" in global_names


def test_extract_discrete_candidates_respects_frozen_shared_names() -> None:
    structure = compile_legacy_structure(
        _template(),
        current_parameter_types={"A_1": "Global", "Lambda": "Local", "A_bg": "Fixed"},
        current_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        parameter_bounds={"A_1": (0.0, 1.0), "Lambda": (0.0, 2.0), "A_bg": (-0.2, 0.2)},
        treat_nonfixed_roles_as_hints=True,
    )
    result = RelaxedFitResult(
        success=True,
        objective_value=1.0,
        base_values={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        local_values_by_run={
            101: {"A_1": 0.2, "Lambda": 0.20, "A_bg": 0.01},
            102: {"A_1": 0.2, "Lambda": 0.55, "A_bg": 0.01},
        },
        deviations_by_run={101: {"Lambda": -0.15}, 102: {"Lambda": 0.20}},
        activity_weights={},
        seed_params_by_run=_base_parameter_sets({101: 0.20, 102: 0.55}),
        frozen_shared_names=("Lambda",),
        continuation_penalties=(2.0, 4.0),
        continuation_objectives=(1.2, 1.0),
    )

    candidates = extract_discrete_candidates(result, structure, max_alternates=3)

    assert candidates
    global_names, local_names, fixed_names = compile_structure_to_legacy_roles(
        candidates[0].structure
    )
    assert "Lambda" in global_names
    assert "Lambda" not in local_names
    assert "A_bg" in fixed_names
