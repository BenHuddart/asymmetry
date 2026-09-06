"""Tests for the partitioned global-fit recommendation.

A temperature series that crosses a transition is not described by one model.
This file pins the whole answer to that, end to end on a planted two-phase
series: screening reads a partition path off the completed per-run table with
the elbow at the planted break, and asking for that ``partition_k`` fits each
phase on its own and verifies the elbow against its neighbours in ``k`` *and* in
position.

The end-to-end tests carry real fits and are marked ``integration``; everything
that is pure bookkeeping — the transitions summary, serialisation, merge, rerank,
argument validation — runs in the standard tier.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import asymmetry.core.fitting.global_fit_wizard as global_fit_wizard_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    SelectionMetric,
    _multiplet_model,
    build_fit_wizard_recommendation_for_templates,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalFitWizardRecommendation,
    build_global_fit_wizard_recommendation,
    build_global_fit_wizard_screening_recommendation,
    deserialize_global_fit_wizard_recommendation,
    merge_global_fit_wizard_recommendations,
    rerank_global_fit_wizard_recommendation,
    serialize_global_fit_wizard_recommendation,
    transitions_summary,
)
from asymmetry.core.fitting.global_search.partition import (
    PartitionPath,
    PartitionSolution,
    Segment,
)

# --------------------------------------------------------------------------- #
# The planted two-phase series
# --------------------------------------------------------------------------- #

EXPONENTIAL = CompositeModel(["Exponential", "Constant"], operators=["+"])
GAUSSIAN = CompositeModel(["Gaussian", "Constant"], operators=["+"])

EXPONENTIAL_TEMPLATE = CandidateTemplate(
    key="exp_constant",
    title="Exponential + Constant",
    category="General",
    rationale="test",
    model=EXPONENTIAL,
)
GAUSSIAN_TEMPLATE = CandidateTemplate(
    key="gauss_constant",
    title="Gaussian + Constant",
    category="General",
    rationale="test",
    model=GAUSSIAN,
)
TEMPLATES = (EXPONENTIAL_TEMPLATE, GAUSSIAN_TEMPLATE)

#: The break is planted between the fifth and sixth run, i.e. between 5 K and
#: 10 K, so the boundary estimate is 7.5 ± 2.5 K.
PLANTED_BREAK = 5
PLANTED_BOUNDARY = (7.5, 2.5)


def _dataset(
    run_number: int,
    model: CompositeModel,
    params: dict[str, float],
    *,
    temperature: float,
    n_points: int = 300,
) -> MuonDataset:
    time = np.linspace(0.0, 8.0, n_points)
    rng = np.random.default_rng(run_number)
    return MuonDataset(
        time=time,
        asymmetry=model.function(time, **params) + rng.normal(scale=0.002, size=n_points),
        error=np.full_like(time, 0.002),
        metadata={
            "run_number": run_number,
            "field": 0.0,
            "temperature": float(temperature),
            "run_label": str(run_number),
        },
    )


def _two_phase_series() -> list[MuonDataset]:
    """Five exponential runs below the transition, five gaussian runs above it.

    Within each phase the amplitude and background are shared and the relaxation
    parameter scans, which is the role structure the search has to recover.
    """
    low = [
        _dataset(
            800 + index,
            EXPONENTIAL,
            {"A_1": 0.2, "Lambda": 0.4 + 0.02 * index, "A_bg": 0.01},
            temperature=1.0 + index,
        )
        for index in range(PLANTED_BREAK)
    ]
    high = [
        _dataset(
            900 + index,
            GAUSSIAN,
            {"A_1": 0.2, "sigma": 0.5 + 0.02 * index, "A_bg": 0.01},
            temperature=10.0 + index,
        )
        for index in range(PLANTED_BREAK)
    ]
    return low + high


def _single_fit_table(datasets):
    """A completed per-run × per-template table, without running phase 1.

    Phase 1's job is to *find* the alphabet; here the alphabet is planted, so the
    two templates are scored on every run directly. The rebin factor and analysed
    point count are what make the table a series table rather than ten unrelated
    ones.
    """
    return {
        int(dataset.run_number): replace(
            build_fit_wizard_recommendation_for_templates(dataset, TEMPLATES),
            rebin_factor=1,
            analysed_points=int(dataset.n_points),
        )
        for dataset in datasets
    }


@pytest.fixture(scope="module")
def planted_series():
    """Screening and per-phase optimisation of the planted series, computed once.

    The end-to-end tests all interrogate the same answer, and the coupled fits
    behind it are the expensive part of this file.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            global_fit_wizard_module,
            "build_candidate_templates",
            lambda fingerprint, current_model=None: TEMPLATES,
        )
        datasets = _two_phase_series()
        table = _single_fit_table(datasets)
        screening = build_global_fit_wizard_screening_recommendation(
            datasets, single_fit_recommendations_by_run=table
        )
        optimised = build_global_fit_wizard_recommendation(
            datasets,
            single_fit_recommendations_by_run=table,
            partition_path=screening.partition_path,
            partition_k=1,
        )
    return datasets, table, screening, optimised


# --------------------------------------------------------------------------- #
# The dense-curve contract
# --------------------------------------------------------------------------- #


def _fully_curved(assessment) -> bool:
    """A displayed assessment: a curve per run it holds a fit result for."""
    for run_number in assessment.fit_results_by_run:
        curve = assessment.fitted_curves_by_run.get(run_number)
        if curve is None:
            return False
        fitted_time, fitted_curve = curve
        if not (np.size(fitted_time) and np.size(fitted_curve)):
            return False
        if run_number not in assessment.component_curves_by_run:
            return False
    return True


@pytest.mark.integration
def test_every_phase_answer_carries_curves_for_its_runs(planted_series):
    """A phase answer is drawn, so it owes a curve per run — and only it does.

    Each phase's search visits and caches many nodes; one per phase is kept, and
    the curves are built for those on the way out (see the dense-curve contract
    on ``GlobalCandidateAssessment``).
    """
    datasets, _table, _screening, optimised = planted_series

    assert optimised.phase_assessments
    for (_k, _segment_index), assessment in optimised.phase_assessments.items():
        assert _fully_curved(assessment)
        assert assessment.fit_results_by_run
        for result in assessment.fit_results_by_run.values():
            assert result.residuals is not None

    # A phase covers only its own runs, so its curves must not span the series.
    series_runs = {int(dataset.run_number) for dataset in datasets}
    covered = {
        run_number
        for assessment in optimised.phase_assessments.values()
        for run_number in assessment.fitted_curves_by_run
    }
    assert covered <= series_runs


@pytest.mark.integration
def test_a_partitioned_recommendation_exposes_only_curved_assessments(planted_series):
    """The series-wide rows a partitioned answer still exposes are drawable too."""
    _datasets, _table, _screening, optimised = planted_series

    assert optimised.assessments
    assert all(_fully_curved(assessment) for assessment in optimised.assessments)


# --------------------------------------------------------------------------- #
# Screening computes the path
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_screening_puts_the_elbow_at_the_planted_break(planted_series):
    _datasets, _table, screening, _optimised = planted_series
    path = screening.partition_path

    assert path is not None
    assert path.selected_k == 1
    selected = path.solutions[1]
    assert [(segment.start, segment.stop) for segment in selected.segments] == [
        (0, PLANTED_BREAK),
        (PLANTED_BREAK, 10),
    ]
    assert len(selected.boundaries) == 1
    assert selected.boundaries[0] == pytest.approx(PLANTED_BOUNDARY)
    # Each phase's cheapest structure is the model that generated it.
    assert selected.segments[0].structure.startswith("exp_constant")
    assert selected.segments[1].structure.startswith("gauss_constant")


@pytest.mark.integration
def test_the_break_clears_the_floor_by_a_wide_margin(planted_series):
    """A structural change is worth far more than the modified-BIC floor."""
    _datasets, _table, screening, _optimised = planted_series
    path = screening.partition_path

    assert path.solutions[1].gain > path.beta_floor
    assert path.solutions[1].admissible
    # ...and with two structures there is no partition with a second break (a
    # stub carrying the body's own structure is not a different phase), so the
    # path stops at one.
    assert len(path.solutions) == 2


@pytest.mark.integration
def test_screening_records_what_the_partition_cost(planted_series):
    datasets, table, _screening, _optimised = planted_series
    instrumentation: dict[str, object] = {}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            global_fit_wizard_module,
            "build_candidate_templates",
            lambda fingerprint, current_model=None: TEMPLATES,
        )
        build_global_fit_wizard_screening_recommendation(
            datasets,
            single_fit_recommendations_by_run=table,
            instrumentation=instrumentation,
        )

    assert instrumentation["partition_selected_k"] == 1
    assert float(instrumentation["partition_seconds"]) >= 0.0
    assert len(instrumentation["partition_gains"]) == len(_screening.partition_path.solutions)


def test_a_series_shorter_than_two_phases_has_no_partition_path():
    """Four runs cannot hold two segments of three, so there is nothing to find."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            global_fit_wizard_module,
            "build_candidate_templates",
            lambda fingerprint, current_model=None: (EXPONENTIAL_TEMPLATE,),
        )
        datasets = [
            _dataset(
                700 + index,
                EXPONENTIAL,
                {"A_1": 0.2, "Lambda": 0.4, "A_bg": 0.01},
                temperature=1.0 + index,
                n_points=120,
            )
            for index in range(4)
        ]
        screening = build_global_fit_wizard_screening_recommendation(
            datasets,
            single_fit_recommendations_by_run=_single_fit_table(datasets),
        )

    assert screening.partition_path is None


# --------------------------------------------------------------------------- #
# Per-phase optimisation
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_each_phase_is_optimised_to_the_model_that_generated_it(planted_series):
    _datasets, _table, _screening, optimised = planted_series

    assert optimised.recommended_partition_k == 1
    low = optimised.phase_assessment(0)
    high = optimised.phase_assessment(1)
    assert low is not None and high is not None
    assert low.template.key == "exp_constant"
    assert high.template.key == "gauss_constant"
    # The planted role structure: the relaxation parameter scans within a phase,
    # the amplitude does not.
    assert "Lambda" in low.local_param_names
    assert "A_1" in low.global_param_names
    assert "sigma" in high.local_param_names
    assert "A_1" in high.global_param_names


@pytest.mark.integration
def test_the_optimised_recommendation_names_its_transition(planted_series):
    _datasets, _table, _screening, optimised = planted_series

    assert optimised.summary == "1 transition found: 7.5 ± 2.5 K."


@pytest.mark.integration
def test_verification_covers_the_elbow_and_the_solution_below_it(planted_series):
    """k* and k*−1 are refitted exactly, and the elbow survives the exact gain."""
    _datasets, _table, _screening, optimised = planted_series
    path = optimised.partition_path

    assert path.selected_k == 1
    assert optimised.recommended_partition_k == 1
    assert [(segment.start, segment.stop) for segment in path.solutions[1].segments] == [
        (0, PLANTED_BREAK),
        (PLANTED_BREAK, 10),
    ]
    assert path.solutions[1].total_ic < path.solutions[0].total_ic


@pytest.mark.integration
def test_every_distinct_verified_segment_is_fitted_exactly_once(planted_series):
    """Neighbouring solutions share segments; a shared segment costs one search."""
    datasets, table, screening, _optimised = planted_series
    instrumentation: dict[str, object] = {}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            global_fit_wizard_module,
            "build_candidate_templates",
            lambda fingerprint, current_model=None: TEMPLATES,
        )
        build_global_fit_wizard_recommendation(
            datasets,
            single_fit_recommendations_by_run=table,
            partition_path=screening.partition_path,
            partition_k=1,
            instrumentation=instrumentation,
        )

    # k = 0 → (0, 10); k = 1 → (0, 5), (5, 10): three distinct segments.
    assert instrumentation["partition_segments_fitted"] == 3


@pytest.mark.integration
def test_an_excluded_end_stub_carries_no_phase_assessment(planted_series):
    """A stub is excluded from the global fit, so nothing fits it."""
    datasets, table, screening, _optimised = planted_series
    base = screening.partition_path
    body = base.solutions[1].segments
    all_runs = base.solutions[0].segments[0].run_numbers
    # Hand the optimiser a solution that sets the last run aside as a stub of a
    # different structure — what the partition produces when an end run looks
    # like yet another phase.
    trimmed = Segment(
        start=body[1].start,
        stop=body[1].stop - 1,
        run_numbers=body[1].run_numbers[:-1],
        structure=body[1].structure,
        ic=body[1].ic,
        excluded=False,
    )
    stub = Segment(
        start=len(all_runs) - 1,
        stop=len(all_runs),
        run_numbers=all_runs[-1:],
        structure="stub",
        ic=1.0,
        excluded=True,
    )
    two = PartitionSolution(
        breaks=2,
        segments=(body[0], trimmed, stub),
        total_ic=body[0].ic + trimmed.ic + stub.ic,
        gain=1.0,
        admissible=False,
        boundaries=(*base.solutions[1].boundaries, (99.0, 0.5)),
    )
    path = replace(base, solutions=(*base.solutions, two))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            global_fit_wizard_module,
            "build_candidate_templates",
            lambda fingerprint, current_model=None: TEMPLATES,
        )
        optimised = build_global_fit_wizard_recommendation(
            datasets,
            single_fit_recommendations_by_run=table,
            partition_path=path,
            partition_k=2,
        )

    assert (2, 0) in optimised.phase_assessments
    assert (2, 1) in optimised.phase_assessments
    assert (2, 2) not in optimised.phase_assessments


@pytest.mark.integration
def test_partition_k_none_runs_the_series_wide_search_and_carries_no_partition(
    planted_series,
):
    """The existing contract: without a ``partition_k`` nothing here applies.

    On *this* series the series-wide answer is that there isn't one — no single
    template describes runs on both sides of the transition well enough to pass
    the residual gate, which is exactly the failure the partitioned path exists to
    replace. What matters here is that the coupled search still ran and that none
    of the partition fields were touched.
    """
    datasets, table, _screening, _optimised = planted_series

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            global_fit_wizard_module,
            "build_candidate_templates",
            lambda fingerprint, current_model=None: TEMPLATES,
        )
        plain = build_global_fit_wizard_recommendation(
            datasets, single_fit_recommendations_by_run=table
        )

    assert plain.partition_path is None
    assert plain.phase_assessments == {}
    assert plain.recommended_partition_k is None
    assert plain.recommended_partition is None
    assert plain.phase_assessment(0) is None
    assert plain.optimized_assessments(), "the series-wide coupled search still ran"
    assert plain.recommended_key is None
    assert "did not pass" in plain.summary or "No globally optimized candidate" in plain.summary


# --------------------------------------------------------------------------- #
# Argument contract
# --------------------------------------------------------------------------- #


def test_a_partition_index_without_a_path_is_refused():
    datasets = _two_phase_series()[:4]
    with pytest.raises(ValueError, match="go together"):
        build_global_fit_wizard_recommendation(datasets, partition_k=1)


def test_a_partition_index_outside_the_path_is_refused():
    datasets = _two_phase_series()[:4]
    path = _fake_path()
    with pytest.raises(ValueError, match="outside the path"):
        build_global_fit_wizard_recommendation(datasets, partition_path=path, partition_k=7)


# --------------------------------------------------------------------------- #
# Transitions summary
# --------------------------------------------------------------------------- #


def _solution(boundaries, segments=None, *, breaks=None) -> PartitionSolution:
    segments = segments or (
        Segment(
            start=index,
            stop=index + 1,
            run_numbers=(index,),
            structure="t",
            ic=1.0,
            excluded=False,
        )
        for index in range(len(boundaries) + 1)
    )
    segments = tuple(segments)
    return PartitionSolution(
        breaks=len(segments) - 1 if breaks is None else breaks,
        segments=segments,
        total_ic=1.0,
        gain=0.0,
        admissible=True,
        boundaries=tuple(boundaries),
    )


def _fake_path() -> PartitionPath:
    return PartitionPath(
        solutions=(_solution(()), _solution(((7.5, 2.5),))),
        selected_k=1,
        beta_floor=16.0,
    )


def test_the_summary_names_each_transition_with_the_axis_unit():
    solution = _solution(((16.5, 0.5), (28.5, 0.5)))
    assert (
        transitions_summary(solution, "Temperature (K)")
        == "2 transitions found: 16.5 ± 0.5 K and 28.5 ± 0.5 K."
    )
    assert (
        transitions_summary(_solution(((120.0, 5.0),)), "Field (G)")
        == "1 transition found: 120 ± 5 G."
    )


def test_a_run_ordered_series_gets_no_invented_unit():
    assert transitions_summary(_solution(((3.5, 0.5),)), "Run") == "1 transition found: 3.5 ± 0.5."


def test_no_breaks_says_so_rather_than_listing_nothing():
    assert transitions_summary(_solution(()), "Temperature (K)") == (
        "No transitions found: one phase describes the whole series."
    )


def test_an_excluded_stub_is_named_in_the_summary():
    segments = (
        Segment(start=0, stop=3, run_numbers=(1, 2, 3), structure="t", ic=1.0, excluded=False),
        Segment(start=3, stop=4, run_numbers=(4,), structure="t", ic=1.0, excluded=True),
    )
    summary = transitions_summary(_solution(((3.5, 0.5),), segments), "Temperature (K)")
    assert "Run 4 is excluded from the global fit" in summary
    assert "looks like a different phase" in summary


# --------------------------------------------------------------------------- #
# Serialisation, merge, rerank
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_partitioned_recommendation_round_trips_through_serialisation(planted_series):
    _datasets, _table, _screening, optimised = planted_series

    payload = serialize_global_fit_wizard_recommendation(optimised, compact=True)
    # Persisted form, so it has to survive a real JSON round trip.
    restored = deserialize_global_fit_wizard_recommendation(json.loads(json.dumps(payload)))

    assert restored is not None
    assert restored.recommended_partition_k == optimised.recommended_partition_k
    assert restored.partition_path.selected_k == optimised.partition_path.selected_k
    assert restored.partition_path.to_payload() == optimised.partition_path.to_payload()
    assert set(restored.phase_assessments) == set(optimised.phase_assessments)
    for key, assessment in optimised.phase_assessments.items():
        restored_assessment = restored.phase_assessments[key]
        assert restored_assessment.template.key == assessment.template.key
        assert restored_assessment.global_param_names == assessment.global_param_names
        assert restored_assessment.local_param_names == assessment.local_param_names
        assert restored_assessment.bic == pytest.approx(assessment.bic)


def test_a_recommendation_without_a_partition_serialises_as_before():
    recommendation = _bare_recommendation()
    payload = serialize_global_fit_wizard_recommendation(recommendation)

    assert payload["partition_path"] is None
    assert payload["phase_assessments"] == []
    assert payload["recommended_partition_k"] is None

    restored = deserialize_global_fit_wizard_recommendation(payload)
    assert restored.partition_path is None
    assert restored.phase_assessments == {}
    assert restored.recommended_partition_k is None


def _bare_recommendation(**overrides) -> GlobalFitWizardRecommendation:
    defaults = {
        "series_axis_key": "temperature",
        "series_axis_label": "Temperature (K)",
        "mixed_axes_warning": None,
        "fingerprints_by_run": {},
        "dataset_order": (1, 2),
        "templates": (),
        "assessments": (),
        "metric": SelectionMetric.AICC,
        "recommended_key": None,
        "comparable_keys": (),
        "summary": "",
    }
    return GlobalFitWizardRecommendation(**{**defaults, **overrides})


def test_merge_adds_phase_assessments_and_takes_the_newer_path(planted_series=None):
    sentinel = object()
    base = _bare_recommendation(
        partition_path=_fake_path(),
        phase_assessments={(1, 0): sentinel},
        recommended_partition_k=1,
    )
    newer_path = replace(_fake_path(), beta_floor=99.0)
    updates = _bare_recommendation(
        partition_path=newer_path,
        phase_assessments={(1, 1): sentinel},
        recommended_partition_k=1,
        metric=SelectionMetric.BIC,
    )

    merged = merge_global_fit_wizard_recommendations(base, updates)

    assert set(merged.phase_assessments) == {(1, 0), (1, 1)}
    assert merged.partition_path.beta_floor == 99.0
    assert merged.recommended_partition_k == 1


def test_merge_keeps_the_existing_path_when_the_update_carries_none():
    base = _bare_recommendation(partition_path=_fake_path(), recommended_partition_k=1)
    merged = merge_global_fit_wizard_recommendations(base, _bare_recommendation())

    assert merged.partition_path is not None
    assert merged.recommended_partition_k == 1


def test_rerank_keeps_the_partition_and_restates_the_transitions():
    """The partition is BIC's answer; the ranking metric does not move it."""
    recommendation = _bare_recommendation(
        partition_path=_fake_path(),
        recommended_partition_k=1,
        summary="stale",
    )

    reranked = rerank_global_fit_wizard_recommendation(recommendation, SelectionMetric.AIC)

    assert reranked.metric == SelectionMetric.AIC
    assert reranked.partition_path is recommendation.partition_path
    assert reranked.recommended_partition_k == 1
    assert reranked.summary == "1 transition found: 7.5 ± 2.5 K."


def test_rerank_of_an_unoptimised_path_still_reports_the_screening_summary():
    """A path alone is not an answer — the user has still to pick a ``k``."""
    recommendation = _bare_recommendation(partition_path=_fake_path())

    reranked = rerank_global_fit_wizard_recommendation(recommendation, SelectionMetric.AIC)

    assert reranked.partition_path is not None
    assert reranked.recommended_partition_k is None
    assert "transition" not in reranked.summary


def _gated_assessment(
    *,
    key: str,
    aicc: float,
    gate_passed: bool,
) -> global_fit_wizard_module.GlobalCandidateAssessment:
    """A converged single-run assessment whose only distinctions are its score and gate."""
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.fit_wizard import CandidateTemplate
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    parameters = ParameterSet([Parameter("A_1", value=0.2, min=0.0, max=1.0)])
    diagnostic = global_fit_wizard_module.RunResidualDiagnostic(
        run_number=1,
        run_label="1",
        axis_value=1.0,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        gate_passed=gate_passed,
        gate_reasons=() if gate_passed else ("runs-test z score suggests structure (2.4)",),
    )
    return global_fit_wizard_module.GlobalCandidateAssessment(
        template=CandidateTemplate(
            key=key,
            title=key,
            category="General",
            rationale="test",
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
        ),
        fit_results_by_run={1: FitResult(success=True, chi_squared=aicc, parameters=parameters)},
        global_parameters=parameters,
        global_param_names=("A_1",),
        local_param_names=(),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(diagnostic,),
        series_warnings=(),
        aic=aicc,
        aicc=aicc,
        bic=aicc,
        selected_score=aicc,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )


def test_a_phase_keeps_the_far_better_model_over_a_gate_clean_one():
    # A phase is homogeneous by construction, so a gate caveat cannot outweigh a
    # criterion gap the search itself treats as decisive.
    better = _gated_assessment(key="damped", aicc=100.0, gate_passed=False)
    clean = _gated_assessment(key="stretched", aicc=1100.0, gate_passed=True)
    chosen = global_fit_wizard_module._recommended_segment_assessment([clean, better], {1: 1000})
    assert chosen is better


def test_a_phase_prefers_the_gate_clean_model_only_within_the_margin():
    better = _gated_assessment(key="damped", aicc=100.0, gate_passed=False)
    clean = _gated_assessment(
        key="stretched",
        aicc=100.0 + global_fit_wizard_module._LAYER_BOUND_MARGIN,
        gate_passed=True,
    )
    chosen = global_fit_wizard_module._recommended_segment_assessment([better, clean], {1: 1000})
    assert chosen is clean


def test_a_phase_with_no_converged_fit_has_no_answer():
    from dataclasses import replace as dc_replace

    from asymmetry.core.fitting.engine import FitResult

    failed = _gated_assessment(key="damped", aicc=100.0, gate_passed=True)
    failed = dc_replace(
        failed,
        fit_results_by_run={1: FitResult(success=False, parameters=failed.global_parameters)},
    )
    assert global_fit_wizard_module._recommended_segment_assessment([failed], {1: 1000}) is None


def test_series_fingerprints_are_computed_once_per_record(monkeypatch, planted_series):
    datasets, _table, _screening, _optimised = planted_series
    calls = {"n": 0}
    real = global_fit_wizard_module.fingerprint_spectrum

    def counting(dataset):
        calls["n"] += 1
        return real(dataset)

    monkeypatch.setattr(global_fit_wizard_module, "fingerprint_spectrum", counting)
    global_fit_wizard_module._SERIES_FINGERPRINT_CACHE.clear()
    first = global_fit_wizard_module._fingerprint_jump_warnings(datasets)
    second = global_fit_wizard_module._fingerprint_jump_warnings(datasets)
    assert first == second
    assert calls["n"] == len(datasets)


def test_a_phase_is_ranked_by_the_partition_score_not_the_search_metric():
    # A richer template can lead on AICc while the partition's per-run BIC —
    # the convention the path is totalled with — prefers the leaner one.
    from dataclasses import replace as dc_replace

    lean = _gated_assessment(key="one_line", aicc=100.0, gate_passed=True)
    rich = _gated_assessment(key="two_line", aicc=90.0, gate_passed=True)
    rich = dc_replace(rich, local_param_names=("A_1", "A_2", "A_3", "A_4"), global_param_names=())
    lean = dc_replace(lean, local_param_names=("A_1",), global_param_names=())
    points = {1: 20000}
    assert global_fit_wizard_module._partition_bic(rich, points) > (
        global_fit_wizard_module._partition_bic(lean, points)
    )
    chosen = global_fit_wizard_module._recommended_segment_assessment([rich, lean], points)
    assert chosen is lean


# --------------------------------------------------------------------------- #
# A vanished oscillation is a change of family
# --------------------------------------------------------------------------- #

#: A relaxing one-line multiplet, ``(Osc x Exp) + Exp + Const``. Its single line
#: amplitude is ``A_1``; ``A_3`` belongs to the relaxation, not to a line.
MULTIPLET_TEMPLATE = CandidateTemplate(
    key="oscillatory1_exp_relax_constant",
    title="1x damped cosine + relaxation + constant",
    category="Oscillatory",
    rationale="test",
    model=_multiplet_model(1, "Exponential", relax=True),
)
OSCILLATION_RUNS = 5
#: The multiplet's parameters on a run where the line is alive, and on one where
#: it has collapsed into the envelope: same shape, same quality of fit, but on
#: the collapsed run ``|A_1|`` sits inside two of its own standard deviations.
LINE_ALIVE = ({"A_1": 0.20}, {"A_1": 0.01})
LINE_VANISHED = ({"A_1": 0.004}, {"A_1": 0.01})


def _multiplet_fit_result(amplitude: dict[str, float], sigma: dict[str, float], chi_squared):
    """A converged multiplet fit whose line amplitude is the point of interest."""
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    values = {name: 0.1 for name in MULTIPLET_TEMPLATE.model.param_names}
    values.update(amplitude)
    uncertainties = {name: 0.001 for name in MULTIPLET_TEMPLATE.model.param_names}
    uncertainties.update(sigma)
    return FitResult(
        success=True,
        chi_squared=float(chi_squared),
        parameters=ParameterSet(
            [Parameter(name=name, value=value) for name, value in values.items()]
        ),
        uncertainties=uncertainties,
    )


def _relaxation_fit_result(chi_squared):
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    return FitResult(
        success=True,
        chi_squared=float(chi_squared),
        parameters=ParameterSet(
            [Parameter(name=name, value=0.1) for name in EXPONENTIAL.param_names]
        ),
        uncertainties=dict.fromkeys(EXPONENTIAL.param_names, 0.001),
    )


def _prescreen_assessment(template, results_by_run):
    """A completed screening row: one converged fit per run, nothing else needed."""
    from asymmetry.core.fitting.parameters import ParameterSet

    return global_fit_wizard_module.GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=results_by_run,
        global_parameters=ParameterSet(),
        global_param_names=(),
        local_param_names=tuple(template.model.param_names),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        aic=0.0,
        aicc=0.0,
        bic=0.0,
        selected_score=0.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )


def _oscillation_that_stops_series(*, line_vanishes: bool):
    """A ten-run series whose oscillation dies halfway, and its screening rows.

    The multiplet is given the *better* per-run cost on every run of the series,
    including the ones where its line has collapsed — so on the criterion alone
    it describes the whole series and there is no break to find. The only thing
    that can separate the phases is the rule: past the transition its lines are
    consistent with zero, so it is not a candidate for the oscillatory family
    there and the relaxation template is all that is left.
    """
    datasets = _two_phase_series()
    multiplet_results = {}
    relaxation_results = {}
    for index, dataset in enumerate(datasets):
        run_number = int(dataset.run_number)
        oscillating = index < OSCILLATION_RUNS
        amplitude, sigma = LINE_ALIVE if oscillating or not line_vanishes else LINE_VANISHED
        multiplet_results[run_number] = _multiplet_fit_result(amplitude, sigma, 250.0)
        relaxation_results[run_number] = _relaxation_fit_result(300.0 if oscillating else 260.0)
    return datasets, {
        MULTIPLET_TEMPLATE.key: _prescreen_assessment(MULTIPLET_TEMPLATE, multiplet_results),
        EXPONENTIAL_TEMPLATE.key: _prescreen_assessment(EXPONENTIAL_TEMPLATE, relaxation_results),
    }


def _partition_of(datasets, assessments, instrumentation=None):
    return global_fit_wizard_module.build_series_partition_path(
        datasets,
        assessments,
        axis_key="temperature",
        analysed_points_by_run={int(d.run_number): int(d.n_points) for d in datasets},
        family_by_template={
            MULTIPLET_TEMPLATE.key: "oscillatory",
            EXPONENTIAL_TEMPLATE.key: "relaxation",
        },
        instrumentation=instrumentation,
    )


def test_a_multiplet_that_keeps_its_lines_describes_the_whole_series():
    """The control: with the lines alive throughout, the cheaper template wins flat.

    This is what makes the next test's break attributable to the rule and not to
    the costs — the same numbers, and no break.
    """
    datasets, assessments = _oscillation_that_stops_series(line_vanishes=False)

    path = _partition_of(datasets, assessments)

    assert path.selected_k == 0
    assert path.solutions[0].segments[0].structure == "oscillatory"


def test_the_series_breaks_where_the_oscillation_vanishes():
    """The multiplet is cheaper on every run, and still cannot span the series.

    Past the transition its line amplitude is inside two sigma, so its cells are
    infeasible for the oscillatory family there. The zero-break solution has to
    fall back to relaxation across the whole series, one break buys the
    oscillatory phase back for the runs that have one, and the break lands
    exactly where the lines stopped.
    """
    datasets, assessments = _oscillation_that_stops_series(line_vanishes=True)
    instrumentation: dict[str, object] = {}

    path = _partition_of(datasets, assessments, instrumentation)

    assert path.selected_k == 1
    low, high = path.solutions[1].segments
    assert low.run_numbers == tuple(int(d.run_number) for d in datasets[:OSCILLATION_RUNS])
    assert (low.structure, high.structure) == ("oscillatory", "relaxation")
    # Nothing but relaxation is available once the lines are gone.
    assert path.solutions[0].segments[0].structure == "relaxation"
    assert instrumentation["counters"]["oscillation_vanished_cells"] == OSCILLATION_RUNS


def test_only_the_runs_whose_lines_vanished_lose_their_multiplet_cell():
    """The rule is per cell: it never removes a template from the whole series."""
    datasets, assessments = _oscillation_that_stops_series(line_vanishes=True)

    table, _estimates, _points = global_fit_wizard_module._partition_inputs_from_prescreen(
        datasets,
        assessments,
        analysed_points_by_run={int(d.run_number): int(d.n_points) for d in datasets},
    )

    for dataset in datasets[:OSCILLATION_RUNS]:
        assert MULTIPLET_TEMPLATE.key in table[int(dataset.run_number)]
    for dataset in datasets[OSCILLATION_RUNS:]:
        run = int(dataset.run_number)
        assert MULTIPLET_TEMPLATE.key not in table[run]
        # The run is still describable — only its *oscillatory* reading is gone.
        assert EXPONENTIAL_TEMPLATE.key in table[run]


def _multiplet_phase_assessment(*, key, chi_squared, vanished_runs=()):
    """A converged two-run phase fit of a multiplet template."""
    from asymmetry.core.fitting.parameters import ParameterSet

    results = {}
    for run_number in (1, 2):
        amplitude, sigma = LINE_VANISHED if run_number in vanished_runs else LINE_ALIVE
        results[run_number] = _multiplet_fit_result(amplitude, sigma, chi_squared)
    return global_fit_wizard_module.GlobalCandidateAssessment(
        template=replace(MULTIPLET_TEMPLATE, key=key),
        fit_results_by_run=results,
        global_parameters=ParameterSet(),
        global_param_names=(),
        local_param_names=("A_1",),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        aic=chi_squared,
        aicc=chi_squared,
        bic=chi_squared,
        selected_score=chi_squared,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )


def test_a_phase_whose_oscillation_dies_on_one_run_is_not_that_phases_answer():
    """Tier 3: a phase fit must describe *every* run of the phase as oscillatory.

    The losing candidate here is the better fit — it is excluded because on one
    run of the phase it has decayed to its envelope, so it is not a description
    of an oscillatory phase at all.
    """
    points = {1: 1000, 2: 1000}
    vanished = _multiplet_phase_assessment(
        key="oscillatory1_exp_relax_constant", chi_squared=100.0, vanished_runs=(2,)
    )
    alive = _multiplet_phase_assessment(key="oscillatory2_exp_relax_constant", chi_squared=400.0)
    instrumentation: dict[str, object] = {}

    candidates = global_fit_wizard_module._oscillatory_admissible_phase_candidates(
        [vanished, alive], instrumentation
    )

    assert candidates == (alive,)
    assert instrumentation["counters"]["oscillation_vanished_phases"] == 1
    assert global_fit_wizard_module._recommended_segment_assessment(candidates, points) is alive


def test_a_phase_with_only_a_vanished_oscillation_has_no_answer():
    """Nothing left to choose makes the containing partition infeasible, as today."""
    vanished = _multiplet_phase_assessment(
        key="oscillatory1_exp_relax_constant", chi_squared=100.0, vanished_runs=(2,)
    )

    candidates = global_fit_wizard_module._oscillatory_admissible_phase_candidates([vanished])

    assert candidates == ()
    assert (
        global_fit_wizard_module._recommended_segment_assessment(candidates, {1: 1000, 2: 1000})
        is None
    )


def test_a_relaxation_phase_is_never_touched_by_the_rule():
    """Only multiplet templates have lines; everything else passes by construction."""
    plain = _gated_assessment(key="exp_constant", aicc=100.0, gate_passed=True)

    assert global_fit_wizard_module._oscillatory_admissible_phase_candidates([plain]) == (plain,)
