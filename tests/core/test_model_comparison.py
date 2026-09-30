"""Tests for the Qt-free model-comparison judgements (core/fitting/model_comparison.py)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.experiment_design import aic_weights
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    RunResidualDiagnostic,
)
from asymmetry.core.fitting.model_comparison import (
    AT_BOUND_RELATIVE_TOLERANCE,
    SHORTLIST_MAX_DELTA,
    SHORTLIST_MAX_SIZE,
    Estimate,
    FitGrade,
    ParameterFlag,
    ParameterRole,
    bound_side,
    compare_parameters,
    grade_reduced_chi_squared,
    information_weights,
    normalised_residuals,
    parameter_flags,
    score_deltas,
    shortlist,
    summarise_candidates,
    summarise_single_candidates,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------

_RUNS = (701, 702, 703)
_FIELDS = {701: 100.0, 702: 200.0, 703: 300.0}
_LAMBDAS = {701: 0.2, 702: 0.3, 703: 0.5}


def _dataset(run: int) -> MuonDataset:
    time = np.linspace(0.0, 8.0, 81)
    return MuonDataset(
        time=time,
        asymmetry=20.0 * np.exp(-_LAMBDAS[run] * time) + 1.0,
        error=np.full_like(time, 0.5),
        metadata={"run_number": run, "run_label": str(run), "field": _FIELDS[run]},
    )


def _datasets() -> list[MuonDataset]:
    return [_dataset(run) for run in _RUNS]


def _fit(run: int, *, a_1: float = 20.0, a_1_err: float = 0.2) -> FitResult:
    return FitResult(
        success=True,
        reduced_chi_squared=1.0 + 0.1 * (run - 700),
        parameters=ParameterSet(
            [
                Parameter("A_1", value=a_1, min=0.0, max=100.0),
                Parameter("Lambda", value=_LAMBDAS[run], min=0.0, max=10.0),
                Parameter("A_bg", value=1.0, fixed=True),
            ]
        ),
        uncertainties={"A_1": a_1_err, "Lambda": 0.01},
    )


def _assessment(
    key: str,
    *,
    aicc: float,
    bic: float = 0.0,
    runs: tuple[int, ...] = _RUNS,
    fits: dict[int, FitResult] | None = None,
    gate_reasons: dict[int, tuple[str, ...]] | None = None,
    global_names: tuple[str, ...] = ("A_1",),
    local_names: tuple[str, ...] = ("Lambda",),
    prescreen: bool = False,
) -> GlobalCandidateAssessment:
    fits = {run: _fit(run) for run in runs} if fits is None else fits
    gate_reasons = gate_reasons or {}
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    curves = {}
    for run in fits:
        # A coarser, narrower grid than the dataset's: the analysis grid differs.
        t = np.linspace(0.05, 7.5, 40)
        curves[run] = (t, 20.0 * np.exp(-_LAMBDAS[run] * t) + 1.0)
    return GlobalCandidateAssessment(
        template=CandidateTemplate(
            key=key.split("|")[0],
            title=f"Title {key}",
            category="General",
            rationale="synthetic",
            model=model,
        ),
        fit_results_by_run=fits,
        global_parameters=ParameterSet(),
        global_param_names=global_names,
        local_param_names=local_names,
        fixed_param_names=("A_bg",),
        parameter_recommendations=(),
        run_diagnostics=tuple(
            RunResidualDiagnostic(
                run_number=run,
                run_label=str(run),
                axis_value=_FIELDS[run],
                residual_rms=1.0,
                runs_z_score=0.0,
                max_abs_autocorrelation=0.0,
                residual_fft_peak_snr=0.0,
                gate_passed=not gate_reasons.get(run),
                gate_reasons=gate_reasons.get(run, ()),
            )
            for run in runs
        ),
        series_warnings=(),
        aic=aicc,
        aicc=aicc,
        bic=bic,
        selected_score=aicc,
        fitted_curves_by_run=curves,
        component_curves_by_run={},
        prescreen_only=prescreen,
        assessment_key=key,
    )


# ---------------------------------------------------------------------------
# Weights and deltas
# ---------------------------------------------------------------------------


def test_information_weights_follow_exp_minus_half_delta() -> None:
    weights = information_weights([10.0, 12.0])
    assert weights[0] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert weights[1] == pytest.approx(math.exp(-1.0) / (1.0 + math.exp(-1.0)))


def test_information_weights_zero_non_finite_scores_and_renormalise() -> None:
    weights = information_weights([1.0, float("nan"), float("inf"), 3.0])
    assert weights[1] == 0.0
    assert weights[2] == 0.0
    assert sum(weights) == pytest.approx(1.0)


def test_information_weights_all_non_finite_or_empty() -> None:
    assert information_weights([float("nan"), float("inf")]) == [0.0, 0.0]
    assert information_weights([]) == []


def test_score_deltas_measure_from_the_best_finite_score() -> None:
    assert score_deltas([5.0, float("nan"), 2.0]) == [3.0, math.inf, 0.0]


def test_aic_weights_are_information_weights_of_aic() -> None:
    chi2, p = [10.0, 11.0, float("nan")], [1, 2, 1]
    assert aic_weights(chi2, p) == information_weights([12.0, 15.0, float("nan")])


# ---------------------------------------------------------------------------
# Shortlist (D7)
# ---------------------------------------------------------------------------


def test_shortlist_includes_exactly_the_delta_threshold() -> None:
    scores = {"best": 0.0, "edge": SHORTLIST_MAX_DELTA, "out": SHORTLIST_MAX_DELTA + 1e-9}
    assert shortlist(scores) == ("best", "edge")


def test_shortlist_is_capped_best_first() -> None:
    scores = {"d": 3.0, "a": 0.0, "nan": float("nan"), "c": 2.0, "b": 1.0}
    assert SHORTLIST_MAX_SIZE == 3
    assert shortlist(scores) == ("a", "b", "c")


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def test_normalised_residuals_use_the_dataset_grid_within_the_curve_span() -> None:
    time = np.linspace(0.0, 10.0, 101)
    dataset = MuonDataset(
        time=time,
        asymmetry=2.0 * time + 1.0,
        error=np.full_like(time, 0.5),
        metadata={"run_number": 1},
    )
    # A linear model on a coarser, narrower grid interpolates exactly.
    curve_t = np.linspace(1.05, 8.0, 7)
    t, r = normalised_residuals(dataset, (curve_t, 2.0 * curve_t))
    assert t[0] == pytest.approx(1.1)
    assert t[-1] == pytest.approx(8.0)
    np.testing.assert_allclose(t, time[(time >= 1.05) & (time <= 8.0)])
    np.testing.assert_allclose(r, np.full_like(t, 2.0))


# ---------------------------------------------------------------------------
# Parameter flags (D8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parameter", "error", "expected"),
    [
        (Parameter("x", value=1.0, min=0.0, max=10.0), 0.1, ()),
        (Parameter("x", value=0.0, min=0.0, max=10.0), 0.0, (ParameterFlag.AT_LOWER_BOUND,)),
        (Parameter("x", value=10.0, min=0.0, max=10.0), 0.1, (ParameterFlag.AT_UPPER_BOUND,)),
        (
            Parameter("x", value=10.0 * (1 - 0.5 * AT_BOUND_RELATIVE_TOLERANCE), max=10.0),
            0.1,
            (ParameterFlag.AT_UPPER_BOUND,),
        ),
        (Parameter("x", value=math.nan), 0.1, (ParameterFlag.NOT_FINITE,)),
        (Parameter("x", value=1.0), 2.0, (ParameterFlag.POORLY_DETERMINED,)),
        (Parameter("x", value=1.0), math.inf, ()),
        (Parameter("x", value=1.0), math.nan, ()),
        (
            Parameter("x", value=0.0, min=0.0),
            0.5,
            (ParameterFlag.AT_LOWER_BOUND, ParameterFlag.POORLY_DETERMINED),
        ),
        (Parameter("x", value=0.0, min=0.0, fixed=True), 0.5, ()),
    ],
)
def test_parameter_flags(parameter: Parameter, error: float, expected: tuple) -> None:
    assert parameter_flags(parameter, error) == expected


def test_bound_side_ignores_infinite_bounds_in_the_tolerance() -> None:
    assert bound_side(Parameter("x", value=5.0, min=0.0)) is None
    assert bound_side(Parameter("x", value=1e-7, min=0.0)) == "lower"
    assert bound_side(Parameter("x", value=3.0, max=3.0)) == "upper"
    assert ParameterFlag.AT_LOWER_BOUND.value == "at lower bound"
    assert ParameterFlag.POORLY_DETERMINED.value == "poorly determined"


# ---------------------------------------------------------------------------
# summarise_candidates
# ---------------------------------------------------------------------------


def test_summaries_carry_delta_and_weight_across_the_given_pool() -> None:
    pool = [_assessment("b|1", aicc=104.0), _assessment("a|1", aicc=100.0)]
    summaries = summarise_candidates(pool, _datasets(), SelectionMetric.AICC)
    assert [s.key for s in summaries] == ["b|1", "a|1"]
    assert [s.delta for s in summaries] == [4.0, 0.0]
    assert [s.weight for s in summaries] == pytest.approx(information_weights([104.0, 100.0]))
    assert summaries[0].title == "Title b|1"
    assert summaries[0].metric_value == 104.0


def test_summaries_rank_on_the_requested_metric() -> None:
    pool = [_assessment("a|1", aicc=100.0, bic=30.0), _assessment("b|1", aicc=104.0, bic=20.0)]
    summaries = summarise_candidates(pool, _datasets(), SelectionMetric.BIC)
    assert [s.delta for s in summaries] == [10.0, 0.0]


def test_summary_runs_follow_series_order_with_residuals_and_chi2() -> None:
    fits = {run: _fit(run) for run in (703, 701, 702)}
    summary = summarise_candidates(
        [_assessment("a|1", aicc=1.0, fits=fits)], list(reversed(_datasets())), SelectionMetric.AICC
    )[0]
    assert [run.run_number for run in summary.runs] == [701, 702, 703]
    assert [run.axis_value for run in summary.runs] == [100.0, 200.0, 300.0]
    run = summary.runs[1]
    assert run.run_label == "702"
    assert run.reduced_chi_squared == pytest.approx(1.2)
    t, r = run.curves.residuals
    assert t[0] >= run.curves.fit[0][0]
    assert t[-1] <= run.curves.fit[0][-1]
    # The curve is the data's own model, so residuals are interpolation error only.
    assert np.max(np.abs(r)) < 0.1


def test_summary_parameter_rows_share_globals_only_after_a_coupled_fit() -> None:
    fits = {run: _fit(run, a_1=20.0, a_1_err=0.3) for run in _RUNS}
    summary = summarise_candidates(
        [_assessment("a|1", aicc=1.0, fits=fits)], _datasets(), SelectionMetric.AICC
    )[0]
    rows = {row.name: row for row in summary.parameters}
    assert [row.name for row in summary.parameters] == ["A_1", "Lambda", "A_bg"]
    assert rows["A_1"].role is ParameterRole.GLOBAL
    assert summary.names(ParameterRole.GLOBAL) == ("A_1",)
    assert summary.names(ParameterRole.LOCAL) == ("Lambda",)
    assert summary.names(ParameterRole.FIXED) == ("A_bg",)
    assert rows["A_1"].values == Estimate(20.0, 0.3)
    assert rows["A_1"].run_flags == ((), (), ())
    assert rows["Lambda"].role is ParameterRole.LOCAL
    assert rows["Lambda"].values == tuple(Estimate(_LAMBDAS[run], 0.01) for run in _RUNS)
    assert rows["A_bg"].role is ParameterRole.FIXED
    assert all(math.isnan(e.error) for e in rows["A_bg"].values)  # type: ignore[union-attr]
    assert all(not row.flags for row in summary.parameters)

    prescreen = summarise_candidates(
        [_assessment("a", aicc=1.0, fits=fits, prescreen=True)], _datasets(), SelectionMetric.AICC
    )[0]
    assert prescreen.prescreen
    assert prescreen.parameters[0].values == (Estimate(20.0, 0.3),) * 3


def test_summary_row_flags_are_kept_per_run_with_their_union() -> None:
    fits = {run: _fit(run) for run in _RUNS}
    fits[702] = _fit(702, a_1=0.0)
    summary = summarise_candidates(
        [_assessment("a|1", aicc=1.0, fits=fits, global_names=(), local_names=("A_1", "Lambda"))],
        _datasets(),
        SelectionMetric.AICC,
    )[0]
    assert summary.parameters[0].flags == (
        ParameterFlag.AT_LOWER_BOUND,
        ParameterFlag.POORLY_DETERMINED,
    )
    assert summary.parameters[0].run_flags == (
        (),
        (ParameterFlag.AT_LOWER_BOUND, ParameterFlag.POORLY_DETERMINED),
        (),
    )


def test_summary_omits_runs_an_incomplete_prescreen_never_fitted() -> None:
    fits = {701: _fit(701), 703: _fit(703)}
    summary = summarise_candidates(
        [
            _assessment(
                "a",
                aicc=math.inf,
                fits=fits,
                prescreen=True,
                gate_reasons={702: ("missing successful single-fit assessment for Title a",)},
            )
        ],
        _datasets(),
        SelectionMetric.AICC,
    )[0]
    assert [run.run_number for run in summary.runs] == [701, 703]
    assert summary.delta == math.inf
    assert summary.weight == 0.0
    assert not summary.gate_passed


def test_summary_failed_run_has_unknown_values_without_flags() -> None:
    fits = {run: _fit(run) for run in _RUNS}
    fits[702] = FitResult(success=False, message="failed")
    summary = summarise_candidates(
        [_assessment("a|1", aicc=1.0, fits=fits)], _datasets(), SelectionMetric.AICC
    )[0]
    lam = summary.parameters[1].values
    assert isinstance(lam, tuple)
    assert math.isnan(lam[1].value) and math.isnan(lam[1].error)
    assert summary.parameters[1].flags == ()
    assert summary.parameters[1].run_flags == ((), (), ())


def test_gate_summary_dedupes_reasons_and_names_the_runs() -> None:
    reasons = {
        701: ("runs-test z score suggests structure",),
        702: ("Lambda at lower bound",),
        703: ("runs-test z score suggests structure",),
    }
    summary = summarise_candidates(
        [_assessment("a|1", aicc=1.0, gate_reasons=reasons)], _datasets(), SelectionMetric.AICC
    )[0]
    assert not summary.gate_passed
    assert summary.gate_summary == (
        "runs-test z score suggests structure (runs 701, 703); Lambda at lower bound (run 702)"
    )
    clean = summarise_candidates([_assessment("a|1", aicc=1.0)], _datasets(), SelectionMetric.AICC)
    assert clean[0].gate_passed
    assert clean[0].gate_summary == ""


def test_gate_summary_leaves_a_flagged_bound_hit_to_the_parameter_flag() -> None:
    fits = {run: _fit(run) for run in _RUNS}
    fits[702] = _fit(702, a_1=0.0)
    reasons = {702: ("A_1 at lower bound", "runs-test z score suggests structure")}
    summary = summarise_candidates(
        [
            _assessment(
                "a|1",
                aicc=1.0,
                fits=fits,
                global_names=(),
                local_names=("A_1", "Lambda"),
                gate_reasons=reasons,
            )
        ],
        _datasets(),
        SelectionMetric.AICC,
    )[0]
    assert summary.gate_summary == "runs-test z score suggests structure (run 702)"
    assert ParameterFlag.AT_LOWER_BOUND in summary.parameters[0].flags


# ---------------------------------------------------------------------------
# A vs B
# ---------------------------------------------------------------------------


def test_compare_orders_globals_first_and_flags_shared_globals_by_sigma() -> None:
    fits_a = {run: _fit(run, a_1=20.0, a_1_err=0.3) for run in _RUNS}
    fits_b = {run: _fit(run, a_1=21.55, a_1_err=0.4) for run in _RUNS}
    a_rows = summarise_candidates(
        [_assessment("t|a", aicc=1.0, fits=fits_a)], _datasets(), SelectionMetric.AICC
    )[0]
    b_rows = summarise_candidates(
        [_assessment("t|b", aicc=2.0, fits=fits_b, global_names=("A_1", "Lambda"), local_names=())],
        _datasets(),
        SelectionMetric.AICC,
    )[0]
    pairs = compare_parameters(a_rows, b_rows)
    assert [pair.name for pair in pairs] == ["A_1", "Lambda", "A_bg"]
    assert pairs[0].sigma_difference == pytest.approx(3.1)
    assert pairs[0].difference_text == "differs by 3.1σ"
    # Lambda is local in A and global in B: not a shared global.
    assert pairs[1].sigma_difference is None
    assert pairs[1].difference_text == ""


def test_compare_places_names_only_b_has_by_their_role_in_b() -> None:
    a = summarise_candidates(
        [_assessment("t|a", aicc=1.0, global_names=(), local_names=("Lambda",))],
        _datasets(),
        SelectionMetric.AICC,
    )[0]
    b = summarise_candidates([_assessment("t|b", aicc=2.0)], _datasets(), SelectionMetric.AICC)[0]
    # A has no A_1 at all; B's global A_1 leads the table.
    a = replace(a, parameters=a.parameters[1:])
    pairs = compare_parameters(a, b)
    assert [pair.name for pair in pairs] == ["A_1", "Lambda", "A_bg"]
    assert pairs[0].a is None
    assert pairs[0].sigma_difference is None


def test_compare_does_not_flag_a_difference_under_two_sigma() -> None:
    a = summarise_candidates(
        [_assessment("t|a", aicc=1.0, fits={r: _fit(r, a_1=20.0, a_1_err=0.3) for r in _RUNS})],
        _datasets(),
        SelectionMetric.AICC,
    )[0]
    b = summarise_candidates(
        [_assessment("t|b", aicc=1.0, fits={r: _fit(r, a_1=20.9, a_1_err=0.4) for r in _RUNS})],
        _datasets(),
        SelectionMetric.AICC,
    )[0]
    pair = compare_parameters(a, b)[0]
    assert pair.sigma_difference == pytest.approx(1.8)
    assert pair.difference_text == ""


def test_compare_leaves_k_unmeasured_when_errors_vanish() -> None:
    a, b = (
        summarise_candidates(
            [_assessment(key, aicc=1.0, fits={r: _fit(r, a_1=a_1, a_1_err=0.0) for r in _RUNS})],
            _datasets(),
            SelectionMetric.AICC,
        )[0]
        for key, a_1 in (("t|a", 20.0), ("t|b", 25.0))
    )
    pair = compare_parameters(a, b)[0]
    assert math.isnan(pair.sigma_difference)
    assert pair.difference_text == ""


@pytest.mark.parametrize(
    ("value", "grade"),
    [
        (0.9, FitGrade.GOOD),
        (1.5, FitGrade.GOOD),
        (1.51, FitGrade.FAIR),
        (5.0, FitGrade.FAIR),
        (5.01, FitGrade.POOR),
        (math.inf, FitGrade.POOR),
        (math.nan, FitGrade.POOR),
    ],
)
def test_grade_reduced_chi_squared_thresholds(value: float, grade: FitGrade) -> None:
    assert grade_reduced_chi_squared(value) is grade


# ---------------------------------------------------------------------------
# summarise_single_candidates (the single-run wizard, N = 1)
# ---------------------------------------------------------------------------


def _single(
    key: str,
    *,
    aicc: float,
    bic: float = 0.0,
    fit: FitResult | None = None,
    curves: bool = True,
    gate_reasons: tuple[str, ...] = (),
    null_baseline: bool = False,
    disqualified: tuple[str, ...] = (),
) -> CandidateAssessment:
    time = np.linspace(0.05, 7.5, 40)
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=CandidateTemplate(
            key=key,
            title=f"Title {key}",
            category="General",
            rationale="synthetic",
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
        ),
        fit_result=_fit(701) if fit is None else fit,
        aic=aicc,
        aicc=aicc,
        bic=bic,
        selected_score=aicc,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=not gate_reasons,
        residual_gate_reasons=gate_reasons,
        bound_hits=(),
        fitted_time=time if curves else empty,
        fitted_curve=20.0 * np.exp(-_LAMBDAS[701] * time) + 1.0 if curves else empty,
        component_curves=(),
        disqualification_reasons=disqualified,
        is_null_baseline=null_baseline,
    )


def _single_recommendation(*assessments: CandidateAssessment) -> FitWizardRecommendation:
    return FitWizardRecommendation(
        fingerprint=SpectrumFingerprint(
            tail_estimate=1.0,
            initial_amplitude_estimate=20.0,
            zero_crossings=0,
            smoothed_zero_crossings=0,
            smoothed_turning_points=0,
            dominant_fft_frequency_mhz=0.0,
            dominant_fft_snr=0.0,
            dominant_fft_cycles_in_window=0.0,
            monotonic_decay_fraction=1.0,
            early_time_curvature=0.0,
            semilog_slope_ratio=1.0,
            late_time_dip_recovery_score=0.0,
            oscillatory_hint=False,
            kt_like_hint=False,
            multi_rate_hint=False,
        ),
        templates=tuple(assessment.template for assessment in assessments),
        assessments=assessments,
        metric=SelectionMetric.AICC,
        recommended_key=assessments[0].template.key,
        comparable_keys=(),
        summary="synthetic",
    )


def test_single_summaries_rank_every_row_with_delta_and_weight_across_all() -> None:
    recommendation = _single_recommendation(
        _single("exp", aicc=100.0),
        _single("flat", aicc=140.0, null_baseline=True),
        _single("osc", aicc=98.0, disqualified=("oscillation amplitude consistent with zero",)),
    )
    summaries = summarise_single_candidates(recommendation, _dataset(701), SelectionMetric.AICC)
    assert [s.key for s in summaries] == ["osc", "exp", "flat"]
    assert [s.delta for s in summaries] == [0.0, 2.0, 42.0]
    assert [s.weight for s in summaries] == pytest.approx(information_weights([98.0, 100.0, 140.0]))
    assert [s.title for s in summaries] == [
        "Title osc (disqualified)",
        "Title exp",
        "Title flat (baseline)",
    ]


def test_single_summaries_rank_on_the_requested_metric() -> None:
    recommendation = _single_recommendation(
        _single("a", aicc=100.0, bic=30.0), _single("b", aicc=104.0, bic=20.0)
    )
    summaries = summarise_single_candidates(recommendation, _dataset(701), SelectionMetric.BIC)
    assert [(s.key, s.delta) for s in summaries] == [("b", 0.0), ("a", 10.0)]


def test_single_summary_is_one_run_with_fitted_and_fixed_parameters() -> None:
    summary = summarise_single_candidates(
        _single_recommendation(_single("exp", aicc=1.0)), _dataset(701), SelectionMetric.AICC
    )[0]
    (run,) = summary.runs
    assert (run.run_number, run.run_label) == (701, "701")
    assert math.isnan(run.axis_value)
    assert run.reduced_chi_squared == pytest.approx(1.1)
    assert summary.names(ParameterRole.FITTED) == ("A_1", "Lambda")
    assert summary.names(ParameterRole.FIXED) == ("A_bg",)
    assert summary.names(ParameterRole.GLOBAL) == ()
    rows = {row.name: row for row in summary.parameters}
    assert rows["A_1"].values == (Estimate(20.0, 0.2),)
    assert rows["A_1"].role is ParameterRole.FITTED
    assert not summary.prescreen


def test_single_summary_leaves_a_bare_row_without_curves() -> None:
    summaries = summarise_single_candidates(
        _single_recommendation(_single("drawn", aicc=1.0), _single("bare", aicc=2.0, curves=False)),
        _dataset(701),
        SelectionMetric.AICC,
    )
    drawn, bare = summaries
    assert drawn.curves_built
    t, r = drawn.runs[0].curves.residuals
    assert t[0] >= drawn.runs[0].curves.fit[0][0]
    assert np.max(np.abs(r)) < 0.1
    assert not bare.curves_built
    assert bare.runs[0].curves is None


def test_single_gate_summary_names_no_run_and_leaves_bound_hits_to_the_flags() -> None:
    summary = summarise_single_candidates(
        _single_recommendation(
            _single(
                "exp",
                aicc=1.0,
                fit=_fit(701, a_1=0.0),
                gate_reasons=("A_1 at lower bound", "runs-test z score suggests structure"),
            )
        ),
        _dataset(701),
        SelectionMetric.AICC,
    )[0]
    assert not summary.gate_passed
    assert summary.gate_summary == "runs-test z score suggests structure"
    assert ParameterFlag.AT_LOWER_BOUND in summary.parameters[0].flags


def test_compare_places_fitted_parameters_before_fixed_ones() -> None:
    a, b = summarise_single_candidates(
        _single_recommendation(_single("a", aicc=1.0), _single("b", aicc=2.0)),
        _dataset(701),
        SelectionMetric.AICC,
    )
    pairs = compare_parameters(a, b)
    assert [pair.name for pair in pairs] == ["A_1", "Lambda", "A_bg"]
    # A single run's values are per run, never a shared global: no kσ judgement.
    assert pairs[0].sigma_difference is None
