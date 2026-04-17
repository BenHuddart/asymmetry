"""Tests for the global fit wizard core service."""

from __future__ import annotations

import numpy as np
import asymmetry.core.fitting.global_fit_wizard as global_fit_wizard_module

from asymmetry.core import fitting as fitting_api
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    RunResidualDiagnostic,
    _canonicalize_parameter_sets,
    _fit_exact_assignment,
    _localisation_penalty,
    _single_run_prefit_parameter_sets,
    _staged_multi_local_assignment,
    _supported_oscillatory_run_numbers,
    _warm_start_parameter_sets,
    build_global_fit_wizard_recommendation,
    rerank_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _dataset_for(
    run_number: int,
    *,
    field: float,
    temperature: float,
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
            "field": field,
            "temperature": temperature,
            "run_label": str(run_number),
        },
    )


def _dense_dataset_for(
    run_number: int,
    *,
    field: float,
    temperature: float,
    model: CompositeModel,
    params: dict[str, float],
    error_level: float = 0.005,
) -> MuonDataset:
    time = np.linspace(0.0, 8.0, 240)
    asymmetry = model.function(time, **params)
    error = np.full_like(time, error_level)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={
            "run_number": run_number,
            "field": field,
            "temperature": temperature,
            "run_label": str(run_number),
        },
    )


def _fingerprint(
    *,
    dominant_fft_snr: float = 0.0,
    cycles: float = 0.0,
    turning_points: int = 0,
    oscillatory_hint: bool = False,
) -> fitting_api.SpectrumFingerprint:
    return fitting_api.SpectrumFingerprint(
        tail_estimate=0.01,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=turning_points,
        dominant_fft_frequency_mhz=0.35 if dominant_fft_snr > 0.0 else 0.0,
        dominant_fft_snr=dominant_fft_snr,
        dominant_fft_cycles_in_window=cycles,
        monotonic_decay_fraction=0.7,
        early_time_curvature=-0.1,
        semilog_slope_ratio=1.2,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=oscillatory_hint,
        kt_like_hint=False,
        multi_rate_hint=False,
    )


def _assessment_with_diagnostics(
    datasets: list[MuonDataset],
    diagnostics: list[RunResidualDiagnostic],
) -> GlobalCandidateAssessment:
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    fit_results = {
        int(dataset.run_number): FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [
                    Parameter("A_1", value=0.2, min=0.0, max=1.0),
                    Parameter("Lambda", value=0.3, min=0.0, max=5.0),
                    Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                ]
            ),
            message="ok",
        )
        for dataset in datasets
    }
    return GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=fit_results[int(datasets[0].run_number)].parameters,
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=tuple(diagnostics),
        series_warnings=(),
        aic=10.0,
        aicc=10.0,
        bic=12.0,
        selected_score=10.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )


def test_global_fit_wizard_prefers_shared_exponential_for_uniform_series() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=100 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)

    assert recommendation.recommended_key == "exp_constant"
    assert recommendation.recommended_assessment is not None
    assert recommendation.recommended_assessment.local_param_names == ()


def test_global_fit_wizard_localizes_lambda_when_series_rate_varies() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    lambdas = [0.15, 0.25, 0.55, 0.9]
    datasets = [
        _dataset_for(
            run_number=200 + idx,
            field=100.0 * idx,
            temperature=10.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)
    assessment = recommendation.recommended_assessment

    assert recommendation.recommended_key == "exp_constant"
    assert assessment is not None
    assert "Lambda" in assessment.local_param_names
    assert "A_1" not in assessment.local_param_names


def test_global_fit_wizard_staged_v2_records_search_instrumentation() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    lambdas = [0.15, 0.25, 0.55, 0.9]
    datasets = [
        _dataset_for(
            run_number=260 + idx,
            field=100.0 * idx,
            temperature=10.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]
    instrumentation: dict[str, object] = {}

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        metric=SelectionMetric.BIC,
        search_strategy="staged_v2",
        instrumentation=instrumentation,
    )
    assessment = recommendation.recommended_assessment

    assert recommendation.recommended_key == "exp_constant"
    assert assessment is not None
    counters = instrumentation.get("counters")
    assert isinstance(counters, dict)
    assert counters.get("exact_fit_invocations", 0) > 0
    assert counters.get("global_fit_calls", 0) > 0
    assert counters.get("curvature_hint_applications", 0) > 0
    assert counters.get("minuit_function_calls", 0) > 0
    assert instrumentation.get("strategy") == "staged_v2"
    assert isinstance(instrumentation.get("staged_frontier_widths"), list)
    assert isinstance(instrumentation.get("relaxed_penalties"), list)
    assert isinstance(instrumentation.get("curvature_hint_sizes"), list)
    assert isinstance(instrumentation.get("minuit_edm"), list)


def test_single_run_prefit_parameter_sets_reuses_cache() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = [
        _dataset_for(
            run_number=610 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35 + (0.05 * idx), "A_bg": 0.01},
        )
        for idx in range(2)
    ]
    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", 0.2, min=0.0, max=1.0),
                Parameter("Lambda", 0.4, min=0.0, max=5.0),
                Parameter("A_bg", 0.01, min=-0.5, max=0.5),
            ]
        )
        for dataset in datasets
    }
    fit_engine = FitEngine()
    fit_calls: list[tuple[int, str]] = []

    def _fake_fit(dataset, _function, params, *, method="migrad"):
        fit_calls.append((int(dataset.run_number), method))
        cloned = ParameterSet(
            [
                Parameter(parameter.name, parameter.value, min=parameter.min, max=parameter.max)
                for parameter in params
            ]
        )
        return FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=cloned,
            message="ok",
        )

    fit_engine.fit = _fake_fit  # type: ignore[method-assign]
    cache: dict[object, dict[int, ParameterSet]] = {}

    first = _single_run_prefit_parameter_sets(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        fixed_param_names=(),
        cache=cache,
    )
    calls_after_first = len(fit_calls)
    second = _single_run_prefit_parameter_sets(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        fixed_param_names=(),
        cache=cache,
    )

    assert calls_after_first > 0
    assert len(fit_calls) == calls_after_first
    assert sorted(first) == sorted(second)


def test_warm_start_parameter_sets_reuses_cache() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = [
        _dataset_for(
            run_number=710 + idx,
            field=25.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + (0.1 * idx), "A_bg": 0.01},
        )
        for idx in range(2)
    ]
    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", 0.2, min=0.0, max=1.0),
                Parameter("Lambda", 0.4, min=0.0, max=5.0),
                Parameter("A_bg", 0.01, min=-0.5, max=0.5),
            ]
        )
        for dataset in datasets
    }
    fit_results = {
        int(dataset.run_number): FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [
                    Parameter("A_1", 0.2, min=0.0, max=1.0),
                    Parameter("Lambda", 0.25 + (0.1 * idx), min=0.0, max=5.0),
                    Parameter("A_bg", 0.01, min=-0.5, max=0.5),
                ]
            ),
            message="ok",
        )
        for idx, dataset in enumerate(datasets)
    }
    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=fit_results[int(datasets[0].run_number)].parameters,
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=(),
        series_warnings=(),
        aic=10.0,
        aicc=10.0,
        bic=10.0,
        selected_score=10.0,
        fitted_curves_by_run={},
        component_curves_by_run={},
    )
    fit_engine = FitEngine()
    fit_calls: list[int] = []

    def _fake_fit(dataset, _function, params, *, method="migrad"):
        fit_calls.append(int(dataset.run_number))
        cloned = ParameterSet(
            [
                Parameter(parameter.name, parameter.value, min=parameter.min, max=parameter.max)
                for parameter in params
            ]
        )
        return FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=cloned,
            message="ok",
        )

    fit_engine.fit = _fake_fit  # type: ignore[method-assign]
    cache: dict[object, dict[int, ParameterSet]] = {}

    first = _warm_start_parameter_sets(
        datasets,
        assessment=assessment,
        base_by_run=base_by_run,
        target_global_names=("A_1", "A_bg"),
        target_local_names=("Lambda",),
        fit_engine=fit_engine,
        template=template,
        cache=cache,
    )
    calls_after_first = len(fit_calls)
    second = _warm_start_parameter_sets(
        datasets,
        assessment=assessment,
        base_by_run=base_by_run,
        target_global_names=("A_1", "A_bg"),
        target_local_names=("Lambda",),
        fit_engine=fit_engine,
        template=template,
        cache=cache,
    )

    assert calls_after_first == len(datasets)
    assert len(fit_calls) == calls_after_first
    assert sorted(first) == sorted(second)


def test_global_fit_wizard_only_builds_detailed_role_recommendations_for_top_candidates(
    monkeypatch,
) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    lambdas = [0.15, 0.25, 0.55, 0.9]
    datasets = [
        _dataset_for(
            run_number=760 + idx,
            field=100.0 * idx,
            temperature=10.0,
            model=model,
            params={"A_1": 0.2, "Lambda": lambdas[idx - 1], "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]
    built_for: list[str] = []

    def _fake_build_parameter_recommendations(*args, **kwargs):
        assessment = args[1]
        built_for.append(assessment.template.key)
        return ()

    monkeypatch.setattr(
        global_fit_wizard_module,
        "_build_parameter_recommendations",
        _fake_build_parameter_recommendations,
    )

    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        search_strategy="staged_v2",
    )

    assert recommendation.recommended_assessment is not None
    assert 1 <= len(built_for) <= 2


def test_global_fit_wizard_warns_for_mixed_field_temperature_grid() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=300 + idx,
            field=50.0 * idx,
            temperature=5.0 * idx,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.35, "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)

    assert recommendation.mixed_axes_warning is not None
    assert recommendation.recommended_key is None


def test_global_fit_wizard_flags_abrupt_regime_changes() -> None:
    exp_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    osc_model = CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
    datasets = [
        _dataset_for(
            run_number=400,
            field=0.0,
            temperature=2.0,
            model=exp_model,
            params={"A_1": 0.2, "Lambda": 0.25, "A_bg": 0.01},
        ),
        _dataset_for(
            run_number=401,
            field=100.0,
            temperature=2.0,
            model=exp_model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        ),
        _dataset_for(
            run_number=402,
            field=200.0,
            temperature=2.0,
            model=osc_model,
            params={"A_1": 0.18, "frequency": 0.6, "phase": 0.0, "Lambda": 0.2, "A_bg": 0.01},
        ),
        _dataset_for(
            run_number=403,
            field=300.0,
            temperature=2.0,
            model=osc_model,
            params={"A_1": 0.18, "frequency": 0.7, "phase": 0.0, "Lambda": 0.2, "A_bg": 0.01},
        ),
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets)

    assert any(
        "Fingerprint features change abruptly" in warning
        for assessment in recommendation.assessments
        for warning in assessment.series_warnings
    )


def test_global_fit_wizard_reranks_existing_assessments() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=500 + idx,
            field=25.0 * idx,
            temperature=8.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.2 + 0.05 * idx, "A_bg": 0.01},
        )
        for idx in range(1, 5)
    ]

    recommendation = build_global_fit_wizard_recommendation(datasets, metric=SelectionMetric.AICC)
    reranked = rerank_global_fit_wizard_recommendation(recommendation, SelectionMetric.BIC)

    assert reranked.metric == SelectionMetric.BIC
    assert reranked.assessments == recommendation.assessments


def test_global_fit_wizard_symbols_are_exported() -> None:
    assert fitting_api.build_global_fit_wizard_recommendation is not None
    assert fitting_api.rerank_global_fit_wizard_recommendation is not None
    assert fitting_api.GlobalFitWizardRecommendation is not None
    assert fitting_api.GlobalCandidateAssessment is not None
    assert fitting_api.GlobalParameterRecommendation is not None


def test_localisation_penalty_prefers_rate_parameters_over_amplitudes() -> None:
    assert _localisation_penalty(("Lambda",)) < _localisation_penalty(("A_1",))
    assert _localisation_penalty(("A_1",)) < _localisation_penalty(("A_bg",))


def test_component_canonicalization_orders_biexponential_components_by_rate() -> None:
    template = CandidateTemplate(
        key="biexp_constant",
        title="Exponential + Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"]),
    )
    params = ParameterSet(
        [
            Parameter("A_1", value=0.1, min=0.0, max=1.0),
            Parameter("Lambda_1", value=0.2, min=0.0, max=10.0),
            Parameter("A_2", value=0.3, min=0.0, max=1.0),
            Parameter("Lambda_2", value=1.4, min=0.0, max=10.0),
            Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
        ]
    )

    canonical = _canonicalize_parameter_sets(
        {1: params},
        template=template,
        global_param_names=("A_1", "Lambda_1", "A_2", "Lambda_2", "A_bg"),
        local_param_names=(),
        fixed_param_names=(),
    )

    assert canonical[1]["Lambda_1"].value == 1.4
    assert canonical[1]["Lambda_2"].value == 0.2
    assert canonical[1]["A_1"].value == 0.3
    assert canonical[1]["A_2"].value == 0.1


def test_single_run_prefits_improve_staged_seed_parameters() -> None:
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="test",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    lambdas = {701: 0.15, 702: 0.30, 703: 0.55, 704: 0.85}
    datasets = [
        _dataset_for(
            run_number=run_number,
            field=25.0 * index,
            temperature=6.0,
            model=template.model,
            params={"A_1": 0.2, "Lambda": lambda_value, "A_bg": 0.01},
        )
        for index, (run_number, lambda_value) in enumerate(sorted(lambdas.items()), start=1)
    ]
    base_by_run = {
        run_number: ParameterSet(
            [
                Parameter("A_1", value=0.5, min=0.0, max=1.0),
                Parameter("Lambda", value=1.5, min=0.0, max=2.0),
                Parameter("A_bg", value=0.1, min=-0.2, max=0.2),
            ]
        )
        for run_number in lambdas
    }

    prefitted = _single_run_prefit_parameter_sets(
        datasets,
        template,
        fit_engine=FitEngine(),
        base_by_run=base_by_run,
        fixed_param_names=(),
    )

    initial_error = sum(
        abs(base_by_run[run_number]["Lambda"].value - truth)
        for run_number, truth in lambdas.items()
    )
    fitted_error = sum(
        abs(prefitted[run_number]["Lambda"].value - truth)
        for run_number, truth in lambdas.items()
    )

    assert fitted_error < initial_error


def test_oscillatory_rescue_requires_repeated_aligned_evidence() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=600 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        )
        for idx in range(6)
    ]
    diagnostics = [
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.3,
            runs_z_score=-3.1 if idx < 3 else 0.2,
            max_abs_autocorrelation=0.2,
            residual_fft_peak_snr=7.4 if idx < 3 else 1.2,
            gate_passed=idx >= 3,
            gate_reasons=(),
        )
        for idx, dataset in enumerate(datasets)
    ]
    fingerprints = {
        int(dataset.run_number): (
            _fingerprint(dominant_fft_snr=5.2, cycles=2.0, turning_points=2)
            if idx < 3
            else _fingerprint()
        )
        for idx, dataset in enumerate(datasets)
    }

    supported_runs = _supported_oscillatory_run_numbers(
        datasets,
        assessment=_assessment_with_diagnostics(datasets, diagnostics),
        fingerprints_by_run=fingerprints,
    )

    assert supported_runs == tuple(int(dataset.run_number) for dataset in datasets[:3])


def test_oscillatory_rescue_ignores_isolated_fft_peaks() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    datasets = [
        _dataset_for(
            run_number=700 + idx,
            field=50.0 * idx,
            temperature=5.0,
            model=model,
            params={"A_1": 0.2, "Lambda": 0.3, "A_bg": 0.01},
        )
        for idx in range(6)
    ]
    diagnostics = [
        RunResidualDiagnostic(
            run_number=int(dataset.run_number),
            run_label=dataset.run_label,
            axis_value=float(dataset.metadata["field"]),
            residual_rms=0.3,
            runs_z_score=-3.3 if idx == 2 else 0.1,
            max_abs_autocorrelation=0.2,
            residual_fft_peak_snr=7.8 if idx == 2 else 1.0,
            gate_passed=idx != 2,
            gate_reasons=(),
        )
        for idx, dataset in enumerate(datasets)
    ]
    fingerprints = {
        int(dataset.run_number): (
            _fingerprint(dominant_fft_snr=5.4, cycles=2.1, turning_points=2)
            if idx == 2
            else _fingerprint()
        )
        for idx, dataset in enumerate(datasets)
    }

    supported_runs = _supported_oscillatory_run_numbers(
        datasets,
        assessment=_assessment_with_diagnostics(datasets, diagnostics),
        fingerprints_by_run=fingerprints,
    )

    assert supported_runs == ()


def test_staged_multi_local_assignment_succeeds_for_large_biexp_gaussian_series() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Gaussian", "Constant"],
        operators=["+", "+", "+"],
    )
    template = CandidateTemplate(
        key="double_exp_gaussian_constant",
        title="Exponential + Exponential + Gaussian + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = []
    for index in range(19):
        run_number = 8000 + index
        datasets.append(
            _dense_dataset_for(
                run_number=run_number,
                field=10.0 * index,
                temperature=5.0,
                model=model,
                params={
                    "A_1": 0.11,
                    "Lambda_1": 0.10 + 0.012 * index,
                    "A_2": 0.06,
                    "Lambda_2": 0.58 + 0.035 * index,
                    "A_3": 0.05,
                    "sigma": 0.18 + 0.022 * index,
                    "A_bg": 0.01,
                },
            )
        )

    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", value=0.14, min=0.0, max=1.0),
                Parameter("Lambda_1", value=0.18, min=0.0, max=2.0),
                Parameter("A_2", value=0.04, min=0.0, max=1.0),
                Parameter("Lambda_2", value=0.95, min=0.0, max=2.0),
                Parameter("A_3", value=0.04, min=0.0, max=1.0),
                Parameter("sigma", value=0.38, min=0.0, max=2.0),
                Parameter("A_bg", value=0.015, min=-0.2, max=0.2),
            ]
        )
        for dataset in datasets
    }
    fit_engine = FitEngine()
    baseline = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        global_param_names=tuple(model.param_names),
        local_param_names=(),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
    )

    assessment, best_partial = _staged_multi_local_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        baseline_assessment=baseline,
        target_local_names=("Lambda_1", "Lambda_2", "sigma"),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={(baseline.global_param_names, baseline.local_param_names): baseline},
    )

    assert baseline.is_successful
    assert best_partial.is_successful
    assert assessment is not None
    assert assessment.is_successful
    assert assessment.local_param_names == ("Lambda_1", "Lambda_2", "sigma")


def test_staged_multi_local_assignment_succeeds_for_large_exp_double_gaussian_series() -> None:
    model = CompositeModel(
        ["Exponential", "Gaussian", "Gaussian", "Constant"],
        operators=["+", "+", "+"],
    )
    template = CandidateTemplate(
        key="exp_double_gaussian_constant",
        title="Exponential + Gaussian + Gaussian + Constant",
        category="General",
        rationale="test",
        model=model,
    )
    datasets = []
    for index in range(19):
        run_number = 8100 + index
        datasets.append(
            _dense_dataset_for(
                run_number=run_number,
                field=12.5 * index,
                temperature=5.0,
                model=model,
                params={
                    "A_1": 0.10,
                    "Lambda": 0.08 + 0.010 * index,
                    "A_2": 0.05,
                    "sigma_2": 0.15 + 0.018 * index,
                    "A_3": 0.05,
                    "sigma_3": 0.42 + 0.028 * index,
                    "A_bg": 0.012,
                },
            )
        )

    base_by_run = {
        int(dataset.run_number): ParameterSet(
            [
                Parameter("A_1", value=0.13, min=0.0, max=1.0),
                Parameter("Lambda", value=0.16, min=0.0, max=2.0),
                Parameter("A_2", value=0.04, min=0.0, max=1.0),
                Parameter("sigma_2", value=0.24, min=0.0, max=2.0),
                Parameter("A_3", value=0.04, min=0.0, max=1.0),
                Parameter("sigma_3", value=0.62, min=0.0, max=2.0),
                Parameter("A_bg", value=0.018, min=-0.2, max=0.2),
            ]
        )
        for dataset in datasets
    }
    fit_engine = FitEngine()
    baseline = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        global_param_names=tuple(model.param_names),
        local_param_names=(),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={},
    )

    assessment, best_partial = _staged_multi_local_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        baseline_assessment=baseline,
        target_local_names=("Lambda", "sigma_2", "sigma_3"),
        fixed_param_names=(),
        axis_key="field",
        metric=SelectionMetric.AICC,
        cache={(baseline.global_param_names, baseline.local_param_names): baseline},
    )

    assert baseline.is_successful
    assert best_partial.is_successful
    assert assessment is not None
    assert assessment.is_successful
    assert assessment.local_param_names == ("Lambda", "sigma_2", "sigma_3")
