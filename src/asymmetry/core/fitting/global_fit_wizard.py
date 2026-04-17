"""Core analysis helpers for the global fit wizard."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    SelectionMetric,
    SpectrumFingerprint,
    _bound_hit_names,
    _clone_parameter_set,
    _dense_fit_curves,
    _initial_parameters_for_template,
    _is_additive_relaxation_mixture_template,
    _parameter_variants,
    _residual_diagnostics,
    _residual_gate_reasons,
    build_candidate_templates,
    compute_information_criteria,
    fingerprint_spectrum,
)
from asymmetry.core.fitting.global_search import (
    GlobalSearchConfig,
    GlobalSearchOrchestrator,
    build_parameter_sets_for_structure,
    compile_legacy_structure,
    compile_structure_to_legacy_roles,
    score_exact_candidate,
)
from asymmetry.core.fitting.global_search.heuristics import parameter_localisation_priority
from asymmetry.core.fitting.global_search.refine import SearchEvaluation
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

_ROLE_DELTA_THRESHOLD = 2.0
_COMPARABLE_SCORE_DELTA = 2.0
_SHORTLIST_COUNT = 4
_SHORTLIST_SCORE_WINDOW = 6.0
_SHORTLIST_CAP = 6
_GLOBAL_FIT_MAX_CALLS = 1200
_GLOBAL_FIT_MAX_CALLS_CAP = 4200
_LOW_RESIDUAL_RMS_FOR_STRUCTURE_WARNINGS = 0.25
_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS = 1800
_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS_CAP = 5600
_STAGED_GLOBAL_FIT_MAX_CALLS = 4200
_STAGED_LOCAL_SEARCH_BEAM_WIDTH = 3
_STAGED_LOCAL_SEARCH_CANDIDATES_PER_BRANCH = 2
_STAGED_V2_LOCAL_SEARCH_BEAM_WIDTH = 5
_STAGED_V2_LOCAL_SEARCH_CANDIDATES_PER_BRANCH = 3
_STAGED_V2_EXACT_CANDIDATES_PER_TIER = 2
_MAX_ROLE_CANDIDATES_PER_TIER = 3
_HIGH_DIMENSION_FREE_COUNT = 40
_EXTREME_DIMENSION_FREE_COUNT = 70
_MAX_TEMPLATE_WORKERS = 4
_OSCILLATORY_RESCUE_RESIDUAL_FFT_SNR = 6.0
_OSCILLATORY_RESCUE_MEDIAN_FFT_SNR = 6.5
_OSCILLATORY_RESCUE_RUNS_Z = 2.5
_OSCILLATORY_RESCUE_FINGERPRINT_FFT_SNR = 4.5
_OSCILLATORY_RESCUE_FINGERPRINT_MIN_CYCLES = 1.75
_OSCILLATORY_RESCUE_FINGERPRINT_MIN_TURNS = 2
_OSCILLATORY_RESCUE_MIN_RUNS = 3
_OSCILLATORY_RESCUE_MIN_FRACTION = 0.25
_OSCILLATORY_RESCUE_MIN_CLUSTER = 2
_OSCILLATORY_RESCUE_MAX_SCOUTS = 3


@dataclass(frozen=True)
class RunResidualDiagnostic:
    """Residual diagnostics for one dataset inside a global-fit candidate."""

    run_number: int
    run_label: str
    axis_value: float
    residual_rms: float
    runs_z_score: float
    max_abs_autocorrelation: float
    residual_fft_peak_snr: float
    gate_passed: bool
    gate_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GlobalParameterRecommendation:
    """Recommended role for one parameter in a global fit."""

    name: str
    recommended_role: str
    global_score: float
    local_score: float
    score_delta: float
    total_variation: float
    roughness: float
    rationale: str


@dataclass(frozen=True)
class GlobalCandidateAssessment:
    """Fit and comparison data for one global-fit candidate."""

    template: CandidateTemplate
    fit_results_by_run: dict[int, FitResult]
    global_parameters: ParameterSet
    global_param_names: tuple[str, ...]
    local_param_names: tuple[str, ...]
    fixed_param_names: tuple[str, ...]
    parameter_recommendations: tuple[GlobalParameterRecommendation, ...]
    run_diagnostics: tuple[RunResidualDiagnostic, ...]
    series_warnings: tuple[str, ...]
    aic: float
    aicc: float | None
    bic: float
    selected_score: float
    fitted_curves_by_run: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]
    component_curves_by_run: dict[int, tuple[tuple[str, NDArray[np.float64]], ...]]

    @property
    def parameter_count(self) -> int:
        return len(self.global_param_names) + (
            len(self.local_param_names) * len(self.fit_results_by_run)
        )

    @property
    def additive_terms(self) -> int:
        return self.template.additive_terms

    def metric_value(self, metric: SelectionMetric) -> float:
        if metric == SelectionMetric.AIC:
            return self.aic
        if metric == SelectionMetric.BIC:
            return self.bic
        if self.aicc is not None and np.isfinite(self.aicc):
            return self.aicc
        return self.aic

    @property
    def is_successful(self) -> bool:
        return bool(self.fit_results_by_run) and all(
            result.success for result in self.fit_results_by_run.values()
        )

    @property
    def residual_gate_passed(self) -> bool:
        return (
            all(diagnostic.gate_passed for diagnostic in self.run_diagnostics)
            and not self.series_warnings
        )


@dataclass(frozen=True)
class GlobalFitWizardRecommendation:
    """Global-fit analysis payload plus the current recommendation."""

    series_axis_key: str
    series_axis_label: str
    mixed_axes_warning: str | None
    fingerprints_by_run: dict[int, SpectrumFingerprint]
    dataset_order: tuple[int, ...]
    templates: tuple[CandidateTemplate, ...]
    assessments: tuple[GlobalCandidateAssessment, ...]
    metric: SelectionMetric
    recommended_key: str | None
    comparable_keys: tuple[str, ...]
    summary: str

    @property
    def recommended_assessment(self) -> GlobalCandidateAssessment | None:
        if not self.recommended_key:
            return None
        for assessment in self.assessments:
            if assessment.template.key == self.recommended_key:
                return assessment
        return None

    def assessment_for_key(self, key: str | None) -> GlobalCandidateAssessment | None:
        if not isinstance(key, str):
            return None
        for assessment in self.assessments:
            if assessment.template.key == key:
                return assessment
        return None

    def sorted_assessments(
        self,
        metric: SelectionMetric | None = None,
    ) -> list[GlobalCandidateAssessment]:
        active_metric = metric or self.metric
        return sorted(
            self.assessments,
            key=lambda assessment: _assessment_sort_key(assessment, active_metric),
        )


@dataclass(frozen=True)
class GlobalFitWizardCandidatePortfolio:
    """Cheap pre-analysis portfolio detection for the global fit wizard."""

    ordered_datasets: tuple[MuonDataset, ...]
    series_axis_key: str
    series_axis_label: str
    mixed_axes_warning: str | None
    fingerprints_by_run: dict[int, SpectrumFingerprint]
    templates: tuple[CandidateTemplate, ...]

    @property
    def dataset_order(self) -> tuple[int, ...]:
        return tuple(int(dataset.run_number) for dataset in self.ordered_datasets)


def build_global_fit_wizard_candidate_portfolio(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
) -> GlobalFitWizardCandidatePortfolio:
    """Return the ordered datasets, fingerprints, and candidate families for one series."""
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    ordered_datasets, axis_key, axis_label, mixed_axes_warning = _ordered_datasets_with_axis(
        datasets
    )
    fingerprints_by_run = {
        int(dataset.run_number): fingerprint_spectrum(dataset) for dataset in ordered_datasets
    }
    aggregate_fingerprint = _aggregate_fingerprints(
        [fingerprints_by_run[int(dataset.run_number)] for dataset in ordered_datasets]
    )
    templates = tuple(
        build_candidate_templates(
            aggregate_fingerprint,
            current_model=current_model,
        )
    )
    return GlobalFitWizardCandidatePortfolio(
        ordered_datasets=tuple(ordered_datasets),
        series_axis_key=axis_key,
        series_axis_label=axis_label,
        mixed_axes_warning=mixed_axes_warning,
        fingerprints_by_run=fingerprints_by_run,
        templates=templates,
    )


def _record_counter(
    instrumentation: dict[str, object] | None,
    name: str,
    delta: int = 1,
) -> None:
    if instrumentation is None:
        return
    counters = instrumentation.setdefault("counters", {})
    if not isinstance(counters, dict):
        return
    counters[name] = int(counters.get(name, 0)) + int(delta)


def _append_metric(
    instrumentation: dict[str, object] | None,
    name: str,
    value: object,
) -> None:
    if instrumentation is None:
        return
    values = instrumentation.setdefault(name, [])
    if isinstance(values, list):
        values.append(value)


def _set_metric(
    instrumentation: dict[str, object] | None,
    name: str,
    value: object,
) -> None:
    if instrumentation is None:
        return
    instrumentation[name] = value


def _positive_uncertainty(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = abs(float(value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _step_hints_from_fit_results(
    datasets: list[MuonDataset],
    results_by_run: dict[int, FitResult],
    *,
    target_global_names: tuple[str, ...],
    target_local_names: tuple[str, ...],
    source_local_names: set[str] | None = None,
) -> dict[str, float]:
    step_hints: dict[str, float] = {}

    for name in target_global_names:
        collected: list[float] = []
        for dataset in datasets:
            result = results_by_run.get(int(dataset.run_number))
            if result is None:
                continue
            uncertainty = _positive_uncertainty(result.uncertainties.get(name))
            if uncertainty is not None:
                collected.append(uncertainty)
        if collected:
            step_hints[name] = float(np.median(np.asarray(collected, dtype=float)))

    for dataset in datasets:
        run_number = int(dataset.run_number)
        result = results_by_run.get(run_number)
        if result is None:
            continue
        for name in target_local_names:
            if source_local_names is not None and name not in source_local_names:
                continue
            uncertainty = _positive_uncertainty(result.uncertainties.get(name))
            if uncertainty is None:
                uncertainty = step_hints.get(name)
            if uncertainty is not None:
                step_hints[f"{name}_{run_number}"] = float(uncertainty)

    return step_hints


def _step_hints_from_assessment(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment | None,
    *,
    target_global_names: tuple[str, ...],
    target_local_names: tuple[str, ...],
) -> dict[str, float]:
    if assessment is None or not assessment.is_successful:
        return {}
    return _step_hints_from_fit_results(
        datasets,
        assessment.fit_results_by_run,
        target_global_names=target_global_names,
        target_local_names=target_local_names,
        source_local_names=set(assessment.local_param_names),
    )


def _record_global_fit_diagnostics(
    instrumentation: dict[str, object] | None,
    results_by_run: dict[int, FitResult],
) -> None:
    if instrumentation is None or not results_by_run:
        return
    first_result = next(iter(results_by_run.values()))
    _record_counter(instrumentation, "minuit_function_calls", first_result.function_calls)
    _record_counter(instrumentation, "minuit_gradient_calls", first_result.gradient_calls)
    _record_counter(instrumentation, "minuit_hessian_calls", first_result.hessian_calls)
    if first_result.edm is not None:
        _append_metric(instrumentation, "minuit_edm", float(first_result.edm))
    if first_result.covariance_accurate:
        _record_counter(instrumentation, "accurate_covariance_fits")


def _staged_orchestrator_config(
    *,
    search_strategy: str,
    metric: SelectionMetric,
    instrumentation: dict[str, object] | None,
) -> GlobalSearchConfig:
    if search_strategy == "staged_v2":
        return GlobalSearchConfig(
            metric=metric,
            deviation_threshold=0.045,
            ambiguity_band=0.025,
            activity_threshold=0.015,
            max_steps=10,
            max_neighbors=10,
            beam_width=3,
            max_exact_evaluations_per_step=6,
            max_alternates=4,
            active_set_threshold=0.012,
            penalty_schedule=(),
            allow_backward_moves=True,
            instrumentation=instrumentation,
        )
    return GlobalSearchConfig(metric=SelectionMetric.BIC, instrumentation=instrumentation)


def _staged_local_search_settings(search_strategy: str) -> tuple[int, int, bool, int | None]:
    if search_strategy == "staged_v2":
        return (
            _STAGED_V2_LOCAL_SEARCH_BEAM_WIDTH,
            _STAGED_V2_LOCAL_SEARCH_CANDIDATES_PER_BRANCH,
            True,
            _STAGED_V2_EXACT_CANDIDATES_PER_TIER,
        )
    return (
        _STAGED_LOCAL_SEARCH_BEAM_WIDTH,
        _STAGED_LOCAL_SEARCH_CANDIDATES_PER_BRANCH,
        False,
        None,
    )


def _build_global_fit_wizard_recommendation_legacy(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    current_parameter_types: dict[str, str] | None = None,
    current_values: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    metric: SelectionMetric = SelectionMetric.AICC,
    progress_callback: Callable[[str], None] | None = None,
) -> GlobalFitWizardRecommendation:
    """Analyze one ordered dataset series and recommend a global-fit candidate."""
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    progress_callback = _threadsafe_progress_callback(progress_callback)

    _progress_log(
        progress_callback,
        f"Preparing global fit wizard analysis for {len(datasets)} datasets.",
    )
    (
        ordered_datasets,
        axis_key,
        axis_label,
        mixed_axes_warning,
    ) = _ordered_datasets_with_axis(datasets)
    _progress_log(
        progress_callback,
        f"Detected ordered series axis: {axis_label}.",
    )
    fingerprints_by_run = {
        int(dataset.run_number): fingerprint_spectrum(dataset) for dataset in ordered_datasets
    }
    aggregate_fingerprint = _aggregate_fingerprints(
        [fingerprints_by_run[int(dataset.run_number)] for dataset in ordered_datasets]
    )
    templates = list(
        build_candidate_templates(
            aggregate_fingerprint,
            current_model=current_model,
        )
    )
    _progress_log(
        progress_callback,
        f"Built candidate portfolio with {len(templates)} model families.",
    )

    if mixed_axes_warning:
        _progress_log(progress_callback, mixed_axes_warning)
        return replace(
            GlobalFitWizardRecommendation(
                series_axis_key=axis_key,
                series_axis_label=axis_label,
                mixed_axes_warning=mixed_axes_warning,
                fingerprints_by_run=fingerprints_by_run,
                dataset_order=tuple(int(dataset.run_number) for dataset in ordered_datasets),
                templates=tuple(templates),
                assessments=(),
                metric=metric,
                recommended_key=None,
                comparable_keys=(),
                summary=mixed_axes_warning,
            ),
            metric=metric,
        )

    current_parameter_types = current_parameter_types or {}
    current_values = current_values or {}
    parameter_bounds = parameter_bounds or {}

    initial_assessments: dict[str, GlobalCandidateAssessment] = {}
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]] = {}
    recommendation_contexts: dict[
        str,
        tuple[
            CandidateTemplate,
            dict[int, ParameterSet],
            tuple[str, ...],
            set[str],
            dict[
                tuple[
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
                ],
                dict[int, ParameterSet],
            ],
        ],
    ] = {}
    single_run_prefit_caches: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...], tuple[bool, ...]],
        dict[
            tuple[tuple[str, ...], tuple[tuple[int, tuple[tuple[str, float], ...]], ...]],
            dict[int, ParameterSet],
        ],
    ] = {}
    warm_start_caches: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...], tuple[bool, ...]],
        dict[
            tuple[
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
            ],
            dict[int, ParameterSet],
        ],
    ] = {}

    def _formula_signature_for_template(
        eval_template: CandidateTemplate,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[bool, ...],
        tuple[bool, ...],
    ]:
        return (
            tuple(eval_template.model.component_names),
            tuple(eval_template.model.operators),
            tuple(eval_template.model.open_parentheses),
            tuple(eval_template.model.close_parentheses),
        )

    def _single_run_prefit_cache_for(
        eval_template: CandidateTemplate,
    ) -> dict[
        tuple[tuple[str, ...], tuple[tuple[int, tuple[tuple[str, float], ...]], ...]],
        dict[int, ParameterSet],
    ]:
        return single_run_prefit_caches.setdefault(
            _formula_signature_for_template(eval_template),
            {},
        )

    def _warm_start_cache_for(
        eval_template: CandidateTemplate,
    ) -> dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ]:
        return warm_start_caches.setdefault(
            _formula_signature_for_template(eval_template),
            {},
        )

    def _initial_screen_task(
        template: CandidateTemplate,
    ) -> tuple[str, dict[int, ParameterSet], tuple[str, ...], GlobalCandidateAssessment]:
        fixed_param_names = _fixed_param_names(template, current_parameter_types)
        base_by_run = _initial_parameter_sets_for_candidate(
            ordered_datasets,
            fingerprints_by_run,
            template,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            fixed_param_names=fixed_param_names,
        )
        template_contexts[template.key] = (base_by_run, fixed_param_names)
        initial_global_names, initial_local_names = _initial_parameter_roles(
            template,
            current_parameter_types=current_parameter_types,
            fixed_param_names=fixed_param_names,
        )
        assignment_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            GlobalCandidateAssessment,
        ] = {}
        assessment = _fit_exact_assignment(
            ordered_datasets,
            template,
            fit_engine=FitEngine(),
            base_by_run=base_by_run,
            global_param_names=initial_global_names,
            local_param_names=initial_local_names,
            fixed_param_names=fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache=assignment_cache,
            progress_callback=progress_callback,
        )
        return template.key, base_by_run, fixed_param_names, assessment

    template_workers = _template_worker_count(len(templates))
    if template_workers <= 1:
        for index, template in enumerate(templates, start=1):
            _progress_log(
                progress_callback,
                f"Initial screening {index}/{len(templates)}: {template.title}.",
            )
            key, base_by_run, fixed_param_names, assessment = _initial_screen_task(template)
            template_contexts[key] = (base_by_run, fixed_param_names)
            initial_assessments[key] = assessment
    else:
        _progress_log(
            progress_callback,
            f"Running initial screening with {template_workers} parallel workers.",
        )
        with ThreadPoolExecutor(
            max_workers=template_workers,
            thread_name_prefix="global-fit-screen",
        ) as executor:
            future_to_template = {}
            for index, template in enumerate(templates, start=1):
                _progress_log(
                    progress_callback,
                    f"Initial screening {index}/{len(templates)}: {template.title}.",
                )
                future_to_template[executor.submit(_initial_screen_task, template)] = template
            for future in as_completed(future_to_template):
                key, base_by_run, fixed_param_names, assessment = future.result()
                template_contexts[key] = (base_by_run, fixed_param_names)
                initial_assessments[key] = assessment

    forced_shortlist_keys = _maybe_expand_oscillatory_shortlist(
        ordered_datasets,
        templates=templates,
        aggregate_fingerprint=aggregate_fingerprint,
        current_model=current_model,
        fit_engine=FitEngine(),
        initial_assessments=initial_assessments,
        template_contexts=template_contexts,
        fingerprints_by_run=fingerprints_by_run,
        current_parameter_types=current_parameter_types,
        current_values=current_values,
        parameter_bounds=parameter_bounds,
        axis_key=axis_key,
        metric=metric,
        progress_callback=progress_callback,
    )
    shortlist_keys = _shortlist_template_keys(
        tuple(templates),
        initial_assessments=initial_assessments,
        metric=metric,
        forced_keys=forced_shortlist_keys,
    )
    shortlisted_titles = [
        template.title for template in templates if template.key in shortlist_keys
    ]
    _progress_log(
        progress_callback,
        "Shortlisted candidates for full parameter-role search: "
        + ", ".join(shortlisted_titles),
    )

    def _full_template_search_task(template: CandidateTemplate) -> GlobalCandidateAssessment:
        initial_assessment = initial_assessments[template.key]
        base_by_run, fixed_param_names = template_contexts[template.key]
        initial_global_names, initial_local_names = _initial_parameter_roles(
            template,
            current_parameter_types=current_parameter_types,
            fixed_param_names=fixed_param_names,
        )
        fit_engine = FitEngine()
        assignment_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            GlobalCandidateAssessment,
        ] = {
            (
                initial_assessment.global_param_names,
                initial_assessment.local_param_names,
            ): initial_assessment
        }
        shortlisted = _search_parameter_roles(
            ordered_datasets,
            template,
            fit_engine=fit_engine,
            base_by_run=base_by_run,
            initial_global_names=initial_global_names,
            initial_local_names=initial_local_names,
            fixed_param_names=fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache=assignment_cache,
            progress_callback=progress_callback,
        )
        return replace(shortlisted, parameter_recommendations=())

    assessments_by_key: dict[str, GlobalCandidateAssessment] = {
        template.key: initial_assessments[template.key]
        for template in templates
        if template.key not in shortlist_keys
    }
    shortlisted_templates = [
        template for template in templates if template.key in shortlist_keys
    ]
    shortlisted_workers = _template_worker_count(len(shortlisted_templates))
    if shortlisted_workers <= 1:
        for template in shortlisted_templates:
            _progress_log(
                progress_callback,
                f"Searching Global/Local parameter roles for {template.title}.",
            )
            assessments_by_key[template.key] = _full_template_search_task(template)
    else:
        _progress_log(
            progress_callback,
            "Running full parameter-role search with "
            f"{shortlisted_workers} parallel workers.",
        )
        with ThreadPoolExecutor(
            max_workers=shortlisted_workers,
            thread_name_prefix="global-fit-search",
        ) as executor:
            future_to_template = {}
            for template in shortlisted_templates:
                _progress_log(
                    progress_callback,
                    f"Searching Global/Local parameter roles for {template.title}.",
                )
                future_to_template[executor.submit(_full_template_search_task, template)] = (
                    template
                )
            for future in as_completed(future_to_template):
                template = future_to_template[future]
                assessments_by_key[template.key] = future.result()

    recommendation = rerank_global_fit_wizard_recommendation(
        GlobalFitWizardRecommendation(
            series_axis_key=axis_key,
            series_axis_label=axis_label,
            mixed_axes_warning=mixed_axes_warning,
            fingerprints_by_run=fingerprints_by_run,
            dataset_order=tuple(int(dataset.run_number) for dataset in ordered_datasets),
            templates=tuple(templates),
            assessments=tuple(assessments_by_key[template.key] for template in templates),
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary="",
        ),
        metric,
    )
    for key in _parameter_recommendation_candidate_keys(recommendation):
        assessment = assessments_by_key.get(key)
        context = template_contexts.get(key)
        template = next((candidate for candidate in templates if candidate.key == key), None)
        if assessment is None or context is None or template is None:
            continue
        base_by_run, fixed_param_names = context
        _progress_log(
            progress_callback,
            f"Building per-parameter role recommendations for {template.title}.",
        )
        parameter_recommendations = _build_parameter_recommendations(
            ordered_datasets,
            assessment,
            template=template,
            fit_engine=FitEngine(),
            base_by_run=base_by_run,
            fixed_param_names=fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache={
                (assessment.global_param_names, assessment.local_param_names): assessment,
            },
            progress_callback=progress_callback,
        )
        assessments_by_key[key] = replace(
            assessment,
            parameter_recommendations=parameter_recommendations,
        )

    recommendation = rerank_global_fit_wizard_recommendation(
        replace(
            recommendation,
            assessments=tuple(assessments_by_key[template.key] for template in templates),
        ),
        metric,
    )
    if recommendation.recommended_assessment is not None:
        _progress_log(
            progress_callback,
            f"Recommended model: {recommendation.recommended_assessment.template.title}.",
        )
    else:
        _progress_log(progress_callback, recommendation.summary)
    return recommendation


def build_global_fit_wizard_recommendation(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    current_parameter_types: dict[str, str] | None = None,
    current_values: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    metric: SelectionMetric = SelectionMetric.AICC,
    progress_callback: Callable[[str], None] | None = None,
    search_strategy: str = "legacy",
    instrumentation: dict[str, object] | None = None,
) -> GlobalFitWizardRecommendation:
    """Analyze one ordered dataset series and recommend a global-fit candidate."""
    strategy = str(search_strategy).strip().lower()
    _set_metric(instrumentation, "strategy", strategy)
    if strategy in {"staged_v1", "staged_v2"}:
        return _build_global_fit_wizard_recommendation_staged(
            datasets,
            current_model=current_model,
            current_parameter_types=current_parameter_types,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            metric=metric,
            progress_callback=progress_callback,
            search_strategy=strategy,
            instrumentation=instrumentation,
        )
    return _build_global_fit_wizard_recommendation_legacy(
        datasets,
        current_model=current_model,
        current_parameter_types=current_parameter_types,
        current_values=current_values,
        parameter_bounds=parameter_bounds,
        metric=metric,
        progress_callback=progress_callback,
    )


def _build_global_fit_wizard_recommendation_staged(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    current_parameter_types: dict[str, str] | None = None,
    current_values: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    metric: SelectionMetric = SelectionMetric.AICC,
    progress_callback: Callable[[str], None] | None = None,
    search_strategy: str = "staged_v1",
    instrumentation: dict[str, object] | None = None,
) -> GlobalFitWizardRecommendation:
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    progress_callback = _threadsafe_progress_callback(progress_callback)
    current_parameter_types = current_parameter_types or {}
    current_values = current_values or {}
    parameter_bounds = parameter_bounds or {}

    _progress_log(
        progress_callback,
        f"Preparing {search_strategy} global fit wizard analysis for {len(datasets)} datasets.",
    )
    (
        ordered_datasets,
        axis_key,
        axis_label,
        mixed_axes_warning,
    ) = _ordered_datasets_with_axis(datasets)
    fingerprints_by_run = {
        int(dataset.run_number): fingerprint_spectrum(dataset) for dataset in ordered_datasets
    }
    aggregate_fingerprint = _aggregate_fingerprints(
        [fingerprints_by_run[int(dataset.run_number)] for dataset in ordered_datasets]
    )
    templates = list(
        build_candidate_templates(
            aggregate_fingerprint,
            current_model=current_model,
        )
    )
    if mixed_axes_warning:
        return replace(
            GlobalFitWizardRecommendation(
                series_axis_key=axis_key,
                series_axis_label=axis_label,
                mixed_axes_warning=mixed_axes_warning,
                fingerprints_by_run=fingerprints_by_run,
                dataset_order=tuple(int(dataset.run_number) for dataset in ordered_datasets),
                templates=tuple(templates),
                assessments=(),
                metric=metric,
                recommended_key=None,
                comparable_keys=(),
                summary=mixed_axes_warning,
            ),
            metric=metric,
        )

    initial_assessments: dict[str, GlobalCandidateAssessment] = {}
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]] = {}
    recommendation_contexts: dict[
        str,
        tuple[
            CandidateTemplate,
            dict[int, ParameterSet],
            tuple[str, ...],
            set[str],
            dict[
                tuple[
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
                ],
                dict[int, ParameterSet],
            ],
        ],
    ] = {}
    single_run_prefit_caches: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...], tuple[bool, ...]],
        dict[
            tuple[tuple[str, ...], tuple[tuple[int, tuple[tuple[str, float], ...]], ...]],
            dict[int, ParameterSet],
        ],
    ] = {}
    warm_start_caches: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...], tuple[bool, ...]],
        dict[
            tuple[
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
            ],
            dict[int, ParameterSet],
        ],
    ] = {}

    def _formula_signature_for_template(
        eval_template: CandidateTemplate,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[bool, ...],
        tuple[bool, ...],
    ]:
        return (
            tuple(eval_template.model.component_names),
            tuple(eval_template.model.operators),
            tuple(eval_template.model.open_parentheses),
            tuple(eval_template.model.close_parentheses),
        )

    def _single_run_prefit_cache_for(
        eval_template: CandidateTemplate,
    ) -> dict[
        tuple[tuple[str, ...], tuple[tuple[int, tuple[tuple[str, float], ...]], ...]],
        dict[int, ParameterSet],
    ]:
        return single_run_prefit_caches.setdefault(
            _formula_signature_for_template(eval_template),
            {},
        )

    def _warm_start_cache_for(
        eval_template: CandidateTemplate,
    ) -> dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ]:
        return warm_start_caches.setdefault(
            _formula_signature_for_template(eval_template),
            {},
        )

    def _initial_screen_task(
        template: CandidateTemplate,
    ) -> tuple[str, dict[int, ParameterSet], tuple[str, ...], GlobalCandidateAssessment]:
        fixed_param_names = _fixed_param_names(template, current_parameter_types)
        base_by_run = _initial_parameter_sets_for_candidate(
            ordered_datasets,
            fingerprints_by_run,
            template,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            fixed_param_names=fixed_param_names,
        )
        template_contexts[template.key] = (base_by_run, fixed_param_names)
        initial_global_names, initial_local_names = _initial_parameter_roles(
            template,
            current_parameter_types=current_parameter_types,
            fixed_param_names=fixed_param_names,
        )
        assessment = _fit_exact_assignment(
            ordered_datasets,
            template,
            fit_engine=FitEngine(),
            base_by_run=base_by_run,
            global_param_names=initial_global_names,
            local_param_names=initial_local_names,
            fixed_param_names=fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache={},
            progress_callback=progress_callback,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
        )
        return template.key, base_by_run, fixed_param_names, assessment

    template_workers = _template_worker_count(len(templates))
    if template_workers <= 1:
        for index, template in enumerate(templates, start=1):
            _progress_log(
                progress_callback,
                f"Initial screening {index}/{len(templates)}: {template.title}.",
            )
            key, base_by_run, fixed_param_names, assessment = _initial_screen_task(template)
            template_contexts[key] = (base_by_run, fixed_param_names)
            initial_assessments[key] = assessment
    else:
        _progress_log(
            progress_callback,
            f"Running staged initial screening with {template_workers} parallel workers.",
        )
        with ThreadPoolExecutor(
            max_workers=template_workers,
            thread_name_prefix="global-fit-staged-screen",
        ) as executor:
            future_to_template = {}
            for index, template in enumerate(templates, start=1):
                _progress_log(
                    progress_callback,
                    f"Initial screening {index}/{len(templates)}: {template.title}.",
                )
                future_to_template[executor.submit(_initial_screen_task, template)] = template
            for future in as_completed(future_to_template):
                key, base_by_run, fixed_param_names, assessment = future.result()
                template_contexts[key] = (base_by_run, fixed_param_names)
                initial_assessments[key] = assessment

    forced_shortlist_keys = _maybe_expand_oscillatory_shortlist(
        ordered_datasets,
        templates=templates,
        aggregate_fingerprint=aggregate_fingerprint,
        current_model=current_model,
        fit_engine=FitEngine(),
        initial_assessments=initial_assessments,
        template_contexts=template_contexts,
        fingerprints_by_run=fingerprints_by_run,
        current_parameter_types=current_parameter_types,
        current_values=current_values,
        parameter_bounds=parameter_bounds,
        axis_key=axis_key,
        metric=metric,
        progress_callback=progress_callback,
    )

    shortlist_keys = _shortlist_template_keys(
        tuple(templates),
        initial_assessments=initial_assessments,
        metric=metric,
        forced_keys=forced_shortlist_keys,
    )
    orchestrator = GlobalSearchOrchestrator()

    def _template_for_structure(
        source_template: CandidateTemplate,
        structure,
    ) -> CandidateTemplate:
        title = source_template.title
        if (
            tuple(structure.model.component_names),
            tuple(structure.model.operators),
            tuple(structure.model.open_parentheses),
            tuple(structure.model.close_parentheses),
        ) != (
            tuple(source_template.model.component_names),
            tuple(source_template.model.operators),
            tuple(source_template.model.open_parentheses),
            tuple(source_template.model.close_parentheses),
        ):
            title = structure.model.formula_string()
        return CandidateTemplate(
            key=source_template.key,
            title=title,
            category=source_template.category,
            rationale=source_template.rationale,
            model=structure.model,
            is_current_model_baseline=source_template.is_current_model_baseline,
        )

    def _staged_assessment_for_template(
        template: CandidateTemplate,
        base_by_run: dict[int, ParameterSet],
        fixed_param_names: tuple[str, ...],
    ) -> GlobalCandidateAssessment:
        fit_engine = FitEngine()
        baseline_assessment = initial_assessments[template.key]
        exact_caches: dict[
            tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...], tuple[bool, ...]],
            dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
        ] = {}
        baseline_by_formula: dict[
            tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...], tuple[bool, ...]],
            GlobalCandidateAssessment,
        ] = {}
        prefit_base_by_run = _single_run_prefit_parameter_sets(
            ordered_datasets,
            template,
            fit_engine=fit_engine,
            base_by_run=base_by_run,
            fixed_param_names=fixed_param_names,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
            cache=_single_run_prefit_cache_for(template),
        )
        structure = compile_legacy_structure(
            template,
            current_parameter_types=current_parameter_types,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            treat_nonfixed_roles_as_hints=True,
        )
        sample_count = int(sum(dataset.n_points for dataset in ordered_datasets))

        def _exact_cache_for(
            eval_template: CandidateTemplate,
        ) -> dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment]:
            signature = _formula_signature_for_template(eval_template)
            return exact_caches.setdefault(signature, {})

        def _baseline_for_structure(
            eval_template: CandidateTemplate,
            *,
            eval_base_by_run: dict[int, ParameterSet],
            fixed_names: tuple[str, ...],
        ) -> GlobalCandidateAssessment:
            signature = _formula_signature_for_template(eval_template)
            cached_baseline = baseline_by_formula.get(signature)
            if cached_baseline is not None:
                return cached_baseline

            template_signature = _formula_signature_for_template(template)
            if signature == template_signature:
                baseline_by_formula[signature] = baseline_assessment
                return baseline_assessment

            all_global_names = tuple(
                name for name in eval_template.model.param_names if name not in fixed_names
            )
            baseline = _fit_exact_assignment(
                ordered_datasets,
                eval_template,
                fit_engine=fit_engine,
                base_by_run=eval_base_by_run,
                global_param_names=all_global_names,
                local_param_names=(),
                fixed_param_names=fixed_names,
                axis_key=axis_key,
                metric=metric,
                cache=_exact_cache_for(eval_template),
                warm_start_by_run=None,
                progress_callback=progress_callback,
            )
            baseline_by_formula[signature] = baseline
            return baseline

        def _evaluate(candidate) -> SearchEvaluation:
            eval_template = _template_for_structure(template, candidate.structure)
            eval_base_by_run = build_parameter_sets_for_structure(
                candidate.structure,
                base_by_run=prefit_base_by_run,
                seed_by_run=candidate.initial_params_by_run,
            )
            global_names, local_names, fixed_names = compile_structure_to_legacy_roles(
                candidate.structure
            )
            exact_cache = _exact_cache_for(eval_template)
            warm_start_by_run = eval_base_by_run
            curvature_source: GlobalCandidateAssessment | None = None
            baseline_for_eval = _baseline_for_structure(
                eval_template,
                eval_base_by_run=eval_base_by_run,
                fixed_names=fixed_names,
            )
            staged_assessment: GlobalCandidateAssessment | None = None
            if len(local_names) >= 2 and baseline_for_eval.is_successful:
                staged_assessment, seed_assessment = _staged_multi_local_assignment(
                    ordered_datasets,
                    eval_template,
                    fit_engine=fit_engine,
                    base_by_run=eval_base_by_run,
                    baseline_assessment=baseline_for_eval,
                    target_local_names=local_names,
                    fixed_param_names=fixed_names,
                    axis_key=axis_key,
                    metric=metric,
                    cache=exact_cache,
                    progress_callback=progress_callback,
                    search_strategy=search_strategy,
                    instrumentation=instrumentation,
                    prefit_base_by_run=eval_base_by_run,
                    warm_start_cache=_warm_start_cache_for(eval_template),
                )
                if staged_assessment is None:
                    curvature_source = seed_assessment
                    warm_start_by_run = _warm_start_parameter_sets(
                        ordered_datasets,
                        assessment=seed_assessment,
                        base_by_run=eval_base_by_run,
                        target_global_names=global_names,
                        target_local_names=local_names,
                        fit_engine=fit_engine,
                        template=eval_template,
                        progress_callback=progress_callback,
                        cache=_warm_start_cache_for(eval_template),
                    )
                else:
                    curvature_source = staged_assessment
            elif baseline_for_eval.is_successful:
                curvature_source = baseline_for_eval
            assessment = staged_assessment
            if assessment is None:
                assessment = _fit_exact_assignment(
                    ordered_datasets,
                    eval_template,
                    fit_engine=fit_engine,
                    base_by_run=eval_base_by_run,
                    global_param_names=global_names,
                    local_param_names=local_names,
                    fixed_param_names=fixed_names,
                    axis_key=axis_key,
                    metric=metric,
                    cache=exact_cache,
                    warm_start_by_run=warm_start_by_run,
                    progress_callback=progress_callback,
                    search_strategy=search_strategy,
                    instrumentation=instrumentation,
                    initial_step_sizes=_step_hints_from_assessment(
                        ordered_datasets,
                        curvature_source,
                        target_global_names=global_names,
                        target_local_names=local_names,
                    ),
                )
            total_chi2 = float(
                sum(result.chi_squared for result in assessment.fit_results_by_run.values())
            )
            score = score_exact_candidate(
                total_chi2,
                assessment.parameter_count,
                sample_count,
                primary_metric=metric.value,
            )
            return SearchEvaluation(
                candidate=candidate,
                score=score,
                payload=assessment,
            )

        evaluation, diagnostics = orchestrator.search(
            structure=structure,
            datasets=ordered_datasets,
            base_by_run=prefit_base_by_run,
            evaluator=_evaluate,
            progress_callback=progress_callback,
            config=_staged_orchestrator_config(
                search_strategy=search_strategy,
                metric=SelectionMetric.BIC,
                instrumentation=instrumentation,
            ),
        )
        for message in diagnostics:
            _progress_log(progress_callback, f"{template.title}: {message}")
        assessment = evaluation.payload
        if not isinstance(assessment, GlobalCandidateAssessment):
            return initial_assessments[template.key]

        eval_template = _template_for_structure(template, evaluation.candidate.structure)
        eval_base_by_run = build_parameter_sets_for_structure(
            evaluation.candidate.structure,
            base_by_run=prefit_base_by_run,
            seed_by_run=evaluation.candidate.initial_params_by_run,
        )
        assessment = _prune_local_assignments(
            ordered_datasets,
            eval_template,
            fit_engine=fit_engine,
            base_by_run=eval_base_by_run,
            fixed_param_names=assessment.fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache={
                (assessment.global_param_names, assessment.local_param_names): assessment,
            },
            progress_callback=progress_callback,
            incumbent=assessment,
            warm_start_cache=_warm_start_cache_for(eval_template),
        )
        fixed_names = assessment.fixed_param_names
        recommendation_contexts[template.key] = (
            eval_template,
            eval_base_by_run,
            fixed_names,
            (
                set(evaluation.candidate.ambiguous_param_names)
                | {
                    name
                    for name in assessment.local_param_names
                    if parameter_localisation_priority(name) >= 3
                }
            ),
            _warm_start_cache_for(eval_template),
        )
        return replace(
            assessment,
            fixed_param_names=fixed_names,
            parameter_recommendations=(),
        )

    assessments_by_key: dict[str, GlobalCandidateAssessment] = {
        template.key: initial_assessments[template.key]
        for template in templates
        if template.key not in shortlist_keys
    }
    shortlisted_templates = [
        template for template in templates if template.key in shortlist_keys
    ]
    shortlisted_workers = _template_worker_count(len(shortlisted_templates))
    if shortlisted_workers <= 1:
        for template in shortlisted_templates:
            _progress_log(
                progress_callback,
                f"Running staged role search for {template.title}.",
            )
            base_by_run, fixed_param_names = template_contexts[template.key]
            assessments_by_key[template.key] = _staged_assessment_for_template(
                template,
                base_by_run,
                fixed_param_names,
            )
    else:
        _progress_log(
            progress_callback,
            "Running staged role search with "
            f"{shortlisted_workers} parallel workers.",
        )
        with ThreadPoolExecutor(
            max_workers=shortlisted_workers,
            thread_name_prefix="global-fit-staged-search",
        ) as executor:
            future_to_template = {}
            for template in shortlisted_templates:
                _progress_log(
                    progress_callback,
                    f"Running staged role search for {template.title}.",
                )
                base_by_run, fixed_param_names = template_contexts[template.key]
                future_to_template[
                    executor.submit(
                        _staged_assessment_for_template,
                        template,
                        base_by_run,
                        fixed_param_names,
                    )
                ] = template
            for future in as_completed(future_to_template):
                template = future_to_template[future]
                assessments_by_key[template.key] = future.result()

    recommendation = rerank_global_fit_wizard_recommendation(
        GlobalFitWizardRecommendation(
            series_axis_key=axis_key,
            series_axis_label=axis_label,
            mixed_axes_warning=mixed_axes_warning,
            fingerprints_by_run=fingerprints_by_run,
            dataset_order=tuple(int(dataset.run_number) for dataset in ordered_datasets),
            templates=tuple(templates),
            assessments=tuple(assessments_by_key[template.key] for template in templates),
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary="",
        ),
        metric,
    )
    for key in _parameter_recommendation_candidate_keys(recommendation):
        assessment = assessments_by_key.get(key)
        context = recommendation_contexts.get(key)
        if assessment is None or context is None:
            continue
        eval_template, eval_base_by_run, fixed_names, names_to_test, warm_start_cache = context
        _progress_log(
            progress_callback,
            f"Building per-parameter role recommendations for {eval_template.title}.",
        )
        parameter_recommendations = _build_parameter_recommendations(
            ordered_datasets,
            assessment,
            template=eval_template,
            fit_engine=FitEngine(),
            base_by_run=eval_base_by_run,
            fixed_param_names=fixed_names,
            axis_key=axis_key,
            metric=metric,
            cache={
                (assessment.global_param_names, assessment.local_param_names): assessment,
            },
            progress_callback=progress_callback,
            names_to_test=names_to_test,
            warm_start_cache=warm_start_cache,
        )
        assessments_by_key[key] = replace(
            assessment,
            fixed_param_names=fixed_names,
            parameter_recommendations=parameter_recommendations,
        )

    return rerank_global_fit_wizard_recommendation(
        replace(
            recommendation,
            assessments=tuple(assessments_by_key[template.key] for template in templates),
        ),
        metric,
    )


def rerank_global_fit_wizard_recommendation(
    recommendation: GlobalFitWizardRecommendation,
    metric: SelectionMetric,
) -> GlobalFitWizardRecommendation:
    """Reuse existing global-fit assessments and recompute the recommendation."""
    if recommendation.mixed_axes_warning:
        return replace(
            recommendation,
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary=recommendation.mixed_axes_warning,
        )

    passing = [
        assessment
        for assessment in recommendation.assessments
        if assessment.is_successful and assessment.residual_gate_passed
    ]
    if not passing:
        return replace(
            recommendation,
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary=(
                "No global candidate passed the automatic residual and continuity "
                "checks. Inspect the comparison table before applying a model."
            ),
        )

    passing_sorted = sorted(
        passing,
        key=lambda assessment: _assessment_sort_key(assessment, metric),
    )
    primary = passing_sorted[0]
    comparable_keys: tuple[str, ...] = ()

    if len(passing_sorted) > 1:
        runner_up = passing_sorted[1]
        score_delta = abs(primary.metric_value(metric) - runner_up.metric_value(metric))
        if score_delta <= _COMPARABLE_SCORE_DELTA:
            primary_complexity = (
                primary.parameter_count,
                len(primary.local_param_names),
                primary.additive_terms,
            )
            runner_up_complexity = (
                runner_up.parameter_count,
                len(runner_up.local_param_names),
                runner_up.additive_terms,
            )
            preferred = runner_up if runner_up_complexity < primary_complexity else primary
            alternate = runner_up if preferred.template.key == primary.template.key else primary
            primary = preferred
            comparable_keys = (preferred.template.key, alternate.template.key)

    compare_summary = (
        ", with a similarly scoring alternative to inspect." if comparable_keys else "."
    )
    return replace(
        recommendation,
        metric=metric,
        recommended_key=primary.template.key,
        comparable_keys=comparable_keys,
        summary=(f"Recommended: {primary.template.title} by {metric.value}{compare_summary}"),
    )


def _parameter_recommendation_candidate_keys(
    recommendation: GlobalFitWizardRecommendation,
) -> tuple[str, ...]:
    keys: list[str] = []
    for key in (recommendation.recommended_key, *recommendation.comparable_keys):
        if isinstance(key, str) and key and key not in keys:
            keys.append(key)
        if len(keys) >= 2:
            return tuple(keys)

    if keys:
        return tuple(keys)

    passing = [
        assessment
        for assessment in recommendation.assessments
        if assessment.is_successful
    ]
    passing.sort(
        key=lambda assessment: _assessment_sort_key(assessment, recommendation.metric)
    )
    for assessment in passing[:2]:
        if assessment.template.key not in keys:
            keys.append(assessment.template.key)
    return tuple(keys)


def serialize_global_fit_wizard_recommendation(
    recommendation: GlobalFitWizardRecommendation,
) -> dict[str, object]:
    """Return a JSON-serialisable snapshot of a global-fit wizard recommendation."""
    return {
        "series_axis_key": recommendation.series_axis_key,
        "series_axis_label": recommendation.series_axis_label,
        "mixed_axes_warning": recommendation.mixed_axes_warning,
        "fingerprints_by_run": {
            str(run_number): _serialize_spectrum_fingerprint(fingerprint)
            for run_number, fingerprint in recommendation.fingerprints_by_run.items()
        },
        "dataset_order": [int(run_number) for run_number in recommendation.dataset_order],
        "templates": [
            _serialize_candidate_template(template)
            for template in recommendation.templates
        ],
        "assessments": [
            _serialize_global_candidate_assessment(assessment)
            for assessment in recommendation.assessments
        ],
        "metric": recommendation.metric.value,
        "recommended_key": recommendation.recommended_key,
        "comparable_keys": list(recommendation.comparable_keys),
        "summary": recommendation.summary,
    }


def deserialize_global_fit_wizard_recommendation(
    payload: object,
) -> GlobalFitWizardRecommendation | None:
    """Rebuild a persisted global-fit wizard recommendation payload."""
    if not isinstance(payload, dict):
        return None

    templates = tuple(
        template
        for entry in payload.get("templates", [])
        if (template := _deserialize_candidate_template(entry)) is not None
    )
    assessments = tuple(
        assessment
        for entry in payload.get("assessments", [])
        if (assessment := _deserialize_global_candidate_assessment(entry)) is not None
    )
    fingerprints_by_run = {
        int(run_number): fingerprint
        for run_number, entry in (payload.get("fingerprints_by_run", {}) or {}).items()
        if (fingerprint := _deserialize_spectrum_fingerprint(entry)) is not None
    }
    dataset_order = tuple(
        int(run_number)
        for run_number in payload.get("dataset_order", [])
        if isinstance(run_number, int | float)
    )
    comparable_keys = tuple(
        key for key in payload.get("comparable_keys", []) if isinstance(key, str)
    )

    return GlobalFitWizardRecommendation(
        series_axis_key=str(payload.get("series_axis_key", "run")),
        series_axis_label=str(payload.get("series_axis_label", "Run")),
        mixed_axes_warning=(
            str(payload["mixed_axes_warning"])
            if payload.get("mixed_axes_warning") is not None
            else None
        ),
        fingerprints_by_run=fingerprints_by_run,
        dataset_order=dataset_order,
        templates=templates,
        assessments=assessments,
        metric=SelectionMetric.from_value(payload.get("metric", SelectionMetric.AICC.value)),
        recommended_key=(
            str(payload["recommended_key"])
            if payload.get("recommended_key") is not None
            else None
        ),
        comparable_keys=comparable_keys,
        summary=str(payload.get("summary", "")),
    )


def _fixed_param_names(
    template: CandidateTemplate,
    current_parameter_types: dict[str, str],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in template.model.param_names
        if str(current_parameter_types.get(name, "")).strip().lower() == "fixed"
    )


def _serialize_candidate_template(template: CandidateTemplate) -> dict[str, object]:
    return {
        "key": template.key,
        "title": template.title,
        "category": template.category,
        "rationale": template.rationale,
        "model": template.model.to_dict(),
        "is_current_model_baseline": bool(template.is_current_model_baseline),
    }


def _deserialize_candidate_template(payload: object) -> CandidateTemplate | None:
    if not isinstance(payload, dict):
        return None
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        return None
    try:
        model = CompositeModel.from_dict(model_payload)
    except ValueError:
        return None
    return CandidateTemplate(
        key=str(payload.get("key", "")),
        title=str(payload.get("title", "")),
        category=str(payload.get("category", "")),
        rationale=str(payload.get("rationale", "")),
        model=model,
        is_current_model_baseline=bool(payload.get("is_current_model_baseline", False)),
    )


def _serialize_spectrum_fingerprint(fingerprint: SpectrumFingerprint) -> dict[str, object]:
    return {
        "tail_estimate": fingerprint.tail_estimate,
        "initial_amplitude_estimate": fingerprint.initial_amplitude_estimate,
        "zero_crossings": fingerprint.zero_crossings,
        "smoothed_zero_crossings": fingerprint.smoothed_zero_crossings,
        "smoothed_turning_points": fingerprint.smoothed_turning_points,
        "dominant_fft_frequency_mhz": fingerprint.dominant_fft_frequency_mhz,
        "dominant_fft_snr": fingerprint.dominant_fft_snr,
        "dominant_fft_cycles_in_window": fingerprint.dominant_fft_cycles_in_window,
        "monotonic_decay_fraction": fingerprint.monotonic_decay_fraction,
        "early_time_curvature": fingerprint.early_time_curvature,
        "semilog_slope_ratio": fingerprint.semilog_slope_ratio,
        "late_time_dip_recovery_score": fingerprint.late_time_dip_recovery_score,
        "oscillatory_hint": fingerprint.oscillatory_hint,
        "kt_like_hint": fingerprint.kt_like_hint,
        "multi_rate_hint": fingerprint.multi_rate_hint,
    }


def _deserialize_spectrum_fingerprint(payload: object) -> SpectrumFingerprint | None:
    if not isinstance(payload, dict):
        return None
    try:
        return SpectrumFingerprint(
            tail_estimate=float(payload.get("tail_estimate", 0.0)),
            initial_amplitude_estimate=float(payload.get("initial_amplitude_estimate", 0.0)),
            zero_crossings=int(payload.get("zero_crossings", 0)),
            smoothed_zero_crossings=int(payload.get("smoothed_zero_crossings", 0)),
            smoothed_turning_points=int(payload.get("smoothed_turning_points", 0)),
            dominant_fft_frequency_mhz=float(payload.get("dominant_fft_frequency_mhz", 0.0)),
            dominant_fft_snr=float(payload.get("dominant_fft_snr", 0.0)),
            dominant_fft_cycles_in_window=float(payload.get("dominant_fft_cycles_in_window", 0.0)),
            monotonic_decay_fraction=float(payload.get("monotonic_decay_fraction", 0.0)),
            early_time_curvature=float(payload.get("early_time_curvature", 0.0)),
            semilog_slope_ratio=float(payload.get("semilog_slope_ratio", 0.0)),
            late_time_dip_recovery_score=float(payload.get("late_time_dip_recovery_score", 0.0)),
            oscillatory_hint=bool(payload.get("oscillatory_hint", False)),
            kt_like_hint=bool(payload.get("kt_like_hint", False)),
            multi_rate_hint=bool(payload.get("multi_rate_hint", False)),
        )
    except (TypeError, ValueError):
        return None


def _serialize_parameter_set(parameters: ParameterSet) -> list[dict[str, object]]:
    return [
        {
            "name": parameter.name,
            "value": float(parameter.value),
            "min": float(parameter.min),
            "max": float(parameter.max),
            "fixed": bool(parameter.fixed),
            "expr": parameter.expr,
        }
        for parameter in parameters
    ]


def _deserialize_parameter_set(payload: object) -> ParameterSet:
    parameters = ParameterSet()
    if not isinstance(payload, list):
        return parameters
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        try:
            parameters.add(
                Parameter(
                    name=name,
                    value=float(entry.get("value", 0.0)),
                    min=float(entry.get("min", -float("inf"))),
                    max=float(entry.get("max", float("inf"))),
                    fixed=bool(entry.get("fixed", False)),
                    expr=str(entry["expr"]) if entry.get("expr") is not None else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return parameters


def _serialize_fit_result(result: FitResult) -> dict[str, object]:
    return {
        "success": bool(result.success),
        "chi_squared": float(result.chi_squared),
        "reduced_chi_squared": float(result.reduced_chi_squared),
        "parameters": _serialize_parameter_set(result.parameters),
        "uncertainties": {name: float(value) for name, value in result.uncertainties.items()},
        "message": result.message,
    }


def _deserialize_fit_result(payload: object) -> FitResult | None:
    if not isinstance(payload, dict):
        return None
    try:
        uncertainties = {
            str(name): float(value)
            for name, value in (payload.get("uncertainties", {}) or {}).items()
        }
    except (TypeError, ValueError):
        uncertainties = {}
    return FitResult(
        success=bool(payload.get("success", False)),
        chi_squared=float(payload.get("chi_squared", 0.0)),
        reduced_chi_squared=float(payload.get("reduced_chi_squared", 0.0)),
        parameters=_deserialize_parameter_set(payload.get("parameters", [])),
        uncertainties=uncertainties,
        message=str(payload.get("message", "")),
    )


def _serialize_run_residual_diagnostic(diagnostic: RunResidualDiagnostic) -> dict[str, object]:
    return {
        "run_number": diagnostic.run_number,
        "run_label": diagnostic.run_label,
        "axis_value": diagnostic.axis_value,
        "residual_rms": diagnostic.residual_rms,
        "runs_z_score": diagnostic.runs_z_score,
        "max_abs_autocorrelation": diagnostic.max_abs_autocorrelation,
        "residual_fft_peak_snr": diagnostic.residual_fft_peak_snr,
        "gate_passed": diagnostic.gate_passed,
        "gate_reasons": list(diagnostic.gate_reasons),
    }


def _deserialize_run_residual_diagnostic(payload: object) -> RunResidualDiagnostic | None:
    if not isinstance(payload, dict):
        return None
    try:
        return RunResidualDiagnostic(
            run_number=int(payload.get("run_number", 0)),
            run_label=str(payload.get("run_label", "")),
            axis_value=float(payload.get("axis_value", 0.0)),
            residual_rms=float(payload.get("residual_rms", 0.0)),
            runs_z_score=float(payload.get("runs_z_score", 0.0)),
            max_abs_autocorrelation=float(payload.get("max_abs_autocorrelation", 0.0)),
            residual_fft_peak_snr=float(payload.get("residual_fft_peak_snr", 0.0)),
            gate_passed=bool(payload.get("gate_passed", False)),
            gate_reasons=tuple(
                reason for reason in payload.get("gate_reasons", []) if isinstance(reason, str)
            ),
        )
    except (TypeError, ValueError):
        return None


def _serialize_global_parameter_recommendation(
    recommendation: GlobalParameterRecommendation,
) -> dict[str, object]:
    return {
        "name": recommendation.name,
        "recommended_role": recommendation.recommended_role,
        "global_score": recommendation.global_score,
        "local_score": recommendation.local_score,
        "score_delta": recommendation.score_delta,
        "total_variation": recommendation.total_variation,
        "roughness": recommendation.roughness,
        "rationale": recommendation.rationale,
    }


def _deserialize_global_parameter_recommendation(
    payload: object,
) -> GlobalParameterRecommendation | None:
    if not isinstance(payload, dict):
        return None
    try:
        return GlobalParameterRecommendation(
            name=str(payload.get("name", "")),
            recommended_role=str(payload.get("recommended_role", "Global")),
            global_score=float(payload.get("global_score", float("inf"))),
            local_score=float(payload.get("local_score", float("inf"))),
            score_delta=float(payload.get("score_delta", float("inf"))),
            total_variation=float(payload.get("total_variation", 0.0)),
            roughness=float(payload.get("roughness", 0.0)),
            rationale=str(payload.get("rationale", "")),
        )
    except (TypeError, ValueError):
        return None


def _serialize_curve_pair(
    curve: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> dict[str, object]:
    time_axis, values = curve
    return {
        "time": np.asarray(time_axis, dtype=float).tolist(),
        "values": np.asarray(values, dtype=float).tolist(),
    }


def _deserialize_curve_pair(
    payload: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if not isinstance(payload, dict):
        return None
    try:
        return (
            np.asarray(payload.get("time", []), dtype=float),
            np.asarray(payload.get("values", []), dtype=float),
        )
    except (TypeError, ValueError):
        return None


def _serialize_component_curves(
    curves: tuple[tuple[str, NDArray[np.float64]], ...],
) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "values": np.asarray(values, dtype=float).tolist(),
        }
        for name, values in curves
    ]


def _deserialize_component_curves(
    payload: object,
) -> tuple[tuple[str, NDArray[np.float64]], ...]:
    if not isinstance(payload, list):
        return ()
    curves: list[tuple[str, NDArray[np.float64]]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        try:
            curves.append((name, np.asarray(entry.get("values", []), dtype=float)))
        except (TypeError, ValueError):
            continue
    return tuple(curves)


def _serialize_global_candidate_assessment(
    assessment: GlobalCandidateAssessment,
) -> dict[str, object]:
    return {
        "template": _serialize_candidate_template(assessment.template),
        "fit_results_by_run": {
            str(run_number): _serialize_fit_result(result)
            for run_number, result in assessment.fit_results_by_run.items()
        },
        "global_parameters": _serialize_parameter_set(assessment.global_parameters),
        "global_param_names": list(assessment.global_param_names),
        "local_param_names": list(assessment.local_param_names),
        "fixed_param_names": list(assessment.fixed_param_names),
        "parameter_recommendations": [
            _serialize_global_parameter_recommendation(recommendation)
            for recommendation in assessment.parameter_recommendations
        ],
        "run_diagnostics": [
            _serialize_run_residual_diagnostic(diagnostic)
            for diagnostic in assessment.run_diagnostics
        ],
        "series_warnings": list(assessment.series_warnings),
        "aic": assessment.aic,
        "aicc": assessment.aicc,
        "bic": assessment.bic,
        "selected_score": assessment.selected_score,
        "fitted_curves_by_run": {
            str(run_number): _serialize_curve_pair(curve)
            for run_number, curve in assessment.fitted_curves_by_run.items()
        },
        "component_curves_by_run": {
            str(run_number): _serialize_component_curves(curves)
            for run_number, curves in assessment.component_curves_by_run.items()
        },
    }


def _deserialize_global_candidate_assessment(
    payload: object,
) -> GlobalCandidateAssessment | None:
    if not isinstance(payload, dict):
        return None
    template = _deserialize_candidate_template(payload.get("template"))
    if template is None:
        return None

    fit_results_by_run = {
        int(run_number): result
        for run_number, entry in (payload.get("fit_results_by_run", {}) or {}).items()
        if (result := _deserialize_fit_result(entry)) is not None
    }
    run_diagnostics = tuple(
        diagnostic
        for entry in payload.get("run_diagnostics", [])
        if (diagnostic := _deserialize_run_residual_diagnostic(entry)) is not None
    )
    parameter_recommendations = tuple(
        recommendation
        for entry in payload.get("parameter_recommendations", [])
        if (recommendation := _deserialize_global_parameter_recommendation(entry)) is not None
    )
    fitted_curves_by_run = {
        int(run_number): curve
        for run_number, entry in (payload.get("fitted_curves_by_run", {}) or {}).items()
        if (curve := _deserialize_curve_pair(entry)) is not None
    }
    component_curves_by_run = {
        int(run_number): _deserialize_component_curves(entry)
        for run_number, entry in (payload.get("component_curves_by_run", {}) or {}).items()
    }

    try:
        return GlobalCandidateAssessment(
            template=template,
            fit_results_by_run=fit_results_by_run,
            global_parameters=_deserialize_parameter_set(payload.get("global_parameters", [])),
            global_param_names=tuple(
                name for name in payload.get("global_param_names", []) if isinstance(name, str)
            ),
            local_param_names=tuple(
                name for name in payload.get("local_param_names", []) if isinstance(name, str)
            ),
            fixed_param_names=tuple(
                name for name in payload.get("fixed_param_names", []) if isinstance(name, str)
            ),
            parameter_recommendations=parameter_recommendations,
            run_diagnostics=run_diagnostics,
            series_warnings=tuple(
                warning
                for warning in payload.get("series_warnings", [])
                if isinstance(warning, str)
            ),
            aic=float(payload.get("aic", float("inf"))),
            aicc=(
                float(payload["aicc"])
                if payload.get("aicc") is not None
                else None
            ),
            bic=float(payload.get("bic", float("inf"))),
            selected_score=float(payload.get("selected_score", float("inf"))),
            fitted_curves_by_run=fitted_curves_by_run,
            component_curves_by_run=component_curves_by_run,
        )
    except (TypeError, ValueError):
        return None


def _initial_parameter_roles(
    template: CandidateTemplate,
    *,
    current_parameter_types: dict[str, str],
    fixed_param_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if template.is_current_model_baseline:
        local_param_names = tuple(
            name
            for name in template.model.param_names
            if name not in fixed_param_names
            and str(current_parameter_types.get(name, "")).strip().lower() == "local"
        )
    else:
        local_param_names = ()
    global_param_names = tuple(
        name
        for name in template.model.param_names
        if name not in fixed_param_names and name not in local_param_names
    )
    return global_param_names, local_param_names


def _shortlist_template_keys(
    templates: tuple[CandidateTemplate, ...],
    *,
    initial_assessments: dict[str, GlobalCandidateAssessment],
    metric: SelectionMetric,
    forced_keys: tuple[str, ...] = (),
) -> set[str]:
    ranked = sorted(
        templates,
        key=lambda template: _assessment_sort_key(
            initial_assessments[template.key],
            metric,
        ),
    )
    if not ranked:
        return set()

    shortlist: list[str] = [template.key for template in ranked[:_SHORTLIST_COUNT]]

    for key in _template_anchor_keys(templates):
        if key not in shortlist and len(shortlist) < _SHORTLIST_CAP:
            shortlist.append(key)

    for template in templates:
        if (
            template.is_current_model_baseline
            and template.key not in shortlist
            and len(shortlist) < _SHORTLIST_CAP
        ):
            shortlist.append(template.key)

    cutoff_index = min(_SHORTLIST_COUNT, len(ranked)) - 1
    cutoff_score = initial_assessments[ranked[cutoff_index].key].metric_value(metric)
    for template in ranked[_SHORTLIST_COUNT:]:
        if len(shortlist) >= _SHORTLIST_CAP:
            break
        score = initial_assessments[template.key].metric_value(metric)
        if score - cutoff_score <= _SHORTLIST_SCORE_WINDOW:
            shortlist.append(template.key)

    for key in forced_keys:
        if key not in shortlist:
            shortlist.append(key)

    return set(shortlist)


def _maybe_expand_oscillatory_shortlist(
    datasets: list[MuonDataset],
    *,
    templates: list[CandidateTemplate],
    aggregate_fingerprint: SpectrumFingerprint,
    current_model: CompositeModel | None,
    fit_engine: FitEngine,
    initial_assessments: dict[str, GlobalCandidateAssessment],
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
    fingerprints_by_run: dict[int, SpectrumFingerprint],
    current_parameter_types: dict[str, str],
    current_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    axis_key: str,
    metric: SelectionMetric,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    scout, supporting_runs = _oscillatory_rescue_scout(
        datasets,
        assessments=tuple(initial_assessments.values()),
        fingerprints_by_run=fingerprints_by_run,
        metric=metric,
    )
    if scout is None or not supporting_runs:
        return ()

    oscillatory_keys = tuple(
        template.key for template in templates if template.category == "Oscillatory"
    )
    if not oscillatory_keys:
        rescue_templates = _oscillatory_rescue_templates(
            aggregate_fingerprint,
            current_model=current_model,
            existing_templates=tuple(templates),
        )
        if rescue_templates:
            templates.extend(rescue_templates)
            cluster_runs = _longest_ordered_run_cluster(datasets, supporting_runs)
            cluster_label = _run_label_span(datasets, cluster_runs or supporting_runs)
            _progress_log(
                progress_callback,
                "Residual-guided oscillatory rescue triggered from "
                f"{scout.template.title} across {cluster_label}. "
                "Adding oscillatory families to the shortlist for a conservative check.",
            )
        for template in rescue_templates:
            fixed_param_names = _fixed_param_names(template, current_parameter_types)
            base_by_run = _initial_parameter_sets_for_candidate(
                datasets,
                fingerprints_by_run,
                template,
                current_values=current_values,
                parameter_bounds=parameter_bounds,
                fixed_param_names=fixed_param_names,
            )
            template_contexts[template.key] = (base_by_run, fixed_param_names)
            initial_global_names, initial_local_names = _initial_parameter_roles(
                template,
                current_parameter_types=current_parameter_types,
                fixed_param_names=fixed_param_names,
            )
            assignment_cache: dict[
                tuple[tuple[str, ...], tuple[str, ...]],
                GlobalCandidateAssessment,
            ] = {}
            _progress_log(
                progress_callback,
                f"Residual-guided rescue screening: {template.title}.",
            )
            initial_assessments[template.key] = _fit_exact_assignment(
                datasets,
                template,
                fit_engine=fit_engine,
                base_by_run=base_by_run,
                global_param_names=initial_global_names,
                local_param_names=initial_local_names,
                fixed_param_names=fixed_param_names,
                axis_key=axis_key,
                metric=metric,
                cache=assignment_cache,
                progress_callback=progress_callback,
            )
        oscillatory_keys = tuple(
            template.key for template in templates if template.category == "Oscillatory"
        )
    else:
        cluster_runs = _longest_ordered_run_cluster(datasets, supporting_runs)
        cluster_label = _run_label_span(datasets, cluster_runs or supporting_runs)
        _progress_log(
            progress_callback,
            "Residual-guided oscillatory rescue promoting existing oscillatory "
            f"families after structured FFT residuals across {cluster_label}.",
        )

    return oscillatory_keys


def _oscillatory_rescue_templates(
    aggregate_fingerprint: SpectrumFingerprint,
    *,
    current_model: CompositeModel | None,
    existing_templates: tuple[CandidateTemplate, ...],
) -> tuple[CandidateTemplate, ...]:
    rescue_fingerprint = replace(aggregate_fingerprint, oscillatory_hint=True)
    rescue_candidates = build_candidate_templates(
        rescue_fingerprint,
        current_model=current_model,
    )
    existing_keys = {template.key for template in existing_templates}
    return tuple(
        template
        for template in rescue_candidates
        if template.category == "Oscillatory" and template.key not in existing_keys
    )


def _oscillatory_rescue_scout(
    datasets: list[MuonDataset],
    *,
    assessments: tuple[GlobalCandidateAssessment, ...],
    fingerprints_by_run: dict[int, SpectrumFingerprint],
    metric: SelectionMetric,
) -> tuple[GlobalCandidateAssessment | None, tuple[int, ...]]:
    non_oscillatory = sorted(
        (
            assessment
            for assessment in assessments
            if assessment.is_successful and assessment.template.category != "Oscillatory"
        ),
        key=lambda assessment: _assessment_sort_key(assessment, metric),
    )
    for assessment in non_oscillatory[:_OSCILLATORY_RESCUE_MAX_SCOUTS]:
        supporting_runs = _supported_oscillatory_run_numbers(
            datasets,
            assessment=assessment,
            fingerprints_by_run=fingerprints_by_run,
        )
        if supporting_runs:
            return assessment, supporting_runs
    return None, ()


def _supported_oscillatory_run_numbers(
    datasets: list[MuonDataset],
    *,
    assessment: GlobalCandidateAssessment,
    fingerprints_by_run: dict[int, SpectrumFingerprint],
) -> tuple[int, ...]:
    diagnostics_by_run = {
        int(diagnostic.run_number): diagnostic for diagnostic in assessment.run_diagnostics
    }
    supported_runs: list[int] = []
    fingerprint_supported_runs: list[int] = []
    residual_fft_snrs: list[float] = []

    for dataset in datasets:
        run_number = int(dataset.run_number)
        diagnostic = diagnostics_by_run.get(run_number)
        fingerprint = fingerprints_by_run.get(run_number)
        if diagnostic is None or fingerprint is None:
            continue
        if diagnostic.residual_fft_peak_snr < _OSCILLATORY_RESCUE_RESIDUAL_FFT_SNR:
            continue

        fingerprint_supported = (
            fingerprint.dominant_fft_snr >= _OSCILLATORY_RESCUE_FINGERPRINT_FFT_SNR
            and (
                fingerprint.dominant_fft_cycles_in_window
                >= _OSCILLATORY_RESCUE_FINGERPRINT_MIN_CYCLES
            )
            and fingerprint.smoothed_turning_points >= _OSCILLATORY_RESCUE_FINGERPRINT_MIN_TURNS
        )
        structured_residual = (
            abs(diagnostic.runs_z_score) >= _OSCILLATORY_RESCUE_RUNS_Z
            or fingerprint.oscillatory_hint
            or fingerprint_supported
        )
        if not structured_residual:
            continue

        supported_runs.append(run_number)
        residual_fft_snrs.append(diagnostic.residual_fft_peak_snr)
        if fingerprint_supported:
            fingerprint_supported_runs.append(run_number)

    if not supported_runs:
        return ()

    minimum_runs = max(
        _OSCILLATORY_RESCUE_MIN_RUNS,
        int(np.ceil(len(datasets) * _OSCILLATORY_RESCUE_MIN_FRACTION)),
    )
    minimum_cluster = max(
        _OSCILLATORY_RESCUE_MIN_CLUSTER,
        int(np.ceil(minimum_runs * 0.6)),
    )
    minimum_fingerprint_runs = max(2, minimum_runs // 2)
    longest_cluster = _longest_ordered_run_cluster(datasets, tuple(supported_runs))

    if len(supported_runs) < minimum_runs:
        return ()
    if len(fingerprint_supported_runs) < minimum_fingerprint_runs:
        return ()
    if len(longest_cluster) < minimum_cluster:
        return ()
    if float(np.median(residual_fft_snrs)) < _OSCILLATORY_RESCUE_MEDIAN_FFT_SNR:
        return ()
    return tuple(supported_runs)


def _longest_ordered_run_cluster(
    datasets: list[MuonDataset],
    run_numbers: tuple[int, ...],
) -> tuple[int, ...]:
    if not run_numbers:
        return ()

    run_number_set = set(run_numbers)
    best_cluster: list[int] = []
    current_cluster: list[int] = []
    for dataset in datasets:
        run_number = int(dataset.run_number)
        if run_number in run_number_set:
            current_cluster.append(run_number)
            continue
        if len(current_cluster) > len(best_cluster):
            best_cluster = current_cluster.copy()
        current_cluster.clear()
    if len(current_cluster) > len(best_cluster):
        best_cluster = current_cluster.copy()
    return tuple(best_cluster)


def _run_label_span(
    datasets: list[MuonDataset],
    run_numbers: tuple[int, ...],
) -> str:
    if not run_numbers:
        return "the series"
    label_by_run = {int(dataset.run_number): dataset.run_label for dataset in datasets}
    start_label = label_by_run.get(run_numbers[0], str(run_numbers[0]))
    end_label = label_by_run.get(run_numbers[-1], str(run_numbers[-1]))
    if start_label == end_label:
        return start_label
    return f"{start_label}-{end_label}"


def _template_anchor_keys(templates: tuple[CandidateTemplate, ...]) -> tuple[str, ...]:
    anchors: list[str] = []
    by_category: dict[str, CandidateTemplate] = {}
    for template in templates:
        incumbent = by_category.get(template.category)
        if incumbent is None:
            by_category[template.category] = template
            continue
        if template.parameter_count < incumbent.parameter_count or (
            template.parameter_count == incumbent.parameter_count
            and template.title < incumbent.title
        ):
            by_category[template.category] = template
    for category in ("General", "Oscillatory", "KT-like", "Baseline"):
        template = by_category.get(category)
        if template is not None:
            anchors.append(template.key)
    return tuple(anchors)


def _search_parameter_roles(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    initial_global_names: tuple[str, ...],
    initial_local_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    progress_callback: Callable[[str], None] | None = None,
) -> GlobalCandidateAssessment:
    best = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        global_param_names=initial_global_names,
        local_param_names=initial_local_names,
        fixed_param_names=fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        cache=cache,
        warm_start_by_run=None,
        progress_callback=progress_callback,
    )

    if not best.is_successful:
        return best

    current_local = list(initial_local_names)
    remaining = [
        name
        for name in template.model.param_names
        if name not in fixed_param_names and name not in current_local
    ]

    while remaining:
        preferred = _best_forward_role_change(
            datasets,
            template,
            fit_engine=fit_engine,
            base_by_run=base_by_run,
            fixed_param_names=fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache=cache,
            progress_callback=progress_callback,
            incumbent=best,
            remaining=tuple(remaining),
        )
        if preferred is None:
            break
        if _prefer_role_change(preferred, best, metric=metric):
            localized = ", ".join(preferred.local_param_names) or "none"
            _progress_log(
                progress_callback,
                f"{template.title}: accepted Local set [{localized}].",
            )
            best = preferred
            current_local = list(preferred.local_param_names)
            remaining = [
                name
                for name in template.model.param_names
                if name not in fixed_param_names and name not in current_local
            ]
            continue
        break

    return _prune_local_assignments(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        fixed_param_names=fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        cache=cache,
        progress_callback=progress_callback,
        incumbent=best,
    )


def _prune_local_assignments(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    progress_callback: Callable[[str], None] | None,
    incumbent: GlobalCandidateAssessment,
    warm_start_cache: dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ] | None = None,
) -> GlobalCandidateAssessment:
    best = incumbent
    for name in list(best.local_param_names):
        candidate_local_names = tuple(pname for pname in best.local_param_names if pname != name)
        candidate_global_names = tuple(
            pname
            for pname in template.model.param_names
            if pname not in fixed_param_names and pname not in candidate_local_names
        )
        candidate = _fit_exact_assignment(
            datasets,
            template,
            fit_engine=fit_engine,
            base_by_run=base_by_run,
            global_param_names=candidate_global_names,
            local_param_names=candidate_local_names,
            fixed_param_names=fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache=cache,
            warm_start_by_run=_warm_start_parameter_sets(
                datasets,
                assessment=best,
                base_by_run=base_by_run,
                target_global_names=candidate_global_names,
                target_local_names=candidate_local_names,
                fit_engine=fit_engine,
                template=template,
                progress_callback=progress_callback,
                cache=warm_start_cache,
            ),
            progress_callback=progress_callback,
        )
        if _prefer_simpler_assignment(candidate, best, metric=metric):
            localized = ", ".join(candidate.local_param_names) or "none"
            _progress_log(
                progress_callback,
                f"{template.title}: pruned Local set back to [{localized}].",
            )
            best = candidate

    return best


def _best_forward_role_change(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    progress_callback: Callable[[str], None] | None,
    incumbent: GlobalCandidateAssessment,
    remaining: tuple[str, ...],
    use_screening: bool = False,
    exact_candidates_per_tier: int | None = None,
    search_strategy: str = "legacy",
    instrumentation: dict[str, object] | None = None,
    warm_start_cache: dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ] | None = None,
) -> GlobalCandidateAssessment | None:
    candidates = _forward_role_change_candidates(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        fixed_param_names=fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        cache=cache,
        progress_callback=progress_callback,
        incumbent=incumbent,
        remaining=remaining,
        use_screening=use_screening,
        exact_candidates_per_tier=exact_candidates_per_tier,
        search_strategy=search_strategy,
        instrumentation=instrumentation,
        warm_start_cache=warm_start_cache,
    )
    if candidates:
        return candidates[0]
    return None


def _forward_role_change_candidates(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    progress_callback: Callable[[str], None] | None,
    incumbent: GlobalCandidateAssessment,
    remaining: tuple[str, ...],
    use_screening: bool = False,
    exact_candidates_per_tier: int | None = None,
    search_strategy: str = "legacy",
    instrumentation: dict[str, object] | None = None,
    warm_start_cache: dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ] | None = None,
) -> tuple[GlobalCandidateAssessment, ...]:
    for tier in _tiered_role_candidates(remaining, incumbent.local_param_names):
        candidate_specs: list[
            tuple[
                float,
                str,
                tuple[str, ...],
                tuple[str, ...],
                dict[int, ParameterSet],
            ]
        ] = []
        for name in tier:
            candidate_local_names = tuple(sorted((*incumbent.local_param_names, name)))
            candidate_global_names = tuple(
                pname
                for pname in template.model.param_names
                if pname not in fixed_param_names and pname not in candidate_local_names
            )
            warm_start_by_run = _warm_start_parameter_sets(
                datasets,
                base_by_run=base_by_run,
                assessment=incumbent,
                target_global_names=candidate_global_names,
                target_local_names=candidate_local_names,
                fit_engine=fit_engine,
                template=template,
                progress_callback=progress_callback,
                cache=warm_start_cache,
            )
            probe_score = 0.0
            if use_screening:
                probe_score = _probe_assignment_candidate(
                    datasets,
                    template,
                    fit_engine=fit_engine,
                    base_by_run=base_by_run,
                    warm_start_by_run=warm_start_by_run,
                    global_param_names=candidate_global_names,
                    local_param_names=candidate_local_names,
                    active_names=set(candidate_local_names),
                    instrumentation=instrumentation,
                )
            candidate_specs.append(
                (
                    probe_score,
                    name,
                    candidate_global_names,
                    candidate_local_names,
                    warm_start_by_run,
                )
            )

        if use_screening:
            candidate_specs.sort(
                key=lambda item: (
                    item[0],
                    parameter_localisation_priority(item[1]),
                    item[1],
                )
            )
        exact_limit = exact_candidates_per_tier or len(candidate_specs)
        if use_screening and len(candidate_specs) > exact_limit:
            _record_counter(
                instrumentation,
                "staged_probe_rejections",
                len(candidate_specs) - exact_limit,
            )

        candidates: list[GlobalCandidateAssessment] = []
        for _probe_score, _name, candidate_global_names, candidate_local_names, warm_start_by_run in candidate_specs[:exact_limit]:
            candidate = _fit_exact_assignment(
                datasets,
                template,
                fit_engine=fit_engine,
                base_by_run=base_by_run,
                global_param_names=candidate_global_names,
                local_param_names=candidate_local_names,
                fixed_param_names=fixed_param_names,
                axis_key=axis_key,
                metric=metric,
                cache=cache,
                warm_start_by_run=warm_start_by_run,
                progress_callback=progress_callback,
                search_strategy=search_strategy,
                instrumentation=instrumentation,
                initial_step_sizes=_step_hints_from_assessment(
                    datasets,
                    incumbent,
                    target_global_names=candidate_global_names,
                    target_local_names=candidate_local_names,
                ),
            )
            if candidate.is_successful:
                candidates.append(candidate)

        if candidates:
            ordered = sorted(
                candidates,
                key=lambda candidate: _assessment_sort_key(candidate, metric),
            )
            return tuple(ordered)
    return ()


def _probe_assignment_candidate(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    warm_start_by_run: dict[int, ParameterSet],
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    active_names: set[str],
    instrumentation: dict[str, object] | None = None,
) -> float:
    probe_params = _parameter_sets_for_stage(warm_start_by_run, active_names=active_names)
    probe_budget = min(
        700,
        _global_fit_call_budget(
            datasets,
            probe_params,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            phase="stage",
        ),
    )
    _record_counter(instrumentation, "staged_probe_calls")
    results_by_run, _ = fit_engine.global_fit(
        datasets,
        template.model.function,
        list(global_param_names),
        list(local_param_names),
        probe_params,
        max_calls=probe_budget,
        migrad_iterations=3,
        use_simplex_rescue=False,
    )
    if all(result.success for result in results_by_run.values()):
        _record_counter(instrumentation, "staged_probe_successes")
        return float(sum(result.chi_squared for result in results_by_run.values()))
    return float("inf")


def _staged_multi_local_assignment(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    baseline_assessment: GlobalCandidateAssessment,
    target_local_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    progress_callback: Callable[[str], None] | None = None,
    search_strategy: str = "staged_v1",
    instrumentation: dict[str, object] | None = None,
    prefit_base_by_run: dict[int, ParameterSet] | None = None,
    warm_start_cache: dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ] | None = None,
) -> tuple[GlobalCandidateAssessment | None, GlobalCandidateAssessment]:
    if len(target_local_names) < 2 or not baseline_assessment.is_successful:
        return None, baseline_assessment

    search_base_by_run = (
        _clone_parameter_sets(prefit_base_by_run)
        if prefit_base_by_run is not None
        else _single_run_prefit_parameter_sets(
            datasets,
            template,
            fit_engine=fit_engine,
            base_by_run=base_by_run,
            fixed_param_names=fixed_param_names,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )
    )
    ordered_target_local_names = tuple(sorted(target_local_names))
    _progress_log(
        progress_callback,
        f"{template.title}: building warm start from all-global baseline "
        "for multi-local assignment.",
    )
    beam_width, branch_limit, use_screening, exact_candidates_per_tier = _staged_local_search_settings(
        search_strategy
    )
    frontier: list[GlobalCandidateAssessment] = [baseline_assessment]
    best_partial = baseline_assessment
    completion_seeds: list[GlobalCandidateAssessment] = [baseline_assessment]

    while frontier:
        _append_metric(instrumentation, "staged_frontier_widths", len(frontier))
        completed = [
            candidate
            for candidate in frontier
            if candidate.local_param_names == ordered_target_local_names
        ]
        if completed:
            best_completed = min(
                completed,
                key=lambda candidate: _assessment_sort_key(candidate, metric),
            )
            return best_completed, best_completed

        next_frontier: list[GlobalCandidateAssessment] = []
        seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for incumbent in frontier:
            remaining = tuple(
                name for name in ordered_target_local_names if name not in incumbent.local_param_names
            )
            if not remaining:
                continue
            candidates = _forward_role_change_candidates(
                datasets,
                template,
                fit_engine=fit_engine,
                base_by_run=search_base_by_run,
                fixed_param_names=fixed_param_names,
                axis_key=axis_key,
                metric=metric,
                cache=cache,
                progress_callback=progress_callback,
                incumbent=incumbent,
                remaining=remaining,
                use_screening=use_screening,
                exact_candidates_per_tier=exact_candidates_per_tier,
                search_strategy=search_strategy,
                instrumentation=instrumentation,
                warm_start_cache=warm_start_cache,
            )
            for candidate in candidates[:branch_limit]:
                signature = (candidate.global_param_names, candidate.local_param_names)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                next_frontier.append(candidate)

        if not next_frontier:
            break

        next_frontier.sort(key=lambda candidate: _assessment_sort_key(candidate, metric))
        frontier = next_frontier[:beam_width]
        best_partial = frontier[0]
        completion_seeds.extend(frontier)

    completion_candidates: list[GlobalCandidateAssessment] = []
    seen_completion_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    completion_global_names = tuple(
        name
        for name in template.model.param_names
        if name not in fixed_param_names and name not in ordered_target_local_names
    )
    direct_completion = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=search_base_by_run,
        global_param_names=completion_global_names,
        local_param_names=ordered_target_local_names,
        fixed_param_names=fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        cache=cache,
        warm_start_by_run=search_base_by_run,
        progress_callback=progress_callback,
        search_strategy=search_strategy,
        instrumentation=instrumentation,
        initial_step_sizes=_step_hints_from_assessment(
            datasets,
            best_partial,
            target_global_names=completion_global_names,
            target_local_names=ordered_target_local_names,
        ),
    )
    _record_counter(instrumentation, "staged_completion_attempts")
    if direct_completion.is_successful:
        completion_candidates.append(direct_completion)
        _record_counter(instrumentation, "staged_completion_successes")

    for seed_assessment in sorted(
        completion_seeds,
        key=lambda candidate: _assessment_sort_key(candidate, metric),
    ):
        seed_signature = (seed_assessment.global_param_names, seed_assessment.local_param_names)
        if seed_signature in seen_completion_signatures:
            continue
        seen_completion_signatures.add(seed_signature)
        warm_start_by_run = _warm_start_parameter_sets(
            datasets,
            assessment=seed_assessment,
            base_by_run=search_base_by_run,
            target_global_names=completion_global_names,
            target_local_names=ordered_target_local_names,
            fit_engine=fit_engine,
            template=template,
            progress_callback=progress_callback,
            cache=warm_start_cache,
        )
        candidate = _fit_exact_assignment(
            datasets,
            template,
            fit_engine=fit_engine,
            base_by_run=search_base_by_run,
            global_param_names=completion_global_names,
            local_param_names=ordered_target_local_names,
            fixed_param_names=fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache=cache,
            warm_start_by_run=warm_start_by_run,
            progress_callback=progress_callback,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
            initial_step_sizes=_step_hints_from_assessment(
                datasets,
                seed_assessment,
                target_global_names=completion_global_names,
                target_local_names=ordered_target_local_names,
            ),
        )
        _record_counter(instrumentation, "staged_completion_attempts")
        if candidate.is_successful:
            completion_candidates.append(candidate)
            _record_counter(instrumentation, "staged_completion_successes")

    if completion_candidates:
        best_completed = min(
            completion_candidates,
            key=lambda candidate: _assessment_sort_key(candidate, metric),
        )
        return best_completed, best_completed

    return None, best_partial


def _fit_exact_assignment(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    warm_start_by_run: dict[int, ParameterSet] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    search_strategy: str = "legacy",
    instrumentation: dict[str, object] | None = None,
    initial_step_sizes: dict[str, float] | None = None,
) -> GlobalCandidateAssessment:
    cache_key = (tuple(global_param_names), tuple(local_param_names))
    cached = cache.get(cache_key)
    if cached is not None:
        _record_counter(instrumentation, "exact_fit_cache_hits")
        return cached
    _record_counter(instrumentation, "exact_fit_invocations")

    _progress_log(
        progress_callback,
        f"{template.title}: fitting assignment "
        f"Global[{', '.join(global_param_names) or 'none'}], "
        f"Local[{', '.join(local_param_names) or 'none'}].",
    )
    attempt_variants = _assignment_attempt_variants(
        base_by_run,
        template,
        warm_start_by_run=warm_start_by_run,
    )
    free_count = _free_parameter_count(
        datasets,
        attempt_variants[0],
        global_param_names=global_param_names,
        local_param_names=local_param_names,
    )
    attempt_variants = _trim_assignment_attempt_variants(
        attempt_variants,
        free_count=free_count,
    )
    attempt_variants = tuple(
        _canonicalize_parameter_sets(
            attempt,
            template=template,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=fixed_param_names,
        )
        for attempt in attempt_variants
    )
    best_results: dict[int, FitResult] | None = None
    best_global = ParameterSet()
    best_score = float("inf")
    best_failure_message = "No fit attempts were created."
    step_hints = dict(initial_step_sizes or {})

    for variant_index, initial_params in enumerate(attempt_variants, start=1):
        _progress_log(
            progress_callback,
            f"{template.title}: trying initial parameter variant "
            f"{variant_index}/{len(attempt_variants)}.",
        )
        staged_initial_params = _staged_assignment_seed(
            datasets,
            template,
            fit_engine=fit_engine,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            initial_params=initial_params,
            progress_callback=progress_callback,
            max_cycles=4 if search_strategy == "staged_v2" else 2,
            include_mixed_polish=search_strategy == "staged_v2",
            instrumentation=instrumentation,
        )
        call_budget = _global_fit_call_budget(
            datasets,
            staged_initial_params,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            phase="full",
        )
        difficult_assignment = free_count >= 20 or len(local_param_names) >= 2
        _record_counter(instrumentation, "global_fit_calls")
        if step_hints:
            _record_counter(instrumentation, "curvature_hint_applications")
            _append_metric(instrumentation, "curvature_hint_sizes", len(step_hints))
        results_by_run, fitted_global = fit_engine.global_fit(
            datasets,
            template.model.function,
            list(global_param_names),
            list(local_param_names),
            staged_initial_params,
            max_calls=call_budget,
            migrad_iterations=7 if difficult_assignment else 5,
            use_simplex_rescue=difficult_assignment,
            minuit_strategy=2 if difficult_assignment else None,
            minuit_tol=0.05 if difficult_assignment else None,
            initial_step_sizes=step_hints or None,
        )
        _record_global_fit_diagnostics(instrumentation, results_by_run)
        results_by_run = _canonicalize_fit_results_by_run(
            results_by_run,
            template=template,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=fixed_param_names,
        )
        if all(result.success for result in results_by_run.values()):
            total_chi2 = float(sum(result.chi_squared for result in results_by_run.values()))
            if total_chi2 < best_score:
                best_score = total_chi2
                best_results = results_by_run
                best_global = fitted_global
                step_hints = _step_hints_from_fit_results(
                    datasets,
                    results_by_run,
                    target_global_names=global_param_names,
                    target_local_names=local_param_names,
                )
                continue
        if best_results is None:
            best_results = results_by_run
            best_global = fitted_global
        failure_message = _assignment_failure_message(results_by_run)
        if failure_message:
            best_failure_message = failure_message

    if best_results is None:
        best_results = {
            int(dataset.run_number): FitResult(
                success=False,
                message="No fit attempts were created.",
            )
            for dataset in datasets
        }

    fit_success = all(result.success for result in best_results.values())
    if not fit_success:
        rescue_params = (
            _clone_parameter_sets(warm_start_by_run)
            if warm_start_by_run is not None
            else _clone_parameter_sets(attempt_variants[0])
        )
        rescue_params = _staged_assignment_seed(
            datasets,
            template,
            fit_engine=fit_engine,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            initial_params=rescue_params,
            progress_callback=progress_callback,
        )
        rescue_budget = _global_fit_call_budget(
            datasets,
            rescue_params,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            phase="simplex",
        )
        _progress_log(
            progress_callback,
            f"{template.title}: retrying failed assignment with simplex.",
        )
        _record_counter(instrumentation, "simplex_rescues")
        _record_counter(instrumentation, "global_fit_calls")
        if step_hints:
            _record_counter(instrumentation, "curvature_hint_applications")
            _append_metric(instrumentation, "curvature_hint_sizes", len(step_hints))
        rescue_results, rescue_global = fit_engine.global_fit(
            datasets,
            template.model.function,
            list(global_param_names),
            list(local_param_names),
            rescue_params,
            method="simplex",
            max_calls=rescue_budget,
            minuit_strategy=2 if free_count >= 20 else None,
            minuit_tol=0.05 if free_count >= 20 else None,
            initial_step_sizes=step_hints or None,
        )
        _record_global_fit_diagnostics(instrumentation, rescue_results)
        rescue_results = _canonicalize_fit_results_by_run(
            rescue_results,
            template=template,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=fixed_param_names,
        )
        if all(result.success for result in rescue_results.values()):
            best_results = rescue_results
            best_global = rescue_global
            fit_success = True
            _progress_log(
                progress_callback,
                f"{template.title}: simplex rescue succeeded.",
            )
        else:
            rescue_message = _assignment_failure_message(rescue_results)
            if rescue_message:
                best_failure_message = rescue_message

    sample_count = int(sum(dataset.n_points for dataset in datasets))
    parameter_count = len(global_param_names) + len(local_param_names) * len(datasets)
    if fit_success:
        total_chi2 = float(sum(result.chi_squared for result in best_results.values()))
        aic, aicc, bic = compute_information_criteria(
            total_chi2,
            parameter_count,
            sample_count,
        )
    else:
        aic = float("inf")
        aicc = float("inf")
        bic = float("inf")

    run_diagnostics: list[RunResidualDiagnostic] = []
    fitted_curves_by_run: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    component_curves_by_run: dict[int, tuple[tuple[str, NDArray[np.float64]], ...]] = {}

    for dataset in datasets:
        run_number = int(dataset.run_number)
        result = best_results.get(
            run_number,
            FitResult(success=False, message="Missing global fit result"),
        )
        (
            residual_rms,
            runs_z_score,
            max_abs_autocorrelation,
            residual_fft_peak_snr,
        ) = _residual_diagnostics(dataset, result)
        bound_hits = _bound_hit_names(result.parameters)
        gate_reasons = _filtered_gate_reasons(
            fit_result=result,
            residual_rms=residual_rms,
            runs_z_score=runs_z_score,
            max_abs_autocorrelation=max_abs_autocorrelation,
            residual_fft_peak_snr=residual_fft_peak_snr,
            bound_hits=bound_hits,
        )
        run_diagnostics.append(
            RunResidualDiagnostic(
                run_number=run_number,
                run_label=dataset.run_label,
                axis_value=_axis_value(dataset, axis_key),
                residual_rms=residual_rms,
                runs_z_score=runs_z_score,
                max_abs_autocorrelation=max_abs_autocorrelation,
                residual_fft_peak_snr=residual_fft_peak_snr,
                gate_passed=not gate_reasons,
                gate_reasons=tuple(gate_reasons),
            )
        )

        fitted_time, fitted_curve, component_curves = _dense_fit_curves(
            dataset,
            template.model,
            result.parameters,
            fallback_parameters=base_by_run.get(run_number),
        )
        fitted_curves_by_run[run_number] = (fitted_time, fitted_curve)
        component_curves_by_run[run_number] = component_curves

    series_warnings = tuple(
        _series_warnings(
            datasets,
            run_diagnostics,
            results_by_run=best_results,
            local_param_names=local_param_names,
        )
    )
    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=best_results,
        global_parameters=best_global,
        global_param_names=tuple(global_param_names),
        local_param_names=tuple(local_param_names),
        fixed_param_names=tuple(fixed_param_names),
        parameter_recommendations=(),
        run_diagnostics=tuple(run_diagnostics),
        series_warnings=series_warnings,
        aic=float(aic),
        aicc=None if aicc is None else float(aicc),
        bic=float(bic),
        selected_score=_metric_value(metric, aic, aicc, bic),
        fitted_curves_by_run=fitted_curves_by_run,
        component_curves_by_run=component_curves_by_run,
    )
    if assessment.is_successful:
        _progress_log(
            progress_callback,
            f"{template.title}: assignment complete with "
            f"{metric.value} = {assessment.metric_value(metric):.3f}.",
        )
    else:
        _progress_log(
            progress_callback,
            f"{template.title}: assignment failed. {best_failure_message}",
        )
    cache[cache_key] = assessment
    return assessment


def _build_parameter_recommendations(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment,
    *,
    template: CandidateTemplate,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    progress_callback: Callable[[str], None] | None = None,
    names_to_test: set[str] | None = None,
    warm_start_cache: dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ] | None = None,
) -> tuple[GlobalParameterRecommendation, ...]:
    recommendations: list[GlobalParameterRecommendation] = []
    current_local = set(assessment.local_param_names)
    fixed_names = set(fixed_param_names)
    current_score = float(assessment.metric_value(metric))

    for name in template.model.param_names:
        if name in fixed_names:
            continue

        total_variation, roughness = _parameter_trace_roughness(
            datasets,
            assessment,
            name,
        )
        if names_to_test is not None and name not in names_to_test:
            current_role = "Local" if name in current_local else "Global"
            recommendations.append(
                GlobalParameterRecommendation(
                    name=name,
                    recommended_role=current_role,
                    global_score=current_score,
                    local_score=current_score,
                    score_delta=0.0,
                    total_variation=total_variation,
                    roughness=roughness,
                    rationale=(
                        f"Staged search kept {name} {current_role}; skipped an extra "
                        "exact role retest because the relaxed proposal was not ambiguous."
                    ),
                )
            )
            continue

        if name in current_local:
            _progress_log(
                progress_callback,
                f"{template.title}: testing whether {name} can be shared globally.",
            )
            global_local_names = tuple(sorted(current_local - {name}))
            global_global_names = tuple(
                pname
                for pname in template.model.param_names
                if pname not in fixed_names and pname not in global_local_names
            )
            alternative = _fit_exact_assignment(
                datasets,
                template,
                fit_engine=fit_engine,
                base_by_run=base_by_run,
                global_param_names=global_global_names,
                local_param_names=global_local_names,
                fixed_param_names=fixed_param_names,
                axis_key=axis_key,
                metric=metric,
                cache=cache,
                warm_start_by_run=_warm_start_parameter_sets(
                    datasets,
                    assessment=assessment,
                    base_by_run=base_by_run,
                    target_global_names=global_global_names,
                    target_local_names=global_local_names,
                    fit_engine=fit_engine,
                    template=template,
                    progress_callback=progress_callback,
                    cache=warm_start_cache,
                ),
                progress_callback=progress_callback,
                initial_step_sizes=_step_hints_from_assessment(
                    datasets,
                    assessment,
                    target_global_names=global_global_names,
                    target_local_names=global_local_names,
                ),
            )
            local_score = assessment.metric_value(metric)
            global_score = alternative.metric_value(metric)
            improvement = global_score - local_score
            keep_local = (
                not alternative.is_successful
                or improvement > _ROLE_DELTA_THRESHOLD
                or (assessment.residual_gate_passed and not alternative.residual_gate_passed)
            )
            recommended_role = "Local" if keep_local else "Global"
            rationale = (
                f"Keeping {name} Local improves the penalized score by {improvement:.2f}."
                if keep_local and np.isfinite(improvement)
                else f"{name} is only weakly supported as Local."
            )
            delta = improvement
        else:
            _progress_log(
                progress_callback,
                f"{template.title}: testing whether {name} should become local.",
            )
            local_local_names = tuple(sorted((*current_local, name)))
            local_global_names = tuple(
                pname
                for pname in template.model.param_names
                if pname not in fixed_names and pname not in local_local_names
            )
            alternative = _fit_exact_assignment(
                datasets,
                template,
                fit_engine=fit_engine,
                base_by_run=base_by_run,
                global_param_names=local_global_names,
                local_param_names=local_local_names,
                fixed_param_names=fixed_param_names,
                axis_key=axis_key,
                metric=metric,
                cache=cache,
                warm_start_by_run=_warm_start_parameter_sets(
                    datasets,
                    assessment=assessment,
                    base_by_run=base_by_run,
                    target_global_names=local_global_names,
                    target_local_names=local_local_names,
                    fit_engine=fit_engine,
                    template=template,
                    progress_callback=progress_callback,
                    cache=warm_start_cache,
                ),
                progress_callback=progress_callback,
                initial_step_sizes=_step_hints_from_assessment(
                    datasets,
                    assessment,
                    target_global_names=local_global_names,
                    target_local_names=local_local_names,
                ),
            )
            global_score = assessment.metric_value(metric)
            local_score = alternative.metric_value(metric)
            improvement = global_score - local_score
            make_local = (
                alternative.is_successful
                and improvement > _ROLE_DELTA_THRESHOLD
                and (alternative.residual_gate_passed or not assessment.residual_gate_passed)
            )
            recommended_role = "Local" if make_local else "Global"
            rationale = (
                f"Localizing {name} improves the penalized score by {improvement:.2f}."
                if make_local and np.isfinite(improvement)
                else (f"Localizing {name} does not overcome the complexity penalty.")
            )
            delta = improvement

        recommendations.append(
            GlobalParameterRecommendation(
                name=name,
                recommended_role=recommended_role,
                global_score=float(global_score),
                local_score=float(local_score),
                score_delta=float(abs(delta)) if np.isfinite(delta) else float("inf"),
                total_variation=total_variation,
                roughness=roughness,
                rationale=rationale,
            )
        )

    return tuple(recommendations)


def _initial_parameter_sets_for_candidate(
    datasets: list[MuonDataset],
    fingerprints_by_run: dict[int, SpectrumFingerprint],
    template: CandidateTemplate,
    *,
    current_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    fixed_param_names: tuple[str, ...],
) -> dict[int, ParameterSet]:
    base_by_run: dict[int, ParameterSet] = {}
    for dataset in datasets:
        run_number = int(dataset.run_number)
        parameters = _initial_parameters_for_template(
            dataset,
            fingerprints_by_run[run_number],
            template,
        )
        for parameter in parameters:
            if parameter.name in current_values:
                try:
                    parameter.value = float(current_values[parameter.name])
                except (TypeError, ValueError):
                    pass
            if parameter.name in parameter_bounds:
                min_val, max_val = parameter_bounds[parameter.name]
                parameter.min = float(min_val)
                parameter.max = float(max_val)
            parameter.value = float(np.clip(parameter.value, parameter.min, parameter.max))
            if parameter.name in fixed_param_names:
                parameter.fixed = True
        base_by_run[run_number] = parameters
    return base_by_run


def _initial_param_variants(
    base_by_run: dict[int, ParameterSet],
    template: CandidateTemplate,
) -> tuple[dict[int, ParameterSet], ...]:
    per_run_variants = {
        run_number: _parameter_variants(base_parameters, template=template)
        for run_number, base_parameters in base_by_run.items()
    }
    variant_count = min(3, min(len(variants) for variants in per_run_variants.values()))
    combined: list[dict[int, ParameterSet]] = []
    for idx in range(variant_count):
        combined.append(
            {
                run_number: _clone_parameter_set(variants[idx])
                for run_number, variants in per_run_variants.items()
            }
        )
    return tuple(combined)


def _single_run_prefit_parameter_sets(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    fixed_param_names: tuple[str, ...],
    progress_callback: Callable[[str], None] | None = None,
    instrumentation: dict[str, object] | None = None,
    cache: dict[
        tuple[tuple[str, ...], tuple[tuple[int, tuple[tuple[str, float], ...]], ...]],
        dict[int, ParameterSet],
    ] | None = None,
) -> dict[int, ParameterSet]:
    cache_key = (
        tuple(fixed_param_names),
        _parameter_set_signature(base_by_run),
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return _clone_parameter_sets(cached)

    global_names = tuple(
        name for name in template.model.param_names if name not in fixed_param_names
    )
    seeded = _canonicalize_parameter_sets(
        base_by_run,
        template=template,
        global_param_names=global_names,
        local_param_names=(),
        fixed_param_names=fixed_param_names,
    )
    success_count = 0
    previous_success: ParameterSet | None = None

    _progress_log(
        progress_callback,
        f"{template.title}: prefitting each dataset individually for staged seeds.",
    )
    for dataset in datasets:
        run_number = int(dataset.run_number)
        attempt_params: list[ParameterSet] = []
        if previous_success is not None:
            neighbor_seed = _clone_parameter_set(seeded[run_number])
            for parameter in neighbor_seed:
                if parameter.name in previous_success:
                    parameter.value = float(
                        np.clip(
                            previous_success[parameter.name].value,
                            parameter.min,
                            parameter.max,
                        )
                    )
            attempt_params.append(neighbor_seed)

        attempt_params.append(_clone_parameter_set(seeded[run_number]))
        for variant_map in _initial_param_variants({run_number: base_by_run[run_number]}, template):
            attempt_params.append(_clone_parameter_set(variant_map[run_number]))

        unique_attempts: list[ParameterSet] = []
        seen_signatures: set[tuple[tuple[str, float], ...]] = set()
        for attempt in attempt_params:
            canonical_attempt = _canonicalize_parameter_sets(
                {run_number: attempt},
                template=template,
                global_param_names=global_names,
                local_param_names=(),
                fixed_param_names=fixed_param_names,
            )[run_number]
            signature = tuple(
                (parameter.name, float(parameter.value)) for parameter in canonical_attempt
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_attempts.append(canonical_attempt)

        best_result: FitResult | None = None
        for attempt in unique_attempts[:3]:
            try:
                result = fit_engine.fit(
                    dataset,
                    template.model.function,
                    _clone_parameter_set(attempt),
                    method="migrad",
                )
            except Exception:
                continue

            if not result.success:
                try:
                    simplex_result = fit_engine.fit(
                        dataset,
                        template.model.function,
                        _clone_parameter_set(attempt),
                        method="simplex",
                    )
                except Exception:
                    simplex_result = FitResult(success=False, message="Single-run simplex failed")
                if simplex_result.success and (
                    best_result is None or simplex_result.chi_squared < best_result.chi_squared
                ):
                    best_result = simplex_result
                elif result.success and (
                    best_result is None or result.chi_squared < best_result.chi_squared
                ):
                    best_result = result
                continue

            if best_result is None or result.chi_squared < best_result.chi_squared:
                best_result = result

        if best_result is None or not best_result.success:
            continue

        canonical_result = _canonicalize_fit_results_by_run(
            {run_number: best_result},
            template=template,
            global_param_names=global_names,
            local_param_names=(),
            fixed_param_names=fixed_param_names,
        )[run_number]
        seeded[run_number] = _merge_result_values_into_parameter_sets(
            {run_number: seeded[run_number]},
            {run_number: canonical_result},
        )[run_number]
        previous_success = canonical_result.parameters
        success_count += 1
        _record_counter(instrumentation, "single_run_prefit_successes")

    _progress_log(
        progress_callback,
        f"{template.title}: single-run prefits succeeded for "
        f"{success_count}/{len(datasets)} datasets.",
    )
    prefitted = _canonicalize_parameter_sets(
        seeded,
        template=template,
        global_param_names=global_names,
        local_param_names=(),
        fixed_param_names=fixed_param_names,
    )
    if cache is not None:
        cache[cache_key] = _clone_parameter_sets(prefitted)
    return prefitted


def _assignment_attempt_variants(
    base_by_run: dict[int, ParameterSet],
    template: CandidateTemplate,
    *,
    warm_start_by_run: dict[int, ParameterSet] | None,
) -> tuple[dict[int, ParameterSet], ...]:
    attempts: list[dict[int, ParameterSet]] = []
    if warm_start_by_run is not None:
        attempts.append(_clone_parameter_sets(warm_start_by_run))
        warm_variants = _initial_param_variants(warm_start_by_run, template)
        attempts.extend(
            _clone_parameter_sets(variant)
            for variant in warm_variants[1:]
        )
    attempts.extend(_initial_param_variants(base_by_run, template))

    unique_attempts: list[dict[int, ParameterSet]] = []
    seen_signatures: set[tuple[tuple[int, tuple[tuple[str, float], ...]], ...]] = set()
    for attempt in attempts:
        signature = _parameter_set_signature(attempt)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_attempts.append(attempt)
    return tuple(unique_attempts)


def _canonicalize_parameter_sets(
    parameter_sets: dict[int, ParameterSet],
    *,
    template: CandidateTemplate,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
) -> dict[int, ParameterSet]:
    groups = _canonical_component_groups(
        template,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        fixed_param_names=fixed_param_names,
    )
    if not groups:
        return _clone_parameter_sets(parameter_sets)

    canonical = _clone_parameter_sets(parameter_sets)
    for params in canonical.values():
        _canonicalize_parameter_set_in_place(params, groups)
    return canonical


def _trim_assignment_attempt_variants(
    attempt_variants: tuple[dict[int, ParameterSet], ...],
    *,
    free_count: int,
) -> tuple[dict[int, ParameterSet], ...]:
    if free_count >= _EXTREME_DIMENSION_FREE_COUNT:
        return attempt_variants[:2]
    if free_count >= _HIGH_DIMENSION_FREE_COUNT:
        return attempt_variants[:3]
    return attempt_variants


def _staged_assignment_seed(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    initial_params: dict[int, ParameterSet],
    progress_callback: Callable[[str], None] | None = None,
    max_cycles: int = 2,
    include_mixed_polish: bool = False,
    instrumentation: dict[str, object] | None = None,
) -> dict[int, ParameterSet]:
    initial_params = _canonicalize_parameter_sets(
        initial_params,
        template=template,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        fixed_param_names=(),
    )
    free_global_names = tuple(
        name
        for name in global_param_names
        if not initial_params[int(datasets[0].run_number)][name].fixed
    )
    free_local_names = tuple(
        name
        for name in local_param_names
        if any(
            not initial_params[int(dataset.run_number)][name].fixed
            for dataset in datasets
        )
    )
    if not free_global_names or not free_local_names:
        return _clone_parameter_sets(initial_params)

    staged_seed = _clone_parameter_sets(initial_params)
    stage_messages: list[str] = []
    completed_cycles = 0
    best_cycle_score = float("inf")
    step_hints: dict[str, float] = {}

    for _cycle_index in range(max_cycles):
        cycle_messages: list[str] = []
        cycle_score = best_cycle_score

        local_only_input = _parameter_sets_for_stage(
            staged_seed,
            active_names=set(free_local_names),
        )
        local_only_budget = _global_fit_call_budget(
            datasets,
            local_only_input,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            phase="stage",
        )
        _record_counter(instrumentation, "global_fit_calls")
        if step_hints:
            _record_counter(instrumentation, "curvature_hint_applications")
        local_results, _ = fit_engine.global_fit(
            datasets,
            template.model.function,
            list(global_param_names),
            list(local_param_names),
            local_only_input,
            max_calls=local_only_budget,
            migrad_iterations=6,
            use_simplex_rescue=True,
            minuit_strategy=2 if len(free_local_names) >= 2 else None,
            minuit_tol=0.05 if len(free_local_names) >= 2 else None,
            initial_step_sizes=step_hints or None,
        )
        _record_global_fit_diagnostics(instrumentation, local_results)
        local_results = _canonicalize_fit_results_by_run(
            local_results,
            template=template,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=(),
        )
        if all(result.success for result in local_results.values()):
            staged_seed = _merge_result_values_into_parameter_sets(staged_seed, local_results)
            cycle_messages.append("local-only refinement")
            step_hints = _step_hints_from_fit_results(
                datasets,
                local_results,
                target_global_names=global_param_names,
                target_local_names=local_param_names,
            )
            cycle_score = min(
                cycle_score,
                float(sum(result.chi_squared for result in local_results.values())),
            )

        global_only_input = _parameter_sets_for_stage(
            staged_seed,
            active_names=set(free_global_names),
        )
        global_only_budget = _global_fit_call_budget(
            datasets,
            global_only_input,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            phase="stage",
        )
        _record_counter(instrumentation, "global_fit_calls")
        if step_hints:
            _record_counter(instrumentation, "curvature_hint_applications")
        global_results, _ = fit_engine.global_fit(
            datasets,
            template.model.function,
            list(global_param_names),
            list(local_param_names),
            global_only_input,
            max_calls=global_only_budget,
            migrad_iterations=6,
            use_simplex_rescue=True,
            minuit_strategy=2 if len(free_global_names) >= 4 else None,
            minuit_tol=0.05 if len(free_global_names) >= 4 else None,
            initial_step_sizes=step_hints or None,
        )
        _record_global_fit_diagnostics(instrumentation, global_results)
        global_results = _canonicalize_fit_results_by_run(
            global_results,
            template=template,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=(),
        )
        if all(result.success for result in global_results.values()):
            staged_seed = _merge_result_values_into_parameter_sets(staged_seed, global_results)
            cycle_messages.append("global-only refinement")
            step_hints = _step_hints_from_fit_results(
                datasets,
                global_results,
                target_global_names=global_param_names,
                target_local_names=local_param_names,
            )
            cycle_score = min(
                cycle_score,
                float(sum(result.chi_squared for result in global_results.values())),
            )

        if include_mixed_polish and cycle_messages:
            mixed_input = _parameter_sets_for_stage(
                staged_seed,
                active_names=set((*free_global_names, *free_local_names)),
            )
            mixed_budget = min(
                1200,
                _global_fit_call_budget(
                    datasets,
                    mixed_input,
                    global_param_names=global_param_names,
                    local_param_names=local_param_names,
                    phase="stage",
                ),
            )
            _record_counter(instrumentation, "global_fit_calls")
            if step_hints:
                _record_counter(instrumentation, "curvature_hint_applications")
            mixed_results, _ = fit_engine.global_fit(
                datasets,
                template.model.function,
                list(global_param_names),
                list(local_param_names),
                mixed_input,
                max_calls=mixed_budget,
                migrad_iterations=4,
                use_simplex_rescue=False,
                initial_step_sizes=step_hints or None,
            )
            _record_global_fit_diagnostics(instrumentation, mixed_results)
            mixed_results = _canonicalize_fit_results_by_run(
                mixed_results,
                template=template,
                global_param_names=global_param_names,
                local_param_names=local_param_names,
                fixed_param_names=(),
            )
            if all(result.success for result in mixed_results.values()):
                mixed_score = float(sum(result.chi_squared for result in mixed_results.values()))
                if mixed_score + 1e-6 < cycle_score:
                    staged_seed = _merge_result_values_into_parameter_sets(staged_seed, mixed_results)
                    step_hints = _step_hints_from_fit_results(
                        datasets,
                        mixed_results,
                        target_global_names=global_param_names,
                        target_local_names=local_param_names,
                    )
                    cycle_score = mixed_score
                    cycle_messages.append("mixed polish")

        if not cycle_messages:
            break
        if cycle_score + 1e-4 >= best_cycle_score:
            break
        completed_cycles += 1
        best_cycle_score = cycle_score
        stage_messages = cycle_messages

    if stage_messages:
        stage_summary = " then ".join(stage_messages)
        cycle_summary = f" over {completed_cycles} cycle" + ("s" if completed_cycles != 1 else "")
        _progress_log(
            progress_callback,
            f"{template.title}: staged {stage_summary} completed{cycle_summary} "
            "before the full solve.",
        )
    return staged_seed


def _parameter_sets_for_stage(
    parameter_sets: dict[int, ParameterSet],
    *,
    active_names: set[str],
) -> dict[int, ParameterSet]:
    staged = _clone_parameter_sets(parameter_sets)
    for params in staged.values():
        for parameter in params:
            if parameter.fixed:
                continue
            parameter.fixed = parameter.name not in active_names
    return staged


def _merge_result_values_into_parameter_sets(
    parameter_sets: dict[int, ParameterSet],
    results_by_run: dict[int, FitResult],
) -> dict[int, ParameterSet]:
    merged = _clone_parameter_sets(parameter_sets)
    for run_number, result in results_by_run.items():
        if not result.success:
            continue
        if int(run_number) not in merged:
            continue
        target_params = merged[int(run_number)]
        for parameter in result.parameters:
            if parameter.name not in target_params:
                continue
            target = target_params[parameter.name]
            target.value = float(np.clip(parameter.value, target.min, target.max))
    return merged


def _clone_parameter_sets(
    parameter_sets: dict[int, ParameterSet],
) -> dict[int, ParameterSet]:
    return {
        int(run_number): _clone_parameter_set(parameters)
        for run_number, parameters in parameter_sets.items()
    }


def _parameter_set_signature(
    parameter_sets: dict[int, ParameterSet],
) -> tuple[tuple[int, tuple[tuple[str, float], ...]], ...]:
    return tuple(
        (
            int(run_number),
            tuple(
                (parameter.name, float(parameter.value))
                for parameter in parameter_sets[run_number]
            ),
        )
        for run_number in sorted(parameter_sets)
    )


def _warm_start_cache_key(
    assessment: GlobalCandidateAssessment,
    base_by_run: dict[int, ParameterSet],
    *,
    target_global_names: tuple[str, ...],
    target_local_names: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
]:
    return (
        tuple(assessment.global_param_names),
        tuple(assessment.local_param_names),
        tuple(target_global_names),
        tuple(target_local_names),
        _parameter_set_signature(base_by_run),
    )


def _canonicalize_fit_results_by_run(
    results_by_run: dict[int, FitResult],
    *,
    template: CandidateTemplate,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
) -> dict[int, FitResult]:
    groups = _canonical_component_groups(
        template,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        fixed_param_names=fixed_param_names,
    )
    if not groups:
        return results_by_run

    canonicalized: dict[int, FitResult] = {}
    for run_number, result in results_by_run.items():
        if not result.success:
            canonicalized[int(run_number)] = result
            continue
        params = _clone_parameter_set(result.parameters)
        swap_plan = _canonicalize_parameter_set_in_place(params, groups)
        if not swap_plan:
            canonicalized[int(run_number)] = result
            continue
        uncertainties = dict(result.uncertainties)
        for source_name, destination_name in swap_plan.items():
            uncertainties[destination_name] = result.uncertainties.get(source_name, 0.0)
        canonicalized[int(run_number)] = replace(
            result,
            parameters=params,
            uncertainties=uncertainties,
        )
    return canonicalized


def _canonical_component_groups(
    template: CandidateTemplate,
    *,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
) -> tuple[tuple[dict[str, str], ...], ...]:
    if not _is_additive_relaxation_mixture_template(template):
        return ()

    fixed_names = set(fixed_param_names)
    global_names = set(global_param_names)
    local_names = set(local_param_names)
    relaxing_components = template.model.component_names[:-1]
    groups: list[tuple[dict[str, str], ...]] = []

    start = 0
    while start < len(relaxing_components):
        end = start + 1
        while (
            end < len(relaxing_components)
            and relaxing_components[end] == relaxing_components[start]
        ):
            end += 1
        if end - start > 1:
            role_buckets: dict[tuple[str, ...], list[dict[str, str]]] = {}
            for component_index in range(start, end):
                mapping = template.model._param_mappings[component_index]  # noqa: SLF001
                role_signature = tuple(
                    _parameter_role_for_name(
                        mapping[parameter_name],
                        fixed_names=fixed_names,
                        global_names=global_names,
                        local_names=local_names,
                    )
                    for parameter_name in template.model.components[component_index].param_names
                )
                role_buckets.setdefault(role_signature, []).append(mapping)
            for mappings in role_buckets.values():
                if len(mappings) > 1:
                    groups.append(tuple(mappings))
        start = end

    return tuple(groups)


def _parameter_role_for_name(
    name: str,
    *,
    fixed_names: set[str],
    global_names: set[str],
    local_names: set[str],
) -> str:
    if name in fixed_names:
        return "fixed"
    if name in local_names:
        return "local"
    if name in global_names:
        return "global"
    return "other"


def _canonicalize_parameter_set_in_place(
    parameters: ParameterSet,
    groups: tuple[tuple[dict[str, str], ...], ...],
) -> dict[str, str]:
    swap_plan: dict[str, str] = {}
    for group in groups:
        ordered = sorted(
            group,
            key=lambda mapping: _component_sort_key(parameters, mapping),
        )
        if ordered == list(group):
            continue

        snapshots = {
            index: {
                base_name: (
                    parameters[mapping[base_name]].value,
                    parameters[mapping[base_name]].fixed,
                    parameters[mapping[base_name]].expr,
                )
                for base_name in mapping
                if mapping[base_name] in parameters
            }
            for index, mapping in enumerate(group)
        }
        for destination_mapping, source_mapping in zip(group, ordered, strict=True):
            source_index = group.index(source_mapping)
            snapshot = snapshots[source_index]
            for base_name, destination_name in destination_mapping.items():
                if destination_name not in parameters or base_name not in snapshot:
                    continue
                value, fixed, expr = snapshot[base_name]
                parameter = parameters[destination_name]
                parameter.value = float(np.clip(value, parameter.min, parameter.max))
                parameter.fixed = fixed
                parameter.expr = expr
                swap_plan[source_mapping[base_name]] = destination_name
    return swap_plan


def _component_sort_key(
    parameters: ParameterSet,
    mapping: dict[str, str],
) -> tuple[float, float, str]:
    shape_name = mapping.get("Lambda", mapping.get("sigma", ""))
    shape_value = float(parameters[shape_name].value) if shape_name in parameters else -float("inf")
    amplitude_name = mapping.get("A", "")
    amplitude_value = (
        float(abs(parameters[amplitude_name].value))
        if amplitude_name in parameters
        else -float("inf")
    )
    return (-shape_value, -amplitude_value, shape_name)


def _global_fit_call_budget(
    datasets: list[MuonDataset],
    parameter_sets: dict[int, ParameterSet],
    *,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    phase: str,
) -> int:
    free_count = _free_parameter_count(
        datasets,
        parameter_sets,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
    )
    dataset_count = len(datasets)
    if phase == "stage":
        budget = 900 + (42 * free_count) + (16 * dataset_count)
        if free_count >= 24:
            budget += 400
        if free_count >= 40:
            budget += 600
        return int(min(max(900, budget), _STAGED_GLOBAL_FIT_MAX_CALLS))
    if phase == "simplex":
        budget = 1200 + (50 * free_count) + (18 * dataset_count)
        return int(
            min(
                max(_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS, budget),
                _GLOBAL_FIT_SIMPLEX_RESCUE_CALLS_CAP,
            )
        )
    budget = 1000 + (40 * free_count) + (14 * dataset_count)
    if free_count >= 40:
        budget += 400
    return int(min(max(_GLOBAL_FIT_MAX_CALLS, budget), _GLOBAL_FIT_MAX_CALLS_CAP))


def _free_parameter_count(
    datasets: list[MuonDataset],
    parameter_sets: dict[int, ParameterSet],
    *,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
) -> int:
    first_params = parameter_sets[int(datasets[0].run_number)]
    n_global = sum(1 for name in global_param_names if not first_params[name].fixed)
    n_local = sum(
        1
        for dataset in datasets
        for name in local_param_names
        if not parameter_sets[int(dataset.run_number)][name].fixed
    )
    return n_global + n_local


def _warm_start_parameter_sets(
    datasets: list[MuonDataset],
    *,
    assessment: GlobalCandidateAssessment,
    base_by_run: dict[int, ParameterSet],
    target_global_names: tuple[str, ...],
    target_local_names: tuple[str, ...],
    fit_engine: FitEngine | None = None,
    template: CandidateTemplate | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cache: dict[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        dict[int, ParameterSet],
    ] | None = None,
) -> dict[int, ParameterSet]:
    cache_key = _warm_start_cache_key(
        assessment,
        base_by_run,
        target_global_names=target_global_names,
        target_local_names=target_local_names,
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return _clone_parameter_sets(cached)

    shared_values = {
        name: _shared_parameter_seed(datasets, assessment, base_by_run, name)
        for name in target_global_names
    }
    seeded: dict[int, ParameterSet] = {}
    for dataset in datasets:
        run_number = int(dataset.run_number)
        base_params = _clone_parameter_set(base_by_run[run_number])
        result = assessment.fit_results_by_run.get(run_number)
        for parameter in base_params:
            if parameter.name in target_global_names:
                parameter.value = shared_values[parameter.name]
            elif parameter.name in target_local_names:
                parameter.value = _local_parameter_seed(
                    parameter.name,
                    result,
                    default=shared_values.get(
                        parameter.name,
                        _shared_parameter_seed(
                            datasets,
                            assessment,
                            base_by_run,
                            parameter.name,
                        ),
                    ),
                )
            elif result is not None and parameter.name in result.parameters:
                parameter.value = result.parameters[parameter.name].value
            parameter.value = float(
                np.clip(parameter.value, parameter.min, parameter.max)
            )
        seeded[run_number] = base_params

    newly_localized = tuple(
        name for name in target_local_names if name not in assessment.local_param_names
    )
    if not newly_localized or fit_engine is None or template is None:
        if cache is not None:
            cache[cache_key] = _clone_parameter_sets(seeded)
        return seeded

    _progress_log(
        progress_callback,
        f"{template.title}: prefitting newly localized parameters "
        f"[{', '.join(newly_localized)}] run-by-run.",
    )
    success_count = 0
    for dataset in datasets:
        run_number = int(dataset.run_number)
        prefit_params = _clone_parameter_set(seeded[run_number])
        for parameter in prefit_params:
            parameter.fixed = parameter.name not in newly_localized

        try:
            result = fit_engine.fit(
                dataset,
                template.model.function,
                prefit_params,
                method="migrad",
            )
        except Exception:
            continue

        if not result.success:
            simplex_result = fit_engine.fit(
                dataset,
                template.model.function,
                prefit_params,
                method="simplex",
            )
            result = simplex_result

        if not result.success:
            continue

        success_count += 1
        target_params = seeded[run_number]
        for parameter in result.parameters:
            if parameter.name in target_params:
                target_params[parameter.name].value = float(
                    np.clip(
                        parameter.value,
                        target_params[parameter.name].min,
                        target_params[parameter.name].max,
                    )
                )

    _progress_log(
        progress_callback,
        f"{template.title}: run-by-run prefits succeeded for "
        f"{success_count}/{len(datasets)} datasets.",
    )
    warmed = _canonicalize_parameter_sets(
        seeded,
        template=template,
        global_param_names=target_global_names,
        local_param_names=target_local_names,
        fixed_param_names=(),
    )
    if cache is not None:
        cache[cache_key] = _clone_parameter_sets(warmed)
    return warmed


def _shared_parameter_seed(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment,
    base_by_run: dict[int, ParameterSet],
    name: str,
) -> float:
    if name in assessment.global_parameters:
        return float(assessment.global_parameters[name].value)

    values: list[float] = []
    for dataset in datasets:
        result = assessment.fit_results_by_run.get(int(dataset.run_number))
        if result is not None and name in result.parameters:
            values.append(float(result.parameters[name].value))
    if values:
        return float(np.median(np.asarray(values, dtype=float)))

    base_values = [
        float(base_by_run[int(dataset.run_number)][name].value)
        for dataset in datasets
        if name in base_by_run[int(dataset.run_number)]
    ]
    if base_values:
        return float(np.median(np.asarray(base_values, dtype=float)))
    return 0.0


def _local_parameter_seed(
    name: str,
    result: FitResult | None,
    *,
    default: float,
) -> float:
    if result is not None and name in result.parameters:
        return float(result.parameters[name].value)
    return float(default)


def _assignment_failure_message(results_by_run: dict[int, FitResult]) -> str:
    messages = [
        str(result.message).strip()
        for result in results_by_run.values()
        if not result.success and isinstance(result.message, str) and result.message.strip()
    ]
    if not messages:
        return "Fit backend returned no detailed failure message."
    unique_messages: list[str] = []
    for message in messages:
        if message not in unique_messages:
            unique_messages.append(message)
    if len(unique_messages) == 1:
        return unique_messages[0]
    return "; ".join(unique_messages[:2])


def _ordered_datasets_with_axis(
    datasets: list[MuonDataset],
) -> tuple[list[MuonDataset], str, str, str | None]:
    field_values = np.array(
        [_field_value(dataset) for dataset in datasets],
        dtype=float,
    )
    temperature_values = np.array(
        [_temperature_value(dataset) for dataset in datasets],
        dtype=float,
    )

    field_unique = len(np.unique(np.round(field_values, 9)))
    temperature_unique = len(np.unique(np.round(temperature_values, 9)))
    field_span = float(np.nanmax(field_values) - np.nanmin(field_values))
    temperature_span = float(np.nanmax(temperature_values) - np.nanmin(temperature_values))

    mixed_axes_warning: str | None = None
    if field_unique > 1 and temperature_unique > 1:
        mixed_axes_warning = (
            "The selected datasets vary along both field and temperature. "
            "Global Fit Wizard v1 only auto-recommends ordered one-axis series."
        )

    if field_unique > 1 and (field_span > temperature_span or temperature_unique <= 1):
        ordered = sorted(
            datasets,
            key=lambda dataset: (_field_value(dataset), int(dataset.run_number)),
        )
        return ordered, "field", "Field (G)", mixed_axes_warning
    if temperature_unique > 1:
        ordered = sorted(
            datasets,
            key=lambda dataset: (
                _temperature_value(dataset),
                int(dataset.run_number),
            ),
        )
        return ordered, "temperature", "Temperature (K)", mixed_axes_warning
    ordered = sorted(datasets, key=lambda dataset: int(dataset.run_number))
    return ordered, "run", "Run", mixed_axes_warning


def _aggregate_fingerprints(
    fingerprints: list[SpectrumFingerprint],
) -> SpectrumFingerprint:
    if not fingerprints:
        raise ValueError("At least one fingerprint is required.")

    def _median(attr: str) -> float:
        return float(np.median([getattr(fingerprint, attr) for fingerprint in fingerprints]))

    def _count_true(attr: str) -> bool:
        return bool(any(getattr(fingerprint, attr) for fingerprint in fingerprints))

    return SpectrumFingerprint(
        tail_estimate=_median("tail_estimate"),
        initial_amplitude_estimate=_median("initial_amplitude_estimate"),
        zero_crossings=int(round(_median("zero_crossings"))),
        smoothed_zero_crossings=int(round(_median("smoothed_zero_crossings"))),
        smoothed_turning_points=int(round(_median("smoothed_turning_points"))),
        dominant_fft_frequency_mhz=_median("dominant_fft_frequency_mhz"),
        dominant_fft_snr=_median("dominant_fft_snr"),
        dominant_fft_cycles_in_window=_median("dominant_fft_cycles_in_window"),
        monotonic_decay_fraction=_median("monotonic_decay_fraction"),
        early_time_curvature=_median("early_time_curvature"),
        semilog_slope_ratio=_median("semilog_slope_ratio"),
        late_time_dip_recovery_score=_median("late_time_dip_recovery_score"),
        oscillatory_hint=_count_true("oscillatory_hint"),
        kt_like_hint=_count_true("kt_like_hint"),
        multi_rate_hint=_count_true("multi_rate_hint"),
    )


def _series_warnings(
    datasets: list[MuonDataset],
    run_diagnostics: list[RunResidualDiagnostic],
    results_by_run: dict[int, FitResult],
    *,
    local_param_names: tuple[str, ...],
) -> list[str]:
    warnings: list[str] = []
    if not run_diagnostics:
        return warnings

    residual_failures = [
        index for index, diagnostic in enumerate(run_diagnostics) if not diagnostic.gate_passed
    ]
    if residual_failures:
        start = residual_failures[0]
        stop = residual_failures[-1]
        if stop > start:
            warnings.append(
                "Residual warnings cluster across runs "
                f"{datasets[start].run_label}-{datasets[stop].run_label}."
            )

    warnings.extend(_fingerprint_jump_warnings(datasets))

    for name in local_param_names:
        total_variation, roughness = _parameter_trace_roughness_from_results(
            datasets,
            results_by_run,
            name,
        )
        if total_variation >= 2.5 or roughness >= 0.9:
            warnings.append(
                f"{name} changes abruptly across the ordered series "
                f"(TV {total_variation:.2f}, roughness {roughness:.2f})."
            )
    return warnings


def _filtered_gate_reasons(
    *,
    fit_result: FitResult,
    residual_rms: float,
    runs_z_score: float,
    max_abs_autocorrelation: float,
    residual_fft_peak_snr: float,
    bound_hits: list[str],
) -> list[str]:
    reasons = _residual_gate_reasons(
        fit_result=fit_result,
        residual_rms=residual_rms,
        runs_z_score=runs_z_score,
        max_abs_autocorrelation=max_abs_autocorrelation,
        residual_fft_peak_snr=residual_fft_peak_snr,
        bound_hits=bound_hits,
    )
    if residual_rms <= _LOW_RESIDUAL_RMS_FOR_STRUCTURE_WARNINGS:
        reasons = [
            reason
            for reason in reasons
            if not (
                reason.startswith("runs-test z score suggests structure")
                or reason.startswith("low-lag residual autocorrelation is high")
                or reason.startswith("residual FFT shows a strong peak")
            )
        ]
    return reasons


def _fingerprint_jump_warnings(datasets: list[MuonDataset]) -> list[str]:
    warnings: list[str] = []
    if len(datasets) < 4:
        return warnings

    fingerprints = [fingerprint_spectrum(dataset) for dataset in datasets]
    features = np.array(
        [
            [
                fingerprint.semilog_slope_ratio,
                fingerprint.dominant_fft_frequency_mhz,
                fingerprint.dominant_fft_snr,
                fingerprint.late_time_dip_recovery_score,
                float(fingerprint.smoothed_turning_points),
            ]
            for fingerprint in fingerprints
        ],
        dtype=float,
    )
    centered = features - np.nanmedian(features, axis=0)
    scale = np.nanstd(centered, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = centered / scale
    jump_strength = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    if jump_strength.size == 0:
        return warnings

    median_jump = float(np.median(jump_strength))
    max_index = int(np.argmax(jump_strength))
    max_jump = float(jump_strength[max_index])
    if max_jump > max(3.0 * max(median_jump, 1e-6), 3.0):
        warnings.append(
            "Fingerprint features change abruptly between "
            f"{datasets[max_index].run_label} and "
            f"{datasets[max_index + 1].run_label}."
        )
    return warnings


def _parameter_trace_roughness(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment,
    name: str,
) -> tuple[float, float]:
    return _parameter_trace_roughness_from_results(
        datasets,
        assessment.fit_results_by_run,
        name,
    )


def _parameter_trace_roughness_from_results(
    datasets: list[MuonDataset],
    results_by_run: dict[int, FitResult],
    name: str,
) -> tuple[float, float]:
    values = np.array(
        [
            results_by_run[int(dataset.run_number)].parameters[name].value
            for dataset in datasets
            if int(dataset.run_number) in results_by_run
            and name in results_by_run[int(dataset.run_number)].parameters
        ],
        dtype=float,
    )
    if values.size < 2:
        return 0.0, 0.0

    span = max(
        float(np.max(values) - np.min(values)),
        float(np.max(np.abs(values))),
        1e-9,
    )
    total_variation = float(np.sum(np.abs(np.diff(values))) / span)
    if values.size < 3:
        return total_variation, 0.0
    roughness = float(np.sqrt(np.mean(np.square(np.diff(values, n=2)))) / span)
    return total_variation, roughness


def _prefer_role_change(
    candidate: GlobalCandidateAssessment,
    incumbent: GlobalCandidateAssessment,
    *,
    metric: SelectionMetric,
) -> bool:
    if not candidate.is_successful:
        return False
    score_delta = incumbent.metric_value(metric) - candidate.metric_value(metric)
    if score_delta > _role_delta_threshold(candidate, incumbent):
        return True
    if (
        not incumbent.residual_gate_passed
        and candidate.residual_gate_passed
        and score_delta >= -1.0
    ):
        return True
    return False


def _prefer_simpler_assignment(
    candidate: GlobalCandidateAssessment,
    incumbent: GlobalCandidateAssessment,
    *,
    metric: SelectionMetric,
) -> bool:
    if not candidate.is_successful:
        return False
    score_penalty = candidate.metric_value(metric) - incumbent.metric_value(metric)
    if (
        score_penalty <= _ROLE_DELTA_THRESHOLD
        and len(candidate.local_param_names) < len(incumbent.local_param_names)
        and (candidate.residual_gate_passed or not incumbent.residual_gate_passed)
    ):
        return True
    return False


def _assessment_sort_key(
    assessment: GlobalCandidateAssessment,
    metric: SelectionMetric,
) -> tuple[float, int, int, int, int, int, str]:
    return (
        float(assessment.metric_value(metric)),
        int(_persistent_lower_bound_penalty(assessment)),
        int(_localisation_penalty(assessment.local_param_names)),
        int(assessment.parameter_count),
        int(len(assessment.local_param_names)),
        int(assessment.additive_terms),
        assessment.template.title,
    )


def _metric_value(
    metric: SelectionMetric,
    aic: float,
    aicc: float | None,
    bic: float,
) -> float:
    if metric == SelectionMetric.AIC:
        return aic
    if metric == SelectionMetric.BIC:
        return bic
    return aicc if aicc is not None else aic


def _role_delta_threshold(
    candidate: GlobalCandidateAssessment,
    incumbent: GlobalCandidateAssessment,
) -> float:
    threshold = _ROLE_DELTA_THRESHOLD
    newly_localized = [
        name for name in candidate.local_param_names if name not in incumbent.local_param_names
    ]
    if len(newly_localized) != 1:
        return threshold
    priority = _parameter_localisation_priority(newly_localized[0])
    if priority >= 3:
        return threshold + 2.0
    if priority == 2:
        return threshold + 1.0
    candidate_penalty = _persistent_lower_bound_penalty(candidate)
    incumbent_penalty = _persistent_lower_bound_penalty(incumbent)
    if candidate_penalty > incumbent_penalty:
        threshold += min(3.0, 0.5 * (candidate_penalty - incumbent_penalty))
    return threshold


def _localisation_penalty(local_param_names: tuple[str, ...]) -> int:
    return sum(_parameter_localisation_priority(name) for name in local_param_names)


def _parameter_localisation_priority(name: str) -> int:
    return parameter_localisation_priority(name)


def _tiered_role_candidates(
    remaining: tuple[str, ...],
    current_local_names: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    remaining_sorted = sorted(
        remaining,
        key=lambda name: (
            _parameter_localisation_priority(name),
            _paired_local_count(name, current_local_names),
            name,
        ),
    )
    grouped: list[tuple[str, ...]] = []
    current_priority: int | None = None
    bucket: list[str] = []
    for name in remaining_sorted:
        priority = _parameter_localisation_priority(name)
        if current_priority is None or priority == current_priority:
            bucket.append(name)
            current_priority = priority
            continue
        grouped.append(tuple(bucket[:_MAX_ROLE_CANDIDATES_PER_TIER]))
        bucket = [name]
        current_priority = priority
    if bucket:
        grouped.append(tuple(bucket[:_MAX_ROLE_CANDIDATES_PER_TIER]))
    return tuple(grouped)


def _paired_local_count(name: str, current_local_names: tuple[str, ...]) -> int:
    suffix = name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        return 0
    return sum(1 for local_name in current_local_names if local_name.endswith(f"_{suffix}"))


def _persistent_lower_bound_penalty(assessment: GlobalCandidateAssessment) -> int:
    repeated_hits: dict[str, int] = {}
    for diagnostic in assessment.run_diagnostics:
        for reason in diagnostic.gate_reasons:
            if " at lower bound" not in reason:
                continue
            parameter_name = reason.split(" at lower bound", 1)[0]
            repeated_hits[parameter_name] = repeated_hits.get(parameter_name, 0) + 1

    if not repeated_hits:
        return 0

    threshold = max(3, len(assessment.run_diagnostics) // 4)
    return sum(
        count - threshold + 1
        for count in repeated_hits.values()
        if count >= threshold
    )


def _axis_value(dataset: MuonDataset, axis_key: str) -> float:
    if axis_key == "field":
        return _field_value(dataset)
    if axis_key == "temperature":
        return _temperature_value(dataset)
    return float(dataset.run_number)


def _field_value(dataset: MuonDataset) -> float:
    field = dataset.run.field if dataset.run is not None else 0.0
    return float(dataset.metadata.get("field", field))


def _temperature_value(dataset: MuonDataset) -> float:
    temperature = dataset.run.temperature if dataset.run is not None else 0.0
    return float(dataset.metadata.get("temperature", temperature))


def _progress_log(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(str(message))


def _threadsafe_progress_callback(
    progress_callback: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    if progress_callback is None:
        return None

    lock = threading.Lock()

    def _wrapped(message: str) -> None:
        with lock:
            progress_callback(str(message))

    return _wrapped


def _template_worker_count(task_count: int) -> int:
    if task_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(_MAX_TEMPLATE_WORKERS, cpu_count, task_count))
