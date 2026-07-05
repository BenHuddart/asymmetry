"""Core analysis helpers for the global fit wizard."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import geometry_from_field_direction
from asymmetry.core.fitting.composite import (
    CompositeModel,
    _legacy_fraction_rename_map,
    migrate_legacy_fraction_parameter_set,
)
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
    _bound_hit_names,
    _clone_parameter_set,
    _dense_fit_curves,
    _initial_parameters_for_template,
    _is_additive_relaxation_mixture_template,
    _migrate_fit_result_fractions,
    _needs_fit_backend_fallback,
    _parameter_variants,
    _residual_diagnostics,
    _residual_gate_reasons,
    _scipy_fit_fallback,
    build_candidate_templates,
    build_fit_wizard_recommendation_for_templates,
    build_wizard_families,
    candidate_template_keys,
    compute_information_criteria,
    fingerprint_spectrum,
    recommendation_template_keys,
    rerank_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search import (
    GlobalSearchConfig,
)
from asymmetry.core.fitting.global_search.heuristics import (
    localisation_threshold_scale,
    parameter_localisation_priority,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.peak_detection import (
    analyze_dataset_peaks,
    match_multiplets,
    merge_user_peaks,
)
from asymmetry.core.fitting.process_pool import open_spawn_pool
from asymmetry.core.fitting.wizard_scope import (
    ScopeResolution,
    WizardScope,
    resolve_scope_for_datasets,
)

_ROLE_DELTA_THRESHOLD = 2.0
_COMPARABLE_SCORE_DELTA = 2.0
#: Safety margin (metric units) for the exact layer-truncation bound (technique
#: A). The bound halts enumeration of a template's remaining Hamming layers once
#: the all-local χ² floor plus the layer's minimum-possible penalty exceeds the
#: incumbent IC by more than this margin. It must dominate every downstream
#: verdict threshold so a pruned assignment can never have altered the winner or
#: its comparable tie-break: the winner selection tie-break is
#: ``_COMPARABLE_SCORE_DELTA`` (2.0) and the per-parameter role recommendations
#: use ``_role_delta_threshold`` (max ≈5.0 across all return paths). 6.0 clears
#: both with headroom — the error is asymmetric (too-high only forfeits a little
#: speedup; too-low silently prunes a verdict-relevant node), so we bias high.
_LAYER_BOUND_MARGIN = 6.0
_SHORTLIST_COUNT = 4
_SHORTLIST_SCORE_WINDOW = 6.0
_SHORTLIST_CAP = 6
_GLOBAL_FIT_MAX_CALLS = 1200
_GLOBAL_FIT_MAX_CALLS_CAP = 4200
_LOW_RESIDUAL_RMS_FOR_STRUCTURE_WARNINGS = 0.25
_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS = 1800
_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS_CAP = 5600
_HIGH_DIMENSION_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS = 9000
_HIGH_DIMENSION_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS_CAP = 9000
_STAGED_GLOBAL_FIT_MAX_CALLS = 4200
_STAGED_LOCAL_SEARCH_BEAM_WIDTH = 3
_STAGED_LOCAL_SEARCH_CANDIDATES_PER_BRANCH = 2
_STAGED_V2_LOCAL_SEARCH_BEAM_WIDTH = 5
_STAGED_V2_LOCAL_SEARCH_CANDIDATES_PER_BRANCH = 3
_STAGED_V2_EXACT_CANDIDATES_PER_TIER = 2
_STAGED_GLOBALIZATION_CANDIDATES_PER_STEP = 3
_CONSOLIDATED_SEARCH_VARIANT = "staged_v2"
_MAX_ROLE_CANDIDATES_PER_TIER = 3
_HIGH_DIMENSION_FREE_COUNT = 40
_EXTREME_DIMENSION_FREE_COUNT = 70
_MAX_TEMPLATE_WORKERS = 4
_MAX_WAVEFRONT_WORKERS = 12
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
    prescreen_only: bool = False
    assessment_key: str | None = None

    @property
    def selection_key(self) -> str:
        return self.assessment_key or self.template.key

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
        return (
            (not self.prescreen_only)
            and bool(self.fit_results_by_run)
            and all(result.success for result in self.fit_results_by_run.values())
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
        return self.assessment_for_key(self.recommended_key)

    def assessment_for_key(self, key: str | None) -> GlobalCandidateAssessment | None:
        if not isinstance(key, str):
            return None
        for assessment in self.assessments:
            if assessment.selection_key == key:
                return assessment

        template_matches = [
            assessment for assessment in self.assessments if assessment.template.key == key
        ]
        if not template_matches:
            return None
        if len(template_matches) == 1:
            return template_matches[0]

        optimized_matches = [
            assessment for assessment in template_matches if not assessment.prescreen_only
        ]
        if len(optimized_matches) == 1:
            return optimized_matches[0]
        if optimized_matches:
            return min(
                optimized_matches,
                key=lambda assessment: _assessment_sort_key(assessment, self.metric),
            )
        for assessment in self.assessments:
            if assessment.template.key == key:
                return assessment
        return None

    def assessments_for_template_key(
        self, template_key: str
    ) -> tuple[GlobalCandidateAssessment, ...]:
        return tuple(
            assessment for assessment in self.assessments if assessment.template.key == template_key
        )

    def sorted_assessments(
        self,
        metric: SelectionMetric | None = None,
    ) -> list[GlobalCandidateAssessment]:
        active_metric = metric or self.metric
        return sorted(
            self.assessments,
            key=lambda assessment: _assessment_sort_key(assessment, active_metric),
        )

    def sorted_prescreen_assessments(
        self,
        metric: SelectionMetric | None = None,
    ) -> list[GlobalCandidateAssessment]:
        active_metric = metric or self.metric
        return sorted(
            (assessment for assessment in self.assessments if assessment.prescreen_only),
            key=lambda assessment: _assessment_sort_key(assessment, active_metric),
        )

    def optimized_assessments(self) -> tuple[GlobalCandidateAssessment, ...]:
        return tuple(assessment for assessment in self.assessments if not assessment.prescreen_only)

    def sorted_optimized_assessments(
        self,
        metric: SelectionMetric | None = None,
    ) -> list[GlobalCandidateAssessment]:
        active_metric = metric or self.metric
        return sorted(
            self.optimized_assessments(),
            key=lambda assessment: _assessment_sort_key(assessment, active_metric),
        )

    def optimization_status_for_key(self, key: str | None) -> str:
        if not isinstance(key, str):
            return "Unknown"
        template_assessments = self.assessments_for_template_key(key)
        optimized = [
            assessment for assessment in template_assessments if not assessment.prescreen_only
        ]
        if not template_assessments:
            return "Unknown"
        if not optimized:
            return "Not optimized"
        if any(assessment.is_successful for assessment in optimized):
            return "Optimized"
        return "Optimization failed"


def _global_candidate_assessment_key(
    template_key: str,
    *,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    prescreen_only: bool = False,
) -> str:
    if prescreen_only:
        return template_key
    global_label = ",".join(global_param_names) or "none"
    local_label = ",".join(local_param_names) or "none"
    return f"{template_key}|g={global_label}|l={local_label}"


@dataclass(frozen=True)
class _WarmStartAssessment:
    fit_results_by_run: dict[int, FitResult]
    global_parameters: ParameterSet
    global_param_names: tuple[str, ...]
    local_param_names: tuple[str, ...]

    @property
    def is_successful(self) -> bool:
        return bool(self.fit_results_by_run) and all(
            result.success for result in self.fit_results_by_run.values()
        )


@dataclass(frozen=True)
class _WavefrontAssignmentTask:
    template_key: str
    template: CandidateTemplate
    datasets: list[MuonDataset]
    base_by_run: dict[int, ParameterSet]
    fixed_param_names: tuple[str, ...]
    global_param_names: tuple[str, ...]
    local_param_names: tuple[str, ...]
    axis_key: str
    metric: SelectionMetric
    search_strategy: str
    warm_start_source: _WarmStartAssessment | None = None
    initial_seed_by_run: dict[int, ParameterSet] | None = None


@dataclass(frozen=True)
class _WavefrontAssignmentResult:
    template_key: str
    global_param_names: tuple[str, ...]
    local_param_names: tuple[str, ...]
    assessment: GlobalCandidateAssessment
    instrumentation: dict[str, object]


@dataclass
class _WavefrontTemplateState:
    template: CandidateTemplate
    fixed_param_names: tuple[str, ...]
    prefit_base_by_run: dict[int, ParameterSet]
    free_param_names: tuple[str, ...]
    exact_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment]
    converged_assessments: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        GlobalCandidateAssessment,
    ]
    best_assessment: GlobalCandidateAssessment | None = None
    #: χ² of the converged all-local anchor (technique A). Every assignment for
    #: this template is nested inside all-local, so its χ² is a lower bound;
    #: ``None`` means the anchor did not converge and the bound is disabled.
    chi2_floor: float | None = None
    #: Best (lowest) IC metric value among this template's converged assignments
    #: so far, used as the layer-bound incumbent. ``inf`` until the first
    #: converged assignment (the anchor) lands.
    incumbent_ic: float = float("inf")
    #: Number of free (localisable) parameters — the maximum Hamming layer.
    free_param_count: int = 0
    #: True once the layer bound has fired and this template's remaining, higher
    #: layers are being skipped.
    layer_bound_fired: bool = False


def _compact_assessment_for_cache(
    assessment: GlobalCandidateAssessment,
) -> GlobalCandidateAssessment:
    return replace(
        assessment,
        fitted_curves_by_run={},
        component_curves_by_run={},
        parameter_recommendations=(),
    )


def _warm_start_source_from_assessment(
    assessment: GlobalCandidateAssessment | None,
) -> _WarmStartAssessment | None:
    if assessment is None:
        return None
    return _WarmStartAssessment(
        fit_results_by_run=assessment.fit_results_by_run,
        global_parameters=assessment.global_parameters,
        global_param_names=assessment.global_param_names,
        local_param_names=assessment.local_param_names,
    )


def _merge_instrumentation(
    instrumentation: dict[str, object] | None,
    delta: dict[str, object] | None,
) -> None:
    if instrumentation is None or not delta:
        return

    counters = delta.get("counters")
    if isinstance(counters, dict):
        target_counters = instrumentation.setdefault("counters", {})
        if isinstance(target_counters, dict):
            for name, value in counters.items():
                target_counters[name] = int(target_counters.get(name, 0)) + int(value)

    for name, value in delta.items():
        if name == "counters":
            continue
        if isinstance(value, list):
            target_values = instrumentation.setdefault(name, [])
            if isinstance(target_values, list):
                target_values.extend(value)


def _wavefront_worker_count(task_count: int) -> int:
    if task_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count, _MAX_WAVEFRONT_WORKERS))


def _single_fit_table_worker_count(task_count: int) -> int:
    if task_count <= 0:
        return 1
    return max(1, min(task_count, _MAX_TEMPLATE_WORKERS))


def _try_open_process_pool(
    *,
    max_workers: int,
    progress_callback: Callable[[str], None] | None = None,
    activity: str,
) -> ProcessPoolExecutor | None:
    executor = open_spawn_pool(max_workers)
    if executor is None:
        _progress_log(
            progress_callback,
            f"{activity}: spawn-safe workers unavailable in this environment; "
            "falling back to serial execution.",
        )
    return executor


def _shutdown_process_pool(executor: ProcessPoolExecutor) -> None:
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _layer_assignments(
    free_param_names: tuple[str, ...],
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    return tuple(
        tuple(tuple(names) for names in combinations(free_param_names, local_count))
        for local_count in range(len(free_param_names) + 1)
    )


def _all_global_seed_parameter_sets(
    base_by_run: dict[int, ParameterSet],
) -> dict[int, ParameterSet]:
    if not base_by_run:
        return {}

    averaged_values: dict[str, float] = {}
    collected_values: dict[str, list[float]] = {}
    for parameters in base_by_run.values():
        for parameter in parameters:
            if parameter.fixed:
                continue
            collected_values.setdefault(parameter.name, []).append(float(parameter.value))

    for name, values in collected_values.items():
        averaged_values[name] = float(np.mean(np.asarray(values, dtype=float)))

    seeded_by_run: dict[int, ParameterSet] = {}
    for run_number, parameters in base_by_run.items():
        cloned = _clone_parameter_set(parameters)
        for parameter in cloned:
            averaged_value = averaged_values.get(parameter.name)
            if averaged_value is None or parameter.fixed:
                continue
            parameter.value = float(np.clip(averaged_value, parameter.min, parameter.max))
        seeded_by_run[run_number] = cloned
    return seeded_by_run


def _best_predecessor_assessment(
    exact_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    *,
    free_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    metric: SelectionMetric,
) -> GlobalCandidateAssessment | None:
    predecessors: list[GlobalCandidateAssessment] = []
    for removed_name in local_param_names:
        predecessor_local = tuple(name for name in local_param_names if name != removed_name)
        predecessor_global = tuple(
            name for name in free_param_names if name not in predecessor_local
        )
        predecessor = exact_cache.get((predecessor_global, predecessor_local))
        if predecessor is not None and predecessor.is_successful:
            predecessors.append(predecessor)
    if not predecessors:
        return None
    return min(predecessors, key=lambda assessment: _assessment_sort_key(assessment, metric))


def _interleave_wavefront_tasks(
    task_groups: list[list[_WavefrontAssignmentTask]],
) -> list[_WavefrontAssignmentTask]:
    ordered_groups = sorted(task_groups, key=len, reverse=True)
    ordered_tasks: list[_WavefrontAssignmentTask] = []
    while ordered_groups:
        next_groups: list[list[_WavefrontAssignmentTask]] = []
        for group in ordered_groups:
            ordered_tasks.append(group[0])
            if len(group) > 1:
                next_groups.append(group[1:])
        ordered_groups = next_groups
    return ordered_tasks


def _run_wavefront_assignment_task(
    task: _WavefrontAssignmentTask,
) -> _WavefrontAssignmentResult:
    task_instrumentation: dict[str, object] = {
        "counters": {},
        "curvature_hint_sizes": [],
        "minuit_edm": [],
        "relaxed_penalties": [],
        "staged_frontier_widths": [],
    }
    fit_engine = FitEngine()
    warm_start_by_run: dict[int, ParameterSet] | None = None
    initial_step_sizes: dict[str, float] = {}

    if task.warm_start_source is not None and task.warm_start_source.is_successful:
        warm_start_by_run = _warm_start_parameter_sets(
            task.datasets,
            assessment=task.warm_start_source,
            base_by_run=task.base_by_run,
            target_global_names=task.global_param_names,
            target_local_names=task.local_param_names,
            fit_engine=fit_engine,
            template=task.template,
            progress_callback=None,
            cache=None,
        )
        initial_step_sizes = _step_hints_from_assessment(
            task.datasets,
            task.warm_start_source,
            target_global_names=task.global_param_names,
            target_local_names=task.local_param_names,
        )
    elif task.initial_seed_by_run is not None:
        warm_start_by_run = _clone_parameter_sets(task.initial_seed_by_run)

    assessment = _fit_exact_assignment(
        task.datasets,
        task.template,
        fit_engine=fit_engine,
        base_by_run=task.base_by_run,
        global_param_names=task.global_param_names,
        local_param_names=task.local_param_names,
        fixed_param_names=task.fixed_param_names,
        axis_key=task.axis_key,
        metric=task.metric,
        cache={},
        warm_start_by_run=warm_start_by_run,
        progress_callback=None,
        search_strategy=task.search_strategy,
        instrumentation=task_instrumentation,
        initial_step_sizes=initial_step_sizes,
    )
    return _WavefrontAssignmentResult(
        template_key=task.template_key,
        global_param_names=task.global_param_names,
        local_param_names=task.local_param_names,
        assessment=assessment,
        instrumentation=task_instrumentation,
    )


def _single_fit_recommendation_task(
    dataset: MuonDataset,
    templates: tuple[CandidateTemplate, ...],
    metric: SelectionMetric,
) -> tuple[int, FitWizardRecommendation]:
    run_number = int(dataset.run_number)
    recommendation = build_fit_wizard_recommendation_for_templates(
        dataset,
        templates,
        metric=metric,
    )
    return run_number, recommendation


@dataclass(frozen=True)
class GlobalFitWizardCandidatePortfolio:
    """Cheap pre-analysis portfolio detection for the global fit wizard."""

    ordered_datasets: tuple[MuonDataset, ...]
    series_axis_key: str
    series_axis_label: str
    mixed_axes_warning: str | None
    fingerprints_by_run: dict[int, SpectrumFingerprint]
    templates: tuple[CandidateTemplate, ...]
    #: Template keys of families supported by the cross-run multiplet pattern
    #: vote; the staged shortlist force-includes them.
    pattern_template_keys: tuple[str, ...] = ()

    @property
    def dataset_order(self) -> tuple[int, ...]:
        return tuple(int(dataset.run_number) for dataset in self.ordered_datasets)


def _series_multiplet_pattern_family_keys(
    ordered_datasets: Sequence[MuonDataset],
    user_frequencies_mhz: Sequence[float] | None = None,
) -> frozenset[str]:
    """Cross-run majority vote on multiplet pattern matches.

    Each run's detected (tail-subtracted) peak set is pattern-matched against
    the known physical multiplets; a candidate family is pattern-supported for
    the series when at least half of the runs (and no fewer than two) show a
    match naming it.
    """
    votes: dict[str, int] = {}
    for dataset in ordered_datasets:
        analysis = analyze_dataset_peaks(dataset)
        if user_frequencies_mhz:
            analysis = merge_user_peaks(analysis, tuple(user_frequencies_mhz))
        direction_text = str(
            dataset.metadata.get("field_direction") or dataset.metadata.get("field_state") or ""
        )
        geometry = geometry_from_field_direction(direction_text)
        matches = match_multiplets(
            analysis,
            field_gauss=dataset.field,
            geometry=geometry.value if geometry is not None else None,
        )
        for family_key in {match.family_key for match in matches}:
            votes[family_key] = votes.get(family_key, 0) + 1
    quorum = max(2, math.ceil(len(ordered_datasets) / 2))
    return frozenset(key for key, count in votes.items() if count >= quorum)


def _scoped_series_templates(
    ordered_datasets: Sequence[MuonDataset],
    aggregate_fingerprint: SpectrumFingerprint,
    current_model: CompositeModel | None,
    *,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
) -> tuple[tuple[CandidateTemplate, ...], tuple[str, ...]]:
    """Return the series candidate templates and pattern-forced template keys.

    With ``scope is None`` the legacy hint-gated portfolio is kept (plus the
    templates of any pattern-supported family, appended additively). With a
    scope, the portfolio is family-based: every in-scope family contributes its
    Stage-1 shapes; full member sets are included for families the aggregate
    fingerprint hints at, the cross-run pattern vote names, or the baseline.
    The returned keys mark pattern-supported families' templates so the staged
    shortlist can never drop them.
    """
    pattern_family_keys = _series_multiplet_pattern_family_keys(
        ordered_datasets, user_frequencies_mhz
    )

    resolution: ScopeResolution | None = None
    if scope is not None:
        resolution = resolve_scope_for_datasets(list(ordered_datasets), scope)
    families = build_wizard_families(
        aggregate_fingerprint, current_model, scope_resolution=resolution
    )

    def _family_templates(family: object, *, members: bool) -> list[CandidateTemplate]:
        chosen = [family.stage1_rep, *family.stage1_extras]
        if members:
            chosen.extend(family.stage2_members)
        return chosen

    pattern_template_keys: list[str] = []
    for family in families:
        if family.key in pattern_family_keys:
            pattern_template_keys.extend(
                template.key for template in _family_templates(family, members=True)
            )

    if scope is None:
        templates = list(
            build_candidate_templates(aggregate_fingerprint, current_model=current_model)
        )
        known_keys = {template.key for template in templates}
        for family in families:
            if family.key not in pattern_family_keys:
                continue
            for template in _family_templates(family, members=True):
                if template.key not in known_keys:
                    templates.append(template)
                    known_keys.add(template.key)
    else:
        templates = []
        known_keys = set()
        for family in families:
            expand = (
                family.priority > 0.0
                or family.key in pattern_family_keys
                or family.key == "baseline"
            )
            for template in _family_templates(family, members=expand):
                if template.key not in known_keys:
                    templates.append(template)
                    known_keys.add(template.key)

    forced = tuple(key for key in dict.fromkeys(pattern_template_keys) if key in known_keys)
    return tuple(templates), forced


def build_global_fit_wizard_candidate_portfolio(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
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
    templates, pattern_template_keys = _scoped_series_templates(
        ordered_datasets,
        aggregate_fingerprint,
        current_model,
        scope=scope,
        user_frequencies_mhz=user_frequencies_mhz,
    )
    return GlobalFitWizardCandidatePortfolio(
        ordered_datasets=tuple(ordered_datasets),
        series_axis_key=axis_key,
        series_axis_label=axis_label,
        mixed_axes_warning=mixed_axes_warning,
        fingerprints_by_run=fingerprints_by_run,
        templates=templates,
        pattern_template_keys=pattern_template_keys,
    )


def build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    existing_recommendations_by_run: dict[int, FitWizardRecommendation] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
) -> tuple[GlobalFitWizardCandidatePortfolio, dict[int, FitWizardRecommendation], tuple[int, ...]]:
    """Return a complete per-run single-fit table set for one global-wizard portfolio."""
    progress_callback = _threadsafe_progress_callback(progress_callback)
    portfolio = build_global_fit_wizard_candidate_portfolio(
        datasets,
        current_model=current_model,
        scope=scope,
        user_frequencies_mhz=user_frequencies_mhz,
    )
    expected_template_keys = candidate_template_keys(portfolio.templates)
    existing = (
        existing_recommendations_by_run if existing_recommendations_by_run is not None else {}
    )

    complete_by_run: dict[int, FitWizardRecommendation] = {}
    for dataset in portfolio.ordered_datasets:
        run_number = int(dataset.run_number)
        recommendation = existing.get(run_number)
        if recommendation is None:
            continue
        if recommendation_template_keys(recommendation) != expected_template_keys:
            continue
        complete_by_run[run_number] = recommendation

    if portfolio.mixed_axes_warning or not portfolio.templates:
        complete_by_run = _sync_single_fit_recommendation_store(
            existing_recommendations_by_run,
            complete_by_run,
        )
        return portfolio, complete_by_run, ()

    missing_datasets = [
        dataset
        for dataset in portfolio.ordered_datasets
        if int(dataset.run_number) not in complete_by_run
    ]
    if not missing_datasets:
        complete_by_run = _sync_single_fit_recommendation_store(
            existing_recommendations_by_run,
            complete_by_run,
        )
        return portfolio, complete_by_run, ()

    _progress_log(
        progress_callback,
        "Preparing per-dataset single-fit comparison tables for "
        f"{len(missing_datasets)} dataset(s) using the shared candidate portfolio.",
    )

    generated_run_numbers: list[int] = []

    worker_count = _single_fit_table_worker_count(len(missing_datasets))
    if worker_count <= 1:
        for dataset in missing_datasets:
            _progress_log(
                progress_callback,
                f"Single-fit table {dataset.run_label}: evaluating shared candidate portfolio.",
            )
            run_number, recommendation = _single_fit_recommendation_task(
                dataset,
                portfolio.templates,
                SelectionMetric.AICC,
            )
            complete_by_run[run_number] = recommendation
            generated_run_numbers.append(run_number)
    else:
        _progress_log(
            progress_callback,
            f"Running phase-1 single-fit table generation with {worker_count} spawn-safe workers.",
        )
        executor = _try_open_process_pool(
            max_workers=worker_count,
            progress_callback=progress_callback,
            activity="Phase-1 single-fit table generation",
        )
        if executor is None:
            for dataset in missing_datasets:
                _progress_log(
                    progress_callback,
                    f"Single-fit table {dataset.run_label}: evaluating shared candidate portfolio.",
                )
                run_number, recommendation = _single_fit_recommendation_task(
                    dataset,
                    portfolio.templates,
                    SelectionMetric.AICC,
                )
                complete_by_run[run_number] = recommendation
                generated_run_numbers.append(run_number)
        else:
            try:
                future_to_dataset = {}
                for dataset in missing_datasets:
                    _progress_log(
                        progress_callback,
                        f"Single-fit table {dataset.run_label}: evaluating shared candidate portfolio.",
                    )
                    future_to_dataset[
                        executor.submit(
                            _single_fit_recommendation_task,
                            dataset,
                            portfolio.templates,
                            SelectionMetric.AICC,
                        )
                    ] = dataset
                for future in as_completed(future_to_dataset):
                    run_number, recommendation = future.result()
                    complete_by_run[run_number] = recommendation
                    generated_run_numbers.append(run_number)
            finally:
                _shutdown_process_pool(executor)

    complete_by_run = _sync_single_fit_recommendation_store(
        existing_recommendations_by_run,
        complete_by_run,
    )
    return portfolio, complete_by_run, tuple(generated_run_numbers)


def build_global_fit_wizard_screening_recommendation(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    current_parameter_types: dict[str, str] | None = None,
    current_values: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    single_fit_recommendations_by_run: dict[int, FitWizardRecommendation] | None = None,
    metric: SelectionMetric = SelectionMetric.AICC,
    progress_callback: Callable[[str], None] | None = None,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
) -> GlobalFitWizardRecommendation:
    """Build the ranking table from per-run single-fit wizard results only."""
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    progress_callback = _threadsafe_progress_callback(progress_callback)
    current_parameter_types = current_parameter_types or {}
    current_values = current_values or {}
    parameter_bounds = parameter_bounds or {}

    portfolio = build_global_fit_wizard_candidate_portfolio(
        datasets,
        current_model=current_model,
        scope=scope,
        user_frequencies_mhz=user_frequencies_mhz,
    )
    templates = list(portfolio.templates)
    if portfolio.mixed_axes_warning:
        return rerank_global_fit_wizard_recommendation(
            GlobalFitWizardRecommendation(
                series_axis_key=portfolio.series_axis_key,
                series_axis_label=portfolio.series_axis_label,
                mixed_axes_warning=portfolio.mixed_axes_warning,
                fingerprints_by_run=portfolio.fingerprints_by_run,
                dataset_order=portfolio.dataset_order,
                templates=portfolio.templates,
                assessments=(),
                metric=metric,
                recommended_key=None,
                comparable_keys=(),
                summary=portfolio.mixed_axes_warning,
            ),
            metric,
        )

    recommendations_by_run = (
        single_fit_recommendations_by_run if single_fit_recommendations_by_run is not None else {}
    )
    expected_template_keys = candidate_template_keys(templates)
    if not recommendations_by_run or not all(
        int(dataset.run_number) in recommendations_by_run
        and recommendation_template_keys(recommendations_by_run[int(dataset.run_number)])
        == expected_template_keys
        for dataset in portfolio.ordered_datasets
    ):
        _progress_log(
            progress_callback,
            "Preparing missing single-fit wizard tables for global screening.",
        )
        _portfolio, recommendations_by_run, _generated_runs = (
            build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
                list(portfolio.ordered_datasets),
                current_model=current_model,
                existing_recommendations_by_run=recommendations_by_run,
                progress_callback=progress_callback,
                scope=scope,
                user_frequencies_mhz=user_frequencies_mhz,
            )
        )

    assessments_by_key, _template_contexts = _build_single_fit_prescreen_assessments(
        list(portfolio.ordered_datasets),
        portfolio.fingerprints_by_run,
        templates,
        single_fit_recommendations_by_run=recommendations_by_run,
        current_parameter_types=current_parameter_types,
        current_values=current_values,
        parameter_bounds=parameter_bounds,
        axis_key=portfolio.series_axis_key,
        metric=metric,
        fit_engine=FitEngine(),
        progress_callback=progress_callback,
    )
    return rerank_global_fit_wizard_recommendation(
        GlobalFitWizardRecommendation(
            series_axis_key=portfolio.series_axis_key,
            series_axis_label=portfolio.series_axis_label,
            mixed_axes_warning=portfolio.mixed_axes_warning,
            fingerprints_by_run=portfolio.fingerprints_by_run,
            dataset_order=portfolio.dataset_order,
            templates=portfolio.templates,
            assessments=tuple(assessments_by_key[template.key] for template in portfolio.templates),
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary="",
        ),
        metric,
    )


def _single_fit_assessment_by_run(
    recommendations_by_run: dict[int, FitWizardRecommendation],
    template_key: str,
) -> dict[int, CandidateAssessment]:
    assessments: dict[int, CandidateAssessment] = {}
    for run_number, recommendation in recommendations_by_run.items():
        assessment = recommendation.assessment_for_key(template_key)
        if assessment is not None:
            assessments[int(run_number)] = assessment
    return assessments


def _sync_single_fit_recommendation_store(
    existing_recommendations_by_run: dict[int, FitWizardRecommendation] | None,
    complete_by_run: dict[int, FitWizardRecommendation],
) -> dict[int, FitWizardRecommendation]:
    if existing_recommendations_by_run is None:
        return complete_by_run
    existing_recommendations_by_run.clear()
    existing_recommendations_by_run.update(complete_by_run)
    return existing_recommendations_by_run


def _merge_repaired_assessments_into_single_fit_recommendations(
    recommendations_by_run: dict[int, FitWizardRecommendation],
    template_key: str,
    repaired_assessments_by_run: dict[int, CandidateAssessment],
) -> None:
    for run_number, repaired_assessment in repaired_assessments_by_run.items():
        recommendation = recommendations_by_run.get(int(run_number))
        if recommendation is None:
            continue
        current_assessment = recommendation.assessment_for_key(template_key)
        if current_assessment is repaired_assessment:
            continue

        replaced = False
        updated_assessments: list[CandidateAssessment] = []
        for assessment in recommendation.assessments:
            if assessment.template.key == template_key:
                updated_assessments.append(repaired_assessment)
                replaced = True
            else:
                updated_assessments.append(assessment)
        if not replaced:
            continue

        recommendations_by_run[int(run_number)] = rerank_fit_wizard_recommendation(
            replace(
                recommendation,
                assessments=tuple(updated_assessments),
            ),
            recommendation.metric,
        )


def _build_single_fit_prescreen_assessments(
    datasets: list[MuonDataset],
    fingerprints_by_run: dict[int, SpectrumFingerprint],
    templates: list[CandidateTemplate],
    *,
    single_fit_recommendations_by_run: dict[int, FitWizardRecommendation],
    current_parameter_types: dict[str, str],
    current_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    axis_key: str,
    metric: SelectionMetric,
    fit_engine: FitEngine | None = None,
    progress_callback: Callable[[str], None] | None = None,
    repair_partial_incomplete: bool = True,
) -> tuple[
    dict[str, GlobalCandidateAssessment],
    dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
]:
    assessments_by_key: dict[str, GlobalCandidateAssessment] = {}
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]] = {}
    fit_engine = fit_engine or FitEngine()

    for template in templates:
        fixed_param_names = _fixed_param_names(template, current_parameter_types)
        seed_assessments_by_run = _single_fit_assessment_by_run(
            single_fit_recommendations_by_run,
            template.key,
        )
        if repair_partial_incomplete:
            seed_assessments_by_run = _repair_partial_single_fit_prescreen_assessments(
                datasets,
                fingerprints_by_run,
                template,
                assessments_by_run=seed_assessments_by_run,
                current_values=current_values,
                parameter_bounds=parameter_bounds,
                fixed_param_names=fixed_param_names,
                metric=metric,
                fit_engine=fit_engine,
                progress_callback=progress_callback,
            )
            _merge_repaired_assessments_into_single_fit_recommendations(
                single_fit_recommendations_by_run,
                template.key,
                seed_assessments_by_run,
            )
        base_by_run = _initial_parameter_sets_for_candidate(
            datasets,
            fingerprints_by_run,
            template,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            fixed_param_names=fixed_param_names,
            seed_assessments_by_run=seed_assessments_by_run,
        )
        template_contexts[template.key] = (base_by_run, fixed_param_names)
        global_param_names, local_param_names = _initial_parameter_roles(
            template,
            current_parameter_types=current_parameter_types,
            fixed_param_names=fixed_param_names,
        )

        fit_results_by_run: dict[int, FitResult] = {}
        fitted_curves_by_run: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        component_curves_by_run: dict[int, tuple[tuple[str, NDArray[np.float64]], ...]] = {}
        run_diagnostics: list[RunResidualDiagnostic] = []
        aic_total = 0.0
        bic_total = 0.0
        aicc_total = 0.0
        all_have_aicc = True
        missing_runs: list[str] = []

        for dataset in datasets:
            run_number = int(dataset.run_number)
            axis_value = _axis_value(dataset, axis_key)
            assessment = seed_assessments_by_run.get(run_number)
            if assessment is None:
                missing_runs.append(dataset.run_label)
                run_diagnostics.append(
                    RunResidualDiagnostic(
                        run_number=run_number,
                        run_label=dataset.run_label,
                        axis_value=axis_value,
                        residual_rms=float("inf"),
                        runs_z_score=float("inf"),
                        max_abs_autocorrelation=float("inf"),
                        residual_fft_peak_snr=float("inf"),
                        gate_passed=False,
                        gate_reasons=(
                            f"missing successful single-fit assessment for {template.title}",
                        ),
                    )
                )
                continue

            fit_results_by_run[run_number] = assessment.fit_result
            fitted_curves_by_run[run_number] = (
                np.asarray(assessment.fitted_time, dtype=float).copy(),
                np.asarray(assessment.fitted_curve, dtype=float).copy(),
            )
            component_curves_by_run[run_number] = tuple(
                (
                    name,
                    np.asarray(values, dtype=float).copy(),
                )
                for name, values in assessment.component_curves
            )
            run_diagnostics.append(
                RunResidualDiagnostic(
                    run_number=run_number,
                    run_label=dataset.run_label,
                    axis_value=axis_value,
                    residual_rms=assessment.residual_rms,
                    runs_z_score=assessment.runs_z_score,
                    max_abs_autocorrelation=assessment.max_abs_autocorrelation,
                    residual_fft_peak_snr=assessment.residual_fft_peak_snr,
                    gate_passed=assessment.residual_gate_passed,
                    gate_reasons=tuple(assessment.residual_gate_reasons),
                )
            )

            if assessment.is_successful:
                aic_total += float(assessment.aic)
                bic_total += float(assessment.bic)
                if assessment.aicc is None:
                    all_have_aicc = False
                else:
                    aicc_total += float(assessment.aicc)
            else:
                missing_runs.append(dataset.run_label)

        complete = not missing_runs and len(fit_results_by_run) == len(datasets)
        if complete:
            aic = float(aic_total)
            aicc = float(aicc_total) if all_have_aicc else None
            bic = float(bic_total)
            selected_score = _metric_value(metric, aic, aicc, bic)
            series_warnings = (
                "Independent single-fit pre-screen only. This candidate was not advanced to coupled global optimisation.",
            )
        else:
            aic = float("inf")
            aicc = None
            bic = float("inf")
            selected_score = float("inf")
            series_warnings = tuple(
                [
                    "Single-fit pre-screen incomplete. This candidate is excluded from the global shortlist.",
                    *[
                        f"Missing or failed single-fit assessment for run {run_label}."
                        for run_label in missing_runs
                    ],
                ]
            )

        assessments_by_key[template.key] = GlobalCandidateAssessment(
            template=template,
            fit_results_by_run=fit_results_by_run,
            global_parameters=ParameterSet(),
            global_param_names=tuple(global_param_names),
            local_param_names=tuple(local_param_names),
            fixed_param_names=tuple(fixed_param_names),
            parameter_recommendations=(),
            run_diagnostics=tuple(run_diagnostics),
            series_warnings=series_warnings,
            aic=aic,
            aicc=aicc,
            bic=bic,
            selected_score=selected_score,
            fitted_curves_by_run=fitted_curves_by_run,
            component_curves_by_run=component_curves_by_run,
            prescreen_only=True,
        )

    return assessments_by_key, template_contexts


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


def build_global_fit_wizard_recommendation(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    current_parameter_types: dict[str, str] | None = None,
    current_values: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    single_fit_recommendations_by_run: dict[int, FitWizardRecommendation] | None = None,
    metric: SelectionMetric = SelectionMetric.AICC,
    progress_callback: Callable[[str], None] | None = None,
    instrumentation: dict[str, object] | None = None,
    selected_template_keys: tuple[str, ...] | None = None,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
) -> GlobalFitWizardRecommendation:
    """Analyze one ordered dataset series and recommend a global-fit candidate."""
    _set_metric(instrumentation, "strategy", "consolidated")
    if instrumentation is not None:
        instrumentation.setdefault("counters", {})
        instrumentation.setdefault("staged_frontier_widths", [])
        instrumentation.setdefault("relaxed_penalties", [])
        instrumentation.setdefault("curvature_hint_sizes", [])
        instrumentation.setdefault("minuit_edm", [])
    return _build_global_fit_wizard_recommendation_staged(
        datasets,
        current_model=current_model,
        current_parameter_types=current_parameter_types,
        current_values=current_values,
        parameter_bounds=parameter_bounds,
        single_fit_recommendations_by_run=single_fit_recommendations_by_run,
        metric=metric,
        progress_callback=progress_callback,
        instrumentation=instrumentation,
        selected_template_keys=selected_template_keys,
        scope=scope,
        user_frequencies_mhz=user_frequencies_mhz,
    )


def _build_global_fit_wizard_recommendation_staged(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    current_parameter_types: dict[str, str] | None = None,
    current_values: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    single_fit_recommendations_by_run: dict[int, FitWizardRecommendation] | None = None,
    metric: SelectionMetric = SelectionMetric.AICC,
    progress_callback: Callable[[str], None] | None = None,
    instrumentation: dict[str, object] | None = None,
    selected_template_keys: tuple[str, ...] | None = None,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
) -> GlobalFitWizardRecommendation:
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    search_strategy = _CONSOLIDATED_SEARCH_VARIANT
    progress_callback = _threadsafe_progress_callback(progress_callback)
    current_parameter_types = current_parameter_types or {}
    current_values = current_values or {}
    parameter_bounds = parameter_bounds or {}
    available_single_fit_recommendations = (
        single_fit_recommendations_by_run if single_fit_recommendations_by_run is not None else {}
    )

    _progress_log(
        progress_callback,
        f"Preparing consolidated global fit wizard analysis for {len(datasets)} datasets.",
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
    scoped_templates, pattern_template_keys = _scoped_series_templates(
        ordered_datasets,
        aggregate_fingerprint,
        current_model,
        scope=scope,
        user_frequencies_mhz=user_frequencies_mhz,
    )
    templates = list(scoped_templates)
    template_by_key = {template.key: template for template in templates}
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
    single_run_prefit_caches: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...], tuple[bool, ...]],
        dict[
            tuple[tuple[str, ...], tuple[tuple[int, tuple[tuple[str, float], ...]], ...]],
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

    normalized_selected_template_keys = tuple(
        key for key in (selected_template_keys or ()) if key in template_by_key
    )
    prescreen_templates = (
        tuple(template_by_key[key] for key in normalized_selected_template_keys)
        if normalized_selected_template_keys
        else templates
    )

    expected_single_fit_template_keys = candidate_template_keys(templates)
    use_single_fit_prescreen = (
        bool(available_single_fit_recommendations)
        and all(
            recommendation_template_keys(
                available_single_fit_recommendations.get(int(dataset.run_number))
            )
            == expected_single_fit_template_keys
            for dataset in ordered_datasets
            if int(dataset.run_number) in available_single_fit_recommendations
        )
        and all(
            int(dataset.run_number) in available_single_fit_recommendations
            for dataset in ordered_datasets
        )
    )

    if use_single_fit_prescreen:
        _progress_log(
            progress_callback,
            "Using completed per-run single-fit wizard tables for aggregated candidate pre-screening.",
        )
        initial_assessments, template_contexts = _build_single_fit_prescreen_assessments(
            ordered_datasets,
            fingerprints_by_run,
            prescreen_templates,
            single_fit_recommendations_by_run=available_single_fit_recommendations,
            current_parameter_types=current_parameter_types,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            axis_key=axis_key,
            metric=metric,
            fit_engine=FitEngine(),
            progress_callback=progress_callback,
            repair_partial_incomplete=not normalized_selected_template_keys,
        )
    else:
        template_workers = _template_worker_count(len(prescreen_templates))
        if template_workers <= 1:
            for index, template in enumerate(prescreen_templates, start=1):
                _progress_log(
                    progress_callback,
                    f"Initial screening {index}/{len(prescreen_templates)}: {template.title}.",
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
                for index, template in enumerate(prescreen_templates, start=1):
                    _progress_log(
                        progress_callback,
                        f"Initial screening {index}/{len(prescreen_templates)}: {template.title}.",
                    )
                    future_to_template[executor.submit(_initial_screen_task, template)] = template
                for future in as_completed(future_to_template):
                    key, base_by_run, fixed_param_names, assessment = future.result()
                    template_contexts[key] = (base_by_run, fixed_param_names)
                    initial_assessments[key] = assessment
    if normalized_selected_template_keys:
        shortlist_keys = set(normalized_selected_template_keys)
        _progress_log(
            progress_callback,
            "Running coupled global optimisation for the selected candidates: "
            + ", ".join(template.title for template in templates if template.key in shortlist_keys)
            + ".",
        )
    else:
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
            forced_keys=tuple(dict.fromkeys((*forced_shortlist_keys, *pattern_template_keys))),
        )
    shortlisted_templates = [template for template in templates if template.key in shortlist_keys]
    if shortlisted_templates:
        _progress_log(
            progress_callback,
            "Coupled global optimisation will evaluate "
            f"{len(shortlisted_templates)} candidate(s) "
            "via exhaustive global/local enumeration.",
        )
    optimized_assessments = _run_exhaustive_wavefront_search(
        ordered_datasets,
        shortlisted_templates=shortlisted_templates,
        template_contexts=template_contexts,
        axis_key=axis_key,
        metric=metric,
        progress_callback=progress_callback,
        search_strategy=search_strategy,
        instrumentation=instrumentation,
        single_run_prefit_cache_for=_single_run_prefit_cache_for,
    )

    prescreen_assessments = tuple(
        initial_assessments[template.key]
        for template in prescreen_templates
        if template.key in initial_assessments
    )

    return rerank_global_fit_wizard_recommendation(
        GlobalFitWizardRecommendation(
            series_axis_key=axis_key,
            series_axis_label=axis_label,
            mixed_axes_warning=mixed_axes_warning,
            fingerprints_by_run=fingerprints_by_run,
            dataset_order=tuple(int(dataset.run_number) for dataset in ordered_datasets),
            templates=tuple(templates),
            assessments=prescreen_assessments + optimized_assessments,
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary="",
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
        optimized_assessments = recommendation.optimized_assessments()
        if not optimized_assessments:
            return replace(
                recommendation,
                metric=metric,
                recommended_key=None,
                comparable_keys=(),
                summary=(
                    "Single-fit screening complete. These scores come from independent "
                    "per-dataset fits only and have not yet been optimized for coupled "
                    "global fitting. Select one or more candidates to continue."
                ),
            )
        return replace(
            recommendation,
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary=(
                "No globally optimized candidate passed the automatic residual and "
                "continuity checks. Inspect the optimized-results table before applying a model."
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
            alternate = primary if preferred.selection_key != primary.selection_key else runner_up
            primary = preferred
            comparable_keys = (preferred.selection_key, alternate.selection_key)

    compare_summary = (
        ", with a similarly scoring alternative to inspect." if comparable_keys else "."
    )
    return replace(
        recommendation,
        metric=metric,
        recommended_key=primary.selection_key,
        comparable_keys=comparable_keys,
        summary=(
            f"Recommended globally optimized candidate: {primary.template.title} "
            f"by {metric.value}{compare_summary}"
        ),
    )


def merge_global_fit_wizard_recommendations(
    base: GlobalFitWizardRecommendation,
    updates: GlobalFitWizardRecommendation,
) -> GlobalFitWizardRecommendation:
    """Merge optimized assessments from one run back into an existing workflow snapshot."""
    updated_template_keys = {
        assessment.template.key
        for assessment in updates.assessments
        if not assessment.prescreen_only
    }
    merged_assessments = [
        assessment
        for assessment in base.assessments
        if assessment.prescreen_only or assessment.template.key not in updated_template_keys
    ]
    merged_assessments.extend(
        assessment for assessment in updates.assessments if not assessment.prescreen_only
    )
    merged = replace(
        base,
        metric=updates.metric,
        assessments=tuple(merged_assessments),
    )
    return rerank_global_fit_wizard_recommendation(merged, updates.metric)


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

    passing = [assessment for assessment in recommendation.assessments if assessment.is_successful]
    passing.sort(key=lambda assessment: _assessment_sort_key(assessment, recommendation.metric))
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
            _serialize_candidate_template(template) for template in recommendation.templates
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
            str(payload["recommended_key"]) if payload.get("recommended_key") is not None else None
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
        "prescreen_only": bool(assessment.prescreen_only),
        "assessment_key": assessment.assessment_key,
    }


def _migrate_global_param_name_tuple(
    names: tuple[str, ...], model: CompositeModel
) -> tuple[str, ...]:
    """Rename/drop legacy ``fraction_<k>`` entries in a cached parameter-role tuple.

    Applies the same rename map as
    :func:`asymmetry.core.fitting.composite.migrate_legacy_fraction_parameter_set`
    to a ``global_param_names``/``local_param_names``/``fixed_param_names`` tuple,
    preserving order and dropping names that map to ``None`` (the derived last
    term of a fraction group, which never has a free-parameter name of its own).
    """
    rename = _legacy_fraction_rename_map(model)
    if not rename:
        return names
    migrated: list[str] = []
    for name in names:
        if name in rename:
            new_name = rename[name]
            if new_name is not None and new_name not in migrated:
                migrated.append(new_name)
        elif name not in migrated:
            migrated.append(name)
    return tuple(migrated)


def _deserialize_global_candidate_assessment(
    payload: object,
) -> GlobalCandidateAssessment | None:
    if not isinstance(payload, dict):
        return None
    template = _deserialize_candidate_template(payload.get("template"))
    if template is None:
        return None

    # A recommendation cached before the fraction rework carries legacy
    # ``fraction_<k>`` names/values across the per-run fit results, the
    # standalone global-parameter set, and the three parameter-role tuples.
    # Migrate all of them against the template's model (mirrors
    # fit_wizard._deserialize_candidate_assessment's single-fit treatment).
    fit_results_by_run = {
        int(run_number): _migrate_fit_result_fractions(result, template.model)
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

    global_parameters = migrate_legacy_fraction_parameter_set(
        template.model, _deserialize_parameter_set(payload.get("global_parameters", []))
    )
    global_param_names = _migrate_global_param_name_tuple(
        tuple(name for name in payload.get("global_param_names", []) if isinstance(name, str)),
        template.model,
    )
    local_param_names = _migrate_global_param_name_tuple(
        tuple(name for name in payload.get("local_param_names", []) if isinstance(name, str)),
        template.model,
    )
    fixed_param_names = _migrate_global_param_name_tuple(
        tuple(name for name in payload.get("fixed_param_names", []) if isinstance(name, str)),
        template.model,
    )

    try:
        return GlobalCandidateAssessment(
            template=template,
            fit_results_by_run=fit_results_by_run,
            global_parameters=global_parameters,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=fixed_param_names,
            parameter_recommendations=parameter_recommendations,
            run_diagnostics=run_diagnostics,
            series_warnings=tuple(
                warning
                for warning in payload.get("series_warnings", [])
                if isinstance(warning, str)
            ),
            aic=float(payload.get("aic", float("inf"))),
            aicc=(float(payload["aicc"]) if payload.get("aicc") is not None else None),
            bic=float(payload.get("bic", float("inf"))),
            selected_score=float(payload.get("selected_score", float("inf"))),
            fitted_curves_by_run=fitted_curves_by_run,
            component_curves_by_run=component_curves_by_run,
            prescreen_only=bool(payload.get("prescreen_only", False)),
            assessment_key=(
                str(payload["assessment_key"])
                if payload.get("assessment_key") is not None
                else None
            ),
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
    ]
    | None = None,
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
    ]
    | None = None,
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
    ]
    | None = None,
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
        for (
            _probe_score,
            _name,
            candidate_global_names,
            candidate_local_names,
            warm_start_by_run,
        ) in candidate_specs[:exact_limit]:
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
    ]
    | None = None,
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
    beam_width, branch_limit, use_screening, exact_candidates_per_tier = (
        _staged_local_search_settings(search_strategy)
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
                name
                for name in ordered_target_local_names
                if name not in incumbent.local_param_names
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


def _staged_globalization_assignment(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    progress_callback: Callable[[str], None] | None = None,
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
    ]
    | None = None,
) -> GlobalCandidateAssessment | None:
    promotable_names = tuple(
        name for name in template.model.param_names if name not in fixed_param_names
    )
    if not promotable_names:
        return None

    _progress_log(
        progress_callback,
        f"{template.title}: starting direct staged globalization from all-local prefits.",
    )
    incumbent = _fit_exact_assignment(
        datasets,
        template,
        fit_engine=fit_engine,
        base_by_run=base_by_run,
        global_param_names=(),
        local_param_names=promotable_names,
        fixed_param_names=fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        cache=cache,
        warm_start_by_run=base_by_run,
        progress_callback=progress_callback,
        search_strategy="staged_v2",
        instrumentation=instrumentation,
    )
    if not incumbent.is_successful:
        _progress_log(
            progress_callback,
            f"{template.title}: all-local globalization baseline failed.",
        )
        return None

    while incumbent.local_param_names:
        ranked_names = _globalization_candidate_order(
            datasets,
            incumbent,
            remaining=incumbent.local_param_names,
        )
        if not ranked_names:
            break

        stage_names = ranked_names[:_STAGED_GLOBALIZATION_CANDIDATES_PER_STEP]
        best_candidate: GlobalCandidateAssessment | None = None
        for name in stage_names:
            candidate_local_names = tuple(
                sorted(
                    local_name for local_name in incumbent.local_param_names if local_name != name
                )
            )
            candidate_global_names = tuple(
                param_name
                for param_name in template.model.param_names
                if param_name not in fixed_param_names and param_name not in candidate_local_names
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
                    assessment=incumbent,
                    base_by_run=base_by_run,
                    target_global_names=candidate_global_names,
                    target_local_names=candidate_local_names,
                    fit_engine=fit_engine,
                    template=template,
                    progress_callback=progress_callback,
                    cache=warm_start_cache,
                ),
                progress_callback=progress_callback,
                search_strategy="staged_v2",
                instrumentation=instrumentation,
                initial_step_sizes=_step_hints_from_assessment(
                    datasets,
                    incumbent,
                    target_global_names=candidate_global_names,
                    target_local_names=candidate_local_names,
                ),
            )
            if not candidate.is_successful:
                continue
            if best_candidate is None or _assessment_sort_key(
                candidate, metric
            ) < _assessment_sort_key(
                best_candidate,
                metric,
            ):
                best_candidate = candidate

        if best_candidate is None or not _prefer_globalization_change(
            best_candidate,
            incumbent,
            metric=metric,
        ):
            break

        promoted_names = sorted(
            name
            for name in best_candidate.global_param_names
            if name not in incumbent.global_param_names
        )
        if promoted_names:
            _progress_log(
                progress_callback,
                f"{template.title}: promoted {', '.join(promoted_names)} to Global; "
                f"{metric.value} improved to {best_candidate.metric_value(metric):.3f}.",
            )
        incumbent = best_candidate

    return incumbent


def _globalization_candidate_order(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment,
    *,
    remaining: tuple[str, ...],
) -> tuple[str, ...]:
    scored_names: list[tuple[float, float, float, float, str]] = []
    for name in remaining:
        total_variation, roughness = _parameter_trace_roughness(
            datasets,
            assessment,
            name,
        )
        effective_variation = (total_variation + roughness) / max(
            localisation_threshold_scale(name),
            1e-9,
        )
        scored_names.append(
            (
                effective_variation,
                total_variation + roughness,
                -float(_parameter_localisation_priority(name)),
                -localisation_threshold_scale(name),
                name,
            )
        )
    scored_names.sort()
    return tuple(name for *_unused, name in scored_names)


def _prefer_globalization_change(
    candidate: GlobalCandidateAssessment,
    incumbent: GlobalCandidateAssessment,
    *,
    metric: SelectionMetric,
) -> bool:
    if not candidate.is_successful:
        return False
    if incumbent.residual_gate_passed and not candidate.residual_gate_passed:
        return False
    score_delta = incumbent.metric_value(metric) - candidate.metric_value(metric)
    if score_delta > 1e-6:
        return True
    if (
        not incumbent.residual_gate_passed
        and candidate.residual_gate_passed
        and score_delta >= -1e-6
    ):
        return True
    return False


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
    if cached is not None and cached.is_successful:
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
    difficult_assignment = free_count >= 20 or len(local_param_names) >= 2

    def _evaluate_attempt_variants(
        variants: tuple[dict[int, ParameterSet], ...],
        *,
        initial_hints: dict[str, float],
    ) -> tuple[
        dict[int, FitResult] | None,
        ParameterSet,
        float,
        str,
        dict[str, float],
    ]:
        local_best_results: dict[int, FitResult] | None = None
        local_best_global = ParameterSet()
        local_best_score = float("inf")
        local_best_failure_message = "No fit attempts were created."
        local_step_hints = dict(initial_hints)

        for variant_index, initial_params in enumerate(variants, start=1):
            _progress_log(
                progress_callback,
                f"{template.title}: trying initial parameter variant "
                f"{variant_index}/{len(variants)}.",
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
            _record_counter(instrumentation, "global_fit_calls")
            if local_step_hints:
                _record_counter(instrumentation, "curvature_hint_applications")
                _append_metric(instrumentation, "curvature_hint_sizes", len(local_step_hints))
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
                initial_step_sizes=local_step_hints or None,
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
                if total_chi2 < local_best_score:
                    local_best_score = total_chi2
                    local_best_results = results_by_run
                    local_best_global = fitted_global
                    local_step_hints = _step_hints_from_fit_results(
                        datasets,
                        results_by_run,
                        target_global_names=global_param_names,
                        target_local_names=local_param_names,
                    )
                    continue
            if local_best_results is None:
                local_best_results = results_by_run
                local_best_global = fitted_global
            failure_message = _assignment_failure_message(results_by_run)
            if failure_message:
                local_best_failure_message = failure_message

        return (
            local_best_results,
            local_best_global,
            local_best_score,
            local_best_failure_message,
            local_step_hints,
        )

    best_results, best_global, best_score, best_failure_message, step_hints = (
        _evaluate_attempt_variants(
            attempt_variants,
            initial_hints=dict(initial_step_sizes or {}),
        )
    )

    fallback_attempt_variants: tuple[dict[int, ParameterSet], ...] = ()
    fit_success = best_results is not None and all(
        result.success for result in best_results.values()
    )
    if not fit_success and warm_start_by_run is not None:
        fallback_attempt_variants = _assignment_attempt_variants(
            base_by_run,
            template,
            warm_start_by_run=None,
        )
        fallback_attempt_variants = _trim_assignment_attempt_variants(
            fallback_attempt_variants,
            free_count=free_count,
        )
        fallback_attempt_variants = tuple(
            _canonicalize_parameter_sets(
                attempt,
                template=template,
                global_param_names=global_param_names,
                local_param_names=local_param_names,
                fixed_param_names=fixed_param_names,
            )
            for attempt in fallback_attempt_variants
        )
        _progress_log(
            progress_callback,
            f"{template.title}: retrying assignment from prefit-only seeds.",
        )
        (
            fallback_results,
            fallback_global,
            fallback_score,
            fallback_failure_message,
            fallback_step_hints,
        ) = _evaluate_attempt_variants(
            fallback_attempt_variants,
            initial_hints={},
        )
        fallback_success = fallback_results is not None and all(
            result.success for result in fallback_results.values()
        )
        if fallback_success and (not fit_success or fallback_score < best_score):
            best_results = fallback_results
            best_global = fallback_global
            best_score = fallback_score
            best_failure_message = fallback_failure_message
            step_hints = fallback_step_hints
            fit_success = True
        elif best_results is None and fallback_results is not None:
            best_results = fallback_results
            best_global = fallback_global
            best_failure_message = fallback_failure_message

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
        rescue_step_hints = {} if difficult_assignment else dict(step_hints)
        rescue_params = (
            _clone_parameter_sets(warm_start_by_run)
            if warm_start_by_run is not None
            else _clone_parameter_sets(fallback_attempt_variants[0])
            if fallback_attempt_variants
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
            max_cycles=4 if difficult_assignment else 2,
            include_mixed_polish=difficult_assignment,
            instrumentation=instrumentation,
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
        if rescue_step_hints:
            _record_counter(instrumentation, "curvature_hint_applications")
            _append_metric(instrumentation, "curvature_hint_sizes", len(rescue_step_hints))
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
            initial_step_sizes=rescue_step_hints or None,
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
    if assessment.is_successful:
        cache[cache_key] = assessment
    return assessment


def _build_parameter_recommendations_from_exact_cache(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment,
    *,
    template: CandidateTemplate,
    fixed_param_names: tuple[str, ...],
    metric: SelectionMetric,
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment],
    names_to_test: set[str] | None = None,
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
        current_role = "Local" if name in current_local else "Global"
        if names_to_test is not None and name not in names_to_test:
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
                        f"Wavefront exhaustive search kept {name} {current_role}; "
                        "no stronger alternative assignment improved the penalized score."
                    ),
                )
            )
            continue

        if name in current_local:
            alternative_local_names = tuple(sorted(current_local - {name}))
            alternative_global_names = tuple(
                pname
                for pname in template.model.param_names
                if pname not in fixed_names and pname not in alternative_local_names
            )
            alternative = cache.get((alternative_global_names, alternative_local_names))
            local_score = current_score
            global_score = (
                float(alternative.metric_value(metric))
                if alternative is not None and alternative.is_successful
                else float("inf")
            )
            improvement = global_score - local_score
            keep_local = (
                alternative is None
                or not alternative.is_successful
                or improvement > _ROLE_DELTA_THRESHOLD
                or (assessment.residual_gate_passed and not alternative.residual_gate_passed)
            )
            recommended_role = "Local" if keep_local else "Global"
            if alternative is None:
                rationale = (
                    f"The exhaustive wavefront cache does not contain a successful shared-{name} "
                    "alternative, so the local role is retained."
                )
            elif not alternative.is_successful:
                rationale = (
                    f"The exhaustive search tried sharing {name}, but that assignment did not converge "
                    "successfully across the full series."
                )
            else:
                rationale = (
                    f"Keeping {name} Local improves the penalized score by {improvement:.2f}."
                    if keep_local and np.isfinite(improvement)
                    else f"The exhaustive search found that {name} is only weakly supported as Local."
                )
            delta = improvement
        else:
            alternative_local_names = tuple(sorted((*current_local, name)))
            alternative_global_names = tuple(
                pname
                for pname in template.model.param_names
                if pname not in fixed_names and pname not in alternative_local_names
            )
            alternative = cache.get((alternative_global_names, alternative_local_names))
            global_score = current_score
            local_score = (
                float(alternative.metric_value(metric))
                if alternative is not None and alternative.is_successful
                else float("inf")
            )
            improvement = global_score - local_score
            make_local = (
                alternative is not None
                and alternative.is_successful
                and improvement > _ROLE_DELTA_THRESHOLD
                and (alternative.residual_gate_passed or not assessment.residual_gate_passed)
            )
            recommended_role = "Local" if make_local else "Global"
            if alternative is None:
                rationale = (
                    f"The exhaustive wavefront cache does not contain a successful localized-{name} "
                    "alternative, so the global role is retained."
                )
            elif not alternative.is_successful:
                rationale = (
                    f"The exhaustive search tried localizing {name}, but that assignment did not converge "
                    "successfully across the full series."
                )
            else:
                rationale = (
                    f"Localizing {name} improves the penalized score by {improvement:.2f}."
                    if make_local and np.isfinite(improvement)
                    else f"The exhaustive search found that localizing {name} does not overcome the complexity penalty."
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
    seed_assessments_by_run: dict[int, CandidateAssessment] | None = None,
) -> dict[int, ParameterSet]:
    base_by_run: dict[int, ParameterSet] = {}
    seeded_by_run = seed_assessments_by_run or {}
    for dataset in datasets:
        run_number = int(dataset.run_number)
        seeded_assessment = seeded_by_run.get(run_number)
        seeded_values = None
        if seeded_assessment is not None and seeded_assessment.fit_result.success:
            seeded_values = {
                parameter.name: float(parameter.value)
                for parameter in seeded_assessment.fit_result.parameters
            }
        parameters = _configured_single_fit_parameter_set(
            dataset,
            fingerprints_by_run[run_number],
            template,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            fixed_param_names=fixed_param_names,
            seeded_values=seeded_values,
        )
        base_by_run[run_number] = parameters
    return base_by_run


def _configured_single_fit_parameter_set(
    dataset: MuonDataset,
    fingerprint: SpectrumFingerprint,
    template: CandidateTemplate,
    *,
    current_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    fixed_param_names: tuple[str, ...],
    seeded_values: dict[str, float] | None = None,
    seeded_values_override_current: bool = False,
) -> ParameterSet:
    parameters = _initial_parameters_for_template(dataset, fingerprint, template)
    seeded_values = dict(seeded_values or {})
    for parameter in parameters:
        if not seeded_values_override_current and parameter.name in seeded_values:
            parameter.value = seeded_values[parameter.name]
        if parameter.name in current_values:
            try:
                parameter.value = float(current_values[parameter.name])
            except (TypeError, ValueError):
                pass
        if (
            seeded_values_override_current
            and parameter.name in seeded_values
            and parameter.name not in fixed_param_names
        ):
            parameter.value = seeded_values[parameter.name]
        if parameter.name in parameter_bounds:
            min_val, max_val = parameter_bounds[parameter.name]
            parameter.min = float(min_val)
            parameter.max = float(max_val)
        parameter.value = float(np.clip(parameter.value, parameter.min, parameter.max))
        if parameter.name in fixed_param_names:
            parameter.fixed = True
    return parameters


def _repair_partial_single_fit_prescreen_assessments(
    datasets: list[MuonDataset],
    fingerprints_by_run: dict[int, SpectrumFingerprint],
    template: CandidateTemplate,
    *,
    assessments_by_run: dict[int, CandidateAssessment],
    current_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    fixed_param_names: tuple[str, ...],
    metric: SelectionMetric,
    fit_engine: FitEngine,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[int, CandidateAssessment]:
    repaired_assessments = dict(assessments_by_run)
    run_order = [int(dataset.run_number) for dataset in datasets]
    successful_runs = [
        run_number
        for run_number in run_order
        if repaired_assessments.get(run_number) is not None
        and repaired_assessments[run_number].is_successful
    ]
    failed_runs = [
        run_number
        for run_number in run_order
        if repaired_assessments.get(run_number) is None
        or (
            not repaired_assessments[run_number].is_successful
            and not repaired_assessments[run_number].repair_attempted
        )
    ]
    if not successful_runs or not failed_runs:
        return repaired_assessments

    _progress_log(
        progress_callback,
        f"{template.title}: repairing partial single-fit screening results "
        f"for {len(failed_runs)} dataset(s) using sibling fit seeds.",
    )

    order_index = {run_number: index for index, run_number in enumerate(run_order)}
    dataset_by_run = {int(dataset.run_number): dataset for dataset in datasets}

    while True:
        successful_runs = [
            run_number
            for run_number in run_order
            if repaired_assessments.get(run_number) is not None
            and repaired_assessments[run_number].is_successful
        ]
        failed_runs = [
            run_number
            for run_number in run_order
            if repaired_assessments.get(run_number) is None
            or (
                not repaired_assessments[run_number].is_successful
                and not repaired_assessments[run_number].repair_attempted
            )
        ]
        if not successful_runs or not failed_runs:
            break

        repaired_any = False
        failed_runs.sort(
            key=lambda run_number: min(
                abs(order_index[run_number] - order_index[other_run])
                for other_run in successful_runs
            )
        )
        for run_number in failed_runs:
            repaired = _repair_single_fit_assessment_from_sibling_runs(
                dataset_by_run[run_number],
                fingerprints_by_run[run_number],
                template,
                donor_assessments=[
                    repaired_assessments[other_run]
                    for other_run in sorted(
                        successful_runs,
                        key=lambda other_run: (
                            abs(order_index[run_number] - order_index[other_run]),
                            order_index[other_run],
                        ),
                    )
                ],
                current_values=current_values,
                parameter_bounds=parameter_bounds,
                fixed_param_names=fixed_param_names,
                metric=metric,
                fit_engine=fit_engine,
            )
            if repaired is None or not repaired.is_successful:
                continue
            repaired_assessments[run_number] = repaired
            repaired_any = True
            _progress_log(
                progress_callback,
                f"{template.title}: repaired single-fit screening seed for run "
                f"{dataset_by_run[run_number].run_label}.",
            )

        if not repaired_any:
            for run_number in failed_runs:
                assessment = repaired_assessments.get(run_number)
                if (
                    assessment is not None
                    and not assessment.is_successful
                    and not assessment.repair_attempted
                ):
                    repaired_assessments[run_number] = replace(assessment, repair_attempted=True)
            break

    return repaired_assessments


def _repair_single_fit_assessment_from_sibling_runs(
    dataset: MuonDataset,
    fingerprint: SpectrumFingerprint,
    template: CandidateTemplate,
    *,
    donor_assessments: list[CandidateAssessment],
    current_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    fixed_param_names: tuple[str, ...],
    metric: SelectionMetric,
    fit_engine: FitEngine,
) -> CandidateAssessment | None:
    attempts: list[ParameterSet] = []
    seen_signatures: set[tuple[tuple[str, float], ...]] = set()

    for donor in donor_assessments:
        if not donor.is_successful:
            continue
        seeded_values = {
            parameter.name: float(parameter.value) for parameter in donor.fit_result.parameters
        }
        seeded_parameters = _configured_single_fit_parameter_set(
            dataset,
            fingerprint,
            template,
            current_values=current_values,
            parameter_bounds=parameter_bounds,
            fixed_param_names=fixed_param_names,
            seeded_values=seeded_values,
            seeded_values_override_current=True,
        )
        for variant in _parameter_variants(seeded_parameters, template=template):
            signature = tuple((parameter.name, float(parameter.value)) for parameter in variant)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            attempts.append(_clone_parameter_set(variant))

    if not attempts:
        return None

    return _assess_single_fit_candidate_from_attempts(
        dataset,
        template,
        attempts=tuple(attempts),
        fit_engine=fit_engine,
        metric=metric,
    )


def _assess_single_fit_candidate_from_attempts(
    dataset: MuonDataset,
    template: CandidateTemplate,
    *,
    attempts: tuple[ParameterSet, ...],
    fit_engine: FitEngine,
    metric: SelectionMetric,
) -> CandidateAssessment:
    best_result: FitResult | None = None
    best_parameters: ParameterSet | None = None
    for parameters in attempts:
        result = fit_engine.fit(dataset, template.model.function, _clone_parameter_set(parameters))
        if _needs_fit_backend_fallback(result):
            result = _scipy_fit_fallback(dataset, template.model.function, parameters)
        if best_result is None:
            best_result = result
            best_parameters = _clone_parameter_set(parameters)
            continue
        if result.success and not best_result.success:
            best_result = result
            best_parameters = _clone_parameter_set(parameters)
            continue
        if result.success == best_result.success and result.chi_squared < best_result.chi_squared:
            best_result = result
            best_parameters = _clone_parameter_set(parameters)

    if best_result is None:
        best_result = FitResult(success=False, message="No fit attempt was created.")
        best_parameters = ParameterSet()

    n_points = int(dataset.n_points)
    k_free = len(best_result.parameters.free_parameters)
    aic, aicc, bic = compute_information_criteria(best_result.chi_squared, k_free, n_points)

    residual_rms, runs_z_score, max_abs_autocorrelation, residual_fft_peak_snr = (
        _residual_diagnostics(dataset, best_result)
    )
    bound_hits = _bound_hit_names(best_result.parameters)
    residual_gate_reasons = _residual_gate_reasons(
        fit_result=best_result,
        residual_rms=residual_rms,
        runs_z_score=runs_z_score,
        max_abs_autocorrelation=max_abs_autocorrelation,
        residual_fft_peak_snr=residual_fft_peak_snr,
        bound_hits=bound_hits,
    )
    fitted_time, fitted_curve, component_curves = _dense_fit_curves(
        dataset,
        template.model,
        best_result.parameters,
        fallback_parameters=best_parameters,
    )
    return CandidateAssessment(
        template=template,
        fit_result=best_result,
        aic=aic,
        aicc=aicc,
        bic=bic,
        selected_score=_metric_value(metric, aic, aicc, bic),
        residual_rms=residual_rms,
        runs_z_score=runs_z_score,
        max_abs_autocorrelation=max_abs_autocorrelation,
        residual_fft_peak_snr=residual_fft_peak_snr,
        residual_gate_passed=not residual_gate_reasons,
        residual_gate_reasons=tuple(residual_gate_reasons),
        bound_hits=tuple(bound_hits),
        fitted_time=fitted_time,
        fitted_curve=fitted_curve,
        component_curves=component_curves,
    )


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
    ]
    | None = None,
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
    base_variants = list(_initial_param_variants(base_by_run, template))
    if warm_start_by_run is not None:
        warm_variants = list(_initial_param_variants(warm_start_by_run, template))
        attempts.append(_clone_parameter_sets(warm_variants[0]))

        base_index = 0
        warm_index = 1
        while base_index < len(base_variants) or warm_index < len(warm_variants):
            if base_index < len(base_variants):
                attempts.append(_clone_parameter_sets(base_variants[base_index]))
                base_index += 1
            if warm_index < len(warm_variants):
                attempts.append(_clone_parameter_sets(warm_variants[warm_index]))
                warm_index += 1
    else:
        attempts.extend(_clone_parameter_sets(variant) for variant in base_variants)

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
        if any(not initial_params[int(dataset.run_number)][name].fixed for dataset in datasets)
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
                    staged_seed = _merge_result_values_into_parameter_sets(
                        staged_seed, mixed_results
                    )
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
                (parameter.name, float(parameter.value)) for parameter in parameter_sets[run_number]
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
        if free_count >= _HIGH_DIMENSION_FREE_COUNT:
            budget = max(_HIGH_DIMENSION_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS, budget)
            budget_cap = _HIGH_DIMENSION_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS_CAP
        else:
            budget_cap = _GLOBAL_FIT_SIMPLEX_RESCUE_CALLS_CAP
        return int(
            min(
                max(_GLOBAL_FIT_SIMPLEX_RESCUE_CALLS, budget),
                budget_cap,
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
    ]
    | None = None,
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
            parameter.value = float(np.clip(parameter.value, parameter.min, parameter.max))
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


def _layer_parameter_count(
    local_count: int,
    *,
    free_param_count: int,
    n_datasets: int,
) -> int:
    """IC ``k`` for a Hamming layer with ``local_count`` localised free params.

    ``parameter_count`` on an assessment is ``n_global + n_local * G`` (fixed
    params excluded). In layer ``m`` there are ``free_param_count - m`` free
    globals and ``m`` locals, so ``k(m) = (P - m) + m * G``. It is monotone
    non-decreasing in ``m`` for ``G >= 2`` (every synthetic/real series), which
    is what makes the layer bound admissible.
    """

    m = max(0, min(int(local_count), int(free_param_count)))
    return (free_param_count - m) + m * int(n_datasets)


def _metric_penalty(parameter_count: int, *, sample_count: int, metric: SelectionMetric) -> float:
    """The additive IC penalty ``IC - chi2`` for ``k`` params over ``n`` points.

    Mirrors :func:`compute_information_criteria` exactly so the layer bound uses
    the same penalty the winning-assessment IC does. All three penalties are
    monotone non-decreasing in ``k`` (hence in the layer index), so the bound
    ``chi2_floor + penalty(k(m))`` lower-bounds every assignment at layer ``>= m``.
    """

    k = max(int(parameter_count), 0)
    n = max(int(sample_count), 1)
    if metric == SelectionMetric.AIC:
        return 2.0 * k
    if metric == SelectionMetric.BIC:
        return k * math.log(n)
    # AICc — fall back to AIC's penalty when the small-sample correction is
    # undefined (n <= k + 1), matching compute_information_criteria which then
    # reports aicc = None and metric_value() falls back to AIC.
    aic_penalty = 2.0 * k
    if n > k + 1:
        return aic_penalty + 2.0 * k * (k + 1) / max(n - k - 1, 1)
    return aic_penalty


def _run_exhaustive_wavefront_search(
    datasets: list[MuonDataset],
    *,
    shortlisted_templates: list[CandidateTemplate],
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
    axis_key: str,
    metric: SelectionMetric,
    progress_callback: Callable[[str], None] | None,
    search_strategy: str,
    instrumentation: dict[str, object] | None,
    single_run_prefit_cache_for: Callable[
        [CandidateTemplate], dict[object, dict[int, ParameterSet]]
    ],
) -> tuple[GlobalCandidateAssessment, ...]:
    if not shortlisted_templates:
        return ()

    states: list[_WavefrontTemplateState] = []
    state_by_key: dict[str, _WavefrontTemplateState] = {}
    layers_by_key: dict[str, tuple[tuple[tuple[str, ...], ...], ...]] = {}
    baseline_seed_by_key: dict[str, dict[int, ParameterSet]] = {}
    total_assignments = 0
    max_rounds = 0

    for template in shortlisted_templates:
        base_by_run, fixed_param_names = template_contexts[template.key]
        prefit_base_by_run = _single_run_prefit_parameter_sets(
            datasets,
            template,
            fit_engine=FitEngine(),
            base_by_run=base_by_run,
            fixed_param_names=fixed_param_names,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
            cache=single_run_prefit_cache_for(template),
        )
        free_param_names = tuple(
            name for name in template.model.param_names if name not in fixed_param_names
        )
        layers = _layer_assignments(free_param_names)
        layers_by_key[template.key] = layers
        baseline_seed_by_key[template.key] = _all_global_seed_parameter_sets(prefit_base_by_run)
        total_assignments += sum(len(layer) for layer in layers)
        max_rounds = max(max_rounds, len(layers))
        state = _WavefrontTemplateState(
            template=template,
            fixed_param_names=fixed_param_names,
            prefit_base_by_run=prefit_base_by_run,
            free_param_names=free_param_names,
            exact_cache={},
            converged_assessments={},
            free_param_count=len(free_param_names),
        )
        states.append(state)
        state_by_key[template.key] = state
        _progress_log(
            progress_callback,
            f"{template.title}: exhaustive role search will enumerate "
            f"{sum(len(layer) for layer in layers)} assignment(s) across "
            f"{len(layers)} Hamming layer(s).",
        )

    # Technique A (exact layer truncation): fit the all-local anchor for each
    # template up front. All-local is the most flexible assignment, so its χ² is
    # a lower bound on every assignment of that template; combined with the
    # penalty that grows monotonically with the Hamming layer, it lets us halt a
    # template's enumeration once no remaining layer can beat the incumbent IC by
    # more than _LAYER_BOUND_MARGIN. Only a *cleanly converged* anchor arms the
    # bound — a mis-converged anchor floor could over-prune once a better low-layer
    # incumbent exists, so we disable the bound for that template instead.
    sample_count = int(sum(dataset.n_points for dataset in datasets))
    for state in states:
        if state.free_param_count == 0:
            continue
        anchor_local = tuple(state.free_param_names)
        anchor_assessment = _fit_exact_assignment(
            datasets,
            state.template,
            fit_engine=FitEngine(),
            base_by_run=state.prefit_base_by_run,
            global_param_names=(),
            local_param_names=anchor_local,
            fixed_param_names=state.fixed_param_names,
            axis_key=axis_key,
            metric=metric,
            cache=state.exact_cache,
            progress_callback=progress_callback,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
        )
        anchor_key = ((), anchor_local)
        state.exact_cache[anchor_key] = _compact_assessment_for_cache(anchor_assessment)
        if anchor_assessment.is_successful:
            state.converged_assessments[anchor_key] = anchor_assessment
            anchor_chi2 = float(
                sum(result.chi_squared for result in anchor_assessment.fit_results_by_run.values())
            )
            state.chi2_floor = anchor_chi2
            state.incumbent_ic = float(anchor_assessment.metric_value(metric))
            if state.best_assessment is None or _assessment_sort_key(
                anchor_assessment, metric
            ) < _assessment_sort_key(state.best_assessment, metric):
                state.best_assessment = anchor_assessment
            _progress_log(
                progress_callback,
                f"{state.template.title}: all-local anchor converged "
                f"(χ²={anchor_chi2:.2f}, {metric.value}={state.incumbent_ic:.2f}); "
                "layer bound armed.",
            )
        else:
            _progress_log(
                progress_callback,
                f"{state.template.title}: all-local anchor did not converge; "
                "layer bound disabled for this template (full enumeration).",
            )

    # Technique B (cross-template incumbent bound): the best IC found across ALL
    # templates so far. A template whose χ²_floor + minimum-possible penalty
    # (penalty at layer 0, the fewest params) can't beat this by more than the
    # margin cannot produce a winner and is skipped wholesale. Seeded and updated
    # ONLY from real converged metric_value()s — never from a floor+penalty
    # estimate (a bound-vs-bound comparison is unsound). A mis-converged anchor
    # never contributes (chi2_floor is None), so it cannot corrupt this incumbent.
    converged_incumbents = [s.incumbent_ic for s in states if s.chi2_floor is not None]
    cross_incumbent = min(converged_incumbents) if converged_incumbents else float("inf")

    worker_count = _wavefront_worker_count(total_assignments)
    if worker_count > 1 and total_assignments > 1:
        _progress_log(
            progress_callback,
            "Using spawn-based wavefront scheduling for exhaustive global/local "
            f"enumeration with {worker_count} worker(s) across "
            f"{len(shortlisted_templates)} shortlisted template(s).",
        )
    else:
        _progress_log(
            progress_callback,
            "Using serial wavefront scheduling for exhaustive global/local enumeration.",
        )

    executor: ProcessPoolExecutor | None = None
    if worker_count > 1 and total_assignments > 1:
        executor = _try_open_process_pool(
            max_workers=worker_count,
            progress_callback=progress_callback,
            activity="Exhaustive global/local role search",
        )

    try:
        for round_index in range(max_rounds):
            task_groups: list[list[_WavefrontAssignmentTask]] = []
            for state in states:
                layers = layers_by_key[state.template.key]
                if round_index >= len(layers):
                    continue
                if state.layer_bound_fired:
                    # Bound already fired in an earlier round; every remaining
                    # (higher) layer only adds penalty, so skip them all.
                    continue
                # Technique B: skip a whole template that cannot beat the best IC
                # found across ANY template. Unlike A this may fire at round 0
                # (its value is skipping a dominated template's all-global fit and
                # everything above it). χ²_floor + penalty(layer 0) is the
                # template's best achievable IC; the winning template's own bound
                # is <= its anchor IC <= cross_incumbent, so it can never trip this.
                if (
                    not state.layer_bound_fired
                    and state.chi2_floor is not None
                    and math.isfinite(cross_incumbent)
                ):
                    best_possible_k = _layer_parameter_count(
                        0,
                        free_param_count=state.free_param_count,
                        n_datasets=len(datasets),
                    )
                    best_possible_ic = state.chi2_floor + _metric_penalty(
                        best_possible_k, sample_count=sample_count, metric=metric
                    )
                    if best_possible_ic > cross_incumbent + _LAYER_BOUND_MARGIN:
                        state.layer_bound_fired = True
                        _record_counter(instrumentation, "cross_template_templates_pruned")
                        _record_counter(
                            instrumentation,
                            "cross_template_layers_pruned",
                            len(layers) - round_index,
                        )
                        _progress_log(
                            progress_callback,
                            f"{state.template.title}: cross-template bound fired "
                            f"(best possible {best_possible_ic:.2f} > cross-incumbent "
                            f"{cross_incumbent:.2f} + {_LAYER_BOUND_MARGIN:.1f}); "
                            "skipping this template entirely.",
                        )
                        continue
                # The all-local anchor (top layer) was already fitted up front and
                # lives in exact_cache/converged_assessments; do not re-fit it.
                if (
                    state.free_param_count > 0
                    and round_index == state.free_param_count
                ):
                    continue
                # Technique A: once χ²_floor + penalty(layer) exceeds the incumbent
                # IC by more than the margin, no assignment in this or any higher
                # layer can win — halt this template's enumeration. Guarded on a
                # cleanly converged anchor (chi2_floor is not None).
                if (
                    state.chi2_floor is not None
                    and math.isfinite(state.incumbent_ic)
                    and round_index > 0
                ):
                    layer_k = _layer_parameter_count(
                        round_index,
                        free_param_count=state.free_param_count,
                        n_datasets=len(datasets),
                    )
                    layer_ic_floor = state.chi2_floor + _metric_penalty(
                        layer_k, sample_count=sample_count, metric=metric
                    )
                    if layer_ic_floor > state.incumbent_ic + _LAYER_BOUND_MARGIN:
                        state.layer_bound_fired = True
                        _record_counter(instrumentation, "layer_bound_templates_pruned")
                        _record_counter(
                            instrumentation,
                            "layer_bound_layers_pruned",
                            len(layers) - round_index,
                        )
                        _progress_log(
                            progress_callback,
                            f"{state.template.title}: layer bound fired at Hamming "
                            f"layer {round_index}/{state.free_param_count} "
                            f"(floor {layer_ic_floor:.2f} > incumbent "
                            f"{state.incumbent_ic:.2f} + {_LAYER_BOUND_MARGIN:.1f}); "
                            "skipping remaining layers.",
                        )
                        continue
                assignment_group: list[_WavefrontAssignmentTask] = []
                for local_param_names in layers[round_index]:
                    global_param_names = tuple(
                        name for name in state.free_param_names if name not in local_param_names
                    )
                    predecessor = None
                    initial_seed_by_run = None
                    if round_index == 0:
                        initial_seed_by_run = baseline_seed_by_key[state.template.key]
                    else:
                        predecessor = _best_predecessor_assessment(
                            state.exact_cache,
                            free_param_names=state.free_param_names,
                            local_param_names=local_param_names,
                            metric=metric,
                        )
                    assignment_group.append(
                        _WavefrontAssignmentTask(
                            template_key=state.template.key,
                            template=state.template,
                            datasets=datasets,
                            base_by_run=state.prefit_base_by_run,
                            fixed_param_names=state.fixed_param_names,
                            global_param_names=global_param_names,
                            local_param_names=local_param_names,
                            axis_key=axis_key,
                            metric=metric,
                            search_strategy=search_strategy,
                            warm_start_source=_warm_start_source_from_assessment(predecessor),
                            initial_seed_by_run=initial_seed_by_run,
                        )
                    )
                if assignment_group:
                    task_groups.append(assignment_group)

            if not task_groups:
                continue

            ordered_tasks = _interleave_wavefront_tasks(task_groups)
            _append_metric(instrumentation, "staged_frontier_widths", len(ordered_tasks))
            _progress_log(
                progress_callback,
                f"Wavefront round {round_index + 1}/{max_rounds}: queueing "
                f"{len(ordered_tasks)} assignment(s) from {len(task_groups)} ready "
                f"template layer(s) on {worker_count} worker(s). Dispatch order is "
                "round-robin across ready templates so each template receives early "
                "slots while surplus workers drain the wider layers.",
            )

            round_results: list[_WavefrontAssignmentResult] = []
            if executor is None:
                round_results = [_run_wavefront_assignment_task(task) for task in ordered_tasks]
            else:
                future_to_task = {
                    executor.submit(_run_wavefront_assignment_task, task): task
                    for task in ordered_tasks
                }
                for future in as_completed(future_to_task):
                    round_results.append(future.result())

            successful_assignments = 0
            for result in round_results:
                state = state_by_key[result.template_key]
                _merge_instrumentation(instrumentation, result.instrumentation)
                state.exact_cache[(result.global_param_names, result.local_param_names)] = (
                    _compact_assessment_for_cache(result.assessment)
                )
                if result.assessment.is_successful:
                    successful_assignments += 1
                    state.converged_assessments[
                        (result.global_param_names, result.local_param_names)
                    ] = result.assessment
                    # Tighten the layer-bound incumbent with any better IC. A
                    # lower incumbent prunes more aggressively next round while
                    # staying admissible (the margin still protects the winner).
                    candidate_ic = float(result.assessment.metric_value(metric))
                    if candidate_ic < state.incumbent_ic:
                        state.incumbent_ic = candidate_ic
                    if candidate_ic < cross_incumbent:
                        cross_incumbent = candidate_ic
                if state.best_assessment is None or _assessment_sort_key(
                    result.assessment,
                    metric,
                ) < _assessment_sort_key(state.best_assessment, metric):
                    state.best_assessment = result.assessment

            _progress_log(
                progress_callback,
                f"Wavefront round {round_index + 1}/{max_rounds} complete: "
                f"{successful_assignments}/{len(round_results)} assignment(s) converged.",
            )
    finally:
        if executor is not None:
            _shutdown_process_pool(executor)

    optimized_assessments: list[GlobalCandidateAssessment] = []
    for state in states:
        if not state.converged_assessments and state.best_assessment is None:
            continue
        exact_cache = dict(state.exact_cache)
        successful_assessments = sorted(
            state.converged_assessments.values(),
            key=lambda assessment: _assessment_sort_key(assessment, metric),
        )
        for assessment in successful_assessments:
            exact_cache[(assessment.global_param_names, assessment.local_param_names)] = assessment

        if successful_assessments:
            for assessment in successful_assessments:
                optimized_assessments.append(
                    replace(
                        assessment,
                        fixed_param_names=state.fixed_param_names,
                        parameter_recommendations=_build_parameter_recommendations_from_exact_cache(
                            datasets,
                            assessment,
                            template=state.template,
                            fixed_param_names=state.fixed_param_names,
                            metric=metric,
                            cache=exact_cache,
                            names_to_test=set(state.free_param_names),
                        ),
                        assessment_key=_global_candidate_assessment_key(
                            state.template.key,
                            global_param_names=assessment.global_param_names,
                            local_param_names=assessment.local_param_names,
                        ),
                    )
                )

            best_assessment = successful_assessments[0]
            _progress_log(
                progress_callback,
                f"Completed exhaustive coupled optimisation for {state.template.title}. "
                f"{len(successful_assessments)} converged assignment(s); best {metric.value} = "
                f"{best_assessment.metric_value(metric):.3f} with "
                f"Global[{', '.join(best_assessment.global_param_names) or 'none'}], "
                f"Local[{', '.join(best_assessment.local_param_names) or 'none'}].",
            )
            continue

        failed_assessment = state.best_assessment
        if failed_assessment is None:
            continue
        optimized_assessments.append(
            replace(
                failed_assessment,
                fixed_param_names=state.fixed_param_names,
                parameter_recommendations=(),
                assessment_key=_global_candidate_assessment_key(
                    state.template.key,
                    global_param_names=failed_assessment.global_param_names,
                    local_param_names=failed_assessment.local_param_names,
                ),
            )
        )
        _progress_log(
            progress_callback,
            f"Completed exhaustive coupled optimisation for {state.template.title}. "
            "No assignment converged; keeping the best failed attempt for status reporting.",
        )

    return tuple(optimized_assessments)


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
    return sum(count - threshold + 1 for count in repeated_hits.values() if count >= threshold)


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
