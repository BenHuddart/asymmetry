"""Core analysis helpers for the global fit wizard."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import (
    CompositeModel,
    _legacy_fraction_rename_map,
    migrate_legacy_fraction_parameter_set,
)
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
    TemplateSeedContext,
    _bound_hit_names,
    _clone_parameter_set,
    _dense_fit_curves,
    _field_seed_context,
    _initial_parameters_for_template,
    _is_additive_relaxation_mixture_template,
    _migrate_fit_result_fractions,
    _needs_fit_backend_fallback,
    _parameter_variants,
    _persisted_curve_list,
    _persisted_curve_stride,
    _residual_diagnostics,
    _residual_gate_reasons,
    _scipy_fit_fallback,
    _template_cost_rank,
    _template_family_map,
    analysis_rebin_factor,
    build_candidate_templates,
    build_fit_wizard_recommendation,
    build_fit_wizard_recommendation_for_templates,
    build_wizard_families,
    compute_information_criteria,
    dataset_field_geometry,
    fingerprint_spectrum,
    is_multiplet_template_key,
    rerank_fit_wizard_recommendation,
    single_fit_build_signature,
)
from asymmetry.core.fitting.global_search import (
    GlobalSearchConfig,
)
from asymmetry.core.fitting.global_search.heuristics import (
    localisation_threshold_scale,
    parameter_localisation_priority,
)
from asymmetry.core.fitting.global_search.homogeneity import (
    ParameterHomogeneity,
    classify_parameter_homogeneity,
    wald_subset_delta_chi2,
)
from asymmetry.core.fitting.global_search.partition import (
    PartitionConfig,
    PartitionPath,
    PartitionSolution,
    Segment,
    partition_series,
    tier2_segment_cost,
)
from asymmetry.core.fitting.global_search.surrogate import (
    CollapseResult,
    RunEstimate,
    collapse_cost,
    rank_assignments,
    run_estimate_from_fit_result,
    surrogate_ic,
)
from asymmetry.core.fitting.global_search.surrogate import (
    metric_penalty as surrogate_metric_penalty,
)
from asymmetry.core.fitting.legacy_product_amplitudes import (
    fold_legacy_product_amplitude_names,
    fold_legacy_product_amplitude_set,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.process_pool import open_spawn_pool, terminate_spawn_pool
from asymmetry.core.fitting.wizard_scope import (
    DEFAULT_EFFORT_TIER,
    EffortTier,
    ScopeResolution,
    WizardScope,
    resolve_scope_for_datasets,
)
from asymmetry.core.fitting.wizard_timing import (
    WizardStageProgress,
    stage_timer,
)

#: Hard cap on the series template alphabet. The alphabet is a union over runs,
#: so on a long series it grows with the number of distinct phases the runs show
#: rather than with the number of runs — but the completion table is
#: (runs x alphabet) fits, so the union needs a ceiling. 24 is comfortably above
#: any single run's assessed set and keeps the table affordable on a 30-run scan.
_SERIES_ALPHABET_CAP = 24

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
#: Wall-clock budget (seconds) for the exhaustive wavefront role search. The loop
#: polls a monotonic deadline between Hamming rounds and between per-assignment
#: fits; on expiry it stops scheduling further work and returns the best-so-far
#: assessments (never hangs, never crashes) with a truncation note. The default
#: sits comfortably below any external watchdog (the GUI's is 240 s), and is a
#: BACKSTOP only — a healthy search finishes on its merits well within it. ``None``
#: disables the budget (unlimited). Kept high so it never trips on the small
#: golden-verdict harness cases.
_WAVEFRONT_TIME_BUDGET_SECONDS: float | None = 180.0
#: Interval (seconds) at which the pooled wavefront loop wakes to re-check the
#: cancel callback and the wall-clock deadline while futures are in flight.
_WAVEFRONT_POLL_INTERVAL_SECONDS = 0.25
#: When the all-local anchor for a template fails to converge, its layer bound is
#: unarmed and it would otherwise enumerate every Hamming layer (2^P assignments).
#: For high-dimensional templates that explosion is the dominant cost of a hung
#: search, so we cap the number of Hamming layers explored for an anchor-failed
#: template with more than ``_ANCHOR_FAILED_CAP_FREE_PARAMS`` free parameters.
#: Low layers (mostly-global assignments) are the physically-plausible role
#: splits for a series, so keeping the first few layers is conservative — it
#: prunes the combinatorially-worst upper layers, not the likely winners. Small
#: templates are unaffected and still enumerate fully (golden-harness parity).
_ANCHOR_FAILED_CAP_FREE_PARAMS = 4
_ANCHOR_FAILED_MAX_LAYERS = 3
#: Tolerance (χ² units) on the technique-D monotonicity certificate. A warm child
#: fit is accepted without escalation when its χ² does not exceed the parent's by
#: more than this. The parent is strictly less flexible, so a correct child fit is
#: expected to land at χ² <= χ²(parent); ε only absorbs Minuit's EDM-scale
#: convergence slop (default EDM tolerance ~1e-3·errordef). 1.0 is loose enough to
#: not fire on genuinely-converged children yet tight enough that a truly stuck
#: child (which would be a different, worse minimum) escalates to the full battery.
_WARM_CERTIFICATE_EPSILON = 1.0
_SHORTLIST_COUNT = 4
_SHORTLIST_SCORE_WINDOW = 6.0
_SHORTLIST_CAP = 6
_GLOBAL_FIT_MAX_CALLS = 1200
_GLOBAL_FIT_MAX_CALLS_CAP = 4200
_LOW_RESIDUAL_RMS_FOR_STRUCTURE_WARNINGS = 0.25
#: Standardized residual RMS at or below which a parameter resting at its bound
#: is accepted as the valid answer rather than a gate failure. A bound-hit only
#: signals a fit pathology when the fit is *also* poor; with a good fit the bound
#: is simply the physical value (e.g. a relaxation/damping rate -> 0, i.e.
#: negligible damping, as in an ordered magnet's low-temperature ZF precession).
#: Matches the ``residual_rms > 1.25`` "high RMS" threshold in
#: :func:`fit_wizard._residual_gate_reasons`.
_GOOD_FIT_RESIDUAL_RMS_FOR_BOUND_HITS = 1.25
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

# --------------------------------------------------------------------------- #
# Non-exhaustive search engine selector (PR 4).
#
# The *engine* axis chooses which enumerator runs the role search. It is
# deliberately distinct from ``search_strategy`` (which drives per-fit local
# search settings via ``_staged_local_search_settings``): a heuristic engine
# still fits every real assignment with the same fidelity as exhaustive, so its
# ICs sit on exactly the same footing as the frozen baseline — the only moving
# variable is *which* assignments get fit. ``"exhaustive"`` and ``"thorough"``
# both resolve to the byte-for-byte current wavefront call, so the exact path is
# untouched by definition. PR 5 maps its ``EffortTier`` enum onto these strings.
# --------------------------------------------------------------------------- #
SEARCH_ENGINE_EXHAUSTIVE = "exhaustive"
SEARCH_ENGINE_THOROUGH = "thorough"
SEARCH_ENGINE_LOW = "low"
SEARCH_ENGINE_BALANCED = "balanced"
#: The separable role search: all-local straight from the per-run fits, a
#: full-covariance GLS surrogate over every sharing pattern, backward
#: elimination as the exact path, then the winner's flip-neighbourhood. It is
#: the default engine for every effort tier; ``"exhaustive"`` stays reachable
#: through the ``search_engine`` seam as the harness referee.
SEARCH_ENGINE_SEPARABLE = "separable"
_DEFAULT_SEARCH_ENGINE = SEARCH_ENGINE_SEPARABLE
#: Engines that resolve to the exact wavefront (no heuristic pre-fixing/skipping).
_EXACT_SEARCH_ENGINES = frozenset({SEARCH_ENGINE_EXHAUSTIVE, SEARCH_ENGINE_THOROUGH})
#: Every supported ``search_engine`` value. An unrecognised value must raise
#: rather than silently fall through to the heuristic path (a typo would
#: otherwise change behaviour without warning).
SEARCH_ENGINES = (
    SEARCH_ENGINE_SEPARABLE,
    SEARCH_ENGINE_EXHAUSTIVE,
    SEARCH_ENGINE_THOROUGH,
    SEARCH_ENGINE_BALANCED,
    SEARCH_ENGINE_LOW,
)

#: Homogeneity (Q) pre-test band edges per heuristic engine (technique E). A
#: parameter is pre-fixed *local* when its Q upper-tail p-value falls below the
#: local edge (strong evidence it varies) and *global* when it exceeds the global
#: edge (no evidence it varies). Low uses wide bands (pre-fix aggressively);
#: Balanced uses conservative bands (pre-fix only the clear tails). PR 5 owns the
#: final per-tier numbers; these are sensible defaults.
_Q_BANDS_LOW = (0.02, 0.80)
_Q_BANDS_BALANCED = (0.002, 0.98)
#: Balanced surrogate verify budget: real-fit the top-K surrogate-ranked subsets
#: per Hamming layer of the ambiguous middle, growing K when the realised order
#: disagrees near the top (technique G). Low skips the surrogate and runs greedy.
_SURROGATE_TOP_K = 3
#: Template racing (technique H): how many templates advance past the shallow
#: (layer 0–1) race into the deeper heuristic search.
_RACING_ADVANCE_COUNT = 2
#: Free-parameter count (``n_global + n_local * G``) at or above which a coupled
#: node in the separable search is solved with the profiled strategy instead of
#: the joint one. See ``_separable_coupled_strategy`` for the trade; 20 is the
#: same "wide assignment" threshold ``_fit_exact_assignment`` already uses to
#: decide an assignment is difficult.
_SEPARABLE_PROFILED_FREE_PARAM_COUNT = 20
#: How many templates advance from the separable engine's race into backward
#: elimination. The others keep their all-local node, which costs no coupled fit
#: at all, so a raced-out template still carries an honest score on the
#: leaderboard rather than vanishing from it.
_SEPARABLE_RACING_ADVANCE_COUNT = 3
_HIGH_DIMENSION_FREE_COUNT = 40
_EXTREME_DIMENSION_FREE_COUNT = 70
_MAX_TEMPLATE_WORKERS = 4
_MAX_WAVEFRONT_WORKERS = 12
#: How often the phase-1 single-fit-table drain re-checks ``cancel_callback``
#: while waiting on the pool. Each table is a minutes-long job, so without this
#: the stage is uncancellable in practice.
_PHASE_ONE_CANCEL_POLL_SECONDS = 0.2
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

# --------------------------------------------------------------------------- #
# Effort tier -> search-engine mapping (PR 5, revised).
#
# ``EffortTier`` is the user-facing slider; ``search_engine`` (PR 4) remains the
# lower-level enumerator selector so existing callers/tests that pass
# ``search_engine=`` directly are unaffected.
#
# REVISION (PR 5 rework): every tier resolves to the exact bounded-wavefront
# engine for the *role search*. PR 2's exact bounds made Exhaustive near-minimal
# (~1000 fits) and 12-way parallel, so empirically the heuristic Low/Balanced
# engines — serial by construction — were *slower* (up to 15x) on real workloads
# with no fit-count headroom left to reclaim. That measurement still stands and
# the mapping below is unchanged; the heuristic engines remain reachable only
# behind the low-level ``search_engine`` string (the PR 4 seam) for future
# large-P use and regression coverage.
#
# REVISION (screening tiers): the search engine was never where a real series
# spends its time. Profiling the screening stage on a 14-dataset synthetic
# series shows ~95 s of CPU *per dataset*, of which ~88 % is the numerically
# integrated longitudinal-field Kubo-Toyabe family — candidate templates the
# portfolio offers on every series whether or not the data ask for them. Because
# the tier knobs above (I portfolio cap / J identifiability demotion /
# K screening decimation) are all gated on the heuristic engine string, no tier
# could buy a cheaper answer at all: the slider was inert end to end.
#
# ``_EFFORT_TIER_SCREENING`` fixes that at the stage that dominates, by giving
# each tier a *portfolio* budget for screening (see ``ScreeningEffortProfile``).
# Low and Balanced drop the numerically expensive candidates and cap the
# portfolio; Thorough and Exhaustive screen everything, exactly as before, so no
# existing caller's answer changes.
# --------------------------------------------------------------------------- #
_EFFORT_TIER_SEARCH_ENGINE: dict[EffortTier, str] = {
    EffortTier.LOW: SEARCH_ENGINE_SEPARABLE,
    EffortTier.BALANCED: SEARCH_ENGINE_SEPARABLE,
    EffortTier.THOROUGH: SEARCH_ENGINE_SEPARABLE,
    EffortTier.EXHAUSTIVE: SEARCH_ENGINE_SEPARABLE,
}


@dataclass(frozen=True)
class ScreeningEffortProfile:
    """What one :class:`EffortTier` may spend on the *screening* stage.

    Screening fits every portfolio template to every dataset independently, so
    its cost is (templates x datasets x per-fit cost) and the only levers that
    matter are which templates are offered and how many. A tier that trims the
    portfolio returns a coarser ranking over a smaller candidate set — a real,
    honest trade, and the one a caller reaches for when the exhaustive answer
    will not finish.
    """

    #: Hard cap on screened templates, or ``None`` for "no cap". Retained
    #: templates are the cheapest, most parsimonious ones (plus every
    #: pattern-matched candidate, which is never dropped).
    max_templates: int | None
    #: Drop templates whose slowest component is tagged
    #: :attr:`ComputationalCost.EXPENSIVE`. These are the numerically integrated
    #: dynamic Kubo-Toyabe / powder-average models; one of them can be an order
    #: of magnitude more expensive than the rest of the portfolio combined.
    drop_expensive_templates: bool
    #: Drop templates with more than this many additive terms, or ``None``.

    max_additive_terms: int | None

    @property
    def prunes(self) -> bool:
        """Does this profile restrict the portfolio at all?"""
        return (
            self.max_templates is not None
            or self.drop_expensive_templates
            or self.max_additive_terms is not None
        )


#: Per-tier screening budget. Thorough/Exhaustive are unrestricted, so every
#: caller that does not pass ``effort_tier`` (the default is
#: :data:`~asymmetry.core.fitting.wizard_scope.DEFAULT_EFFORT_TIER`, i.e.
#: Exhaustive) screens exactly the portfolio it screened before.
_EFFORT_TIER_SCREENING: dict[EffortTier, ScreeningEffortProfile] = {
    EffortTier.LOW: ScreeningEffortProfile(
        max_templates=6,
        drop_expensive_templates=True,
        max_additive_terms=3,
    ),
    EffortTier.BALANCED: ScreeningEffortProfile(
        max_templates=10,
        drop_expensive_templates=True,
        max_additive_terms=None,
    ),
    EffortTier.THOROUGH: ScreeningEffortProfile(
        max_templates=None,
        drop_expensive_templates=False,
        max_additive_terms=None,
    ),
    EffortTier.EXHAUSTIVE: ScreeningEffortProfile(
        max_templates=None,
        drop_expensive_templates=False,
        max_additive_terms=None,
    ),
}

#: Technique I (Low portfolio cap): Low shortlists at most this many templates
#: (vs. ``_SHORTLIST_COUNT``/``_SHORTLIST_CAP`` for Balanced/Thorough/Exhaustive).
_LOW_SHORTLIST_CAP = 3
#: Technique I: Low skips any template whose parameter count exceeds this —
#: a 3-4-additive-component model must earn its place at Balanced+ instead.
_LOW_MAX_TEMPLATE_PARAM_COUNT = 5
#: Technique I: extra IC penalty per additive component applied *only* at Low
#: ranking time (never baked into the stored aic/aicc/bic fields other tiers and
#: the harness baseline read) — a complexity prior so more additive terms must
#: buy a proportionally larger fit improvement to be shortlisted/preferred.
_LOW_COMPLEXITY_PRIOR_PER_ADDITIVE_TERM = 3.0

#: Technique J (identifiability demotion): a template is demoted (sorted to the
#: back of the Low shortlist ranking, never hard-dropped) when any pair of its
#: free parameters is this correlated in the initial-screen fit's covariance.
_IDENTIFIABILITY_CORRELATION_THRESHOLD = 0.98
#: Technique J: a template is demoted when any parameter's relative uncertainty
#: (sigma / |value|) spans at least this many decades from the tightest one —
#: a proxy for "some parameters are essentially unconstrained by this template".
_IDENTIFIABILITY_ERROR_DECADE_SPAN = 3.0

#: Technique K (screening decimation): rebin factor applied to the *search*
#: phase's datasets at Low/Balanced. The winner and its full flip-neighbourhood
#: are always refitted at native resolution afterward (see
#: ``_fill_winner_flip_neighbourhood``), so no decimated IC ever reaches the
#: returned leaderboard.
_DECIMATION_FACTOR_LOW = 4
_DECIMATION_FACTOR_BALANCED = 2
#: Technique K Nyquist gate: never decimate when the aggregate fingerprint's
#: dominant FFT content would alias under the candidate rebin factor. Comparing
#: the dominant frequency's period (in samples) against the rebin factor is a
#: conservative proxy for a real Nyquist check without needing the raw dwell
#: time here (``dominant_fft_cycles_in_window`` free of dt).
_DECIMATION_MIN_CYCLES_MARGIN = 2.0


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
    """Global-fit analysis payload plus the current recommendation.

    Two answers live here, and they do not compete. ``recommended_key``,
    ``comparable_keys`` and ``assessments`` are the **series-wide** answer: one
    template, one role assignment, every run. ``partition_path`` and
    ``phase_assessments`` are the **partitioned** answer: the series split into
    contiguous phases at structural breaks, each phase with its own template and
    assignment. A screening pass always computes the path (it is closed-form and
    cheap); the per-phase coupled fits happen only when
    :func:`build_global_fit_wizard_recommendation` is asked for a ``partition_k``.

    ``partition_path`` is ``None`` when the series cannot be partitioned at all —
    fewer than ``2 × min_segment`` runs, or no template scoreable on every run.
    ``phase_assessments`` is keyed ``(k, segment_index)`` so the assessments of
    the neighbouring solutions tier 3 verified stay addressable next to the
    selected one; ``recommended_partition_k`` names which ``k`` was optimised.
    """

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
    partition_path: PartitionPath | None = None
    phase_assessments: dict[tuple[int, int], GlobalCandidateAssessment] = field(
        default_factory=dict
    )
    recommended_partition_k: int | None = None

    @property
    def recommended_assessment(self) -> GlobalCandidateAssessment | None:
        return self.assessment_for_key(self.recommended_key)

    @property
    def recommended_partition(self) -> PartitionSolution | None:
        """The optimised partition solution, when one was optimised."""

        if self.partition_path is None or self.recommended_partition_k is None:
            return None
        return self.partition_path.solutions[self.recommended_partition_k]

    def phase_assessment(self, segment_index: int) -> GlobalCandidateAssessment | None:
        """The recommended assessment of one phase of the optimised partition.

        ``None`` for an excluded end stub, which never receives a coupled fit.
        """

        if self.recommended_partition_k is None:
            return None
        return self.phase_assessments.get((self.recommended_partition_k, segment_index))

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
    #: Hard cap on the highest Hamming layer this template may enumerate. ``None``
    #: means no cap (the admissible layer bound governs pruning). It is set only
    #: when the all-local anchor did NOT converge for a high-dimensional template:
    #: with no ``chi2_floor`` the admissible bound cannot arm, so without a cap the
    #: template would enumerate all 2^P assignments — the dominant cost of a hung
    #: search. Capping to the first few (mostly-global) layers keeps pruning
    #: conservative instead of removing it entirely.
    layer_cap: int | None = None


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
    """Workers for phase-1 single-fit table generation — one task per *dataset*.

    Bounded by the host's CPU count: each table is minutes of CPU-bound fitting,
    so running more of them than there are cores only lengthens the time to the
    *first* completed table — which is the stage's only sign of life.

    It is deliberately **not** bounded by ``_MAX_TEMPLATE_WORKERS``. That cap
    sizes template-level fan-out inside one series fit, where the tasks share
    caches and the deeper stages fan out again; phase-1 tables are independent
    whole-dataset jobs, and capping them at four left a 14-dataset series using
    a fraction of a large host while taking 3.5 sequential rounds to finish.
    """
    if task_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count))


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


def _shutdown_process_pool(
    executor: ProcessPoolExecutor,
    *,
    wait: bool = True,
    cancel_futures: bool = False,
) -> None:
    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            # Executor wrappers that predate cancel_futures (Python <3.9 signature).
            shutdown(wait=wait)


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
    warm_start_chi2: float | None = None
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
        # Parent χ² for technique D's monotonicity certificate: the predecessor is
        # a strict single-flip-simpler assignment (one fewer local), so the warm
        # child should not exceed it.
        warm_start_chi2 = float(
            sum(result.chi_squared for result in task.warm_start_source.fit_results_by_run.values())
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
        warm_start_chi2=warm_start_chi2,
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


def _single_fit_completion_task(
    dataset: MuonDataset,
    templates: tuple[CandidateTemplate, ...],
    source: FitWizardRecommendation,
    rebin_factor: int,
    metric: SelectionMetric,
    sibling_values_by_template: dict[str, ParameterSet],
) -> tuple[int, FitWizardRecommendation, int]:
    """Score one run against the whole series alphabet at the series resolution.

    Plain-data in, plain-data out, so this can cross a process boundary. Cells
    the run's own single-fit table already holds *at this rebin factor* are kept
    verbatim; everything else is fitted on the rebinned record, warm-started
    from the run's own values for that template when it has them at another
    factor and from the best sibling run's otherwise, and seeded from the run's
    own peak analysis so a multiplet template's lines start on the lines this
    run actually shows.

    Returns the run number, the completed table, and the number of fits it cost.
    """
    run_number = int(dataset.run_number)
    analysis_dataset = dataset.rebin(rebin_factor) if rebin_factor > 1 else dataset
    same_resolution = int(source.rebin_factor) == int(rebin_factor)

    assessments_by_key: dict[str, CandidateAssessment] = {}
    warm_templates: list[CandidateTemplate] = []
    warm_starts: dict[str, ParameterSet] = {}
    cold_templates: list[CandidateTemplate] = []
    for template in templates:
        assessment = source.assessment_for_key(template.key)
        if assessment is not None and assessment.is_successful and same_resolution:
            assessments_by_key[template.key] = assessment
            continue
        if assessment is not None and assessment.is_successful:
            warm_templates.append(template)
            warm_starts[template.key] = assessment.fit_result.parameters
        elif template.key in sibling_values_by_template:
            warm_templates.append(template)
            warm_starts[template.key] = sibling_values_by_template[template.key]
        else:
            cold_templates.append(template)

    seed_context = TemplateSeedContext(
        peak_analysis=source.peak_analysis,
        multiplet_matches=source.multiplet_matches,
        field_gauss=dataset.field,
        geometry=dataset_field_geometry(dataset),
    )
    # Two calls, one ladder depth each: a cell that starts from an already
    # fitted answer needs the answer tried, not a ladder around it, while a cell
    # no run has ever fitted gets the normal seed ladder.
    for group, budget in ((warm_templates, 1), (cold_templates, 5)):
        if not group:
            continue
        completed = build_fit_wizard_recommendation_for_templates(
            analysis_dataset,
            tuple(group),
            fingerprint=source.fingerprint,
            metric=metric,
            seed_context=seed_context,
            warm_start_by_template=warm_starts,
            variant_budget=budget,
        )
        for assessment in completed.assessments:
            assessments_by_key[assessment.template.key] = assessment

    recommendation = rerank_fit_wizard_recommendation(
        FitWizardRecommendation(
            fingerprint=source.fingerprint,
            templates=tuple(templates),
            assessments=tuple(assessments_by_key[template.key] for template in templates),
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary="",
            peak_analysis=source.peak_analysis,
            multiplet_matches=source.multiplet_matches,
            family_reports=source.family_reports,
            scope_note=source.scope_note,
            rebin_factor=int(rebin_factor),
            analysed_points=int(analysis_dataset.n_points),
        ),
        metric,
    )
    return run_number, recommendation, len(warm_templates) + len(cold_templates)


@dataclass(frozen=True)
class GlobalFitWizardCandidatePortfolio:
    """Cheap pre-analysis portfolio detection for the global fit wizard."""

    ordered_datasets: tuple[MuonDataset, ...]
    series_axis_key: str
    series_axis_label: str
    mixed_axes_warning: str | None
    fingerprints_by_run: dict[int, SpectrumFingerprint]
    templates: tuple[CandidateTemplate, ...]
    #: Alphabet templates something in the data positively identified — a
    #: multiplet built from a run's detected lines, or a template whose family a
    #: run's multiplet pattern match named. The effort tier never trims them and
    #: the staged shortlist force-includes them. Empty on a preview portfolio,
    #: which has run no peak search of its own.
    pattern_template_keys: tuple[str, ...] = ()

    @property
    def dataset_order(self) -> tuple[int, ...]:
        return tuple(int(dataset.run_number) for dataset in self.ordered_datasets)


def _preview_series_templates(
    ordered_datasets: Sequence[MuonDataset],
    aggregate_fingerprint: SpectrumFingerprint,
    current_model: CompositeModel | None,
    *,
    scope: WizardScope | None = None,
) -> tuple[CandidateTemplate, ...]:
    """The candidate list to *quote* for a series before phase 1 has run.

    A preview, not the portfolio the screen will score: it is built from the
    scope and the median fingerprint alone, so the Setup page can size the job
    without paying for a single fit. The portfolio that is actually screened is
    the series alphabet — the union of what the per-run single-fit wizard
    assessed (:func:`series_template_alphabet`) — which no aggregate can
    anticipate, because a template that describes a minority phase of the series
    is invisible in the median.

    With ``scope is None`` this is the legacy hint-gated candidate list. With a
    scope it is family-based: every in-scope family contributes its Stage-1
    shapes, and the hinted families and the baseline contribute their full
    member sets.
    """
    resolution: ScopeResolution | None = None
    if scope is not None:
        resolution = resolve_scope_for_datasets(list(ordered_datasets), scope)
    families = build_wizard_families(
        aggregate_fingerprint, current_model, scope_resolution=resolution
    )

    if scope is None:
        return tuple(build_candidate_templates(aggregate_fingerprint, current_model=current_model))

    templates: list[CandidateTemplate] = []
    known_keys: set[str] = set()
    for family in families:
        chosen = [family.stage1_rep, *family.stage1_extras]
        if family.priority > 0.0 or family.key == "baseline":
            chosen.extend(family.stage2_members)
        for template in chosen:
            if template.key not in known_keys:
                templates.append(template)
                known_keys.add(template.key)
    return tuple(templates)


def series_template_alphabet(
    recommendations_by_run: Mapping[int, FitWizardRecommendation],
    *,
    cap: int = _SERIES_ALPHABET_CAP,
) -> tuple[CandidateTemplate, ...]:
    """The series' candidate alphabet: the union of what its runs assessed.

    Every template any run's single-fit wizard successfully fitted is a
    candidate for the series — including a multiplet template only a couple of
    runs produced, which is exactly the shape a median-fingerprint portfolio
    loses. Null baselines are excluded: they are a per-run reference for "no
    significant structure", never a model for a coupled series fit.

    Order is the order to spend a bounded budget in: templates a run *chose*
    (recommended, or scored comparable to the recommendation) first, then by the
    best rank any single run gave them, then by first appearance in run order.
    The first ``cap`` survive.
    """
    templates_by_key: dict[str, CandidateTemplate] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    chosen_keys: set[str] = set()
    for run_index, run_number in enumerate(sorted(recommendations_by_run)):
        recommendation = recommendations_by_run[run_number]
        ranked = [
            assessment
            for assessment in recommendation.sorted_assessments()
            if not assessment.is_null_baseline and assessment.is_successful
        ]
        for rank, assessment in enumerate(ranked):
            key = assessment.template.key
            templates_by_key.setdefault(key, assessment.template)
            first_seen.setdefault(key, (run_index, rank))
            best_rank[key] = min(best_rank.get(key, rank), rank)
        if recommendation.recommended_key is not None:
            chosen_keys.add(recommendation.recommended_key)
        chosen_keys.update(recommendation.comparable_keys)

    ordered = sorted(
        templates_by_key,
        key=lambda key: (0 if key in chosen_keys else 1, best_rank[key], first_seen[key]),
    )
    return tuple(templates_by_key[key] for key in ordered[: int(cap)])


def series_rebin_factor(
    datasets: Sequence[MuonDataset],
    recommendations_by_run: Mapping[int, FitWizardRecommendation],
) -> int:
    """One value-rebinning factor for the whole series: the smallest per run.

    Every run's own factor already respects that run's bandwidth (see
    :func:`~asymmetry.core.fitting.fit_wizard.analysis_rebin_factor`), so the
    minimum respects all of them. The series needs a single factor because the
    information criteria of two runs are only summable when both refer to
    records of the same construction.
    """
    return max(
        1,
        min(
            int(recommendations_by_run[int(dataset.run_number)].rebin_factor)
            for dataset in datasets
        ),
    )


def _series_pattern_template_keys(
    templates: Sequence[CandidateTemplate],
    recommendations_by_run: Mapping[int, FitWizardRecommendation],
) -> tuple[str, ...]:
    """Alphabet templates something in the data positively identified.

    Two kinds: a multiplet template (its lines came from a run's detected peak
    set) and a template whose family a run's multiplet pattern match named.
    These are never trimmed by an effort tier — dropping them would change the
    answer rather than coarsen it.
    """
    protected: set[str] = set()
    for recommendation in recommendations_by_run.values():
        family_by_template = _template_family_map(recommendation.family_reports)
        matched_families = {match.family_key for match in recommendation.multiplet_matches}
        for template in templates:
            if is_multiplet_template_key(template.key):
                protected.add(template.key)
            elif family_by_template.get(template.key) in matched_families:
                protected.add(template.key)
    return tuple(template.key for template in templates if template.key in protected)


def _sibling_warm_starts(
    templates: Sequence[CandidateTemplate],
    recommendations_by_run: Mapping[int, FitWizardRecommendation],
    metric: SelectionMetric,
) -> dict[str, ParameterSet]:
    """Best per-template fitted values across the series, for completion seeds.

    A run that never fitted a template has no values of its own to start from;
    the same model fitted on the sibling run it scored best on is the closest
    thing the series has to an answer. Chosen deterministically (best metric
    value, ties broken by run order) so a completion table does not depend on
    which worker finished first.
    """
    best: dict[str, tuple[float, ParameterSet]] = {}
    for run_number in sorted(recommendations_by_run):
        recommendation = recommendations_by_run[run_number]
        for template in templates:
            assessment = recommendation.assessment_for_key(template.key)
            if assessment is None or not assessment.is_successful:
                continue
            score = float(assessment.metric_value(metric))
            if not np.isfinite(score):
                continue
            incumbent = best.get(template.key)
            if incumbent is None or score < incumbent[0]:
                best[template.key] = (score, assessment.fit_result.parameters)
    return {key: parameters for key, (_score, parameters) in best.items()}


def _scope_filtered_templates(
    templates: Sequence[CandidateTemplate],
    ordered_datasets: Sequence[MuonDataset],
    scope: WizardScope | None,
) -> tuple[CandidateTemplate, ...]:
    """Drop templates whose components are out of scope for *every* run."""
    if scope is None:
        return tuple(templates)
    resolution = resolve_scope_for_datasets(list(ordered_datasets), scope)
    return tuple(
        template
        for template in templates
        if all(name in resolution.included_set for name in template.model.component_names)
    )


def single_fit_table_covers_portfolio(
    datasets: Sequence[MuonDataset],
    templates: Sequence[CandidateTemplate],
    recommendations_by_run: Mapping[int, FitWizardRecommendation],
) -> bool:
    """True when ``recommendations_by_run`` is a usable series score table.

    Two conditions, and both are what the series arithmetic needs rather than
    bookkeeping: every run carries a score for every template (a missing cell
    makes that template's series sum meaningless), and every run was scored at
    the *same* rebin factor (an information criterion computed on a coarser
    record is not comparable with one computed on a finer one). A cell that
    *failed* still covers: the completion pass already retried it from a
    sibling run's values, so a failure in a completed table is the honest
    answer that the template does not describe that run, not work left
    undone. Coverage is not identity: a run's table may hold more templates
    than the series ranks, which is exactly what a genuine single-run analysis
    does.
    """
    if not recommendations_by_run or not templates:
        return False
    factors: set[int] = set()
    for dataset in datasets:
        recommendation = recommendations_by_run.get(int(dataset.run_number))
        if recommendation is None:
            return False
        factors.add(int(recommendation.rebin_factor))
        for template in templates:
            if recommendation.assessment_for_key(template.key) is None:
                return False
    return len(factors) == 1


def build_global_fit_wizard_candidate_portfolio(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    scope: WizardScope | None = None,
    single_fit_recommendations_by_run: Mapping[int, FitWizardRecommendation] | None = None,
) -> GlobalFitWizardCandidatePortfolio:
    """Return the ordered datasets, fingerprints, and candidate portfolio for one series.

    With ``single_fit_recommendations_by_run`` the portfolio is the **series
    alphabet**: the union of the templates those per-run single-fit wizard
    recommendations assessed, scope-filtered and capped
    (:func:`series_template_alphabet`). Without them it is the cheap
    **preview** list the Setup page quotes before phase 1 has run — the scope's
    families around the median fingerprint, no fits, and no claim to be the set
    that will actually be screened.
    """
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    ordered_datasets, axis_key, axis_label, mixed_axes_warning = _ordered_datasets_with_axis(
        datasets
    )
    fingerprints_by_run = {
        int(dataset.run_number): fingerprint_spectrum(dataset) for dataset in ordered_datasets
    }
    if single_fit_recommendations_by_run:
        templates = _scope_filtered_templates(
            series_template_alphabet(single_fit_recommendations_by_run),
            ordered_datasets,
            scope,
        )
        pattern_template_keys = _series_pattern_template_keys(
            templates, single_fit_recommendations_by_run
        )
    else:
        aggregate_fingerprint = _aggregate_fingerprints(
            [fingerprints_by_run[int(dataset.run_number)] for dataset in ordered_datasets]
        )
        templates = _preview_series_templates(
            ordered_datasets,
            aggregate_fingerprint,
            current_model,
            scope=scope,
        )
        pattern_template_keys = ()
    return GlobalFitWizardCandidatePortfolio(
        ordered_datasets=tuple(ordered_datasets),
        series_axis_key=axis_key,
        series_axis_label=axis_label,
        mixed_axes_warning=mixed_axes_warning,
        fingerprints_by_run=fingerprints_by_run,
        templates=templates,
        pattern_template_keys=pattern_template_keys,
    )


#: Cost rank at or above which a template counts as numerically expensive.
_EXPENSIVE_COST_RANK = 2


def screening_templates_for_effort_tier(
    templates: Sequence[CandidateTemplate],
    effort_tier: EffortTier,
    *,
    pattern_template_keys: Sequence[str] = (),
) -> tuple[tuple[CandidateTemplate, ...], tuple[CandidateTemplate, ...]]:
    """Split a candidate portfolio into ``(screened, skipped)`` for one effort tier.

    Templates named by a multiplet *pattern match* are never skipped: those are
    positively identified by the data, so dropping them would change the answer
    rather than coarsen it. Everything else is ranked cheapest-and-simplest
    first — the order in which a shorter portfolio should be spent — and the
    tier's budget applied.

    Model order of the retained templates is preserved, so a caller comparing
    two tiers' tables reads the same rows in the same order.
    """
    profile = _EFFORT_TIER_SCREENING.get(effort_tier)
    ordered = tuple(templates)
    if profile is None or not profile.prunes or not ordered:
        return ordered, ()

    protected = set(pattern_template_keys)
    index_by_key = {template.key: index for index, template in enumerate(ordered)}

    def _priority(template: CandidateTemplate) -> tuple[int, int, int, int]:
        return (
            0 if template.key in protected else 1,
            _template_cost_rank(template),
            int(template.additive_terms),
            index_by_key[template.key],
        )

    kept_keys: set[str] = set()
    for template in sorted(ordered, key=_priority):
        if template.key in protected:
            kept_keys.add(template.key)
            continue
        if profile.drop_expensive_templates and _template_cost_rank(template) >= (
            _EXPENSIVE_COST_RANK
        ):
            continue
        if (
            profile.max_additive_terms is not None
            and int(template.additive_terms) > profile.max_additive_terms
        ):
            continue
        if profile.max_templates is not None and len(kept_keys) >= profile.max_templates:
            continue
        kept_keys.add(template.key)

    screened = tuple(template for template in ordered if template.key in kept_keys)
    skipped = tuple(template for template in ordered if template.key not in kept_keys)
    return screened, skipped


def _apply_screening_effort_tier(
    portfolio: GlobalFitWizardCandidatePortfolio,
    effort_tier: EffortTier,
    *,
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
) -> GlobalFitWizardCandidatePortfolio:
    """Return ``portfolio`` narrowed to what ``effort_tier`` will screen.

    The skipped templates are announced, never silently dropped: a coarser answer
    is only honest if the caller can see what it was coarsened by.
    """
    screened, skipped = screening_templates_for_effort_tier(
        portfolio.templates,
        effort_tier,
        pattern_template_keys=portfolio.pattern_template_keys,
    )
    if not skipped:
        return portfolio
    _progress_log(
        progress_callback,
        f"Effort tier '{effort_tier.value}': screening {len(screened)} of "
        f"{len(portfolio.templates)} candidates; skipping "
        + ", ".join(template.key for template in skipped)
        + ".",
    )
    _set_metric(instrumentation, "screening_effort_tier", effort_tier.value)
    _set_metric(
        instrumentation,
        "screening_skipped_template_keys",
        [template.key for template in skipped],
    )
    return replace(portfolio, templates=screened)


@dataclass(frozen=True)
class GlobalFitWizardScreeningTable:
    """Phase 1's product: a series alphabet and a complete score table for it.

    Two per-run mappings, deliberately distinct:

    ``single_fit_recommendations_by_run``
        the runs' own **single-run Fit Wizard** analyses — full family screening,
        peak analysis, damped-line scan, each at that run's own rebin factor.
        This is what a caller caches (and what the fit tabs show), and what a
        later series analysis reuses.
    ``recommendations_by_run``
        the **completed table**: every run scored against every alphabet
        template at one common ``series_rebin_factor``, so the information
        criteria of two runs, or of two templates, may be summed and compared.
        Derived, never a substitute for a run's own analysis.
    """

    portfolio: GlobalFitWizardCandidatePortfolio
    recommendations_by_run: dict[int, FitWizardRecommendation]
    single_fit_recommendations_by_run: dict[int, FitWizardRecommendation]
    generated_run_numbers: tuple[int, ...]
    series_rebin_factor: int


def build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
    datasets: list[MuonDataset],
    current_model: CompositeModel | None = None,
    *,
    existing_recommendations_by_run: dict[int, FitWizardRecommendation] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    effort_tier: EffortTier = DEFAULT_EFFORT_TIER,
    metric: SelectionMetric = SelectionMetric.AICC,
    instrumentation: dict[str, object] | None = None,
    stage_callback: Callable[[WizardStageProgress], None] | None = None,
) -> GlobalFitWizardScreeningTable:
    """Run phase 1 of the global fit wizard: per-run analysis, then completion.

    Two stages, and the first is the whole point:

    1. **Every run goes through the single-run Fit Wizard**
       (:func:`~asymmetry.core.fitting.fit_wizard.build_fit_wizard_recommendation`)
       — tiered family screening, peak analysis including the matched-apodisation
       damped-line scan, peak-seeded multiplet templates, null baselines and
       refinement. That is what finds the heavily damped pair describing only the
       low-temperature half of a series; a portfolio built from the *median*
       fingerprint cannot. Runs are processed serially in this process because
       each build opens its own worker pool, and pools must not nest. An existing
       recommendation is reused when it answers the same question — same scope,
       same user-declared frequencies, encoded by
       :func:`~asymmetry.core.fitting.fit_wizard.single_fit_build_signature` —
       which is the whole of the reuse rule; a table derived from some other
       portfolio carries no signature and is never reused.

    2. **Completion.** The series alphabet is the union of what those runs
       assessed (:func:`series_template_alphabet`), narrowed by the effort tier;
       the series search resolution is the smallest per-run rebin factor
       (:func:`series_rebin_factor`). Every (run, template) cell the run's own
       table does not already hold *at that factor* is fitted, warm-started and
       peak-seeded by :func:`_single_fit_completion_task`. The result is a
       rectangular table whose information criteria are mutually comparable.

    ``cancel_callback`` is polled between runs in stage 1 and, in stage 2's
    pooled path, *while draining* the submitted futures: a ``wait`` with a
    ``_PHASE_ONE_CANCEL_POLL_SECONDS`` timeout re-checks it each iteration, so a
    cancel is honoured within one poll interval instead of one in-flight table's
    duration (a fit already running in a worker still cannot be interrupted). A
    truthy callback raises :class:`FitCancelledError` and tears the pool down
    with :func:`terminate_spawn_pool` — non-blocking, and force-killing the
    workers so nothing is orphaned. This matters because phase 1 is a
    *minutes*-long job: a drain that only blocks on completion, followed by a
    ``shutdown(wait=True)``, is indistinguishable from a deadlock from the
    caller's side.

    ``existing_recommendations_by_run`` is *replaced in place* with the run's own
    single-run analyses (stage 1's product), never with the completed table, so a
    caller's cache keeps answers that stay reusable.
    """
    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Global fit wizard analysis cancelled.")
    progress_callback = _threadsafe_progress_callback(progress_callback)
    preview = build_global_fit_wizard_candidate_portfolio(
        datasets,
        current_model=current_model,
        scope=scope,
    )
    existing = (
        existing_recommendations_by_run if existing_recommendations_by_run is not None else {}
    )

    if preview.mixed_axes_warning:
        return GlobalFitWizardScreeningTable(
            portfolio=preview,
            recommendations_by_run=_sync_single_fit_recommendation_store(
                existing_recommendations_by_run, {}
            ),
            single_fit_recommendations_by_run={},
            generated_run_numbers=(),
            series_rebin_factor=1,
        )

    expected_signature = single_fit_build_signature(scope, user_frequencies_mhz)
    source_by_run: dict[int, FitWizardRecommendation] = {
        int(dataset.run_number): existing[int(dataset.run_number)]
        for dataset in preview.ordered_datasets
        if int(dataset.run_number) in existing
        and existing[int(dataset.run_number)].build_signature == expected_signature
    }
    missing_datasets = [
        dataset
        for dataset in preview.ordered_datasets
        if int(dataset.run_number) not in source_by_run
    ]
    generated_run_numbers: list[int] = []

    if missing_datasets:
        _progress_log(
            progress_callback,
            f"Running the single-run Fit Wizard on {len(missing_datasets)} dataset(s) "
            "to build the series candidate alphabet.",
        )
    with stage_timer(
        instrumentation,
        "screening.single_fit_tables",
        items_total=len(missing_datasets),
        stage_callback=stage_callback,
        message=f"Analysing {len(missing_datasets)} dataset(s) with the single-run Fit Wizard.",
    ) as advance:
        # Serial, deliberately: ``build_fit_wizard_recommendation`` opens its own
        # spawn pool for the Stage-1/Stage-2 fan-outs, and a pool inside a pool
        # oversubscribes the host while making the outer stage's progress
        # invisible. One run at a time, each using the whole machine.
        for dataset in missing_datasets:
            if cancel_callback is not None and cancel_callback():
                raise FitCancelledError("Global fit wizard analysis cancelled.")
            _progress_log(
                progress_callback,
                f"Single-fit analysis {dataset.run_label}: screening candidate families.",
            )
            recommendation = build_fit_wizard_recommendation(
                dataset,
                current_model,
                metric=metric,
                scope=scope,
                user_frequencies_mhz=user_frequencies_mhz,
                cancel_callback=cancel_callback,
            )
            run_number = int(dataset.run_number)
            source_by_run[run_number] = recommendation
            generated_run_numbers.append(run_number)
            message = (
                f"Single-fit analysis {dataset.run_label}: done "
                f"({len(generated_run_numbers)}/{len(missing_datasets)})."
            )
            _progress_log(progress_callback, message)
            advance(len(generated_run_numbers), message)

    _sync_single_fit_recommendation_store(existing_recommendations_by_run, source_by_run)

    alphabet = _scope_filtered_templates(
        series_template_alphabet(source_by_run),
        preview.ordered_datasets,
        scope,
    )
    portfolio = replace(
        preview,
        templates=alphabet,
        pattern_template_keys=_series_pattern_template_keys(alphabet, source_by_run),
    )
    portfolio = _apply_screening_effort_tier(
        portfolio,
        effort_tier,
        progress_callback=progress_callback,
        instrumentation=instrumentation,
    )
    rebin_factor = series_rebin_factor(preview.ordered_datasets, source_by_run)
    _set_metric(instrumentation, "alphabet_size", len(portfolio.templates))
    _set_metric(instrumentation, "series_rebin_factor", rebin_factor)

    if not portfolio.templates:
        _set_metric(instrumentation, "completion_fits", 0)
        return GlobalFitWizardScreeningTable(
            portfolio=portfolio,
            recommendations_by_run={},
            single_fit_recommendations_by_run=source_by_run,
            generated_run_numbers=tuple(generated_run_numbers),
            series_rebin_factor=rebin_factor,
        )

    _progress_log(
        progress_callback,
        f"Series alphabet: {len(portfolio.templates)} candidate(s) at rebin factor "
        f"x{rebin_factor}; completing the per-run score table.",
    )
    completed_by_run, completion_fits = _complete_single_fit_table(
        preview.ordered_datasets,
        portfolio.templates,
        source_by_run,
        rebin_factor=rebin_factor,
        metric=metric,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        instrumentation=instrumentation,
        stage_callback=stage_callback,
    )
    _set_metric(instrumentation, "completion_fits", completion_fits)
    return GlobalFitWizardScreeningTable(
        portfolio=portfolio,
        recommendations_by_run=completed_by_run,
        single_fit_recommendations_by_run=source_by_run,
        generated_run_numbers=tuple(generated_run_numbers),
        series_rebin_factor=rebin_factor,
    )


def _complete_single_fit_table(
    ordered_datasets: Sequence[MuonDataset],
    templates: tuple[CandidateTemplate, ...],
    source_by_run: dict[int, FitWizardRecommendation],
    *,
    rebin_factor: int,
    metric: SelectionMetric,
    progress_callback: Callable[[str], None] | None,
    cancel_callback: Callable[[], bool] | None,
    instrumentation: dict[str, object] | None,
    stage_callback: Callable[[WizardStageProgress], None] | None,
) -> tuple[dict[int, FitWizardRecommendation], int]:
    """Score every run against every alphabet template at one rebin factor.

    Each run is one independent task, so this fans out over the spawn pool; the
    per-task fits are serial, so no pool nests inside another.
    """
    sibling_values = _sibling_warm_starts(templates, source_by_run, metric)
    completed_by_run: dict[int, FitWizardRecommendation] = {}
    completion_fits = 0

    def _task_args(dataset: MuonDataset) -> tuple[object, ...]:
        return (
            dataset,
            templates,
            source_by_run[int(dataset.run_number)],
            rebin_factor,
            metric,
            sibling_values,
        )

    with stage_timer(
        instrumentation,
        "screening.completion_fits",
        items_total=len(ordered_datasets),
        stage_callback=stage_callback,
        message=(
            f"Completing the per-run score table for {len(ordered_datasets)} dataset(s) "
            f"against {len(templates)} candidate(s)."
        ),
    ) as advance:
        worker_count = _single_fit_table_worker_count(len(ordered_datasets))
        executor = (
            _try_open_process_pool(
                max_workers=worker_count,
                progress_callback=progress_callback,
                activity="Phase-1 completion fits",
            )
            if worker_count > 1
            else None
        )
        if executor is None:
            for dataset in ordered_datasets:
                if cancel_callback is not None and cancel_callback():
                    raise FitCancelledError("Global fit wizard analysis cancelled.")
                run_number, recommendation, fits = _single_fit_completion_task(*_task_args(dataset))
                completed_by_run[run_number] = recommendation
                completion_fits += fits
                message = (
                    f"Score table {dataset.run_label}: done "
                    f"({len(completed_by_run)}/{len(ordered_datasets)})."
                )
                _progress_log(progress_callback, message)
                advance(len(completed_by_run), message)
            return completed_by_run, completion_fits

        try:
            future_to_dataset = {
                executor.submit(_single_fit_completion_task, *_task_args(dataset)): dataset
                for dataset in ordered_datasets
            }
            pending = set(future_to_dataset)
            total = len(pending)
            while pending:
                if cancel_callback is not None and cancel_callback():
                    raise FitCancelledError("Global fit wizard analysis cancelled.")
                done, pending = wait(
                    pending,
                    timeout=_PHASE_ONE_CANCEL_POLL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    run_number, recommendation, fits = future.result()
                    completed_by_run[run_number] = recommendation
                    completion_fits += fits
                    message = (
                        f"Score table {future_to_dataset[future].run_label}: "
                        f"done ({len(completed_by_run)}/{total})."
                    )
                    _progress_log(progress_callback, message)
                    advance(len(completed_by_run), message)
        except BaseException:
            # Cancel, a worker crash, or a Ctrl-C: never wait on the fits still
            # in flight. Drop queued work and force-kill the workers, so the
            # caller gets control back at once and nothing is orphaned.
            terminate_spawn_pool(executor)
            raise
        else:
            _shutdown_process_pool(executor)
    return completed_by_run, completion_fits


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
    cancel_callback: Callable[[], bool] | None = None,
    effort_tier: EffortTier = DEFAULT_EFFORT_TIER,
    portfolio: GlobalFitWizardCandidatePortfolio | None = None,
    instrumentation: dict[str, object] | None = None,
    stage_callback: Callable[[WizardStageProgress], None] | None = None,
) -> GlobalFitWizardRecommendation:
    """Build the ranking table from per-run single-fit wizard results only.

    Phase 1 comes first and the portfolio comes out of it: every run is analysed
    by the single-run Fit Wizard, the series alphabet is the union of what those
    runs assessed, and every run is then scored against every alphabet template
    at one series rebin factor
    (:func:`build_or_complete_single_fit_wizard_recommendations_for_global_portfolio`).
    Pass ``portfolio`` together with a ``single_fit_recommendations_by_run`` that
    already covers it — the pair a previous phase-1 call returned — to skip that
    work; anything less and phase 1 runs here.

    ``effort_tier`` narrows the alphabet before the completion fits (see
    :func:`screening_templates_for_effort_tier`): Low and Balanced score a
    trimmed candidate set, while Thorough and Exhaustive — the default — score
    all of it. Templates a run's peak search positively identified are never
    trimmed. The skipped candidates are announced through ``progress_callback``
    and recorded in ``instrumentation``.

    ``instrumentation`` additionally receives the standard timing block
    (:mod:`asymmetry.core.fitting.wizard_timing`): wall-clock, CPU seconds
    including reaped pool workers, and a per-stage breakdown. ``stage_callback``
    receives :class:`~asymmetry.core.fitting.wizard_timing.WizardStageProgress`
    events as each stage starts, advances and ends, which is what a caller needs
    to time out on *lack of progress* rather than on total runtime.
    """
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Global fit wizard analysis cancelled.")
    progress_callback = _threadsafe_progress_callback(progress_callback)
    current_parameter_types = current_parameter_types or {}
    current_values = current_values or {}
    parameter_bounds = parameter_bounds or {}

    if portfolio is None:
        portfolio = build_global_fit_wizard_candidate_portfolio(
            datasets,
            current_model=current_model,
            scope=scope,
            single_fit_recommendations_by_run=single_fit_recommendations_by_run,
        )
        portfolio = _apply_screening_effort_tier(
            portfolio,
            effort_tier,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )
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
    if not single_fit_table_covers_portfolio(
        portfolio.ordered_datasets, portfolio.templates, recommendations_by_run
    ):
        _progress_log(
            progress_callback,
            "Preparing missing single-fit wizard tables for global screening.",
        )
        table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
            list(portfolio.ordered_datasets),
            current_model=current_model,
            existing_recommendations_by_run=recommendations_by_run,
            progress_callback=progress_callback,
            scope=scope,
            user_frequencies_mhz=user_frequencies_mhz,
            cancel_callback=cancel_callback,
            effort_tier=effort_tier,
            metric=metric,
            instrumentation=instrumentation,
            stage_callback=stage_callback,
        )
        portfolio = table.portfolio
        recommendations_by_run = table.recommendations_by_run
    templates = list(portfolio.templates)

    with stage_timer(
        instrumentation,
        "screening.aggregate_assessments",
        items_total=len(templates),
        stage_callback=stage_callback,
        message=f"Aggregating single-fit screening scores for {len(templates)} candidate(s).",
    ):
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
        
            # The completed table already tried every cell with sibling warm
            # starts (phase 1); the serial repair pass would only repeat that.
            repair_partial_incomplete=False,
        )

    partition_started = time.monotonic()
    with stage_timer(
        instrumentation,
        "screening.partition_path",
        stage_callback=stage_callback,
        message="Looking for structural transitions along the series.",
    ):
        partition_path = build_series_partition_path(
            portfolio.ordered_datasets,
            assessments_by_key,
            axis_key=portfolio.series_axis_key,
            analysed_points_by_run={
                int(run_number): int(recommendation.analysed_points)
                for run_number, recommendation in recommendations_by_run.items()
            },
            family_by_template=series_template_families(recommendations_by_run),
        )
    _set_metric(instrumentation, "partition_seconds", time.monotonic() - partition_started)
    if partition_path is not None:
        _set_metric(instrumentation, "partition_selected_k", partition_path.selected_k)
        _set_metric(
            instrumentation,
            "partition_gains",
            [float(solution.gain) for solution in partition_path.solutions],
        )
        selected = partition_path.solutions[partition_path.selected_k]
        _progress_log(
            progress_callback,
            transitions_summary(selected, portfolio.series_axis_label),
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
            partition_path=partition_path,
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

    fixed_names_by_key = {
        template.key: _fixed_param_names(template, current_parameter_types)
        for template in templates
    }
    seeds_by_key = {
        template.key: _single_fit_assessment_by_run(
            single_fit_recommendations_by_run,
            template.key,
        )
        for template in templates
    }
    if repair_partial_incomplete:
        # NOTE (measured, deliberately still serial): repair is a *fit* per failed
        # (template, run) pair and is the largest remaining serial section of the
        # screening stage — 22 s of one-core time on a 14-dataset synthetic series
        # against 3.5 s for the whole parallel phase-1 stage. Fanning it out over
        # a spawn pool was tried and reverted: the repair path is the one place a
        # caller can inject a fit engine and observe per-template progress, and a
        # process boundary silently breaks both for a stage whose visible cost is
        # now bounded by the effort tier anyway. The stage timer below reports
        # what it costs, so the trade is measurable rather than assumed.
        for template in templates:
            seeds_by_key[template.key] = _repair_partial_single_fit_prescreen_assessments(
                datasets,
                fingerprints_by_run,
                template,
                assessments_by_run=seeds_by_key[template.key],
                current_values=current_values,
                parameter_bounds=parameter_bounds,
                fixed_param_names=fixed_names_by_key[template.key],
                metric=metric,
                fit_engine=fit_engine,
                progress_callback=progress_callback,
            )
        for template in templates:
            _merge_repaired_assessments_into_single_fit_recommendations(
                single_fit_recommendations_by_run,
                template.key,
                seeds_by_key[template.key],
            )

    for template in templates:
        fixed_param_names = fixed_names_by_key[template.key]
        seed_assessments_by_run = seeds_by_key[template.key]
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


#: Structural breaks are scored with BIC whatever ranking metric the user chose.
#:
#: A transition is a claim about *structure*, and BIC is the consistent structure
#: selector. Measured on a real 29-run zero-field scan: two heavily damped lines
#: plus a relaxation nests plain relaxation, so under AIC the extra parameters
#: cost 2 apiece per run and a genuine structural change was worth almost nothing
#: at tier 1 — the elbow never appeared. Under BIC the same change scored ~100 per
#: break (``ln n ≈ 10`` per parameter there) and the path had a clean elbow. The
#: user's metric still decides which candidate wins *within* a phase; it does not
#: decide where the phases are.
_PARTITION_METRIC = SelectionMetric.BIC


def _partition_inputs_from_prescreen(
    ordered_datasets: Sequence[MuonDataset],
    prescreen_assessments: Mapping[str, GlobalCandidateAssessment],
    *,
    analysed_points_by_run: Mapping[int, int],
) -> tuple[dict[int, dict[str, float]], dict[str, dict[int, RunEstimate]], int]:
    """The per-run BIC table and per-run estimates the partition search runs on.

    **A cell is feasible when the fit converged** — ``FitResult.success`` — and
    nothing more. The single-run wizard's own disqualifiers ("amplitude consistent
    with zero", "rate unresolved") are deliberately *not* consulted: treating a
    disqualified cell as infeasible makes the template infeasible on every segment
    containing that run, which on a real series forced breaks around single weak
    runs *inside* a phase. Handing the IC the run's real χ² instead lets the
    partition cost decide, which is what it is for.

    Per-run costs are ``χ² + k·ln(n_run)`` at the series search resolution, with
    ``n_run`` the points that run was actually fitted over
    (:attr:`FitWizardRecommendation.analysed_points`), so tier 1's sums and tier
    2's segment scores are on one scale.
    """

    table: dict[int, dict[str, float]] = {
        int(dataset.run_number): {} for dataset in ordered_datasets
    }
    estimates_by_template: dict[str, dict[int, RunEstimate]] = {}

    for template_key, assessment in prescreen_assessments.items():
        free_names = tuple(
            name
            for name in assessment.template.model.param_names
            if name not in assessment.fixed_param_names
        )
        if not free_names:
            continue
        per_run: dict[int, RunEstimate] = {}
        for dataset in ordered_datasets:
            run_number = int(dataset.run_number)
            result = assessment.fit_results_by_run.get(run_number)
            if result is None or not result.success:
                continue
            sample_count = int(analysed_points_by_run[run_number])
            table[run_number][template_key] = float(result.chi_squared) + surrogate_metric_penalty(
                len(free_names), sample_count=sample_count, metric=_PARTITION_METRIC
            )
            per_run[run_number] = run_estimate_from_fit_result(
                result,
                free_names,
                run_number=run_number,
                n_points=sample_count,
                at_bound=_bound_hit_names(result.parameters),
            )
        if per_run:
            estimates_by_template[template_key] = per_run

    n_total_points = sum(
        int(analysed_points_by_run[int(dataset.run_number)]) for dataset in ordered_datasets
    )
    return table, estimates_by_template, n_total_points


def series_template_families(
    recommendations_by_run: Mapping[int, FitWizardRecommendation],
) -> dict[str, str]:
    """Map every template key the series knows to its family key.

    The union of the per-run family reports, plus every multiplet template
    (built from a run's detected lines rather than listed in a family report)
    under ``"oscillatory"``. A key absent from the result belongs to no known
    family and stands as its own structure.
    """

    families: dict[str, str] = {}
    for recommendation in recommendations_by_run.values():
        families.update(_template_family_map(recommendation.family_reports))
        for template in recommendation.templates:
            if is_multiplet_template_key(template.key):
                families[template.key] = "oscillatory"
    return families


def build_series_partition_path(
    ordered_datasets: Sequence[MuonDataset],
    prescreen_assessments: Mapping[str, GlobalCandidateAssessment],
    *,
    axis_key: str,
    analysed_points_by_run: Mapping[int, int],
    family_by_template: Mapping[str, str],
    config: PartitionConfig = PartitionConfig(),
) -> PartitionPath | None:
    """Where the series breaks, read off the completed phase-1 table.

    Closed form and cheap — no fit happens here — so a screening pass always
    computes it. ``None`` when the series cannot be partitioned at all: a series
    shorter than two minimum segments has no admissible break to look for, and an
    empty candidate alphabet has nothing to score.

    ``analysed_points_by_run`` gives every ordered run's fitted point count at the
    series search resolution. A non-empty ``prescreen_assessments`` implies a
    completed per-run table, which is exactly when that mapping is total.

    ``family_by_template`` names each template's family
    (:func:`series_template_families`); a break is a change of *family*
    between adjacent phases. Which template within the family, and which
    parameters it shares, are priced into the segment cost but are not breaks
    — a two-line and a one-line damped oscillation are the same ordered phase,
    and a background that becomes shareable is not a transition. A template
    absent from the map is its own family.
    """

    order = [int(dataset.run_number) for dataset in ordered_datasets]
    if len(order) < 2 * config.min_segment or not prescreen_assessments:
        return None

    table, estimates_by_template, n_total_points = _partition_inputs_from_prescreen(
        ordered_datasets,
        prescreen_assessments,
        analysed_points_by_run=analysed_points_by_run,
    )
    if not any(table[run] for run in order):
        return None

    axis_values = {
        int(dataset.run_number): _axis_value(dataset, axis_key) for dataset in ordered_datasets
    }
    def family_of(template_key: str) -> str:
        return family_by_template.get(template_key, template_key)

    return partition_series(
        order,
        axis_values,
        tier2_segment_cost(
            table, order, estimates_by_template, _PARTITION_METRIC, family_of=family_of
        ),
        config,
        n_total_points=n_total_points,
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
    search_engine: str | None = None,
    effort_tier: EffortTier = DEFAULT_EFFORT_TIER,
    portfolio: GlobalFitWizardCandidatePortfolio | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    partition_path: PartitionPath | None = None,
    partition_k: int | None = None,
) -> GlobalFitWizardRecommendation:
    """Analyze one ordered dataset series and recommend a global-fit candidate.

    ``portfolio`` is the candidate set to search, normally the series alphabet a
    phase-1 call returned together with the ``single_fit_recommendations_by_run``
    table that covers it; without it the cheap preview portfolio is used and the
    per-run pre-screen is only taken up when the supplied table happens to cover
    that.

    ``effort_tier`` is the user-facing effort slider: ``LOW``, ``BALANCED``,
    ``THOROUGH``, ``EXHAUSTIVE`` (default ``EXHAUSTIVE``). **Every tier resolves
    to the separable role-search engine** (see ``_EFFORT_TIER_SEARCH_ENGINE``):
    the separable search takes the all-local assignment straight from the per-run
    fits, ranks every sharing pattern with a full-covariance GLS surrogate, and
    walks backward elimination with one warm profiled fit per step, so it costs
    O(P) coupled fits per template where the wavefront costs O(2^P) — and it is
    the honest answer at every tier, not a coarser one. The enum and its payload
    are retained so a future scope-based quick-look tier can be added without a
    schema/UI change.

    ``search_engine`` remains available as a lower-level override for existing
    callers/tests — when given explicitly it takes precedence over
    ``effort_tier`` for engine selection. ``"separable"`` is the default engine;
    ``"exhaustive"`` and ``"thorough"`` run the exact bounded wavefront
    (byte-for-byte the frozen-baseline path, and the harness referee the
    separable engine is measured against); ``"low"`` and ``"balanced"`` run the
    retained non-exhaustive heuristic engines (techniques E/F/G/H and the I/J/K
    knobs), reachable only through this seam for large-P use and regression
    coverage — never through ``effort_tier``.

    ``partition_k`` switches the whole run from **one series-wide answer** to
    **one answer per phase**: solution ``k`` of ``partition_path`` (the path a
    screening pass computed and stored on its recommendation) is optimised
    segment by segment, together with the neighbours tier 3 verifies, and the
    result carries ``phase_assessments`` and a re-scored path instead of a
    ``recommended_key``. The two arrive together — a ``k`` names a row of a
    specific path — so passing one without the other is a programming error and
    is refused. With ``partition_k=None`` nothing about this function changes.
    """
    if (partition_k is None) != (partition_path is None):
        raise ValueError(
            "partition_k and partition_path go together: a partition index names a "
            "row of a particular path."
        )
    if partition_path is not None and not 0 <= partition_k < len(partition_path.solutions):
        raise ValueError(
            f"partition_k={partition_k} is outside the path's "
            f"{len(partition_path.solutions)} solution(s)."
        )
    _set_metric(instrumentation, "strategy", "consolidated")
    if instrumentation is not None:
        instrumentation.setdefault("counters", {})
        instrumentation.setdefault("staged_frontier_widths", [])
        instrumentation.setdefault("relaxed_penalties", [])
        instrumentation.setdefault("curvature_hint_sizes", [])
        instrumentation.setdefault("minuit_edm", [])
    resolved_engine = (
        search_engine if search_engine is not None else _EFFORT_TIER_SEARCH_ENGINE[effort_tier]
    )
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
        search_engine=resolved_engine,
        portfolio=portfolio,
        cancel_callback=cancel_callback,
        partition_path=partition_path,
        partition_k=partition_k,
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
    search_engine: str = _DEFAULT_SEARCH_ENGINE,
    portfolio: GlobalFitWizardCandidatePortfolio | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    partition_path: PartitionPath | None = None,
    partition_k: int | None = None,
) -> GlobalFitWizardRecommendation:
    if len(datasets) < 2:
        raise ValueError("Global fit wizard requires at least two datasets.")

    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Global fit wizard analysis cancelled.")
    search_engine = search_engine or _DEFAULT_SEARCH_ENGINE
    if search_engine not in SEARCH_ENGINES:
        raise ValueError(
            f"Unknown search_engine {search_engine!r}; valid options are "
            f"{', '.join(SEARCH_ENGINES)}."
        )
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
    if portfolio is None:
        portfolio = build_global_fit_wizard_candidate_portfolio(
            datasets,
            current_model=current_model,
            scope=scope,
            single_fit_recommendations_by_run=available_single_fit_recommendations,
        )
    ordered_datasets = list(portfolio.ordered_datasets)
    axis_key = portfolio.series_axis_key
    axis_label = portfolio.series_axis_label
    mixed_axes_warning = portfolio.mixed_axes_warning
    fingerprints_by_run = portfolio.fingerprints_by_run
    aggregate_fingerprint = _aggregate_fingerprints(
        [fingerprints_by_run[int(dataset.run_number)] for dataset in ordered_datasets]
    )
    pattern_template_keys = portfolio.pattern_template_keys
    templates = list(portfolio.templates)
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

    # Coverage, not identity: the per-run table has to *hold a score for* every
    # candidate this search will rank, and nothing more. Demanding that its
    # template list match the portfolio's exactly rejected every genuine
    # single-run analysis, which carries the run's own fuller candidate set.
    use_single_fit_prescreen = single_fit_table_covers_portfolio(
        ordered_datasets, templates, available_single_fit_recommendations
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
            # A supplied completed table (``portfolio`` given) already retried
            # every failed cell from a sibling in phase 1; the serial repair pass
            # is only for a bare per-run table handed in without one.
            repair_partial_incomplete=portfolio is None and not normalized_selected_template_keys,
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
    if partition_k is not None:
        # One answer per phase, not one for the series. The pre-screen table is
        # what the path was computed from, so it is also what each segment's
        # search starts from — at the series search resolution, where those
        # per-run fits *are* the segment's all-local anchor and cost no fit.
        search_rebin_factor = series_rebin_factor(
            ordered_datasets, available_single_fit_recommendations
        )
        analysed_points_by_run = {
            int(run_number): int(recommendation.analysed_points)
            for run_number, recommendation in available_single_fit_recommendations.items()
        }
        updated_path, phase_assessments, recommended_partition_k = _optimise_partition_phases(
            ordered_datasets,
            path=partition_path,
            partition_k=partition_k,
            templates=templates,
            template_contexts=template_contexts,
            prescreen_assessments=initial_assessments,
            analysed_points_by_run=analysed_points_by_run,
            axis_key=axis_key,
            metric=metric,
            search_rebin_factor=search_rebin_factor,
            progress_callback=progress_callback,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
            single_run_prefit_cache_for=_single_run_prefit_cache_for,
            cancel_callback=cancel_callback,
        )
        return rerank_global_fit_wizard_recommendation(
            GlobalFitWizardRecommendation(
                series_axis_key=axis_key,
                series_axis_label=axis_label,
                mixed_axes_warning=mixed_axes_warning,
                fingerprints_by_run=fingerprints_by_run,
                dataset_order=tuple(int(dataset.run_number) for dataset in ordered_datasets),
                templates=tuple(templates),
                assessments=tuple(
                    initial_assessments[template.key]
                    for template in prescreen_templates
                    if template.key in initial_assessments
                ),
                metric=metric,
                recommended_key=None,
                comparable_keys=(),
                summary="",
                partition_path=updated_path,
                phase_assessments=phase_assessments,
                recommended_partition_k=recommended_partition_k,
            ),
            metric,
        )

    if normalized_selected_template_keys:
        if search_engine == SEARCH_ENGINE_LOW:
            # Technique I/J still apply within an explicit user selection on the
            # retained Low heuristic engine: "screening-grade" means the
            # cap/complexity-prior/demotion narrow *what gets the expensive
            # coupled search*, not just the auto-shortlist path. Every exact
            # engine (which is what every user-facing tier now resolves to)
            # honours the user's selection verbatim.
            selected_templates = tuple(
                template_by_key[key] for key in normalized_selected_template_keys
            )
            shortlist_keys = _shortlist_template_keys(
                selected_templates,
                initial_assessments=initial_assessments,
                metric=metric,
                search_engine=search_engine,
                progress_callback=progress_callback,
            )
        else:
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
            search_engine=search_engine,
            progress_callback=progress_callback,
        )
    shortlisted_templates = [template for template in templates if template.key in shortlist_keys]
    if shortlisted_templates:
        _progress_log(
            progress_callback,
            "Coupled global optimisation will evaluate "
            f"{len(shortlisted_templates)} candidate(s) "
            f"via the {search_engine} global/local role search.",
        )
    _set_metric(instrumentation, "search_engine", search_engine)
    if search_engine == SEARCH_ENGINE_SEPARABLE:
        optimized_assessments = _run_separable_search(
            ordered_datasets,
            shortlisted_templates=shortlisted_templates,
            template_contexts=template_contexts,
            prescreen_assessments=initial_assessments,
            axis_key=axis_key,
            metric=metric,
            progress_callback=progress_callback,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
            single_run_prefit_cache_for=_single_run_prefit_cache_for,
            cancel_callback=cancel_callback,
        )
    elif search_engine in _EXACT_SEARCH_ENGINES:
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
            cancel_callback=cancel_callback,
        )
    else:
        optimized_assessments = _run_heuristic_search(
            ordered_datasets,
            shortlisted_templates=shortlisted_templates,
            template_contexts=template_contexts,
            axis_key=axis_key,
            metric=metric,
            progress_callback=progress_callback,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
            single_run_prefit_cache_for=_single_run_prefit_cache_for,
            engine=search_engine,
            aggregate_fingerprint=aggregate_fingerprint,
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


def _screening_no_recommendation_summary(
    recommendation: GlobalFitWizardRecommendation,
) -> str:
    """Say *why* screening produced no recommendation, not merely that it did.

    A screening-only pass *always* returns ``recommended_key=None``: a
    pre-screen assessment is ``prescreen_only``, hence never ``is_successful``,
    because independent per-dataset fits are not evidence about a coupled global
    fit. The caller is meant to read the ranked table and select candidates for
    optimisation. That is by design — but the previous single generic sentence
    said the same thing whether the table held a clean ranking of every
    candidate or nothing scoreable at all, which let a *failed* screen (no
    candidate could be scored, because per-run single-fit tables were missing or
    failed) pass for an ordinary one. A caller reading ``recommended_key``, as
    the natural reading of "recommendation" invites, saw ``None`` either way and
    silently proceeded with an empty selection.

    So the two cases now name themselves, with counts, the top-ranked key to
    select, and the underlying per-candidate reasons when there is no ranking.
    """
    assessments = tuple(recommendation.assessments)
    if not assessments:
        return (
            "Screening produced no candidate assessments at all. This is a failure, "
            "not a ranking: the candidate portfolio was empty for this series, so "
            "there was nothing to score."
        )
    scored = sorted(
        (
            assessment
            for assessment in assessments
            if np.isfinite(assessment.metric_value(recommendation.metric))
        ),
        key=lambda assessment: assessment.metric_value(recommendation.metric),
    )
    if scored:
        return (
            f"Single-fit screening complete: {len(scored)} of {len(assessments)} "
            "candidates scored, best-ranked "
            f"'{scored[0].template.key}'. These scores come from independent "
            "per-dataset fits only and have not yet been optimized for coupled "
            "global fitting, so no candidate is recommended yet — select one or "
            "more from the ranked screening table to continue."
        )
    reasons: list[str] = []
    for assessment in assessments:
        for warning in assessment.series_warnings:
            if warning not in reasons:
                reasons.append(warning)
    detail = (" Reported causes: " + " ".join(reasons[:5])) if reasons else ""
    return (
        f"Screening scored none of its {len(assessments)} candidates: every one is "
        "missing a successful single-fit assessment for at least one run, so the "
        "table carries no usable ranking. Treat this as a failed screen rather than "
        "an absence of structure." + detail
    )


def _axis_unit(axis_label: str) -> str:
    """The unit out of an axis label — ``"Temperature (K)"`` → ``"K"``.

    ``"Run"`` carries no unit and yields ``""``, so a boundary on a run-ordered
    series reads as a bare number rather than inventing one.
    """

    opening = axis_label.find("(")
    closing = axis_label.rfind(")")
    if opening < 0 or closing < opening:
        return ""
    return axis_label[opening + 1 : closing].strip()


def _format_boundary(estimate: float, half_gap: float, unit: str) -> str:
    """``"16.5 ± 0.5 K"`` — the boundary estimate as the card states it."""

    text = f"{estimate:.4g} ± {half_gap:.2g}"
    return f"{text} {unit}" if unit else text


def format_transition_boundaries(
    solution: PartitionSolution,
    axis_label: str,
) -> str:
    """The solution's boundary estimates as one phrase, or ``""`` for no break.

    ``"16.5 ± 0.5 K and 28.5 ± 0.5 K"`` — the same phrase
    :func:`transitions_summary` embeds in its sentence, so the wizard's
    Transitions table can tabulate the boundaries without re-deriving the
    formatting (or the unit) from the axis label itself.
    """

    unit = _axis_unit(axis_label)
    formatted = [
        _format_boundary(estimate, half_gap, unit) for estimate, half_gap in solution.boundaries
    ]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + f" and {formatted[-1]}"


def transitions_summary(
    solution: PartitionSolution,
    axis_label: str,
) -> str:
    """Plain-language summary of a partition solution's transitions.

    ``"2 transitions found: 16.5 ± 0.5 K and 28.5 ± 0.5 K"``. A solution with no
    breaks says so; an excluded end stub is named separately, because "excluded
    from the global fit" is the one part of the answer a reader must not miss.
    """

    excluded = [segment for segment in solution.segments if segment.excluded]
    excluded_note = ""
    if excluded:
        runs = [run for segment in excluded for run in segment.run_numbers]
        labels = ", ".join(str(run) for run in runs)
        excluded_note = (
            f" Run {labels} is excluded from the global fit: it looks like a different phase."
            if len(runs) == 1
            else f" Runs {labels} are excluded from the global fit: they look like a"
            " different phase."
        )

    if not solution.boundaries:
        return f"No transitions found: one phase describes the whole series.{excluded_note}"

    listed = format_transition_boundaries(solution, axis_label)
    count = len(solution.boundaries)
    noun = "transition" if count == 1 else "transitions"
    return f"{count} {noun} found: {listed}.{excluded_note}"


def rerank_global_fit_wizard_recommendation(
    recommendation: GlobalFitWizardRecommendation,
    metric: SelectionMetric,
) -> GlobalFitWizardRecommendation:
    """Reuse existing global-fit assessments and recompute the recommendation.

    A **partitioned** recommendation — one whose ``recommended_partition_k`` names
    an optimised solution — keeps that partition and summarises its transitions
    instead of the series-wide winner. The partition itself is not re-selected:
    breaks are scored with BIC whatever ranking metric the user picked (see
    :func:`build_global_fit_wizard_screening_recommendation`), so changing the
    metric changes which candidate wins *within* a phase, never where the phases
    are.
    """
    if recommendation.recommended_partition is not None:
        return replace(
            recommendation,
            metric=metric,
            summary=transitions_summary(
                recommendation.recommended_partition, recommendation.series_axis_label
            ),
        )
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
    tentative = False
    if not passing:
        # Recommend-with-caveat: when no candidate passes strictly but the fit is
        # demonstrably excellent -- every run clears its per-run residual gate --
        # a heuristic *series-consistency* warning (a fingerprint jump across a
        # transition, a rough local-parameter trace) should not hard-veto to None.
        # Surface the best such candidate as a tentative recommendation with the
        # series_warnings as a caveat instead. (Per-run gate failures still block:
        # those mean the model genuinely does not fit some runs.)
        run_gated = [
            assessment
            for assessment in recommendation.assessments
            if assessment.is_successful
            and assessment.run_diagnostics
            and all(diagnostic.gate_passed for diagnostic in assessment.run_diagnostics)
        ]
        if run_gated:
            passing = run_gated
            tentative = True
    if not passing:
        optimized_assessments = recommendation.optimized_assessments()
        if not optimized_assessments:
            return replace(
                recommendation,
                metric=metric,
                recommended_key=None,
                comparable_keys=(),
                summary=_screening_no_recommendation_summary(recommendation),
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

    # At an exact tie the searched assignment wins over the screening row that
    # found the same minimum. They are the same fit with the same score, but only
    # the searched one carries ``parameter_recommendations`` — the per-parameter
    # role justification read from its flip-neighbourhood — so preferring it costs
    # the reader nothing and hands them the reasoning. Which of the two landed
    # first in ``assessments`` is an accident of engine ordering, and a
    # last-float-bit coin flip is not a basis for dropping the justification.
    passing_sorted = sorted(
        passing,
        key=lambda assessment: (
            _assessment_sort_key(assessment, metric),
            0 if assessment.assessment_key else 1,
        ),
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
    if tentative and primary.series_warnings:
        summary = (
            f"Recommended (tentative): {primary.template.title} by {metric.value}"
            f"{compare_summary} The coupled fit is strong (every run passes the residual "
            f"gate), but a series-consistency check flagged: "
            f"{' '.join(primary.series_warnings)} Review before applying."
        )
    else:
        summary = (
            f"Recommended globally optimized candidate: {primary.template.title} "
            f"by {metric.value}{compare_summary}"
        )
    return replace(
        recommendation,
        metric=metric,
        recommended_key=primary.selection_key,
        comparable_keys=comparable_keys,
        summary=summary,
    )


def merge_global_fit_wizard_recommendations(
    base: GlobalFitWizardRecommendation,
    updates: GlobalFitWizardRecommendation,
) -> GlobalFitWizardRecommendation:
    """Merge optimized assessments from one run back into an existing workflow snapshot.

    Phase assessments merge by ``(k, segment_index)`` — a later optimisation of a
    different ``k`` adds its phases beside the ones already there rather than
    replacing them. The **path** is not merged: it is replaced whole whenever the
    update carries one, because an optimisation pass re-scores its rows with exact
    per-segment ICs and a half-exact, half-surrogate path is not a path anybody can
    read a gain off.
    """
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
    merged_phase_assessments = dict(base.phase_assessments)
    merged_phase_assessments.update(updates.phase_assessments)
    merged = replace(
        base,
        metric=updates.metric,
        assessments=tuple(merged_assessments),
        partition_path=(
            updates.partition_path if updates.partition_path is not None else base.partition_path
        ),
        phase_assessments=merged_phase_assessments,
        recommended_partition_k=(
            updates.recommended_partition_k
            if updates.recommended_partition_k is not None
            else base.recommended_partition_k
        ),
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
    *,
    compact: bool = False,
) -> dict[str, object]:
    """Return a JSON-serialisable snapshot of a global-fit wizard recommendation.

    ``compact=True`` is the persistence form: every stored curve is strided
    down to :data:`~asymmetry.core.fitting.fit_wizard.PERSISTED_CURVE_MAX_POINTS`
    points, exactly as for the single-fit recommendation (this payload holds a
    fitted curve *per run per candidate*, so it grows fastest of the two).
    Both forms deserialise through
    :func:`deserialize_global_fit_wizard_recommendation`.
    """
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
            _serialize_global_candidate_assessment(assessment, compact=compact)
            for assessment in recommendation.assessments
        ],
        "metric": recommendation.metric.value,
        "recommended_key": recommendation.recommended_key,
        "comparable_keys": list(recommendation.comparable_keys),
        "summary": recommendation.summary,
        "partition_path": (
            None
            if recommendation.partition_path is None
            else recommendation.partition_path.to_payload()
        ),
        # A list of entries rather than a mapping: the key is the pair
        # ``(k, segment_index)``, and JSON object keys are strings.
        "phase_assessments": [
            {
                "k": int(k),
                "segment_index": int(segment_index),
                "assessment": _serialize_global_candidate_assessment(assessment, compact=compact),
            }
            for (k, segment_index), assessment in sorted(recommendation.phase_assessments.items())
        ],
        "recommended_partition_k": recommendation.recommended_partition_k,
        # Marks the payload as curve-decimated (read by nothing —
        # deserialisation tolerates both shapes).
        "compact": bool(compact),
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
    partition_payload = payload.get("partition_path")
    partition_path = (
        PartitionPath.from_payload(partition_payload)
        if isinstance(partition_payload, dict)
        else None
    )
    phase_assessments = {
        (int(entry["k"]), int(entry["segment_index"])): assessment
        for entry in payload.get("phase_assessments", [])
        if (assessment := _deserialize_global_candidate_assessment(entry["assessment"])) is not None
    }
    recommended_partition_k = payload.get("recommended_partition_k")

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
        partition_path=partition_path,
        phase_assessments=phase_assessments,
        recommended_partition_k=(
            int(recommended_partition_k) if recommended_partition_k is not None else None
        ),
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
        "damped_line_frequency_mhz": fingerprint.damped_line_frequency_mhz,
        "damped_line_snr": fingerprint.damped_line_snr,
        "damped_line_crop_us": fingerprint.damped_line_crop_us,
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
            damped_line_frequency_mhz=float(payload.get("damped_line_frequency_mhz", 0.0)),
            damped_line_snr=float(payload.get("damped_line_snr", 0.0)),
            damped_line_crop_us=float(payload.get("damped_line_crop_us", 0.0)),
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
    *,
    stride: int = 1,
    compact: bool = False,
) -> dict[str, object]:
    time_axis, values = curve
    return {
        "time": _persisted_curve_list(time_axis, stride=stride, compact=compact),
        "values": _persisted_curve_list(values, stride=stride, compact=compact),
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
    *,
    stride: int = 1,
    compact: bool = False,
) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "values": _persisted_curve_list(values, stride=stride, compact=compact),
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
    *,
    compact: bool = False,
) -> dict[str, object]:
    # One stride per run: that run's fitted curve, its time axis and its
    # component curves share a grid, so they are sampled together.
    strides = {
        run_number: (_persisted_curve_stride(int(np.asarray(curve[0]).size)) if compact else 1)
        for run_number, curve in assessment.fitted_curves_by_run.items()
    }
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
            str(run_number): _serialize_curve_pair(
                curve, stride=strides.get(run_number, 1), compact=compact
            )
            for run_number, curve in assessment.fitted_curves_by_run.items()
        },
        "component_curves_by_run": {
            str(run_number): _serialize_component_curves(
                curves, stride=strides.get(run_number, 1), compact=compact
            )
            for run_number, curves in assessment.component_curves_by_run.items()
        },
        "prescreen_only": bool(assessment.prescreen_only),
        "assessment_key": assessment.assessment_key,
    }


def _migrate_global_param_name_tuple(
    names: tuple[str, ...], model: CompositeModel
) -> tuple[str, ...]:
    """Rename/drop legacy entries in a cached parameter-role tuple.

    Applies the same rename maps as
    :func:`asymmetry.core.fitting.composite.migrate_legacy_fraction_parameter_set`
    and
    :func:`asymmetry.core.fitting.legacy_product_amplitudes.fold_legacy_product_amplitude_names`
    to a ``global_param_names``/``local_param_names``/``fixed_param_names`` tuple,
    preserving order and dropping names that have no parameter under the current
    schemes (a fraction group's derived last term, a folded-away product
    amplitude).
    """
    rename = _legacy_fraction_rename_map(model)
    migrated: list[str] = []
    for name in names:
        if name in rename:
            new_name = rename[name]
            if new_name is not None and new_name not in migrated:
                migrated.append(new_name)
        elif name not in migrated:
            migrated.append(name)
    return fold_legacy_product_amplitude_names(model, migrated)


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

    global_parameters, _ = fold_legacy_product_amplitude_set(
        template.model,
        migrate_legacy_fraction_parameter_set(
            template.model, _deserialize_parameter_set(payload.get("global_parameters", []))
        ),
        {},
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


#: A single relaxation envelope plus a background term (e.g. Exponential +
#: Constant) is the simplest additive shape the portfolio offers, so it is the
#: complexity prior's zero point — only additive terms *beyond* this baseline
#: draw a penalty. Every built-in template includes a background term.
_LOW_COMPLEXITY_PRIOR_BASELINE_ADDITIVE_TERMS = 2


def _low_complexity_prior_penalty(template: CandidateTemplate) -> float:
    """Technique I: extra Low-only ranking penalty per additive component.

    Applied purely at shortlist-ranking time (never stored on the assessment's
    ``aic``/``aicc``/``bic``, which the harness baseline and every other tier
    read), so a 3-4-additive-component template must beat a simpler one by more
    than this margin to be preferred at Low. Balanced/Thorough/Exhaustive see
    zero penalty — this must never move their shortlist.
    """
    extra_terms = max(0, template.additive_terms - _LOW_COMPLEXITY_PRIOR_BASELINE_ADDITIVE_TERMS)
    return _LOW_COMPLEXITY_PRIOR_PER_ADDITIVE_TERM * extra_terms


def _initial_assessment_is_identifiability_degenerate(
    assessment: GlobalCandidateAssessment,
) -> bool:
    """Technique J: does this template's initial fit look near-unidentifiable?

    True when any pair of free parameters is highly correlated (|rho| above
    :data:`_IDENTIFIABILITY_CORRELATION_THRESHOLD` in a per-run covariance) or
    any parameter's relative uncertainty spans
    :data:`_IDENTIFIABILITY_ERROR_DECADE_SPAN` decades from the tightest one —
    both are proxies for "this template's free parameters are not jointly
    constrained by the data", independent of which specific role split wins.
    Demotion only ever reorders the Low shortlist ranking; it never hard-drops
    a template (Balanced+ ignore this signal entirely).
    """
    relative_errors: list[float] = []
    for result in assessment.fit_results_by_run.values():
        covariance = result.covariance
        cov_names = result.covariance_parameters
        if covariance is not None and len(cov_names) >= 2:
            diag = np.diag(covariance)
            for i in range(len(cov_names)):
                for j in range(i + 1, len(cov_names)):
                    denom = diag[i] * diag[j]
                    if denom <= 0 or not math.isfinite(denom):
                        continue
                    rho = covariance[i, j] / math.sqrt(denom)
                    if abs(rho) >= _IDENTIFIABILITY_CORRELATION_THRESHOLD:
                        return True
        for name, sigma in result.uncertainties.items():
            if name not in result.parameters or not math.isfinite(sigma) or sigma <= 0:
                continue
            magnitude = abs(result.parameters[name].value)
            if magnitude <= 0:
                continue
            relative_errors.append(sigma / magnitude)
    if len(relative_errors) >= 2:
        finite = [error for error in relative_errors if error > 0 and math.isfinite(error)]
        if len(finite) >= 2:
            span = math.log10(max(finite)) - math.log10(min(finite))
            if span >= _IDENTIFIABILITY_ERROR_DECADE_SPAN:
                return True
    return False


def _shortlist_template_keys(
    templates: tuple[CandidateTemplate, ...],
    *,
    initial_assessments: dict[str, GlobalCandidateAssessment],
    metric: SelectionMetric,
    forced_keys: tuple[str, ...] = (),
    search_engine: str = _DEFAULT_SEARCH_ENGINE,
    progress_callback: Callable[[str], None] | None = None,
) -> set[str]:
    # Techniques I (portfolio cap), J (identifiability demotion) and the Low
    # complexity prior are gated on the *heuristic Low engine string*, never on
    # ``effort_tier``. Every user-facing tier now resolves to the exact engine
    # (see ``_EFFORT_TIER_SEARCH_ENGINE``), so these knobs are inert unless a
    # caller opts into the retained ``search_engine="low"`` seam directly.
    is_low = search_engine == SEARCH_ENGINE_LOW

    def _rank_key(template: CandidateTemplate) -> tuple[float, int, tuple]:
        assessment = initial_assessments[template.key]
        base = _assessment_sort_key(assessment, metric)
        if not is_low:
            return (0.0, 0, base)
        prior = _low_complexity_prior_penalty(template)
        demoted = 1 if _initial_assessment_is_identifiability_degenerate(assessment) else 0
        return (prior, demoted, base)

    eligible = list(templates)
    if is_low:
        # Technique I (Low portfolio cap): skip templates whose parameter count
        # exceeds the Low cap outright — a 3-4-additive-component model must earn
        # its place at Balanced+ instead of ever reaching Low's shortlist.
        capped = [
            template
            for template in eligible
            if template.parameter_count <= _LOW_MAX_TEMPLATE_PARAM_COUNT
        ]
        over_budget = [template for template in eligible if template not in capped]
        if over_budget and capped:
            _progress_log(
                progress_callback,
                f"Low effort: skipped {len(over_budget)} template(s) over the "
                f"P<={_LOW_MAX_TEMPLATE_PARAM_COUNT} cap ("
                + ", ".join(template.title for template in over_budget)
                + ") — they will not receive the coupled role search at this tier.",
            )
        # Never let the cap remove every candidate (a run with only "expensive"
        # templates in scope still needs a shortlist).
        eligible = capped or eligible

    ranked = sorted(eligible, key=_rank_key)
    if not ranked:
        return set()

    shortlist_count = _LOW_SHORTLIST_CAP if is_low else _SHORTLIST_COUNT
    shortlist_cap = _LOW_SHORTLIST_CAP if is_low else _SHORTLIST_CAP
    shortlist_window = _SHORTLIST_SCORE_WINDOW

    shortlist: list[str] = [template.key for template in ranked[:shortlist_count]]

    if not is_low:
        for key in _template_anchor_keys(templates):
            if key not in shortlist and len(shortlist) < shortlist_cap:
                shortlist.append(key)

    for template in templates:
        if (
            template.is_current_model_baseline
            and template.key not in shortlist
            and len(shortlist) < shortlist_cap
        ):
            shortlist.append(template.key)

    cutoff_index = min(shortlist_count, len(ranked)) - 1
    cutoff_score = initial_assessments[ranked[cutoff_index].key].metric_value(metric)
    for template in ranked[shortlist_count:]:
        if len(shortlist) >= shortlist_cap:
            break
        score = initial_assessments[template.key].metric_value(metric)
        if score - cutoff_score <= shortlist_window:
            shortlist.append(template.key)

    for key in forced_keys:
        if key not in shortlist and (not is_low or len(shortlist) < shortlist_cap):
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


def _warm_certificate_fit(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
    warm_start_by_run: dict[int, ParameterSet],
    initial_step_sizes: dict[str, float],
    difficult_assignment: bool,
    screening: bool,
    strategy: str,
    instrumentation: dict[str, object] | None,
) -> tuple[dict[int, FitResult] | None, ParameterSet, float, dict[str, float]]:
    """One warm single-cycle global fit for technique D's monotonicity certificate.

    A single migrad from the warm-start seed, with no staged multi-cycle
    re-seeding and no multi-start variants — the cheapest converging step. The
    caller decides whether the result clears the certificate; on failure the
    caller escalates to the full battery. Returns
    ``(results_by_run | None, fitted_global, total_chi2, step_hints)``.

    ``screening`` is passed separately from ``difficult_assignment`` because the
    separable engine's warm-only nodes are *deliberately* screened (Minuit
    strategy 0) even when the assignment is wide: the GLS collapse has already
    placed the seed in the right basin, so a strategy-2 hunt buys nothing.
    """

    warm_seed = _clone_parameter_sets(warm_start_by_run)
    call_budget = _global_fit_call_budget(
        datasets,
        warm_seed,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        phase="full",
    )
    _record_counter(instrumentation, "global_fit_calls")
    _record_counter(instrumentation, "warm_certificate_fits")
    if initial_step_sizes:
        _record_counter(instrumentation, "curvature_hint_applications")
        _append_metric(instrumentation, "curvature_hint_sizes", len(initial_step_sizes))
    results_by_run, fitted_global = fit_engine.global_fit(
        datasets,
        template.model.function,
        list(global_param_names),
        list(local_param_names),
        warm_seed,
        max_calls=call_budget,
        migrad_iterations=7 if difficult_assignment else 5,
        use_simplex_rescue=difficult_assignment,
        minuit_strategy=2 if difficult_assignment else None,
        minuit_tol=0.05 if difficult_assignment else None,
        initial_step_sizes=initial_step_sizes or None,
        screening=screening,
        strategy=strategy,
    )
    _record_global_fit_diagnostics(instrumentation, results_by_run)
    results_by_run = _canonicalize_fit_results_by_run(
        results_by_run,
        template=template,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        fixed_param_names=fixed_param_names,
    )
    if not all(result.success for result in results_by_run.values()):
        return None, ParameterSet(), float("inf"), dict(initial_step_sizes)
    total_chi2 = float(sum(result.chi_squared for result in results_by_run.values()))
    step_hints = _step_hints_from_fit_results(
        datasets,
        results_by_run,
        target_global_names=global_param_names,
        target_local_names=local_param_names,
    )
    return results_by_run, fitted_global, total_chi2, step_hints


def _assemble_assignment_assessment(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    base_by_run: dict[int, ParameterSet],
    results_by_run: dict[int, FitResult],
    fitted_global: ParameterSet,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    fit_success: bool,
) -> GlobalCandidateAssessment:
    """Score one role assignment's per-run results into a candidate assessment.

    Information criteria, residual diagnostics, dense curves and series warnings
    — everything that turns ``results_by_run`` into a leaderboard row. Split out
    of :func:`_fit_exact_assignment` so the separable engine's all-local node,
    which is assembled from the phase-1 per-run fits and is never refitted
    jointly, is scored by *exactly* the same code as a coupled node rather than a
    parallel reimplementation that could drift on ``k``, ``n`` or the gates.
    """

    sample_count = int(sum(dataset.n_points for dataset in datasets))
    parameter_count = len(global_param_names) + len(local_param_names) * len(datasets)
    if fit_success:
        total_chi2 = float(sum(result.chi_squared for result in results_by_run.values()))
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
        result = results_by_run.get(
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
            results_by_run=results_by_run,
            local_param_names=local_param_names,
        )
    )
    return GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=results_by_run,
        global_parameters=fitted_global,
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
    warm_start_chi2: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
    search_strategy: str = "legacy",
    strategy: str = "joint",
    warm_start_only: bool = False,
    instrumentation: dict[str, object] | None = None,
    initial_step_sizes: dict[str, float] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> GlobalCandidateAssessment:
    """Fit one global/local role assignment and score it.

    ``strategy`` selects the engine's minimiser architecture (``"joint"`` — the
    historical path — or ``"profiled"``) and reaches *every* ``global_fit`` call
    this function makes, staged seeding and simplex rescue included, so a node is
    never half-profiled.

    ``warm_start_only`` is the separable engine's cheap node: the caller has
    already placed the seed in the right basin with the GLS collapse, so a single
    screened warm migrad is the whole fit. The multi-start battery runs only when
    that one fit fails to converge (counted as ``warm_only_escalations``). It
    requires ``warm_start_by_run`` — without a warm start there is nothing to
    start only from.
    """

    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Global fit wizard analysis cancelled.")
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
                strategy=strategy,
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
                screening=not difficult_assignment,
                strategy=strategy,
                cancel_callback=cancel_callback,
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

    # Technique D (escalation-on-anomaly): a warm-started child node (from its
    # best Hamming-neighbour predecessor) is strictly more flexible than that
    # predecessor, so a good fit lands at χ² <= χ²(parent) + ε. If ONE warm
    # single-cycle migrad clears that monotonicity certificate and Minuit reports
    # a valid minimum, accept it and skip the full multi-start / staged / fallback
    # / simplex battery. Escalate to the battery only when the certificate fails.
    # Anchor nodes (all-global round 0 has no warm start; all-local is fitted
    # separately; layer-1 is the first coupled step) keep the full battery, so the
    # fast path is gated on a warm start plus >= 2 localised params.
    best_results = None
    best_global = ParameterSet()
    best_score = float("inf")
    best_failure_message = "No fit attempts were created."
    step_hints = dict(initial_step_sizes or {})
    certificate_passed = False
    if warm_start_only:
        # Separable engine: the GLS collapse already placed globals and
        # conditional locals at the second-order optimum, so one screened warm
        # migrad IS the fit. No certificate is checked here — backward
        # elimination owns the (opposite-direction) monotonicity certificate,
        # comparing the child's χ² against its *parent*'s from the outside.
        (
            warm_results,
            warm_global,
            warm_score,
            warm_step_hints,
        ) = _warm_certificate_fit(
            datasets,
            template,
            fit_engine=fit_engine,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=fixed_param_names,
            warm_start_by_run=warm_start_by_run,
            initial_step_sizes=step_hints,
            difficult_assignment=False,
            screening=True,
            strategy=strategy,
            instrumentation=instrumentation,
        )
        if warm_results is not None and all(r.success for r in warm_results.values()):
            best_results = warm_results
            best_global = warm_global
            best_score = warm_score
            step_hints = warm_step_hints
            certificate_passed = True
            _record_counter(instrumentation, "warm_only_accepts")
        else:
            _record_counter(instrumentation, "warm_only_escalations")
    elif (
        warm_start_by_run is not None
        and warm_start_chi2 is not None
        and math.isfinite(warm_start_chi2)
        and len(local_param_names) >= 2
    ):
        (
            warm_results,
            warm_global,
            warm_score,
            warm_step_hints,
        ) = _warm_certificate_fit(
            datasets,
            template,
            fit_engine=fit_engine,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
            fixed_param_names=fixed_param_names,
            warm_start_by_run=warm_start_by_run,
            initial_step_sizes=step_hints,
            difficult_assignment=difficult_assignment,
            screening=not difficult_assignment,
            strategy=strategy,
            instrumentation=instrumentation,
        )
        if warm_results is not None and all(r.success for r in warm_results.values()):
            # Certificate: the child (more free params) must not do worse than its
            # parent by more than ε. ε absorbs Minuit's EDM-scale numerical slop.
            certificate_ok = warm_score <= warm_start_chi2 + _WARM_CERTIFICATE_EPSILON
            if certificate_ok:
                best_results = warm_results
                best_global = warm_global
                best_score = warm_score
                step_hints = warm_step_hints
                certificate_passed = True
                _record_counter(instrumentation, "warm_certificate_accepts")
            else:
                _record_counter(instrumentation, "warm_certificate_escalations")
        else:
            # The warm single-cycle fit itself failed to converge; we still fall
            # through to the full battery below, so count it as an escalation too
            # (otherwise the counter only sees certificate failures, not warm-fit
            # failures, and undercounts how often the fast path falls through).
            _record_counter(instrumentation, "warm_certificate_escalations")

    if not certificate_passed:
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
            strategy=strategy,
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
            strategy=strategy,
            cancel_callback=cancel_callback,
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

    assessment = _assemble_assignment_assessment(
        datasets,
        template,
        base_by_run=base_by_run,
        results_by_run=best_results,
        fitted_global=best_global,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        fixed_param_names=fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        fit_success=fit_success,
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
    # The applied field is run metadata, so the series candidates get the same
    # field/B_L policy the per-run screening does — otherwise a zero-field
    # series would carry a free B_L per run that its own screening had pinned.
    # Everything below still overrides it: an explicit current value, a caller
    # bound, or a caller-declared pin.
    parameters = _initial_parameters_for_template(
        dataset, fingerprint, template, seed_context=_field_seed_context(dataset)
    )
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


def _best_single_run_fit_result(
    dataset: MuonDataset,
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    seed_params: ParameterSet,
    base_params: ParameterSet,
    global_names: tuple[str, ...],
    fixed_param_names: tuple[str, ...],
    previous_success: ParameterSet | None,
) -> FitResult | None:
    """Best of up to three independent single-run fits of ``template``.

    The per-run half of :func:`_single_run_prefit_parameter_sets`, lifted out so
    the separable engine's all-local node — which needs the ``FitResult``s
    themselves (χ², covariance, bound hits), not just the fitted values — is
    produced by the same attempt ladder rather than a second, subtly different
    one. Attempts are, in order: the previous run's converged values mapped onto
    this run's seed (the series is ordered, so the neighbour is the best guess),
    the caller's seed, then the template's initial-parameter variants;
    duplicates are dropped and each attempt falls back to simplex on a failed
    migrad. ``None`` when no attempt produced a fit at all.
    """

    run_number = int(dataset.run_number)
    attempt_params: list[ParameterSet] = []
    if previous_success is not None:
        neighbor_seed = _clone_parameter_set(seed_params)
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

    attempt_params.append(_clone_parameter_set(seed_params))
    for variant_map in _initial_param_variants({run_number: base_params}, template):
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

    return best_result


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
        best_result = _best_single_run_fit_result(
            dataset,
            template,
            fit_engine=fit_engine,
            seed_params=seeded[run_number],
            base_params=base_by_run[run_number],
            global_names=global_names,
            fixed_param_names=fixed_param_names,
            previous_success=previous_success,
        )
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
    strategy: str = "joint",
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
            screening=len(free_local_names) < 2,
            initial_step_sizes=step_hints or None,
            strategy=strategy,
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
            screening=len(free_global_names) < 4,
            strategy=strategy,
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
                screening=True,
                strategy=strategy,
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

    def _median_over_candidates(attr: str) -> float:
        # Only runs that actually carry a damped-line candidate contribute: a
        # zero from a run where the early pass found nothing is an absence, not
        # a measurement, and would drag the aggregate to a meaningless value.
        values = [
            float(getattr(fingerprint, attr))
            for fingerprint in fingerprints
            if fingerprint.has_damped_line_candidate
        ]
        return float(np.median(values)) if values else 0.0

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
        damped_line_frequency_mhz=_median_over_candidates("damped_line_frequency_mhz"),
        damped_line_snr=_median_over_candidates("damped_line_snr"),
        damped_line_crop_us=_median_over_candidates("damped_line_crop_us"),
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
    if fit_result.success and residual_rms <= _GOOD_FIT_RESIDUAL_RMS_FOR_BOUND_HITS:
        # A parameter resting at its bound is the valid answer when the fit is
        # otherwise good -- e.g. a relaxation/damping rate driven to 0 (negligible
        # damping, as deep in an ordered magnet's ZF precession). A bound-hit
        # should only veto the recommendation when the fit is *also* poor, so drop
        # bound-hit reasons here and let the ranking's persistent-lower-bound
        # penalty handle any residual preference between passing candidates.
        reasons = [
            reason
            for reason in reasons
            if not (reason.endswith(" at lower bound") or reason.endswith(" at upper bound"))
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

    The one implementation lives in :mod:`global_search.surrogate` so the tier-2
    surrogate's IC and the exact path's IC can never drift apart.
    """

    return surrogate_metric_penalty(parameter_count, sample_count=sample_count, metric=metric)


@dataclass
class _HeuristicTemplateState:
    """Per-template working state for the heuristic (Low/Balanced) search.

    Mirrors the fields the exhaustive wavefront threads through, but the search
    populates ``exact_cache``/``converged_assessments`` sparsely (only the
    assignments the heuristic actually fits) — with one hard guarantee: after
    :func:`_fill_winner_flip_neighbourhood` runs, the cache contains the winner
    plus *all* single-flip neighbours over the full free-param set, so the
    downstream ``_build_parameter_recommendations_from_exact_cache`` (which reads
    the winner's flip-neighbourhood to justify each role) is never starved.
    """

    template: CandidateTemplate
    fixed_param_names: tuple[str, ...]
    prefit_base_by_run: dict[int, ParameterSet]
    free_param_names: tuple[str, ...]
    exact_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], GlobalCandidateAssessment]
    converged_assessments: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        GlobalCandidateAssessment,
    ]
    anchor_assessment: GlobalCandidateAssessment | None = None
    best_assessment: GlobalCandidateAssessment | None = None
    #: Q pre-test outcome per free parameter (technique E), for progress/logging.
    homogeneity: dict[str, ParameterHomogeneity] = field(default_factory=dict)
    #: Per-parameter all-local estimates / errors (technique E + G source).
    estimates: dict[str, tuple[float, ...]] = field(default_factory=dict)
    estimate_errors: dict[str, tuple[float, ...]] = field(default_factory=dict)


def _local_names_for(
    free_param_names: tuple[str, ...],
    local_set: set[str],
) -> tuple[str, ...]:
    """Canonical sorted local-name tuple over the free params of a template."""

    return tuple(sorted(name for name in free_param_names if name in local_set))


def _global_names_for(
    free_param_names: tuple[str, ...],
    local_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Free globals = free params not in ``local_names`` (order preserved)."""

    local_set = set(local_names)
    return tuple(name for name in free_param_names if name not in local_set)


def _fit_heuristic_assignment(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    local_names: tuple[str, ...],
    *,
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
    warm_start_source: GlobalCandidateAssessment | None = None,
) -> GlobalCandidateAssessment:
    """Fit one role assignment for a heuristic template and cache it.

    Reuses :func:`_fit_exact_assignment` verbatim (same fidelity, same
    instrumentation counters the harness reads) so every heuristic-fit IC is on
    the same footing as the exhaustive baseline. Warm-starts from a neighbour
    assessment when one is supplied.
    """

    global_names = _global_names_for(state.free_param_names, local_names)
    warm_start_by_run: dict[int, ParameterSet] | None = None
    warm_start_chi2: float | None = None
    initial_step_sizes: dict[str, float] = {}
    if warm_start_source is not None and warm_start_source.is_successful:
        warm_start_by_run = _warm_start_parameter_sets(
            datasets,
            assessment=warm_start_source,
            base_by_run=state.prefit_base_by_run,
            target_global_names=global_names,
            target_local_names=local_names,
            fit_engine=FitEngine(),
            template=state.template,
            progress_callback=progress_callback,
        )
        initial_step_sizes = _step_hints_from_assessment(
            datasets,
            warm_start_source,
            target_global_names=global_names,
            target_local_names=local_names,
        )
        # Technique D certificate: only arm it when the warm source is a strictly
        # *simpler* parent (its locals are a subset of the child's) — then the
        # child, being strictly more flexible, cannot honestly exceed the parent's
        # χ², so a warm child landing at χ² <= parent + ε skips the multi-start
        # battery. A backward-prune trial (fewer locals than its warm incumbent)
        # is the reverse and must NOT arm the certificate; it simply escalates.
        source_local = set(warm_start_source.local_param_names)
        if source_local < set(local_names):
            warm_start_chi2 = float(
                sum(result.chi_squared for result in warm_start_source.fit_results_by_run.values())
            )
    assessment = _fit_exact_assignment(
        datasets,
        state.template,
        fit_engine=FitEngine(),
        base_by_run=state.prefit_base_by_run,
        global_param_names=global_names,
        local_param_names=local_names,
        fixed_param_names=state.fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        cache=state.exact_cache,
        warm_start_by_run=warm_start_by_run,
        warm_start_chi2=warm_start_chi2,
        progress_callback=progress_callback,
        search_strategy=search_strategy,
        instrumentation=instrumentation,
        initial_step_sizes=initial_step_sizes,
    )
    key = (global_names, local_names)
    state.exact_cache[key] = assessment
    if assessment.is_successful:
        state.converged_assessments[key] = assessment
        if state.best_assessment is None or _assessment_sort_key(
            assessment, metric
        ) < _assessment_sort_key(state.best_assessment, metric):
            state.best_assessment = assessment
    return assessment


def _extract_anchor_estimates(
    datasets: list[MuonDataset],
    anchor: GlobalCandidateAssessment,
    free_param_names: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]], set[str]]:
    """Per-parameter (θ_g, σ_g) and the at-limit set from the all-local anchor.

    The all-local joint fit is equivalent to G independent per-dataset fits (no
    shared params → block-diagonal covariance), so each per-run ``FitResult``
    carries the local estimate ``θ_g`` and its 1σ ``σ_g`` for every free
    parameter — exactly the source technique E (Q-test) and technique G (Wald
    surrogate) both need. Returns ``(estimates, errors, at_limit_params)`` where
    ``at_limit_params`` names any parameter that pinned a bound in *any* dataset
    (the Q-test must never pre-fix such a parameter).
    """

    estimates: dict[str, list[float]] = {name: [] for name in free_param_names}
    errors: dict[str, list[float]] = {name: [] for name in free_param_names}
    at_limit: set[str] = set()

    for diagnostic in anchor.run_diagnostics:
        for reason in diagnostic.gate_reasons:
            for edge in (" at lower bound", " at upper bound"):
                if edge in reason:
                    at_limit.add(reason.split(edge, 1)[0])

    for dataset in datasets:
        result = anchor.fit_results_by_run.get(int(dataset.run_number))
        if result is None:
            continue
        for name in free_param_names:
            parameter = result.parameters[name] if name in result.parameters else None
            if parameter is None:
                estimates[name].append(float("nan"))
                errors[name].append(float("nan"))
                continue
            estimates[name].append(float(parameter.value))
            sigma = _positive_uncertainty(result.uncertainties.get(name))
            errors[name].append(sigma if sigma is not None else float("nan"))

    return (
        {name: tuple(values) for name, values in estimates.items()},
        {name: tuple(values) for name, values in errors.items()},
        at_limit,
    )


def _homogeneity_pretests(
    state: _HeuristicTemplateState,
    at_limit: set[str],
    *,
    q_bands: tuple[float, float],
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
) -> tuple[set[str], set[str], list[str]]:
    """Technique E: partition free params into fixed-local / fixed-global / middle.

    Returns ``(fixed_local, fixed_global, ambiguous)``. A parameter is only
    pre-fixed when its Q classification is unambiguous *and* its single fits were
    clean (finite errors, not at a limit); everything else stays ambiguous and
    is enumerated. The ambiguous middle preserves the template's free-param
    order so downstream enumeration is deterministic.
    """

    p_local, p_global = q_bands
    fixed_local: set[str] = set()
    fixed_global: set[str] = set()
    ambiguous: list[str] = []
    for name in state.free_param_names:
        outcome = classify_parameter_homogeneity(
            name,
            state.estimates.get(name, ()),
            state.estimate_errors.get(name, ()),
            at_limit=name in at_limit,
            p_local_threshold=p_local,
            p_global_threshold=p_global,
        )
        state.homogeneity[name] = outcome
        if outcome.role == "local":
            fixed_local.add(name)
            _record_counter(instrumentation, "q_pretest_fixed_local")
        elif outcome.role == "global":
            fixed_global.add(name)
            _record_counter(instrumentation, "q_pretest_fixed_global")
        else:
            ambiguous.append(name)
            _record_counter(instrumentation, "q_pretest_ambiguous")
    _progress_log(
        progress_callback,
        f"{state.template.title}: Q pre-tests fixed "
        f"{len(fixed_local)} local, {len(fixed_global)} global; "
        f"{len(ambiguous)} ambiguous → enumerate.",
    )
    return fixed_local, fixed_global, ambiguous


def _greedy_role_search(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    *,
    base_local: set[str],
    searchable: tuple[str, ...],
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
) -> GlobalCandidateAssessment:
    """Technique F: forward-select from all-global, then backward-prune. O(P²).

    Starts from the Q-fixed-local set (``base_local``) with every ``searchable``
    param global, greedily flips the single best global→local move while it
    improves the penalised score beyond the role-delta threshold, then prunes any
    local flip that no longer earns its complexity. Only ``searchable`` params
    ever change role; Q-fixed-local stay local, Q-fixed-global stay global.
    """

    incumbent = _fit_heuristic_assignment(
        datasets,
        state,
        _local_names_for(state.free_param_names, base_local),
        axis_key=axis_key,
        metric=metric,
        search_strategy=search_strategy,
        progress_callback=progress_callback,
        instrumentation=instrumentation,
    )
    if not incumbent.is_successful:
        return incumbent

    current_local = set(base_local)
    remaining = [name for name in searchable if name not in current_local]

    # Forward selection.
    while remaining:
        best_candidate: GlobalCandidateAssessment | None = None
        best_name: str | None = None
        for name in remaining:
            trial_local = _local_names_for(state.free_param_names, current_local | {name})
            candidate = _fit_heuristic_assignment(
                datasets,
                state,
                trial_local,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
                warm_start_source=incumbent,
            )
            if not candidate.is_successful:
                continue
            if best_candidate is None or _assessment_sort_key(
                candidate, metric
            ) < _assessment_sort_key(best_candidate, metric):
                best_candidate = candidate
                best_name = name
        if best_candidate is None or best_name is None:
            break
        if not _prefer_role_change(best_candidate, incumbent, metric=metric):
            break
        incumbent = best_candidate
        current_local.add(best_name)
        remaining = [name for name in searchable if name not in current_local]
        localized = ", ".join(sorted(current_local)) or "none"
        _progress_log(
            progress_callback,
            f"{state.template.title}: greedy localised {best_name} (Local set [{localized}]).",
        )

    # Backward pruning: drop a searchable local that no longer earns its keep.
    for name in [n for n in searchable if n in current_local]:
        trial_local = _local_names_for(state.free_param_names, current_local - {name})
        candidate = _fit_heuristic_assignment(
            datasets,
            state,
            trial_local,
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
            warm_start_source=incumbent,
        )
        if candidate.is_successful and _prefer_simpler_assignment(
            candidate, incumbent, metric=metric
        ):
            incumbent = candidate
            current_local.discard(name)
            _progress_log(
                progress_callback,
                f"{state.template.title}: greedy pruned {name} back to Global.",
            )
    return incumbent


def _surrogate_ranked_search(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    *,
    base_local: set[str],
    ambiguous: tuple[str, ...],
    top_k: int,
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
) -> GlobalCandidateAssessment:
    """Technique G: Wald-rank the ambiguous subsets, real-fit only the top-K/layer.

    For each Hamming layer of the ambiguous middle (how many ambiguous params to
    localise) the Wald surrogate predicts the Δχ² of *not* globalising each
    subset; the caller ranks by surrogate IC and real-fits the ``top_k`` per
    layer. K grows by one whenever the realised best of a layer lands at the K-th
    rank (the surrogate order disagreed near the top), so a mis-ranked winner is
    still fitted. Q-fixed-local (``base_local``) params are always local.
    """

    incumbent = _fit_heuristic_assignment(
        datasets,
        state,
        _local_names_for(state.free_param_names, base_local),
        axis_key=axis_key,
        metric=metric,
        search_strategy=search_strategy,
        progress_callback=progress_callback,
        instrumentation=instrumentation,
    )
    if not incumbent.is_successful:
        return incumbent

    penalty_all_local = wald_subset_delta_chi2(ambiguous, state.estimates, state.estimate_errors)
    for layer_size in range(1, len(ambiguous) + 1):
        subsets = list(combinations(ambiguous, layer_size))
        # Surrogate IC ordering: globalising the *complement* of ``subset`` costs
        # the sum of its members' collapse penalties, so a subset whose localised
        # members carry the most globalisation cost (i.e. best to keep local)
        # ranks first. Rank ascending by surrogate IC = χ²_floor proxy + penalty.
        scored: list[tuple[float, tuple[str, ...]]] = []
        for subset in subsets:
            # ``subset`` is the set localised this layer; the surrogate cost is the
            # residual Δχ² of globalising the *remaining* ambiguous params (those
            # not in ``subset``) = penalty_all_local − Δχ²(subset).
            predicted_cost = penalty_all_local - wald_subset_delta_chi2(
                subset, state.estimates, state.estimate_errors
            )
            # Fewer localised params is cheaper in penalty; the surrogate IC adds
            # that penalty to the residual globalisation cost. Lower is better.
            surrogate_ic = predicted_cost + 2.0 * len(subset) * len(datasets)
            scored.append((surrogate_ic, subset))
        scored.sort(key=lambda item: item[0])

        verify = min(top_k, len(scored))
        rank = 0
        realised_best_rank: int | None = None
        layer_best: GlobalCandidateAssessment | None = None
        while rank < len(scored) and rank < verify:
            _surrogate_ic, subset = scored[rank]
            trial_local = _local_names_for(state.free_param_names, base_local | set(subset))
            candidate = _fit_heuristic_assignment(
                datasets,
                state,
                trial_local,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
                warm_start_source=incumbent,
            )
            _record_counter(instrumentation, "surrogate_real_fits")
            if candidate.is_successful:
                if layer_best is None or _assessment_sort_key(
                    candidate, metric
                ) < _assessment_sort_key(layer_best, metric):
                    layer_best = candidate
                    realised_best_rank = rank
                if _assessment_sort_key(candidate, metric) < _assessment_sort_key(
                    incumbent, metric
                ):
                    incumbent = candidate
            # Grow K when the realised best sits at the current verify frontier —
            # the surrogate mis-ranked and a better subset may lurk just past K.
            if realised_best_rank is not None and realised_best_rank == verify - 1:
                verify = min(verify + 1, len(scored))
                _record_counter(instrumentation, "surrogate_k_grown")
            rank += 1
        _record_metric_max(instrumentation, "surrogate_rank_of_winner", realised_best_rank)
    return incumbent


def _record_metric_max(
    instrumentation: dict[str, object] | None,
    name: str,
    value: int | None,
) -> None:
    """Track the running max of a metric in instrumentation (0-based rank)."""

    if instrumentation is None or value is None:
        return
    counters = instrumentation.setdefault("counters", {})
    if isinstance(counters, dict):
        counters[name] = max(int(counters.get(name, 0)), int(value))


def _fill_winner_flip_neighbourhood(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    winner: GlobalCandidateAssessment,
    *,
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
) -> None:
    """Fit every single-flip neighbour of ``winner`` over the FULL free set.

    This is the correctness linchpin of the heuristic path (verification-plan
    item 1 / item 6): a sparse search leaves the exact cache missing the winner's
    single-role-flip neighbours, and ``_build_parameter_recommendations_from_
    exact_cache`` reads exactly those neighbours to justify each parameter's role.
    Critically the neighbourhood spans *all* free params — including any Q
    pre-fixed as clearly-global — so a wrong pre-fix is still caught by the
    flip-recheck rather than silently trusted.
    """

    winner_local = set(winner.local_param_names)
    for name in state.free_param_names:
        if name in winner_local:
            flipped = _local_names_for(state.free_param_names, winner_local - {name})
        else:
            flipped = _local_names_for(state.free_param_names, winner_local | {name})
        global_names = _global_names_for(state.free_param_names, flipped)
        if (global_names, flipped) in state.exact_cache:
            continue
        _record_counter(instrumentation, "flip_neighbourhood_fits")
        _fit_heuristic_assignment(
            datasets,
            state,
            flipped,
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
            warm_start_source=winner,
        )


def _decimation_factor_for_engine(engine: str) -> int:
    """Technique K's rebin factor for the search phase, by heuristic engine."""
    if engine == SEARCH_ENGINE_LOW:
        return _DECIMATION_FACTOR_LOW
    if engine == SEARCH_ENGINE_BALANCED:
        return _DECIMATION_FACTOR_BALANCED
    return 1


def _decimation_is_nyquist_safe(
    datasets: list[MuonDataset],
    aggregate_fingerprint: SpectrumFingerprint | None,
    factor: int,
) -> bool:
    """Technique K's Nyquist gate: would rebinning alias the dominant content?

    Compares the *decimated* sample rate's Nyquist frequency against the
    aggregate fingerprint's dominant FFT frequency, with a conservative margin
    (:data:`_DECIMATION_MIN_CYCLES_MARGIN`): decimation is refused whenever the
    dominant oscillatory content sits within that margin of the coarser
    Nyquist limit, so an oscillatory template is never searched against an
    aliased spectrum. A fingerprint with no detected oscillatory content (or no
    fingerprint at all) is always safe to decimate — there is nothing to alias.
    """
    if factor <= 1 or aggregate_fingerprint is None:
        return True
    if not aggregate_fingerprint.oscillatory_hint:
        return True
    dominant_mhz = float(aggregate_fingerprint.dominant_fft_frequency_mhz)
    if dominant_mhz <= 0.0:
        return True
    for dataset in datasets:
        time = np.asarray(dataset.time, dtype=float)
        if time.size < 2 * factor:
            return False
        dt = float(np.mean(np.diff(time)))
        if dt <= 0.0:
            return False
        decimated_dt = dt * factor
        decimated_nyquist_mhz = 0.5 / decimated_dt
        if dominant_mhz * _DECIMATION_MIN_CYCLES_MARGIN >= decimated_nyquist_mhz:
            return False
    return True


def _decimated_datasets_for_search(
    datasets: list[MuonDataset],
    *,
    engine: str,
    aggregate_fingerprint: SpectrumFingerprint | None,
    instrumentation: dict[str, object] | None,
) -> tuple[list[MuonDataset], int]:
    """Technique K: rebin the search-phase datasets, or return them unchanged.

    Returns ``(search_datasets, factor)``; ``factor == 1`` means no decimation
    happened (either the engine doesn't decimate, the factor is a no-op, or the
    Nyquist gate refused it). The caller is responsible for refitting the
    winner and its flip-neighbourhood at full resolution before any assessment
    reaches the returned leaderboard — decimated ICs must never be compared
    against full-resolution ones on the same leaderboard.
    """
    factor = _decimation_factor_for_engine(engine)
    if factor <= 1:
        return datasets, 1
    if not _decimation_is_nyquist_safe(datasets, aggregate_fingerprint, factor):
        _record_counter(instrumentation, "decimation_skipped_nyquist_gate")
        return datasets, 1
    try:
        decimated = [dataset.rebin(factor) for dataset in datasets]
    except ValueError:
        return datasets, 1
    _record_counter(instrumentation, "decimation_applied")
    _append_metric(instrumentation, "decimation_factor", float(factor))
    return decimated, factor


def _new_heuristic_template_state(
    template: CandidateTemplate,
    *,
    datasets: list[MuonDataset],
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
    single_run_prefit_cache_for: Callable[
        [CandidateTemplate], dict[object, dict[int, ParameterSet]]
    ],
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
    prefit_cache_override: dict[object, dict[int, ParameterSet]] | None = None,
) -> _HeuristicTemplateState:
    """Build a fresh, empty-role-cache state for one template.

    ``prefit_cache_override`` lets a caller isolate the per-run prefit cache
    from ``single_run_prefit_cache_for``'s shared one — needed by technique K's
    full-resolution refit pass (:func:`_refit_states_at_full_resolution`),
    whose cache key (``fixed_param_names`` + the seed *values*, not resolution)
    would otherwise collide with entries the decimated search phase already
    populated and silently reuse decimated-data prefit seeds.
    """
    base_by_run, fixed_param_names = template_contexts[template.key]
    prefit_base_by_run = _single_run_prefit_parameter_sets(
        datasets,
        template,
        fit_engine=FitEngine(),
        base_by_run=base_by_run,
        fixed_param_names=fixed_param_names,
        progress_callback=progress_callback,
        instrumentation=instrumentation,
        cache=(
            prefit_cache_override
            if prefit_cache_override is not None
            else single_run_prefit_cache_for(template)
        ),
    )
    free_param_names = tuple(
        name for name in template.model.param_names if name not in fixed_param_names
    )
    return _HeuristicTemplateState(
        template=template,
        fixed_param_names=fixed_param_names,
        prefit_base_by_run=prefit_base_by_run,
        free_param_names=free_param_names,
        exact_cache={},
        converged_assessments={},
    )


def _refit_states_at_full_resolution(
    datasets: list[MuonDataset],
    decimated_states: list[_HeuristicTemplateState],
    *,
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
    single_run_prefit_cache_for: Callable[
        [CandidateTemplate], dict[object, dict[int, ParameterSet]]
    ],
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
) -> list[_HeuristicTemplateState]:
    """Technique K's correctness step: redo the winner at native resolution.

    ``decimated_states`` carries each template's *winning role split* (picked
    from decimated-resolution fits), but none of those fits belong on a
    leaderboard alongside full-resolution ICs (mixed-n AICc/BIC is meaningless
    per verification-plan item 5). This builds one fresh, empty-cache state per
    template against the original ``datasets`` and refits exactly the winning
    assignment plus its full single-flip neighbourhood — nothing decimated ever
    reaches :func:`_finalise_heuristic_assessments`.

    Uses a fresh per-template prefit-seed cache (``prefit_cache_override``)
    rather than ``single_run_prefit_cache_for``'s shared one: that cache's key
    is ``(fixed_param_names, base_by_run signature)`` with no resolution term,
    so reusing it here would silently warm-start the full-resolution refit from
    prefit seeds computed against the decimated data.
    """

    refit_states: list[_HeuristicTemplateState] = []
    for decimated_state in decimated_states:
        template = decimated_state.template
        winner = decimated_state.best_assessment
        full_state = _new_heuristic_template_state(
            template,
            datasets=datasets,
            template_contexts=template_contexts,
            single_run_prefit_cache_for=single_run_prefit_cache_for,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
            prefit_cache_override={},
        )
        refit_states.append(full_state)
        if winner is None or not full_state.free_param_names:
            if not full_state.free_param_names:
                _fit_heuristic_assignment(
                    datasets,
                    full_state,
                    (),
                    axis_key=axis_key,
                    metric=metric,
                    search_strategy=search_strategy,
                    progress_callback=progress_callback,
                    instrumentation=instrumentation,
                )
            continue
        _record_counter(instrumentation, "decimation_full_res_refits")
        full_winner = _fit_heuristic_assignment(
            datasets,
            full_state,
            winner.local_param_names,
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )
        if full_winner.is_successful:
            _fill_winner_flip_neighbourhood(
                datasets,
                full_state,
                full_winner,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
            )
        else:
            # The decimated winner failed to reproduce at full resolution — fall
            # back to the all-local anchor so the template still contributes a
            # (worse-ranked but real, full-resolution) assessment rather than
            # silently dropping out of the leaderboard.
            _record_counter(instrumentation, "decimation_full_res_refit_failed")
            anchor = _fit_heuristic_assignment(
                datasets,
                full_state,
                tuple(full_state.free_param_names),
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
            )
            if anchor.is_successful:
                _fill_winner_flip_neighbourhood(
                    datasets,
                    full_state,
                    anchor,
                    axis_key=axis_key,
                    metric=metric,
                    search_strategy=search_strategy,
                    progress_callback=progress_callback,
                    instrumentation=instrumentation,
                )
    return refit_states


def _run_heuristic_search(
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
    engine: str,
    aggregate_fingerprint: SpectrumFingerprint | None = None,
) -> tuple[GlobalCandidateAssessment, ...]:
    """Low/Balanced non-exhaustive role search (techniques E/F/G/H + K).

    Returns the same ``tuple[GlobalCandidateAssessment, ...]`` contract as
    :func:`_run_exhaustive_wavefront_search` — the winner plus its fully-fitted
    single-flip neighbourhood per template — so the downstream verdict/rerank
    layer is engine-agnostic. Serial by design: the entire point is far fewer
    real fits, so it stays off the process pool and avoids that lifecycle.

    Technique K (screening decimation): the role search itself (anchor, Q
    pre-tests, greedy/surrogate search, racing) runs on coarser-rebinned
    datasets at Low/Balanced, gated on the Nyquist-safety check above. Once a
    winning role split is picked per template, it — and its full single-flip
    neighbourhood — is refitted from scratch on the *original* full-resolution
    ``datasets`` into a fresh cache, so every assessment this function returns
    sits on the same full-resolution footing as every other tier (no mixed-n
    ICs ever reach the leaderboard).
    """

    if not shortlisted_templates:
        return ()

    search_datasets, decimation_factor = _decimated_datasets_for_search(
        datasets,
        engine=engine,
        aggregate_fingerprint=aggregate_fingerprint,
        instrumentation=instrumentation,
    )
    if decimation_factor > 1:
        _progress_log(
            progress_callback,
            f"Screening decimation: searching at {decimation_factor}x coarser binning; "
            "the winner and its flip-neighbourhood will be refitted at full resolution.",
        )

    q_bands = _Q_BANDS_LOW if engine == SEARCH_ENGINE_LOW else _Q_BANDS_BALANCED
    states: list[_HeuristicTemplateState] = [
        _new_heuristic_template_state(
            template,
            datasets=search_datasets,
            template_contexts=template_contexts,
            single_run_prefit_cache_for=single_run_prefit_cache_for,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )
        for template in shortlisted_templates
    ]

    # --- All-local anchor + Q pre-tests + shallow race (layers 0-1) ---------
    for state in states:
        if not state.free_param_names:
            # No promotable params — the single all-global assessment is the
            # verdict; fit it so the tuple is non-empty.
            _fit_heuristic_assignment(
                search_datasets,
                state,
                (),
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
            )
            continue
        anchor = _fit_heuristic_assignment(
            search_datasets,
            state,
            tuple(state.free_param_names),
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )
        state.anchor_assessment = anchor
        if anchor.is_successful:
            state.estimates, state.estimate_errors, at_limit = _extract_anchor_estimates(
                search_datasets, anchor, state.free_param_names
            )
            _homogeneity_pretests(
                state,
                at_limit,
                q_bands=q_bands,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
            )
        # All-global anchor (layer 0) too, so racing has a shallow score and the
        # search has a warm all-global start.
        _fit_heuristic_assignment(
            search_datasets,
            state,
            (),
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )

    # --- Template racing (technique H): advance only the top templates -------
    ranked = sorted(
        (state for state in states if state.best_assessment is not None),
        key=lambda s: _assessment_sort_key(s.best_assessment, metric),
    )
    advance = ranked[:_RACING_ADVANCE_COUNT] if engine == SEARCH_ENGINE_BALANCED else ranked
    if engine == SEARCH_ENGINE_BALANCED and len(ranked) > len(advance):
        _record_counter(instrumentation, "raced_templates_dropped", len(ranked) - len(advance))
        _progress_log(
            progress_callback,
            f"Template racing advanced {len(advance)}/{len(ranked)} template(s) "
            "past the shallow layer-0/1 race.",
        )

    # --- Deep search on the advanced templates -------------------------------
    for state in advance:
        if not state.free_param_names:
            continue
        # Read the Q pre-test outcomes computed during the anchor phase. A param
        # with no stored outcome (anchor failed) or an "ambiguous" outcome stays
        # searchable; only clear tails were pre-fixed. Any at-limit / invalid
        # param was already forced to "ambiguous" (skipped) inside the pre-test.
        base_local = {n for n, o in state.homogeneity.items() if o.role == "local"}
        ambiguous = tuple(
            n
            for n in state.free_param_names
            if state.homogeneity.get(n) is None or state.homogeneity[n].role == "ambiguous"
        )

        if engine == SEARCH_ENGINE_LOW:
            searchable = tuple(
                n
                for n in state.free_param_names
                if n not in base_local
                and (state.homogeneity.get(n) is None or state.homogeneity[n].role != "global")
            )
            winner = _greedy_role_search(
                search_datasets,
                state,
                base_local=base_local,
                searchable=searchable,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
            )
        else:
            winner = _surrogate_ranked_search(
                search_datasets,
                state,
                base_local=base_local,
                ambiguous=ambiguous,
                top_k=_SURROGATE_TOP_K,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
            )

        # The winner is whatever converged best across the whole cache (greedy /
        # surrogate incumbent may have been beaten by a shallow-race assignment).
        winner = state.best_assessment or winner
        if winner is not None and winner.is_successful:
            _fill_winner_flip_neighbourhood(
                search_datasets,
                state,
                winner,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                progress_callback=progress_callback,
                instrumentation=instrumentation,
            )

    if decimation_factor > 1:
        states = _refit_states_at_full_resolution(
            datasets,
            states,
            template_contexts=template_contexts,
            single_run_prefit_cache_for=single_run_prefit_cache_for,
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )

    return _finalise_heuristic_assessments(
        datasets, states, metric=metric, progress_callback=progress_callback
    )


def _finalise_heuristic_assessments(
    datasets: list[MuonDataset],
    states: list[_HeuristicTemplateState],
    *,
    metric: SelectionMetric,
    progress_callback: Callable[[str], None] | None,
) -> tuple[GlobalCandidateAssessment, ...]:
    """Build the returned assessments exactly like the exhaustive wavefront.

    Each template contributes its converged assignments with per-parameter role
    recommendations resolved from the (now flip-complete) exact cache, so the
    verdict layer treats a heuristic winner identically to an exhaustive one.
    """

    optimized_assessments: list[GlobalCandidateAssessment] = []
    for state in states:
        if not state.converged_assessments and state.best_assessment is None:
            continue
        exact_cache = dict(state.exact_cache)
        successful = sorted(
            state.converged_assessments.values(),
            key=lambda assessment: _assessment_sort_key(assessment, metric),
        )
        for assessment in successful:
            exact_cache[(assessment.global_param_names, assessment.local_param_names)] = assessment

        if successful:
            for assessment in successful:
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
            best = successful[0]
            _progress_log(
                progress_callback,
                f"Completed heuristic coupled optimisation for {state.template.title}. "
                f"{len(successful)} converged assignment(s); best {metric.value} = "
                f"{best.metric_value(metric):.3f} with "
                f"Global[{', '.join(best.global_param_names) or 'none'}], "
                f"Local[{', '.join(best.local_param_names) or 'none'}].",
            )
            continue

        failed = state.best_assessment
        if failed is None:
            continue
        optimized_assessments.append(
            replace(
                failed,
                fixed_param_names=state.fixed_param_names,
                parameter_recommendations=(),
                assessment_key=_global_candidate_assessment_key(
                    state.template.key,
                    global_param_names=failed.global_param_names,
                    local_param_names=failed.local_param_names,
                ),
            )
        )

    return tuple(optimized_assessments)


# --------------------------------------------------------------------------- #
# Separable role search (the default engine).
#
# The exhaustive wavefront enumerates 2^P role assignments per template and
# refits the all-local anchor as one joint (n_global + n_local*G)-parameter
# Minuit problem although phase 1 already holds every per-run fit. On a real
# ordered series that is where the search's whole budget goes.
#
# The separable engine replaces enumeration with a statistical model of it:
#
#   1. **All-local comes for free.** Independent per-run fits *are* the
#      all-local assignment, so it is assembled from the phase-1 results and
#      never fitted jointly.
#   2. **A full-covariance surrogate scores every assignment at once.** The GLS
#      collapse (``global_search.surrogate``) predicts the IC of globalising any
#      subset from the per-run estimates and covariances, and hands back the
#      warm start (shared values + conditional locals) the exact fit needs.
#   3. **Backward elimination is the exact path.** One parameter is globalised
#      at a time, cheapest-by-surrogate first, each step fitted exactly
#      (profiled, one warm variant) and accepted only when the *exact* IC
#      improves. At most P coupled fits, not 2^P.
#   4. **The winner's single-flip neighbourhood is fitted** so the per-parameter
#      role recommendations are exact, then winner + neighbourhood are refitted
#      jointly at full resolution — the leaderboard never mixes resolutions.
#
# The exhaustive wavefront stays reachable behind ``search_engine`` as the
# harness referee.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _SeparableAnchorTask:
    """Build one template's all-local node (no coupled fit) plus its estimates."""

    template_key: str
    template: CandidateTemplate
    datasets: list[MuonDataset]
    base_by_run: dict[int, ParameterSet]
    fixed_param_names: tuple[str, ...]
    axis_key: str
    metric: SelectionMetric
    search_strategy: str
    #: Per-run all-local ``FitResult``s reusable verbatim from the phase-1
    #: pre-screen, or ``None`` when they must be fitted here (either the
    #: pre-screen holds coupled fits rather than independent per-run ones, or it
    #: sits at a different resolution than the search).
    prescreen_results_by_run: dict[int, FitResult] | None


@dataclass(frozen=True)
class _SeparableTemplateResult:
    template_key: str
    state: _HeuristicTemplateState
    estimates: tuple[RunEstimate, ...]
    instrumentation: dict[str, object]


@dataclass(frozen=True)
class _SeparableEliminationTask:
    """Run one template's backward elimination and winner flip-neighbourhood."""

    template_key: str
    state: _HeuristicTemplateState
    estimates: tuple[RunEstimate, ...]
    datasets: list[MuonDataset]
    axis_key: str
    metric: SelectionMetric
    search_strategy: str


def _fresh_task_instrumentation() -> dict[str, object]:
    return {
        "counters": {},
        "curvature_hint_sizes": [],
        "minuit_edm": [],
        "relaxed_penalties": [],
        "staged_frontier_widths": [],
    }


def _counter_value(instrumentation: dict[str, object], name: str) -> int:
    """Read one counter out of a task's instrumentation dict.

    Callers hold a :func:`_fresh_task_instrumentation` dict, which seeds
    ``counters`` before anything can read it, so the lookup is total. A counter
    nobody has recorded yet is honestly zero.
    """

    counters: dict[str, int] = instrumentation["counters"]  # type: ignore[assignment]
    return int(counters.get(name, 0))


def _separable_search_rebin_factor(datasets: list[MuonDataset]) -> int:
    """One rebin factor for the whole series: the *smallest* run's factor.

    Every search fit runs at this resolution so the ICs on one leaderboard share
    an ``n``. Taking the minimum over runs means the series is never rebinned
    past what the most bandwidth-hungry run can afford — going past a run's own
    factor would alias the very line its seeds name.
    """

    return max(
        1,
        min(
            int(analysis_rebin_factor(dataset, field_gauss=_field_value(dataset)))
            for dataset in datasets
        ),
    )


def _separable_search_datasets(
    datasets: list[MuonDataset],
    *,
    search_rebin_factor: int,
    instrumentation: dict[str, object] | None,
) -> list[MuonDataset]:
    if search_rebin_factor <= 1:
        return datasets
    _record_counter(instrumentation, "separable_search_rebin_applied")
    return [dataset.rebin(search_rebin_factor) for dataset in datasets]


def _all_local_run_fit_results(
    datasets: list[MuonDataset],
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    base_by_run: dict[int, ParameterSet],
    fixed_param_names: tuple[str, ...],
    instrumentation: dict[str, object] | None,
) -> dict[int, FitResult]:
    """Fit ``template`` to each run independently — the all-local assignment.

    Runs are walked in series order and each converged run seeds the next
    (``_best_single_run_fit_result``'s first attempt), which is why this stays
    serial: along an ordered series the neighbour is the best available start,
    and the process pool is already saturated one level up, across templates.
    """

    free_names = tuple(name for name in template.model.param_names if name not in fixed_param_names)
    seeded = _canonicalize_parameter_sets(
        base_by_run,
        template=template,
        global_param_names=free_names,
        local_param_names=(),
        fixed_param_names=fixed_param_names,
    )
    results: dict[int, FitResult] = {}
    previous_success: ParameterSet | None = None
    for dataset in datasets:
        run_number = int(dataset.run_number)
        _record_counter(instrumentation, "separable_all_local_fits")
        best_result = _best_single_run_fit_result(
            dataset,
            template,
            fit_engine=fit_engine,
            seed_params=seeded[run_number],
            base_params=base_by_run[run_number],
            global_names=free_names,
            fixed_param_names=fixed_param_names,
            previous_success=previous_success,
        )
        if best_result is None:
            results[run_number] = FitResult(
                success=False,
                message="No single-run fit attempt produced a result.",
            )
            continue
        canonical = _canonicalize_fit_results_by_run(
            {run_number: best_result},
            template=template,
            global_param_names=free_names,
            local_param_names=(),
            fixed_param_names=fixed_param_names,
        )[run_number]
        results[run_number] = canonical
        if canonical.success:
            previous_success = canonical.parameters
    return results


def _run_estimates_from_results(
    datasets: list[MuonDataset],
    results_by_run: dict[int, FitResult],
    free_param_names: tuple[str, ...],
) -> tuple[RunEstimate, ...]:
    return tuple(
        run_estimate_from_fit_result(
            results_by_run[int(dataset.run_number)],
            free_param_names,
            run_number=int(dataset.run_number),
            n_points=int(dataset.n_points),
            at_bound=_bound_hit_names(results_by_run[int(dataset.run_number)].parameters),
        )
        for dataset in datasets
    )


def _run_separable_anchor_task(task: _SeparableAnchorTask) -> _SeparableTemplateResult:
    """Assemble one template's all-local node — the search's starting point.

    No coupled fit happens here: G independent per-run fits *are* the all-local
    assignment, and scoring them through :func:`_assemble_assignment_assessment`
    puts that node on exactly the same IC footing as every coupled node.
    """

    instrumentation = _fresh_task_instrumentation()
    template = task.template
    fixed_param_names = task.fixed_param_names
    free_param_names = tuple(
        name for name in template.model.param_names if name not in fixed_param_names
    )
    state = _HeuristicTemplateState(
        template=template,
        fixed_param_names=fixed_param_names,
        prefit_base_by_run=_clone_parameter_sets(task.base_by_run),
        free_param_names=free_param_names,
        exact_cache={},
        converged_assessments={},
    )

    if not free_param_names:
        # Nothing is promotable: the single all-fixed assignment is the verdict.
        _fit_heuristic_assignment(
            task.datasets,
            state,
            (),
            axis_key=task.axis_key,
            metric=task.metric,
            search_strategy=task.search_strategy,
            progress_callback=None,
            instrumentation=instrumentation,
        )
        return _SeparableTemplateResult(
            template_key=task.template_key,
            state=state,
            estimates=(),
            instrumentation=instrumentation,
        )

    if task.prescreen_results_by_run is None:
        results_by_run = _all_local_run_fit_results(
            task.datasets,
            template,
            fit_engine=FitEngine(),
            base_by_run=task.base_by_run,
            fixed_param_names=fixed_param_names,
            instrumentation=instrumentation,
        )
    else:
        _record_counter(instrumentation, "separable_all_local_from_prescreen")
        results_by_run = _canonicalize_fit_results_by_run(
            dict(task.prescreen_results_by_run),
            template=template,
            global_param_names=free_param_names,
            local_param_names=(),
            fixed_param_names=fixed_param_names,
        )

    local_param_names = _local_names_for(free_param_names, set(free_param_names))
    fit_success = len(results_by_run) == len(task.datasets) and all(
        result.success for result in results_by_run.values()
    )
    state.prefit_base_by_run = _merge_result_values_into_parameter_sets(
        task.base_by_run,
        results_by_run,
    )
    assessment = _assemble_assignment_assessment(
        task.datasets,
        template,
        base_by_run=state.prefit_base_by_run,
        results_by_run=results_by_run,
        fitted_global=ParameterSet(),
        global_param_names=(),
        local_param_names=local_param_names,
        fixed_param_names=fixed_param_names,
        axis_key=task.axis_key,
        metric=task.metric,
        fit_success=fit_success,
    )
    anchor_key = ((), local_param_names)
    state.exact_cache[anchor_key] = assessment
    state.anchor_assessment = assessment
    if assessment.is_successful:
        state.converged_assessments[anchor_key] = assessment
        state.best_assessment = assessment
        estimates = _run_estimates_from_results(task.datasets, results_by_run, free_param_names)
    else:
        # With no usable all-local node there is no surrogate and no warm start.
        # The template still deserves one real coupled row on the leaderboard, so
        # fit the cheapest assignment (all-global) from the base seeds.
        _record_counter(instrumentation, "separable_all_local_failed")
        _fit_heuristic_assignment(
            task.datasets,
            state,
            (),
            axis_key=task.axis_key,
            metric=task.metric,
            search_strategy=task.search_strategy,
            progress_callback=None,
            instrumentation=instrumentation,
        )
        estimates = ()

    return _SeparableTemplateResult(
        template_key=task.template_key,
        state=state,
        estimates=estimates,
        instrumentation=instrumentation,
    )


def _collapse_warm_start_parameter_sets(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    collapse: CollapseResult,
    *,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
) -> dict[int, ParameterSet]:
    """Turn a GLS collapse into per-run start values for the coupled fit.

    Shared parameters start at the pooled value; every other free parameter
    starts at its run's *conditional* value — the shift the collapse says it
    takes once the shared ones move. That is the second-order optimum of the
    coupled problem, which is why one screened migrad from here is enough.
    """

    seeded: dict[int, ParameterSet] = {}
    for dataset in datasets:
        run_number = int(dataset.run_number)
        parameters = _clone_parameter_set(state.prefit_base_by_run[run_number])
        conditional = collapse.conditional_locals_by_run[run_number]
        for parameter in parameters:
            value = collapse.shared_values.get(parameter.name, conditional.get(parameter.name))
            if value is None:
                continue
            parameter.value = float(np.clip(value, parameter.min, parameter.max))
        seeded[run_number] = parameters
    return _canonicalize_parameter_sets(
        seeded,
        template=state.template,
        global_param_names=global_param_names,
        local_param_names=local_param_names,
        fixed_param_names=state.fixed_param_names,
    )


def _separable_warm_start(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    estimates: tuple[RunEstimate, ...],
    incumbent: GlobalCandidateAssessment,
    *,
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
) -> dict[int, ParameterSet]:
    """The warm start for one candidate assignment.

    Globalising a parameter — the elimination's forward move, and the
    flip-neighbourhood's "share this one too" — is exactly what the collapse
    models, so it supplies the seed. Freeing a global back to per-run values is
    the reverse move, which the collapse says nothing about; there the
    incumbent's own fitted values, spread per run, are the honest start. A
    collapse whose pooled weight is unusable reports ``inf`` and hands back no
    shared values, so it falls to the same reverse-move seed.
    """

    collapse = collapse_cost(estimates, global_param_names)
    globalising = bool(set(global_param_names) - set(incumbent.global_param_names))
    if globalising and math.isfinite(collapse.delta_chi2):
        return _collapse_warm_start_parameter_sets(
            datasets,
            state,
            collapse,
            global_param_names=global_param_names,
            local_param_names=local_param_names,
        )
    return _warm_start_parameter_sets(
        datasets,
        assessment=incumbent,
        base_by_run=state.prefit_base_by_run,
        target_global_names=global_param_names,
        target_local_names=local_param_names,
        fit_engine=FitEngine(),
        template=state.template,
        progress_callback=None,
    )


def _separable_coupled_strategy(
    datasets: list[MuonDataset],
    global_param_names: tuple[str, ...],
    local_param_names: tuple[str, ...],
) -> str:
    """Which minimiser architecture solves this node: ``"joint"`` or ``"profiled"``.

    The joint solver builds one Minuit problem over ``n_global + n_local*G``
    parameters, so its Hessian cost grows as the square of that; the profiled
    solver replaces it with ``n_global**2`` plus ``G`` small per-dataset blocks,
    but pays for it by re-solving every dataset on *each* outer iteration. That
    trade only pays once the joint problem is genuinely large: on a short series
    the joint problem is already tiny and profiled's outer loop dominates
    (measured on the harness's 5-parameter, 3-run near-degenerate case, profiled
    cost roughly 5x the joint path for the same verdict). Above the threshold the
    joint Hessian is what dominates and profiled wins by the margin that motivates
    it in the first place.
    """

    free_count = len(global_param_names) + len(local_param_names) * len(datasets)
    return "profiled" if free_count >= _SEPARABLE_PROFILED_FREE_PARAM_COUNT else "joint"


def _fit_separable_assignment(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    local_names: tuple[str, ...],
    *,
    estimates: tuple[RunEstimate, ...],
    warm_start_source: GlobalCandidateAssessment,
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    instrumentation: dict[str, object],
    cancel_callback: Callable[[], bool] | None = None,
) -> GlobalCandidateAssessment:
    """One exact, warm-started, profiled coupled fit; cached on ``state``.

    Goes through :func:`_fit_exact_assignment` so its IC, diagnostics and
    instrumentation counters are identical to an exhaustive node's — only the
    seed and the minimiser architecture differ. The warm start is built here,
    *after* the cache lookup, because building one can itself cost a per-run
    prefit and an already-fitted node needs neither.
    """

    global_names = _global_names_for(state.free_param_names, local_names)
    cached = state.exact_cache.get((global_names, local_names))
    if cached is not None and cached.is_successful:
        _record_counter(instrumentation, "exact_fit_cache_hits")
        return cached
    strategy = _separable_coupled_strategy(datasets, global_names, local_names)
    _record_counter(instrumentation, f"separable_{strategy}_fits")
    warm_start_by_run = _separable_warm_start(
        datasets,
        state,
        estimates,
        warm_start_source,
        global_param_names=global_names,
        local_param_names=local_names,
    )
    assessment = _fit_exact_assignment(
        datasets,
        state.template,
        fit_engine=FitEngine(),
        base_by_run=state.prefit_base_by_run,
        global_param_names=global_names,
        local_param_names=local_names,
        fixed_param_names=state.fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        cache=state.exact_cache,
        warm_start_by_run=warm_start_by_run,
        progress_callback=None,
        search_strategy=search_strategy,
        strategy=strategy,
        warm_start_only=True,
        instrumentation=instrumentation,
        initial_step_sizes=_step_hints_from_assessment(
            datasets,
            warm_start_source,
            target_global_names=global_names,
            target_local_names=local_names,
        ),
        cancel_callback=cancel_callback,
    )
    key = (global_names, local_names)
    state.exact_cache[key] = assessment
    if assessment.is_successful:
        state.converged_assessments[key] = assessment
        if state.best_assessment is None or _assessment_sort_key(
            assessment, metric
        ) < _assessment_sort_key(state.best_assessment, metric):
            state.best_assessment = assessment
    return assessment


def _assessment_chi2(assessment: GlobalCandidateAssessment) -> float:
    return float(sum(result.chi_squared for result in assessment.fit_results_by_run.values()))


def _refit_certificate_violating_parent(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    parent: GlobalCandidateAssessment,
    child: GlobalCandidateAssessment,
    *,
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    instrumentation: dict[str, object],
) -> GlobalCandidateAssessment:
    """Redo a parent the child proved mis-converged, from the child's values.

    The child shares one more parameter than its parent and so has *fewer* free
    parameters; an honest pair therefore satisfies
    ``chi2(child) >= chi2(parent) - eps``. A violation is not evidence about the
    child — it says the parent settled in a worse minimum. The child's fitted
    values, spread back to per-run locals, are a start the parent cannot do worse
    from. Returns whichever of the two parent fits is better.
    """

    _record_counter(instrumentation, "separable_certificate_refits")
    parent_local = parent.local_param_names
    parent_global = parent.global_param_names
    warm_start_by_run = _warm_start_parameter_sets(
        datasets,
        assessment=child,
        base_by_run=state.prefit_base_by_run,
        target_global_names=parent_global,
        target_local_names=parent_local,
        fit_engine=FitEngine(),
        template=state.template,
        progress_callback=None,
    )
    refit = _fit_exact_assignment(
        datasets,
        state.template,
        fit_engine=FitEngine(),
        base_by_run=state.prefit_base_by_run,
        global_param_names=parent_global,
        local_param_names=parent_local,
        fixed_param_names=state.fixed_param_names,
        axis_key=axis_key,
        metric=metric,
        # A fresh cache: the mis-converged parent already sits in ``state``'s
        # cache and would otherwise be handed back instead of refitted.
        cache={},
        warm_start_by_run=warm_start_by_run,
        progress_callback=None,
        search_strategy=search_strategy,
        strategy="profiled",
        warm_start_only=True,
        instrumentation=instrumentation,
        initial_step_sizes=_step_hints_from_assessment(
            datasets,
            child,
            target_global_names=parent_global,
            target_local_names=parent_local,
        ),
    )
    if not refit.is_successful or _assessment_sort_key(parent, metric) <= _assessment_sort_key(
        refit, metric
    ):
        return parent
    key = (parent_global, parent_local)
    state.exact_cache[key] = refit
    state.converged_assessments[key] = refit
    if state.best_assessment is None or _assessment_sort_key(refit, metric) < _assessment_sort_key(
        state.best_assessment, metric
    ):
        state.best_assessment = refit
    return refit


def _separable_backward_elimination(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    estimates: tuple[RunEstimate, ...],
    *,
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    instrumentation: dict[str, object],
) -> GlobalCandidateAssessment:
    """Globalise one parameter at a time, cheapest first, while the IC improves.

    The incumbent starts at all-local. Each round asks the surrogate which single
    remaining parameter is cheapest to share, fits *that* assignment exactly, and
    keeps it only when the exact IC beats the incumbent. The first rejection ends
    the walk.

    A parameter resting on a bound in any run is never proposed: its curvature is
    not the curvature of an interior minimum, so the collapse would be modelling
    numerical noise rather than the parameter's spread.
    """

    # The walk's incumbent is the current node of the *chain*, which starts at
    # all-local — not ``state.best_assessment``, which may already hold phase 1's
    # adopted screening node from somewhere else in the assignment lattice.
    incumbent = state.anchor_assessment
    shared: tuple[str, ...] = ()
    at_bound: set[str] = set()
    for estimate in estimates:
        at_bound |= estimate.at_bound

    while True:
        candidates = [
            name for name in state.free_param_names if name not in shared and name not in at_bound
        ]
        if not candidates:
            return incumbent

        scored = sorted(
            (surrogate_ic(estimates, (*shared, name), metric), name) for name in candidates
        )
        predicted_ic, best_name = scored[0]
        if not math.isfinite(predicted_ic):
            # Every remaining single addition has an unusable pooled weight, so
            # the surrogate can no longer say which parameter to share next.
            _record_counter(instrumentation, "separable_surrogate_exhausted")
            return incumbent

        candidate_shared = tuple(
            name for name in state.free_param_names if name in {*shared, best_name}
        )
        local_names = _local_names_for(
            state.free_param_names,
            set(state.free_param_names) - set(candidate_shared),
        )
        _record_counter(instrumentation, "separable_steps")
        escalations_before = _counter_value(instrumentation, "warm_only_escalations")
        candidate = _fit_separable_assignment(
            datasets,
            state,
            local_names,
            estimates=estimates,
            warm_start_source=incumbent,
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
        )
        if _counter_value(instrumentation, "warm_only_escalations") > escalations_before:
            _record_counter(instrumentation, "separable_escalations")
        if not candidate.is_successful:
            # The warm fit failed and the multi-start battery behind it failed
            # too. Sharing yet more parameters would only start from a worse
            # seed, so the walk ends at the incumbent.
            return incumbent

        if _assessment_chi2(candidate) < _assessment_chi2(incumbent) - _WARM_CERTIFICATE_EPSILON:
            incumbent = _refit_certificate_violating_parent(
                datasets,
                state,
                incumbent,
                candidate,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                instrumentation=instrumentation,
            )

        if _assessment_sort_key(candidate, metric) >= _assessment_sort_key(incumbent, metric):
            return incumbent
        incumbent = candidate
        shared = candidate_shared


def _fill_separable_flip_neighbourhood(
    datasets: list[MuonDataset],
    state: _HeuristicTemplateState,
    winner: GlobalCandidateAssessment,
    estimates: tuple[RunEstimate, ...],
    *,
    axis_key: str,
    metric: SelectionMetric,
    search_strategy: str,
    instrumentation: dict[str, object],
) -> None:
    """Fit every single-role flip of ``winner`` the walk did not already visit.

    ``_build_parameter_recommendations_from_exact_cache`` justifies each
    parameter's role by comparing the winner against exactly these neighbours, so
    without them the verdict layer is starved. Elimination visits only the
    accepted chain plus its one rejected step, so most flips land here.
    """

    winner_local = set(winner.local_param_names)
    for name in state.free_param_names:
        if name in winner_local:
            flipped = _local_names_for(state.free_param_names, winner_local - {name})
        else:
            flipped = _local_names_for(state.free_param_names, winner_local | {name})
        global_names = _global_names_for(state.free_param_names, flipped)
        if (global_names, flipped) in state.exact_cache:
            continue
        _record_counter(instrumentation, "flip_neighbourhood_fits")
        _record_counter(instrumentation, "separable_flip_fits")
        _fit_separable_assignment(
            datasets,
            state,
            flipped,
            estimates=estimates,
            warm_start_source=winner,
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            instrumentation=instrumentation,
        )


def _run_separable_elimination_task(
    task: _SeparableEliminationTask,
) -> _SeparableTemplateResult:
    instrumentation = _fresh_task_instrumentation()
    state = task.state
    _separable_backward_elimination(
        task.datasets,
        state,
        task.estimates,
        axis_key=task.axis_key,
        metric=task.metric,
        search_strategy=task.search_strategy,
        instrumentation=instrumentation,
    )
    # The flips are centred on the template's best converged node, not on the
    # elimination's last incumbent: the walk stops at the first rejection, and a
    # node it rejected (or an adopted screening fit) may still be the best one in
    # the cache. A template only reaches elimination once its all-local anchor
    # converged, and that anchor is what sets ``best_assessment`` in the first
    # place, so there is always a node here.
    _fill_separable_flip_neighbourhood(
        task.datasets,
        state,
        state.best_assessment,
        task.estimates,
        axis_key=task.axis_key,
        metric=task.metric,
        search_strategy=task.search_strategy,
        instrumentation=instrumentation,
    )
    return _SeparableTemplateResult(
        template_key=task.template_key,
        state=state,
        estimates=task.estimates,
        instrumentation=instrumentation,
    )


def _drain_separable_tasks(
    tasks: Sequence[_SeparableAnchorTask] | Sequence[_SeparableEliminationTask],
    runner: Callable[..., _SeparableTemplateResult],
    *,
    activity: str,
    progress_callback: Callable[[str], None] | None,
    instrumentation: dict[str, object] | None,
    cancel_callback: Callable[[], bool] | None,
    deadline: float | None,
    shared_executor: ProcessPoolExecutor | None = None,
) -> list[_SeparableTemplateResult]:
    """Run per-template tasks on the spawn pool, polling cancel and the deadline.

    The wall budget is a backstop: a healthy search finishes on its merits long
    before it. On expiry the pool is torn down *without* waiting, so the wall we
    were bounding is not simply relocated into ``shutdown``, and whatever
    completed is kept.

    ``shared_executor`` is a pool somebody else owns and will close. Opening one
    costs a spawn per worker, which is nothing beside a template's coupled fits
    but everything beside a *series* of short searches: per-phase optimisation
    calls this twice per segment, so on the eight segments of a verified
    two-break path it is sixteen pools. Borrowing one is measurably the whole
    difference there (8.3 s to 1.4 s on the two-phase synthetic).
    """

    def _check_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Global fit wizard analysis cancelled.")

    def _budget_exceeded() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    results: list[_SeparableTemplateResult] = []
    executor = shared_executor
    if executor is None:
        worker_count = _template_worker_count(len(tasks))
        if worker_count > 1 and len(tasks) > 1:
            executor = _try_open_process_pool(
                max_workers=worker_count,
                progress_callback=progress_callback,
                activity=activity,
            )

    truncated = False
    try:
        if executor is None:
            for task in tasks:
                _check_cancelled()
                if _budget_exceeded():
                    truncated = True
                    break
                results.append(runner(task))
        else:
            pending = {executor.submit(runner, task) for task in tasks}
            while pending:
                _check_cancelled()
                if _budget_exceeded():
                    truncated = True
                    break
                done, pending = wait(
                    pending,
                    timeout=_WAVEFRONT_POLL_INTERVAL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    results.append(future.result())
    finally:
        # A borrowed pool outlives this call and is its owner's to close.
        if executor is not None and executor is not shared_executor:
            _shutdown_process_pool(executor, wait=not truncated, cancel_futures=truncated)

    if truncated:
        _progress_log(
            progress_callback,
            f"{activity} hit its wall-clock budget; keeping "
            f"{len(results)}/{len(tasks)} completed template(s).",
        )
        _record_counter(instrumentation, "separable_budget_truncations")
    for result in results:
        _merge_instrumentation(instrumentation, result.instrumentation)
    return results


def _separable_prescreen_results_for_template(
    datasets: list[MuonDataset],
    assessment: GlobalCandidateAssessment | None,
    *,
    resolution_matches: bool,
) -> dict[int, FitResult] | None:
    """The pre-screen's per-run fits, when they really *are* the all-local node.

    They qualify only when the pre-screen ran the independent per-run single-fit
    path (``prescreen_only``), every run converged, and the table sits at the
    search resolution — a full-resolution chi-squared inside an IC computed over
    a rebinned ``n`` is meaningless. Otherwise the caller fits the node itself.
    """

    if assessment is None or not assessment.prescreen_only or not resolution_matches:
        return None
    results: dict[int, FitResult] = {}
    for dataset in datasets:
        result = assessment.fit_results_by_run.get(int(dataset.run_number))
        if result is None or not result.success:
            return None
        results[int(dataset.run_number)] = result
    return results


def _adopt_screening_assessment(
    state: _HeuristicTemplateState,
    assessment: GlobalCandidateAssessment | None,
    *,
    metric: SelectionMetric,
) -> None:
    """Adopt phase 1's coupled screening fit as a node of the search's cache.

    Screening already fitted one role assignment of this template exactly.
    Re-fitting it during the search would spend a coupled fit rediscovering the
    same minimum and then land, a last-float-bit apart, next to the screening row
    on the leaderboard — so the two would compete over convergence noise. Adopting
    it instead makes the searched node and the screening row *the same fit*, gives
    the walk a better starting incumbent for the cross-template bound, and lets
    the finaliser attach the flip-neighbourhood role justification to it.

    A pre-screen row is independent per-run fits, not a coupled fit, and is never
    adopted; nor is a screening fit that did not converge.
    """

    if assessment is None or assessment.prescreen_only or not assessment.is_successful:
        return
    local_names = _local_names_for(state.free_param_names, set(assessment.local_param_names))
    global_names = _global_names_for(state.free_param_names, local_names)
    adopted = replace(
        assessment,
        global_param_names=global_names,
        local_param_names=local_names,
        fixed_param_names=state.fixed_param_names,
    )
    key = (global_names, local_names)
    state.exact_cache[key] = adopted
    state.converged_assessments[key] = adopted
    if state.best_assessment is None or _assessment_sort_key(
        adopted, metric
    ) < _assessment_sort_key(state.best_assessment, metric):
        state.best_assessment = adopted


def _run_separable_search(
    datasets: list[MuonDataset],
    *,
    shortlisted_templates: list[CandidateTemplate],
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
    prescreen_assessments: dict[str, GlobalCandidateAssessment],
    axis_key: str,
    metric: SelectionMetric,
    progress_callback: Callable[[str], None] | None,
    search_strategy: str,
    instrumentation: dict[str, object] | None,
    single_run_prefit_cache_for: Callable[
        [CandidateTemplate], dict[object, dict[int, ParameterSet]]
    ],
    cancel_callback: Callable[[], bool] | None = None,
    search_rebin_factor: int | None = None,
    prescreen_rebin_factor: int = 1,
    time_budget_seconds: float | None = _WAVEFRONT_TIME_BUDGET_SECONDS,
    shared_executor: ProcessPoolExecutor | None = None,
) -> tuple[GlobalCandidateAssessment, ...]:
    """Separable global/local role search — the default engine.

    Returns the same ``tuple[GlobalCandidateAssessment, ...]`` contract as the
    exhaustive wavefront (each template's converged assignments, carrying the
    per-parameter role recommendations resolved from that template's exact
    cache), so the verdict layer is engine-agnostic.

    ``search_rebin_factor`` is the series search resolution; by default the
    minimum bandwidth-aware factor across runs. ``prescreen_rebin_factor`` says
    what resolution the pre-screen table sits at — when the two agree its per-run
    fits *are* the all-local node and no fit is spent on it at all.

    ``shared_executor`` lets a caller that runs several searches back to back —
    per-phase optimisation does, once per segment — pay for one spawn pool rather
    than two per call.
    """

    if not shortlisted_templates:
        return ()

    if search_rebin_factor is None:
        search_rebin_factor = _separable_search_rebin_factor(datasets)
    _set_metric(instrumentation, "separable_search_rebin_factor", int(search_rebin_factor))
    search_datasets = _separable_search_datasets(
        datasets,
        search_rebin_factor=search_rebin_factor,
        instrumentation=instrumentation,
    )
    if search_rebin_factor > 1:
        _progress_log(
            progress_callback,
            f"Separable role search: searching at {search_rebin_factor}x coarser binning; "
            "the winner and its flip-neighbourhood are refitted at full resolution.",
        )

    deadline: float | None = None
    if time_budget_seconds is not None and time_budget_seconds > 0.0:
        deadline = time.monotonic() + float(time_budget_seconds)

    resolution_matches = int(prescreen_rebin_factor) == int(search_rebin_factor)
    anchor_tasks: list[_SeparableAnchorTask] = []
    for template in shortlisted_templates:
        base_by_run, fixed_param_names = template_contexts[template.key]
        anchor_tasks.append(
            _SeparableAnchorTask(
                template_key=template.key,
                template=template,
                datasets=search_datasets,
                base_by_run=base_by_run,
                fixed_param_names=fixed_param_names,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
                prescreen_results_by_run=_separable_prescreen_results_for_template(
                    search_datasets,
                    prescreen_assessments.get(template.key),
                    resolution_matches=resolution_matches,
                ),
            )
        )

    _progress_log(
        progress_callback,
        f"Separable role search: assembling all-local anchors for {len(anchor_tasks)} "
        "shortlisted template(s) without a joint fit.",
    )
    anchor_results = _drain_separable_tasks(
        anchor_tasks,
        _run_separable_anchor_task,
        activity="Separable role search (all-local anchors)",
        progress_callback=progress_callback,
        instrumentation=instrumentation,
        cancel_callback=cancel_callback,
        deadline=deadline,
        shared_executor=shared_executor,
    )
    result_by_key = {result.template_key: result for result in anchor_results}
    # A screening fit is a node of *this* search's lattice only when it was
    # measured over the same points the search measures. At a coarser search
    # resolution its chi-squared is computed over a different ``n``, so adopting
    # it would put a full-resolution IC next to search-resolution ones inside the
    # same state — the very mixing the resolution rule forbids.
    if resolution_matches:
        for result in anchor_results:
            _adopt_screening_assessment(
                result.state,
                prescreen_assessments.get(result.template_key),
                metric=metric,
            )
    states = [
        result_by_key[template.key].state
        for template in shortlisted_templates
        if template.key in result_by_key
    ]

    # --- Template racing -----------------------------------------------------
    # Rank by the better of what the pre-screen already measured and what the
    # surrogate predicts the best sharing pattern reaches. Only the top few earn
    # coupled fits; the rest keep their (free) all-local node on the leaderboard.
    # The pre-screen's measurement counts only when it shares the search's
    # resolution: comparing an IC over the full record against a surrogate IC over
    # a rebinned one ranks templates by their point count, not their fit.
    def _race_score(result: _SeparableTemplateResult) -> float:
        scores = [float("inf")]
        prescreen = prescreen_assessments.get(result.template_key)
        if resolution_matches and prescreen is not None:
            prescreen_score = float(prescreen.metric_value(metric))
            if np.isfinite(prescreen_score):
                scores.append(prescreen_score)
        ranked = rank_assignments(result.estimates, result.state.free_param_names, metric)
        if ranked:
            scores.append(float(ranked[0][1]))
        return min(scores)

    contenders = [
        result for result in anchor_results if result.estimates and result.state.free_param_names
    ]
    ranked_results = sorted(contenders, key=_race_score)
    advancing = ranked_results[:_SEPARABLE_RACING_ADVANCE_COUNT]
    if len(ranked_results) > len(advancing):
        _record_counter(
            instrumentation,
            "separable_templates_raced_out",
            len(ranked_results) - len(advancing),
        )
        _progress_log(
            progress_callback,
            f"Template racing advanced {len(advancing)}/{len(ranked_results)} template(s) "
            "into backward elimination; the rest keep their all-local score.",
        )

    # --- Cross-template incumbent bound (technique B) ------------------------
    # A template whose all-local chi-squared floor plus the smallest penalty any
    # of its assignments could carry still loses to the best converged IC found
    # so far cannot produce the winner, so it never earns a coupled fit.
    sample_count = int(sum(dataset.n_points for dataset in search_datasets))
    converged_ics = [
        float(result.state.best_assessment.metric_value(metric))
        for result in anchor_results
        if result.state.best_assessment is not None and result.state.best_assessment.is_successful
    ]
    cross_incumbent = min(converged_ics) if converged_ics else float("inf")

    elimination_tasks: list[_SeparableEliminationTask] = []
    for result in advancing:
        state = result.state
        best_possible_ic = _assessment_chi2(state.anchor_assessment) + _metric_penalty(
            _layer_parameter_count(
                0,
                free_param_count=len(state.free_param_names),
                n_datasets=len(search_datasets),
            ),
            sample_count=sample_count,
            metric=metric,
        )
        if (
            math.isfinite(cross_incumbent)
            and best_possible_ic > cross_incumbent + _LAYER_BOUND_MARGIN
        ):
            _record_counter(instrumentation, "cross_template_templates_pruned")
            _progress_log(
                progress_callback,
                f"{state.template.title}: cross-template bound fired (best possible "
                f"{best_possible_ic:.2f} > cross-incumbent {cross_incumbent:.2f} + "
                f"{_LAYER_BOUND_MARGIN:.1f}); skipping backward elimination.",
            )
            continue
        elimination_tasks.append(
            _SeparableEliminationTask(
                template_key=result.template_key,
                state=state,
                estimates=result.estimates,
                datasets=search_datasets,
                axis_key=axis_key,
                metric=metric,
                search_strategy=search_strategy,
            )
        )

    if elimination_tasks:
        _progress_log(
            progress_callback,
            f"Separable role search: backward elimination on {len(elimination_tasks)} template(s).",
        )
        elimination_results = _drain_separable_tasks(
            elimination_tasks,
            _run_separable_elimination_task,
            activity="Separable role search (backward elimination)",
            progress_callback=progress_callback,
            instrumentation=instrumentation,
            cancel_callback=cancel_callback,
            deadline=deadline,
            shared_executor=shared_executor,
        )
        searched_by_key = {result.template_key: result.state for result in elimination_results}
        states = [searched_by_key.get(state.template.key, state) for state in states]

    if search_rebin_factor > 1:
        states = _refit_states_at_full_resolution(
            datasets,
            states,
            template_contexts=template_contexts,
            single_run_prefit_cache_for=single_run_prefit_cache_for,
            axis_key=axis_key,
            metric=metric,
            search_strategy=search_strategy,
            progress_callback=progress_callback,
            instrumentation=instrumentation,
        )

    return _finalise_heuristic_assessments(
        datasets, states, metric=metric, progress_callback=progress_callback
    )


# --------------------------------------------------------------------------- #
# Per-phase optimisation (tier 3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _PhaseWindow:
    """One segment of a *candidate* partition, as positions into the series order.

    Deliberately not a :class:`Segment`: a shifted-break candidate has no cost or
    structure until its segments have been fitted, and a placeholder ``inf`` on a
    ``Segment`` would be indistinguishable from a segment nothing could describe.
    """

    start: int
    stop: int
    excluded: bool


def _windows_of(solution: PartitionSolution) -> tuple[_PhaseWindow, ...]:
    return tuple(
        _PhaseWindow(segment.start, segment.stop, segment.excluded) for segment in solution.segments
    )


def _shifted_break_variants(
    windows: tuple[_PhaseWindow, ...], min_segment: int
) -> list[tuple[_PhaseWindow, ...]]:
    """Every one-run shift of a single break that keeps each phase legal.

    The elbow is a claim about *where* the series changes as much as about how
    many times, and the DP found the break from a closed-form surrogate. Fitting
    the two neighbouring positions is what turns "the surrogate liked run 16" into
    "the exact fit prefers run 16 to runs 15 and 17".

    Only breaks between two ordinary phases move. An end stub is short *because*
    it is excluded, so a shift into it would have to redefine what it is; it is
    left where the DP put it.
    """

    variants: list[tuple[_PhaseWindow, ...]] = []
    for index in range(len(windows) - 1):
        left, right = windows[index], windows[index + 1]
        if left.excluded or right.excluded:
            continue
        for delta in (-1, 1):
            boundary = left.stop + delta
            if boundary - left.start < min_segment or right.stop - boundary < min_segment:
                continue
            variants.append(
                (
                    *windows[:index],
                    _PhaseWindow(left.start, boundary, False),
                    _PhaseWindow(boundary, right.stop, False),
                    *windows[index + 2 :],
                )
            )
    return variants


def _restrict_prescreen_assessment(
    assessment: GlobalCandidateAssessment,
    datasets: Sequence[MuonDataset],
    *,
    metric: SelectionMetric,
    analysed_points_by_run: Mapping[int, int],
) -> GlobalCandidateAssessment:
    """One phase-1 pre-screen row, restricted to a segment of the series.

    The per-run fits carry over verbatim — they are independent fits, so a
    segment's rows really are its own — but the information criteria are
    recomputed over the segment's runs and point count. A series-wide IC says
    nothing about how a template does on one phase, and it is what the separable
    search's template racing reads.
    """

    run_numbers = [int(dataset.run_number) for dataset in datasets]
    wanted = set(run_numbers)
    free_names = tuple(
        name
        for name in assessment.template.model.param_names
        if name not in assessment.fixed_param_names
    )
    results = {
        run_number: assessment.fit_results_by_run[run_number]
        for run_number in run_numbers
        if run_number in assessment.fit_results_by_run
    }
    complete = len(results) == len(run_numbers) and all(
        result.success for result in results.values()
    )
    if complete:
        aic, aicc, bic = compute_information_criteria(
            math.fsum(float(result.chi_squared) for result in results.values()),
            len(free_names) * len(run_numbers),
            sum(int(analysed_points_by_run[run_number]) for run_number in run_numbers),
        )
    else:
        aic, aicc, bic = float("inf"), None, float("inf")

    return replace(
        assessment,
        fit_results_by_run=results,
        run_diagnostics=tuple(
            diagnostic
            for diagnostic in assessment.run_diagnostics
            if diagnostic.run_number in wanted
        ),
        aic=float(aic),
        aicc=None if aicc is None else float(aicc),
        bic=float(bic),
        selected_score=_metric_value(metric, aic, aicc, bic),
        fitted_curves_by_run={
            run_number: curves
            for run_number, curves in assessment.fitted_curves_by_run.items()
            if run_number in wanted
        },
        component_curves_by_run={
            run_number: curves
            for run_number, curves in assessment.component_curves_by_run.items()
            if run_number in wanted
        },
    )


def _partition_bic(
    assessment: GlobalCandidateAssessment, points_by_run: Mapping[int, int]
) -> float:
    """BIC of one fitted phase under the partition convention.

    ``Σ_r χ²_r + n_shared·ln N_phase + n_local·Σ_r ln n_r`` — the convention
    :meth:`~asymmetry.core.fitting.global_search.surrogate.OrderedCollapse.partition_ic`
    scores the surrogate path with, so an exactly refitted row and a surrogate
    row sit on one scale. ``points_by_run`` is each run's fitted point count
    at the resolution the assessment was fitted at.
    """

    chi_squared = math.fsum(
        float(result.chi_squared) for result in assessment.fit_results_by_run.values()
    )
    points = [int(points_by_run[run]) for run in assessment.fit_results_by_run]
    shared = len(assessment.global_param_names)
    local = len(assessment.local_param_names)
    return (
        chi_squared
        + shared * math.log(max(sum(points), 1))
        + local * math.fsum(math.log(max(count, 1)) for count in points)
    )


def _recommended_segment_assessment(
    assessments: Sequence[GlobalCandidateAssessment],
    metric: SelectionMetric,
) -> GlobalCandidateAssessment | None:
    """The phase's answer, chosen exactly as the series-wide verdict is chosen.

    Same two tiers as :func:`rerank_global_fit_wizard_recommendation`: candidates
    that converged and cleared every gate, else candidates that converged and
    cleared every *per-run* gate with only a series-consistency caveat. ``None``
    means nothing describes this phase — which makes the candidate partition
    containing it infeasible, and is a real answer rather than a failure.
    """

    passing = [
        assessment
        for assessment in assessments
        if assessment.is_successful and assessment.residual_gate_passed
    ]
    if not passing:
        passing = [
            assessment
            for assessment in assessments
            if assessment.is_successful
            and assessment.run_diagnostics
            and all(diagnostic.gate_passed for diagnostic in assessment.run_diagnostics)
        ]
    if not passing:
        return None
    return min(passing, key=lambda assessment: _assessment_sort_key(assessment, metric))


def _optimise_partition_phases(
    ordered_datasets: list[MuonDataset],
    *,
    path: PartitionPath,
    partition_k: int,
    templates: list[CandidateTemplate],
    template_contexts: dict[str, tuple[dict[int, ParameterSet], tuple[str, ...]]],
    prescreen_assessments: dict[str, GlobalCandidateAssessment],
    analysed_points_by_run: Mapping[int, int],
    axis_key: str,
    metric: SelectionMetric,
    search_rebin_factor: int,
    progress_callback: Callable[[str], None] | None,
    search_strategy: str,
    instrumentation: dict[str, object] | None,
    single_run_prefit_cache_for: Callable[
        [CandidateTemplate], dict[object, dict[int, ParameterSet]]
    ],
    cancel_callback: Callable[[], bool] | None,
    config: PartitionConfig = PartitionConfig(),
) -> tuple[PartitionPath, dict[tuple[int, int], GlobalCandidateAssessment], int]:
    """Tier 3: fit each phase of the selected partition, and verify the elbow.

    Verification covers the elbow's neighbours in **both** senses — ``k*−1``,
    ``k*`` and ``k*+1`` where they exist, and each break of ``k*`` shifted one run
    either way — because a partition can be wrong about how many transitions there
    are or about where one of them is, and the closed-form path cannot tell those
    apart. Every distinct segment across all of those is fitted **once**
    (neighbouring solutions share most of their segments), and the path rows they
    touch are re-scored with the exact per-segment BIC, still on the BIC scale
    (:data:`_PARTITION_METRIC`) whatever ranking metric the user chose. ``selected_k``
    is then re-derived from the exact gains.

    An excluded end stub never receives a coupled fit and keeps its per-run cost
    from the closed-form path, which is precisely what "not part of any phase"
    means. Rows outside the verified window keep their surrogate totals; a gain
    that straddles the edge therefore compares an exact total against a surrogate
    one, which is honest — the search measured what it could afford to measure.

    Segments run serially here and each one's templates fan out over the existing
    spawn pool. Pools must not nest, so the choice is which level gets the
    workers, and the alphabet is always larger than the number of phases: two
    segment-level tasks would leave most of ``_MAX_TEMPLATE_WORKERS`` idle where
    the per-template fan-out fills them.
    """

    solutions = path.solutions
    candidates: dict[int, list[tuple[_PhaseWindow, ...]]] = {}
    for k in (partition_k - 1, partition_k, partition_k + 1):
        if 0 <= k < len(solutions):
            candidates[k] = [_windows_of(solutions[k])]
    candidates[partition_k].extend(
        _shifted_break_variants(candidates[partition_k][0], config.min_segment)
    )

    stub_ic: dict[tuple[int, int], float] = {
        (segment.start, segment.stop): float(segment.ic)
        for solution in solutions
        for segment in solution.segments
        if segment.excluded
    }
    stub_structure: dict[tuple[int, int], str] = {
        (segment.start, segment.stop): segment.structure
        for solution in solutions
        for segment in solution.segments
        if segment.excluded
    }

    targets = sorted(
        {
            (window.start, window.stop)
            for windows_list in candidates.values()
            for windows in windows_list
            for window in windows
            if not window.excluded
        }
    )
    _set_metric(instrumentation, "partition_segments_fitted", len(targets))
    _progress_log(
        progress_callback,
        f"Optimising {len(targets)} distinct phase(s) across the {partition_k}-break "
        "solution and its verified neighbours.",
    )

    searched_by_window: dict[tuple[int, int], tuple[GlobalCandidateAssessment, ...]] = {}
    worker_count = _template_worker_count(len(templates))
    executor = (
        _try_open_process_pool(
            max_workers=worker_count,
            progress_callback=progress_callback,
            activity="Per-phase separable role search",
        )
        if worker_count > 1 and len(templates) > 1
        else None
    )
    try:
        for start, stop in targets:
            if cancel_callback is not None and cancel_callback():
                raise FitCancelledError("Global fit wizard analysis cancelled.")
            segment_datasets = ordered_datasets[start:stop]
            segment_runs = {int(dataset.run_number) for dataset in segment_datasets}
            _progress_log(
                progress_callback,
                f"Phase {segment_datasets[0].run_label}–{segment_datasets[-1].run_label}: "
                f"separable role search over {len(templates)} candidate(s).",
            )
            searched_by_window[(start, stop)] = _run_separable_search(
                segment_datasets,
                shortlisted_templates=list(templates),
                template_contexts={
                    key: (
                        {
                            run_number: parameters
                            for run_number, parameters in base_by_run.items()
                            if run_number in segment_runs
                        },
                        fixed_param_names,
                    )
                    for key, (base_by_run, fixed_param_names) in template_contexts.items()
                },
                prescreen_assessments={
                    key: _restrict_prescreen_assessment(
                        assessment,
                        segment_datasets,
                        metric=metric,
                        analysed_points_by_run=analysed_points_by_run,
                    )
                    for key, assessment in prescreen_assessments.items()
                },
                axis_key=axis_key,
                metric=metric,
                progress_callback=progress_callback,
                search_strategy=search_strategy,
                instrumentation=instrumentation,
                single_run_prefit_cache_for=single_run_prefit_cache_for,
                cancel_callback=cancel_callback,
                search_rebin_factor=search_rebin_factor,
                prescreen_rebin_factor=search_rebin_factor,
                shared_executor=executor,
            )
    except BaseException:
        if executor is not None:
            terminate_spawn_pool(executor)
        raise
    else:
        if executor is not None:
            _shutdown_process_pool(executor)

    # The exact rows are refitted at full resolution, so they are scored with
    # each run's full point count.
    full_points_by_run = {
        int(dataset.run_number): int(dataset.n_points) for dataset in ordered_datasets
    }

    def _score(
        windows: tuple[_PhaseWindow, ...],
    ) -> tuple[float, tuple[GlobalCandidateAssessment | None, ...]]:
        total = 0.0
        chosen: list[GlobalCandidateAssessment | None] = []
        for window in windows:
            if window.excluded:
                total += stub_ic[(window.start, window.stop)]
                chosen.append(None)
                continue
            assessment = _recommended_segment_assessment(
                searched_by_window[(window.start, window.stop)], metric
            )
            if assessment is None:
                return math.inf, ()
            total += _partition_bic(assessment, full_points_by_run)
            chosen.append(assessment)
        return total, tuple(chosen)

    phase_assessments: dict[tuple[int, int], GlobalCandidateAssessment] = {}
    rescored: dict[int, PartitionSolution] = {}
    for k, windows_list in candidates.items():
        scored = [(_score(windows), windows) for windows in windows_list]
        (total, chosen), windows = min(scored, key=lambda item: item[0][0])
        if not math.isfinite(total):
            continue
        segments: list[Segment] = []
        for index, (window, assessment) in enumerate(zip(windows, chosen, strict=True)):
            run_numbers = tuple(
                int(dataset.run_number) for dataset in ordered_datasets[window.start : window.stop]
            )
            if assessment is None:
                segments.append(
                    Segment(
                        start=window.start,
                        stop=window.stop,
                        run_numbers=run_numbers,
                        structure=stub_structure[(window.start, window.stop)],
                        ic=stub_ic[(window.start, window.stop)],
                        excluded=True,
                    )
                )
                continue
            phase_assessments[(k, index)] = assessment
            segments.append(
                Segment(
                    start=window.start,
                    stop=window.stop,
                    run_numbers=run_numbers,
                    structure=assessment.selection_key,
                    ic=_partition_bic(assessment, full_points_by_run),
                    excluded=False,
                )
            )
        rescored[k] = replace(
            solutions[k],
            breaks=len(segments) - 1,
            segments=tuple(segments),
            total_ic=total,
            boundaries=_partition_boundaries(ordered_datasets, segments, axis_key),
        )

    updated = _rescored_partition_path(path, rescored)
    # The elbow may move onto a neighbour tier 3 measured — that is what verifying
    # the neighbours is for. It is never followed outside the verified window,
    # where the gains are still the surrogate's.
    recommended_k = updated.selected_k if updated.selected_k in rescored else partition_k
    return updated, phase_assessments, recommended_k


def _partition_boundaries(
    ordered_datasets: Sequence[MuonDataset],
    segments: Sequence[Segment],
    axis_key: str,
) -> tuple[tuple[float, float], ...]:
    """``((x_a + x_b)/2, (x_b − x_a)/2)`` for each adjacent pair, as tier 1 does."""

    axis_values = {
        int(dataset.run_number): _axis_value(dataset, axis_key) for dataset in ordered_datasets
    }
    return tuple(
        (
            0.5 * (axis_values[left.run_numbers[-1]] + axis_values[right.run_numbers[0]]),
            0.5 * (axis_values[right.run_numbers[0]] - axis_values[left.run_numbers[-1]]),
        )
        for left, right in zip(segments, segments[1:], strict=False)
    )


def _rescored_partition_path(
    path: PartitionPath,
    rescored: Mapping[int, PartitionSolution],
) -> PartitionPath:
    """Replace the rows tier 3 measured, then re-read the elbow off the new gains.

    The gains are recomputed for the whole path so the exact rows and the
    surrogate rows around them stay one sequence, and ``selected_k`` is re-derived
    by the same rule the closed-form path uses: the largest ``k`` whose gains are
    *all* admissible, so a rejected break still ends the path.
    """

    solutions: list[PartitionSolution] = []
    previous_total = math.inf
    for index, solution in enumerate(path.solutions):
        current = rescored.get(index, solution)
        gain = 0.0 if index == 0 else previous_total - current.total_ic
        solutions.append(
            replace(
                current,
                gain=gain,
                admissible=True if index == 0 else gain >= path.beta_floor,
            )
        )
        previous_total = current.total_ic

    selected_k = 0
    for index in range(1, len(solutions)):
        if not solutions[index].admissible:
            break
        selected_k = index

    return replace(path, solutions=tuple(solutions), selected_k=selected_k)


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
    cancel_callback: Callable[[], bool] | None = None,
    time_budget_seconds: float | None = _WAVEFRONT_TIME_BUDGET_SECONDS,
) -> tuple[GlobalCandidateAssessment, ...]:
    if not shortlisted_templates:
        return ()

    deadline: float | None = None
    if time_budget_seconds is not None and time_budget_seconds > 0.0:
        deadline = time.monotonic() + float(time_budget_seconds)

    def _budget_exceeded() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _check_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Global fit wizard analysis cancelled.")

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
        # Cooperative cancel between templates (in-process anchor fits).
        _check_cancelled()
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
            cancel_callback=cancel_callback,
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
            # The admissible layer bound cannot arm without a converged anchor
            # (no chi2_floor). Rather than let this template enumerate every
            # Hamming layer (2^P assignments) — the dominant cost of the hang on
            # hard oscillatory/KT families — impose a conservative layer cap for
            # high-dimensional templates: keep the low, mostly-global layers (the
            # physically-plausible role splits for a series) and drop the
            # combinatorially-worst upper layers. Small templates are left
            # uncapped and still enumerate fully.
            if state.free_param_count > _ANCHOR_FAILED_CAP_FREE_PARAMS:
                state.layer_cap = _ANCHOR_FAILED_MAX_LAYERS
                _progress_log(
                    progress_callback,
                    f"{state.template.title}: all-local anchor did not converge; "
                    "no admissible layer bound, so capping enumeration to Hamming "
                    f"layer {state.layer_cap}/{state.free_param_count} "
                    "(conservative pruning instead of full 2^P enumeration).",
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

    #: Set when the wall-clock budget expires. On truncation we stop scheduling
    #: further work and return the best-so-far assessments; the pool is torn down
    #: NON-blocking (cancel_futures) so teardown does not re-block on the in-flight
    #: fits that the budget was meant to bound.
    budget_truncated = False
    try:
        for round_index in range(max_rounds):
            # Cooperative cancel between Hamming layers. Under a process pool we
            # cannot kill in-flight futures, so cancel stops scheduling further
            # rounds/templates, then the finally-block shuts the pool down.
            _check_cancelled()
            if _budget_exceeded():
                budget_truncated = True
                _progress_log(
                    progress_callback,
                    "Wavefront role search hit its wall-clock budget "
                    f"({time_budget_seconds:.0f} s) before round {round_index + 1}"
                    f"/{max_rounds}; returning the best assignments found so far.",
                )
                break
            task_groups: list[list[_WavefrontAssignmentTask]] = []
            for state in states:
                layers = layers_by_key[state.template.key]
                if round_index >= len(layers):
                    continue
                if state.layer_bound_fired:
                    # Bound already fired in an earlier round; every remaining
                    # (higher) layer only adds penalty, so skip them all.
                    continue
                if state.layer_cap is not None and round_index > state.layer_cap:
                    # Conservative cap for an anchor-failed high-dimensional
                    # template: the admissible bound could not arm, so cap the
                    # enumeration at the low (mostly-global) layers instead of
                    # exploring all 2^P assignments.
                    if not state.layer_bound_fired:
                        state.layer_bound_fired = True
                        _progress_log(
                            progress_callback,
                            f"{state.template.title}: enumeration cap reached at "
                            f"Hamming layer {round_index}/{state.free_param_count}; "
                            "skipping remaining layers (anchor-failed conservative "
                            "pruning).",
                        )
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
                if state.free_param_count > 0 and round_index == state.free_param_count:
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

            # Cooperative cancel before dispatching this layer's assignment fits.
            _check_cancelled()

            round_results: list[_WavefrontAssignmentResult] = []
            if executor is None:
                for task in ordered_tasks:
                    # Serial (in-process) path: check cancel and the wall-clock
                    # budget between per-assignment fits, keeping whatever has
                    # already been fitted so best-so-far includes it.
                    _check_cancelled()
                    if _budget_exceeded():
                        budget_truncated = True
                        _progress_log(
                            progress_callback,
                            "Wavefront role search hit its wall-clock budget "
                            f"({time_budget_seconds:.0f} s) mid-round "
                            f"{round_index + 1}/{max_rounds}; keeping "
                            f"{len(round_results)} completed assignment(s) and "
                            "stopping.",
                        )
                        break
                    round_results.append(_run_wavefront_assignment_task(task))
            else:
                future_to_task = {
                    executor.submit(_run_wavefront_assignment_task, task): task
                    for task in ordered_tasks
                }
                # Poll cancel and the wall-clock deadline WHILE the pool works,
                # rather than blocking in ``as_completed`` with no deadline. On
                # budget expiry we stop collecting and let the (non-blocking)
                # teardown cancel the still-pending futures — otherwise a single
                # slow, non-convergent oscillatory fit could keep the loop alive
                # past the budget indefinitely.
                pending = set(future_to_task)
                while pending:
                    _check_cancelled()
                    if _budget_exceeded():
                        budget_truncated = True
                        _progress_log(
                            progress_callback,
                            "Wavefront role search hit its wall-clock budget "
                            f"({time_budget_seconds:.0f} s) mid-round "
                            f"{round_index + 1}/{max_rounds} with "
                            f"{len(pending)} assignment(s) still in flight; "
                            f"keeping {len(round_results)} completed assignment(s) "
                            "and stopping.",
                        )
                        break
                    done, pending = wait(
                        pending,
                        timeout=_WAVEFRONT_POLL_INTERVAL_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
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

            if budget_truncated:
                # Best-so-far has been merged from the completed results above;
                # stop scheduling further rounds.
                break
    finally:
        if executor is not None:
            if budget_truncated:
                # NON-blocking teardown: a bare shutdown() waits on every
                # in-flight fit, which would just relocate the wall we were
                # trying to bound. cancel_futures drops the not-yet-started
                # tasks; the running ones are abandoned.
                _shutdown_process_pool(executor, wait=False, cancel_futures=True)
            else:
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
