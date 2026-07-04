"""Core analysis helpers for the single-spectrum fit wizard."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import (
    ComputationalCost,
    geometry_from_field_direction,
)
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
from asymmetry.core.fitting.envelope_match import match_envelope_banks
from asymmetry.core.fitting.muonium import VACUUM_MUONIUM_A_HF_MHZ
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.fitting.peak_detection import (
    DetectedPeak,
    MultipletMatch,
    PeakAnalysis,
    analyze_dataset_peaks,
    deserialize_multiplet_match,
    deserialize_peak_analysis,
    effective_analysis_window,
    match_multiplets,
    merge_user_peaks,
    serialize_multiplet_match,
    serialize_peak_analysis,
)
from asymmetry.core.fitting.process_pool import open_spawn_pool
from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz
from asymmetry.core.fitting.wizard_scope import (
    ScopeResolution,
    WizardScope,
    dataset_suggests_fluorine,
    resolve_scope_for_dataset,
)
from asymmetry.core.fourier.fft import fft_asymmetry

# ``ComputationalCost`` is a ``str``-Enum, so ``max()``/``<`` compares members
# alphabetically ("cheap" < "expensive" < "moderate") — wrong. Rank explicitly.
# (Mirrors ``wizard_scope._COST_RANK``; kept local so we do not reach into that
# module's private name.)
_COST_RANK: dict[ComputationalCost, int] = {
    ComputationalCost.CHEAP: 0,
    ComputationalCost.MODERATE: 1,
    ComputationalCost.EXPENSIVE: 2,
}

_EPS = 1e-12
_DEFAULT_FFT_WINDOW = "hann"
_DEFAULT_FFT_PADDING = 4
_COMPARABLE_SCORE_DELTA = 2.0
_FIT_WIZARD_TITLES = {
    "exp_constant": "Exponential + Constant",
    "biexp_constant": "Exponential + Exponential + Constant",
    "exp_gaussian_constant": "Exponential + Gaussian + Constant",
    "gaussian_constant": "Gaussian + Constant",
    "double_gaussian_constant": "Gaussian + Gaussian + Constant",
    "triple_exp_constant": "Exponential + Exponential + Exponential + Constant",
    "double_exp_gaussian_constant": "Exponential + Exponential + Gaussian + Constant",
    "exp_double_gaussian_constant": "Exponential + Gaussian + Gaussian + Constant",
    "triple_gaussian_constant": "Gaussian + Gaussian + Gaussian + Constant",
    "stretched_constant": "Stretched Exponential + Constant",
    "static_gkt_constant": "Static GKT + Constant",
    "dynamic_gkt_constant": "Dynamic GKT + Constant",
    "static_gkt_exp_constant": "Static GKT * Exponential + Constant",
    "oscillatory_exp_constant": "Oscillatory * Exponential + Constant",
    "oscillatory_gaussian_constant": "Oscillatory * Gaussian + Constant",
    "bessel_exp_constant": "Bessel * Exponential + Constant",
    "gbkt_constant": "Gaussian-broadened KT + Constant",
    "risch_kehr_constant": "Risch-Kehr + Constant",
    "current_model": "Current Fit Function",
    # Tiered-screening family templates (build_wizard_families).
    "dynamic_lkt_constant": "Dynamic Lorentzian KT + Constant",
    "lf_kt_constant": "Longitudinal-field KT + Constant",
    "vortex_lattice_constant": "Vortex Lattice + Constant",
    "vortex_lattice_powder_constant": "Vortex Lattice (powder) + Constant",
    "muonium_low_tf_constant": "Muonium (low TF) + Constant",
    "muonium_tf_constant": "Muonium (TF) + Constant",
    "muonium_high_tf_constant": "Muonium (high TF) + Constant",
    "muonium_zf_constant": "Muonium (ZF) + Constant",
    "muonium_lf_relax_constant": "Muonium LF relaxation + Constant",
    "fmuf_linear_exp_constant": "F-mu-F (collinear) * Exponential + Constant",
    "muf_exp_constant": "mu-F * Exponential + Constant",
    "dynamic_fmuf_constant": "Dynamic F-mu-F + Constant",
    "fmuf_general_constant": "F-mu-F (general) + Constant",
    "dipolar_pair_constant": "Dipolar pair field + Constant",
    "baseline": "Current model",
}


class ConfidenceTier(str, Enum):
    """Confidence a recommendation carries, decoupled from the veto policy.

    The residual gates no longer suppress a recommendation; they classify it.
    ``HIGH`` means the recommended template's residuals passed every gate;
    ``MEDIUM`` means it is the clear metric winner but leaves structured
    residuals (a caveat, surfaced to the user, not a veto). ``NONE`` is carried
    when there is no confident recommendation at all (out of scope, no
    successful fit, or the null-baseline verdict fired).
    """

    HIGH = "high"
    MEDIUM = "medium"
    NONE = "none"

    @classmethod
    def from_value(cls, value: object) -> ConfidenceTier:
        if isinstance(value, cls):
            return value
        # A missing field (``None``) takes the default explicitly — do not let
        # ``str(None) == "none"`` masquerade as a real serialized value.
        if value is None:
            return cls.NONE
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        return cls.NONE


class RecommendationVerdict(str, Enum):
    """Top-level meaning of a recommendation, orthogonal to the metric ranking.

    ``STRUCTURED`` — the recommended template describes real structure (the
    normal case). ``NO_SIGNIFICANT_STRUCTURE`` — the best template does not beat
    a strictly-simpler null baseline by a meaningful AICc margin, so the data are
    consistent with a plain relaxation/constant and no oscillatory or multi-rate
    claim is warranted (fixes pure-noise over-confidence, F6). ``NONE`` — no
    recommendation could be formed (out of scope or no successful fit).
    """

    STRUCTURED = "structured"
    NO_SIGNIFICANT_STRUCTURE = "no_significant_structure"
    NONE = "none"

    @classmethod
    def from_value(cls, value: object) -> RecommendationVerdict:
        if isinstance(value, cls):
            return value
        # A missing field (``None``) means the payload predates this enum, so it
        # defaults to STRUCTURED — not the "none" member that ``str(None)`` would
        # otherwise select.
        if value is None:
            return cls.STRUCTURED
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        return cls.STRUCTURED


class SelectionMetric(str, Enum):
    """Model-selection metrics exposed in the fit wizard."""

    AIC = "AIC"
    AICC = "AICc"
    BIC = "BIC"

    @classmethod
    def from_value(cls, value: object) -> SelectionMetric:
        text = str(value).strip()
        for metric in cls:
            if metric.value == text:
                return metric
        return cls.AICC


@dataclass(frozen=True)
class SpectrumFingerprint:
    """Deterministic summary of a single asymmetry spectrum."""

    tail_estimate: float
    initial_amplitude_estimate: float
    zero_crossings: int
    smoothed_zero_crossings: int
    smoothed_turning_points: int
    dominant_fft_frequency_mhz: float
    dominant_fft_snr: float
    dominant_fft_cycles_in_window: float
    monotonic_decay_fraction: float
    early_time_curvature: float
    semilog_slope_ratio: float
    late_time_dip_recovery_score: float
    oscillatory_hint: bool
    kt_like_hint: bool
    multi_rate_hint: bool


@dataclass(frozen=True)
class CandidateTemplate:
    """One curated candidate model considered by the fit wizard."""

    key: str
    title: str
    category: str
    rationale: str
    model: CompositeModel
    is_current_model_baseline: bool = False

    @property
    def parameter_count(self) -> int:
        return len(self.model.param_names)

    @property
    def additive_terms(self) -> int:
        return len(self.model.additive_component_indices())


@dataclass(frozen=True)
class CandidateAssessment:
    """Fit and comparison data for one candidate model."""

    template: CandidateTemplate
    fit_result: FitResult
    aic: float
    aicc: float | None
    bic: float
    selected_score: float
    residual_rms: float
    runs_z_score: float
    max_abs_autocorrelation: float
    residual_fft_peak_snr: float
    residual_gate_passed: bool
    residual_gate_reasons: tuple[str, ...]
    bound_hits: tuple[str, ...]
    fitted_time: NDArray[np.float64]
    fitted_curve: NDArray[np.float64]
    component_curves: tuple[tuple[str, NDArray[np.float64]], ...]
    repair_attempted: bool = False
    #: Screening stage that produced this assessment: 1 = Stage-1 family
    #: representative (cheap first-pass fit), 2 = full assessment.
    stage: int = 2
    #: Targeted-disqualifier reasons that make this candidate unfit to recommend
    #: even when it wins by metric (frequency at the resolution floor / pinned at
    #: a bound, oscillation amplitude consistent with zero). Empty when the
    #: candidate is eligible. Additive — old payloads deserialize to ``()``.
    disqualification_reasons: tuple[str, ...] = ()
    #: True for the flat/plain-exponential null baselines fitted unconditionally
    #: as a "no significant structure" reference; these never win a normal
    #: recommendation and are excluded from the ranked candidate pool.
    is_null_baseline: bool = False

    @property
    def is_disqualified(self) -> bool:
        return bool(self.disqualification_reasons)

    @property
    def parameter_count(self) -> int:
        return len(self.fit_result.parameters.free_parameters)

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
        return bool(self.fit_result.success)


@dataclass(frozen=True)
class FamilyScreeningReport:
    """Outcome of Stage-1 screening for one wizard family.

    ``stage1_metric_value`` may be ``math.inf`` when the representative fit
    failed; ``promoted`` records whether the family advanced to Stage-2.
    """

    family_key: str
    title: str
    stage1_template_key: str
    stage1_metric_value: float
    stage1_gate_passed: bool
    promoted: bool
    reason: str
    stage2_template_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class FitWizardRecommendation:
    """Full fit-wizard analysis payload plus the current recommendation."""

    fingerprint: SpectrumFingerprint
    templates: tuple[CandidateTemplate, ...]
    assessments: tuple[CandidateAssessment, ...]
    metric: SelectionMetric
    recommended_key: str | None
    comparable_keys: tuple[str, ...]
    summary: str
    peak_analysis: PeakAnalysis | None = None
    multiplet_matches: tuple[MultipletMatch, ...] = ()
    family_reports: tuple[FamilyScreeningReport, ...] = ()
    #: Confidence tier of ``recommended_key`` (High = gates passed, Medium =
    #: metric winner with structured residuals, None = no recommendation).
    #: Additive — old payloads default to ``NONE``.
    confidence: ConfidenceTier = ConfidenceTier.NONE
    #: Whether the recommendation describes real structure or the data are
    #: consistent with a null baseline. Additive — old payloads default to
    #: ``NONE`` (they predate the null-baseline test).
    verdict: RecommendationVerdict = RecommendationVerdict.NONE
    #: Human-readable caveat carried when ``confidence`` is Medium (the residual
    #: gate reasons for the recommended template) or when the null-baseline
    #: verdict fires. Empty for a clean High recommendation. GUI-facing.
    caveat: str = ""

    @property
    def recommended_assessment(self) -> CandidateAssessment | None:
        if not self.recommended_key:
            return None
        for assessment in self.assessments:
            if assessment.template.key == self.recommended_key:
                return assessment
        return None

    def assessment_for_key(self, key: str | None) -> CandidateAssessment | None:
        if not isinstance(key, str):
            return None
        for assessment in self.assessments:
            if assessment.template.key == key:
                return assessment
        return None

    def sorted_assessments(
        self, metric: SelectionMetric | None = None
    ) -> list[CandidateAssessment]:
        active_metric = metric or self.metric
        return sorted(
            self.assessments,
            key=lambda assessment: _assessment_sort_key(assessment, active_metric),
        )


def compute_information_criteria(
    chi_squared: float,
    parameter_count: int,
    sample_count: int,
) -> tuple[float, float | None, float]:
    """Return AIC, AICc, and BIC from a least-squares objective."""
    k = max(int(parameter_count), 0)
    n = max(int(sample_count), 1)
    chi2 = float(chi_squared)

    aic = chi2 + 2.0 * k
    aicc: float | None
    if n > k + 1:
        aicc = aic + 2.0 * k * (k + 1) / max(n - k - 1, 1)
    else:
        aicc = None
    bic = chi2 + k * math.log(n)
    return aic, aicc, bic


def fingerprint_spectrum(dataset: MuonDataset) -> SpectrumFingerprint:
    """Extract deterministic features used to shortlist candidate models."""
    n_points = max(int(dataset.n_points), 1)
    early_count = min(n_points, max(5, n_points // 20))
    late_count = min(n_points, max(5, n_points // 10))

    y = np.asarray(dataset.asymmetry, dtype=float)
    t = np.asarray(dataset.time, dtype=float)
    tail_estimate = float(np.mean(y[-late_count:]))
    early_mean = float(np.mean(y[:early_count]))
    initial_amplitude_estimate = float(early_mean - tail_estimate)

    centered = y - tail_estimate
    zero_crossings = _count_zero_crossings(centered)
    smoothed_centered = _moving_average(centered, _smoothing_window_points(n_points))
    smoothed_zero_crossings = _count_zero_crossings(smoothed_centered)
    smoothed_turning_points = _count_turning_points(smoothed_centered)

    # Fingerprint the same SNR-truncated window the peak detector uses.  Real
    # μSR error bars grow exponentially with time (capped at 100 %); over the
    # full record the pure-noise late tail both whitens the FFT and — because
    # the smoothed decay/slope features are then computed over noise — reads as
    # a monotonic non-oscillatory envelope, so clean TF precession is mislabelled
    # as KT/multi-rate.  Truncating to the informative early window fixes both.
    # Flat-error records (constant σ) truncate to the full length, so this is a
    # no-op there.  Purely early-time features (curvature, early mean) are
    # unaffected; the effective window only ever discards the noisy tail.
    error_full = np.asarray(dataset.error, dtype=float)
    fft_end = effective_analysis_window(t, error_full)
    t_win = t[:fft_end]
    centered_win = centered[:fft_end]
    smoothed_win = smoothed_centered[:fft_end]
    error_win = error_full[:fft_end]
    fft_dataset = MuonDataset(
        time=t_win.copy(),
        asymmetry=centered_win.copy(),
        error=error_win.copy(),
        metadata=dict(dataset.metadata),
        run=dataset.run,
    )
    frequencies, _real, magnitude = fft_asymmetry(
        fft_dataset,
        window=_DEFAULT_FFT_WINDOW,
        padding_factor=_DEFAULT_FFT_PADDING,
    )
    dominant_fft_frequency_mhz, dominant_fft_snr = _dominant_peak_metrics(frequencies, magnitude)
    duration = max(float(t_win[-1] - t_win[0]), _EPS) if t_win.size > 1 else 1.0
    dominant_fft_cycles_in_window = float(dominant_fft_frequency_mhz * duration)

    curvature_count = min(n_points, max(8, n_points // 6))
    early_time_curvature = _quadratic_curvature(t[:curvature_count], centered[:curvature_count])
    monotonic_decay_fraction = _monotonic_decay_fraction(
        smoothed_win,
        error_win,
        initial_amplitude_estimate,
    )
    semilog_slope_ratio = _semilog_slope_ratio(
        t_win,
        smoothed_win,
        error_win,
        initial_amplitude_estimate,
    )

    mid_start = early_count
    mid_stop = max(mid_start + 1, n_points - late_count)
    if mid_stop > mid_start:
        dip_minimum = float(np.min(y[mid_start:mid_stop]))
    else:
        dip_minimum = float(np.min(y))
    signal_scale = max(abs(initial_amplitude_estimate), np.ptp(y), _EPS)
    late_time_dip_recovery_score = max(0.0, tail_estimate - dip_minimum) / signal_scale

    oscillatory_hint = bool(
        dominant_fft_snr >= 3.0
        and dominant_fft_cycles_in_window >= 1.5
        and smoothed_turning_points >= 2
        and monotonic_decay_fraction <= 0.85
    )
    kt_like_hint = bool(late_time_dip_recovery_score >= 0.05 and initial_amplitude_estimate > 0.0)
    multi_rate_hint = bool(
        semilog_slope_ratio >= 1.5 and monotonic_decay_fraction >= 0.7 and not oscillatory_hint
    )

    return SpectrumFingerprint(
        tail_estimate=tail_estimate,
        initial_amplitude_estimate=initial_amplitude_estimate,
        zero_crossings=zero_crossings,
        smoothed_zero_crossings=smoothed_zero_crossings,
        smoothed_turning_points=smoothed_turning_points,
        dominant_fft_frequency_mhz=float(dominant_fft_frequency_mhz),
        dominant_fft_snr=float(dominant_fft_snr),
        dominant_fft_cycles_in_window=dominant_fft_cycles_in_window,
        monotonic_decay_fraction=float(monotonic_decay_fraction),
        early_time_curvature=float(early_time_curvature),
        semilog_slope_ratio=float(semilog_slope_ratio),
        late_time_dip_recovery_score=float(late_time_dip_recovery_score),
        oscillatory_hint=oscillatory_hint,
        kt_like_hint=kt_like_hint,
        multi_rate_hint=multi_rate_hint,
    )


def build_candidate_templates(
    fingerprint: SpectrumFingerprint,
    current_model: CompositeModel | None = None,
) -> tuple[CandidateTemplate, ...]:
    """Return the curated v1 model portfolio for the fit wizard."""
    templates: list[CandidateTemplate] = []

    def _add(
        key: str,
        model: CompositeModel,
        *,
        category: str,
        rationale: str,
        baseline: bool = False,
    ) -> None:
        if any(_model_identity(existing.model) == _model_identity(model) for existing in templates):
            return
        templates.append(
            CandidateTemplate(
                key=key,
                title=_FIT_WIZARD_TITLES.get(key, model.formula_string()),
                category=category,
                rationale=rationale,
                model=model,
                is_current_model_baseline=baseline,
            )
        )

    _add(
        "exp_constant",
        CompositeModel(["Exponential", "Constant"], operators=["+"]),
        category="General",
        rationale="Baseline single-relaxation model with a constant background.",
    )
    _add(
        "biexp_constant",
        CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"]),
        category="Multi-rate",
        rationale=(
            "Allows two exponential relaxation rates."
            if not fingerprint.multi_rate_hint
            else "Semilog envelope curvature suggests more than one relaxation rate."
        ),
    )
    _add(
        "exp_gaussian_constant",
        CompositeModel(["Exponential", "Gaussian", "Constant"], operators=["+", "+"]),
        category="Multi-rate",
        rationale="Allows an additive mix of exponential and Gaussian relaxation channels.",
    )
    _add(
        "gaussian_constant",
        CompositeModel(["Gaussian", "Constant"], operators=["+"]),
        category="General",
        rationale="Useful when early-time curvature suggests Gaussian broadening.",
    )
    _add(
        "double_gaussian_constant",
        CompositeModel(["Gaussian", "Gaussian", "Constant"], operators=["+", "+"]),
        category="Multi-rate",
        rationale="Allows two Gaussian relaxation channels with different widths.",
    )
    _add(
        "triple_exp_constant",
        CompositeModel(
            ["Exponential", "Exponential", "Exponential", "Constant"], operators=["+", "+", "+"]
        ),
        category="Multi-rate",
        rationale="Three additive exponential channels for strongly multi-rate monotonic relaxation.",
    )
    _add(
        "double_exp_gaussian_constant",
        CompositeModel(
            ["Exponential", "Exponential", "Gaussian", "Constant"], operators=["+", "+", "+"]
        ),
        category="Multi-rate",
        rationale="Three relaxing components with two exponential channels and one Gaussian channel.",
    )
    _add(
        "exp_double_gaussian_constant",
        CompositeModel(
            ["Exponential", "Gaussian", "Gaussian", "Constant"], operators=["+", "+", "+"]
        ),
        category="Multi-rate",
        rationale="Three relaxing components with one exponential channel and two Gaussian channels.",
    )
    _add(
        "triple_gaussian_constant",
        CompositeModel(["Gaussian", "Gaussian", "Gaussian", "Constant"], operators=["+", "+", "+"]),
        category="Multi-rate",
        rationale="Three additive Gaussian channels for broad distributed relaxation.",
    )
    _add(
        "stretched_constant",
        CompositeModel(["StretchedExponential", "Constant"], operators=["+"]),
        category="General",
        rationale="Adds one shape parameter when relaxation looks broader than a simple exponential.",
    )

    if fingerprint.multi_rate_hint:
        # Risch-Kehr: 1D-diffusion relaxation e^{Gt}erfc(sqrt(Gt)), curved on a
        # semilog axis (fast early, ~(pi G t)^{-1/2} tail). A natural automatic
        # alternative to the stretched exponential for broader-than-exponential
        # monotonic decay; one shape parameter, closed form (A1 in the
        # wimda-fit-function-parity study).
        _add(
            "risch_kehr_constant",
            CompositeModel(["RischKehr", "Constant"], operators=["+"]),
            category="Multi-rate",
            rationale="Curved semilog envelope is consistent with Risch-Kehr 1D-diffusion relaxation.",
        )

    if fingerprint.kt_like_hint:
        _add(
            "static_gkt_constant",
            CompositeModel(["StaticGKT_ZF", "Constant"], operators=["+"]),
            category="KT-like",
            rationale="Late-time dip and recovery are consistent with static Gaussian Kubo-Toyabe relaxation.",
        )
        # Dynamic Gaussian KT: the strong-collision generalisation where the
        # static dip-recovery is washed out by fluctuations at rate nu. Offered
        # alongside the static candidate so the wizard can distinguish a frozen
        # from a fluctuating local field. Delta is seeded from the early-time
        # curvature (see _initial_parameters_for_template); nu seeding is a
        # documented follow-up.
        _add(
            "dynamic_gkt_constant",
            CompositeModel(["DynamicGaussianKT", "Constant"], operators=["+"]),
            category="KT-like",
            rationale="A partially-washed-out KT dip suggests dynamic (fluctuating) Gaussian Kubo-Toyabe relaxation.",
        )
        _add(
            "static_gkt_exp_constant",
            CompositeModel(["StaticGKT_ZF", "Exponential", "Constant"], operators=["*", "+"]),
            category="KT-like",
            rationale="Adds a phenomenological exponential envelope to a static KT-like shape.",
        )
        # Gaussian-broadened KT: a distribution of Delta for disordered/dilute
        # moments where a single static KT is too sharp. Competes directly with
        # StaticGKT_ZF in the same regime; defaults to ZF (B_L = 0).
        _add(
            "gbkt_constant",
            CompositeModel(["GaussianBroadenedKT", "Constant"], operators=["+"]),
            category="KT-like",
            rationale="Dip-and-recovery with a softened minimum suggests a distribution of KT widths (disordered moments).",
        )

    if fingerprint.oscillatory_hint:
        _add(
            "oscillatory_exp_constant",
            CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"]),
            category="Oscillatory",
            rationale="Resolved turning points and an FFT peak spanning multiple cycles suggest a damped oscillatory component.",
        )
        _add(
            "oscillatory_gaussian_constant",
            CompositeModel(["Oscillatory", "Gaussian", "Constant"], operators=["*", "+"]),
            category="Oscillatory",
            rationale="Oscillatory candidate with Gaussian envelope for broader field distributions.",
        )
        # Bessel j0 oscillation (Overhauser): the field distribution of an
        # incommensurate/SDW internal field gives a J0 line shape rather than a
        # cosine. Offered alongside the cosine oscillatory candidates so the
        # wizard can distinguish the two envelopes on oscillatory data.
        _add(
            "bessel_exp_constant",
            CompositeModel(["Bessel", "Exponential", "Constant"], operators=["*", "+"]),
            category="Oscillatory",
            rationale="A J0 (Bessel) line shape fits incommensurate/SDW internal-field distributions better than a cosine.",
        )

    if current_model is not None:
        _add(
            "current_model",
            current_model,
            category="Baseline",
            rationale="Compares the wizard recommendation against the function already active in the fit tab.",
            baseline=True,
        )

    return tuple(templates)


@dataclass(frozen=True)
class WizardFamily:
    """A tiered-screening family: one Stage-1 representative + Stage-2 members.

    ``stage2_members`` never includes ``stage1_rep``. ``stage1_extras`` are
    additional cheap Stage-1 shapes fitted alongside the representative when a
    single rep cannot proxy the family's shape space (e.g. exponential vs
    Gaussian relaxation); the family screens on the best of them. ``priority``
    is a soft ordering hint (higher runs first); ``must_run_stage1`` marks
    families whose representative is always fitted in Stage-1.
    """

    key: str
    title: str
    stage1_rep: CandidateTemplate
    stage2_members: tuple[CandidateTemplate, ...]
    priority: float = 0.0
    must_run_stage1: bool = True
    stage1_extras: tuple[CandidateTemplate, ...] = ()


#: Canonical family order used as the stable tie-break after priority.
_WIZARD_FAMILY_ORDER: tuple[str, ...] = (
    "relaxation",
    "multi_rate",
    "kt",
    "oscillatory",
    "muonium",
    "fmuf",
    "baseline",
)


def _family_template(
    key: str,
    model: CompositeModel,
    *,
    category: str,
    rationale: str,
    baseline: bool = False,
) -> CandidateTemplate:
    """Build one :class:`CandidateTemplate`, titling it from ``_FIT_WIZARD_TITLES``."""
    return CandidateTemplate(
        key=key,
        title=_FIT_WIZARD_TITLES.get(key, model.formula_string()),
        category=category,
        rationale=rationale,
        model=model,
        is_current_model_baseline=baseline,
    )


def _template_cost_rank(template: CandidateTemplate) -> int:
    """Worst-component computational-cost rank of a template's model."""
    ranks = [
        _COST_RANK.get(definition.cost, 1)
        for name in template.model.component_names
        if (definition := COMPONENTS.get(name)) is not None
    ]
    return max(ranks, default=1)


def _template_in_scope(template: CandidateTemplate, included: frozenset[str]) -> bool:
    """True iff every component of ``template``'s model is in ``included``."""
    return all(name in included for name in template.model.component_names)


def _scope_filter_family(
    family: WizardFamily,
    included: frozenset[str],
) -> WizardFamily | None:
    """Filter one family to the in-scope templates, promoting a member if needed.

    Returns ``None`` when nothing survives. The current-model baseline family is
    never passed here (it is exempt from scope filtering).
    """
    rep_ok = _template_in_scope(family.stage1_rep, included)
    surviving_extras = tuple(
        extra for extra in family.stage1_extras if _template_in_scope(extra, included)
    )
    surviving_members = tuple(
        member for member in family.stage2_members if _template_in_scope(member, included)
    )
    if rep_ok:
        return replace(family, stage1_extras=surviving_extras, stage2_members=surviving_members)
    pool = surviving_extras + surviving_members
    if not pool:
        return None
    # Promote the cheapest surviving template to representative; ties broken by
    # template key (alphabetical) for determinism.
    promoted = min(pool, key=lambda t: (_template_cost_rank(t), t.key))
    return replace(
        family,
        stage1_rep=promoted,
        stage1_extras=tuple(e for e in surviving_extras if e.key != promoted.key),
        stage2_members=tuple(m for m in surviving_members if m.key != promoted.key),
    )


def build_wizard_families(
    fingerprint: SpectrumFingerprint,
    current_model: CompositeModel | None = None,
    *,
    scope_resolution: ScopeResolution | None = None,
) -> tuple[WizardFamily, ...]:
    """Return the tiered-screening family table for the fit wizard.

    Unlike :func:`build_candidate_templates`, every family and every member is
    always present — the fingerprint hints only raise a family's ``priority``.
    When ``scope_resolution`` is given, templates whose model uses an
    out-of-scope component are dropped; a family whose representative is dropped
    promotes its cheapest surviving member, and a family with nothing surviving
    is omitted. The current-model baseline family is never scope-filtered.
    """
    families: list[WizardFamily] = []

    # -- Relaxation -------------------------------------------------------
    families.append(
        WizardFamily(
            key="relaxation",
            title="Relaxation",
            stage1_rep=_family_template(
                "exp_constant",
                CompositeModel(["Exponential", "Constant"], operators=["+"]),
                category="General",
                rationale="Baseline single-relaxation model with a constant background.",
            ),
            stage1_extras=(
                # A single exponential rep cannot proxy Gaussian-shaped
                # relaxation (very different early-time curvature), so both
                # cheap shapes screen in Stage 1 and the family takes the best.
                _family_template(
                    "gaussian_constant",
                    CompositeModel(["Gaussian", "Constant"], operators=["+"]),
                    category="General",
                    rationale="Useful when early-time curvature suggests Gaussian broadening.",
                ),
            ),
            stage2_members=(
                _family_template(
                    "stretched_constant",
                    CompositeModel(["StretchedExponential", "Constant"], operators=["+"]),
                    category="General",
                    rationale=(
                        "Adds one shape parameter when relaxation looks broader than a "
                        "simple exponential."
                    ),
                ),
            ),
        )
    )

    # -- Multi-rate relaxation --------------------------------------------
    families.append(
        WizardFamily(
            key="multi_rate",
            title="Multi-rate relaxation",
            stage1_rep=_family_template(
                "biexp_constant",
                CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"]),
                category="Multi-rate",
                rationale="Allows two exponential relaxation rates.",
            ),
            stage2_members=(
                _family_template(
                    "exp_gaussian_constant",
                    CompositeModel(["Exponential", "Gaussian", "Constant"], operators=["+", "+"]),
                    category="Multi-rate",
                    rationale=(
                        "Allows an additive mix of exponential and Gaussian relaxation channels."
                    ),
                ),
                _family_template(
                    "double_gaussian_constant",
                    CompositeModel(["Gaussian", "Gaussian", "Constant"], operators=["+", "+"]),
                    category="Multi-rate",
                    rationale="Allows two Gaussian relaxation channels with different widths.",
                ),
                _family_template(
                    "triple_exp_constant",
                    CompositeModel(
                        ["Exponential", "Exponential", "Exponential", "Constant"],
                        operators=["+", "+", "+"],
                    ),
                    category="Multi-rate",
                    rationale=(
                        "Three additive exponential channels for strongly multi-rate "
                        "monotonic relaxation."
                    ),
                ),
                _family_template(
                    "double_exp_gaussian_constant",
                    CompositeModel(
                        ["Exponential", "Exponential", "Gaussian", "Constant"],
                        operators=["+", "+", "+"],
                    ),
                    category="Multi-rate",
                    rationale=(
                        "Three relaxing components with two exponential channels and one "
                        "Gaussian channel."
                    ),
                ),
                _family_template(
                    "exp_double_gaussian_constant",
                    CompositeModel(
                        ["Exponential", "Gaussian", "Gaussian", "Constant"],
                        operators=["+", "+", "+"],
                    ),
                    category="Multi-rate",
                    rationale=(
                        "Three relaxing components with one exponential channel and two "
                        "Gaussian channels."
                    ),
                ),
                _family_template(
                    "triple_gaussian_constant",
                    CompositeModel(
                        ["Gaussian", "Gaussian", "Gaussian", "Constant"],
                        operators=["+", "+", "+"],
                    ),
                    category="Multi-rate",
                    rationale="Three additive Gaussian channels for broad distributed relaxation.",
                ),
                _family_template(
                    "risch_kehr_constant",
                    CompositeModel(["RischKehr", "Constant"], operators=["+"]),
                    category="Multi-rate",
                    rationale=(
                        "Curved semilog envelope is consistent with Risch-Kehr "
                        "1D-diffusion relaxation."
                    ),
                ),
            ),
            priority=1.0 if fingerprint.multi_rate_hint else 0.0,
        )
    )

    # -- Kubo-Toyabe ------------------------------------------------------
    families.append(
        WizardFamily(
            key="kt",
            title="Kubo-Toyabe",
            stage1_rep=_family_template(
                "static_gkt_constant",
                CompositeModel(["StaticGKT_ZF", "Constant"], operators=["+"]),
                category="KT-like",
                rationale=(
                    "Late-time dip and recovery are consistent with static Gaussian "
                    "Kubo-Toyabe relaxation."
                ),
            ),
            stage2_members=(
                _family_template(
                    "dynamic_gkt_constant",
                    CompositeModel(["DynamicGaussianKT", "Constant"], operators=["+"]),
                    category="KT-like",
                    rationale=(
                        "A partially-washed-out KT dip suggests dynamic (fluctuating) "
                        "Gaussian Kubo-Toyabe relaxation."
                    ),
                ),
                _family_template(
                    "static_gkt_exp_constant",
                    CompositeModel(
                        ["StaticGKT_ZF", "Exponential", "Constant"], operators=["*", "+"]
                    ),
                    category="KT-like",
                    rationale="Adds a phenomenological exponential envelope to a static KT-like shape.",
                ),
                _family_template(
                    "gbkt_constant",
                    CompositeModel(["GaussianBroadenedKT", "Constant"], operators=["+"]),
                    category="KT-like",
                    rationale=(
                        "Dip-and-recovery with a softened minimum suggests a distribution of "
                        "KT widths (disordered moments)."
                    ),
                ),
                _family_template(
                    "dynamic_lkt_constant",
                    CompositeModel(["DynamicLorentzianKT", "Constant"], operators=["+"]),
                    category="KT-like",
                    rationale=(
                        "Dynamic Lorentzian KT for dilute-moment strong-collision relaxation."
                    ),
                ),
                _family_template(
                    "lf_kt_constant",
                    CompositeModel(["LongitudinalFieldKT", "Constant"], operators=["+"]),
                    category="KT-like",
                    rationale="Static Gaussian KT under a longitudinal field.",
                ),
            ),
            priority=1.0 if fingerprint.kt_like_hint else 0.0,
        )
    )

    # -- Precession (oscillatory) -----------------------------------------
    families.append(
        WizardFamily(
            key="oscillatory",
            title="Precession",
            stage1_rep=_family_template(
                "oscillatory_exp_constant",
                CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"]),
                category="Oscillatory",
                rationale=(
                    "Resolved turning points and an FFT peak spanning multiple cycles "
                    "suggest a damped oscillatory component."
                ),
            ),
            stage2_members=(
                _family_template(
                    "oscillatory_gaussian_constant",
                    CompositeModel(["Oscillatory", "Gaussian", "Constant"], operators=["*", "+"]),
                    category="Oscillatory",
                    rationale=(
                        "Oscillatory candidate with Gaussian envelope for broader field "
                        "distributions."
                    ),
                ),
                _family_template(
                    "bessel_exp_constant",
                    CompositeModel(["Bessel", "Exponential", "Constant"], operators=["*", "+"]),
                    category="Oscillatory",
                    rationale=(
                        "A J0 (Bessel) line shape fits incommensurate/SDW internal-field "
                        "distributions better than a cosine."
                    ),
                ),
                _family_template(
                    "vortex_lattice_constant",
                    CompositeModel(["VortexLattice", "Constant"], operators=["+"]),
                    category="Oscillatory",
                    rationale=(
                        "Vortex-lattice TF line shape for a superconducting field distribution."
                    ),
                ),
                _family_template(
                    "vortex_lattice_powder_constant",
                    CompositeModel(["VortexLatticePowder", "Constant"], operators=["+"]),
                    category="Oscillatory",
                    rationale=(
                        "Powder-averaged vortex-lattice TF line shape for a superconducting "
                        "field distribution."
                    ),
                ),
            ),
            priority=1.0 if fingerprint.oscillatory_hint else 0.0,
        )
    )

    # -- Muonium ----------------------------------------------------------
    # The anisotropic high-TF form (MuoniumHighTFAniso) is deliberately omitted:
    # it is too expensive for automatic screening.
    families.append(
        WizardFamily(
            key="muonium",
            title="Muonium",
            stage1_rep=_family_template(
                "muonium_low_tf_constant",
                CompositeModel(["MuoniumLowTF", "Constant"], operators=["+"]),
                category="Muonium",
                rationale="Low-transverse-field muonium two-satellite precession.",
            ),
            stage2_members=(
                _family_template(
                    "muonium_tf_constant",
                    CompositeModel(["MuoniumTF", "Constant"], operators=["+"]),
                    category="Muonium",
                    rationale="Exact four-frequency transverse-field muonium form.",
                ),
                _family_template(
                    "muonium_high_tf_constant",
                    CompositeModel(["MuoniumHighTF", "Constant"], operators=["+"]),
                    category="Muonium",
                    rationale="High-transverse-field intratriplet muonium precession.",
                ),
                _family_template(
                    "muonium_zf_constant",
                    CompositeModel(["MuoniumZF", "Constant"], operators=["+"]),
                    category="Muonium",
                    rationale="Zero-field muonium precession.",
                ),
                _family_template(
                    "muonium_lf_relax_constant",
                    CompositeModel(["MuoniumLFRelax", "Constant"], operators=["+"]),
                    category="Muonium",
                    rationale="Longitudinal-field muonium repolarisation/relaxation.",
                ),
            ),
        )
    )

    # -- Fluorine dipolar (F-mu-F) ----------------------------------------
    # FmuF_General / FmuF_Triangle are expensive powder averages — they belong in
    # Stage-2 only and are never used as representatives.
    families.append(
        WizardFamily(
            key="fmuf",
            title="Fluorine dipolar",
            stage1_rep=_family_template(
                "fmuf_linear_exp_constant",
                CompositeModel(["FmuF_Linear", "Exponential", "Constant"], operators=["*", "+"]),
                category="Nuclear dipolar",
                rationale=(
                    "Collinear F-mu-F polarisation with a phenomenological relaxing envelope."
                ),
            ),
            stage2_members=(
                _family_template(
                    "muf_exp_constant",
                    CompositeModel(["MuF", "Exponential", "Constant"], operators=["*", "+"]),
                    category="Nuclear dipolar",
                    rationale=(
                        "Single-fluorine mu-F polarisation with a phenomenological relaxing "
                        "envelope."
                    ),
                ),
                _family_template(
                    "dynamic_fmuf_constant",
                    CompositeModel(["DynamicFmuF", "Constant"], operators=["+"]),
                    category="Nuclear dipolar",
                    rationale="Dynamic F-mu-F polarisation with hopping/fluctuation.",
                ),
                _family_template(
                    "fmuf_general_constant",
                    CompositeModel(["FmuF_General", "Constant"], operators=["+"]),
                    category="Nuclear dipolar",
                    rationale="General (powder-averaged) F-mu-F polarisation.",
                ),
                _family_template(
                    "dipolar_pair_constant",
                    CompositeModel(["DipolarPairField", "Constant"], operators=["+"]),
                    category="Nuclear dipolar",
                    rationale="Dipolar-pair local-field polarisation.",
                ),
            ),
        )
    )

    if scope_resolution is not None:
        included = scope_resolution.included_set
        families = [
            filtered
            for family in families
            if (filtered := _scope_filter_family(family, included)) is not None
        ]

    # The current-model baseline family is appended after scope filtering and is
    # never scope-filtered itself.
    if current_model is not None:
        families.append(
            WizardFamily(
                key="baseline",
                title="Current model",
                stage1_rep=_family_template(
                    "baseline",
                    current_model,
                    category="Baseline",
                    rationale=(
                        "Compares the wizard recommendation against the function already "
                        "active in the fit tab."
                    ),
                    baseline=True,
                ),
                stage2_members=(),
                must_run_stage1=True,
            )
        )

    order_index = {key: idx for idx, key in enumerate(_WIZARD_FAMILY_ORDER)}
    families.sort(
        key=lambda family: (
            -family.priority,
            order_index.get(family.key, len(_WIZARD_FAMILY_ORDER)),
        )
    )
    return tuple(families)


#: Stage-1 family representatives get a reduced parameter-variant ladder.
_STAGE1_VARIANT_BUDGET = 3
#: Families within this metric distance of the best Stage-1 representative are
#: promoted to Stage 2 even when their residual gates fail (Burnham & Anderson:
#: "essentially no support" only beyond ~10).
_STAGE1_PROMOTE_DELTA = 10.0
#: Cap on Stage-2 families promoted by score/gates; pattern-matched and
#: baseline families never count against it.
_STAGE2_MAX_FAMILIES = 4
#: Detected peaks need this SNR to seed a component in a multiplet template
#: (user-declared peaks always qualify).
_MULTIPLET_MIN_SNR = 4.0

#: The best template must beat the better *strictly-simpler* null baseline by at
#: least this much AICc to count as describing significant structure. Burnham &
#: Anderson (2002): ΔAICc > 10 ⇒ the simpler (null) model has "essentially no
#: support", so below this margin the extra complexity is not warranted and the
#: verdict becomes "no significant structure". Nulls with the same or more free
#: parameters than the candidate are not a required hurdle (an exponential
#: candidate is not asked to out-score the exponential null it equals).
_NULL_BASELINE_MIN_DELTA_AICC = 10.0

#: Sigma multiple below which a fitted oscillation amplitude counts as
#: consistent with zero (|A| < k·σ_A). k = 2 ≈ the 95% two-sided bound.
_ZERO_AMPLITUDE_SIGMA = 2.0

#: A fitted frequency within this fractional slack of the 1/T resolution floor
#: is treated as "at the floor" — a free-phase cosine that completes barely one
#: cycle in the window is indistinguishable from a smooth envelope.
_FREQUENCY_FLOOR_SLACK = 0.05

#: Template keys of the two null baselines fitted unconditionally.
_NULL_CONSTANT_KEY = "null_constant"
_NULL_EXPONENTIAL_KEY = "null_exp"

#: Minimum oscillation cycles inside the SNR-truncated *effective* window for a
#: fitted frequency to be a supportable claim. Real muSR errors explode with t,
#: so the statistically informative record is often much shorter than the fit
#: window; a free-phase oscillation completing fewer than ~2 cycles there is
#: indistinguishable from a smooth envelope + systematics, however well it
#: scores by AICc (the validation programme's flat-Ag/Cu-ZF and KT-plus-drift
#: false positives were all of this kind). Generalises the 1/T_full floor in
#: ``_disqualification_reasons`` to the window that actually carries
#: information.
_MIN_CYCLES_IN_EFFECTIVE_WINDOW = 2.0

#: Fractional tolerance for matching a fitted frequency to a detected peak in
#: the spectral-corroboration check (widened by the effective resolution).
_FREQUENCY_SUPPORT_REL_TOL = 0.10

#: Effective-window cycle count above which a recommended oscillation is no
#: longer flagged as sitting at the "edge" of resolvability. Below
#: ``_MIN_CYCLES_IN_EFFECTIVE_WINDOW`` (2.0) the frequency is disqualified
#: outright by ``_apply_frequency_support_disqualifiers``; the [2.0, 3.0) band
#: is the zone where the fit survives that gate but is still marginal, so a
#: caveat rather than a veto is appropriate when spectral support is also
#: weak. Calibration anchors from the validation programme: the genuine S1
#: Larmor evidence case sits at ~2.1 effective cycles but its peak SNR is ~37
#: (see ``_EDGE_WINDOW_WEAK_SNR`` below) — no caveat, the strong line saves
#: it. The Ag-candlestick real-data case sits at ~2.25 cycles with peak SNR
#: 4.4 — the caveat fires.
_EDGE_WINDOW_MAX_CYCLES = 3.0

#: Peak SNR below which the spectral support for an edge-of-window
#: recommended oscillation counts as "weak" for the caveat in
#: ``rerank_fit_wizard_recommendation``. Set so the S1 evidence case (SNR ~37)
#: is well clear of the line while the Ag-candlestick case (SNR ~4.4) is
#: caught.
_EDGE_WINDOW_WEAK_SNR = 6.0


@dataclass(frozen=True)
class TemplateSeedContext:
    """Peak/pattern context threaded into data-driven template seeding."""

    peak_analysis: PeakAnalysis | None = None
    multiplet_matches: tuple[MultipletMatch, ...] = ()
    field_gauss: float | None = None

    def best_match(self, kinds: tuple[str, ...]) -> MultipletMatch | None:
        """Return the highest-quality match of one of ``kinds``, if any."""
        candidates = [m for m in self.multiplet_matches if m.kind in kinds]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.quality)


def _multiplet_seed_peaks(
    analysis: PeakAnalysis | None, max_components: int
) -> tuple[DetectedPeak, ...]:
    """Peaks strong enough to seed one oscillatory component each."""
    if analysis is None:
        return ()
    eligible = [
        peak for peak in analysis.peaks if peak.source == "user" or peak.snr >= _MULTIPLET_MIN_SNR
    ]
    return tuple(eligible[:max_components])


def build_oscillatory_multiplet_templates(
    analysis: PeakAnalysis | None,
    *,
    max_components: int = 3,
    envelopes: tuple[str, ...] = ("Exponential", "Gaussian"),
) -> tuple[CandidateTemplate, ...]:
    """Build ``(Osc×Env) + … + Const`` templates, one component per strong peak.

    Returns an empty tuple below two qualifying peaks — a single line is already
    covered by the plain oscillatory candidates.
    """
    peaks = _multiplet_seed_peaks(analysis, max_components)
    n = len(peaks)
    if n < 2:
        return ()
    freq_text = ", ".join(f"{peak.frequency_mhz:.3g}" for peak in peaks)
    templates: list[CandidateTemplate] = []
    for envelope in envelopes:
        names: list[str] = []
        operators: list[str] = []
        opens: list[int] = []
        closes: list[int] = []
        for k in range(n):
            names.extend(["Oscillatory", envelope])
            opens.extend([1, 0])
            closes.extend([0, 1])
            operators.append("*")
            if k < n - 1:
                operators.append("+")
        names.append("Constant")
        operators.append("+")
        opens.append(0)
        closes.append(0)
        env_tag = "exp" if envelope == "Exponential" else "gaussian"
        templates.append(
            CandidateTemplate(
                key=f"oscillatory{n}_{env_tag}_constant",
                title=f"{n}× damped cosine ({envelope} envelopes) + constant",
                category="Oscillatory",
                rationale=(
                    f"{n} spectral lines detected at {freq_text} MHz; one damped cosine per line."
                ),
                model=CompositeModel(
                    names,
                    operators=operators,
                    open_parentheses=opens,
                    close_parentheses=closes,
                ),
            )
        )
    return tuple(templates)


def build_null_baseline_templates() -> tuple[CandidateTemplate, ...]:
    """The two cheap null models fitted unconditionally as a significance floor.

    A flat constant (1 free parameter) and a plain exponential + constant
    (3 free parameters). Any real recommendation must beat the better
    *strictly-simpler* of these by a meaningful AICc margin; otherwise the data
    carry no structure worth a richer model (fixes pure-noise over-confidence).
    They are tagged as baselines so callers can keep them out of the ranked
    candidate pool.
    """
    return (
        CandidateTemplate(
            key=_NULL_CONSTANT_KEY,
            title="Null baseline: constant",
            category="Baseline",
            rationale="Flat null model — the no-structure reference.",
            model=CompositeModel(["Constant"], operators=[]),
        ),
        CandidateTemplate(
            key=_NULL_EXPONENTIAL_KEY,
            title="Null baseline: exponential + constant",
            category="Baseline",
            rationale="Plain relaxation null model — the no-oscillation reference.",
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
        ),
    )


_MULTIPLET_TEMPLATE_KEY_RE = re.compile(r"^oscillatory(\d+)_(exp|gaussian)_constant$")

#: Muonium TF templates whose ``field``/``A_hf`` seeding shares one rule.
_MUONIUM_TF_TEMPLATE_KEYS = frozenset(
    {"muonium_low_tf_constant", "muonium_tf_constant", "muonium_high_tf_constant"}
)
#: Fluorine-dipolar templates seeded from a matched r_muF.
_FMUF_TEMPLATE_KEYS = frozenset(
    {
        "fmuf_linear_exp_constant",
        "muf_exp_constant",
        "dynamic_fmuf_constant",
        "fmuf_general_constant",
        "dipolar_pair_constant",
    }
)
#: Fluorine-dipolar template parameter names carrying an F-mu distance (Angstrom).
_FMUF_DISTANCE_PARAM_BASES = frozenset({"r_muF", "r1", "r2"})
#: Default r_muF seed (a typical F-mu bond length) when no match constrains it.
_FMUF_DEFAULT_R_SEED = 1.17
#: r_muF ladder (Angstrom) spanning typical F-mu bond lengths. The lone default
#: seed converges to a spurious ~0.53 A minimum on some F-mu-F data, losing the
#: family; the ladder brackets the physical 1.06-1.28 A range so at least one
#: rung lands in the right basin. A match-derived r (when present) replaces the
#: first rung. Sized to ``_STAGE1_VARIANT_BUDGET`` so it fully populates Stage 1.
_FMUF_R_LADDER = (1.17, 1.06, 1.28)


def _effective_hint_keys(
    hint_keys: frozenset[str],
    pattern_keys: frozenset[str],
    sniff_keys: frozenset[str],
) -> frozenset[str]:
    """Suppress fingerprint hints once a stronger signal already names a family.

    Fingerprint hints are weak evidence used to rescue under-promotion when
    nothing stronger fired. A multiplet pattern match or a fluorine sniff is
    concrete evidence that already names the families it names; letting hints
    additionally expand the promotion set in that case only adds runtime for
    families the stronger signal did not ask for (failure F3's
    over-promotion), so hints are dropped entirely whenever either fires.
    """
    if pattern_keys or sniff_keys:
        return frozenset()
    return hint_keys


def _decide_family_promotions(
    families: Sequence[WizardFamily],
    rep_assessments: Sequence[CandidateAssessment],
    pattern_family_keys: frozenset[str],
    metric: SelectionMetric,
    hint_family_keys: frozenset[str] = frozenset(),
    sniff_family_keys: frozenset[str] = frozenset(),
) -> list[tuple[WizardFamily, CandidateAssessment, bool, str]]:
    """Decide which families expand to Stage 2, with a concrete reason each.

    Promotion: residual gates passed, metric within ``_STAGE1_PROMOTE_DELTA``
    of the best representative, a multiplet pattern names the family, a fluorine
    sniff names it (a chemical-formula scope prior for fmuf), a fingerprint hint
    points at it (a family's Stage-2 member can differ enough in shape from its
    representative — e.g. damped KT vs bare KT — that the rep's score alone
    under-promotes), or the family is the best/baseline one. Score/hint
    promotions are capped at ``_STAGE2_MAX_FAMILIES`` by metric rank; pattern,
    fluorine-sniff and baseline promotions are exempt from the cap (they are
    scope priors / structural, not score/hint over-expansion).
    """
    reps = dict(zip((family.key for family in families), rep_assessments))
    successful = [a for a in rep_assessments if a.is_successful]
    best_value = min(a.metric_value(metric) for a in successful) if successful else math.inf

    provisional: list[tuple[WizardFamily, CandidateAssessment, bool, str, bool]] = []
    for family in families:
        assessment = reps[family.key]
        value = assessment.metric_value(metric) if assessment.is_successful else math.inf
        delta = value - best_value
        exempt = False
        if family.key in pattern_family_keys:
            promoted, reason, exempt = True, "pattern match promotes this family", True
        elif family.key in sniff_family_keys:
            promoted, reason, exempt = (
                True,
                "sample name suggests fluorine — promoting the F-mu-F family",
                True,
            )
        elif family.key == "baseline":
            promoted, reason, exempt = True, "current-model baseline", True
        elif family.key == "relaxation":
            # Always expand the smooth-relaxation portfolio: exotic families
            # (fmuf/KT with free envelopes) are flexible enough to win Stage-1
            # screening on smooth data, and the Δ cutoff would then lock the
            # simpler true model (e.g. a stretched exponential) out of the
            # final metric comparison entirely. These are cheap 1-D relaxation
            # fits — the guard-rail is worth far more than they cost.
            promoted, reason, exempt = (
                True,
                "smooth-relaxation reference family — always expanded",
                True,
            )
        elif not assessment.is_successful:
            promoted, reason = False, "not promoted: Stage-1 representative fit failed"
        elif delta <= 0.0:
            promoted, reason = True, "best Stage-1 score"
        elif assessment.residual_gate_passed:
            promoted, reason = True, "residual gates passed"
        elif delta <= _STAGE1_PROMOTE_DELTA:
            promoted, reason = (
                True,
                f"within {_STAGE1_PROMOTE_DELTA:.0f} of the best score "
                f"(Δ{metric.value} = {delta:.1f})",
            )
        elif family.key in hint_family_keys:
            promoted, reason = (
                True,
                "fingerprint hint suggests this family despite the representative score",
            )
        else:
            gate_text = "; ".join(assessment.residual_gate_reasons[:2]) or "gates failed"
            promoted, reason = (
                False,
                f"not promoted: Δ{metric.value} = {delta:.1f} above best and {gate_text}",
            )
        provisional.append((family, assessment, promoted, reason, exempt))

    capped_keys = [
        family.key
        for family, assessment, promoted, _reason, exempt in sorted(
            provisional, key=lambda entry: entry[1].metric_value(metric)
        )
        if promoted and not exempt
    ]
    overflow = set(capped_keys[_STAGE2_MAX_FAMILIES:])

    decisions: list[tuple[WizardFamily, CandidateAssessment, bool, str]] = []
    for family, assessment, promoted, reason, _exempt in provisional:
        if family.key in overflow:
            promoted = False
            reason = "not promoted: within margin but beyond the Stage-2 family cap"
        decisions.append((family, assessment, promoted, reason))
    return decisions


#: Stage-1 screening fits only need to rank families, not converge to full
#: precision: a fit that has not settled in 3000 calls is either pathological
#: or seeded in the wrong basin, and the parameter-variant ladder (not a longer
#: migrad run) is what rescues a bad seed. Stage-2 refits of promoted families
#: run at full precision with no cap.
_SCREENING_MIGRAD_NCALL = 3000

#: Full parameter-variant budget for a Stage-2 member (the blind-seed ladder that
#: rescues a bad initial guess).
_STAGE2_FULL_VARIANT_BUDGET = 5
#: Reduced variant budget for Stage-2 members of a family promoted with no
#: independent support (no pattern match, no fingerprint hint, no fluorine
#: sniff). Such a family reached Stage 2 only by Stage-1 score / Δ-margin / gates
#: — it is a "screen everything" hedge, not a positively-identified shape — so a
#: full blind-seed ladder buys little and costs the family's slowest members a
#: seconds-per-call fit each. Two rungs keep the seed + one perturbation that
#: catches a mis-scaled guess.
_STAGE2_UNSUPPORTED_VARIANT_BUDGET = 2

# NOTE: a fixed migrad ncall cap on EXPENSIVE Stage-2 members was investigated
# and deliberately rejected. iminuit's default ``migrad(ncall=None)`` uses an
# adaptive per-problem heuristic that already stops these powder-average /
# strong-collision fits at a sensible point; forcing an explicit cap (e.g. 5000)
# instead *raises* the ceiling and, because ``migrad`` auto-iterates up to five
# times when a run hits its call limit without converging, multiplies the work
# on exactly the slow non-converging fits it was meant to bound (a clean TF@20G
# worst case regressed ~5x). The support-gated variant-budget trim below is the
# safe win; EXPENSIVE members keep the adaptive drive so their final AICc is
# unchanged.

#: Families whose Stage-2 effort is never trimmed regardless of support, because
#: their promotion is a structural guard-rail rather than a score/hint hedge:
#: ``relaxation`` is the always-expanded smooth-relaxation reference, ``baseline``
#: is the current-model reference. (Null baselines are fitted outside the
#: promotion loop and so are inherently unaffected.)
_STAGE2_NEVER_TRIM_FAMILY_KEYS = frozenset({"relaxation", "baseline"})


def _stage2_variant_budget(
    *,
    is_expensive: bool,
    is_peak_seeded: bool,
    family_key: str,
    supported: bool,
) -> int:
    """Pure per-member Stage-2 variant budget (no fits — unit-testable).

    ``supported`` is TRUE when the member's family is named by a multiplet
    pattern match, a fingerprint hint, or a fluorine sniff (set membership, not
    the promotion reason string — a hinted family can be promoted by score, so
    the reason alone under-reports support).

    Ordering of the rules:

    * EXPENSIVE members always get the reduced ladder (their match-derived seeds
      are already tight and each extra variant is a full slow fit — unchanged
      from the prior behaviour).
    * Peak-seeded multiplet members, never-trim families, and supported families
      keep the full ladder.
    * Everything else — an unsupported family that reached Stage 2 only by
      score / Δ-margin / gates — gets the reduced ladder.
    """
    if is_expensive:
        return _STAGE2_UNSUPPORTED_VARIANT_BUDGET
    if is_peak_seeded or family_key in _STAGE2_NEVER_TRIM_FAMILY_KEYS or supported:
        return _STAGE2_FULL_VARIANT_BUDGET
    return _STAGE2_UNSUPPORTED_VARIANT_BUDGET


@dataclass(frozen=True)
class _AssessmentTask:
    """One self-contained unit of work for :func:`_run_template_assessments`.

    A plain-data payload (rather than a closure) so a task can cross a process
    boundary via ``pickle`` under ``spawn``. ``seed_context`` may be ``None``
    (the Stage-1 pass runs without one); ``screening_cap`` is explicit rather
    than keyed off ``stage`` because the null-baseline templates are fitted as
    stage-1-style cheap fits but must never be call-capped (they are 1-3
    parameter fits already).
    """

    dataset: MuonDataset
    fingerprint: SpectrumFingerprint
    template: CandidateTemplate
    metric: SelectionMetric
    seed_context: TemplateSeedContext | None
    variant_budget: int
    stage: int
    screening_cap: bool = False


def _execute_assessment_task(
    task: _AssessmentTask,
    cancel_callback: Callable[[], bool] | None = None,
) -> CandidateAssessment:
    """Run one :class:`_AssessmentTask` to a :class:`CandidateAssessment`.

    Module-level (not a closure) so it can be pickled and sent to a worker
    process. Builds its own :class:`FitEngine` per call — engines carry no
    state worth sharing, and a fresh one keeps each task fully self-contained.
    """
    migrad_ncall = _SCREENING_MIGRAD_NCALL if task.screening_cap else None
    return _assess_candidate_template(
        task.dataset,
        task.fingerprint,
        task.template,
        fit_engine=FitEngine(),
        metric=task.metric,
        seed_context=task.seed_context,
        variant_budget=task.variant_budget,
        stage=task.stage,
        cancel_callback=cancel_callback,
        migrad_ncall=migrad_ncall,
    )


def _run_template_assessments(
    tasks: Sequence[_AssessmentTask],
    *,
    max_workers: int | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    executor: Executor | None = None,
) -> list[CandidateAssessment]:
    """Run candidate-assessment tasks, returning results in task order.

    Tasks are plain-data :class:`_AssessmentTask` payloads (not closures) so
    they can be fanned out across worker *processes* via ``pickle`` under
    ``spawn`` — the GIL makes a thread pool of CPU-bound fits effectively
    serial, so processes are what actually buys parallelism here.

    Worker count: ``max_workers`` wins when given; otherwise
    ``min(len(tasks), max(1, cpu_count - 2))``. A resolved width ``<= 1`` (or
    at most one task) runs a plain serial loop calling
    :func:`_execute_assessment_task` with ``cancel_callback`` — this is the
    deterministic test path and it is also the only path that can honour an
    *in-fit* cancellation (a cancel_callback cannot cross a process boundary).

    Process path: reuses ``executor`` if given (a caller-managed shared pool);
    otherwise opens one via :func:`open_spawn_pool` and closes it in a
    ``finally`` — but only the pool this call opened; a caller-supplied
    ``executor`` remains the caller's to shut down. ``cancel_callback`` cannot
    be forwarded across the process boundary, so cancellation is instead
    polled between completions via an ``as_completed`` loop; on cancellation
    the *locally opened* pool is shut down with
    ``cancel_futures=True`` and :class:`FitCancelledError` is raised (never
    swallowed). A pool that could not start (``open_spawn_pool`` returns
    ``None``) falls back to the historical thread-pool path, which keeps
    in-fit cancellation working.

    Per-task resilience: if a submitted future raises (a pickling failure or a
    worker crash), that one task is retried serially in-parent with
    ``cancel_callback``; only if the retry also raises does the exception
    propagate. :class:`FitCancelledError` is never treated as a per-task
    failure — it always propagates immediately.
    """
    task_list = list(tasks)
    if not task_list:
        return []

    if max_workers is not None:
        workers = int(max_workers)
    else:
        workers = min(len(task_list), max(1, (os.cpu_count() or 4) - 2))

    if workers <= 1 or len(task_list) <= 1:
        return [_execute_assessment_task(task, cancel_callback) for task in task_list]

    opened_here = executor is None
    pool: Executor | None = executor
    if opened_here:
        pool = open_spawn_pool(workers)

    if pool is None:
        # spawn unavailable in this environment (e.g. a restricted sandbox) —
        # fall back to the thread pool, which still honours in-fit cancellation.
        results: list[CandidateAssessment | None] = [None] * len(task_list)
        with ThreadPoolExecutor(max_workers=workers) as thread_executor:
            future_to_index = {
                thread_executor.submit(_execute_assessment_task, task, cancel_callback): index
                for index, task in enumerate(task_list)
            }
            for future in future_to_index:
                results[future_to_index[future]] = future.result()
        return [result for result in results if result is not None]

    shutdown_on_exit = opened_here
    try:
        results: list[CandidateAssessment | None] = [None] * len(task_list)
        try:
            future_to_index = {
                pool.submit(_execute_assessment_task, task): index
                for index, task in enumerate(task_list)
            }
        except Exception:
            # A pool broken at submission time (e.g. spawn workers cannot
            # re-import __main__ under an interactive/stdin host) leaves NO
            # futures to drain — run everything serially in-parent instead.
            # FitCancelledError cannot arise from submit, so a blanket except
            # is safe here.
            return [_execute_assessment_task(task, cancel_callback) for task in task_list]
        remaining = len(task_list)
        completed = as_completed(future_to_index)
        while remaining:
            if cancel_callback is not None and cancel_callback():
                if opened_here:
                    pool.shutdown(wait=False, cancel_futures=True)
                    shutdown_on_exit = False
                raise FitCancelledError("Fit wizard analysis cancelled.")
            future = next(completed)
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except FitCancelledError:
                raise
            except Exception:
                # Pickling failure or worker crash: retry this one task
                # serially in-parent before giving up on it.
                results[index] = _execute_assessment_task(task_list[index], cancel_callback)
            remaining -= 1
        return [result for result in results if result is not None]
    finally:
        if shutdown_on_exit:
            pool.shutdown()


def build_fit_wizard_recommendation(
    dataset: MuonDataset,
    current_model: CompositeModel | None = None,
    *,
    metric: SelectionMetric = SelectionMetric.AICC,
    scope: WizardScope | None = None,
    user_frequencies_mhz: Sequence[float] | None = None,
    max_workers: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitWizardRecommendation:
    """Analyze one asymmetry spectrum and recommend a fit candidate.

    Tiered screening: every in-scope candidate family gets its cheap Stage-1
    representative fitted (fingerprint hints only prioritise — they no longer
    exclude); families that pass the residual gates, score within
    ``_STAGE1_PROMOTE_DELTA`` of the best, or are named by a multiplet pattern
    match expand to their full Stage-2 portfolios. ``scope`` restricts the
    families physically (``None`` screens the default superset);
    ``user_frequencies_mhz`` adds trusted peak seeds; ``max_workers=1`` gives
    a deterministic serial path. ``cancel_callback`` is polled between and
    inside fits when running serially or on the thread-pool fallback (engine
    in-fit abort); when the resolved worker count allows a shared *process*
    pool, cancellation instead takes effect only between fits (a
    cancel_callback cannot cross a process boundary) — either way,
    cancellation raises :class:`FitCancelledError`.
    """
    fingerprint = fingerprint_spectrum(dataset)
    resolution: ScopeResolution | None = None
    if scope is not None:
        resolution = resolve_scope_for_dataset(dataset, scope)
    families = build_wizard_families(fingerprint, current_model, scope_resolution=resolution)

    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def _check_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Fit wizard analysis cancelled.")

    if not families:
        return FitWizardRecommendation(
            fingerprint=fingerprint,
            templates=(),
            assessments=(),
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary=(
                "No candidate families are in scope — widen the scope selection "
                "to run the analysis."
            ),
        )

    field_gauss = dataset.field
    direction_text = str(
        dataset.metadata.get("field_direction") or dataset.metadata.get("field_state") or ""
    )
    geometry = geometry_from_field_direction(direction_text)
    geometry_token = geometry.value if geometry is not None else None

    # Peak pass A: tail-subtracted spectrum, plus any user-declared seeds.
    peak_analysis = analyze_dataset_peaks(dataset)
    if user_frequencies_mhz:
        peak_analysis = merge_user_peaks(peak_analysis, tuple(user_frequencies_mhz))

    stage1_context = TemplateSeedContext(peak_analysis=peak_analysis, field_gauss=field_gauss)

    _progress(f"Stage 1: screening {len(families)} candidate families")

    def _stage1_task(template: CandidateTemplate) -> _AssessmentTask:
        return _AssessmentTask(
            dataset=dataset,
            fingerprint=fingerprint,
            template=template,
            metric=metric,
            seed_context=stage1_context,
            variant_budget=_STAGE1_VARIANT_BUDGET,
            stage=1,
            screening_cap=True,
        )

    stage1_groups = [(family, [family.stage1_rep, *family.stage1_extras]) for family in families]
    flat_stage1_templates = [template for _family, group in stage1_groups for template in group]

    # One process pool for the whole build, shared across the three fan-outs
    # below (Stage 1, Stage 2, null baselines) — opening a spawn pool per call
    # would pay the process-startup cost three times over for no benefit, since
    # the pool is idle between fan-outs anyway.
    resolved_workers = (
        int(max_workers)
        if max_workers is not None
        else min(
            max(len(flat_stage1_templates), 1),
            max(1, (os.cpu_count() or 4) - 2),
        )
    )
    shared_pool = open_spawn_pool(resolved_workers) if resolved_workers > 1 else None

    try:
        flat_stage1 = _run_template_assessments(
            [_stage1_task(template) for template in flat_stage1_templates],
            max_workers=max_workers,
            cancel_callback=cancel_callback,
            executor=shared_pool,
        )

        # Regroup and pick each family's screening representative: gate-passers
        # first, then best metric (a family screens on the best of its cheap
        # Stage-1 shapes).
        grouped_stage1: list[list[CandidateAssessment]] = []
        cursor = 0
        for _family, group in stage1_groups:
            grouped_stage1.append(flat_stage1[cursor : cursor + len(group)])
            cursor += len(group)

        def _family_best(group: list[CandidateAssessment]) -> CandidateAssessment:
            return min(
                group,
                key=lambda a: (
                    not a.is_successful,
                    not a.residual_gate_passed,
                    a.metric_value(metric) if a.is_successful else math.inf,
                ),
            )

        stage1_assessments = [_family_best(group) for group in grouped_stage1]

        _check_cancelled()

        # Peak pass B: FFT of the best smooth (non-oscillatory) fit's residuals
        # kills relaxation leakage and exposes weak lines. This is the first-class
        # generalisation of the global wizard's oscillatory rescue.
        detrend_pool = [
            assessment
            for (family, _group), group_assessments in zip(stage1_groups, grouped_stage1)
            for assessment in group_assessments
            if family.key in ("relaxation", "multi_rate", "kt") and assessment.is_successful
        ]
        if detrend_pool:
            best_smooth = min(detrend_pool, key=lambda a: a.metric_value(metric))
            curve_values = _curve_parameter_values(
                best_smooth.template.model, best_smooth.fit_result.parameters
            )
            detrend_curve = np.asarray(
                best_smooth.template.model.function(
                    np.asarray(dataset.time, dtype=float), **curve_values
                ),
                dtype=float,
            )
            detrended_analysis = analyze_dataset_peaks(
                dataset,
                detrend_curve=detrend_curve,
                detrend_template_key=best_smooth.template.key,
            )
            if user_frequencies_mhz:
                detrended_analysis = merge_user_peaks(
                    detrended_analysis, tuple(user_frequencies_mhz)
                )
            if detrended_analysis.peaks or not peak_analysis.peaks:
                peak_analysis = detrended_analysis

        multiplet_matches = match_multiplets(
            peak_analysis, field_gauss=field_gauss, geometry=geometry_token
        )
        # Time-domain matched filter for damped-envelope families (F-mu-F / mu-F /
        # KT): their signatures are envelopes, not sharp lines, so the line-based
        # ``match_multiplets`` above rarely fires on them (the circular-dependency
        # failure). Run the banks on the tail-centred raw signal — NOT the Peak-pass-B
        # residual (the KT family is itself that residual's detrend model, so it would
        # subtract the shape it looks for). Gate to the in-scope envelope families so
        # out-of-scope runs skip the work.
        in_scope_family_keys = frozenset(family.key for family in families)
        envelope_scope = frozenset({"fmuf", "kt"}) & in_scope_family_keys
        if envelope_scope:
            multiplet_matches = (
                *multiplet_matches,
                *match_envelope_banks(
                    dataset,
                    field_gauss=field_gauss,
                    include_families=envelope_scope,
                ),
            )
        pattern_family_keys = frozenset(
            match.family_key
            for match in multiplet_matches
            if match.family_key in in_scope_family_keys
        )

        # Fluorine sniff: a chemical-formula fluorine in the sample/title name is a
        # physics-scope prior for the fmuf family. It previously only annotated the
        # scope note; here it *promotes* fmuf (exempt from the Stage-2 family cap, the
        # same class as a pattern match — it is a scope prior, not a score/hint
        # over-expansion), but only when fmuf is actually in scope so the promotion
        # cannot force the EXPENSIVE fmuf powder-average members on a non-fluoride run.
        sniff_family_keys: frozenset[str] = frozenset()
        if "fmuf" in in_scope_family_keys and dataset_suggests_fluorine(dataset):
            sniff_family_keys = frozenset({"fmuf"})

        hint_family_keys = _effective_hint_keys(
            frozenset(
                key
                for key, hinted in (
                    ("multi_rate", fingerprint.multi_rate_hint),
                    ("kt", fingerprint.kt_like_hint),
                    ("oscillatory", fingerprint.oscillatory_hint),
                )
                if hinted
            ),
            pattern_family_keys,
            sniff_family_keys,
        )
        decisions = _decide_family_promotions(
            families,
            stage1_assessments,
            pattern_family_keys,
            metric,
            hint_family_keys=hint_family_keys,
            sniff_family_keys=sniff_family_keys,
        )

        stage2_context = TemplateSeedContext(
            peak_analysis=peak_analysis,
            multiplet_matches=multiplet_matches,
            field_gauss=field_gauss,
        )

        # A family reached Stage 2 with *independent support* when it is named by
        # a multiplet pattern match, a fingerprint hint, or a fluorine sniff —
        # i.e. something positively identified its shape, rather than it merely
        # surviving the "screen everything" Stage-1 score / Δ-margin / gate hedge.
        # This is set membership, not the promotion reason string: a hinted family
        # can still be promoted by "best Stage-1 score" (the hint check is a later
        # elif in _decide_family_promotions), so the reason alone under-reports
        # support. Unsupported families run a reduced Stage-2 variant ladder.
        supported_family_keys = pattern_family_keys | hint_family_keys | sniff_family_keys

        # Stage 2: expand promoted families; the oscillatory family additionally
        # receives multiplet templates generated from the detected peak set.
        seen_identities = {_model_identity(template.model) for template in flat_stage1_templates}
        stage2_templates: list[CandidateTemplate] = []
        stage2_keys_by_family: dict[str, list[str]] = {}
        # Per-member Stage-2 fit-effort metadata, keyed by the (unique) member key:
        # (family_key, family_supported, is_peak_seeded_multiplet). Peak-seeded
        # multiplet members keep the full ladder even inside an unsupported family
        # (their frequencies come straight from detected peaks — already tight).
        stage2_member_meta: dict[str, tuple[str, bool, bool]] = {}
        for family, _assessment, promoted, _reason in decisions:
            if not promoted:
                continue
            supported = family.key in supported_family_keys
            members = [(member, False) for member in family.stage2_members]
            if family.key == "oscillatory":
                for extra in build_oscillatory_multiplet_templates(peak_analysis):
                    if resolution is not None and not all(
                        name in resolution.included_set for name in extra.model.component_names
                    ):
                        continue
                    members.append((extra, True))
            kept: list[str] = []
            for member, is_peak_seeded in members:
                identity = _model_identity(member.model)
                if identity in seen_identities:
                    continue
                seen_identities.add(identity)
                stage2_templates.append(member)
                stage2_member_meta[member.key] = (family.key, supported, is_peak_seeded)
                kept.append(member.key)
            stage2_keys_by_family[family.key] = kept

        _check_cancelled()
        if stage2_templates:
            _progress(f"Stage 2: fitting {len(stage2_templates)} expanded candidates")

        def _stage2_task(template: CandidateTemplate) -> _AssessmentTask:
            family_key, supported, is_peak_seeded = stage2_member_meta[template.key]
            is_expensive = _template_cost_rank(template) >= _COST_RANK[ComputationalCost.EXPENSIVE]
            budget = _stage2_variant_budget(
                is_expensive=is_expensive,
                is_peak_seeded=is_peak_seeded,
                family_key=family_key,
                supported=supported,
            )
            return _AssessmentTask(
                dataset=dataset,
                fingerprint=fingerprint,
                template=template,
                metric=metric,
                seed_context=stage2_context,
                variant_budget=budget,
                stage=2,
                screening_cap=False,
            )

        stage2_assessments = _run_template_assessments(
            [_stage2_task(template) for template in stage2_templates],
            max_workers=max_workers,
            cancel_callback=cancel_callback,
            executor=shared_pool,
        )

        _check_cancelled()
        # Null baselines: fitted unconditionally (independent of scope/promotion
        # and of the Stage-1/2 dedup) so the "no significant structure" verdict
        # always has a reference. They are 1-2 free-parameter fits, so cheap —
        # never call-capped (screening_cap=False) even though they are tagged
        # stage=1 like the other cheap first-pass fits.
        null_templates = build_null_baseline_templates()
        null_assessments = _run_template_assessments(
            [
                _AssessmentTask(
                    dataset=dataset,
                    fingerprint=fingerprint,
                    template=template,
                    metric=metric,
                    seed_context=stage2_context,
                    variant_budget=_STAGE1_VARIANT_BUDGET,
                    stage=1,
                    screening_cap=False,
                )
                for template in null_templates
            ],
            max_workers=max_workers,
            cancel_callback=cancel_callback,
            executor=shared_pool,
        )
    finally:
        if shared_pool is not None:
            shared_pool.shutdown()

    family_reports = tuple(
        FamilyScreeningReport(
            family_key=family.key,
            title=family.title,
            stage1_template_key=assessment.template.key,
            stage1_metric_value=(
                assessment.metric_value(metric) if assessment.is_successful else math.inf
            ),
            stage1_gate_passed=assessment.residual_gate_passed,
            promoted=promoted,
            reason=reason,
            stage2_template_keys=tuple(stage2_keys_by_family.get(family.key, ())),
        )
        for family, assessment, promoted, reason in decisions
    )

    all_templates = tuple(flat_stage1_templates) + tuple(stage2_templates) + tuple(null_templates)
    all_assessments = _apply_frequency_support_disqualifiers(
        dataset,
        tuple(flat_stage1) + tuple(stage2_assessments) + tuple(null_assessments),
        peak_analysis,
    )

    return rerank_fit_wizard_recommendation(
        FitWizardRecommendation(
            fingerprint=fingerprint,
            templates=all_templates,
            assessments=all_assessments,
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary="",
            peak_analysis=peak_analysis,
            multiplet_matches=multiplet_matches,
            family_reports=family_reports,
        ),
        metric,
    )


def build_fit_wizard_recommendation_for_templates(
    dataset: MuonDataset,
    templates: tuple[CandidateTemplate, ...] | list[CandidateTemplate],
    *,
    fingerprint: SpectrumFingerprint | None = None,
    metric: SelectionMetric = SelectionMetric.AICC,
) -> FitWizardRecommendation:
    """Evaluate one dataset against an explicit candidate-template list."""
    active_fingerprint = fingerprint or fingerprint_spectrum(dataset)
    active_templates = tuple(templates)
    # max_workers=1 preserves this function's historical serial semantics (one
    # fit engine's worth of work at a time, in list order); the previous single
    # shared FitEngine() is replaced by one-per-task, which is equivalent since
    # FitEngine carries no state across calls.
    assessments = tuple(
        _run_template_assessments(
            [
                _AssessmentTask(
                    dataset=dataset,
                    fingerprint=active_fingerprint,
                    template=template,
                    metric=metric,
                    seed_context=None,
                    variant_budget=5,
                    stage=2,
                    screening_cap=False,
                )
                for template in active_templates
            ],
            max_workers=1,
        )
    )
    return rerank_fit_wizard_recommendation(
        FitWizardRecommendation(
            fingerprint=active_fingerprint,
            templates=active_templates,
            assessments=assessments,
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary="",
        ),
        metric,
    )


def _better_simpler_null(
    candidate: CandidateAssessment,
    null_assessments: Sequence[CandidateAssessment],
    metric: SelectionMetric,
) -> CandidateAssessment | None:
    """Best null baseline strictly simpler than ``candidate``, or ``None``.

    "Strictly simpler" = fewer free parameters. Same/richer nulls are not a
    required hurdle, so a plain-exponential candidate is never asked to out-score
    the equal-complexity exponential null it equals — only the flat 1-parameter
    null. This is what lets richer families (oscillatory/KT/F-µ-F) be forced to
    clear both nulls while the exponential family clears just the flat one.
    """
    simpler = [
        null
        for null in null_assessments
        if null.is_successful
        and np.isfinite(null.metric_value(metric))
        and null.parameter_count < candidate.parameter_count
    ]
    if not simpler:
        return None
    return min(simpler, key=lambda null: null.metric_value(metric))


def _template_family_map(
    family_reports: Sequence[FamilyScreeningReport],
) -> dict[str, str]:
    """Map each known template key to its owning family key.

    Built from ``family_reports`` (Stage-1 representative + Stage-2 members of
    every screened family), so it reflects the actual family membership used
    for promotion rather than requiring a fresh lookup against
    ``build_wizard_families``. Template keys absent from every report (old
    payloads that predate ``family_reports``, the explicit-template path, or
    multiplet/extra templates never surfaced in a report) are simply absent
    from the returned map — callers must treat that as "unknown family", not
    as membership in some default family.
    """
    mapping: dict[str, str] = {}
    for report in family_reports:
        mapping[report.stage1_template_key] = report.family_key
        for key in report.stage2_template_keys:
            mapping[key] = report.family_key
    return mapping


def _same_family(
    key_a: str,
    key_b: str,
    family_map: dict[str, str],
) -> bool:
    """True only when both keys resolve to the same known family.

    Either key missing from ``family_map`` (unknown family — old payloads, the
    explicit-template path, or a template never listed in a screening report)
    is treated as "not the same family": the safe direction, since it only
    suppresses the simpler-preferred swap rather than risking a wrong-family
    promotion.
    """
    family_a = family_map.get(key_a)
    family_b = family_map.get(key_b)
    if family_a is None or family_b is None:
        return False
    return family_a == family_b


def _edge_of_window_caveat(
    assessment: CandidateAssessment,
    peak_analysis: PeakAnalysis | None,
) -> str:
    """Caveat text when ``assessment``'s oscillation sits at the edge of resolvability.

    Computed entirely from serialized state (``peak_analysis``), so a metric
    rerank or an ``.asymp`` reload sees the same verdict as the original build.
    Fires when a ``frequency``-named fitted parameter completes
    ``[_MIN_CYCLES_IN_EFFECTIVE_WINDOW, _EDGE_WINDOW_MAX_CYCLES)`` cycles in
    the effective window (fewer would already be disqualified by
    ``_apply_frequency_support_disqualifiers``) AND its supporting spectral
    peak — matched with the same tolerance rule as that disqualifier — is
    either absent or weak (SNR < ``_EDGE_WINDOW_WEAK_SNR``). A strong
    supporting line (e.g. the S1 evidence case at SNR ~37) is exempted even at
    the edge of the window. Returns ``""`` when no trigger fires or
    ``peak_analysis`` is unavailable (explicit-template path, old payloads).
    """
    if peak_analysis is None:
        return ""
    resolution_mhz = peak_analysis.resolution_mhz
    if not np.isfinite(resolution_mhz) or resolution_mhz <= _EPS:
        return ""
    for parameter in assessment.fit_result.parameters:
        if split_parameter_name(parameter.name)[0] != "frequency":
            continue
        frequency = abs(float(parameter.value))
        if frequency <= _EPS:
            continue
        cycles = frequency / resolution_mhz
        if not (_MIN_CYCLES_IN_EFFECTIVE_WINDOW <= cycles < _EDGE_WINDOW_MAX_CYCLES):
            continue
        tolerance = max(2.0 * resolution_mhz, _FREQUENCY_SUPPORT_REL_TOL * frequency)
        supporting_peak = min(
            (
                peak
                for peak in peak_analysis.peaks
                if abs(peak.frequency_mhz - frequency) <= tolerance
            ),
            key=lambda peak: abs(peak.frequency_mhz - frequency),
            default=None,
        )
        snr = supporting_peak.snr if supporting_peak is not None else 0.0
        if snr >= _EDGE_WINDOW_WEAK_SNR:
            continue
        return (
            f"the {frequency:.4g} MHz oscillation completes only {cycles:.2g} cycles "
            "inside the statistically informative window and its spectral support is "
            f"weak (peak SNR {snr:.2g}) — verify against a longer or higher-statistics run."
        )
    return ""


def rerank_fit_wizard_recommendation(
    recommendation: FitWizardRecommendation,
    metric: SelectionMetric,
) -> FitWizardRecommendation:
    """Apply the recommendation policy against the (already fitted) assessments.

    Policy (gates classify, they no longer veto):

    * Recommend the **best-by-metric** successful, non-null, non-disqualified
      candidate. Targeted disqualifiers (frequency at the resolution floor /
      pinned at a bound, zero-consistent oscillation amplitude) drop a candidate
      to the next survivor.
    * Confidence tier: **High** when the winner's residuals pass every gate,
      **Medium** when it is the clear metric winner but leaves structured
      residuals (its gate reasons ride along as a ``caveat``).
    * Null-baseline verdict: if the winner does not beat the better
      *strictly-simpler* null baseline by ``_NULL_BASELINE_MIN_DELTA_AICC``, the
      verdict becomes ``NO_SIGNIFICANT_STRUCTURE`` and the recommendation points
      at the winning null (fixes pure-noise over-confidence). Tolerates missing
      nulls (old payloads, the explicit-template path): the test is skipped.
    """
    null_assessments = [
        assessment
        for assessment in recommendation.assessments
        if assessment.is_null_baseline and assessment.is_successful
    ]
    candidates = [
        assessment
        for assessment in recommendation.assessments
        if assessment.is_successful and not assessment.is_null_baseline
    ]

    eligible = sorted(
        (a for a in candidates if not a.is_disqualified),
        key=lambda assessment: _assessment_sort_key(assessment, metric),
    )

    if not eligible:
        # Nothing survives the disqualifiers (or nothing fitted). Fall back to a
        # null baseline if one is available so callers still get a concrete,
        # low-confidence anchor; otherwise there is genuinely no recommendation.
        best_null = min(
            (a for a in null_assessments if np.isfinite(a.metric_value(metric))),
            key=lambda a: a.metric_value(metric),
            default=None,
        )
        if best_null is None:
            summary = (
                "No candidate could be recommended — every successful fit was "
                "disqualified and no null baseline is available. Inspect the "
                "comparison table before applying a model."
            )
            return replace(
                recommendation,
                metric=metric,
                recommended_key=None,
                comparable_keys=(),
                summary=summary,
                confidence=ConfidenceTier.NONE,
                verdict=RecommendationVerdict.NONE,
                caveat="",
            )
        caveat = (
            "Every candidate model was disqualified (e.g. an oscillation at the "
            "resolution floor or with amplitude consistent with zero); the data "
            "are consistent with the null baseline."
        )
        summary = (
            f"No significant structure — recommending the {best_null.template.title} "
            f"null baseline by {metric.value}."
        )
        return replace(
            recommendation,
            metric=metric,
            recommended_key=best_null.template.key,
            comparable_keys=(),
            summary=summary,
            confidence=ConfidenceTier.NONE,
            verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
            caveat=caveat,
        )

    primary = eligible[0]
    comparable_keys: tuple[str, ...] = ()
    if len(eligible) > 1:
        runner_up = eligible[1]
        score_delta = abs(primary.metric_value(metric) - runner_up.metric_value(metric))
        if score_delta <= _COMPARABLE_SCORE_DELTA:
            # The simpler-preferred swap only applies within one family: a
            # cross-family tie (e.g. stretched-exponential vs. Risch-Kehr, S7)
            # must keep the metric winner primary — swapping to the simpler
            # template across families let a wrong-family model win on
            # param-count alone. Family membership is derived from
            # ``family_reports``; a key missing from that map is "unknown
            # family" and never swaps (the safe direction).
            family_map = _template_family_map(recommendation.family_reports)
            if _same_family(primary.template.key, runner_up.template.key, family_map):
                preferred = min(
                    (primary, runner_up),
                    key=lambda assessment: (
                        assessment.parameter_count,
                        assessment.additive_terms,
                        assessment.template.title,
                    ),
                )
                alternate = runner_up if preferred.template.key == primary.template.key else primary
                primary = preferred
                comparable_keys = (preferred.template.key, alternate.template.key)
            else:
                comparable_keys = (primary.template.key, runner_up.template.key)

    # Null-baseline significance test against the best strictly-simpler null.
    reference_null = _better_simpler_null(primary, null_assessments, metric)
    if reference_null is not None:
        delta = reference_null.metric_value(metric) - primary.metric_value(metric)
        if delta < _NULL_BASELINE_MIN_DELTA_AICC:
            caveat = (
                f"{primary.template.title} improves on the {reference_null.template.title} "
                f"null baseline by only Δ{metric.value} = {delta:.1f} "
                f"(< {_NULL_BASELINE_MIN_DELTA_AICC:.0f}); the data show no significant "
                "structure beyond the null."
            )
            summary = (
                f"No significant structure — {primary.template.title} does not beat the "
                f"{reference_null.template.title} null baseline by {metric.value}."
            )
            return replace(
                recommendation,
                metric=metric,
                recommended_key=reference_null.template.key,
                comparable_keys=(),
                summary=summary,
                confidence=ConfidenceTier.NONE,
                verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
                caveat=caveat,
            )

    # A genuine structured recommendation. Gates classify the confidence.
    if primary.residual_gate_passed:
        confidence = ConfidenceTier.HIGH
        caveat = ""
    else:
        confidence = ConfidenceTier.MEDIUM
        caveat = (
            "Structured residuals remain: "
            + ("; ".join(primary.residual_gate_reasons) or "residual checks flagged this fit")
            + "."
        )

    # Edge-of-window caveat: keep the tier as computed, but flag an
    # oscillation that only just clears the cycle-count disqualifier and has
    # weak (or no) spectral corroboration. Appended rather than replacing any
    # policy caveat already set above.
    edge_caveat = _edge_of_window_caveat(primary, recommendation.peak_analysis)
    if edge_caveat:
        caveat = (
            f"{caveat} {edge_caveat}".strip()
            if caveat
            else edge_caveat[0].upper() + edge_caveat[1:]
        )

    compare_summary = (
        ", with a similarly scoring alternative to inspect." if comparable_keys else "."
    )
    tier_note = "" if confidence is ConfidenceTier.HIGH else " (medium confidence)"
    summary = f"Recommended: {primary.template.title} by {metric.value}{tier_note}{compare_summary}"
    return replace(
        recommendation,
        metric=metric,
        recommended_key=primary.template.key,
        comparable_keys=comparable_keys,
        summary=summary,
        confidence=confidence,
        verdict=RecommendationVerdict.STRUCTURED,
        caveat=caveat,
    )


def serialize_fit_wizard_recommendation(
    recommendation: FitWizardRecommendation,
) -> dict[str, object]:
    """Return a JSON-serialisable snapshot of a single-fit wizard recommendation."""
    return {
        "fingerprint": _serialize_spectrum_fingerprint(recommendation.fingerprint),
        "templates": [
            _serialize_candidate_template(template) for template in recommendation.templates
        ],
        "assessments": [
            _serialize_candidate_assessment(assessment) for assessment in recommendation.assessments
        ],
        "metric": recommendation.metric.value,
        "recommended_key": recommendation.recommended_key,
        "comparable_keys": list(recommendation.comparable_keys),
        "summary": recommendation.summary,
        "peak_analysis": (
            serialize_peak_analysis(recommendation.peak_analysis)
            if recommendation.peak_analysis is not None
            else None
        ),
        "multiplet_matches": [
            serialize_multiplet_match(match) for match in recommendation.multiplet_matches
        ],
        "family_reports": [
            serialize_family_screening_report(report) for report in recommendation.family_reports
        ],
        "confidence": recommendation.confidence.value,
        "verdict": recommendation.verdict.value,
        "caveat": recommendation.caveat,
    }


def deserialize_fit_wizard_recommendation(
    payload: object,
) -> FitWizardRecommendation | None:
    """Rebuild a persisted single-fit wizard recommendation payload."""
    if not isinstance(payload, dict):
        return None

    fingerprint = _deserialize_spectrum_fingerprint(payload.get("fingerprint"))
    if fingerprint is None:
        return None

    templates = tuple(
        template
        for entry in payload.get("templates", [])
        if (template := _deserialize_candidate_template(entry)) is not None
    )
    assessments = tuple(
        assessment
        for entry in payload.get("assessments", [])
        if (assessment := _deserialize_candidate_assessment(entry)) is not None
    )
    comparable_keys = tuple(
        key for key in payload.get("comparable_keys", []) if isinstance(key, str)
    )
    peak_analysis = deserialize_peak_analysis(payload.get("peak_analysis"))
    multiplet_matches = tuple(
        match
        for entry in payload.get("multiplet_matches", []) or ()
        if (match := deserialize_multiplet_match(entry)) is not None
    )
    family_reports = tuple(
        report
        for entry in payload.get("family_reports", []) or ()
        if (report := deserialize_family_screening_report(entry)) is not None
    )
    return FitWizardRecommendation(
        fingerprint=fingerprint,
        templates=templates,
        assessments=assessments,
        metric=SelectionMetric.from_value(payload.get("metric", SelectionMetric.AICC.value)),
        recommended_key=(
            str(payload["recommended_key"]) if payload.get("recommended_key") is not None else None
        ),
        comparable_keys=comparable_keys,
        summary=str(payload.get("summary", "")),
        peak_analysis=peak_analysis,
        multiplet_matches=multiplet_matches,
        family_reports=family_reports,
        # Additive: tolerate old payloads that predate the confidence/verdict
        # fields — ``from_value`` defaults them to NONE/STRUCTURED sensibly.
        confidence=ConfidenceTier.from_value(payload.get("confidence")),
        verdict=RecommendationVerdict.from_value(payload.get("verdict")),
        caveat=str(payload.get("caveat", "")),
    )


def candidate_template_keys(
    templates: tuple[CandidateTemplate, ...] | list[CandidateTemplate],
) -> tuple[str, ...]:
    """Return the ordered template-key sequence for one candidate portfolio."""
    return tuple(template.key for template in templates)


def recommendation_template_keys(
    recommendation: FitWizardRecommendation,
) -> tuple[str, ...]:
    """Return the ordered template-key sequence embedded in one recommendation."""
    return candidate_template_keys(recommendation.templates)


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


def serialize_family_screening_report(report: FamilyScreeningReport) -> dict[str, object]:
    """Return a JSON-safe dict snapshot of a :class:`FamilyScreeningReport`.

    ``json`` cannot hold ``inf``, so a non-finite ``stage1_metric_value`` is
    serialized as ``None`` (and deserializes back to ``math.inf``).
    """
    metric = report.stage1_metric_value
    return {
        "family_key": str(report.family_key),
        "title": str(report.title),
        "stage1_template_key": str(report.stage1_template_key),
        "stage1_metric_value": (float(metric) if math.isfinite(metric) else None),
        "stage1_gate_passed": bool(report.stage1_gate_passed),
        "promoted": bool(report.promoted),
        "reason": str(report.reason),
        "stage2_template_keys": [str(key) for key in report.stage2_template_keys],
    }


def deserialize_family_screening_report(payload: object) -> FamilyScreeningReport | None:
    """Rebuild a :class:`FamilyScreeningReport` from a persisted dict, tolerating gaps."""
    if not isinstance(payload, dict):
        return None
    metric_payload = payload.get("stage1_metric_value", None)
    metric = math.inf if metric_payload is None else float(metric_payload)
    return FamilyScreeningReport(
        family_key=str(payload.get("family_key", "")),
        title=str(payload.get("title", "")),
        stage1_template_key=str(payload.get("stage1_template_key", "")),
        stage1_metric_value=metric,
        stage1_gate_passed=bool(payload.get("stage1_gate_passed", False)),
        promoted=bool(payload.get("promoted", False)),
        reason=str(payload.get("reason", "")),
        stage2_template_keys=tuple(
            str(key) for key in payload.get("stage2_template_keys", []) if isinstance(key, str)
        ),
    )


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
        "residuals": (
            np.asarray(result.residuals, dtype=float).tolist()
            if result.residuals is not None
            else None
        ),
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
    residuals_payload = payload.get("residuals")
    residuals: NDArray[np.float64] | None
    if residuals_payload is None:
        residuals = None
    else:
        try:
            residuals = np.asarray(residuals_payload, dtype=float)
        except (TypeError, ValueError):
            residuals = None
    return FitResult(
        success=bool(payload.get("success", False)),
        chi_squared=float(payload.get("chi_squared", 0.0)),
        reduced_chi_squared=float(payload.get("reduced_chi_squared", 0.0)),
        parameters=_deserialize_parameter_set(payload.get("parameters", [])),
        uncertainties=uncertainties,
        residuals=residuals,
        message=str(payload.get("message", "")),
    )


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


def _serialize_candidate_assessment(
    assessment: CandidateAssessment,
) -> dict[str, object]:
    return {
        "template": _serialize_candidate_template(assessment.template),
        "fit_result": _serialize_fit_result(assessment.fit_result),
        "aic": assessment.aic,
        "aicc": assessment.aicc,
        "bic": assessment.bic,
        "selected_score": assessment.selected_score,
        "residual_rms": assessment.residual_rms,
        "runs_z_score": assessment.runs_z_score,
        "max_abs_autocorrelation": assessment.max_abs_autocorrelation,
        "residual_fft_peak_snr": assessment.residual_fft_peak_snr,
        "residual_gate_passed": bool(assessment.residual_gate_passed),
        "residual_gate_reasons": list(assessment.residual_gate_reasons),
        "bound_hits": list(assessment.bound_hits),
        "fitted_time": np.asarray(assessment.fitted_time, dtype=float).tolist(),
        "fitted_curve": np.asarray(assessment.fitted_curve, dtype=float).tolist(),
        "component_curves": _serialize_component_curves(assessment.component_curves),
        "stage": assessment.stage,
        "disqualification_reasons": list(assessment.disqualification_reasons),
        "is_null_baseline": bool(assessment.is_null_baseline),
    }


def _deserialize_candidate_assessment(
    payload: object,
) -> CandidateAssessment | None:
    if not isinstance(payload, dict):
        return None
    template = _deserialize_candidate_template(payload.get("template"))
    fit_result = _deserialize_fit_result(payload.get("fit_result"))
    if template is None or fit_result is None:
        return None
    try:
        return CandidateAssessment(
            template=template,
            fit_result=fit_result,
            aic=float(payload.get("aic", float("inf"))),
            aicc=(float(payload["aicc"]) if payload.get("aicc") is not None else None),
            bic=float(payload.get("bic", float("inf"))),
            selected_score=float(payload.get("selected_score", float("inf"))),
            residual_rms=float(payload.get("residual_rms", float("inf"))),
            runs_z_score=float(payload.get("runs_z_score", float("inf"))),
            max_abs_autocorrelation=float(payload.get("max_abs_autocorrelation", float("inf"))),
            residual_fft_peak_snr=float(payload.get("residual_fft_peak_snr", float("inf"))),
            residual_gate_passed=bool(payload.get("residual_gate_passed", False)),
            residual_gate_reasons=tuple(
                reason
                for reason in payload.get("residual_gate_reasons", [])
                if isinstance(reason, str)
            ),
            bound_hits=tuple(
                name for name in payload.get("bound_hits", []) if isinstance(name, str)
            ),
            fitted_time=np.asarray(payload.get("fitted_time", []), dtype=float),
            fitted_curve=np.asarray(payload.get("fitted_curve", []), dtype=float),
            component_curves=_deserialize_component_curves(payload.get("component_curves", [])),
            stage=int(payload.get("stage", 2)),
            disqualification_reasons=tuple(
                reason
                for reason in payload.get("disqualification_reasons", [])
                if isinstance(reason, str)
            ),
            is_null_baseline=bool(payload.get("is_null_baseline", False)),
        )
    except (TypeError, ValueError):
        return None


def _assess_candidate_template(
    dataset: MuonDataset,
    fingerprint: SpectrumFingerprint,
    template: CandidateTemplate,
    *,
    fit_engine: FitEngine,
    metric: SelectionMetric,
    seed_context: TemplateSeedContext | None = None,
    variant_budget: int = 5,
    stage: int = 2,
    cancel_callback: Callable[[], bool] | None = None,
    migrad_ncall: int | None = None,
) -> CandidateAssessment:
    # Frequencies measured from spectral peaks are trusted seeds: the 0.5x/2x
    # variant scaling that rescues a blind FFT guess would only destroy them.
    frozen_scale_names: frozenset[str] = frozenset()
    if (
        seed_context is not None
        and seed_context.peak_analysis is not None
        and seed_context.peak_analysis.peaks
    ):
        frozen_scale_names = frozenset({"frequency"})
    attempts = _parameter_variants(
        _initial_parameters_for_template(dataset, fingerprint, template, seed_context=seed_context),
        template=template,
        variant_budget=variant_budget,
        frozen_scale_names=frozen_scale_names,
    )

    # Screening cap (Stage 1 only): forwarded to the engine's migrad drive, not
    # the scipy fallback below (scipy has no call-count knob).
    migrad_kwargs = {"ncall": migrad_ncall} if migrad_ncall is not None else None

    best_result: FitResult | None = None
    best_parameters: ParameterSet | None = None
    for parameters in attempts:
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Fit wizard analysis cancelled.")
        result = fit_engine.fit(
            dataset,
            template.model.function,
            _clone_parameter_set(parameters),
            cancel_callback=cancel_callback,
            migrad_kwargs=migrad_kwargs,
        )
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
    residual_gate_passed = not residual_gate_reasons
    disqualification_reasons = _disqualification_reasons(dataset, template, best_result, bound_hits)

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
        residual_gate_passed=residual_gate_passed,
        residual_gate_reasons=tuple(residual_gate_reasons),
        bound_hits=tuple(bound_hits),
        fitted_time=fitted_time,
        fitted_curve=fitted_curve,
        component_curves=component_curves,
        stage=stage,
        disqualification_reasons=tuple(disqualification_reasons),
        is_null_baseline=template.key in (_NULL_CONSTANT_KEY, _NULL_EXPONENTIAL_KEY),
    )


def _metric_value(metric: SelectionMetric, aic: float, aicc: float | None, bic: float) -> float:
    if metric == SelectionMetric.AIC:
        return aic
    if metric == SelectionMetric.BIC:
        return bic
    return aicc if aicc is not None else aic


def _assessment_sort_key(
    assessment: CandidateAssessment,
    metric: SelectionMetric,
) -> tuple[float, int, int, str]:
    return (
        float(assessment.metric_value(metric)),
        int(assessment.parameter_count),
        int(assessment.additive_terms),
        assessment.template.title,
    )


def _model_identity(
    model: CompositeModel,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(model.component_names),
        tuple(model.operators),
        tuple(model.open_parentheses),
        tuple(model.close_parentheses),
    )


def _count_zero_crossings(values: NDArray[np.float64]) -> int:
    if values.size < 2:
        return 0
    signs = np.sign(values)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))


def _count_turning_points(values: NDArray[np.float64]) -> int:
    if values.size < 3:
        return 0
    diffs = np.diff(np.asarray(values, dtype=float))
    if diffs.size < 2:
        return 0
    scale = max(float(np.max(np.abs(values))), _EPS)
    tol = 0.01 * scale
    signs = np.sign(np.where(np.abs(diffs) >= tol, diffs, 0.0))
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))


def _smoothing_window_points(n_points: int) -> int:
    if n_points <= 7:
        return 1
    window = max(5, n_points // 40)
    if window % 2 == 0:
        window += 1
    return min(window, 51)


def _moving_average(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    kernel = np.full(window, 1.0 / float(window), dtype=float)
    return np.convolve(padded, kernel, mode="valid")


def _dominant_peak_metrics(
    frequencies: NDArray[np.float64],
    magnitude: NDArray[np.float64],
) -> tuple[float, float]:
    if frequencies.size < 2 or magnitude.size < 2:
        return 0.0, 0.0

    freqs = np.asarray(frequencies, dtype=float)
    mags = np.asarray(magnitude, dtype=float)
    positive = np.flatnonzero(freqs > 0.0)
    if positive.size == 0:
        return 0.0, 0.0

    positive_mags = mags[positive]
    peak_idx_local = int(np.argmax(positive_mags))
    peak_idx = int(positive[peak_idx_local])
    peak_mag = float(mags[peak_idx])

    noise = np.delete(positive_mags, peak_idx_local)
    noise_floor = float(np.median(noise)) if noise.size else 0.0
    snr = peak_mag / max(noise_floor, _EPS)
    return float(freqs[peak_idx]), float(snr)


def _quadratic_curvature(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 3 or y.size < 3:
        return 0.0
    try:
        coeffs = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), deg=2)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 0.0
    return float(2.0 * coeffs[0])


def _signal_region_end(
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    initial_amplitude_estimate: float,
) -> int:
    if values.size == 0:
        return 0
    initial_sign = 1.0 if initial_amplitude_estimate >= 0.0 else -1.0
    amplitude = max(abs(initial_amplitude_estimate), _EPS)
    error_level = float(np.median(np.abs(errors))) if errors.size else 0.0
    threshold = max(0.1 * amplitude, 2.0 * error_level, _EPS)
    signed_values = initial_sign * np.asarray(values, dtype=float)
    min_points = min(values.size, max(12, values.size // 8))

    end = min_points
    for idx, value in enumerate(signed_values):
        if idx < min_points or value > threshold:
            end = idx + 1
            continue
        break
    return max(0, min(end, values.size))


def _monotonic_decay_fraction(
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    initial_amplitude_estimate: float,
) -> float:
    if values.size < 3:
        return 1.0
    initial_sign = 1.0 if initial_amplitude_estimate >= 0.0 else -1.0
    segment = np.asarray(values, dtype=float)
    diffs = np.diff(segment)
    expected_sign = -initial_sign
    amplitude = max(abs(initial_amplitude_estimate), _EPS)
    error_level = float(np.median(np.abs(errors))) if errors.size else 0.0
    threshold = max(0.08 * amplitude, 2.0 * error_level, _EPS)
    tol = max(0.005 * max(amplitude, float(np.max(np.abs(segment)))), error_level, _EPS)

    local_signal = np.maximum(np.abs(segment[:-1]), np.abs(segment[1:])) >= threshold
    informative = local_signal & (np.abs(diffs) >= tol)
    if not np.any(informative):
        return 1.0
    aligned = expected_sign * diffs[informative] >= 0.0
    return float(np.mean(aligned))


def _semilog_slope_ratio(
    x: NDArray[np.float64],
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    initial_amplitude_estimate: float,
) -> float:
    if x.size < 12 or values.size < 12:
        return 1.0
    end = _signal_region_end(values, errors, initial_amplitude_estimate)
    if end < 12:
        return 1.0

    initial_sign = 1.0 if initial_amplitude_estimate >= 0.0 else -1.0
    amplitude = max(abs(initial_amplitude_estimate), _EPS)
    error_level = float(np.median(np.abs(errors[:end]))) if errors.size else 0.0
    threshold = max(0.05 * amplitude, 2.0 * error_level, _EPS)

    segment_x = np.asarray(x[:end], dtype=float)
    segment_y = np.maximum(initial_sign * np.asarray(values[:end], dtype=float), threshold)
    if segment_x.size < 12 or np.any(~np.isfinite(segment_y)):
        return 1.0

    third = max(4, segment_x.size // 3)
    early_x = segment_x[:third]
    early_log_y = np.log(segment_y[:third])
    late_x = segment_x[-third:]
    late_log_y = np.log(segment_y[-third:])

    early_slope = _linear_slope(early_x, early_log_y)
    late_slope = _linear_slope(late_x, late_log_y)
    if not np.isfinite(early_slope) or not np.isfinite(late_slope):
        return 1.0

    early_mag = abs(early_slope)
    late_mag = abs(late_slope)
    if early_mag < _EPS or late_mag < _EPS:
        return 1.0
    return float(np.clip(max(early_mag, late_mag) / min(early_mag, late_mag), 1.0, 25.0))


def _initial_parameters_for_template(
    dataset: MuonDataset,
    fingerprint: SpectrumFingerprint,
    template: CandidateTemplate,
    seed_context: TemplateSeedContext | None = None,
) -> ParameterSet:
    t = np.asarray(dataset.time, dtype=float)
    y = np.asarray(dataset.asymmetry, dtype=float)
    duration = max(float(t[-1] - t[0]), _EPS) if t.size > 1 else 1.0
    dt = max(float(np.mean(np.diff(t))), _EPS) if t.size > 1 else 1.0
    nyquist = 0.5 / dt

    data_min = float(np.min(y)) if y.size else 0.0
    data_max = float(np.max(y)) if y.size else 0.0
    data_span = max(data_max - data_min, abs(fingerprint.initial_amplitude_estimate), 1.0)

    early_count = min(max(5, dataset.n_points // 20), dataset.n_points)
    early_t = t[:early_count]
    early_y = y[:early_count] - fingerprint.tail_estimate
    slope = _linear_slope(early_t, early_y)
    lambda_guess = max(
        0.05, min(20.0, -slope / max(abs(fingerprint.initial_amplitude_estimate), _EPS))
    )
    gaussian_width = math.sqrt(
        max(
            -fingerprint.early_time_curvature
            / max(2.0 * abs(fingerprint.initial_amplitude_estimate), _EPS),
            0.0,
        )
    )
    gaussian_width = max(0.05, min(20.0, gaussian_width if np.isfinite(gaussian_width) else 0.5))
    # Frequency seed from the dominant FFT line. FOLLOW-UP (high-TF envelope
    # seeding, GUI Round-4 AFM finding): for high transverse fields the precession
    # (e.g. ~800-1100 MHz) sits near/above the bin Nyquist, so the FFT peak and
    # this seed under-resolve it and the Oscillatory*Gaussian envelope fit stays
    # seed-fragile. A robust fix needs a field-derived frequency seed
    # (gamma_mu * B via spectral.field_gauss_to_frequency_mhz) plus finer
    # rebinning, which spans the GUI populate path; tracked as a separate change.
    frequency_guess = max(fingerprint.dominant_fft_frequency_mhz, 0.25 / duration)
    phase_guess = 0.0 if (y[0] - fingerprint.tail_estimate) >= 0.0 else math.pi

    # Peak-detection seeds supersede the single-line fingerprint guess; a
    # field-derived Larmor seed (gamma_mu * B) covers the high-TF case where the
    # dominant FFT bin under-resolves the precession (the FOLLOW-UP above).
    seed_peaks: tuple[DetectedPeak, ...] = ()
    if seed_context is not None and seed_context.peak_analysis is not None:
        seed_peaks = seed_context.peak_analysis.peaks
    if seed_peaks:
        frequency_guess = seed_peaks[0].frequency_mhz
    elif seed_context is not None and seed_context.field_gauss:
        larmor = field_gauss_to_frequency_mhz(seed_context.field_gauss)
        if 0.0 < larmor < nyquist:
            frequency_guess = larmor

    overrides: dict[str, float] = {}
    bounds_overrides: dict[str, tuple[float, float]] = {}
    fixed_names: set[str] = set()

    def _narrow_frequency_bounds(name: str, peak: DetectedPeak) -> None:
        half_width = max(5.0 * peak.width_mhz, 0.25 * peak.frequency_mhz)
        bounds_overrides[name] = (
            max(0.0, peak.frequency_mhz - half_width),
            min(peak.frequency_mhz + half_width, 0.98 * nyquist),
        )

    if template.key == "exp_constant":
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = {"A": amplitude, "Lambda": lambda_guess, "A_bg": fingerprint.tail_estimate}
    elif template.key == "gaussian_constant":
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = {"A": amplitude, "sigma": gaussian_width, "A_bg": fingerprint.tail_estimate}
    elif _is_additive_relaxation_mixture_template(template):
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = _additive_relaxation_mixture_overrides(
            template,
            amplitude=amplitude,
            lambda_guess=lambda_guess,
            gaussian_width=gaussian_width,
            tail_estimate=fingerprint.tail_estimate,
        )
    elif template.key == "stretched_constant":
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = {
            "A": amplitude,
            "Lambda": lambda_guess,
            "beta": 1.5,
            "A_bg": fingerprint.tail_estimate,
        }
    elif template.key in {"static_gkt_constant", "dynamic_gkt_constant"}:
        # Static and dynamic GKT share the same A/Delta/A_bg seed: the 1/3-tail
        # KT baseline and Delta from the early-time curvature. For the dynamic
        # variant, nu is left at the component default (FOLLOW-UP, KT hop-rate
        # seeding): a single static seed cannot span the static -> fast-
        # fluctuation range (~0.1 to >10 us^-1) and any fixed guess measurably
        # regresses one end (verified by an A/B sweep across regimes), so robust
        # multi-decade nu seeding (e.g. regime detection + a coarse nu variant
        # ladder) is deferred to a focused follow-up. The wizard still offers the
        # dynamic model and recovers weak/moderate dynamics from the Delta seed.
        amplitude = max(1.5 * abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        # A time-domain KT envelope match (dip + 1/3 tail) supplies a Delta seed
        # directly; it supersedes the early-time curvature guess and narrows the
        # bounds around the recognised width.
        kt_match = seed_context.best_match(("kt_envelope",)) if seed_context else None
        delta_seed = (kt_match.derived("Delta") if kt_match else None) or gaussian_width
        overrides = {
            "A": amplitude,
            "Delta": delta_seed,
            "A_bg": fingerprint.tail_estimate - amplitude / 3.0,
        }
        if kt_match:
            bounds_overrides["Delta"] = (0.5 * delta_seed, 2.0 * delta_seed)
    elif template.key == "static_gkt_exp_constant":
        amplitude = max(1.5 * abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = {
            "A": amplitude,
            "Delta": gaussian_width,
            "Lambda": max(0.02, lambda_guess * 0.5),
            "A_bg": fingerprint.tail_estimate - amplitude / 3.0,
        }
    elif template.key == "oscillatory_exp_constant":
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = {
            "A": amplitude,
            "frequency": min(frequency_guess, 0.98 * nyquist),
            "phase": phase_guess,
            "Lambda": lambda_guess,
            "A_bg": fingerprint.tail_estimate,
        }
    elif template.key == "oscillatory_gaussian_constant":
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = {
            "A": amplitude,
            "frequency": min(frequency_guess, 0.98 * nyquist),
            "phase": phase_guess,
            "sigma": gaussian_width,
            "A_bg": fingerprint.tail_estimate,
        }
    elif (multiplet := _MULTIPLET_TEMPLATE_KEY_RE.match(template.key)) is not None:
        # One damped cosine per detected line: frequencies/amplitudes from the
        # peaks; the envelope amplitude of each (Osc x Env) pair is fixed at 1
        # so the pair's scale lives in the oscillatory amplitude alone (the
        # parenthesised product would otherwise be A_i*A_j degenerate).
        n_components = int(multiplet.group(1))
        pair_peaks = _multiplet_seed_peaks(
            seed_context.peak_analysis if seed_context else None, n_components
        )
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        total = sum(max(peak.amplitude, 0.0) for peak in pair_peaks)
        overrides = {"A_bg": fingerprint.tail_estimate}
        for k in range(n_components):
            osc = 2 * k + 1
            env = 2 * k + 2
            share = 1.0 / max(n_components, 1)
            if k < len(pair_peaks) and total > 0.0:
                share = max(pair_peaks[k].amplitude, 0.0) / total or share
            overrides[f"A_{osc}"] = max(amplitude * share, _EPS)
            overrides[f"phase_{osc}"] = phase_guess
            overrides[f"A_{env}"] = 1.0
            fixed_names.add(f"A_{env}")
            overrides[f"Lambda_{env}"] = lambda_guess
            overrides[f"sigma_{env}"] = gaussian_width
            if k < len(pair_peaks):
                overrides[f"frequency_{osc}"] = pair_peaks[k].frequency_mhz
                _narrow_frequency_bounds(f"frequency_{osc}", pair_peaks[k])
            else:
                overrides[f"frequency_{osc}"] = min(frequency_guess, 0.98 * nyquist)
    elif template.key in _MUONIUM_TF_TEMPLATE_KEYS:
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        match = (
            seed_context.best_match(("muonium_low_tf", "muonium_high_tf")) if seed_context else None
        )
        a_hf = (match.derived("a_hf_mhz") if match else None) or VACUUM_MUONIUM_A_HF_MHZ
        overrides = {
            "A": amplitude,
            "A_hf": a_hf,
            "phase": phase_guess,
            "A_bg": fingerprint.tail_estimate,
        }
        bounds_overrides["A_hf"] = (0.5 * a_hf, 2.0 * a_hf) if match else (1.0, 4700.0)
    elif template.key == "muonium_zf_constant":
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        match = seed_context.best_match(("muonium_zf",)) if seed_context else None
        overrides = {"A": amplitude, "A_bg": fingerprint.tail_estimate}
        a_hf_zf = match.derived("a_hf_mhz") if match else None
        d_zf = match.derived("d_mhz") if match else None
        if a_hf_zf:
            overrides["A_hf"] = a_hf_zf
            bounds_overrides["A_hf"] = (0.5 * a_hf_zf, 2.0 * a_hf_zf)
        if d_zf:
            overrides["D_mu"] = d_zf
    elif template.key in _FMUF_TEMPLATE_KEYS:
        amplitude = max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        # Line-based (fmuf_linear/muf) and time-domain envelope matches both carry
        # a derived r_muF; either seeds the shape-critical distance.
        match = (
            seed_context.best_match(("fmuf_linear", "muf", "fmuf_envelope", "muF_envelope"))
            if seed_context
            else None
        )
        r_seed = (match.derived("r_muF_angstrom") if match else None) or _FMUF_DEFAULT_R_SEED
        overrides = {
            "A": amplitude,
            "r_muF": r_seed,
            "r1": r_seed,
            "r2": r_seed,
            "Lambda": max(0.02, 0.5 * lambda_guess),
            "A_bg": fingerprint.tail_estimate,
        }
        if match:
            # omega_d ~ r^-3, so +-40 % in frequency is roughly (0.7r, 1.4r).
            r_bounds = (0.7 * r_seed, 1.4 * r_seed)
            bounds_overrides["r_muF"] = r_bounds
            bounds_overrides["r1"] = r_bounds
            bounds_overrides["r2"] = r_bounds
    else:
        overrides = {
            "A": max(abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS),
            "A_bg": fingerprint.tail_estimate,
            "Lambda": lambda_guess,
            "sigma": gaussian_width,
            "Delta": gaussian_width,
            "beta": 1.5,
            "frequency": min(frequency_guess, 0.98 * nyquist),
            "phase": phase_guess,
        }

    # A measured peak narrows any remaining free frequency (single-oscillator
    # and Bessel templates reach here via their branches or the generic one).
    if seed_peaks and "frequency" not in bounds_overrides:
        _narrow_frequency_bounds("frequency", seed_peaks[0])

    # Applied-field parameters: seed from run metadata and pin them — the
    # field is measured, not fitted (muonium/vortex 'field'; LF 'B_L' is
    # seeded but left free so a miscalibrated magnet cannot wedge the fit).
    model_bases = {split_parameter_name(name)[0] for name in template.model.param_names}
    if seed_context is not None and seed_context.field_gauss:
        if "field" in model_bases:
            overrides.setdefault("field", seed_context.field_gauss)
            fixed_names.add("field")
        if "B_L" in model_bases:
            overrides.setdefault("B_L", seed_context.field_gauss)

    # Honour component-definition fixed parameters (e.g. VortexLattice 'field',
    # MuoniumLFRelax 'A_hf').
    definition_fixed = {
        fixed for component in template.model.components for fixed in component.fixed_params
    }

    parameters = ParameterSet()
    for name in template.model.param_names:
        base_name, _index = split_parameter_name(name)
        value = float(
            overrides.get(
                name,
                overrides.get(
                    base_name,
                    template.model.param_defaults.get(
                        name, template.model.param_defaults.get(base_name, 0.0)
                    ),
                ),
            )
        )
        bounds = bounds_overrides.get(name, bounds_overrides.get(base_name))
        if bounds is not None:
            p_min, p_max = bounds
        else:
            p_min, p_max = _parameter_bounds(
                base_name,
                value,
                data_min=data_min,
                data_max=data_max,
                data_span=data_span,
                duration=duration,
                nyquist=nyquist,
            )
        parameters.add(
            Parameter(
                name=name,
                value=float(np.clip(value, p_min, p_max)),
                min=p_min,
                max=p_max,
                fixed=(
                    name in fixed_names or base_name in fixed_names or base_name in definition_fixed
                ),
            )
        )
    return parameters


def _linear_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    try:
        coeffs = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), deg=1)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 0.0
    return float(coeffs[0])


def _parameter_bounds(
    base_name: str,
    value: float,
    *,
    data_min: float,
    data_max: float,
    data_span: float,
    duration: float,
    nyquist: float,
) -> tuple[float, float]:
    if base_name == "A":
        return 0.0, max(4.0 * abs(value), 4.0 * data_span, 1.0)
    if base_name == "A_bg":
        lower = min(data_min - data_span, value - data_span)
        upper = max(data_max + data_span, value + data_span)
        return lower, upper
    if base_name in {"Lambda", "sigma", "Delta"}:
        upper = max(8.0 * abs(value), 20.0 / max(duration, _EPS), 1.0)
        return 0.0, upper
    if base_name == "frequency":
        upper = max(min(0.98 * nyquist, max(8.0 * abs(value), 0.5)), 0.5)
        # An oscillation must complete at least one cycle in the observation
        # window (the spectral resolution limit): below 1/T a free-phase cosine
        # degenerates into a smooth envelope and lets oscillatory templates
        # "cheat" on non-oscillatory data. Peak-derived bounds bypass this via
        # bounds_overrides.
        lower = min(1.0 / max(duration, _EPS), 0.5 * upper)
        return lower, upper
    if base_name == "phase":
        return -math.pi, math.pi
    if base_name == "beta":
        return 0.1, 3.0
    if base_name in {"r_muF", "r1", "r2"}:
        return 0.0, max(10.0, 4.0 * abs(value))
    if base_name == "theta":
        return 0.0, 180.0
    if base_name == "Gamma":
        # Risch-Kehr 1D-diffusion rate (>= 0); a finite cap keeps the fit and
        # the χ²-quality verdict away from the degenerate Gamma -> 0 plateau.
        return 0.0, max(8.0 * abs(value), 20.0 / max(duration, _EPS), 1.0)
    if base_name == "B_L":
        # Longitudinal field magnitude (Gauss) >= 0; an unbounded-negative B_L
        # lets a Gaussian-broadened-KT candidate wander into a spurious
        # alternate minimum with structured residuals.
        return 0.0, max(8.0 * abs(value), 1000.0)
    if base_name == "w_rel":
        # Relative width of the Δ distribution (dimensionless, >= 0).
        return 0.0, max(4.0 * abs(value), 2.0)

    info = get_param_info(base_name)
    lower = float(info.default_min) if info.default_min is not None else -float("inf")
    upper = float("inf")
    return lower, upper


def _parameter_variants(
    base_parameters: ParameterSet,
    *,
    template: CandidateTemplate,
    variant_budget: int = 5,
    frozen_scale_names: frozenset[str] = frozenset(),
) -> tuple[ParameterSet, ...]:
    budget = max(1, int(variant_budget))
    if _is_additive_relaxation_mixture_template(template):
        return _additive_relaxation_mixture_variants(base_parameters, template)[:budget]
    if template.key in _FMUF_TEMPLATE_KEYS:
        return _fmuf_r_ladder_variants(base_parameters)[:budget]

    def _adjust(
        params: ParameterSet,
        *,
        scale: float = 1.0,
        amplitude_bias: float = 0.0,
        phase_shift: float = 0.0,
    ) -> ParameterSet:
        clone = _clone_parameter_set(params)
        for parameter in clone:
            base_name, _index = split_parameter_name(parameter.name)
            if parameter.fixed or base_name in frozen_scale_names:
                continue
            if base_name in {"Lambda", "sigma", "Delta", "frequency"} and parameter.value > 0.0:
                parameter.value = float(
                    np.clip(parameter.value * scale, parameter.min, parameter.max)
                )
            elif base_name == "A" and amplitude_bias != 0.0:
                parameter.value = float(
                    np.clip(parameter.value * (1.0 + amplitude_bias), parameter.min, parameter.max)
                )
            elif base_name == "A_bg" and amplitude_bias != 0.0:
                shift = amplitude_bias * 0.25 * max(abs(parameter.value), 1.0)
                parameter.value = float(
                    np.clip(parameter.value - shift, parameter.min, parameter.max)
                )
            elif base_name == "phase" and (phase_shift != 0.0 or "oscillatory" in template.key):
                wrapped = ((parameter.value + phase_shift + math.pi) % (2.0 * math.pi)) - math.pi
                parameter.value = float(np.clip(wrapped, parameter.min, parameter.max))
        return clone

    base = _clone_parameter_set(base_parameters)
    variants = (
        base,
        _adjust(base_parameters, scale=0.5),
        _adjust(base_parameters, scale=2.0),
        _adjust(base_parameters, amplitude_bias=0.25, phase_shift=math.pi / 6.0),
        _adjust(base_parameters, amplitude_bias=-0.25, phase_shift=-math.pi / 6.0),
    )
    return variants[:budget]


def _fmuf_r_ladder_variants(base_parameters: ParameterSet) -> tuple[ParameterSet, ...]:
    """Seed variants stepping ``r_muF`` (and ``r1``/``r2``) across the physical ladder.

    F-mu-F fits are shape-critical in the F-mu distance: a lone default seed can
    slide into a spurious ~0.53 A minimum (frequency ~1/r^3, so a far-off r gives
    a completely wrong beat), the representative then scores badly and the family
    loses promotion (failure F5). Each rung reseeds every distance parameter to a
    physical F-mu bond length so at least one attempt starts in the right basin.

    The **first** rung preserves the base seed — that is where a matched r_muF
    (from a time-domain envelope or line match) or the default already sits — so a
    trusted distance is tried first; the remaining rungs bracket it. Non-distance
    parameters are inherited unchanged from the base (amplitude/relaxation/bg were
    already seeded from the fingerprint).
    """
    distance_names = [
        parameter.name
        for parameter in base_parameters
        if split_parameter_name(parameter.name)[0] in _FMUF_DISTANCE_PARAM_BASES
        and not parameter.fixed
    ]
    if not distance_names:
        return (_clone_parameter_set(base_parameters),)

    variants: list[ParameterSet] = [_clone_parameter_set(base_parameters)]
    for r_value in _FMUF_R_LADDER[1:]:
        clone = _clone_parameter_set(base_parameters)
        for parameter in clone:
            if parameter.name in distance_names:
                parameter.value = float(np.clip(r_value, parameter.min, parameter.max))
        variants.append(clone)
    return tuple(variants)


def _parameter_variant_by_name(
    parameters: ParameterSet,
    scale_overrides: dict[str, float],
) -> ParameterSet:
    clone = _clone_parameter_set(parameters)
    for parameter in clone:
        scale = scale_overrides.get(parameter.name)
        if scale is None:
            continue
        parameter.value = float(np.clip(parameter.value * scale, parameter.min, parameter.max))
    return clone


def _is_additive_relaxation_mixture_template(template: CandidateTemplate) -> bool:
    component_names = template.model.component_names
    if len(component_names) < 3:
        return False
    if component_names[-1] != "Constant":
        return False
    if any(operator != "+" for operator in template.model.operators):
        return False
    relaxing_components = component_names[:-1]
    return len(relaxing_components) >= 2 and all(
        component_name in {"Exponential", "Gaussian"} for component_name in relaxing_components
    )


def _additive_relaxation_mixture_overrides(
    template: CandidateTemplate,
    *,
    amplitude: float,
    lambda_guess: float,
    gaussian_width: float,
    tail_estimate: float,
) -> dict[str, float]:
    overrides: dict[str, float] = {"A_bg": tail_estimate}
    relaxing_components = template.model.component_names[:-1]
    weights = _component_amplitude_weights(len(relaxing_components))
    exp_scales = _component_rate_scales(relaxing_components.count("Exponential"))
    gauss_scales = _component_rate_scales(relaxing_components.count("Gaussian"))

    exp_index = 0
    gauss_index = 0
    for component_index, component_name in enumerate(relaxing_components):
        mapping = template.model._param_mappings[component_index]  # noqa: SLF001
        overrides[mapping["A"]] = amplitude * weights[component_index]
        if component_name == "Exponential":
            overrides[mapping["Lambda"]] = max(
                0.01, min(20.0, lambda_guess * exp_scales[exp_index])
            )
            exp_index += 1
        else:
            overrides[mapping["sigma"]] = max(
                0.05, min(20.0, gaussian_width * gauss_scales[gauss_index])
            )
            gauss_index += 1
    return overrides


def _component_amplitude_weights(component_count: int) -> tuple[float, ...]:
    presets: dict[int, tuple[float, ...]] = {
        2: (0.65, 0.35),
        3: (0.55, 0.30, 0.15),
    }
    if component_count in presets:
        return presets[component_count]
    weights = np.geomspace(1.0, 0.35, component_count)
    weights = weights / np.sum(weights)
    return tuple(float(weight) for weight in weights)


def _component_rate_scales(component_count: int) -> tuple[float, ...]:
    presets: dict[int, tuple[float, ...]] = {
        1: (1.0,),
        2: (3.0, 0.3),
        3: (5.0, 1.0, 0.2),
    }
    if component_count in presets:
        return presets[component_count]
    scales = np.geomspace(4.0, 0.25, component_count)
    return tuple(float(scale) for scale in scales)


def _additive_relaxation_mixture_variants(
    base_parameters: ParameterSet,
    template: CandidateTemplate,
) -> tuple[ParameterSet, ...]:
    base = _clone_parameter_set(base_parameters)
    relaxing_components = template.model.component_names[:-1]
    component_count = len(relaxing_components)

    front_loaded_amp = tuple(float(scale) for scale in np.linspace(1.3, 0.75, component_count))
    back_loaded_amp = tuple(reversed(front_loaded_amp))
    front_loaded_rate = tuple(float(scale) for scale in np.geomspace(1.8, 0.6, component_count))
    back_loaded_rate = tuple(reversed(front_loaded_rate))

    return (
        base,
        _mixture_component_variant(
            base_parameters,
            template,
            component_amplitude_scales=tuple(1.0 for _ in relaxing_components),
            component_shape_scales=front_loaded_rate,
        ),
        _mixture_component_variant(
            base_parameters,
            template,
            component_amplitude_scales=tuple(1.0 for _ in relaxing_components),
            component_shape_scales=back_loaded_rate,
        ),
        _mixture_component_variant(
            base_parameters,
            template,
            component_amplitude_scales=front_loaded_amp,
            component_shape_scales=front_loaded_rate,
        ),
        _mixture_component_variant(
            base_parameters,
            template,
            component_amplitude_scales=back_loaded_amp,
            component_shape_scales=back_loaded_rate,
        ),
    )


def _mixture_component_variant(
    parameters: ParameterSet,
    template: CandidateTemplate,
    *,
    component_amplitude_scales: tuple[float, ...],
    component_shape_scales: tuple[float, ...],
) -> ParameterSet:
    scale_overrides: dict[str, float] = {}
    relaxing_components = template.model.component_names[:-1]
    for component_index, component_name in enumerate(relaxing_components):
        mapping = template.model._param_mappings[component_index]  # noqa: SLF001
        scale_overrides[mapping["A"]] = component_amplitude_scales[component_index]
        if component_name == "Exponential":
            scale_overrides[mapping["Lambda"]] = component_shape_scales[component_index]
        elif component_name == "Gaussian":
            scale_overrides[mapping["sigma"]] = component_shape_scales[component_index]
    return _parameter_variant_by_name(parameters, scale_overrides)


def _clone_parameter_set(parameters: ParameterSet) -> ParameterSet:
    return ParameterSet(
        [
            Parameter(
                name=parameter.name,
                value=parameter.value,
                min=parameter.min,
                max=parameter.max,
                fixed=parameter.fixed,
                expr=parameter.expr,
            )
            for parameter in parameters
        ]
    )


def _needs_fit_backend_fallback(result: FitResult) -> bool:
    return (
        not result.success
        and isinstance(result.message, str)
        and "iminuit import error" in result.message.lower()
    )


def _scipy_fit_fallback(
    dataset: MuonDataset,
    model_fn,
    parameters: ParameterSet,
) -> FitResult:
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return FitResult(
            success=False,
            message="SciPy fallback unavailable and iminuit fit backend could not be imported.",
        )

    free = parameters.free_parameters
    fixed = {parameter.name: parameter.value for parameter in parameters if parameter.fixed}
    names = [parameter.name for parameter in free]
    x0 = np.asarray([parameter.value for parameter in free], dtype=float)
    lower = np.asarray(
        [parameter.min if np.isfinite(parameter.min) else -np.inf for parameter in free],
        dtype=float,
    )
    upper = np.asarray(
        [parameter.max if np.isfinite(parameter.max) else np.inf for parameter in free],
        dtype=float,
    )
    errors = np.asarray(dataset.error, dtype=float)
    safe_errors = np.where(np.isfinite(errors) & (errors > 0.0), errors, 1.0)

    def _residual_vector(values: NDArray[np.float64]) -> NDArray[np.float64]:
        kwargs = {**fixed, **dict(zip(names, values, strict=True))}
        model_values = np.asarray(model_fn(dataset.time, **kwargs), dtype=float)
        return (np.asarray(dataset.asymmetry, dtype=float) - model_values) / safe_errors

    result = least_squares(_residual_vector, x0=x0, bounds=(lower, upper), max_nfev=4000)
    fitted_values = result.x if result.x.size else x0

    fitted_parameters = ParameterSet()
    uncertainties: dict[str, float] = {}
    cov = None
    if result.success and result.jac.size and free:
        try:
            jac = np.asarray(result.jac, dtype=float)
            jt_j = jac.T @ jac
            cov = np.linalg.pinv(jt_j)
            reduced = float(np.sum(np.square(_residual_vector(fitted_values)))) / max(
                len(dataset.time) - len(free), 1
            )
            diag = np.diag(cov) * max(reduced, 0.0)
            for name, variance in zip(names, diag, strict=True):
                if variance >= 0.0:
                    uncertainties[name] = float(np.sqrt(variance))
        except (TypeError, ValueError, np.linalg.LinAlgError):
            cov = None

    for parameter in parameters:
        if parameter.fixed:
            fitted_parameters.add(
                Parameter(
                    name=parameter.name,
                    value=parameter.value,
                    min=parameter.min,
                    max=parameter.max,
                    fixed=True,
                    expr=parameter.expr,
                )
            )
            continue
        idx = names.index(parameter.name)
        fitted_parameters.add(
            Parameter(
                name=parameter.name,
                value=float(fitted_values[idx]),
                min=parameter.min,
                max=parameter.max,
                fixed=False,
                expr=parameter.expr,
            )
        )

    fitted_kwargs = {parameter.name: parameter.value for parameter in fitted_parameters}
    fitted_model = np.asarray(model_fn(dataset.time, **fitted_kwargs), dtype=float)
    residuals = np.asarray(dataset.asymmetry, dtype=float) - fitted_model
    chi_squared = float(np.sum(np.square(residuals / safe_errors)))
    reduced_chi_squared = chi_squared / max(len(dataset.time) - len(free), 1)
    return FitResult(
        success=bool(result.success),
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        parameters=fitted_parameters,
        uncertainties=uncertainties,
        covariance=cov,
        residuals=residuals,
        message="Fit successful (SciPy fallback)" if result.success else str(result.message),
    )


def _residual_diagnostics(
    dataset: MuonDataset,
    fit_result: FitResult,
) -> tuple[float, float, float, float]:
    if fit_result.residuals is None or len(fit_result.residuals) == 0:
        return float("inf"), 0.0, 0.0, float("inf")

    residuals = np.asarray(fit_result.residuals, dtype=float)
    errors = np.asarray(dataset.error, dtype=float)
    if residuals.size != errors.size:
        errors = errors[: residuals.size]
    safe_errors = np.where(np.isfinite(errors) & (errors > 0.0), errors, 1.0)
    standardized = residuals / safe_errors
    residual_rms = (
        float(np.sqrt(np.mean(np.square(standardized)))) if standardized.size else float("inf")
    )

    runs_z_score = _runs_test_z_score(standardized)
    max_abs_autocorrelation = _max_abs_autocorrelation(standardized)
    if residual_rms <= 0.1 or float(np.max(np.abs(standardized), initial=0.0)) <= 0.25:
        return residual_rms, runs_z_score, max_abs_autocorrelation, 0.0

    residual_dataset = MuonDataset(
        time=np.asarray(dataset.time, dtype=float)[: residuals.size].copy(),
        asymmetry=residuals.copy(),
        error=safe_errors.copy(),
        metadata=dict(dataset.metadata),
        run=dataset.run,
    )
    frequencies, _real, magnitude = fft_asymmetry(
        residual_dataset,
        window=_DEFAULT_FFT_WINDOW,
        padding_factor=_DEFAULT_FFT_PADDING,
    )
    _peak_freq, residual_fft_peak_snr = _dominant_peak_metrics(frequencies, magnitude)
    return residual_rms, runs_z_score, max_abs_autocorrelation, residual_fft_peak_snr


def _runs_test_z_score(values: NDArray[np.float64]) -> float:
    if values.size < 3:
        return 0.0
    threshold = max(1e-6, 1e-3 * float(np.max(np.abs(values))))
    filtered = np.asarray(values, dtype=float).copy()
    filtered[np.abs(filtered) <= threshold] = 0.0
    if np.max(np.abs(filtered)) <= threshold:
        return 0.0
    signs = np.sign(filtered)
    signs = signs[signs != 0]
    if signs.size < 3:
        return 0.0
    positives = int(np.count_nonzero(signs > 0))
    negatives = int(np.count_nonzero(signs < 0))
    if positives == 0 or negatives == 0:
        return 0.0
    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
    total = positives + negatives
    mean = 1.0 + 2.0 * positives * negatives / total
    variance = (
        2.0
        * positives
        * negatives
        * (2.0 * positives * negatives - positives - negatives)
        / max(total * total * (total - 1), 1)
    )
    if variance <= 0.0:
        return 0.0
    return float((runs - mean) / math.sqrt(variance))


def _max_abs_autocorrelation(values: NDArray[np.float64]) -> float:
    if values.size < 4:
        return 0.0
    threshold = max(1e-6, 1e-3 * float(np.max(np.abs(values))))
    filtered = np.asarray(values, dtype=float).copy()
    filtered[np.abs(filtered) <= threshold] = 0.0
    if np.max(np.abs(filtered)) <= threshold:
        return 0.0
    centered = filtered - float(np.mean(filtered))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return 0.0
    max_lag = min(4, centered.size // 2)
    if max_lag <= 0:
        return 0.0
    correlations = []
    for lag in range(1, max_lag + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        correlations.append(abs(corr))
    return float(max(correlations, default=0.0))


def _fit_window_duration(dataset: MuonDataset) -> float:
    """Length T of the fit window (µs); sets the 1/T spectral-resolution floor."""
    if dataset.n_points < 2:
        return 0.0
    time = np.asarray(dataset.time, dtype=float)
    return float(time.max() - time.min())


def _paired_amplitude(parameters: ParameterSet, freq_index: str | None) -> Parameter | None:
    """Return the oscillation amplitude sharing a frequency parameter's suffix.

    An ``Oscillatory``/``Bessel`` component names its amplitude ``A`` and its
    frequency ``frequency`` with the same numeric suffix (``A_2`` ↔
    ``frequency_2``); an unindexed single-oscillator model uses bare ``A`` ↔
    ``frequency``. Match the amplitude to the frequency by that suffix so the
    zero-amplitude test targets the right component.
    """
    target = "A" if freq_index is None else f"A_{freq_index}"
    for parameter in parameters:
        if parameter.name == target:
            return parameter
    return None


def _apply_frequency_support_disqualifiers(
    dataset: MuonDataset,
    assessments: tuple[CandidateAssessment, ...],
    peak_analysis: PeakAnalysis | None,
) -> tuple[CandidateAssessment, ...]:
    """Disqualify fitted frequencies the data cannot actually support.

    Applied once at build time against the FINAL peak set (after the
    residual-FFT pass), not per-stage — a line found only by the residual FFT
    must still corroborate the Stage-1 assessment that carries it. Two prongs,
    both targeted at free-frequency templates (plain oscillatory/bessel; the
    multiplet templates seed their frequencies from detected peaks, and the
    muonium/fluorine families carry no literal ``frequency`` parameter):

    * **Effective-window floor** — the frequency completes fewer than
      ``_MIN_CYCLES_IN_EFFECTIVE_WINDOW`` cycles inside the SNR-truncated
      informative window (see :func:`effective_analysis_window`); such a claim
      rests on the noise-dominated tail or on smooth systematics.
    * **Spectral corroboration** — the frequency is resolvable in the effective
      window but no detected or user-declared peak lies within
      ``max(2·1/T_eff, _FREQUENCY_SUPPORT_REL_TOL·f)``. The validation
      programme's surviving false positives (flat-Ag-ZF 0.28 MHz cosine,
      Cu-ZF floor bessel, KT-plus-drift cosine) were all AICc-winning
      oscillations with no spectral support.

    Reasons are appended to the assessments' ``disqualification_reasons`` so
    the policy in :func:`rerank_fit_wizard_recommendation` (and any later
    metric rerank of the serialized payload) treats them like every other
    targeted disqualifier.
    """
    time = np.asarray(dataset.time, dtype=float)
    error = np.asarray(dataset.error, dtype=float)
    if time.size < 2:
        return assessments
    end = effective_analysis_window(time, error)
    if end < 2:
        return assessments
    t_eff = float(time[end - 1] - time[0])
    if t_eff <= _EPS:
        return assessments
    resolution_eff = 1.0 / t_eff
    peak_freqs = (
        tuple(peak.frequency_mhz for peak in peak_analysis.peaks)
        if peak_analysis is not None
        else ()
    )

    updated: list[CandidateAssessment] = []
    for assessment in assessments:
        if not assessment.is_successful or assessment.is_null_baseline:
            updated.append(assessment)
            continue
        reasons: list[str] = []
        for parameter in assessment.fit_result.parameters:
            if split_parameter_name(parameter.name)[0] != "frequency":
                continue
            frequency = abs(float(parameter.value))
            cycles = frequency * t_eff
            if cycles < _MIN_CYCLES_IN_EFFECTIVE_WINDOW:
                reasons.append(
                    f"{parameter.name} completes only {cycles:.1f} cycles inside the "
                    f"statistically informative window ({t_eff:.1f} µs)"
                )
                continue
            tolerance = max(2.0 * resolution_eff, _FREQUENCY_SUPPORT_REL_TOL * frequency)
            if not any(abs(peak - frequency) <= tolerance for peak in peak_freqs):
                reasons.append(
                    f"{parameter.name} = {frequency:.4g} MHz has no supporting "
                    "detected spectral peak"
                )
        if reasons:
            updated.append(
                replace(
                    assessment,
                    disqualification_reasons=(
                        tuple(assessment.disqualification_reasons) + tuple(reasons)
                    ),
                )
            )
        else:
            updated.append(assessment)
    return tuple(updated)


def _disqualification_reasons(
    dataset: MuonDataset,
    template: CandidateTemplate,
    fit_result: FitResult,
    bound_hits: Sequence[str],
) -> list[str]:
    """Targeted disqualifiers that suppress an otherwise metric-winning candidate.

    Metric-independent, so computed once at assessment time and carried on the
    assessment. Two families of failure, both specific to oscillatory templates
    (any component contributing a ``frequency`` parameter):

    * **Resolution-floor frequency** — a fitted ``frequency`` at or below the
      1/T spectral-resolution floor (within ``_FREQUENCY_FLOOR_SLACK``), or one
      pinned at its lower/upper bound. Below 1/T a free-phase cosine degenerates
      into a smooth envelope; such a "fit" is a systematics artefact, not a real
      line. This complements — never replaces — the 1/T anti-cheating bound on
      the seeds.
    * **Zero-consistent amplitude** — the paired oscillation amplitude with
      ``|A| < k·σ_A`` (``k = _ZERO_AMPLITUDE_SIGMA``). Skipped when the fitted
      error is missing/non-finite (we do not suppress on unknown uncertainty).
    """
    if not fit_result.success:
        return []
    parameters = fit_result.parameters
    freq_params = [
        parameter
        for parameter in parameters
        if split_parameter_name(parameter.name)[0] == "frequency"
    ]
    if not freq_params:
        return []

    reasons: list[str] = []
    duration = _fit_window_duration(dataset)
    floor = 1.0 / duration if duration > _EPS else 0.0
    bound_hit_set = set(bound_hits)
    uncertainties = fit_result.uncertainties or {}

    for parameter in freq_params:
        _base, index = split_parameter_name(parameter.name)
        value = abs(float(parameter.value))
        if floor > 0.0 and value <= floor * (1.0 + _FREQUENCY_FLOOR_SLACK):
            reasons.append(
                f"{parameter.name} at the 1/T resolution floor ({value:.4g} ≤ {floor:.4g} MHz)"
            )
        elif f"{parameter.name} at lower bound" in bound_hit_set:
            reasons.append(f"{parameter.name} pinned at its lower bound")
        elif f"{parameter.name} at upper bound" in bound_hit_set:
            reasons.append(f"{parameter.name} pinned at its upper bound")

        amplitude = _paired_amplitude(parameters, index)
        if amplitude is not None:
            sigma = float(uncertainties.get(amplitude.name, float("nan")))
            if np.isfinite(sigma) and sigma > 0.0:
                if abs(float(amplitude.value)) < _ZERO_AMPLITUDE_SIGMA * sigma:
                    reasons.append(
                        f"oscillation amplitude {amplitude.name} consistent with zero "
                        f"(|{amplitude.value:.3g}| < {_ZERO_AMPLITUDE_SIGMA:.0f}·{sigma:.3g})"
                    )
    return reasons


def _bound_hit_names(parameters: ParameterSet) -> list[str]:
    hits: list[str] = []
    for parameter in parameters:
        # The tolerance scale must ignore infinite bounds: an infinite |max|
        # would make ``tol`` infinite and flag every value as "at lower bound"
        # (any finite offset is <= inf). Components with one-sided bounds — e.g.
        # Risch-Kehr's Gamma in [0, inf) — would otherwise be spuriously gated.
        scale = max(
            abs(parameter.value),
            abs(parameter.min) if np.isfinite(parameter.min) else 0.0,
            abs(parameter.max) if np.isfinite(parameter.max) else 0.0,
            1.0,
        )
        tol = 1e-6 * scale
        if np.isfinite(parameter.min) and abs(parameter.value - parameter.min) <= tol:
            hits.append(f"{parameter.name} at lower bound")
        elif np.isfinite(parameter.max) and abs(parameter.value - parameter.max) <= tol:
            hits.append(f"{parameter.name} at upper bound")
    return hits


def _residual_gate_reasons(
    *,
    fit_result: FitResult,
    residual_rms: float,
    runs_z_score: float,
    max_abs_autocorrelation: float,
    residual_fft_peak_snr: float,
    bound_hits: list[str],
) -> list[str]:
    reasons: list[str] = []
    if not fit_result.success:
        reasons.append(fit_result.message or "Fit failed")
    if residual_rms > 1.25:
        reasons.append(f"standardized residual RMS is high ({residual_rms:.2f})")
    if abs(runs_z_score) > 2.0:
        reasons.append(f"runs-test z score suggests structure ({runs_z_score:.2f})")
    if max_abs_autocorrelation > 0.35:
        reasons.append(f"low-lag residual autocorrelation is high ({max_abs_autocorrelation:.2f})")
    if residual_fft_peak_snr > 6.0:
        reasons.append(f"residual FFT shows a strong peak (SNR {residual_fft_peak_snr:.2f})")
    reasons.extend(bound_hits)
    return reasons


def _dense_fit_curves(
    dataset: MuonDataset,
    model: CompositeModel,
    parameters: ParameterSet,
    *,
    fallback_parameters: ParameterSet | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[tuple[str, NDArray[np.float64]], ...]]:
    if dataset.n_points == 0:
        empty = np.array([], dtype=float)
        return empty, empty, ()

    param_values = _curve_parameter_values(
        model, parameters, fallback_parameters=fallback_parameters
    )
    n_samples = _curve_sample_count(dataset, param_values)
    fitted_time = np.linspace(float(dataset.time.min()), float(dataset.time.max()), n_samples)
    fitted_curve = np.asarray(model.function(fitted_time, **param_values), dtype=float)
    component_curves = tuple(
        model.evaluate_components(
            fitted_time,
            additive_only=True,
            **param_values,
        )
    )
    return fitted_time, fitted_curve, component_curves


def _curve_parameter_values(
    model: CompositeModel,
    parameters: ParameterSet,
    *,
    fallback_parameters: ParameterSet | None = None,
) -> dict[str, float]:
    param_values = {parameter.name: parameter.value for parameter in parameters}
    if fallback_parameters is not None:
        for parameter in fallback_parameters:
            param_values.setdefault(parameter.name, parameter.value)
    for name in model.param_names:
        param_values.setdefault(name, float(model.param_defaults.get(name, 0.0)))
    return param_values


def _curve_sample_count(dataset: MuonDataset, param_values: dict[str, float]) -> int:
    duration = float(dataset.time.max() - dataset.time.min()) if dataset.n_points > 1 else 0.0
    base_points = max(500, dataset.n_points * 4)
    if duration <= 0.0:
        return base_points

    max_frequency = 0.0
    for name, value in param_values.items():
        base_name, _index = split_parameter_name(name)
        if base_name == "frequency":
            max_frequency = max(max_frequency, abs(float(value)))

    if max_frequency <= 0.0:
        return base_points
    cycles = max_frequency * duration
    return int(max(base_points, min(20000, math.ceil(cycles * 40.0) + 1)))
