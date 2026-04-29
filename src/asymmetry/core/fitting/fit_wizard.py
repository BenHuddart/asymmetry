"""Core analysis helpers for the single-spectrum fit wizard."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.fourier.fft import fft_asymmetry

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
    "static_gkt_exp_constant": "Static GKT * Exponential + Constant",
    "oscillatory_exp_constant": "Oscillatory * Exponential + Constant",
    "oscillatory_gaussian_constant": "Oscillatory * Gaussian + Constant",
    "current_model": "Current Fit Function",
}


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
class FitWizardRecommendation:
    """Full fit-wizard analysis payload plus the current recommendation."""

    fingerprint: SpectrumFingerprint
    templates: tuple[CandidateTemplate, ...]
    assessments: tuple[CandidateAssessment, ...]
    metric: SelectionMetric
    recommended_key: str | None
    comparable_keys: tuple[str, ...]
    summary: str

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

    fft_dataset = MuonDataset(
        time=t.copy(),
        asymmetry=centered.copy(),
        error=np.asarray(dataset.error, dtype=float).copy(),
        metadata=dict(dataset.metadata),
        run=dataset.run,
    )
    frequencies, _real, magnitude = fft_asymmetry(
        fft_dataset,
        window=_DEFAULT_FFT_WINDOW,
        padding_factor=_DEFAULT_FFT_PADDING,
    )
    dominant_fft_frequency_mhz, dominant_fft_snr = _dominant_peak_metrics(frequencies, magnitude)
    duration = max(float(t[-1] - t[0]), _EPS) if t.size > 1 else 1.0
    dominant_fft_cycles_in_window = float(dominant_fft_frequency_mhz * duration)

    curvature_count = min(n_points, max(8, n_points // 6))
    early_time_curvature = _quadratic_curvature(t[:curvature_count], centered[:curvature_count])
    monotonic_decay_fraction = _monotonic_decay_fraction(
        smoothed_centered,
        np.asarray(dataset.error, dtype=float),
        initial_amplitude_estimate,
    )
    semilog_slope_ratio = _semilog_slope_ratio(
        t,
        smoothed_centered,
        np.asarray(dataset.error, dtype=float),
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

    if fingerprint.kt_like_hint:
        _add(
            "static_gkt_constant",
            CompositeModel(["StaticGKT_ZF", "Constant"], operators=["+"]),
            category="KT-like",
            rationale="Late-time dip and recovery are consistent with static Gaussian Kubo-Toyabe relaxation.",
        )
        _add(
            "static_gkt_exp_constant",
            CompositeModel(["StaticGKT_ZF", "Exponential", "Constant"], operators=["*", "+"]),
            category="KT-like",
            rationale="Adds a phenomenological exponential envelope to a static KT-like shape.",
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

    if current_model is not None:
        _add(
            "current_model",
            current_model,
            category="Baseline",
            rationale="Compares the wizard recommendation against the function already active in the fit tab.",
            baseline=True,
        )

    return tuple(templates)


def build_fit_wizard_recommendation(
    dataset: MuonDataset,
    current_model: CompositeModel | None = None,
    *,
    metric: SelectionMetric = SelectionMetric.AICC,
) -> FitWizardRecommendation:
    """Analyze one asymmetry spectrum and recommend a fit candidate."""
    fingerprint = fingerprint_spectrum(dataset)
    templates = tuple(build_candidate_templates(fingerprint, current_model=current_model))
    return build_fit_wizard_recommendation_for_templates(
        dataset,
        templates,
        fingerprint=fingerprint,
        metric=metric,
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
    fit_engine = FitEngine()
    assessments = tuple(
        _assess_candidate_template(
            dataset,
            active_fingerprint,
            template,
            fit_engine=fit_engine,
            metric=metric,
        )
        for template in active_templates
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


def rerank_fit_wizard_recommendation(
    recommendation: FitWizardRecommendation,
    metric: SelectionMetric,
) -> FitWizardRecommendation:
    """Reuse existing fit assessments and compute a recommendation for a new metric."""
    passing = [
        assessment
        for assessment in recommendation.assessments
        if assessment.is_successful and assessment.residual_gate_passed
    ]
    if not passing:
        summary = (
            "No candidate passed the residual checks automatically. "
            "Inspect the comparison table and residual warnings before applying a model."
        )
        return replace(
            recommendation,
            metric=metric,
            recommended_key=None,
            comparable_keys=(),
            summary=summary,
        )

    passing_sorted = sorted(
        passing, key=lambda assessment: _assessment_sort_key(assessment, metric)
    )
    primary = passing_sorted[0]
    comparable_keys: tuple[str, ...] = ()

    if len(passing_sorted) > 1:
        runner_up = passing_sorted[1]
        score_delta = abs(primary.metric_value(metric) - runner_up.metric_value(metric))
        if score_delta <= _COMPARABLE_SCORE_DELTA:
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

    if comparable_keys:
        compare_summary = ", with a similarly scoring alternative to inspect."
    else:
        compare_summary = "."

    summary = f"Recommended: {primary.template.title} by {metric.value}{compare_summary}"
    return replace(
        recommendation,
        metric=metric,
        recommended_key=primary.template.key,
        comparable_keys=comparable_keys,
        summary=summary,
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
) -> CandidateAssessment:
    attempts = _parameter_variants(
        _initial_parameters_for_template(dataset, fingerprint, template),
        template=template,
    )

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
    residual_gate_passed = not residual_gate_reasons

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
    frequency_guess = max(fingerprint.dominant_fft_frequency_mhz, 0.25 / duration)
    phase_guess = 0.0 if (y[0] - fingerprint.tail_estimate) >= 0.0 else math.pi

    overrides: dict[str, float] = {}
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
    elif template.key == "static_gkt_constant":
        amplitude = max(1.5 * abs(fingerprint.initial_amplitude_estimate), 0.25 * data_span, _EPS)
        overrides = {
            "A": amplitude,
            "Delta": gaussian_width,
            "A_bg": fingerprint.tail_estimate - amplitude / 3.0,
        }
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
            Parameter(name=name, value=float(np.clip(value, p_min, p_max)), min=p_min, max=p_max)
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
        return 0.0, upper
    if base_name == "phase":
        return -math.pi, math.pi
    if base_name == "beta":
        return 0.1, 3.0
    if base_name in {"r_muF", "r1", "r2"}:
        return 0.0, max(10.0, 4.0 * abs(value))
    if base_name == "theta":
        return 0.0, 180.0

    info = get_param_info(base_name)
    lower = float(info.default_min) if info.default_min is not None else -float("inf")
    upper = float("inf")
    return lower, upper


def _parameter_variants(
    base_parameters: ParameterSet,
    *,
    template: CandidateTemplate,
) -> tuple[ParameterSet, ...]:
    if _is_additive_relaxation_mixture_template(template):
        return _additive_relaxation_mixture_variants(base_parameters, template)

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
    return (
        base,
        _adjust(base_parameters, scale=0.5),
        _adjust(base_parameters, scale=2.0),
        _adjust(base_parameters, amplitude_bias=0.25, phase_shift=math.pi / 6.0),
        _adjust(base_parameters, amplitude_bias=-0.25, phase_shift=-math.pi / 6.0),
    )


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


def _bound_hit_names(parameters: ParameterSet) -> list[str]:
    hits: list[str] = []
    for parameter in parameters:
        tol = 1e-6 * max(abs(parameter.value), abs(parameter.min), abs(parameter.max), 1.0)
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
