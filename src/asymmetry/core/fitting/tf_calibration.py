"""Shared TF count calibration with beta fixed at one or fitted."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_domain import _fit_fb_count_groups
from asymmetry.core.fitting.count_results import CountFitParameter, JointCountFit
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.grouped_time_domain import build_count_groups
from asymmetry.core.fitting.likelihood import CountSource, FitInputError, Objective, PoissonCounts
from asymmetry.core.fitting.models import gaussian_relaxation, oscillatory, stretched_exponential
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.io.periods import select_period, source_run_of
from asymmetry.core.transform.corrections import CorrectionStatus
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

TF_CALIBRATION_METHOD = "tf_calibration_fit"

_MIN_AMPLITUDE_SIGNIFICANCE = 3.0
_MIN_FREQUENCY_SIGNIFICANCE = 3.0
_FALLBACK_FREQUENCY_MHZ = 1.0


class TFEnvelope(str, Enum):
    """Declared candidate order, also used to resolve exact cost ties."""

    UNDAMPED = "undamped"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    STRETCHED = "stretched"


def _undamped_tf(t, A0, frequency, phase=0.0, baseline=0.0):  # noqa: N803
    """Undamped cosine with no unused damping parameter."""
    return oscillatory(t, A0, frequency, phase, baseline=baseline)


def _gaussian_tf(t, A0, frequency, phase=0.0, sigma=0.1, baseline=0.0):  # noqa: N803
    """Gaussian exp[-(sigma*t)^2], with sigma in inverse microseconds."""
    return oscillatory(t, A0, frequency, phase) * gaussian_relaxation(t, 1.0, sigma) + baseline


def _stretched_tf(t, A0, frequency, phase=0.0, Lambda=0.1, stretch=1.0, baseline=0.0):  # noqa: N803
    """Positive shape exponent is independent of detector-amplitude beta."""
    return (
        oscillatory(t, A0, frequency, phase) * stretched_exponential(t, 1.0, Lambda, beta=stretch)
        + baseline
    )


_TF_MODELS = {
    TFEnvelope.UNDAMPED: (_undamped_tf, ()),
    TFEnvelope.EXPONENTIAL: (oscillatory, (CountFitParameter("Lambda", 0.1, min=0.0),)),
    TFEnvelope.GAUSSIAN: (_gaussian_tf, (CountFitParameter("sigma", 0.1, min=0.0),)),
    # A positive search floor avoids the singular p=0 limit; there is no upper cap.
    TFEnvelope.STRETCHED: (
        _stretched_tf,
        (CountFitParameter("Lambda", 0.1, min=0.0), CountFitParameter("stretch", 1.0, min=0.05)),
    ),
}


@dataclass(frozen=True)
class TFCalibrationRequest:
    """Intent applied to the supplied acquisition's existing grouping/corrections.

    ``period=None`` uses the supplied run; an explicit selector delegates to
    ``select_period``. Parameter overrides use the existing count-model names.
    The amplitude is percent, frequency MHz, phase radians and rate inverse µs.
    """

    forward_group: int
    backward_group: int
    fit_beta: bool = False
    period: int | str | None = None
    t_min: float | None = None
    t_max: float | None = None
    exclude: tuple[float, float] | None = None
    frequency_mhz: float | None = None
    parameters: tuple[CountFitParameter, ...] = ()
    pulse_separation_us: float | None = None
    minos: bool = False

    def __post_init__(self):
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if self.exclude is not None:
            object.__setattr__(self, "exclude", tuple(self.exclude))
        if self.forward_group == self.backward_group:
            raise ValueError("TF calibration needs two distinct detector groups")
        if self.pulse_separation_us is not None:
            raise FitInputError("TF calibration pulse-response fitting is unsupported")


@dataclass(frozen=True)
class TFCalibrationResult:
    """Complete fit and a separate calibration verdict; errors are conditional.

    Successful minimisation can have unavailable/unreliable covariance. Free
    beta also requires the existing resolved-precession/error safeguards; fixed
    beta does not acquire those beta-identifiability gates or a fitted error.
    """

    request: TFCalibrationRequest
    fit: JointCountFit
    source_run_number: int
    period_number: int | None
    source_file: str
    ok: bool
    message: str
    envelope: TFEnvelope = TFEnvelope.EXPONENTIAL

    @property
    def n_bins_used(self) -> int:
        return len(self.fit.domains[0].time_us)

    @property
    def method(self) -> str:
        return TF_CALIBRATION_METHOD

    @property
    def model(self) -> str:
        return f"{self.envelope.value}_common_phase"

    @property
    def alpha(self) -> float:
        return next(p.value for p in self.fit.parameters if p.name == "alpha")

    @property
    def beta(self) -> float:
        return next(p.value for p in self.fit.parameters if p.name == "beta")

    @property
    def beta_fixed(self) -> bool:
        return not self.request.fit_beta

    @property
    def alpha_error(self) -> float | None:
        return self.fit.standard_error("alpha")

    @property
    def beta_error(self) -> float | None:
        return self.fit.standard_error("beta")

    @property
    def alpha_beta_correlation(self) -> float | None:
        return self.fit.correlation("alpha", "beta")


def _seed_frequency_mhz(dataset: MuonDataset, field_tesla: float | None) -> float:
    """Precession-frequency seed (MHz) for the oscillatory model.

    Prefers an explicit ``field_tesla``; otherwise falls back to the dataset's
    recorded field (Gauss). A missing or zero field yields
    :data:`_FALLBACK_FREQUENCY_MHZ` — a calibration run normally carries the
    field, so this is only a floor to keep the fit runnable.
    """
    if field_tesla is not None and np.isfinite(field_tesla) and field_tesla != 0.0:
        return float(MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * abs(float(field_tesla)))
    field_gauss = float(dataset.run.field) if dataset.run is not None else 0.0
    if np.isfinite(field_gauss) and field_gauss != 0.0:
        return float(MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * abs(field_gauss))
    return _FALLBACK_FREQUENCY_MHZ


def _amplitude_is_significant(value: float | None, error: float | None) -> bool:
    """Whether a fitted amplitude is resolved above zero (precession present)."""
    if value is None or not np.isfinite(value):
        return False
    if abs(value) <= 1.0e-9:
        return False
    if error is None or not np.isfinite(error) or error <= 0.0:
        # No usable error: accept only a clearly non-zero amplitude.
        return abs(value) > 1.0e-6
    return abs(value) >= _MIN_AMPLITUDE_SIGNIFICANCE * error


def _frequency_is_resolved(value: float | None, error: float | None) -> bool:
    """Whether a fitted precession frequency is resolved above zero.

    A flat / non-precessing asymmetry drives the fitted frequency to zero; when
    it is consistent with zero (or has no usable error) there is no resolved
    precession and β is degenerate.
    """
    if value is None or not np.isfinite(value):
        return False
    if error is None or not np.isfinite(error) or error <= 0.0:
        return False
    return abs(value) >= _MIN_FREQUENCY_SIGNIFICANCE * error


def _require_acquired_tf_counts(run, groups):
    """Correction changes means, not the independent integer acquisition precondition."""
    for group in groups:
        offsets = dict(group.metadata["grouped_alignment"]["detector_offsets"])
        for index in group.detector_indices:
            bins = np.concatenate(
                [
                    np.arange(first - offsets[index], last - offsets[index] + 1)
                    for first, last in group.metadata["count_fit_raw_windows"]
                ]
            )
            histogram = run.histograms[index]
            PoissonCounts(
                histogram.time_axis[bins], histogram.counts[bins], CountSource.RAW, run.run_number
            )


def _prepare_tf_counts(dataset, request, cancel_callback):
    """Resolve the acquisition and one bank pair for every candidate."""
    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Fit cancelled.")
    if request.period is not None:
        dataset = select_period(dataset, request.period)
    run = dataset.run
    if run is None or not run.histograms:
        raise FitInputError("TF calibration needs a run with detector histograms")
    from asymmetry.core.transform.constituent_counts import single_acquisition_model_source

    source = single_acquisition_model_source(run)
    if source is None:
        raise FitInputError(
            "TF calibration requires an original acquisition; calibrate a source run and apply its shared alpha/beta to the combination."
        )
    if source is not run:
        run = source
        dataset = replace(dataset, run=source, metadata=source.metadata)
    if run.metadata.get("raw_period_histograms_available") is False:
        raise FitInputError(
            "TF calibration needs acquired period histograms, not a cached reduction"
        )
    indices = [
        effective_group_indices(run.grouping, gid, n_histograms=len(run.histograms))
        for gid in (request.forward_group, request.backward_group)
    ]
    if set(indices[0]) & set(indices[1]):
        raise FitInputError("Independent Poisson TF banks must have disjoint detectors")
    labels = run.metadata.get("histogram_labels") or run.grouping.get("histogram_labels", ())
    processed = [
        i + 1
        for bank in indices
        for i in bank
        if i < len(labels) and "postpileup" in labels[i].casefold().replace(" ", "")
    ]
    if processed:
        raise FitInputError(f"TF calibration cannot use declared PostPileUp channels {processed}")
    groups = build_count_groups(
        dataset,
        [request.forward_group, request.backward_group],
        t_min=request.t_min,
        t_max=request.t_max,
        exclude=request.exclude,
        lifetime_corrected=False,
    )
    objective = (
        Objective.GAUSSIAN
        if groups[0].deadtime.status is CorrectionStatus.APPLIED
        else Objective.POISSON
    )
    if objective is Objective.GAUSSIAN:
        _require_acquired_tf_counts(run, groups)
    if objective is Objective.POISSON and any(
        run.histograms[i].count_variance is not None for bank in indices for i in bank
    ):
        raise FitInputError("Raw Poisson TF calibration cannot use supplied counting variance")
    return dataset, groups, objective


def _tf_parameters(dataset, request, envelope):
    """Validate user constraints against the effective candidate's domain."""
    frequency = (
        request.frequency_mhz
        if request.frequency_mhz is not None
        else _seed_frequency_mhz(dataset, None)
    )
    # A positive background seed avoids the zero-gradient bound transform at zero.
    defaults = [
        Parameter("alpha", 1.0, min=1e-6, max=100.0),
        Parameter("N0", float("nan"), min=0.0),
        Parameter("background", 1.0, min=0.0),
        Parameter("background_b", 1.0, min=0.0),
        Parameter("A0", 20.0, min=0.0, max=100.0),
        Parameter("frequency", frequency, min=0.0),
        Parameter("phase", 0.0),
        *(Parameter(p.name, p.value, min=p.min, max=p.max) for p in _TF_MODELS[envelope][1]),
        Parameter("baseline", 0.0, fixed=True),
        Parameter(
            "beta",
            1.0,
            min=0.01 if request.fit_beta else -np.inf,
            max=10.0 if request.fit_beta else np.inf,
            fixed=not request.fit_beta,
        ),
    ]
    overrides = {p.name: p for p in request.parameters}
    if len(overrides) != len(request.parameters) or overrides.keys() - {p.name for p in defaults}:
        raise ValueError("TF parameter overrides need unique supported count-model names")
    if "beta" in overrides and (
        overrides["beta"].fixed == request.fit_beta
        or (not request.fit_beta and overrides["beta"].value != 1.0)
    ):
        raise ValueError("TF beta must be free for joint calibration or fixed at 1 for alpha-only")
    if (
        request.fit_beta
        and "frequency" in overrides
        and overrides["frequency"].fixed
        and (not np.isfinite(overrides["frequency"].value) or overrides["frequency"].value <= 0)
    ):
        raise ValueError("Joint TF fixed frequency must be finite and positive")
    if "baseline" in overrides and (
        not overrides["baseline"].fixed or overrides["baseline"].value != 0
    ):
        raise ValueError("TF baseline must be fixed at zero; count backgrounds own constant levels")
    params = ParameterSet()
    for default in defaults:
        p = overrides.get(default.name)
        if p is None:
            params.add(default)
            continue
        # Fixed declarations do not inherit numerical search bounds.
        lower = default.min if not p.fixed and p.min == -np.inf else p.min
        upper = default.max if not p.fixed and p.max == np.inf else p.max
        strictly_positive = p.name in {"alpha", "beta", "stretch"}
        nonnegative = default.min >= 0
        amplitude_ceiling = default.max if p.name == "A0" else np.inf
        if (
            np.isnan(lower)
            or np.isnan(upper)
            or lower > upper
            or (not p.fixed and lower == upper)
            or (nonnegative and np.isfinite(lower) and lower < 0)
            or (strictly_positive and np.isfinite(lower) and lower <= 0)
            or (np.isfinite(upper) and upper > amplitude_ceiling)
        ):
            raise ValueError(f"TF parameter bounds for {p.name!r} must retain its physical domain")
        if p.fixed and (
            not np.isfinite(p.value)
            or (nonnegative and p.value < 0)
            or (strictly_positive and p.value <= 0)
            or p.value > amplitude_ceiling
            or p.value < lower
            or p.value > upper
        ):
            raise ValueError(
                f"TF fixed parameter {p.name!r} must lie in its physical domain and explicit bounds"
            )
        params.add(Parameter(p.name, p.value, min=lower, max=upper, fixed=p.fixed))
    return params


def _fit_prepared_tf(dataset, request, groups, objective, params, envelope, cancel_callback):
    """Fit and assess one candidate with the shared full nuisance covariance."""
    run = dataset.run
    grouped = _fit_fb_count_groups(
        dataset,
        groups,
        _TF_MODELS[envelope][0],
        params,
        estimate_beta=True,
        cost=objective.value,
        minos=request.minos,
        cancel_callback=cancel_callback,
    )
    fit = grouped.joint_fit
    if fit is None:
        raise RuntimeError("TF calibration requires the full joint count-fit solution")
    ok, message = fit.success, fit.message
    if ok and request.fit_beta:
        fitted = {p.name: p for p in fit.parameters}
        if not _amplitude_is_significant(fitted["A0"].value, fit.standard_error("A0")) or (
            not fitted["frequency"].fixed
            and not _frequency_is_resolved(
                fitted["frequency"].value, fit.standard_error("frequency")
            )
        ):
            ok, message = (
                False,
                "No resolved precession signal (amplitude or frequency consistent with zero); β is degenerate without precession.",
            )
        elif fit.standard_error("beta") is None:
            ok, message = (
                False,
                "β could not be constrained (no finite error); the run may lack a resolved precession signal.",
            )
    result = TFCalibrationResult(
        request,
        fit,
        source_run_of(run),
        run.metadata.get("period_number"),
        run.source_file,
        ok,
        message,
        envelope,
    )
    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Fit cancelled.")
    return result


def fit_tf_calibration(
    dataset: MuonDataset,
    request: TFCalibrationRequest,
    *,
    envelope: TFEnvelope = TFEnvelope.EXPONENTIAL,
    cancel_callback: Callable[[], bool] | None = None,
) -> TFCalibrationResult:
    """Fit one explicit envelope (exponential/common-phase by default).

    Applied deadtime uses retained-variance weighted counts; off/skipped correction
    uses raw Poisson deviance. Input limitations raise FitInputError/ValueError;
    numerical failures retain the fit with ok=False. Cancellation raises
    FitCancelledError. This single-model operation does not select an envelope.
    """
    envelope = TFEnvelope(envelope)
    dataset, groups, objective = _prepare_tf_counts(dataset, request, cancel_callback)
    params = _tf_parameters(dataset, request, envelope)
    return _fit_prepared_tf(dataset, request, groups, objective, params, envelope, cancel_callback)


@dataclass(frozen=True)
class TFCalibrationSelection:
    """All four candidate fits and their separate calibration/covariance states.

    The recommendation uses minimum summed cost among usable candidates. Errors
    are conditional on each envelope and exclude model-selection uncertainty.
    """

    candidates: tuple[TFCalibrationResult, ...]

    @property
    def selected(self) -> TFCalibrationResult | None:
        return min(
            (
                candidate
                for candidate in self.candidates
                if candidate.ok and np.isfinite(candidate.fit.statistic)
            ),
            key=lambda candidate: candidate.fit.statistic,
            default=None,
        )


def select_tf_calibration(
    dataset: MuonDataset,
    request: TFCalibrationRequest,
    *,
    envelope_parameters: Mapping[TFEnvelope, tuple[CountFitParameter, ...]] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> TFCalibrationSelection:
    """Compare four envelopes on one prepared F/B count pair.

    Request overrides constrain common parameters in every candidate. Damping
    overrides belong in ``envelope_parameters``: Lambda for exponential,
    sigma for Gaussian, Lambda and stretch for stretched; none for undamped.
    All constraints are validated before fitting. Exact ties follow TFEnvelope
    order. A failed fit retains its result/message and does not remove other
    candidates; no usable candidate gives ``selected=None``. Cancellation raises
    FitCancelledError, including between candidates, and returns no partial set.
    """
    specific = envelope_parameters or {}
    if set(specific) - set(TFEnvelope):
        raise ValueError("Unknown TF envelope in parameter overrides")
    damping_names = {p.name for _, defaults in _TF_MODELS.values() for p in defaults}
    if any(p.name in damping_names for p in request.parameters):
        raise ValueError("Selection damping overrides belong in envelope_parameters")
    requests = []
    for envelope in TFEnvelope:
        overrides = tuple(specific.get(envelope, ()))
        if {p.name for p in overrides} - {p.name for p in _TF_MODELS[envelope][1]}:
            raise ValueError(f"Unsupported damping override for {envelope.value}")
        requests.append(replace(request, parameters=(*request.parameters, *overrides)))
    dataset, groups, objective = _prepare_tf_counts(dataset, request, cancel_callback)
    parameters = [_tf_parameters(dataset, r, e) for r, e in zip(requests, TFEnvelope, strict=True)]
    candidates = []
    for envelope, effective_request, params in zip(TFEnvelope, requests, parameters, strict=True):
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Fit cancelled.")
        candidates.append(
            _fit_prepared_tf(
                dataset, effective_request, groups, objective, params, envelope, cancel_callback
            )
        )
    return TFCalibrationSelection(tuple(candidates))
