"""Shared TF count calibration with beta fixed at one or fitted."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_domain import _fit_fb_count_groups
from asymmetry.core.fitting.count_results import CountFitParameter, JointCountFit
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.grouped_time_domain import build_count_groups
from asymmetry.core.fitting.likelihood import CountSource, FitInputError, Objective, PoissonCounts
from asymmetry.core.fitting.models import oscillatory
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.io.periods import select_period, source_run_of
from asymmetry.core.transform.corrections import CorrectionStatus
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

_MIN_AMPLITUDE_SIGNIFICANCE = 3.0
_MIN_FREQUENCY_SIGNIFICANCE = 3.0
_FALLBACK_FREQUENCY_MHZ = 1.0


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

    @property
    def n_bins_used(self) -> int:
        return len(self.fit.domains[0].time_us)

    @property
    def method(self) -> str:
        return "tf_calibration_fit"

    @property
    def model(self) -> str:
        return "exponential_common_phase"

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


def fit_tf_calibration(
    dataset: MuonDataset,
    request: TFCalibrationRequest,
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> TFCalibrationResult:
    """Fit one exponential/common-phase TF count model under fixed/free beta.

    Uses existing native alignment, exclusions, bunching and background preflight.
    Applied deadtime corrections use weighted corrected counts with retained
    measured/supplied variance; otherwise raw counts use Poisson deviance.
    Arbitrary supplied variance and explicit pulse-response requests are unsupported.
    Input limitations raise ``FitInputError``/``ValueError``; numerical failures
    retain their full fit with ``ok=False``. Cancellation raises ``FitCancelledError``.
    """
    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Fit cancelled.")
    if request.period is not None:
        dataset = select_period(dataset, request.period)
    run = dataset.run
    if run is None or not run.histograms:
        raise FitInputError("TF calibration needs a run with detector histograms")
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
        raise FitInputError(
            f"Raw Poisson TF calibration cannot use declared PostPileUp channels {processed}"
        )
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
        Parameter("Lambda", 0.1, min=0.0),
        Parameter("baseline", 0.0, fixed=True),
        Parameter("beta", 1.0, min=0.01, max=10.0, fixed=not request.fit_beta),
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
        lower = default.min if p.min == -np.inf else p.min
        upper = default.max if p.max == np.inf else p.max
        positive_balance = p.name in {"alpha", "beta"}
        nonnegative = default.min >= 0
        amplitude_ceiling = default.max if p.name == "A0" else np.inf
        if (
            np.isnan(lower)
            or np.isnan(upper)
            or lower > upper
            or (not p.fixed and lower == upper)
            or (nonnegative and lower < 0)
            or (positive_balance and lower <= 0)
            or upper > amplitude_ceiling
        ):
            raise ValueError(f"TF parameter bounds for {p.name!r} must retain its physical domain")
        if p.fixed and (
            not np.isfinite(p.value)
            or (nonnegative and p.value < 0)
            or (positive_balance and p.value <= 0)
            or p.value > amplitude_ceiling
        ):
            raise ValueError(f"TF fixed parameter {p.name!r} must lie in its physical domain")
        params.add(Parameter(p.name, p.value, min=lower, max=upper, fixed=p.fixed))
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
    grouped = _fit_fb_count_groups(
        dataset,
        groups,
        oscillatory,
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
    )
    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Fit cancelled.")
    return result
