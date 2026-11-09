"""Explicit script calibration delegates preparation and fitting to core owners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from uuid import uuid4

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_results import CountFitParameter
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.tf_calibration import (
    TF_CALIBRATION_METHOD,
    TFCalibrationRequest,
    TFEnvelope,
    fit_tf_calibration,
    select_tf_calibration,
)
from asymmetry.core.io.periods import period_count, select_period, source_run_of
from asymmetry.core.project.calibration import CalibrationRecord, calibration_payload
from asymmetry.core.project.json_values import encode_non_finite
from asymmetry.core.transform.asymmetry import (
    ALPHA_ESTIMATION_METHODS,
    AlphaEstimate,
    estimate_alpha,
    estimate_alpha_detailed,
)
from asymmetry.core.transform.corrections import reference_observation_metadata
from asymmetry.core.transform.fb_counts import ForwardBackwardCounts, native_count_time
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.reduce import corrected_period_counts, correction_flags_from_grouping
from asymmetry.core.workflow.reduction import (
    GREEN_MINUS_RED,
    ReductionSettings,
    resolve_reduction_grouping,
)

CALIBRATION_METHODS = (*ALPHA_ESTIMATION_METHODS, TF_CALIBRATION_METHOD)


@dataclass(frozen=True)
class CalibrationOptions:
    """Explicit operation; an omitted method preserves the automatic ratio policy.

    TF with no envelope selects among TFEnvelope in declared order. Parameter
    constraints use the shared TF request vocabulary. Period is the source's
    acquired period, independent of a target ReductionSettings.period.
    """

    method: str | None = None
    fit_beta: bool = False
    envelope: TFEnvelope | None = None
    period: int | str | None = None
    frequency_mhz: float | None = None
    parameters: tuple[CountFitParameter, ...] = ()
    exclude: tuple[float, float] | None = None
    minos: bool = False

    def __post_init__(self):
        if self.method is not None and self.method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown calibration method {self.method!r}")
        if self.method != TF_CALIBRATION_METHOD and (
            self.fit_beta
            or self.envelope is not None
            or self.frequency_mhz is not None
            or self.parameters
            or self.exclude is not None
            or self.minos
        ):
            raise ValueError(
                "Envelope, frequency, parameters and free beta require tf_calibration_fit"
            )
        if self.envelope is not None:
            object.__setattr__(self, "envelope", TFEnvelope(self.envelope))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if self.exclude is not None:
            object.__setattr__(self, "exclude", tuple(self.exclude))


@dataclass(frozen=True)
class CalibrationOutcome:
    """A shared typed result with source/pair context, without copied coefficients."""

    result: AlphaEstimate | CalibrationRecord
    run_number: int
    forward_group: int
    backward_group: int
    period_number: int | None
    automatic_ratio: bool = False
    counts: ForwardBackwardCounts | None = None

    @property
    def record(self) -> CalibrationRecord | None:
        return self.result if isinstance(self.result, CalibrationRecord) else None

    @property
    def selected(self):
        return self.record.selected if self.record is not None else self.result

    @property
    def alpha(self) -> float | None:
        return self.selected.alpha if self.selected is not None and self.selected.ok else None

    @property
    def method(self) -> str:
        if self.automatic_ratio:
            return "per_run_estimate"
        return TF_CALIBRATION_METHOD if self.record is not None else self.result.method

    def to_dict(self) -> dict:
        """Derive scalar views; preserve the canonical block before JSON sanitising."""
        selected = self.selected
        data = {
            "run_number": self.run_number,
            "period_number": self.period_number,
            "forward_group": self.forward_group,
            "backward_group": self.backward_group,
            "method": self.method,
            "alpha": self.alpha,
            "ok": selected is not None and selected.ok,
            "message": "No usable TF candidate" if selected is None else selected.message,
        }
        if self.record is not None:
            data["calibration"] = encode_non_finite(calibration_payload(self.record))
            data.update(
                alpha_error=None if selected is None else selected.alpha_error,
                beta=None if selected is None else selected.beta,
                beta_error=None if selected is None else selected.beta_error,
                beta_fixed=None if selected is None else selected.beta_fixed,
                alpha_beta_correlation=None
                if selected is None
                else selected.alpha_beta_correlation,
                model=None if selected is None else selected.model,
                uncertainty_status=None if selected is None else selected.fit.uncertainty_status,
            )
        else:
            data.update(
                asdict(self.result),
                alpha=self.alpha,
                method=self.method,
                beta=1.0,
                beta_fixed=True,
                beta_error=None,
            )
            data["corrections"] = self.counts.corrections.to_dict()
            data["alignment"] = self.counts.alignment.to_dict()
            data["window"] = asdict(self.counts.window)
        return data


def calibrate_run(
    loaded: MuonDataset | list[MuonDataset],
    settings: ReductionSettings,
    options: CalibrationOptions = CalibrationOptions(),
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> CalibrationOutcome:
    """Calibrate the source acquisition, never a target's cached or mapped curve.

    Explicit TF requests on multi-period acquisitions require a selected acquired
    period. The historical ratio uses red for a difference and preserves its
    single-period source adaptation. Corrections/timing come from the source
    profile; TF owns its own native-window and fitted-background preparation.
    """

    def check_cancel():
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Fit cancelled.")

    check_cancel()
    dataset = calibration_source(loaded, settings, options)
    tf = options.method == TF_CALIBRATION_METHOD
    run = dataset.run
    if run is None or not run.histograms:
        raise ValueError("Calibration requires source detector histograms")
    source_settings = replace(settings, alpha=1.0, alpha_source="assumed", period=None)
    grouping = resolve_reduction_grouping(run, source_settings)
    forward, backward = int(grouping["forward_group"]), int(grouping["backward_group"])
    prepared = replace(dataset, run=replace(run, grouping=dict(run.grouping) | grouping))
    if tf:
        request = TFCalibrationRequest(
            forward,
            backward,
            options.fit_beta,
            t_min=settings.t_min,
            t_max=settings.t_max,
            frequency_mhz=options.frequency_mhz,
            parameters=options.parameters,
            exclude=options.exclude,
            minos=options.minos,
        )
        result = (
            select_tf_calibration(prepared, request, cancel_callback=cancel_callback)
            if options.envelope is None
            else fit_tf_calibration(
                prepared, request, envelope=options.envelope, cancel_callback=cancel_callback
            )
        )
        result = CalibrationRecord(str(uuid4()), result)
        counts = None
    else:
        flags = correction_flags_from_grouping(grouping)
        n = len(run.histograms)
        counts = next(
            iter(
                corrected_period_counts(
                    run.histograms,
                    grouping,
                    "red",
                    binning=BinningSpec(),
                    forward_idx=effective_group_indices(grouping, forward, n_histograms=n),
                    backward_idx=effective_group_indices(grouping, backward, n_histograms=n),
                    use_deadtime=flags.use_deadtime,
                    deadtime_mode=flags.deadtime_mode,
                    use_background=flags.use_background,
                    metadata=reference_observation_metadata(run),
                    cancel_check=check_cancel,
                ).values()
            )
        )
        if options.method is None:
            result = AlphaEstimate(
                estimate_alpha(
                    counts.forward.counts,
                    counts.backward.counts,
                    first_good_bin=counts.window.first,
                    last_good_bin=counts.window.last,
                ),
                None,
                "per_run_estimate",
                counts.window.last - counts.window.first + 1,
                None,
                True,
            )
        else:
            result = estimate_alpha_detailed(
                counts.forward,
                counts.backward,
                method=options.method,
                time_us=native_count_time(
                    counts.forward.counts.size,
                    counts.bin_width_us,
                    counts.common_t0_bin,
                    counts.t0_time_us,
                ),
                first_good_bin=counts.window.first,
                last_good_bin=counts.window.last,
            )
    check_cancel()
    return CalibrationOutcome(
        result,
        source_run_of(run),
        forward,
        backward,
        run.metadata.get("period_number"),
        options.method is None,
        counts,
    )


def calibration_source(
    loaded, settings: ReductionSettings, options: CalibrationOptions
) -> MuonDataset:
    """Resolve a source period separately from target reduction and cached views."""
    tf = options.method == TF_CALIBRATION_METHOD
    period = options.period if options.period is not None else settings.period
    if tf and (period == GREEN_MINUS_RED or (period is None and period_count(loaded) > 1)):
        raise ValueError("TF calibration requires an explicit acquired source period")
    if period == GREEN_MINUS_RED:
        period = "red"
    if not tf and options.period is None and period_count(loaded) == 1:
        period = None
    return (
        select_period(loaded, period)
        if period is not None
        else (loaded[0] if isinstance(loaded, list) else loaded)
    )
