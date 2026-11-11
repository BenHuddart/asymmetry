"""Explicit script calibration delegates preparation and fitting to core owners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import uuid4

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.count_results import CountFitParameter
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.likelihood import FitInputError
from asymmetry.core.fitting.tf_calibration import (
    TF_CALIBRATION_METHOD,
    TFCalibrationRequest,
    TFEnvelope,
    fit_tf_calibration,
    select_tf_calibration,
)
from asymmetry.core.instrument import derive_projection_pairs
from asymmetry.core.io.periods import (
    period_count,
    period_run,
    reference_mapping_requires_replay,
    resolve_period_index,
    select_period,
    source_run_of,
)
from asymmetry.core.io.reference_selection import select_reference_run
from asymmetry.core.project.calibration import CalibrationRecord
from asymmetry.core.transform.asymmetry import (
    ALPHA_ESTIMATION_METHODS,
    AlphaEstimate,
    estimate_alpha,
    estimate_alpha_detailed,
)
from asymmetry.core.transform.corrections import reference_observation_metadata
from asymmetry.core.transform.fb_counts import native_count_time
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.reduce import corrected_period_counts, correction_flags_from_grouping
from asymmetry.core.workflow.calibration_results import CalibrationOutcome, QuickCalibrationResult
from asymmetry.core.workflow.reduction import (
    GREEN_MINUS_RED,
    ReductionSettings,
    resolve_reduction_grouping,
)

CALIBRATION_METHODS = (*ALPHA_ESTIMATION_METHODS, TF_CALIBRATION_METHOD)


def supports_joint_calibration(method: str | None, *, projection_layout: bool = False) -> bool:
    """Joint detector beta is supported by TF calibration on a single F/B pair."""
    return method == TF_CALIBRATION_METHOD and not projection_layout


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
    pulse_separation_us: float | None = None

    def __post_init__(self):
        if self.method is not None and self.method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown calibration method {self.method!r}")
        if not supports_joint_calibration(self.method) and (
            self.fit_beta
            or self.envelope is not None
            or self.frequency_mhz is not None
            or self.parameters
            or self.exclude is not None
            or self.minos
            or self.pulse_separation_us is not None
        ):
            raise ValueError(
                "Envelope, frequency, parameters, pulse separation and free beta require tf_calibration_fit"
            )
        if self.envelope is not None:
            object.__setattr__(self, "envelope", TFEnvelope(self.envelope))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if self.exclude is not None:
            object.__setattr__(self, "exclude", tuple(self.exclude))


def calibrate_run(
    loaded: MuonDataset | list[MuonDataset],
    settings: ReductionSettings,
    options: CalibrationOptions | None = None,
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> CalibrationOutcome:
    """Calibrate source histograms; TF requires one acquired period.

    Explicit TF requests on multi-period acquisitions require a selected acquired
    period. The historical ratio uses red for a difference and preserves its
    single-period source adaptation. Corrections/timing come from the source
    profile; TF owns its own native-window and fitted-background preparation.
    """

    def check_cancel():
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Fit cancelled.")

    check_cancel()
    options = explicit_calibration_options(options)
    dataset = calibration_source(loaded, settings, options)
    run = dataset.run
    if run is None or not run.histograms:
        raise ValueError("Calibration requires source detector histograms")
    source_settings = replace(settings, alpha=1.0, alpha_source="assumed", period=None)
    grouping = resolve_reduction_grouping(run, source_settings)
    prepared = replace(dataset, run=replace(run, grouping=grouping))
    return calibrate_source(
        prepared,
        options,
        t_min=settings.t_min,
        t_max=settings.t_max,
        cancel_callback=cancel_callback,
    )


def explicit_calibration_options(options: CalibrationOptions | None = None) -> CalibrationOptions:
    """Resolve omitted explicit-operation intent without altering automatic policies."""
    return CalibrationOptions(TF_CALIBRATION_METHOD) if options is None else options


def calibrate_source(
    dataset: MuonDataset,
    options: CalibrationOptions | None = None,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    reference_resolver=None,
    cancel_callback: Callable[[], bool] | None = None,
) -> CalibrationOutcome:
    """Calibrate an identified source with its fully resolved grouping and corrections.

    Front ends resolve source selection/profile through their existing owners.
    This boundary never replaces those policies with CLI reduction defaults.
    """

    def check_cancel():
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Fit cancelled.")

    check_cancel()
    options = explicit_calibration_options(options)
    run = dataset.run
    if run is None or not run.histograms:
        raise ValueError("Calibration requires source detector histograms")
    grouping = run.grouping
    projections = derive_projection_pairs(
        grouping.get("groups"), grouping.get("group_names"), grouping.get("projections")
    )
    if options.fit_beta and not supports_joint_calibration(
        options.method, projection_layout=bool(projections)
    ):
        raise ValueError("Joint beta calibration requires a single F/B pair layout")
    forward, backward = int(grouping["forward_group"]), int(grouping["backward_group"])
    prepared = dataset
    tf = options.method == TF_CALIBRATION_METHOD
    if tf:
        request = TFCalibrationRequest(
            forward,
            backward,
            options.fit_beta,
            t_min=t_min,
            t_max=t_max,
            frequency_mhz=options.frequency_mhz,
            parameters=options.parameters,
            exclude=options.exclude,
            minos=options.minos,
            pulse_separation_us=options.pulse_separation_us,
        )
        result = (
            select_tf_calibration(prepared, request, cancel_callback=cancel_callback)
            if options.envelope is None
            else fit_tf_calibration(
                prepared, request, envelope=options.envelope, cancel_callback=cancel_callback
            )
        )
        result = CalibrationRecord(str(uuid4()), result)
    else:
        flags = correction_flags_from_grouping(grouping)
        n = len(run.histograms)
        counts = next(
            iter(
                corrected_period_counts(
                    run.histograms,
                    grouping,
                    grouping.get("period_mode", "red"),
                    binning=BinningSpec(),
                    forward_idx=effective_group_indices(grouping, forward, n_histograms=n),
                    backward_idx=effective_group_indices(grouping, backward, n_histograms=n),
                    use_deadtime=flags.use_deadtime,
                    deadtime_mode=flags.deadtime_mode,
                    use_background=flags.use_background,
                    metadata=reference_observation_metadata(run),
                    cancel_check=check_cancel,
                    reference_resolver=reference_resolver,
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
        mapping = run.grouping.get("period_mapping", run.metadata.get("period_mapping", {}))
        mapped = reference_mapping_requires_replay(mapping)
        result = QuickCalibrationResult(
            result,
            source_run_of(run),
            forward,
            backward,
            None if mapped else run.metadata.get("period_number"),
            run.source_file,
            counts.corrections,
            counts.alignment,
            counts.window,
            counts.bin_width_us,
            counts.t0_time_us,
            tuple((int(number), target) for number, target in mapping.items()),
            mapped_period_number=run.metadata.get("period_number") if mapped else None,
        )
    check_cancel()
    return CalibrationOutcome(result)


def calibration_source(
    loaded: MuonDataset | list[MuonDataset],
    settings: ReductionSettings,
    options: CalibrationOptions,
) -> MuonDataset:
    """Resolve a source period separately from target reduction and cached views.

    Keep the selected cached curve when available; a held-only acquisition has
    empty curve arrays, so a parent's curve cannot acquire another period's label.
    """
    tf = options.method == TF_CALIBRATION_METHOD
    period = options.period if options.period is not None else settings.period
    if period == GREEN_MINUS_RED:
        if tf:
            raise ValueError("TF calibration requires an explicit acquired source period")
        period = "red"
    if not tf and options.period is None and period_count(loaded) == 1:
        period = None
    if isinstance(loaded, list):
        source = loaded[0] if period is None else select_period(loaded, period)
    else:
        source = loaded
    run = source.run
    if (
        tf
        and run is not None
        and reference_mapping_requires_replay(
            run.grouping.get("period_mapping", run.metadata.get("period_mapping"))
        )
    ):
        raise ValueError("TF calibration source is mapped; select an original acquired period")
    if isinstance(loaded, list):
        if tf and period is None and len(loaded) > 1:
            raise ValueError("TF calibration requires an explicit acquired source period")
    elif run is not None and (
        "period_histograms" in run.grouping or "period_reduced" in run.grouping
    ):
        if tf and period is None and period_count(run) > 1:
            raise ValueError("TF calibration requires an explicit acquired source period")
        index = resolve_period_index(1 if period is None else period, period_count(run))
        if len(run.grouping.get("period_reduced") or ()) >= 2:
            source = select_period(source, index + 1)
        else:
            selected = period_run(run, index)
            empty = np.empty(0)
            source = MuonDataset(empty, empty, empty, metadata=selected.metadata, run=selected)
    elif run is not None and "period_number" in run.metadata:
        number = run.metadata["period_number"]
        count = max(2, number, period_count(source))
        requested = number if period is None else resolve_period_index(period, count) + 1
        selected = select_reference_run(run, {"period_number": requested})
        if selected is None:
            raise FitInputError(
                "Calibration needs the requested acquired period histograms; reload the source"
            )
    elif run is not None and period_count(run) > 1:
        raise FitInputError(
            "Calibration needs acquired period histograms for an identified source period"
        )
    elif period is not None:
        source = select_period(source, period)
    if (
        source.run is not None
        and source.run.metadata.get("raw_period_histograms_available") is False
    ):
        raise FitInputError("Calibration needs acquired period histograms; reload the source")
    return source
