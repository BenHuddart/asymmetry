"""Coordinate acquired count preparation for the requested period statistic."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.constituent_counts import (
    pool_constituents,
    resolve_constituent_counts,
)
from asymmetry.core.transform.corrections import result_tail_skip
from asymmetry.core.transform.deadtime import deadtime_operation_inputs
from asymmetry.core.transform.fb_counts import (
    ForwardBackwardCounts,
    native_count_time,
    native_time_window,
)
from asymmetry.core.transform.grouping import shared_automatic_alpha
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.t0 import resolve_selected_timing

PERIOD_SUM_POLICY = "pooled_prepared_counts_v1"


def period_statistic_policy(grouping):
    """Identify the changed G+R statistic without invalidating unrelated modes."""
    return (
        {"period_sum_policy": PERIOD_SUM_POLICY}
        if grouping.get("period_mode") == "green_plus_red"
        else {}
    )


def corrected_period_counts(
    histograms: list[Histogram],
    grouping: dict,
    period_mode: str,
    *,
    binning: BinningSpec,
    forward_idx: list[int],
    backward_idx: list[int],
    use_deadtime: bool,
    deadtime_mode: str,
    use_background: bool,
    output_time_range: tuple[float | None, float | None] | None = None,
    facility: str | None = None,
    metadata: dict[str, Any] | None = None,
    reference_resolver: Callable | None = None,
    cancel_check: Callable[[], None] | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    timing_search: Callable | None = None,
    native_time_range: tuple[float | None, float | None] | None = None,
) -> dict[str, ForwardBackwardCounts]:
    """Coordinate F/B and contributing-period corrections before any statistic."""
    from asymmetry.core.io.periods import select_period_histograms
    from asymmetry.core.transform.reduce import grouped_correction_plan, prepare_grouped_corrections
    from asymmetry.core.utils.constants import PeriodMode

    mode = PeriodMode(period_mode)
    combined = mode in {PeriodMode.GREEN_MINUS_RED, PeriodMode.GREEN_PLUS_RED}
    labels = (PeriodMode.RED, PeriodMode.GREEN) if combined else (mode,)
    periods = grouping.get("period_histograms")
    if periods is not None and (not isinstance(periods, list) or len(periods) not in {1, 2}):
        raise ValueError("Period asymmetry reduction requires one or two held periods.")
    if periods is not None and len(periods) == 1 and mode != PeriodMode.RED:
        raise ValueError("A green selection or two-set operation requires a real green set.")
    if combined and periods is None:
        raise ValueError("A period sum or difference requires two held periods.")
    if periods is not None and any(
        not isinstance(period, list)
        or not period
        or any(not isinstance(hist, Histogram) for hist in period)
        for period in periods
    ):
        raise ValueError("Held periods must contain nonempty histogram lists.")
    selections = {
        label: select_period_histograms(histograms, grouping, int(label == PeriodMode.GREEN))
        if periods is not None
        else (histograms, grouping)
        for label in labels
    }
    selections = {
        label: (selected, dict(effective, period_mode=str(label)))
        for label, (selected, effective) in selections.items()
    }
    if mode == PeriodMode.GREEN_MINUS_RED and shared_automatic_alpha(grouping, metadata):
        selections[PeriodMode.GREEN][1].pop("alpha_policy", None)
    for label, (selected, effective) in selections.items():
        if (
            periods is not None
            and "period_inputs" not in effective
            and "t0_policy" in effective
            and "effective_t0_sources" not in effective
        ):
            policy = effective["t0_policy"]
            resolve_selected_timing(
                selected,
                effective,
                metadata,
                mode=policy["mode"],
                cancel_check=cancel_check,
                offset_bins=policy.get("offset_bins", 0),
                search_provider=partial(timing_search, period_index=int(label == PeriodMode.GREEN))
                if timing_search is not None
                else None,
            )
    pair = (tuple(forward_idx), tuple(backward_idx))
    observations = {
        label: resolve_constituent_counts(
            selected,
            effective,
            pair,
            dict(metadata or {}, period_number=int(label == PeriodMode.GREEN) + 1)
            if periods is not None
            else metadata,
            cancel_check=cancel_check,
        )
        for label, (selected, effective) in selections.items()
    }
    layouts = {}
    if combined:
        members = [
            replace(
                member, period_number=member.period_number or int(label == PeriodMode.GREEN) + 1
            )
            for label, observation in observations.items()
            for member in observation.members
        ]
        pooled = pool_constituents(members, grouping, pair)
        common = pooled.layout
        if mode == PeriodMode.GREEN_PLUS_RED:
            from asymmetry.core.transform.reference_scale import ReferenceExposure

            facts = [
                ReferenceExposure.from_grouping(effective) for _, effective in selections.values()
            ]
            exposure = ReferenceExposure()
            if all(fact.value is not None and fact.unit == facts[0].unit for fact in facts):
                exposure = ReferenceExposure(
                    sum(fact.value for fact in facts),
                    "acquired_period_sum:" + ",".join(fact.origin for fact in facts),
                    facts[0].unit,
                )
            effective = dict(grouping, reference_exposure=exposure.to_dict())
            selections = {mode: (histograms, effective)}
            observations = {mode: pooled}
            layouts = {mode: common}
        else:
            offset = 0
            for label, observation in observations.items():
                size = len(observation.histograms)
                crop = common.alignment

                def local(values):
                    return tuple((i - offset, v) for i, v in values if offset <= i < offset + size)

                alignment = replace(
                    crop,
                    detector_offsets=local(crop.detector_offsets),
                    timing_residuals_us=local(crop.timing_residuals_us),
                    timing_sources=local(crop.timing_sources),
                    acquired_detectors=(),
                )
                layouts[label] = replace(
                    common, alignment=observation.acquired_alignment(alignment)
                )
                offset += size
    output_windows, plans = {}, {}
    for label, (selected, effective) in selections.items():
        if native_time_range is not None:
            layout = layouts.get(label) or observations[label].layout
            time = native_count_time(
                layout.alignment.length,
                layout.bin_width_us,
                layout.common_t0_bin,
                layout.t0_time_us,
            )
            output_windows[label] = native_time_window(time, layout.window, native_time_range)
        selected_metadata = (
            dict(metadata or {}, period_number=int(label == PeriodMode.GREEN) + 1)
            if periods is not None
            else metadata
        )
        plans[label] = grouped_correction_plan(
            selected,
            effective,
            (tuple(forward_idx), tuple(backward_idx)),
            use_background=use_background,
            facility=facility,
            metadata=selected_metadata,
            reference_resolver=reference_resolver,
            cancel_check=cancel_check,
            observation=observations[label],
            layout=layouts.get(label),
            output_window=output_windows.get(label),
            output_binning=binning,
            output_time_range=output_time_range,
        )
    inputs = [item for plan in plans.values() for item in plan.inputs]
    domains = [item for plan in plans.values() for item in plan.domains]
    coordinated = deadtime_operation_inputs(inputs, use_deadtime, deadtime_mode, domains=domains)
    first = 0
    for label, plan in plans.items():
        plans[label] = replace(plan, inputs=coordinated[first : first + len(plan.inputs)])
        first += len(plan.inputs)
    prepared = {}
    for label, (selected, effective) in selections.items():
        prepared[label] = prepare_grouped_corrections(
            plan=plans[label],
            histograms=selected,
            grouping=effective,
            output_binning=binning,
            output_window=output_windows.get(label),
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            use_deadtime=use_deadtime,
            deadtime_mode=deadtime_mode,
            use_background=use_background,
            output_time_range=output_time_range,
            facility=facility,
            metadata=dict(metadata or {}, period_number=int(label == PeriodMode.GREEN) + 1)
            if periods is not None
            else metadata,
            reference_resolver=reference_resolver,
            cancel_check=cancel_check,
            allow_incomplete=allow_incomplete,
            context=f"{context} {label}".strip() if combined else context,
        )
    skip = result_tail_skip(
        {str(label): item.corrected.corrections.background for label, item in prepared.items()}
    )
    if skip is not None:
        warnings.warn(skip.warning, UserWarning, stacklevel=2)
    return {str(label): item.finish(skip) for label, item in prepared.items()}
