"""Group detector counts and apply requested corrections before forming asymmetry.

Final analysis requires complete correction outcomes. A preview may explicitly
retain an incomplete result, carrying its unavailable or failed status.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    resolve_background_mode,
)
from asymmetry.core.transform.corrections import (
    BackgroundReference,
    CorrectionOutcome,
    CorrectionOutcomes,
    CorrectionStatus,
    result_tail_skip,
)
from asymmetry.core.transform.deadtime import (
    deadtime_operation_inputs,
    normalise_deadtime_mode,
)
from asymmetry.core.transform.fb_counts import (
    AlignmentCrop,
    BinWindow,
    ForwardBackwardCounts,
    GroupCounts,
    native_count_time,
    native_time_window,
)
from asymmetry.core.transform.fb_deadtime import deadtime_corrected_pair, fb_deadtime_domain
from asymmetry.core.transform.grouping import (
    independent_pair_indices,
    resolve_alignment,
    resolve_forward_backward_layout,
)
from asymmetry.core.transform.rebin import BinningSpec, binned_fb_asymmetry
from asymmetry.core.transform.t0 import effective_detector_t0_bins, resolve_selected_timing
from asymmetry.core.transform.units import (
    ASYMMETRY_PERCENT,
    PERCENT_PER_FRACTION,
    AsymmetryUnit,
)
from asymmetry.core.utils.perf import perf_timer

#: Resolve reference identity, histograms, and the exposure used for subtraction.
ReferenceResolver = Callable[[dict[str, Any]], BackgroundReference | None]


@dataclass(frozen=True)
class GroupedAsymmetryReduction:
    """Final percent asymmetry, its µs axis, and requested/effective corrections."""

    time: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    corrections: CorrectionOutcomes
    binning: BinningSpec
    alignment: AlignmentCrop
    #: Scale of :attr:`asymmetry` and :attr:`error`. Always
    #: :data:`~asymmetry.core.transform.units.ASYMMETRY_PERCENT` — the reduction
    #: has no other mode; the field exists so a consumer can assert the scale it
    #: assumed instead of trusting a comment, and so a future fraction-scale
    #: reduction could not arrive silently.
    units: AsymmetryUnit = ASYMMETRY_PERCENT


@dataclass(frozen=True)
class CorrectionFlags:
    """The three correction switches a grouping payload implies.

    ``deadtime_mode`` uses the core's canonical request names, including
    ``from_file`` and the authored ``manual`` aliases, for :func:`corrected_grouped_counts` /
    :func:`reduce_grouped_asymmetry`.
    """

    use_deadtime: bool
    deadtime_mode: str
    use_background: bool


def correction_flags_from_grouping(grouping: dict[str, Any] | None) -> CorrectionFlags:
    """Derive the deadtime/background correction switches from a grouping payload.

    This is the single source of truth for how a grouping dict maps to the
    ``use_deadtime`` / ``deadtime_mode`` / ``use_background`` values the reduction
    and the alpha estimate both consume, so a preview, a reduction and an alpha
    estimate built from the same grouping agree on which corrections are active.
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    use_deadtime = bool(grouping.get("deadtime_correction", False))
    deadtime_mode = (
        str(
            grouping.get(
                "deadtime_mode", grouping.get("deadtime_method", "file" if use_deadtime else "off")
            )
        )
        .strip()
        .lower()
        or "off"
    )
    deadtime_mode = normalise_deadtime_mode(deadtime_mode)
    background_mode = resolve_background_mode(grouping)
    use_background = bool(
        grouping.get("background_correction", False) and background_mode != "none"
    )
    return CorrectionFlags(use_deadtime, deadtime_mode, use_background)


@dataclass(frozen=True)
class GroupedCorrectionPreparation:
    """Retain deadtime-corrected counts until a result-wide background decision."""

    corrected: ForwardBackwardCounts
    forward: GroupCounts
    backward: GroupCounts

    def finish(self, skip: CorrectionOutcome | None) -> ForwardBackwardCounts:
        if skip is None:
            return self.corrected
        return replace(
            self.corrected,
            forward=self.forward,
            backward=self.backward,
            corrections=replace(self.corrected.corrections, background=skip),
        )


def corrected_grouped_counts(
    *,
    histograms: list[Histogram],
    grouping: dict,
    forward_idx: list[int],
    backward_idx: list[int],
    use_deadtime: bool,
    deadtime_mode: str,
    use_background: bool,
    facility: str | None = None,
    metadata: dict[str, Any] | None = None,
    reference_resolver: ReferenceResolver | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    output_window: BinWindow | None = None,
    output_binning: BinningSpec | None = None,
    output_time_range: tuple[float | None, float | None] | None = None,
) -> ForwardBackwardCounts:
    """Return corrected counts with the canonical requested/effective outcomes."""
    prepared = prepare_grouped_corrections(
        histograms=histograms,
        grouping=grouping,
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        use_deadtime=use_deadtime,
        deadtime_mode=deadtime_mode,
        use_background=use_background,
        facility=facility,
        metadata=metadata,
        reference_resolver=reference_resolver,
        allow_incomplete=allow_incomplete,
        context=context,
        output_window=output_window,
        output_binning=output_binning,
        output_time_range=output_time_range,
    )
    outcome = prepared.corrected.corrections.background
    if outcome.warning is not None:
        warnings.warn(outcome.warning, UserWarning, stacklevel=2)
    return prepared.corrected


def prepare_grouped_corrections(
    *,
    histograms: list[Histogram],
    grouping: dict,
    forward_idx: list[int],
    backward_idx: list[int],
    use_deadtime: bool,
    deadtime_mode: str,
    use_background: bool,
    facility: str | None = None,
    metadata: dict[str, Any] | None = None,
    reference_resolver: ReferenceResolver | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    output_window: BinWindow | None = None,
    output_binning: BinningSpec | None = None,
    output_time_range: tuple[float | None, float | None] | None = None,
) -> GroupedCorrectionPreparation:
    """Run the reduction's correction stages up to (not including) the asymmetry.

    Applies the detector-first reduction order: deadtime correction
    (per-detector, on raw histograms), forward/backward grouping onto a common
    t0, then optional background subtraction on the grouped counts. The returned
    forward/backward counts are exactly what :func:`reduce_grouped_asymmetry`
    feeds to :func:`binned_fb_asymmetry`, and what the alpha estimate must
    consume so ``alpha`` balances the corrected spectra rather than the raw
    totals (see ``docs/porting/correction-order-alpha-estimation``).

    Only required output samples and active background estimators are
    deadtime-corrected. ``output_window`` selects native bins;
    ``output_binning`` and ``output_time_range`` select consumed merged-bin
    edges and centres without shifting the binning origin. Outside-domain grouped values
    retain raw context for preview plots; they are not corrected observations.
    The outcome records the required detector/raw-bin intervals. No scoped
    detector histograms are exposed for reuse as whole-acquisition preparation.

    ``facility`` guides native-bin period adjustment of automatic range defaults;
    authored intervals remain exact. ``None`` derives the label from ``metadata``
    (pass ``run.metadata``) or the grouping instrument. An empty string disables
    that automatic adjustment; a nonempty string overrides the metadata.
    """
    with perf_timer(
        "core.reduce.corrected_counts",
        n_forward=len(forward_idx),
        n_backward=len(backward_idx),
        deadtime_mode=deadtime_mode,
        background=use_background,
    ) as perf:
        forward_idx, backward_idx = independent_pair_indices(
            grouping,
            n_histograms=len(histograms),
            resolved_indices=(forward_idx, backward_idx),
        )
        resolved = (
            reference_resolver(grouping)
            if use_background
            and resolve_background_mode(grouping) == "reference_run"
            and reference_resolver is not None
            else None
        )
        grouped = resolve_forward_backward_layout(
            histograms, grouping, resolved_indices=(forward_idx, backward_idx)
        )
        crop = grouped.alignment
        common_t0, n_grouped, window = grouped.common_t0_bin, crop.length, grouped.window
        domain = fb_deadtime_domain(
            grouped,
            grouping,
            forward_idx,
            backward_idx,
            use_background=use_background,
            facility=facility,
            metadata=metadata,
            output_window=output_window,
            output_binning=output_binning,
            output_time_range=output_time_range,
        )
        inputs = [(histograms, grouping)]
        domains = [domain]
        if resolved is not None:
            reference_crop = resolve_alignment(
                resolved.histograms,
                forward_idx,
                backward_idx,
                detector_t0_bins=effective_detector_t0_bins(resolved.histograms, grouping),
                common_t0_bin=common_t0 + crop.first_bin,
                first_bin=crop.first_bin,
                sample_length=n_grouped,
            )
            inputs.append((resolved.histograms, resolved.deadtime_grouping(grouping)))
            domains.append(
                fb_deadtime_domain(
                    grouped,
                    grouping,
                    forward_idx,
                    backward_idx,
                    use_background=False,
                    facility=facility,
                    metadata=metadata,
                    alignment=reference_crop,
                    output_window=output_window,
                    output_binning=output_binning,
                    output_time_range=output_time_range,
                )
            )
        inputs = deadtime_operation_inputs(inputs, use_deadtime, deadtime_mode, domains=domains)
        forward, backward, deadtime_outcome = deadtime_corrected_pair(
            *inputs[0],
            forward_idx,
            backward_idx,
            crop,
            domain,
            use_deadtime=use_deadtime,
            mode=deadtime_mode,
        )

        background_outcome = CorrectionOutcome("background", "none", CorrectionStatus.OFF)
        if use_background:
            bin_width = float(histograms[forward_idx[0]].bin_width)
            reference_forward = None
            reference_backward = None
            reference_scale = None
            if resolved is not None:
                reference_scale = resolved.scale
                reference_forward, reference_backward, reference_outcome = deadtime_corrected_pair(
                    *inputs[1],
                    forward_idx,
                    backward_idx,
                    reference_crop,
                    domains[1],
                    use_deadtime=use_deadtime,
                    mode=deadtime_mode,
                )
                if reference_outcome.status in {
                    CorrectionStatus.UNAVAILABLE,
                    CorrectionStatus.FAILED,
                }:
                    deadtime_outcome = replace(
                        reference_outcome, reason=f"reference run: {reference_outcome.reason}"
                    )
            bkg_result = apply_grouped_background_correction(
                forward,
                backward,
                grouping=grouping,
                acquisition_start_bin=crop.first_bin,
                t0_bin=common_t0,
                bin_width_us=bin_width,
                facility=facility,
                metadata=metadata,
                first_good_bin=window.first,
                last_good_bin=window.last,
                reference_forward=reference_forward,
                reference_backward=reference_backward,
                reference_scale=reference_scale,
            )
            forward_counts = bkg_result.forward
            backward_counts = bkg_result.backward
            background_outcome = bkg_result.outcome
            if resolved is not None and background_outcome.status == CorrectionStatus.APPLIED:
                effective = background_outcome.to_dict()["effective"]
                effective["reference_alignment"] = reference_crop.to_dict()
                effective["reference"] = {
                    "run_number": resolved.run_number,
                    "file_name": resolved.source_file,
                    "scale": resolved.scale,
                    "scale_source": resolved.scale_source,
                    "good_frames_sample": resolved.good_frames_sample,
                    "good_frames_reference": resolved.good_frames_reference,
                }
                if use_deadtime:
                    effective["reference"]["deadtime"] = reference_outcome.to_dict()
                background_outcome = replace(background_outcome, effective=effective)

        bin_width = float(histograms[forward_idx[0]].bin_width)
        t0_time_us = grouped.t0_time_us
        if not use_background:
            forward_counts = forward
            backward_counts = backward

        corrections = CorrectionOutcomes(deadtime_outcome, background_outcome)
        if not allow_incomplete:
            corrections.require_complete(context=context)
        perf.detail(
            bins=n_grouped,
            deadtime_applied=deadtime_outcome.status == CorrectionStatus.APPLIED,
            background_applied=background_outcome.status == CorrectionStatus.APPLIED,
        )
        corrected = ForwardBackwardCounts(
            forward=forward_counts,
            backward=backward_counts,
            alignment=crop,
            window=window if output_window is None else output_window,
            bin_width_us=bin_width,
            t0_time_us=t0_time_us,
            corrections=corrections,
        )
        return GroupedCorrectionPreparation(corrected, forward, backward)


def reduce_grouped_asymmetry(
    *,
    histograms: list[Histogram],
    grouping: dict,
    binning: BinningSpec,
    forward_idx: list[int],
    backward_idx: list[int],
    alpha: float,
    use_deadtime: bool,
    deadtime_mode: str,
    use_background: bool,
    facility: str | None = None,
    metadata: dict[str, Any] | None = None,
    reference_resolver: ReferenceResolver | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    beta: float = 1.0,
    output_time_range: tuple[float | None, float | None] | None = None,
) -> GroupedAsymmetryReduction:
    """Reduce grouped histograms to a forward/backward asymmetry curve.

    The correction stages (deadtime → grouping → background) are shared with the
    alpha estimate via :func:`corrected_grouped_counts`; this function adds only
    the final :func:`binned_fb_asymmetry` step. The returned arrays are final —
    the good window is applied, the binning (fixed bunching included) is applied,
    and no further slicing or bunching is left for the caller.

    ``facility``/``metadata`` behave exactly as in
    :func:`corrected_grouped_counts`: leave ``facility`` at ``None`` and pass
    ``metadata=run.metadata`` to have the accelerator-period trimming of a
    pre-t0 background window engage by itself.

    Returns
    -------
    GroupedAsymmetryReduction
        Whose ``asymmetry`` and ``error`` are **in percent (0–100)** — ``units``
        is :data:`~asymmetry.core.transform.units.ASYMMETRY_PERCENT`. This is the
        one step on the reduction path that applies the ×100, so its output drops
        straight into a :class:`~asymmetry.core.data.dataset.MuonDataset` and the
        percent-scale fit models, unlike the **fraction** returned by
        :func:`binned_fb_asymmetry` and
        :func:`~asymmetry.core.transform.compute_asymmetry` underneath it. See
        "Asymmetry units across the API" in the documentation.
    """
    with perf_timer(
        "core.reduce.grouped_asymmetry",
        n_forward=len(forward_idx),
        n_backward=len(backward_idx),
        deadtime_mode=deadtime_mode,
        background=use_background,
    ) as perf:
        corrected = corrected_grouped_counts(
            histograms=histograms,
            grouping=grouping,
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            use_deadtime=use_deadtime,
            deadtime_mode=deadtime_mode,
            use_background=use_background,
            facility=facility,
            metadata=metadata,
            reference_resolver=reference_resolver,
            allow_incomplete=allow_incomplete,
            context=context,
            output_binning=binning,
            output_time_range=output_time_range,
        )
        result = grouped_asymmetry_from_counts(corrected, binning, alpha, beta, output_time_range)
        perf.detail(bins=len(result.time))
        return result


def grouped_asymmetry_from_counts(counts, binning, alpha, beta, time_range):
    """Form the ratio once, converting fraction to public percent units."""
    time, asymmetry, error = binned_fb_asymmetry(
        counts, binning=binning, alpha=alpha, beta=beta, time_range=time_range
    )
    return GroupedAsymmetryReduction(
        time,
        asymmetry * PERCENT_PER_FRACTION,
        error * PERCENT_PER_FRACTION,
        counts.corrections,
        binning,
        counts.alignment,
    )


@dataclass(frozen=True)
class PeriodAsymmetryReduction:
    """Percent asymmetry after binning each period; keep every correction outcome."""

    time: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    corrections: dict[str, CorrectionOutcomes]
    binning: BinningSpec
    alignments: dict[str, AlignmentCrop]

    def metadata_with_alignment(self, metadata: dict) -> dict:
        """Replace inherited alignment with the actual selected period resolutions."""
        result = {
            key: value
            for key, value in metadata.items()
            if key not in {"grouped_alignment", "period_alignments", "correction_error"}
        }
        if len(self.alignments) == 1:
            result["grouped_alignment"] = next(iter(self.alignments.values())).to_dict()
        else:
            result["period_alignments"] = {
                label: value.to_dict() for label, value in self.alignments.items()
            }
        result["correction_outcomes"] = {
            label: item.to_dict() for label, item in self.corrections.items()
        }
        return result


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
    reference_resolver: ReferenceResolver | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    timing_search: Callable | None = None,
    native_time_range: tuple[float | None, float | None] | None = None,
) -> dict[str, ForwardBackwardCounts]:
    """Coordinate F/B and contributing-period corrections before any statistic."""
    from asymmetry.core.io.periods import select_period_histograms
    from asymmetry.core.utils.constants import PeriodMode

    mode = PeriodMode(period_mode)
    combined = mode in {PeriodMode.GREEN_MINUS_RED, PeriodMode.GREEN_PLUS_RED}
    labels = (PeriodMode.RED, PeriodMode.GREEN) if combined else (mode,)
    periods = grouping.get("period_histograms")
    if periods is not None and (not isinstance(periods, list) or len(periods) != 2):
        raise ValueError("Period asymmetry reduction requires exactly two held periods.")
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
    for label, (selected, effective) in selections.items():
        if (
            periods is not None
            and "t0_policy" in effective
            and "effective_t0_sources" not in effective
        ):
            policy = effective["t0_policy"]
            resolve_selected_timing(
                selected,
                effective,
                metadata,
                mode=policy["mode"],
                offset_bins=policy.get("offset_bins", 0),
                search_provider=partial(timing_search, period_index=int(label == PeriodMode.GREEN))
                if timing_search is not None
                else None,
            )
    output_windows = {}
    if native_time_range is not None:
        for label, (selected, effective) in selections.items():
            layout = resolve_forward_backward_layout(
                selected, effective, resolved_indices=(forward_idx, backward_idx)
            )
            time = native_count_time(
                layout.alignment.length,
                layout.bin_width_us,
                layout.common_t0_bin,
                layout.t0_time_us,
            )
            output_windows[label] = native_time_window(time, layout.window, native_time_range)
    if combined:
        inputs, domains, layouts = [], [], []
        for label, (selected, effective) in selections.items():
            pair = independent_pair_indices(
                effective, n_histograms=len(selected), resolved_indices=(forward_idx, backward_idx)
            )
            grouped = resolve_forward_backward_layout(selected, effective, resolved_indices=pair)
            layouts.append((grouped, pair))
            inputs.append((selected, effective))
            domains.append(
                fb_deadtime_domain(
                    grouped,
                    effective,
                    *pair,
                    use_background=use_background,
                    facility=facility,
                    metadata=metadata,
                    output_window=output_windows.get(label),
                    output_binning=binning,
                    output_time_range=output_time_range,
                )
            )
        if (
            use_background
            and resolve_background_mode(grouping) == "reference_run"
            and reference_resolver is not None
        ):
            for (label, (_, effective)), (grouped, pair) in zip(
                selections.items(), layouts, strict=True
            ):
                reference = reference_resolver(effective)
                if reference is not None:
                    crop = grouped.alignment
                    reference_crop = resolve_alignment(
                        reference.histograms,
                        *pair,
                        detector_t0_bins=effective_detector_t0_bins(
                            reference.histograms, effective
                        ),
                        common_t0_bin=grouped.common_t0_bin + crop.first_bin,
                        first_bin=crop.first_bin,
                        sample_length=crop.length,
                    )
                    inputs.append((reference.histograms, reference.deadtime_grouping(effective)))
                    domains.append(
                        fb_deadtime_domain(
                            grouped,
                            effective,
                            *pair,
                            use_background=False,
                            facility=facility,
                            metadata=metadata,
                            output_binning=binning,
                            output_time_range=output_time_range,
                            alignment=reference_crop,
                            output_window=output_windows.get(label),
                        )
                    )
        prepared_inputs = deadtime_operation_inputs(
            inputs, use_deadtime, deadtime_mode, domains=domains
        )
        selections = dict(zip(labels, prepared_inputs[: len(labels)], strict=True))
    prepared = {}
    for label, (selected, effective) in selections.items():
        prepared[label] = prepare_grouped_corrections(
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
            metadata=metadata,
            reference_resolver=reference_resolver,
            allow_incomplete=allow_incomplete,
            context=f"{context} {label}".strip() if combined else context,
        )
    skip = result_tail_skip(
        {str(label): item.corrected.corrections.background for label, item in prepared.items()}
    )
    if skip is not None:
        warnings.warn(skip.warning, UserWarning, stacklevel=2)
    return {str(label): item.finish(skip) for label, item in prepared.items()}


def reduce_period_asymmetry(
    histograms: list[Histogram],
    grouping: dict,
    period_mode: str,
    *,
    binning: BinningSpec,
    forward_idx: list[int],
    backward_idx: list[int],
    alpha: float,
    use_deadtime: bool,
    deadtime_mode: str,
    use_background: bool,
    beta: float = 1.0,
    output_time_range: tuple[float | None, float | None] | None = None,
    facility: str | None = None,
    metadata: dict[str, Any] | None = None,
    reference_resolver: ReferenceResolver | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    timing_search: Callable | None = None,
) -> PeriodAsymmetryReduction:
    """Reduce identical raw-bin edges per period before combining G +/- R.

    Period selection supplies each period's own exposure and deadtime table.
    A sum or difference combines asymmetries, with independent variances added;
    it never sums or averages already reduced bins to implement bunching.
    """
    from asymmetry.core.io.periods import combine_period_asymmetry
    from asymmetry.core.utils.constants import PeriodMode

    mode = PeriodMode(period_mode)
    combined = mode in {PeriodMode.GREEN_MINUS_RED, PeriodMode.GREEN_PLUS_RED}
    corrected = corrected_period_counts(
        histograms,
        grouping,
        period_mode,
        binning=binning,
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        use_deadtime=use_deadtime,
        deadtime_mode=deadtime_mode,
        use_background=use_background,
        output_time_range=output_time_range,
        facility=facility,
        metadata=metadata,
        reference_resolver=reference_resolver,
        allow_incomplete=allow_incomplete,
        context=context,
        timing_search=timing_search,
    )
    reduced = {
        label: grouped_asymmetry_from_counts(counts, binning, alpha, beta, output_time_range)
        for label, counts in corrected.items()
    }
    if combined:
        red, green = reduced[PeriodMode.RED], reduced[PeriodMode.GREEN]
        time, asymmetry, error = combine_period_asymmetry(
            red.time,
            red.asymmetry,
            red.error,
            green.time,
            green.asymmetry,
            green.error,
            mode,
        )
    else:
        single = reduced[mode]
        time, asymmetry, error = single.time, single.asymmetry, single.error
    return PeriodAsymmetryReduction(
        time,
        asymmetry,
        error,
        {str(label): result.corrections for label, result in reduced.items()},
        binning,
        {str(label): result.alignment for label, result in reduced.items()},
    )


__all__ = [
    "CorrectionFlags",
    "GroupedAsymmetryReduction",
    "ReferenceResolver",
    "corrected_grouped_counts",
    "correction_flags_from_grouping",
    "reduce_grouped_asymmetry",
    "reduce_period_asymmetry",
    "PeriodAsymmetryReduction",
]
