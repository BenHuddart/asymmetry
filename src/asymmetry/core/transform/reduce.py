"""Group detector counts and apply requested corrections before forming asymmetry.

Final analysis requires complete correction outcomes. A preview may explicitly
retain an incomplete result, carrying its unavailable or failed status.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.data.ratio_validity import RatioValidity
from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    resolve_background_mode,
)
from asymmetry.core.transform.constituent_counts import (
    ConstituentCounts,
    resolve_constituent_counts,
)
from asymmetry.core.transform.corrections import (
    BackgroundReference,
    CorrectionOutcome,
    CorrectionOutcomes,
    CorrectionStatus,
)
from asymmetry.core.transform.deadtime import (
    DeadtimeDomain,
    deadtime_operation_inputs,
    normalise_deadtime_mode,
)
from asymmetry.core.transform.fb_counts import (
    AlignmentCrop,
    BinWindow,
    ForwardBackwardCounts,
    GroupCounts,
)
from asymmetry.core.transform.fb_deadtime import deadtime_corrected_pair, fb_deadtime_domain
from asymmetry.core.transform.grouping import (
    ForwardBackwardLayout,
    independent_pair_indices,
)
from asymmetry.core.transform.period_reduction import corrected_period_counts
from asymmetry.core.transform.rebin import BinningSpec, binned_fb_observation
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
    ratio_validity: RatioValidity
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
    cancel_check: Callable[[], None] | None = None,
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
        cancel_check=cancel_check,
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


@dataclass(frozen=True)
class GroupedCorrectionPlan:
    """Required acquired observations and reference on the requested set's grid."""

    observation: ConstituentCounts
    grouped: ForwardBackwardLayout
    reference: BackgroundReference | None
    reference_crop: AlignmentCrop | None
    inputs: list[tuple[list[Histogram], dict]]
    domains: list[DeadtimeDomain]


def grouped_correction_plan(
    histograms,
    grouping,
    pair,
    *,
    use_background,
    facility,
    metadata,
    reference_resolver,
    cancel_check,
    output_window,
    output_binning,
    output_time_range,
    observation=None,
    layout=None,
):
    """Resolve membership/support before any correction availability or saturation work."""
    if observation is None:
        observation = resolve_constituent_counts(
            histograms, grouping, pair, metadata, cancel_check=cancel_check
        )
    grouped = observation.layout if layout is None else layout
    domain = fb_deadtime_domain(
        grouped,
        grouping,
        *observation.pair,
        use_background=use_background,
        facility=facility,
        metadata=metadata,
        output_window=output_window,
        output_binning=output_binning,
        output_time_range=output_time_range,
    )
    inputs, domains = observation.participants(domain)
    reference = (
        reference_resolver(grouping)
        if use_background
        and resolve_background_mode(grouping) == "reference_run"
        and reference_resolver is not None
        else None
    )
    reference_crop = None
    if reference is not None:
        sample = observation.reference_sample
        reference_crop = reference.alignment(
            sample.histograms, sample.grouping, grouped.alignment, *pair, cancel_check=cancel_check
        )
        for member in observation.members:
            reference.require_independent(member.histograms, member.metadata, list(sum(pair, ())))
        inputs.append((reference.histograms, reference.deadtime_grouping(grouping)))
        domains.append(
            fb_deadtime_domain(
                grouped,
                grouping,
                *pair,
                use_background=False,
                facility=facility,
                metadata=metadata,
                alignment=reference_crop,
                output_window=output_window,
                output_binning=output_binning,
                output_time_range=output_time_range,
            )
        )
    return GroupedCorrectionPlan(observation, grouped, reference, reference_crop, inputs, domains)


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
    cancel_check: Callable[[], None] | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    output_window: BinWindow | None = None,
    output_binning: BinningSpec | None = None,
    output_time_range: tuple[float | None, float | None] | None = None,
    plan: GroupedCorrectionPlan | None = None,
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
        if plan is None:
            plan = grouped_correction_plan(
                histograms,
                grouping,
                (forward_idx, backward_idx),
                use_background=use_background,
                facility=facility,
                metadata=metadata,
                reference_resolver=reference_resolver,
                cancel_check=cancel_check,
                output_window=output_window,
                output_binning=output_binning,
                output_time_range=output_time_range,
            )
        resolved, grouped = plan.reference, plan.grouped
        reference_crop = plan.reference_crop
        crop = grouped.alignment
        common_t0, n_grouped, window = grouped.common_t0_bin, crop.length, grouped.window
        inputs = deadtime_operation_inputs(
            plan.inputs, use_deadtime, deadtime_mode, domains=plan.domains
        )
        n_members = len(plan.observation.members)
        forward, backward, deadtime_outcome = plan.observation.corrected_pair(
            inputs[:n_members],
            plan.domains[:n_members],
            crop,
            use_deadtime=use_deadtime,
            mode=deadtime_mode,
            cancel_check=cancel_check,
        )

        background_outcome = CorrectionOutcome("background", "none", CorrectionStatus.OFF)
        if use_background:
            bin_width = grouped.bin_width_us
            reference_forward = None
            reference_backward = None
            reference_scale = None
            if resolved is not None:
                reference_scale = resolved.scale
                reference_forward, reference_backward, reference_outcome = deadtime_corrected_pair(
                    *inputs[-1],
                    forward_idx,
                    backward_idx,
                    reference_crop,
                    plan.domains[-1],
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
                effective["reference"] = resolved.provenance()
                if use_deadtime:
                    effective["reference"]["deadtime"] = reference_outcome.to_dict()
                background_outcome = replace(background_outcome, effective=effective)

        bin_width = grouped.bin_width_us
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
    cancel_check: Callable[[], None] | None = None,
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
        Whose ``asymmetry`` and ``error`` are **in percent (fraction × 100)** — ``units``
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
            cancel_check=cancel_check,
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
    time, asymmetry, error, validity = binned_fb_observation(
        counts, binning=binning, alpha=alpha, beta=beta, time_range=time_range
    )
    return GroupedAsymmetryReduction(
        time,
        asymmetry * PERCENT_PER_FRACTION,
        error * PERCENT_PER_FRACTION,
        counts.corrections,
        binning,
        counts.alignment,
        validity,
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
    ratio_validity: RatioValidity

    def metadata_with_alignment(self, metadata: dict) -> dict:
        """Replace inherited alignment with the actual selected period resolutions."""
        result = {}
        if len(self.alignments) == 1:
            result["grouped_alignment"] = next(iter(self.alignments.values())).to_dict()
        else:
            result["period_alignments"] = {
                label: value.to_dict() for label, value in self.alignments.items()
            }
        result["correction_outcomes"] = {
            label: item.to_dict() for label, item in self.corrections.items()
        }
        return asymmetry_reduction_metadata(metadata, result)


def asymmetry_reduction_metadata(metadata: dict, reduced: dict) -> dict:
    """Refresh live metadata while retaining independent reduction-owned provenance."""
    owned = {"grouped_alignment", "period_alignments", "correction_outcomes"}
    return {
        key: value
        for key, value in metadata.items()
        if key not in owned and key != "correction_error"
    } | deepcopy({key: value for key, value in reduced.items() if key in owned})


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
    cancel_check: Callable[[], None] | None = None,
    allow_incomplete: bool = False,
    context: str = "",
    timing_search: Callable | None = None,
) -> PeriodAsymmetryReduction:
    """Form pooled G+R once or difference independently reduced sets on common support."""
    from asymmetry.core.io.periods import combine_period_asymmetry
    from asymmetry.core.utils.constants import PeriodMode

    mode = PeriodMode(period_mode)
    combined = mode == PeriodMode.GREEN_MINUS_RED
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
        cancel_check=cancel_check,
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
            red.ratio_validity.mathematical_error(red.error),
            green.time,
            green.asymmetry,
            green.ratio_validity.mathematical_error(green.error),
            mode,
        )
        validity = red.ratio_validity.combine(green.ratio_validity, error)
    else:
        single = reduced[mode]
        time, asymmetry, error = single.time, single.asymmetry, single.error
        validity = single.ratio_validity
    return PeriodAsymmetryReduction(
        time,
        asymmetry,
        error,
        {str(label): result.corrections for label, result in reduced.items()},
        binning,
        {str(label): result.alignment for label, result in reduced.items()},
        validity,
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
