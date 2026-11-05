"""Group detector counts and apply requested corrections before forming asymmetry.

Final analysis requires complete correction outcomes. A preview may explicitly
retain an incomplete result, carrying its unavailable or failed status.
"""

from __future__ import annotations

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
)
from asymmetry.core.transform.deadtime import (
    deadtime_operation_inputs,
    prepare_histograms_with_deadtime,
)
from asymmetry.core.transform.fb_counts import AlignmentCrop, ForwardBackwardCounts
from asymmetry.core.transform.grouping import (
    align_forward_backward_counts,
    group_forward_backward,
    independent_pair_indices,
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

    ``deadtime_mode`` is normalised (lower-cased, ``load`` folded to ``manual``)
    so it can be passed straight to :func:`corrected_grouped_counts` /
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
    if deadtime_mode == "load":
        deadtime_mode = "manual"
    background_mode = resolve_background_mode(grouping)
    use_background = bool(
        grouping.get("background_correction", False) and background_mode != "none"
    )
    return CorrectionFlags(use_deadtime, deadtime_mode, use_background)


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
) -> ForwardBackwardCounts:
    """Run the reduction's correction stages up to (not including) the asymmetry.

    Applies, in the order every reference program uses: deadtime correction
    (per-detector, on raw histograms), forward/backward grouping onto a common
    t0, then optional background subtraction on the grouped counts. The returned
    forward/backward counts are exactly what :func:`reduce_grouped_asymmetry`
    feeds to :func:`binned_fb_asymmetry`, and what the alpha estimate must
    consume so ``alpha`` balances the corrected spectra rather than the raw
    totals (see ``docs/porting/correction-order-alpha-estimation``).

    ``facility`` decides whether a ``range``-mode background window is trimmed to
    whole accelerator periods. The default ``None`` means *derive it* — from
    ``metadata`` when given (pass ``run.metadata``; this function receives
    histograms rather than a run, so it cannot fetch them itself), otherwise from
    the grouping payload's canonical instrument name. Pass ``""`` for the explicit
    "no facility, no trimming" opt-out. It used to default to ``""``, so a script
    that did not know to pass a facility silently reduced with an untrimmed
    window while the GUI, which does pass one, trimmed — the same run reduced two
    ways.
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
        inputs = [(histograms, grouping)]
        if resolved is not None:
            inputs.append(
                (
                    resolved.histograms,
                    resolved.deadtime_grouping(grouping),
                )
            )
        inputs = deadtime_operation_inputs(inputs, use_deadtime, deadtime_mode)
        working_histograms, deadtime_outcome = prepare_histograms_with_deadtime(
            *inputs[0],
            use_deadtime,
            mode=deadtime_mode,
            allow_incomplete=True,
        )

        grouped = group_forward_backward(
            working_histograms, grouping, resolved_indices=(forward_idx, backward_idx)
        ).counts
        forward, backward, crop = grouped.forward, grouped.backward, grouped.alignment
        common_t0, n_grouped, window = grouped.common_t0_bin, crop.length, grouped.window

        background_outcome = CorrectionOutcome("background", "none", CorrectionStatus.OFF)
        if use_background:
            bin_width = float(working_histograms[forward_idx[0]].bin_width)
            reference_forward = None
            reference_backward = None
            reference_scale = None
            if resolved is not None:
                reference_scale = resolved.scale
                reference_prepared, reference_outcome = prepare_histograms_with_deadtime(
                    *inputs[1],
                    use_deadtime,
                    mode=deadtime_mode,
                    allow_incomplete=True,
                )
                if reference_outcome.status in {
                    CorrectionStatus.UNAVAILABLE,
                    CorrectionStatus.FAILED,
                }:
                    deadtime_outcome = replace(
                        reference_outcome, reason=f"reference run: {reference_outcome.reason}"
                    )
                reference_forward, reference_backward, reference_crop = (
                    align_forward_backward_counts(
                        reference_prepared,
                        forward_idx,
                        backward_idx,
                        effective_detector_t0_bins(reference_prepared, grouping),
                        common_t0_bin=common_t0 + crop.first_bin,
                        first_bin=crop.first_bin,
                        sample_length=n_grouped,
                    )
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

        bin_width = float(working_histograms[forward_idx[0]].bin_width)
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
        return ForwardBackwardCounts(
            forward=forward_counts,
            backward=backward_counts,
            alignment=crop,
            window=window,
            bin_width_us=bin_width,
            t0_time_us=t0_time_us,
            corrections=corrections,
        )


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
        )
        time_axis, asymmetry, error = binned_fb_asymmetry(
            corrected,
            binning=binning,
            alpha=alpha,
            beta=beta,
        )
        perf.detail(
            bins=len(time_axis),
            deadtime_applied=corrected.corrections.deadtime.status == CorrectionStatus.APPLIED,
            background_applied=corrected.corrections.background.status == CorrectionStatus.APPLIED,
        )
        # binned_fb_asymmetry returns the fraction; the public contract of this
        # reduction (and of MuonDataset) is percent, so scale here — the one
        # place the ×100 happens on this path.
        return GroupedAsymmetryReduction(
            time=np.asarray(time_axis, dtype=np.float64),
            asymmetry=np.asarray(asymmetry * PERCENT_PER_FRACTION, dtype=np.float64),
            error=np.asarray(error * PERCENT_PER_FRACTION, dtype=np.float64),
            corrections=corrected.corrections,
            binning=binning,
            alignment=corrected.alignment,
            units=ASYMMETRY_PERCENT,
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
            if key not in {"grouped_alignment", "period_alignments"}
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
    from asymmetry.core.io.periods import combine_period_asymmetry, select_period_histograms
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
    if combined:
        inputs = list(selections.values())
        if (
            use_background
            and resolve_background_mode(grouping) == "reference_run"
            and reference_resolver is not None
        ):
            for selected, effective in selections.values():
                reference = reference_resolver(effective)
                if reference is not None:
                    inputs.append(
                        (
                            reference.histograms,
                            reference.deadtime_grouping(effective),
                        )
                    )
        prepared_inputs = deadtime_operation_inputs(inputs, use_deadtime, deadtime_mode)
        selections = dict(zip(labels, prepared_inputs[: len(labels)], strict=True))
    reduced = {}
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
        reduced[label] = reduce_grouped_asymmetry(
            histograms=selected,
            grouping=effective,
            binning=binning,
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            alpha=alpha,
            use_deadtime=use_deadtime,
            deadtime_mode=deadtime_mode,
            use_background=use_background,
            beta=beta,
            facility=facility,
            metadata=metadata,
            reference_resolver=reference_resolver,
            allow_incomplete=allow_incomplete,
            context=f"{context} {label}".strip() if combined else context,
        )
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
