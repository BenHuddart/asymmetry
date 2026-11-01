"""Group detector counts and apply requested corrections before forming asymmetry.

Final analysis requires complete correction outcomes. A preview may explicitly
retain an incomplete result, carrying its unavailable or failed status.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
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
    prepare_histograms_with_deadtime,
)
from asymmetry.core.transform.fb_counts import ForwardBackwardCounts, GroupCounts, good_window
from asymmetry.core.transform.grouping import (
    align_forward_backward,
)
from asymmetry.core.transform.rebin import binned_fb_asymmetry
from asymmetry.core.transform.t0 import common_t0_time_us, effective_detector_t0_bins
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
        working_histograms, deadtime_outcome = prepare_histograms_with_deadtime(
            histograms,
            grouping,
            use_deadtime,
            mode=deadtime_mode,
            allow_incomplete=True,
        )

        # A *manual* T0Policy carries effective per-detector t0 bins in the grouping
        # so alignment shifts without the histograms' own t0_bin being rewritten.
        detector_t0 = effective_detector_t0_bins(working_histograms, grouping)
        forward, backward, crop = align_forward_backward(
            working_histograms, forward_idx, backward_idx, detector_t0
        )
        common_t0 = crop.common_t0_bin
        n_grouped = crop.length
        window = good_window(grouping, n_grouped)

        background_outcome = CorrectionOutcome("background", "none", CorrectionStatus.OFF)
        if use_background:
            bin_width = float(working_histograms[0].bin_width) if working_histograms else 1.0
            reference_forward = None
            reference_backward = None
            reference_scale = None
            resolved = None
            if (
                resolve_background_mode(grouping) == "reference_run"
                and reference_resolver is not None
            ):
                resolved = reference_resolver(grouping)
                if resolved is not None:
                    reference_scale = resolved.scale
                    reference_grouping = dict(grouping, good_frames=resolved.good_frames_reference)
                    reference_prepared, reference_outcome = prepare_histograms_with_deadtime(
                        resolved.histograms,
                        reference_grouping,
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
                    reference_forward, reference_backward, reference_crop = align_forward_backward(
                        reference_prepared,
                        forward_idx,
                        backward_idx,
                        effective_detector_t0_bins(reference_prepared, grouping),
                        common_t0_bin=common_t0,
                        sample_length=n_grouped,
                    )
            bkg_result = apply_grouped_background_correction(
                forward,
                backward,
                grouping=grouping,
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
                effective["reference_alignment"] = asdict(reference_crop)
                effective["reference"] = {
                    "run_number": resolved.run_number,
                    "file_name": resolved.source_file,
                    "scale": resolved.scale,
                    "scale_source": resolved.scale_source,
                    "good_frames_sample": resolved.good_frames_sample,
                    "good_frames_reference": resolved.good_frames_reference,
                }
                background_outcome = replace(background_outcome, effective=effective)

        bin_width = float(working_histograms[0].bin_width) if working_histograms else 1.0
        t0_time_us = common_t0_time_us(working_histograms, grouping, common_t0)
        if not use_background:
            forward_counts = GroupCounts(forward)
            backward_counts = GroupCounts(backward)

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
            grouping=grouping,
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
            units=ASYMMETRY_PERCENT,
        )


__all__ = [
    "CorrectionFlags",
    "GroupedAsymmetryReduction",
    "ReferenceResolver",
    "corrected_grouped_counts",
    "correction_flags_from_grouping",
    "reduce_grouped_asymmetry",
]
