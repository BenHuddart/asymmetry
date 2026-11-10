"""Request-scoped detector correction into F/B counts, never prepared acquisitions."""

from __future__ import annotations

from dataclasses import replace

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.background import (
    grouped_background_ranges,
    resolve_background_mode,
    resolve_facility,
    tail_background_window,
)
from asymmetry.core.transform.corrections import CorrectionOutcome
from asymmetry.core.transform.deadtime import (
    DeadtimeDomain,
)
from asymmetry.core.transform.fb_counts import (
    AlignmentCrop,
    BinWindow,
    GroupCounts,
)
from asymmetry.core.transform.grouping import ForwardBackwardLayout
from asymmetry.core.transform.rebin import BinningSpec, fb_binning_edges
from asymmetry.core.transform.scoped_counts import ScopedCounts


def fb_deadtime_domain(
    counts: ForwardBackwardLayout,
    grouping: dict,
    forward_idx: tuple[int, ...],
    backward_idx: tuple[int, ...],
    *,
    use_background: bool,
    facility: str | None,
    metadata: dict | None,
    alignment: AlignmentCrop | None = None,
    output_window: BinWindow | None = None,
    output_binning: BinningSpec | None = None,
    output_time_range: tuple[float | None, float | None] | None = None,
) -> DeadtimeDomain:
    """Map the exact output/active-estimator union through the existing alignment.

    Local r = raw k + offset_i, so raw k = r - offset_i. A reference uses
    the sample's output window with its own resolved detector offsets.
    """
    crop = counts.alignment if alignment is None else alignment
    output = counts.window if output_window is None else output_window
    output.validate_length(crop.length)
    windows = [[output], [output]]
    if output_binning is not None:
        edges, _ = fb_binning_edges(
            replace(counts, window=output), output_binning, time_range=output_time_range
        )
        windows = (
            [[], []]
            if edges.size < 2
            else [[BinWindow(int(edges[0]), int(edges[-1]) - 1)] for _ in range(2)]
        )
    if use_background:
        mode = resolve_background_mode(grouping)
        if mode == "range":
            ranges = grouped_background_ranges(
                grouping,
                t0_bin=counts.common_t0_bin,
                acquisition_start_bin=counts.alignment.first_bin,
                length=counts.alignment.length,
                bin_width_us=counts.bin_width_us,
                facility=resolve_facility(metadata=metadata, grouping=grouping)
                if facility is None
                else facility,
            )
            for required, window in zip(windows, ranges, strict=True):
                required.append(window)
        elif mode == "tail_fit":
            first, last = tail_background_window(
                counts.alignment.length,
                counts.common_t0_bin,
                counts.window.last,
                first_good_bin=counts.window.first,
            )
            if first <= last:
                for required in windows:
                    required.append(BinWindow(first, last))
    observations = []
    offsets = dict(crop.detector_offsets)
    for indices, required in zip((forward_idx, backward_idx), windows, strict=True):
        merged = []
        for window in sorted(required, key=lambda window: window.first):
            if merged and window.first <= merged[-1].last + 1:
                merged[-1] = BinWindow(merged[-1].first, max(merged[-1].last, window.last))
            else:
                merged.append(window)
        for index in indices if merged else ():
            observations.append(
                (
                    index,
                    tuple(
                        BinWindow(window.first - offsets[index], window.last - offsets[index])
                        for window in merged
                    ),
                )
            )
    return DeadtimeDomain(tuple(sorted(observations)))


def deadtime_corrected_pair(
    histograms: list[Histogram],
    grouping: dict,
    forward_idx: tuple[int, ...],
    backward_idx: tuple[int, ...],
    alignment: AlignmentCrop,
    domain: DeadtimeDomain,
    *,
    use_deadtime: bool,
    mode: str,
    retain_variance: bool = False,
) -> tuple[GroupCounts, GroupCounts, CorrectionOutcome]:
    """Correct required raw samples before summing; retain acquired noise.

    The temporary arrays outside the declared domain retain raw values for
    context plots. They are never returned as corrected detector histograms.
    Only the F/B window and active estimators consume corrected samples.
    """
    prepared = ScopedCounts.prepare(
        histograms, grouping, domain, use_deadtime=use_deadtime, mode=mode, allow_incomplete=True
    )
    return (
        prepared.aligned(list(forward_idx), alignment, retain_variance=retain_variance),
        prepared.aligned(list(backward_idx), alignment, retain_variance=retain_variance),
        prepared.outcome,
    )
