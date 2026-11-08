"""Background dependencies and availability for grouped count domains."""

from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    grouped_background_ranges,
    resolve_background_mode,
    resolve_facility,
    tail_background_window,
)
from asymmetry.core.transform.corrections import (
    CorrectionOutcome,
    CorrectionOutcomes,
    result_tail_skip,
)
from asymmetry.core.transform.fb_counts import BinWindow
from asymmetry.core.transform.grouping import resolve_alignment
from asymmetry.core.transform.t0 import effective_detector_t0_bins


def count_background_domain(histograms, grouping, group_specs, metadata):
    """Retain observations consumed by the existing count-background availability check."""
    groups = (
        [indices for _, indices in group_specs]
        if resolve_background_mode(grouping) == "tail_fit"
        else [group_specs[0][1], group_specs[-1][1]]
    )
    crop = resolve_alignment(
        histograms, *groups, detector_t0_bins=effective_detector_t0_bins(histograms, grouping)
    )
    mode = resolve_background_mode(grouping)
    windows = [[] for _ in groups]
    if mode == "range":
        ranges = grouped_background_ranges(
            grouping,
            t0_bin=crop.common_t0_bin,
            acquisition_start_bin=crop.first_bin,
            length=crop.length,
            bin_width_us=histograms[groups[0][0]].bin_width,
            facility=resolve_facility(metadata=metadata, grouping=grouping),
        )
        windows = [[window] for window in ranges]
    elif mode == "tail_fit":
        first, last = tail_background_window(
            crop.length,
            crop.common_t0_bin,
            crop.window(grouping).last,
            first_good_bin=crop.window(grouping).first,
        )
        if first <= last:
            windows = [[BinWindow(first, last)] for _ in groups]
    return [(crop, indices, required) for indices, required in zip(groups, windows, strict=True)]


def require_count_background(prepared, grouping, group_specs, metadata):
    """Check availability on corrected count groups without inventing a ratio."""
    if resolve_background_mode(grouping) == "reference_run":
        CorrectionOutcomes(
            background=CorrectionOutcome(
                "background",
                "reference_run",
                "unavailable",
                reason="reference-run subtraction is unsupported for grouped/raw count fitting. "
                "Disable background correction or choose a non-reference background mode "
                "for count fitting; use an asymmetry fit for reference-subtracted data.",
            )
        ).require_complete()
    histograms = prepared.histograms
    groups = (
        [indices for _, indices in group_specs]
        if resolve_background_mode(grouping) == "tail_fit"
        else [group_specs[0][1], group_specs[-1][1]]
    )
    crop = resolve_alignment(
        histograms, *groups, detector_t0_bins=effective_detector_t0_bins(histograms, grouping)
    )
    pairs = [("pair", groups[0], groups[-1])]
    if resolve_background_mode(grouping) == "tail_fit":
        pairs = [(str(gid), indices, indices) for gid, indices in group_specs]
    outcomes = {}
    for label, forward, backward in pairs:
        background = apply_grouped_background_correction(
            prepared.aligned(forward, crop),
            prepared.aligned(backward, crop),
            grouping=grouping,
            t0_bin=crop.common_t0_bin,
            acquisition_start_bin=crop.first_bin,
            bin_width_us=histograms[groups[0][0]].bin_width,
            first_good_bin=crop.window(grouping).first,
            last_good_bin=crop.window(grouping).last,
            metadata=metadata,
        )
        CorrectionOutcomes(background=background.outcome).require_complete()
        outcomes[label] = background.outcome
    skip = result_tail_skip(outcomes)
    return {label: skip or outcome for label, outcome in outcomes.items()}
