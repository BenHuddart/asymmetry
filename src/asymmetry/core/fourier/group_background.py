"""Helpers for WiMDA-style grouped detector Fourier inputs."""

from __future__ import annotations

from asymmetry.core.transform.background import (
    BackgroundCorrectionResult,
    apply_grouped_background_correction,
    background_window,
    fixed_background_values,
    resolve_background_mode,
    resolve_background_ranges,
)
from asymmetry.core.transform.corrections import (
    CorrectionOutcome,
    CorrectionOutcomes,
    CorrectionStatus,
)
from asymmetry.core.transform.fb_counts import GroupCounts


def _group_background_value_for_group(
    grouping: dict,
    group_id: int,
) -> float | None:
    """Return a fixed background value for one group when explicitly configured."""
    raw_values = fixed_background_values(grouping)
    if raw_values is None:
        raw_values = grouping.get("background_values")
    if isinstance(raw_values, dict):
        value = raw_values.get(group_id, raw_values.get(str(group_id)))
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if not isinstance(raw_values, (list, tuple)):
        return None

    try:
        forward_group = int(grouping.get("forward_group", 1))
    except (TypeError, ValueError):
        forward_group = None
    try:
        backward_group = int(grouping.get("backward_group", 2))
    except (TypeError, ValueError):
        backward_group = None

    if len(raw_values) >= 2 and forward_group is not None and backward_group is not None:
        try:
            if int(group_id) == forward_group:
                return float(raw_values[0])
            if int(group_id) == backward_group:
                return float(raw_values[1])
        except (TypeError, ValueError):
            return None

    if len(raw_values) >= 1:
        try:
            return float(raw_values[0])
        except (TypeError, ValueError):
            return None
    return None


def _group_background_range_for_group(
    grouping: dict,
    group_id: int,
) -> tuple[int, int] | None:
    """Return a background range for one group when explicitly configured."""

    def _parse_range(value: object) -> tuple[int, int] | None:
        if value is None:
            return None
        window = background_window(value)
        return window.first, window.last

    raw_ranges = grouping.get("background_ranges")
    if isinstance(raw_ranges, dict):
        direct = _parse_range(raw_ranges.get(group_id, raw_ranges.get(str(group_id))))
        if direct is not None:
            return direct
        forward = _parse_range(raw_ranges.get("forward"))
        backward = _parse_range(raw_ranges.get("backward"))
        if forward is not None and backward is not None:
            try:
                if int(group_id) == int(grouping.get("forward_group", 1)):
                    return forward
                if int(group_id) == int(grouping.get("backward_group", 2)):
                    return backward
            except (TypeError, ValueError):
                pass
            return forward

    if isinstance(raw_ranges, (list, tuple)) and len(raw_ranges) >= 2:
        if not isinstance(raw_ranges[0], (list, tuple)):
            return _parse_range(raw_ranges)
        first = _parse_range(raw_ranges[0])
        second = _parse_range(raw_ranges[1])
        if first is not None and second is not None:
            try:
                if int(group_id) == int(grouping.get("forward_group", 1)):
                    return first
                if int(group_id) == int(grouping.get("backward_group", 2)):
                    return second
            except (TypeError, ValueError):
                pass
            return first
        shared = _parse_range(raw_ranges)
        if shared is not None:
            return shared

    shared = _parse_range(grouping.get("background_range"))
    if shared is not None:
        return shared
    if "background_forward_range" in grouping or "background_backward_range" in grouping:
        pair = resolve_background_ranges(grouping, 0)
        window = pair[int(group_id == int(grouping.get("backward_group", 2)))]
        return window.first, window.last
    return None


def group_background_payload(grouping, group_id):
    """Retain authored acquisition bounds while routing group-specific settings."""
    mode = resolve_background_mode(grouping)
    payload = dict(grouping)
    # An acquisition window defines both the level and its shared uncertainty.
    # Cached levels do not determine the interval or its estimation uncertainty.
    if mode in ("fixed", "range"):
        selected_value = (
            _group_background_value_for_group(grouping, group_id) if mode == "fixed" else None
        )
        selected_range = (
            _group_background_range_for_group(grouping, group_id) if mode == "range" else None
        )
        if selected_range is not None:
            payload["background_ranges"] = [list(selected_range), list(selected_range)]
            payload["background_mode"] = "range"
        elif mode == "fixed" and selected_value is not None:
            payload["background_fixed_values"] = [selected_value, selected_value]
            payload["background_mode"] = "fixed"

    return payload


def _apply_group_background_correction(
    counts: GroupCounts,
    *,
    grouping: dict,
    group_id: int,
    t0_bin: int,
    bin_width: float,
    metadata: dict,
    source_file: str,
    reference_counts: GroupCounts | None = None,
    reference_scale: float | None = None,
    background_t0_bin: int | None = None,
) -> BackgroundCorrectionResult:
    """Apply optional grouped background subtraction for one detector group."""
    if not bool(grouping.get("background_correction", False)):
        return BackgroundCorrectionResult(
            counts, counts, CorrectionOutcome("background", "none", CorrectionStatus.OFF)
        )

    mode = resolve_background_mode(grouping)
    if mode == "reference_run" and reference_counts is not None:
        length = counts.counts.size
        if reference_counts.counts.size < length:
            raise ValueError(
                f"reference run has {reference_counts.counts.size} aligned bins; sample requires {length}."
            )
        variance = reference_counts.variance
        reference_counts = GroupCounts(
            reference_counts.counts[:length], None if variance is None else variance[:length]
        )
    payload = group_background_payload(grouping, group_id)
    last_good_bin = payload.get("last_good_bin")
    try:
        last_good_bin = int(last_good_bin) if last_good_bin is not None else None
    except (TypeError, ValueError):
        last_good_bin = None
    correction = apply_grouped_background_correction(
        counts,
        counts,
        reference_forward=reference_counts,
        reference_backward=reference_counts,
        reference_scale=reference_scale,
        grouping=payload,
        t0_bin=int(t0_bin),
        bin_width_us=float(bin_width),
        # facility left to derive from the run metadata this helper already has.
        metadata=metadata,
        first_good_bin=payload.get("first_good_bin"),
        last_good_bin=last_good_bin,
        acquisition_start_bin=0 if background_t0_bin is None else background_t0_bin - t0_bin,
    )
    CorrectionOutcomes(background=correction.outcome).require_complete()
    return correction
