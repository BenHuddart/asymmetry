"""Helpers for WiMDA-style grouped detector Fourier inputs."""

from __future__ import annotations

from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    available_background_modes,
    background_window,
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
        forward_group = int(grouping.get("forward_group"))
    except (TypeError, ValueError):
        forward_group = None
    try:
        backward_group = int(grouping.get("backward_group"))
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
                if int(group_id) == int(grouping.get("forward_group")):
                    return forward
                if int(group_id) == int(grouping.get("backward_group")):
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
                if int(group_id) == int(grouping.get("forward_group")):
                    return first
                if int(group_id) == int(grouping.get("backward_group")):
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
        window = pair[int(group_id == grouping.get("backward_group", 2))]
        return window.first, window.last
    return None


def group_background_payload(grouping, group_id, t0_bin, background_t0_bin):
    """Resolve group-specific estimator settings on the signal grid."""
    mode = resolve_background_mode(grouping)
    payload = dict(grouping)
    # An acquisition window defines both the level and its shared uncertainty.
    # Cached levels alone are exact constants, not a substitute for that window.
    if mode in ("fixed", "range"):
        selected_value = _group_background_value_for_group(grouping, group_id)
        selected_range = _group_background_range_for_group(grouping, group_id)
        source_t0 = t0_bin if background_t0_bin is None else background_t0_bin
        if mode == "range" and selected_range is None and selected_value is None:
            window = resolve_background_ranges({}, source_t0)[0]
            selected_range = window.first, window.last
        if selected_range is not None and (mode == "range" or selected_value is None):
            offset = t0_bin - source_t0
            selected_range = tuple(bound + offset for bound in selected_range)
            payload["background_ranges"] = [list(selected_range), list(selected_range)]
            payload["background_mode"] = "range"
        elif selected_value is not None:
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
) -> tuple[GroupCounts, bool, float | None]:
    """Apply optional grouped background subtraction for one detector group."""
    if not bool(grouping.get("background_correction", False)):
        return counts, False, None

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
    if mode != "reference_run" and mode not in available_background_modes(
        metadata=metadata, source_file=source_file
    ):
        CorrectionOutcomes(
            background=CorrectionOutcome(
                "background",
                mode,
                CorrectionStatus.UNAVAILABLE,
                "mode is unavailable for this acquisition",
            )
        ).require_complete()

    payload = group_background_payload(grouping, group_id, t0_bin, background_t0_bin)
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
        last_good_bin=last_good_bin,
    )
    CorrectionOutcomes(background=correction.outcome).require_complete()
    values = (
        correction.outcome.effective.get("values")
        if correction.outcome.effective is not None
        else None
    )
    background_value = values[0] if values is not None else None
    return (
        correction.forward,
        correction.outcome.status == CorrectionStatus.APPLIED,
        background_value,
    )
