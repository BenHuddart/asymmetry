"""Identity of grouped Fourier observations and their conditioning policy."""

from __future__ import annotations

import hashlib
import json

from asymmetry.core.data.dataset import Run
from asymmetry.core.fourier.conditioning import CONDITIONING_REVISION
from asymmetry.core.fourier.group_preparation import histogram_grid_identity
from asymmetry.core.io.period_counts import run_arithmetic_policy
from asymmetry.core.transform.background import (
    background_range_is_explicit,
    resolve_background_mode,
    resolve_facility,
)
from asymmetry.core.transform.constituent_counts import constituent_metadata, held_period_grids
from asymmetry.core.transform.corrections import TAIL_BACKGROUND_POLICY_REVISION
from asymmetry.core.transform.deadtime import file_deadtime_table
from asymmetry.core.transform.grouping import grouping_balances
from asymmetry.core.transform.period_reduction import period_statistic_policy
from asymmetry.core.transform.rebin import BINNING_POLICY, binning_spec
from asymmetry.core.transform.t0 import detector_t0_overrides


def _digest_groups(groups: object) -> dict[int, list[int]] | None:
    """Normalise parseable group/detector IDs for metadata-only cache comparisons."""
    if not isinstance(groups, dict):
        return None
    normalized: dict[int, list[int]] = {}
    for raw_gid, entries in groups.items():
        try:
            gid = int(raw_gid)
        except (TypeError, ValueError):
            continue
        if not isinstance(entries, (list, tuple)):
            continue
        detectors: list[int] = []
        for entry in entries:
            detector = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            try:
                detectors.append(int(detector))
            except (TypeError, ValueError):
                continue
        normalized[gid] = sorted(detectors)
    return normalized


def _digest_int_list(value: object) -> list[int] | None:
    """Parse a sequence of ids into a sorted ``list[int]``, or ``None``."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(ids)


def _digest_int(value: object) -> int | None:
    """Parse a scalar id/bin count, or ``None`` when unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fourier_grouping_digest(run: Run | None, signal_source: str = "grouped_average") -> str:
    if run is None:
        return ""
    grouping = run.grouping
    payload = {
        "deadtime_scope": "required_grouped_observations",
        "spectrum_conditioning": CONDITIONING_REVISION,
        "tail_background_policy": TAIL_BACKGROUND_POLICY_REVISION,
        "binning_policy": BINNING_POLICY,
        "histogram_grid": histogram_grid_identity(run.histograms),
        **run_arithmetic_policy(run.metadata),
    }

    if (groups := _digest_groups(grouping.get("groups"))) is not None:
        payload["groups"] = groups
    payload["excluded_detectors"] = _digest_int_list(grouping.get("excluded_detectors")) or []

    deadtime_correction = bool(grouping.get("deadtime_correction", False))
    payload["deadtime_correction"] = deadtime_correction
    if deadtime_correction:
        mode = (
            str(grouping.get("deadtime_mode", grouping.get("deadtime_method", "file")))
            .strip()
            .lower()
        )
        payload["deadtime_mode"] = mode
        if mode not in {"off", "none"}:
            payload["good_frames"] = grouping.get("good_frames")
        payload["deadtime_method"] = grouping.get("deadtime_method")
        payload["dead_time_us"] = (
            file_deadtime_table(grouping)
            if mode in {"", "file", "from_file"}
            else grouping.get("dead_time_us", grouping.get("deadtime_loaded_us"))
        )

    background_correction = bool(grouping.get("background_correction", False))
    payload["background_correction"] = background_correction
    if background_correction:
        mode = resolve_background_mode(grouping)
        payload["background_mode"] = mode
        if mode == "reference_run":
            from asymmetry.core.io import reference_file_identity

            request = grouping.get("background_run")
            payload["reference_file_identity"] = reference_file_identity(request)
            for key in ("reference_exposure", "period_reference_exposure"):
                payload[key] = grouping.get(key)
        elif mode == "range" and not background_range_is_explicit(grouping):
            payload["background_facility"] = resolve_facility(
                metadata=run.metadata, grouping=grouping
            )
        for key in (
            "background_values",
            "background_ranges",
            "background_range",
            "background_forward_range",
            "background_backward_range",
            "background_fixed_values",
            "background_fix",
            "bkg_fix",
            "background_run",
        ):
            if key in grouping:
                payload[key] = grouping[key]
        if (
            mode == "range"
            or isinstance(grouping.get("background_values"), (list, tuple))
            or any(
                key in grouping for key in ("background_fixed_values", "background_fix", "bkg_fix")
            )
        ):
            forward_group = _digest_int(grouping.get("forward_group", 1))
            backward_group = _digest_int(grouping.get("backward_group", 2))
            if forward_group is not None:
                payload["forward_group"] = forward_group
            if backward_group is not None:
                payload["backward_group"] = backward_group

    for key in ("t0_bin", "first_good_bin", "last_good_bin", "bunching_factor"):
        parsed = _digest_int(grouping.get(key))
        if parsed is not None:
            payload[key] = parsed
    t0_time_us = grouping.get("t0_time_us")
    if t0_time_us is not None:
        payload["t0_time_us"] = float(t0_time_us)
    t0_source = grouping.get("t0_source")
    if t0_source is not None:
        payload["t0_source"] = str(t0_source)

    if "period_inputs" in grouping:
        payload["constituents"] = constituent_metadata(grouping)
    if signal_source == "fb_asymmetry":
        payload.update(period_statistic_policy(grouping))
        payload["signal_source"] = "fb_asymmetry"
        forward_group = _digest_int(grouping.get("forward_group", 1))
        backward_group = _digest_int(grouping.get("backward_group", 2))
        payload["fb_forward_group"] = forward_group if forward_group is not None else 1
        payload["fb_backward_group"] = backward_group if backward_group is not None else 2
        alpha, beta = grouping_balances(grouping)
        payload["fb_alpha"] = alpha
        if beta != 1.0:
            payload["fb_beta"] = beta
        overrides = detector_t0_overrides(grouping, len(run.histograms))
        if overrides is not None:
            payload["fb_detector_t0_bins"] = overrides
        binning = binning_spec(grouping)
        payload["fb_binning_mode"] = binning.mode
        if binning.mode != "fixed":
            payload["fb_bin0_us"] = binning.bin0_us
            payload["fb_bin10_us"] = binning.bin10_us
        period_lists = grouping.get("period_histograms")
        has_periods = (
            isinstance(period_lists, list)
            and len(period_lists) >= 2
            and all(isinstance(period, list) and period for period in period_lists)
        )
        if has_periods:
            payload["fb_period_mode"] = str(grouping.get("period_mode", "red"))
            payload["fb_period_grids"] = held_period_grids(grouping)

    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]
