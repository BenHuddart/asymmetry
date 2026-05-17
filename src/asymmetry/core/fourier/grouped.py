"""Helpers for WiMDA-style grouped detector Fourier inputs."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    supports_background_correction,
)
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.transform.grouping import apply_grouping_aligned, common_t0_for_groups
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _normalize_group_entries(values) -> list[int]:
    """Return zero-based detector indices for one grouping entry list."""
    indices: list[int] = []
    if not isinstance(values, list):
        return indices
    for value in values:
        detector = value[0] if isinstance(value, (list, tuple)) and value else value
        try:
            indices.append(max(0, int(detector) - 1))
        except (TypeError, ValueError):
            continue
    return indices


def _rebin_group_counts(
    time: np.ndarray,
    counts: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return WiMDA-style bunched grouped counts.

    Grouped Fourier inputs are count-like signals, so bunching should merge
    neighboring bins by summing counts while moving the time coordinate to the
    center of the wider effective bin.
    """
    bunch_factor = max(1, int(factor))
    if bunch_factor <= 1 or counts.size < bunch_factor:
        return np.asarray(time, dtype=np.float64), np.asarray(counts, dtype=np.float64)

    n_new = counts.size // bunch_factor
    trimmed = n_new * bunch_factor
    rebinned_time = np.asarray(time[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).mean(axis=1)
    rebinned_counts = (
        np.asarray(counts[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).sum(axis=1)
    )
    return rebinned_time, rebinned_counts


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
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            lo = int(float(value[0]))
            hi = int(float(value[1]))
        except (TypeError, ValueError):
            return None
        return (lo, hi) if lo <= hi else (hi, lo)

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
    return None


def _apply_group_background_correction(
    counts: np.ndarray,
    *,
    grouping: dict,
    group_id: int,
    t0_bin: int,
    bin_width: float,
    metadata: dict,
    source_file: str,
) -> tuple[np.ndarray, bool, float | None]:
    """Apply optional grouped background subtraction for one detector group."""
    if not bool(grouping.get("background_correction", False)):
        return counts, False, None

    if not supports_background_correction(metadata=metadata, source_file=source_file):
        return counts, False, None

    payload = dict(grouping)
    selected_value = _group_background_value_for_group(grouping, group_id)
    selected_range = _group_background_range_for_group(grouping, group_id)
    if selected_value is not None:
        payload["background_fixed_values"] = [selected_value, selected_value]
    elif selected_range is not None:
        payload["background_ranges"] = [list(selected_range), list(selected_range)]

    correction = apply_grouped_background_correction(
        np.asarray(counts, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
        grouping=payload,
        t0_bin=int(t0_bin),
        bin_width_us=float(bin_width),
        facility=str(metadata.get("facility", "")),
    )
    background_value = correction.values[0] if correction.values is not None else None
    return correction.forward, bool(correction.applied), background_value


def build_group_signal_dataset(
    run: Run,
    group_id: int,
    *,
    center_signal: bool = True,
    apply_lifetime_correction: bool = True,
    use_deadtime: bool | None = None,
    reference_t0_bin: int | None = None,
    prepared_histograms: list[Histogram] | None = None,
) -> MuonDataset:
    """Build a time-domain grouped detector signal for Fourier analysis.

    This mirrors WiMDA's grouped-detector Fourier input more closely than the
    standard forward/backward asymmetry dataset used elsewhere in the GUI.
    The returned signal is centred by default so the FFT is not dominated by
    the zero-frequency component.

    WiMDA's grouped Fourier path uses lifetime-corrected detector counts as the
    FFT source. This helper mirrors that by default via ``exp(t / tau_mu)``
    before later mean subtraction and FFT filtering.
    """
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    if not isinstance(groups, dict):
        raise ValueError("Run does not define detector groups for Fourier analysis.")

    if group_id not in groups:
        raise ValueError(f"Unknown detector group {group_id!r}.")

    indices = _normalize_group_entries(groups[group_id])
    if not indices:
        raise ValueError(f"Detector group {group_id!r} does not contain any detectors.")

    histograms = list(run.histograms)
    if not histograms:
        raise ValueError("Run does not contain any histograms.")

    if prepared_histograms is None:
        apply_deadtime = bool(grouping.get("deadtime_correction", False))
        if use_deadtime is not None:
            apply_deadtime = bool(use_deadtime)
        prepared_histograms, _ = prepare_histograms_with_deadtime(histograms, grouping, apply_deadtime)
    else:
        prepared_histograms = list(prepared_histograms)
        if not prepared_histograms:
            raise ValueError("Prepared histograms are empty for Fourier analysis.")

    if reference_t0_bin is None:
        all_group_indices = [
            normalized
            for normalized in (_normalize_group_entries(values) for values in groups.values())
            if normalized
        ]
        reference_t0_bin = common_t0_for_groups(prepared_histograms, *all_group_indices)
    common_t0 = max(0, int(reference_t0_bin))
    counts = apply_grouping_aligned(
        prepared_histograms,
        indices,
        common_t0_bin=common_t0,
    )
    if counts.size == 0:
        raise ValueError(f"Detector group {group_id!r} produced an empty signal.")

    source_file = str(getattr(run, "source_file", "") or "")
    counts, background_applied, background_value = _apply_group_background_correction(
        counts,
        grouping=grouping,
        group_id=int(group_id),
        t0_bin=int(grouping.get("t0_bin", common_t0)),
        bin_width=float(prepared_histograms[0].bin_width),
        metadata=run.metadata,
        source_file=source_file,
    )

    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = int(grouping.get("last_good_bin", counts.size - 1))
    except (TypeError, ValueError):
        last_good = counts.size - 1
    last_good = min(last_good, counts.size - 1)
    if first_good > last_good:
        first_good = 0
        last_good = counts.size - 1

    trimmed_counts = np.asarray(counts[first_good : last_good + 1], dtype=np.float64)
    try:
        bunch_factor = max(1, int(grouping.get("bunching_factor", 1)))
    except (TypeError, ValueError):
        bunch_factor = 1

    bin_width = float(prepared_histograms[0].bin_width)
    axis_start = first_good - common_t0
    time = (np.arange(trimmed_counts.size, dtype=float) + float(axis_start)) * bin_width
    if bunch_factor > 1:
        time, trimmed_counts = _rebin_group_counts(time, trimmed_counts, bunch_factor)

    scale = np.ones_like(trimmed_counts, dtype=np.float64)
    if apply_lifetime_correction and trimmed_counts.size > 0:
        scale = np.exp(np.asarray(time, dtype=np.float64) / float(MUON_LIFETIME_US))

    signal = trimmed_counts * scale
    if center_signal and signal.size > 0:
        signal -= float(np.mean(signal))
    error = np.sqrt(np.clip(trimmed_counts, 1.0, None)) * scale

    group_names = grouping.get("group_names") if isinstance(grouping, dict) else None
    group_name = None
    if isinstance(group_names, dict):
        group_name = group_names.get(group_id)
    if group_name is None:
        group_name = f"Group {group_id}"

    metadata = dict(run.metadata)
    metadata.update(
        {
            "run_number": run.run_number,
            "run_label": f"{run.run_number} {group_name}",
            "fourier_source": "group_signal",
            "fourier_lifetime_corrected": bool(apply_lifetime_correction),
            "fourier_background_corrected": bool(background_applied),
            "fourier_background_value": (
                float(background_value) if background_value is not None else None
            ),
            "group_id": int(group_id),
            "group_name": str(group_name),
        }
    )

    return MuonDataset(
        time=time,
        asymmetry=signal,
        error=error,
        metadata=metadata,
        run=run,
    )