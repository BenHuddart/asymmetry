"""Deadtime correction utilities for raw detector histograms."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram


def apply_deadtime_correction(
    counts,
    tau_us: float,
    bin_width_us: float,
    *,
    num_good_frames: float = 1.0,
):
    """Apply the musrfit/Mantid non-paralyzable deadtime correction.

    ``N_corr = N / (1 - N * tau / (dt * n_frames))``
    """
    n = np.asarray(counts, dtype=np.float64)
    tau_us = float(tau_us)
    if not np.isfinite(tau_us) or tau_us == 0.0 or bin_width_us <= 0.0 or num_good_frames <= 0.0:
        return n.copy()

    denom = 1.0 - (n * tau_us / (float(bin_width_us) * float(num_good_frames)))
    denom = np.clip(denom, 1.0e-6, None)
    return n / denom


def has_file_deadtime(grouping: dict, n_histograms: int = 0) -> bool:
    """Return ``True`` when grouping metadata contains file deadtime values."""
    if not isinstance(grouping, dict):
        return False
    dead_time_us = grouping.get("dead_time_us")
    if not isinstance(dead_time_us, list):
        return False
    required = max(1, int(n_histograms))
    return len(dead_time_us) >= required and any(_finite_nonzero(value) for value in dead_time_us)


def prepare_histograms_with_deadtime(
    histograms: list[Histogram],
    grouping: dict,
    use_deadtime: bool,
) -> tuple[list[Histogram], bool]:
    """Return histograms with optional deadtime correction applied."""
    if not use_deadtime:
        return list(histograms), False

    dead_time_us = grouping.get("dead_time_us") if isinstance(grouping, dict) else None
    if isinstance(dead_time_us, list) and len(dead_time_us) >= len(histograms):
        return _prepare_histograms_with_file_deadtime(histograms, grouping, dead_time_us)

    return list(histograms), False


def _prepare_histograms_with_file_deadtime(
    histograms: list[Histogram],
    grouping: dict,
    dead_time_us: list,
) -> tuple[list[Histogram], bool]:
    try:
        good_frames = float(grouping.get("good_frames", 1.0))
    except (TypeError, ValueError):
        good_frames = 1.0
    if good_frames <= 0.0:
        good_frames = 1.0

    corrected: list[Histogram] = []
    applied_any = False
    for i, hist in enumerate(histograms):
        try:
            tau_us = float(dead_time_us[i])
        except (TypeError, ValueError):
            tau_us = 0.0

        counts = np.asarray(hist.counts, dtype=np.float64)
        if tau_us != 0.0 and np.isfinite(tau_us):
            counts = apply_deadtime_correction(
                counts,
                tau_us,
                hist.bin_width,
                num_good_frames=good_frames,
            )
            applied_any = True

        corrected.append(_copy_histogram_with_counts(hist, counts))

    if applied_any and isinstance(grouping, dict):
        grouping["deadtime_method"] = "file"

    return corrected, applied_any


def _finite_nonzero(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number != 0.0)


def _copy_histogram_with_counts(histogram: Histogram, counts) -> Histogram:
    return Histogram(
        counts=np.asarray(counts, dtype=np.float64),
        bin_width=histogram.bin_width,
        t0_bin=histogram.t0_bin,
        good_bin_start=histogram.good_bin_start,
        good_bin_end=histogram.good_bin_end,
    )
