"""Detector grouping utilities.

Groups individual detector histograms into forward / backward (or custom)
groups by summing their counts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram


def apply_grouping(
    histograms: list[Histogram],
    group_indices: list[int],
) -> NDArray[np.float64]:
    """Sum the counts of the listed histograms into a single group array.

    Parameters
    ----------
    histograms
        All histograms from the run.
    group_indices
        0-based indices of the histograms to include in this group.

    Returns
    -------
    NDArray
        Summed counts array.
    """
    arrays = [histograms[i].counts for i in group_indices]
    # Truncate to shortest length in case of mismatch
    min_len = min(len(a) for a in arrays)
    total = np.zeros(min_len, dtype=np.float64)
    for a in arrays:
        total += a[:min_len]
    return total


def apply_grouping_aligned(
    histograms: list[Histogram],
    group_indices: list[int],
    *,
    common_t0_bin: int | None = None,
) -> NDArray[np.float64]:
    """Sum detector counts after aligning each detector's own ``t0_bin``.

    ISIS NeXus files normally provide one common ``t0`` for every detector, so
    :func:`apply_grouping` is sufficient. PSI BIN/MDU data can carry different
    ``t0`` values per detector. This helper shifts each selected detector so
    that its local ``t0_bin`` lands on ``common_t0_bin`` before summing.
    """
    selected = [histograms[i] for i in group_indices]
    if not selected:
        return np.array([], dtype=np.float64)

    if common_t0_bin is None:
        common_t0_bin = max(0, max(int(hist.t0_bin) for hist in selected))
    common_t0_bin = max(0, int(common_t0_bin))

    shifted: list[NDArray[np.float64]] = []
    for hist in selected:
        counts = np.asarray(hist.counts, dtype=np.float64)
        offset = common_t0_bin - int(hist.t0_bin)
        if offset <= 0:
            shifted.append(counts[-offset:].copy() if offset < 0 else counts.copy())
            continue

        out = np.zeros(len(counts) + offset, dtype=np.float64)
        out[offset:] = counts
        shifted.append(out)

    min_len = min(len(a) for a in shifted)
    total = np.zeros(min_len, dtype=np.float64)
    for a in shifted:
        total += a[:min_len]
    return total


def common_t0_for_groups(
    histograms: list[Histogram],
    *group_indices: list[int],
) -> int:
    """Return a common t0 suitable for comparing multiple detector groups."""
    indices = sorted({idx for group in group_indices for idx in group})
    if not indices:
        return int(histograms[0].t0_bin) if histograms else 0
    return max(0, max(int(histograms[idx].t0_bin) for idx in indices))
