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
