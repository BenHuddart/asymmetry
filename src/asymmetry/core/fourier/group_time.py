"""Existing grouped spectral bin centres and retained raw edges."""

import numpy as np

from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.t0 import common_t0_time_us


def group_time_grid(histograms, grouping, indices, alignment, window):
    """Retain complete bunches on the exact-t0 integer-aligned signal grid."""
    factor = binning_spec(grouping).factor
    width = float(histograms[indices[0]].bin_width)
    t0 = alignment.common_t0_bin
    residual = (t0 + 0.5) * width - common_t0_time_us(
        histograms, grouping, t0, selected_indices=indices
    )
    size = (window.last - window.first + 1) // factor
    edges = window.first + np.arange(size + 1) * factor
    time = (np.arange(size * factor) + float(window.first - t0)) * width + residual
    return time.reshape(-1, factor).mean(axis=1), edges
