"""Factories for synthetic counts that already share an acquisition grid."""

from __future__ import annotations

import numpy as np

from asymmetry.core.transform.fb_counts import (
    AlignmentCrop,
    BinWindow,
    ForwardBackwardCounts,
    GroupCounts,
    count_timing,
)


def aligned_counts(
    forward,
    backward,
    *,
    common_t0: int = 0,
    bin_width_us: float = 0.01,
    first_good_bin: int = 0,
    last_good_bin: int | None = None,
    forward_error=None,
    backward_error=None,
    t0_time_us: float | None = None,
) -> ForwardBackwardCounts:
    """Describe an already aligned synthetic F/B record, without cropping it."""
    f = GroupCounts(forward, None if forward_error is None else np.asarray(forward_error) ** 2)
    b = GroupCounts(backward, None if backward_error is None else np.asarray(backward_error) ** 2)
    width, t0 = count_timing(bin_width_us, common_t0, t0_time_us)
    return ForwardBackwardCounts(
        f,
        b,
        t0,
        width,
        BinWindow(first_good_bin, f.counts.size - 1 if last_good_bin is None else last_good_bin),
        AlignmentCrop(common_t0, (), (f.counts.size, b.counts.size), f.counts.size),
    )
