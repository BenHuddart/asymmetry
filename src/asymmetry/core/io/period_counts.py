"""Independent count and variance pooling on stored period bins."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from asymmetry.core.data.dataset import Histogram


def sum_period_histograms(
    period_histograms: list[list[Histogram]], *, cancel_check: Callable[[], None] | None = None
) -> list[Histogram]:
    """Sum per-period histogram lists detector-wise (count level).

    Independent supplied variances add; raw Poisson contributions use their
    counts. Timing follows the first contributor and bins use common raw-index
    support. This operation does not align different acquisition clocks.
    A singleton is copied without inferring a variance or statistical eligibility.
    """
    if cancel_check is not None:
        cancel_check()
    if not period_histograms:
        raise ValueError("No period histograms to sum")
    n_detectors = len(period_histograms[0])
    if any(len(period) != n_detectors for period in period_histograms):
        raise ValueError("Mapped periods require corresponding detector lists")
    if n_detectors == 0:
        raise ValueError("Period histogram lists are empty")
    if len(period_histograms) == 1:
        copied = []
        for hist in period_histograms[0]:
            if cancel_check is not None:
                cancel_check()
            copied.append(replace(hist, counts=np.array(hist.counts, dtype=float, copy=True)))
        return copied
    summed: list[Histogram] = []
    for det in range(n_detectors):
        if cancel_check is not None:
            cancel_check()
        first = period_histograms[0][det]
        n_bins = min(len(period[det].counts) for period in period_histograms)
        if any(
            period[det].count_variance is None and np.any(period[det].counts[:n_bins] < 0)
            for period in period_histograms
        ):
            raise ValueError(
                "Signed period counts require supplied variance for independent pooling"
            )
        counts = np.zeros(n_bins, dtype=np.float64)
        for period in period_histograms:
            counts += np.asarray(period[det].counts[:n_bins], dtype=np.float64)
        variance = None
        if any(period[det].count_variance is not None for period in period_histograms):
            variance = sum(
                (
                    period[det].count_variance
                    if period[det].count_variance is not None
                    else period[det].counts
                )[:n_bins]
                for period in period_histograms
            )
        summed.append(
            Histogram(
                counts=counts,
                count_variance=variance,
                bin_width=first.bin_width,
                t0_bin=first.t0_bin,
                good_bin_start=max(period[det].good_bin_start for period in period_histograms),
                good_bin_end=min(
                    period[det].good_bin_end if period[det].good_bin_end >= 0 else n_bins - 1
                    for period in period_histograms
                ),
                t0_time_us=first.t0_time_us,
            )
        )
    return summed
