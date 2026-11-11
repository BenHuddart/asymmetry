"""Independent count and variance pooling on stored period bins."""

from collections.abc import Callable

import numpy as np

from asymmetry.core.data.dataset import Histogram

RUN_ARITHMETIC_POLICY = "independent_signed_count_variance_v1"


def run_arithmetic_policy(metadata):
    """Scope numerical identity to actual run arithmetic, including unchanged means."""
    return {"run_arithmetic_policy": RUN_ARITHMETIC_POLICY} if metadata.get("combination") else {}


def sum_period_histograms(
    period_histograms: list[list[Histogram]],
    *,
    coefficients: list[float] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> list[Histogram]:
    """Sum per-period histogram lists detector-wise (count level).

    Independent supplied variances add; genuinely raw Poisson contributions use
    their counts. NaN retains unknown source noise for scoped consumer validation. Timing follows the first contributor and bins use common raw-index
    support. This operation does not align different acquisition clocks.
    A unit singleton preserves supplied noise and records unknown raw noise too.
    """
    if cancel_check is not None:
        cancel_check()
    if not period_histograms:
        raise ValueError("No period histograms to sum")
    coefficients = [1.0] * len(period_histograms) if coefficients is None else coefficients
    n_detectors = len(period_histograms[0])
    if any(len(period) != n_detectors for period in period_histograms):
        raise ValueError("Mapped periods require corresponding detector lists")
    if n_detectors == 0:
        raise ValueError("Period histogram lists are empty")
    summed: list[Histogram] = []
    for det in range(n_detectors):
        if cancel_check is not None:
            cancel_check()
        first = period_histograms[0][det]
        n_bins = min(len(period[det].counts) for period in period_histograms)
        counts = np.zeros(n_bins, dtype=np.float64)
        variance = np.zeros(n_bins, dtype=np.float64)
        retain_noise = any(c != 1 for c in coefficients)
        for period, coefficient in zip(period_histograms, coefficients, strict=True):
            histogram = period[det]
            values = np.asarray(histogram.counts[:n_bins], dtype=np.float64)
            counts += coefficient * values
            noise = histogram.count_variance
            if noise is None:
                # A signed/fractional source is not an acquired Poisson draw.
                # Unknown noise stays unknown even when the total looks integral.
                raw_poisson = np.isfinite(values) & (values >= 0) & (values == np.floor(values))
                noise = np.where(raw_poisson, values, np.nan)
            else:
                noise = noise[:n_bins]
                retain_noise = True
            retain_noise |= bool(np.any(np.isnan(noise)))
            variance += coefficient**2 * noise
        summed.append(
            Histogram(
                counts=counts,
                count_variance=variance if retain_noise else None,
                bin_width=first.bin_width,
                t0_bin=first.t0_bin,
                good_bin_start=max(period[det].good_bin_start for period in period_histograms),
                good_bin_end=first.good_bin_end
                if len(period_histograms) == 1
                else min(
                    period[det].good_bin_end if period[det].good_bin_end >= 0 else n_bins - 1
                    for period in period_histograms
                ),
                t0_time_us=first.t0_time_us,
            )
        )
    return summed
