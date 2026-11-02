"""Counts-first histogram binning and independent curve-bin merging.

BinningSpec fixes the raw edges for fixed, exponentially variable, or
constant-error binning. Corrected counts and their shared-background noise are
combined on those edges before forming a forward/backward asymmetry ratio.
The raw acquisition record is unchanged. Curve-only inputs use
merge_curve_bins, whose errors assume independent points.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.transform.asymmetry import (
    compute_asymmetry_with_count_errors,
)
from asymmetry.core.transform.fb_counts import ForwardBackwardCounts, linear_combination_variance
from asymmetry.core.utils.constants import MUON_LIFETIME_US

BINNING_MODES = ("fixed", "variable", "constant_error")


def merge_curve_bins(
    time: NDArray[np.float64],
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    factor: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Merge independent curve bins by a linear mean, without using raw counts.

    Parameters
    ----------
    time, values, errors
        Equal-length arrays of the original data. ``values`` may be asymmetry on
        **either** scale — fraction or percent (see
        :mod:`asymmetry.core.transform.units`).
    factor
        Number of bins to merge into one.

    Returns
    -------
    (time_rebinned, values_rebinned, errors_rebinned)
        **On the same scale as the input.** The merge is a linear mean of
        ``values`` with errors added in quadrature, so this transform is
        unit-preserving: it never converts between fraction and percent.
    """
    BinningSpec(factor=factor)
    if factor == 1:
        return time.copy(), values.copy(), errors.copy()

    n = len(time)
    n_new = n // factor
    trimmed = n_new * factor

    t = time[:trimmed].reshape(n_new, factor).mean(axis=1)
    v = values[:trimmed].reshape(n_new, factor).mean(axis=1)
    e = np.sqrt((errors[:trimmed].reshape(n_new, factor) ** 2).sum(axis=1)) / factor

    return t, v, e


def rebin_counts(
    time: NDArray[np.float64],
    counts: NDArray[np.float64],
    factor: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return count-preserving bunched traces (sum counts, mean times).

    Unlike :func:`merge_curve_bins`, which is value-domain (mean of values, errors in
    quadrature ÷ factor), this merges neighbouring bins by *summing* the
    counts while moving the time coordinate to the centre of the wider
    effective bin — the right combiner for grouped count-like signals
    (fit-input traces, grouped Fourier inputs). The two are deliberately
    distinct: a value-domain mean would corrupt Poisson statistics on counts.

    ``factor`` must be an integer >= 1. A factor of one returns float64 copies;
    an incomplete trailing group is dropped.
    """
    bunch_factor = BinningSpec(factor=factor).factor
    if bunch_factor == 1:
        return np.asarray(time, dtype=np.float64).copy(), np.asarray(
            counts, dtype=np.float64
        ).copy()

    n_new = counts.size // bunch_factor
    trimmed = n_new * bunch_factor
    rebinned_time = (
        np.asarray(time[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).mean(axis=1)
    )
    rebinned_counts = (
        np.asarray(counts[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).sum(axis=1)
    )
    return rebinned_time, rebinned_counts


@dataclass(frozen=True)
class BinningSpec:
    """Validated raw-bin edges and the number of recorded bins to merge."""

    mode: str = "fixed"
    factor: int = 1
    bin0_us: float = 0.08
    bin10_us: float = 0.25
    merge: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in BINNING_MODES:
            raise ValueError(
                f"binning mode {self.mode!r} is not one of {', '.join(BINNING_MODES)}."
            )
        for name, value in (("bunching factor", self.factor), ("bin merge", self.merge)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1; got {value!r}.")
        for name, value in (("bin0_us", self.bin0_us), ("bin10_us", self.bin10_us)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and > 0; got {value!r}.")

    def edges(self, n_bins: int, bin_width_us: float, t_start_us: float) -> NDArray[np.intp]:
        """Raw edges; retain a variable-mode tail, drop an incomplete merged group."""
        if self.mode == "fixed":
            edges = np.arange(
                0, n_bins // self.factor * self.factor + 1, self.factor, dtype=np.intp
            )
        else:
            boundaries = [0]
            index = 0
            while index < n_bins:
                time = t_start_us + index * bin_width_us
                width = _target_width_us(self.mode, time, self.bin0_us, self.bin10_us)
                step = max(1, int(np.ceil(width / bin_width_us - 1e-9)))
                index = min(index + step, n_bins)
                boundaries.append(index)
            edges = np.asarray(boundaries, dtype=np.intp)
        return edges[:: self.merge]


def binning_spec(grouping: dict[str, Any] | None) -> BinningSpec:
    """Parse file/project/user grouping binning, rejecting present invalid values."""
    values = {} if grouping is None else grouping
    return BinningSpec(
        mode=values.get("binning_mode", "fixed"),
        factor=values.get("bunching_factor", 1),
        bin0_us=values.get("bin0_us", 0.08),
        bin10_us=values.get("bin10_us", 0.25),
    )


def _target_width_us(mode: str, t_us: float, bin0_us: float, bin10_us: float) -> float:
    if mode == "variable":
        return bin0_us * (bin10_us / bin0_us) ** (max(t_us, 0.0) / 10.0)
    return bin0_us * np.exp(max(t_us, 0.0) / MUON_LIFETIME_US)


def binned_fb_asymmetry(
    counts: ForwardBackwardCounts,
    *,
    binning: BinningSpec,
    alpha: float,
    beta: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reduce grouped counts to a binned asymmetry curve (all binning modes).

    The validated spec selects raw edges inside the good-bin window, then
    merges ``binning.merge`` recorded output bins. Counts and their full
    variance model are summed before the ratio is formed. Fixed binning drops
    a partial raw group; variable modes keep their last partial bin. An
    incomplete group of output bins is dropped when merge is greater than one.

    Raw bin ``k`` is stamped at its **centre** measured from the run's exact t0,
    ``(k + 0.5)·w − t0_time_us``. The validated counts carry the exact t0
    and inclusive aligned-bin window.

    Returns
    -------
    (time, asymmetry, error)
        ``time`` in µs relative to t0; ``asymmetry`` and ``error`` **as a
        fraction** (``A ∈ [-1, 1]``) — *not* percent, since this delegates
        straight to :func:`~asymmetry.core.transform.compute_asymmetry`. Callers
        that need the percent scale apply the ×100 themselves;
        :func:`~asymmetry.core.transform.reduce_grouped_asymmetry` is the
        reduction that does, and its output is percent.
    """
    window = counts.window
    f = counts.forward.counts[window.slice]
    b = counts.backward.counts[window.slice]
    bin_width_us = counts.bin_width_us
    residual_us = (counts.common_t0_bin + 0.5) * bin_width_us - counts.t0_time_us
    t_start = (window.first - counts.common_t0_bin) * bin_width_us + residual_us

    edges = binning.edges(f.size, float(bin_width_us), t_start)
    if edges.size < 2:
        empty = np.array([], dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    # reduceat runs its last segment to the array end; exclude dropped raw bins.
    f = f[: edges[-1]]
    b = b[: edges[-1]]
    starts = edges[:-1]
    f_out = np.add.reduceat(f, starts)
    b_out = np.add.reduceat(b, starts)
    absolute_edges = edges + window.first
    vf = linear_combination_variance(counts.forward, absolute_edges)
    vb = linear_combination_variance(counts.backward, absolute_edges)
    asymmetry, error = compute_asymmetry_with_count_errors(
        f_out, b_out, np.sqrt(vf), np.sqrt(vb), alpha=alpha, beta=beta
    )
    # Mean of the merged raw bins' stamps: for the slice [e0, e1) that is
    # t_start + ((e0 + e1 − 1)/2)·w — matching the fixed-mode path, where
    # merge_curve_bins() averages the same per-bin stamps.
    time = t_start + (edges[:-1] + edges[1:] - 1) * 0.5 * float(bin_width_us)
    return np.asarray(time, dtype=np.float64), asymmetry, error
