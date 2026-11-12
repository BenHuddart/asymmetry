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
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.ratio_validity import RatioValidity
from asymmetry.core.transform.counting_noise import CountingNoise, bin_matrix, combine_noise
from asymmetry.core.transform.fb_counts import ForwardBackwardCounts, linear_combination_variance
from asymmetry.core.utils.constants import MUON_LIFETIME_US

if TYPE_CHECKING:
    from asymmetry.core.transform.grouping import ForwardBackwardLayout


BINNING_MODES = ("fixed", "variable", "constant_error")
BINNING_POLICY = "nearest_half_up_left_edge_v1"


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
    #: Successive (first, stop, merge) selections of recorded output bins.
    selections: tuple[tuple[int, int, int], ...] = ()
    #: Centre-selected support after base binning, before later selections.
    time_range: tuple[float | None, float | None] | None = None

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
        """Raw edges from the first native centre ``t_start_us``.

        Nonuniform targets use the actual left edge and nearest whole native
        width (half-up, at least one). Retain their final partial bin; fixed
        packing and an additional merge drop only incomplete groups.
        """
        if self.mode == "fixed":
            edges = np.arange(
                0, n_bins // self.factor * self.factor + 1, self.factor, dtype=np.intp
            )
        else:
            boundaries = [0]
            index = 0
            while index < n_bins:
                time = t_start_us + (index - 0.5) * bin_width_us
                width = _target_width_us(self.mode, time, self.bin0_us, self.bin10_us)
                whole, fraction = divmod(width / bin_width_us, 1.0)
                # Compare the fractional quotient directly: adding 0.5 can
                # round the representable value below a half-integer to a tie.
                step = max(1, int(whole) + int(fraction >= 0.5))
                index = min(index + step, n_bins)
                boundaries.append(index)
            edges = np.asarray(boundaries, dtype=np.intp)
        edges = edges[:: self.merge]
        if self.time_range is not None and edges.size > 1:
            times = t_start_us + (edges[:-1] + edges[1:] - 1) * 0.5 * bin_width_us
            lower, upper = self.time_range
            mask = np.ones(times.shape, dtype=bool)
            if lower is not None:
                mask &= times >= lower
            if upper is not None:
                mask &= times <= upper
            selected = np.flatnonzero(mask)
            edges = edges[:1] if not selected.size else edges[selected[0] : selected[-1] + 2]
        for first, stop, merge in self.selections:
            edges = edges[first : stop + 1 : merge]
        return edges


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


def fb_binning_edges(
    counts: ForwardBackwardCounts | ForwardBackwardLayout,
    binning: BinningSpec,
    *,
    time_range: tuple[float | None, float | None] | None = None,
) -> tuple[NDArray[np.intp], float]:
    """Resolve consumed local acquisition edges and the first bin's timestamp."""
    window = counts.window
    width = counts.bin_width_us
    residual = (counts.common_t0_bin + 0.5) * width - counts.t0_time_us
    t_start = (window.first - counts.common_t0_bin) * width + residual
    edges = binning.edges(window.last - window.first + 1, width, t_start)
    if time_range is not None and edges.size > 1:
        times = t_start + (edges[:-1] + edges[1:] - 1) * 0.5 * width
        lower, upper = time_range
        mask = np.ones(times.shape, dtype=bool)
        if lower is not None:
            mask &= times >= lower
        if upper is not None:
            mask &= times <= upper
        selected = np.flatnonzero(mask)
        edges = edges[:1] if not selected.size else edges[selected[0] : selected[-1] + 2]
    return edges + window.first, t_start


class BinnedRatio(NamedTuple):
    """Fractional curve and its retained acquisition response."""

    time: NDArray[np.float64]
    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    validity: RatioValidity
    noise: CountingNoise | None
    edges: NDArray[np.intp]


def binned_ratio_observation(
    counts: ForwardBackwardCounts,
    *,
    binning: BinningSpec,
    alpha: float,
    beta: float = 1.0,
    time_range: tuple[float | None, float | None] | None = None,
) -> BinnedRatio:
    """Reduce grouped counts to a binned asymmetry curve (all binning modes).

    ``time_range`` optionally selects complete output bins by their centres;
    it never moves the bin edges. The validated spec selects raw edges inside
    the good-bin window, then
    merges ``binning.merge`` recorded output bins. Counts and their retained
    variance model are summed before the ratio is formed. Fixed binning drops
    a partial raw group; variable modes keep their last partial bin. An
    incomplete group of output bins is dropped when merge is greater than one.

    Raw bin ``k`` is stamped at its **centre** measured from the run's exact t0,
    ``(k + 0.5)·w − t0_time_us``. The validated counts carry the exact t0
    and inclusive aligned-bin window.

    Returns
    -------
    (time, asymmetry, error, validity)
        ``validity`` classifies each final ratio before display fallbacks.
        ``time`` in µs relative to t0; ``asymmetry`` and ``error`` **as a
        fraction** — *not* percent, since this delegates
        to :func:`~asymmetry.core.transform.compute_asymmetry_with_count_errors`. Callers
        that need the percent scale apply the ×100 themselves;
        :func:`~asymmetry.core.transform.reduce_grouped_asymmetry` is the
        reduction that does, and its output is percent.
    """
    window = counts.window
    f = counts.forward.counts[window.slice]
    b = counts.backward.counts[window.slice]
    bin_width_us = counts.bin_width_us
    absolute_edges, t_start = fb_binning_edges(counts, binning, time_range=time_range)
    edges = absolute_edges - window.first
    if edges.size < 2:
        empty = np.array([], dtype=np.float64)
        return BinnedRatio(
            empty,
            empty.copy(),
            empty.copy(),
            RatioValidity(np.array([], dtype=np.uint8)),
            None,
            absolute_edges,
        )
    # reduceat runs its last segment to the array end; exclude dropped raw bins.
    f = f[: edges[-1]]
    b = b[: edges[-1]]
    starts = edges[:-1]
    f_out = np.add.reduceat(f, starts)
    b_out = np.add.reduceat(b, starts)
    vf = linear_combination_variance(counts.forward, absolute_edges)
    vb = linear_combination_variance(counts.backward, absolute_edges)
    from asymmetry.core.transform.asymmetry import ratio_with_count_errors

    asymmetry, error, validity = ratio_with_count_errors(
        f_out, b_out, np.sqrt(vf), np.sqrt(vb), alpha=alpha, beta=beta
    )
    # Mean of the merged raw bins' stamps: for the slice [e0, e1) that is
    # t_start + ((e0 + e1 − 1)/2)·w — matching the fixed-mode path, where
    # merge_curve_bins() averages the same per-bin stamps.
    time = t_start + (edges[:-1] + edges[1:] - 1) * 0.5 * float(bin_width_us)
    noise = None
    if counts.forward.noise is not None:
        matrix = bin_matrix(absolute_edges, counts.forward.counts.size)
        denominator = beta * f_out + alpha * b_out
        valid = denominator != 0
        forward_derivative = np.divide(
            alpha * (1 + beta) * b_out, denominator**2, out=np.zeros_like(f_out), where=valid
        )
        backward_derivative = np.divide(
            -alpha * (1 + beta) * f_out, denominator**2, out=np.zeros_like(b_out), where=valid
        )
        noise = combine_noise(
            counts.forward.noise.transform(matrix).scale(forward_derivative),
            counts.backward.noise.transform(matrix).scale(backward_derivative),
        )
    return BinnedRatio(
        np.asarray(time, dtype=np.float64), asymmetry, error, validity, noise, absolute_edges
    )


def binned_fb_observation(counts, *, binning, alpha, beta=1.0, time_range=None):
    """Return the fractional ratio grid, errors and mathematical validity."""
    result = binned_ratio_observation(
        counts, binning=binning, alpha=alpha, beta=beta, time_range=time_range
    )
    return result.time, result.asymmetry, result.error, result.validity


def binned_fb_asymmetry(
    counts: ForwardBackwardCounts,
    *,
    binning: BinningSpec,
    alpha: float,
    beta: float = 1.0,
    time_range: tuple[float | None, float | None] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the full binned time, fractional ratio and display-error arrays.

    Known dataset producers use :func:`binned_fb_observation` to retain the
    mathematical classification for Gaussian fits alongside these full grids.
    """
    time, asymmetry, error, _ = binned_fb_observation(
        counts, binning=binning, alpha=alpha, beta=beta, time_range=time_range
    )
    return time, asymmetry, error
