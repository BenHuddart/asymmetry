"""Rebinning utilities for time-domain data.

Three binning modes (WiMDA ``Group.pas:1411–1418``, textbook Fig. 15.7):

- **fixed** — every output bin merges the same number of raw bins
  (:func:`rebin`, the ``bunching_factor`` grouping key);
- **variable** — output width grows exponentially from ``bin0_us`` at t = 0
  to ``bin10_us`` at t = 10 µs (WiMDA's formula folds 1/(10·λ_µ) into the
  exponent with a rounded constant, 0.22; the exact law is implemented here —
  study divergence D8, < 0.6 % difference over a 32 µs window);
- **constant_error** — output width grows as exp(t/τ_µ), so each output bin
  holds roughly equal counts and the Poisson error per bin stays flat while
  the polarization varies slowly.

All modes are display/fit-input transformations on the reduced data: raw
histograms are never modified (provenance invariant). Non-fixed modes bin the
*counts* and then form the asymmetry per output bin — at late times the raw
bins hold few or zero counts, where a weighted mean of per-bin asymmetry
ratios is undefined; summed counts stay exactly Poisson.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.transform.asymmetry import (
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
)
from asymmetry.core.utils.constants import MUON_LIFETIME_US

BINNING_MODES = ("fixed", "variable", "constant_error")


def rebin(
    time: NDArray[np.float64],
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    factor: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Rebin data by combining *factor* consecutive bins.

    Parameters
    ----------
    time, values, errors
        Equal-length arrays of the original data.
    factor
        Number of bins to merge into one.

    Returns
    -------
    (time_rebinned, values_rebinned, errors_rebinned)
    """
    if factor < 1:
        raise ValueError("Rebinning factor must be >= 1")
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

    Unlike :func:`rebin`, which is value-domain (mean of values, errors in
    quadrature ÷ factor), this merges neighbouring bins by *summing* the
    counts while moving the time coordinate to the centre of the wider
    effective bin — the right combiner for grouped count-like signals
    (fit-input traces, grouped Fourier inputs). The two are deliberately
    distinct: a value-domain mean would corrupt Poisson statistics on counts.

    ``factor`` is clamped to ``max(1, int(factor))``; a no-op factor or a
    series shorter than the factor returns ``float64`` copies of the inputs.
    """
    bunch_factor = max(1, int(factor))
    if bunch_factor <= 1 or counts.size < bunch_factor:
        return np.asarray(time, dtype=np.float64), np.asarray(counts, dtype=np.float64)

    n_new = counts.size // bunch_factor
    trimmed = n_new * bunch_factor
    rebinned_time = (
        np.asarray(time[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).mean(axis=1)
    )
    rebinned_counts = (
        np.asarray(counts[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).sum(axis=1)
    )
    return rebinned_time, rebinned_counts


def resolve_binning_mode(grouping: dict[str, Any] | None) -> tuple[str, float, float]:
    """Return ``(mode, bin0_us, bin10_us)`` from a grouping payload.

    Defaults mirror WiMDA's ``InitializeGlobalVars``: bin0 = 0.08 µs,
    bin10 = 0.25 µs. Unknown modes fall back to ``fixed``.
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    mode = str(grouping.get("binning_mode", "fixed")).strip().lower()
    if mode not in BINNING_MODES:
        mode = "fixed"

    def _positive(key: str, default: float) -> float:
        try:
            value = float(grouping.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) and value > 0.0 else default

    return mode, _positive("bin0_us", 0.08), _positive("bin10_us", 0.25)


def _target_width_us(mode: str, t_us: float, bin0_us: float, bin10_us: float) -> float:
    if mode == "variable":
        return bin0_us * (bin10_us / bin0_us) ** (max(t_us, 0.0) / 10.0)
    return bin0_us * np.exp(max(t_us, 0.0) / MUON_LIFETIME_US)


def binning_slice_edges(
    n_bins: int,
    *,
    mode: str,
    bin_width_us: float,
    t_start_us: float,
    bin0_us: float = 0.08,
    bin10_us: float = 0.25,
) -> NDArray[np.intp]:
    """Output-bin boundaries (indices 0..n_bins) for a non-fixed binning mode.

    Raw bins are accumulated until the running edge passes the target width
    at the output bin's start time — WiMDA's ``Regroup`` loop, snapped to
    integer raw-bin boundaries. ``t_start_us`` is the time (relative to t0)
    of the window's first raw-bin edge; widths before t = 0 use the t = 0
    target.
    """
    if mode not in ("variable", "constant_error"):
        raise ValueError(f"binning_slice_edges expects a non-fixed mode, got {mode!r}")
    if bin_width_us <= 0.0:
        raise ValueError("bin_width_us must be positive")
    edges = [0]
    i = 0
    while i < n_bins:
        t1 = t_start_us + i * bin_width_us
        width = _target_width_us(mode, t1, bin0_us, bin10_us)
        step = max(1, int(np.ceil(width / bin_width_us - 1e-9)))
        i = min(i + step, n_bins)
        edges.append(i)
    return np.asarray(edges, dtype=np.intp)


def binned_fb_asymmetry(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    grouping: dict[str, Any] | None,
    common_t0: int,
    bin_width_us: float,
    alpha: float,
    first_good_bin: int,
    last_good_bin: int,
    forward_error: NDArray[np.float64] | None = None,
    backward_error: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reduce grouped counts to an asymmetry curve with variable-width bins.

    Counts (and, when supplied, count variances) are summed onto the output
    bins first and the asymmetry formed per output bin — the counts-then-
    ratio order all reference programs use. Returns ``(time, asymmetry,
    error)`` in µs relative to t0; the asymmetry is fractional (callers
    scale to percent). Output times are the mean of the merged raw bins'
    reduction time stamps ``(k − t0)·w`` — the same convention the
    fixed-mode path and :func:`rebin` use, so switching binning modes never
    shifts the time axis.
    """
    mode, bin0_us, bin10_us = resolve_binning_mode(grouping)
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    n = min(f.size, b.size)
    lo = max(0, int(first_good_bin))
    hi = min(n - 1, int(last_good_bin))
    if lo > hi:
        lo, hi = 0, n - 1
    f = f[lo : hi + 1]
    b = b[lo : hi + 1]
    t_start = (lo - int(common_t0)) * float(bin_width_us)

    if mode == "fixed":
        raise ValueError("binned_fb_asymmetry handles variable/constant_error modes only")
    edges = binning_slice_edges(
        f.size,
        mode=mode,
        bin_width_us=float(bin_width_us),
        t_start_us=t_start,
        bin0_us=bin0_us,
        bin10_us=bin10_us,
    )
    starts = edges[:-1]
    f_out = np.add.reduceat(f, starts)
    b_out = np.add.reduceat(b, starts)
    if forward_error is not None and backward_error is not None:
        ef = np.asarray(forward_error, dtype=np.float64)[lo : hi + 1]
        eb = np.asarray(backward_error, dtype=np.float64)[lo : hi + 1]
        ef_out = np.sqrt(np.add.reduceat(ef * ef, starts))
        eb_out = np.sqrt(np.add.reduceat(eb * eb, starts))
        asymmetry, error = compute_asymmetry_with_count_errors(
            f_out, b_out, ef_out, eb_out, alpha=alpha
        )
    else:
        asymmetry, error = compute_asymmetry(f_out, b_out, alpha=alpha)
    # Mean of the merged raw bins' time stamps (k − t0)·w: for the slice
    # [e0, e1) that is ((e0 + e1 − 1)/2)·w — matching the fixed-mode path,
    # where rebin() averages the same left-edge stamps.
    time = t_start + (edges[:-1] + edges[1:] - 1) * 0.5 * float(bin_width_us)
    return np.asarray(time, dtype=np.float64), asymmetry, error
