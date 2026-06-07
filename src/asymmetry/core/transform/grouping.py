"""Detector grouping utilities.

Groups individual detector histograms into forward / backward (or custom)
groups by summing their counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, Run


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


def resolve_group_indices(groups: dict, group_id: int) -> list[int]:
    """Return zero-based detector indices for *group_id*.

    Grouping entries are 1-based detector numbers (matching the convention in
    ``fourier.grouped`` / ``grouped_time_domain``); they are converted to
    zero-based histogram indices here.  Group keys may be ``int`` or ``str``.

    This is the single source of truth shared by the time-domain F-B asymmetry
    representation and the time-integral observable, so the two cannot decode a
    run's grouping differently.
    """
    entries = groups.get(group_id)
    if entries is None:
        entries = groups.get(str(group_id))
    if not isinstance(entries, list):
        return []
    indices: list[int] = []
    for value in entries:
        detector = value[0] if isinstance(value, (list, tuple)) and value else value
        try:
            indices.append(max(0, int(detector) - 1))
        except (TypeError, ValueError):
            continue
    return indices


def effective_grouping(run: Run, grouping_ref: dict | None = None) -> dict:
    """Merge a run's grouping with an optional override (override wins).

    ``grouping_ref`` is the recipe-level grouping override (forward/backward
    group ids, alpha, good-bin window, …).  Returning a fresh dict keeps the
    run's stored grouping immutable.
    """
    base = dict(run.grouping) if isinstance(run.grouping, dict) else {}
    if isinstance(grouping_ref, dict):
        base.update(grouping_ref)
    return base


@dataclass
class GroupedForwardBackward:
    """Forward/backward grouped counts plus the metadata to reduce them."""

    forward: NDArray[np.float64]
    backward: NDArray[np.float64]
    common_t0: int
    alpha: float
    forward_gid: int
    backward_gid: int


def group_forward_backward(
    histograms: list[Histogram],
    grouping: dict,
) -> GroupedForwardBackward:
    """Form aligned forward/backward groups from a (effective) grouping dict.

    This is the shared core of the time-domain F-B asymmetry and the
    time-integral observable: it resolves the forward/backward group ids to
    detector indices, aligns each detector to a common ``t0``, sums the groups,
    and reads the balance ``alpha`` (leniently, defaulting to ``1.0``).  Callers
    own the good-bin window, the time axis, ``compute_asymmetry``, and any
    rebinning, so the two reductions agree on grouping by construction.

    Raises :class:`ValueError` when the grouping lacks a ``groups`` definition or
    the forward/backward groups reference no detectors.
    """
    groups = grouping.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            "Forward-backward grouping requires a detector grouping definition "
            "(grouping['groups'])."
        )

    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    forward_indices = resolve_group_indices(groups, forward_gid)
    backward_indices = resolve_group_indices(groups, backward_gid)
    if not forward_indices or not backward_indices:
        raise ValueError("Forward/backward groups do not reference any detectors.")

    try:
        alpha = float(grouping.get("alpha", 1.0))
    except (TypeError, ValueError):
        alpha = 1.0
    # A degenerate balance (non-finite or non-positive) is meaningless; fall back
    # to 1.0 rather than propagate NaN/0 into the asymmetry.
    if not np.isfinite(alpha) or alpha <= 0.0:
        alpha = 1.0

    common_t0 = common_t0_for_groups(histograms, forward_indices, backward_indices)
    forward = apply_grouping_aligned(histograms, forward_indices, common_t0_bin=common_t0)
    backward = apply_grouping_aligned(histograms, backward_indices, common_t0_bin=common_t0)
    return GroupedForwardBackward(
        forward=forward,
        backward=backward,
        common_t0=int(common_t0),
        alpha=alpha,
        forward_gid=forward_gid,
        backward_gid=backward_gid,
    )
