"""Detector grouping utilities.

Groups individual detector histograms into forward / backward (or custom)
groups by summing their counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, Run


def _present_indices(group_indices: list[int], n_histograms: int) -> list[int]:
    """Drop indices that fall outside ``[0, n_histograms)``.

    A grouping (e.g. a HAL-9500 preset that names the backward ring) can
    reference detectors a particular run does not contain. Filtering here keeps
    the summing helpers from indexing past the histogram list.
    """
    return [i for i in group_indices if 0 <= i < n_histograms]


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
    arrays = [histograms[i].counts for i in _present_indices(group_indices, len(histograms))]
    if not arrays:
        return np.array([], dtype=np.float64)
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
    selected = [histograms[i] for i in _present_indices(group_indices, len(histograms))]
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
    indices = _present_indices(indices, len(histograms))
    if not indices:
        return int(histograms[0].t0_bin) if histograms else 0
    return max(0, max(int(histograms[idx].t0_bin) for idx in indices))


def good_frames(grouping: dict | None, default: float = 1.0) -> float:
    """Positive ``good_frames`` from a grouping, falling back to *default*.

    ``good_frames`` is the universal dead-time normaliser (rate = counts /
    (dt · good_frames)); a missing, unparseable or non-positive value collapses
    to *default* so it can never zero the correction. Pass ``default=0.0`` (and
    treat a falsy result as "unknown", e.g. ``good_frames(g, 0.0) or None``)
    when the caller wants to fall back to a snapshot instead.
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    try:
        value = float(grouping.get("good_frames", default))
    except (TypeError, ValueError):
        return default
    return value if value > 0.0 else default


def excluded_detector_indices(grouping: dict | None) -> frozenset[int]:
    """0-based indices of detectors excluded by the grouping.

    The ``excluded_detectors`` grouping key holds 1-based detector ids
    (WiMDA ``Group2.pas`` semantics). Exclusion is applied at grouping time
    — excluded detectors are dropped from every group sum — so the raw
    histograms stay intact and no reload is needed (study divergence D10;
    WiMDA zeroes the counts in its file-read path instead).
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    raw = grouping.get("excluded_detectors")
    if not isinstance(raw, (list, tuple, set)):
        return frozenset()
    indices: set[int] = set()
    for value in raw:
        try:
            detector = int(value)
        except (TypeError, ValueError):
            continue
        if detector >= 1:
            indices.add(detector - 1)
    return frozenset(indices)


def filter_excluded_indices(indices: list[int], grouping: dict | None) -> list[int]:
    """Drop excluded detectors (0-based) from a group index list."""
    excluded = excluded_detector_indices(grouping)
    if not excluded:
        return list(indices)
    return [i for i in indices if i not in excluded]


def parse_detector_list(text: str) -> list[int]:
    """Parse a WiMDA-style detector list, e.g. ``"1,5,10-15"``.

    Separators may be commas or whitespace; ranges use ``-`` and may run in
    either direction (``"15-10"`` equals ``"10-15"``). Returns sorted unique
    1-based detector ids. Raises ``ValueError`` on unparseable fragments so
    typos surface instead of silently excluding nothing.
    """
    ids: set[int] = set()
    for fragment in str(text).replace(";", ",").replace(" ", ",").split(","):
        fragment = fragment.strip()
        if not fragment:
            continue
        if "-" in fragment:
            parts = fragment.split("-")
            if len(parts) != 2:
                raise ValueError(f"Cannot parse detector range {fragment!r}")
            start, end = (int(parts[0]), int(parts[1]))
            if start > end:
                start, end = end, start
            if start < 1:
                raise ValueError(f"Detector ids start at 1, got {fragment!r}")
            ids.update(range(start, end + 1))
        else:
            value = int(fragment)
            if value < 1:
                raise ValueError(f"Detector ids start at 1, got {fragment!r}")
            ids.add(value)
    return sorted(ids)


def format_detector_list(ids: list[int]) -> str:
    """Format detector ids compactly with ranges, e.g. ``"1,5,10-15"``."""
    unique = sorted({int(v) for v in ids})
    if not unique:
        return ""
    parts: list[str] = []
    start = previous = unique[0]
    for value in unique[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ",".join(parts)


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


def effective_group_indices(
    grouping: dict | None,
    group_id: int,
    *,
    n_histograms: int | None = None,
) -> list[int]:
    """Resolve a group's detector indices for **reduction**, exclusion applied.

    This is the single exclusion-aware chokepoint: every reduction path that
    turns a ``grouping`` + ``group_id`` into the 0-based detector indices it will
    sum MUST go through here, so detector exclusion (the ``excluded_detectors``
    grouping key) can never be silently skipped at a new call site.

    The raw decoder :func:`resolve_group_indices` is reserved for non-reduction
    uses that legitimately want every named detector regardless of exclusion
    (synthetic-run generation, NeXus writing); reduction code should not call it
    directly.

    Parameters
    ----------
    grouping
        An effective grouping dict (must carry ``groups``; may carry
        ``excluded_detectors``).
    group_id
        The forward/backward (or custom) group id to resolve.
    n_histograms
        When given, indices outside ``[0, n_histograms)`` are dropped too — a
        grouping or preset may name detectors a particular run does not contain.
    """
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    if not isinstance(groups, dict):
        return []
    indices = resolve_group_indices(groups, group_id)
    if n_histograms is not None:
        indices = _present_indices(indices, n_histograms)
    return filter_excluded_indices(indices, grouping)


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
    n = len(histograms)
    forward_indices = effective_group_indices(grouping, forward_gid, n_histograms=n)
    backward_indices = effective_group_indices(grouping, backward_gid, n_histograms=n)
    if not forward_indices or not backward_indices:
        raise ValueError(
            "Forward/backward groups do not reference any detectors present in this run "
            "(after detector exclusion)."
        )

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
