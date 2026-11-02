"""Detector grouping utilities.

Groups individual detector histograms into forward / backward (or custom)
groups by summing their counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.transform.fb_counts import (
    AlignmentCrop,
    ForwardBackwardCounts,
    GroupCounts,
    good_window,
)
from asymmetry.core.transform.t0 import common_t0_time_us, effective_detector_t0_bins
from asymmetry.core.utils.perf import perf_timer


class InvalidGroupingError(ValueError):
    """An unavailable request retained verbatim so project/UI boundaries can edit it."""

    def __init__(self, message: str, grouping: dict):
        super().__init__(message)
        self.grouping = grouping


@dataclass(frozen=True)
class ChannelMembership:
    """Unique acquired-channel ordinals; physical detector identity is separate."""

    ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ids, tuple):
            raise ValueError("Detector membership must be an immutable tuple.")
        for value in self.ids:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"Detector IDs must be positive integers; got {value!r}.")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"Detector membership contains duplicate IDs: {self.ids!r}.")

    @classmethod
    def from_entries(cls, entries: object) -> ChannelMembership:
        """Decode structured membership without coercion or accidental weights."""
        if not isinstance(entries, (list, tuple)):
            raise ValueError("Detector membership must be a list of integer IDs.")
        return cls(tuple(entries))

    def indices(self, n_histograms: int | None = None) -> tuple[int, ...]:
        """Resolve declared ordinals against the acquired channels, without trimming."""
        if n_histograms is not None:
            missing = [d for d in self.ids if d > n_histograms]
            if missing:
                raise ValueError(
                    f"Detector(s) {format_detector_list(missing)} do not exist in this "
                    f"{n_histograms}-detector run; choose a matching preset or edit the group."
                )
        return tuple(d - 1 for d in self.ids)


def _group_membership(groups: dict, group_id: int) -> ChannelMembership:
    entries = groups.get(group_id, groups.get(str(group_id)))
    if entries is None:
        raise ValueError(f"Group {group_id!r} is not defined.")
    membership = ChannelMembership.from_entries(entries)
    if not membership.ids:
        raise ValueError(f"Group {group_id!r} needs at least one detector.")
    return membership


def _validated_group_indices(group_indices: list[int], n_histograms: int) -> list[int]:
    """Validate a public zero-based selection before accessing count observations."""
    membership = ChannelMembership.from_entries(
        [i + 1 if isinstance(i, Integral) and not isinstance(i, bool) else i for i in group_indices]
    )
    if not membership.ids:
        raise ValueError("A group needs at least one detector.")
    return list(membership.indices(n_histograms))


def _detector_t0(
    histograms: list[Histogram],
    index: int,
    detector_t0_bins: list[int] | None,
) -> int:
    """The t0 bin to align detector ``index`` to.

    Prefers a per-detector override from ``detector_t0_bins`` (the effective
    t0 a T0Policy resolved for this run without mutating the histograms) and
    falls back to the histogram's own file-derived ``t0_bin``. The override
    lets a *manual* t0 policy shift the alignment point non-destructively —
    the loaded ``Histogram.t0_bin`` values stay as read from the file.
    """
    if detector_t0_bins is not None and 0 <= index < len(detector_t0_bins):
        override = detector_t0_bins[index]
        if override is not None:
            return int(override)
    return int(histograms[index].t0_bin)


def apply_grouping_aligned(
    histograms: list[Histogram],
    group_indices: list[int],
    *,
    common_t0_bin: int | None = None,
    detector_t0_bins: list[int] | None = None,
    variance: bool = False,
) -> NDArray[np.float64]:
    """Sum detector counts after aligning each detector's own ``t0_bin``.

    ISIS NeXus files normally provide one common ``t0`` for every detector.
    PSI BIN/MDU data can carry different
    ``t0`` values per detector. This helper shifts each selected detector so
    that its local ``t0_bin`` lands on ``common_t0_bin`` before summing.

    ``detector_t0_bins`` optionally overrides each detector's alignment t0
    (indexed 0-based across the full histogram list) — the non-destructive
    route a *manual* T0Policy uses to shift the effective t0 without rewriting
    ``Histogram.t0_bin``. When ``None``, each histogram's own ``t0_bin`` is used.
    With ``variance=True``, sum supplied independent variances, using measured
    counts for detectors without an explicit variance. Padding contributes zero.
    """
    with perf_timer("core.grouping.apply_aligned", n_indices=len(group_indices)) as perf:
        present = _validated_group_indices(group_indices, len(histograms))
        selected = [(i, histograms[i]) for i in present]
        if common_t0_bin is None:
            common_t0_bin = max(
                0, max(_detector_t0(histograms, i, detector_t0_bins) for i, _ in selected)
            )
        common_t0_bin = max(0, int(common_t0_bin))

        # Each detector, once aligned, contributes an array of length
        # ``len(counts) + offset`` (offset < 0 trims the head, offset > 0
        # zero-pads it), so the summed length is known from the offsets alone —
        # no need to materialise the shifted arrays to find it. Accumulate each
        # detector's aligned slice directly into one buffer instead of building
        # a list of full-length copies and summing them: same float64 additions
        # in the same order (bit-identical result), but only the accumulator is
        # allocated rather than one padded copy per detector in the group.
        offsets = [
            common_t0_bin - _detector_t0(histograms, i, detector_t0_bins) for i, _ in selected
        ]
        min_len = min(len(hist.counts) + offset for (_, hist), offset in zip(selected, offsets))

        total = np.zeros(min_len, dtype=np.float64)
        for (_, hist), offset in zip(selected, offsets):
            values = (
                hist.count_variance if variance and hist.count_variance is not None else hist.counts
            )
            counts = np.asarray(values, dtype=np.float64)
            if offset <= 0:
                # Aligned array is ``counts[-offset:]`` (``counts`` itself when
                # offset == 0); add its first ``min_len`` samples.
                src_start = -offset
                total += counts[src_start : src_start + min_len]
            elif min_len > offset:
                # Aligned array is ``offset`` leading zeros then ``counts``; the
                # zeros contribute nothing, so add ``counts`` from ``offset`` on.
                total[offset:min_len] += counts[: min_len - offset]
            # offset >= min_len: the aligned array is all zeros within the
            # summed window, so it contributes nothing.
        perf.detail(bins=min_len)
        return total


def common_t0_for_groups(
    histograms: list[Histogram],
    *group_indices: list[int],
    detector_t0_bins: list[int] | None = None,
) -> int:
    """Return a common t0 over a nonempty selection of acquired channels.

    ``detector_t0_bins`` optionally overrides the per-detector t0 (see
    :func:`apply_grouping_aligned`) so a *manual* T0Policy can shift the common
    t0 without mutating ``Histogram.t0_bin``.
    """
    indices = sorted({idx for group in group_indices for idx in group})
    indices = _validated_group_indices(indices, len(histograms))
    return max(0, max(_detector_t0(histograms, idx, detector_t0_bins) for idx in indices))


def group_reference_t0(histograms: list[Histogram], grouping: dict) -> int:
    """Use every authored group for the shared spectral alignment reference.

    Exclusions do not move this reference. Invalid auxiliary groups remain
    unavailable instead of changing which acquired channels define the grid.
    """
    groups = grouping.get("groups")
    if not isinstance(groups, dict):
        raise InvalidGroupingError("Group definitions must be a dictionary.", grouping)
    indices = [resolve_group_indices(groups, gid, n_histograms=len(histograms)) for gid in groups]
    return common_t0_for_groups(
        histograms, *indices, detector_t0_bins=effective_detector_t0_bins(histograms, grouping)
    )


def align_forward_backward(
    histograms: list[Histogram],
    forward_indices: list[int],
    backward_indices: list[int],
    detector_t0_bins: list[int] | None = None,
    *,
    common_t0_bin: int | None = None,
    sample_length: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], AlignmentCrop]:
    """Align detector t0 bins, then crop both sums to their common terminal bin.

    Each group ends at its shortest shifted detector. The pair then ends at
    the shorter group, recording the original group lengths and every shift.
    An explicit common t0 and sample length align and crop a reference onto
    the sample's grid; the reference must cover the full sample record.
    """
    common = (
        common_t0_bin
        if common_t0_bin is not None
        else common_t0_for_groups(
            histograms, forward_indices, backward_indices, detector_t0_bins=detector_t0_bins
        )
    )
    forward, backward = (
        apply_grouping_aligned(
            histograms, indices, common_t0_bin=common, detector_t0_bins=detector_t0_bins
        )
        for indices in (forward_indices, backward_indices)
    )
    lengths = (forward.size, backward.size)
    available = min(lengths)
    if sample_length is not None and sample_length > available:
        raise ValueError(
            f"reference run has {available} aligned bins; sample requires {sample_length}."
        )
    length = available if sample_length is None else sample_length
    selected = sorted(set(forward_indices + backward_indices))
    crop = AlignmentCrop(
        common,
        tuple(
            (index, common - _detector_t0(histograms, index, detector_t0_bins))
            for index in selected
        ),
        lengths,
        length,
    )
    return forward[:length], backward[:length], crop


def _aligned_count_variance(histograms, indices, **alignment):
    """Keep Poisson inference only when every contributing detector is raw."""
    if not any(
        histograms[i].count_variance is not None
        for i in _validated_group_indices(indices, len(histograms))
    ):
        return None
    return apply_grouping_aligned(histograms, indices, variance=True, **alignment)


def group_counts_aligned(
    histograms: list[Histogram],
    group_indices: list[int],
    *,
    common_t0_bin: int | None = None,
    detector_t0_bins: list[int] | None = None,
) -> GroupCounts:
    """Sum counts and supplied variances on the same detector alignment grid."""
    options = dict(common_t0_bin=common_t0_bin, detector_t0_bins=detector_t0_bins)
    counts = apply_grouping_aligned(histograms, group_indices, **options)
    return GroupCounts(counts, _aligned_count_variance(histograms, group_indices, **options))


def align_forward_backward_counts(
    histograms: list[Histogram],
    forward_indices: list[int],
    backward_indices: list[int],
    detector_t0_bins: list[int] | None = None,
    *,
    common_t0_bin: int | None = None,
    sample_length: int | None = None,
) -> tuple[GroupCounts, GroupCounts, AlignmentCrop]:
    """Align both count groups and crop their noise onto the same terminal bin."""
    forward, backward, crop = align_forward_backward(
        histograms,
        forward_indices,
        backward_indices,
        detector_t0_bins,
        common_t0_bin=common_t0_bin,
        sample_length=sample_length,
    )
    groups = []
    for indices, counts in ((forward_indices, forward), (backward_indices, backward)):
        variance = _aligned_count_variance(
            histograms,
            indices,
            common_t0_bin=crop.common_t0_bin,
            detector_t0_bins=detector_t0_bins,
        )
        groups.append(GroupCounts(counts, None if variance is None else variance[: crop.length]))
    return groups[0], groups[1], crop


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


def good_event_count(
    histograms: list[Histogram] | None,
    grouping: dict | None,
) -> float | None:
    """Total raw counts in the good-bin range over the forward+backward groups.

    Mirrors WiMDA's logbook "good events": the sum of
    detector counts between ``first_good_bin`` and ``last_good_bin`` (inclusive)
    across the detectors of the forward and backward groups. Returns ``None``
    when the grouping lacks a good-bin range or named forward/backward groups
    (e.g. an ungrouped run), so callers can fall back to a total-count display.

    Detector ids in ``grouping['groups']`` are 1-based (WiMDA convention), so
    histogram index = id − 1. This is the single source of truth for the
    good-range event total: the data-browser "Good Events" column and the
    export ``events_grouped`` header both call it, so they agree by construction.
    """
    if not histograms or not isinstance(grouping, dict):
        return None

    first_good = grouping.get("first_good_bin")
    last_good = grouping.get("last_good_bin")
    if first_good is None or last_good is None or not grouping.get("groups"):
        return None
    try:
        lo = max(0, min(int(first_good), int(last_good)))
        hi = max(int(first_good), int(last_good))
        selected = [
            index
            for key, default in (("forward_group", 1), ("backward_group", 2))
            for index in effective_group_indices(
                grouping, grouping.get(key, default), n_histograms=len(histograms)
            )
        ]
    except (TypeError, ValueError):
        return None
    total = sum(float(np.sum(histograms[i].counts[lo : hi + 1])) for i in selected)
    return total if total > 0 else None


def excluded_detector_indices(grouping: dict | None) -> frozenset[int]:
    """0-based indices of detectors excluded by the grouping.

    The ``excluded_detectors`` grouping key holds 1-based detector ids
    (WiMDA's detector-exclusion convention). Exclusion is applied at grouping time
    — excluded detectors are dropped from every group sum — so the raw
    histograms stay intact and no reload is needed (study divergence D10;
    WiMDA zeroes the counts in its file-read path instead).
    """
    raw = grouping.get("excluded_detectors", []) if grouping is not None else []
    return frozenset(ChannelMembership.from_entries(raw).indices())


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
        parts = fragment.split("-")
        if len(parts) > 2 or not all(part.isdecimal() for part in parts):
            raise ValueError(f"Cannot read {fragment!r} as a detector id or range")
        start, end = sorted(int(part) for part in (parts[0], parts[-1]))
        if start < 1:
            raise ValueError(f"Detector ids start at 1, got {fragment!r}")
        ids.update(range(start, end + 1))
    return sorted(ids)


def parse_group_detectors(text: str, n_detectors: int) -> list[int]:
    """Parse one group's typed detector list for a run of *n_detectors*.

    :func:`parse_detector_list`, plus the two rules a group adds: it needs at
    least one detector, and every id must exist on the run. Raises
    ``ValueError`` with a message fit to show the user.
    """
    ids = parse_detector_list(text)
    membership = _group_membership({1: ids}, 1)
    membership.indices(n_detectors)
    return list(membership.ids)


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


def resolve_group_indices(
    groups: dict, group_id: int, *, n_histograms: int | None = None
) -> list[int]:
    """Resolve unique authored 1-based channel IDs (integer or JSON group keys)."""
    return list(_group_membership(groups, group_id).indices(n_histograms))


def group_detectors_outside_run(groups: dict, group_id: int, n_histograms: int) -> list[int]:
    """Name absent authored channels for diagnostics; no reduction trims them."""
    return [d for d in _group_membership(groups, group_id).ids if d > n_histograms]


def effective_group_indices(
    grouping: dict | None,
    group_id: int,
    *,
    n_histograms: int | None = None,
) -> list[int]:
    """Validate authored membership, then subtract explicit acquired-channel exclusions.

    Missing preset channels remain a mismatch even when excluded. A required
    group with no effective members cannot be reduced. Raw counts are untouched.
    """
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    try:
        if not isinstance(groups, dict):
            raise ValueError("Group definitions must be a dictionary.")
        indices = _group_membership(groups, group_id).indices(n_histograms)
        excluded = ChannelMembership.from_entries(grouping.get("excluded_detectors", []))
        excluded_indices = frozenset(excluded.indices(n_histograms))
        effective = [i for i in indices if i not in excluded_indices]
        if not effective:
            raise ValueError(f"Group {group_id!r} has no detectors after exclusion.")
    except ValueError as exc:
        raise InvalidGroupingError(str(exc), grouping) from exc
    return effective


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


def group_names(run: Run) -> dict[int, str]:
    """Return ``{group_id: display name}`` for a run's grouping.

    Reads the optional ``group_names`` map from the run's grouping payload
    (keyed by ``int`` or stringified id), falling back to ``"Group <id>"`` for
    any group without an explicit name. Returns ``{}`` when the run has no
    group definitions.
    """
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    groups = grouping.get("groups") if isinstance(grouping, dict) else None
    if not isinstance(groups, dict):
        return {}
    raw_names = grouping.get("group_names")
    names = raw_names if isinstance(raw_names, dict) else {}
    resolved: dict[int, str] = {}
    for raw_id in groups:
        try:
            gid = int(raw_id)
        except (TypeError, ValueError):
            continue
        name = names.get(gid, names.get(str(gid)))
        resolved[gid] = str(name) if name is not None else f"Group {gid}"
    return resolved


@dataclass
class GroupedForwardBackward:
    """Forward/backward grouped counts plus the metadata to reduce them."""

    counts: ForwardBackwardCounts
    alpha: float
    forward_gid: int
    backward_gid: int
    beta: float = 1.0

    @property
    def forward(self) -> NDArray[np.float64]:
        return self.counts.forward.counts

    @property
    def backward(self) -> NDArray[np.float64]:
        return self.counts.backward.counts

    @property
    def common_t0(self) -> int:
        return self.counts.common_t0_bin


def group_forward_backward(
    histograms: list[Histogram],
    grouping: dict,
) -> GroupedForwardBackward:
    """Form aligned forward/backward groups from a (effective) grouping dict.

    This is the shared core of the time-domain F-B asymmetry and the
    time-integral observable: it resolves the forward/backward group ids to
    detector indices, aligns each detector to a common ``t0``, sums the groups,
    and reads the balances ``alpha`` and ``beta`` (leniently, defaulting to
    ``1.0``).  Callers
    consume the validated window and time grid in ``counts`` for asymmetry
    and rebinning, so reductions share the same aligned coordinates.

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
    try:
        alpha = float(grouping.get("alpha", 1.0))
    except (TypeError, ValueError):
        alpha = 1.0
    # A degenerate balance (non-finite or non-positive) is meaningless; fall back
    # to 1.0 rather than propagate NaN/0 into the asymmetry.
    if not np.isfinite(alpha) or alpha <= 0.0:
        alpha = 1.0
    try:
        beta = float(grouping.get("beta", 1.0))
    except (TypeError, ValueError):
        beta = 1.0
    if not np.isfinite(beta) or beta <= 0.0:
        beta = 1.0

    detector_t0_bins = effective_detector_t0_bins(histograms, grouping)
    forward, backward, crop = align_forward_backward_counts(
        histograms, forward_indices, backward_indices, detector_t0_bins
    )
    width = float(histograms[0].bin_width)
    counts = ForwardBackwardCounts(
        forward,
        backward,
        common_t0_time_us(histograms, grouping, crop.common_t0_bin),
        width,
        good_window(grouping, crop.length),
        crop,
    )
    return GroupedForwardBackward(
        counts=counts,
        alpha=alpha,
        forward_gid=forward_gid,
        backward_gid=backward_gid,
        beta=beta,
    )
