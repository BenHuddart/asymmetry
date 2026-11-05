"""Detector grouping utilities.

Groups individual detector histograms into forward / backward (or custom)
groups by summing their counts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.transform.fb_counts import (
    AlignmentCrop,
    BinWindow,
    ForwardBackwardCounts,
    GroupCounts,
    count_timing,
)
from asymmetry.core.transform.t0 import (
    EFFECTIVE_DETECTOR_T0_KEY,
    common_t0_time_us,
    effective_detector_t0_bins,
    effective_t0_sources,
    require_selected_timing,
    resolve_selected_timing,
    selected_t0_reference,
)


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


def resolve_alignment(
    histograms: list[Histogram],
    *group_indices: list[int],
    detector_t0_bins: list[int] | None = None,
    common_t0_bin: int | None = None,
    sample_length: int | None = None,
    first_bin: int | None = None,
) -> AlignmentCrop:
    """Resolve selected grids, integer shifts, common observations and residuals.

    Raw k maps to local r = k + c - n_i - L. Both the integer and exact
    origins move by L*w, so removing unobserved padding preserves timestamps.
    """
    groups = [_validated_group_indices(indices, len(histograms)) for indices in group_indices]
    selected = sorted({i for indices in groups for i in indices})
    override = {} if detector_t0_bins is None else {EFFECTIVE_DETECTOR_T0_KEY: detector_t0_bins}
    bins = effective_detector_t0_bins(histograms, override)
    width = count_timing(histograms[selected[0]].bin_width, bins[selected[0]], None)[0]
    for i in selected:
        current, _ = count_timing(histograms[i].bin_width, bins[i], histograms[i].t0_time_us)
        # Widths are canonical float64 scalars on zero-origin acquisition edges.
        # One ULP admits representation rounding, not different acquisition grids.
        if abs(current - width) > max(np.spacing(current), np.spacing(width)):
            raise ValueError("Selected channels require a compatible acquisition grid.")
    common = max(bins[i] for i in selected) if common_t0_bin is None else common_t0_bin
    count_timing(width, common, None)
    shifts = {i: common - bins[i] for i in selected}
    lower = max(0, max(shifts.values()))
    upper = min(histograms[i].n_bins + shifts[i] for i in selected)
    if first_bin is not None:
        if first_bin < lower:
            raise ValueError("Reference observed support does not cover the sample start.")
        lower = first_bin
    available = upper - lower
    if available < 1 or any(histograms[i].n_bins == 0 for i in selected):
        raise ValueError("Selected channels have empty observed support.")
    if sample_length is not None and sample_length > available:
        raise ValueError(
            f"reference run has {available} aligned bins; sample requires {sample_length}."
        )
    length = available if sample_length is None else sample_length
    local_common = common - lower
    reference = selected_t0_reference(histograms, override, selected_indices=selected)
    fractions = {
        i: (
            0.0
            if histograms[i].t0_time_us is None
            else histograms[i].t0_time_us_effective - (histograms[i].t0_bin + 0.5) * width
        )
        for i in selected
    }
    residuals = tuple((i, fractions[i] - fractions[reference]) for i in selected)
    lengths = tuple(
        min(histograms[i].n_bins + shifts[i] for i in indices) - lower for indices in groups
    )
    if len(lengths) != 2:
        lengths = (min(lengths), min(lengths))
    return AlignmentCrop(
        local_common,
        tuple((i, shifts[i] - lower) for i in selected),
        lengths,
        length,
        lower,
        residuals,
    )


def sum_aligned_counts(
    histograms: list[Histogram],
    group_indices: list[int],
    alignment: AlignmentCrop,
    *,
    variance: bool = False,
) -> NDArray[np.float64]:
    """Sum acquired slices only, with identical mapping for counts and variance."""
    offsets = dict(alignment.detector_offsets)
    total = np.zeros(alignment.length, dtype=np.float64)
    for i in group_indices:
        hist = histograms[i]
        values = (
            hist.count_variance if variance and hist.count_variance is not None else hist.counts
        )
        start = -offsets[i]
        total += np.asarray(values, dtype=np.float64)[start : start + alignment.length]
    return total


def apply_grouping_aligned(
    histograms: list[Histogram],
    group_indices: list[int],
    *,
    common_t0_bin: int | None = None,
    detector_t0_bins: list[int] | None = None,
    variance: bool = False,
) -> NDArray[np.float64]:
    """Sum selected observations on their common integer-aligned support."""
    alignment = resolve_alignment(
        histograms, group_indices, common_t0_bin=common_t0_bin, detector_t0_bins=detector_t0_bins
    )
    return sum_aligned_counts(histograms, group_indices, alignment, variance=variance)


def common_t0_for_groups(
    histograms: list[Histogram],
    *group_indices: list[int],
    detector_t0_bins: list[int] | None = None,
) -> int:
    """Maximum selected effective anchor in uncropped aligned coordinates."""
    indices = _validated_group_indices(
        sorted({i for group in group_indices for i in group}), len(histograms)
    )
    override = {} if detector_t0_bins is None else {EFFECTIVE_DETECTOR_T0_KEY: detector_t0_bins}
    bins = effective_detector_t0_bins(histograms, override)
    return max(bins[i] for i in indices)


def group_reference_t0(histograms: list[Histogram], grouping: dict) -> int:
    """Reference over effective authored groups for a shared spectral grid."""
    groups = grouping.get("groups")
    if not isinstance(groups, dict):
        raise InvalidGroupingError("Group definitions must be a dictionary.", grouping)
    excluded = set(
        ChannelMembership.from_entries(grouping.get("excluded_detectors", [])).indices(
            len(histograms)
        )
    )
    indices = sorted(
        {
            i
            for gid in groups
            for i in resolve_group_indices(groups, gid, n_histograms=len(histograms))
        }
        - excluded
    )
    return common_t0_for_groups(
        histograms, indices, detector_t0_bins=effective_detector_t0_bins(histograms, grouping)
    )


def align_forward_backward(
    histograms: list[Histogram],
    forward_indices: list[int],
    backward_indices: list[int],
    detector_t0_bins: list[int] | None = None,
    *,
    common_t0_bin: int | None = None,
    sample_length: int | None = None,
    first_bin: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], AlignmentCrop]:
    """Align both groups onto their common observed support without padding."""
    crop = resolve_alignment(
        histograms,
        forward_indices,
        backward_indices,
        detector_t0_bins=detector_t0_bins,
        common_t0_bin=common_t0_bin,
        sample_length=sample_length,
        first_bin=first_bin,
    )
    return (
        sum_aligned_counts(histograms, forward_indices, crop),
        sum_aligned_counts(histograms, backward_indices, crop),
        crop,
    )


def group_counts_aligned(
    histograms: list[Histogram],
    group_indices: list[int],
    *,
    common_t0_bin: int | None = None,
    detector_t0_bins: list[int] | None = None,
    alignment: AlignmentCrop | None = None,
) -> GroupCounts:
    """Sum counts and supplied variances with one observed-bin mapping."""
    if alignment is None:
        alignment = resolve_alignment(
            histograms,
            group_indices,
            common_t0_bin=common_t0_bin,
            detector_t0_bins=detector_t0_bins,
        )
    counts = sum_aligned_counts(histograms, group_indices, alignment)
    variance = (
        sum_aligned_counts(histograms, group_indices, alignment, variance=True)
        if any(histograms[i].count_variance is not None for i in group_indices)
        else None
    )
    return GroupCounts(counts, variance)


def align_forward_backward_counts(
    histograms: list[Histogram],
    forward_indices: list[int],
    backward_indices: list[int],
    detector_t0_bins: list[int] | None = None,
    *,
    common_t0_bin: int | None = None,
    sample_length: int | None = None,
    first_bin: int | None = None,
) -> tuple[GroupCounts, GroupCounts, AlignmentCrop]:
    """Resolve the pair once and use identical observed support for its noise."""
    crop = resolve_alignment(
        histograms,
        forward_indices,
        backward_indices,
        detector_t0_bins=detector_t0_bins,
        common_t0_bin=common_t0_bin,
        sample_length=sample_length,
        first_bin=first_bin,
    )
    groups = [
        group_counts_aligned(histograms, indices, alignment=crop)
        for indices in (forward_indices, backward_indices)
    ]
    if any(group.variance is not None for group in groups):
        groups = [GroupCounts(group.counts, group.independent_variance) for group in groups]
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
    """Raw F/B observations inside the current aligned window and selected period.

    Good Events and the export header share this uncorrected observation sum.
    An unavailable request returns None while acquired counts remain available.
    """
    from asymmetry.core.utils.constants import PeriodMode

    if not histograms or not isinstance(grouping, dict):
        return None
    if grouping.get("period_mode") in (PeriodMode.GREEN_MINUS_RED, PeriodMode.GREEN_PLUS_RED):
        return None
    if (
        grouping.get("first_good_bin") is None
        or grouping.get("last_good_bin") is None
        or not grouping.get("groups")
    ):
        return None
    from asymmetry.core.io.periods import (
        held_period_histograms,
        histogram_grouping_facts,
        selected_period_index,
    )

    period_index = selected_period_index(grouping)
    if period_index is not None:
        histograms = held_period_histograms(histograms, grouping, period_index)
        grouping = histogram_grouping_facts(histograms, grouping)
    try:
        counts = group_forward_backward(histograms, grouping).counts
    except InvalidGroupingError:
        return None
    total = sum(
        float(np.sum(group.counts[counts.window.slice]))
        for group in (counts.forward, counts.backward)
    )
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


def independent_pair_indices(
    grouping: dict,
    *,
    n_histograms: int,
    resolved_indices: tuple[list[int], list[int]] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve one independent F/B pair, after exclusions, without changing it.

    Resolved-index entry points supply their effective selections explicitly.
    Identity is the declared acquired ordinal, never count or array equality.
    Auxiliary groups and other projections do not participate in this contract.
    """
    gids = tuple(
        int(grouping.get(key, default))
        for key, default in (("forward_group", 1), ("backward_group", 2))
    )
    if resolved_indices is None:
        pair = tuple(
            tuple(effective_group_indices(grouping, gid, n_histograms=n_histograms)) for gid in gids
        )
    else:
        try:
            pair = tuple(
                tuple(_validated_group_indices(indices, n_histograms))
                for indices in resolved_indices
            )
        except ValueError as exc:
            raise InvalidGroupingError(str(exc), grouping) from exc
    shared = sorted(set(pair[0]).intersection(pair[1]))
    if shared:
        raw_names = grouping.get("group_names")
        names = raw_names if isinstance(raw_names, dict) else {}
        labels = [f"{names.get(gid, names.get(str(gid), 'Group'))} ({gid})" for gid in gids]
        raise InvalidGroupingError(
            f"Forward/backward groups {labels[0]} and {labels[1]} have shared acquired "
            f"channels {format_detector_list([i + 1 for i in shared])}. "
            "Independent F/B analysis requires disjoint groups; edit the groups "
            "or exclude shared channels while keeping both sides nonempty.",
            grouping,
        )
    return pair


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


@dataclass(frozen=True)
class ForwardBackwardLayout:
    """Resolved F/B timing, windows, and balances without evaluating count sums."""

    t0_time_us: float
    bin_width_us: float
    window: BinWindow
    alignment: AlignmentCrop
    alpha: float
    beta: float
    forward_gid: int
    backward_gid: int

    @property
    def common_t0_bin(self) -> int:
        """Local integer anchor owned by the alignment."""
        return self.alignment.common_t0_bin


def grouping_balances(grouping: dict) -> tuple[float, float]:
    """Resolve finite positive F/B balances under the existing unity fallback."""
    balances = []
    for key in ("alpha", "beta"):
        try:
            value = float(grouping.get(key, 1.0))
        except (TypeError, ValueError):
            value = 1.0
        balances.append(value if np.isfinite(value) and value > 0.0 else 1.0)
    return tuple(balances)


def resolve_forward_backward_layout(
    histograms: list[Histogram],
    grouping: dict,
    *,
    resolved_indices: tuple[list[int], list[int]] | None = None,
) -> ForwardBackwardLayout:
    """Resolve a selected pair onto common observations and one typed window.

    Acquired k maps to local r = k + c - t0_i - L. A carried window policy
    resolves for this actual pair/period, rather than reusing another pair's
    scalar bounds. Legacy scalars retain their recorded coordinate meaning.
    """
    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    forward_indices, backward_indices = independent_pair_indices(
        grouping, n_histograms=len(histograms), resolved_indices=resolved_indices
    )
    alpha, beta = grouping_balances(grouping)

    try:
        policy = grouping.get("t0_policy", {})
        if (
            isinstance(policy, dict)
            and policy.get("mode") == "manual"
            and "offset_bins" in policy
            and EFFECTIVE_DETECTOR_T0_KEY not in grouping
        ):
            grouping = dict(grouping)
            resolve_selected_timing(
                histograms,
                grouping,
                mode="manual",
                offset_bins=policy["offset_bins"],
                resolved_indices=(forward_indices, backward_indices),
            )
        require_selected_timing(grouping, forward_indices + backward_indices, len(histograms))
        detector_t0_bins = effective_detector_t0_bins(histograms, grouping)
        crop = resolve_alignment(
            histograms, forward_indices, backward_indices, detector_t0_bins=detector_t0_bins
        )
        sources = effective_t0_sources(grouping, len(histograms))
        crop = replace(
            crop,
            timing_sources=tuple((i, sources[i]) for i, _ in crop.detector_offsets),
            timing_strategy=grouping.get("t0_search_strategy", ""),
        )
        window_grouping = grouping
        if "good_window_policy" in grouping:
            from asymmetry.core.project.profiles import GoodWindowPolicy, resolve_run_good_window

            first, last = resolve_run_good_window(
                GoodWindowPolicy.from_dict(grouping["good_window_policy"]),
                Run(histograms=histograms, grouping=grouping),
                grouping,
                alignment=crop,
            )
            window_grouping = dict(grouping, first_good_bin=first, last_good_bin=last)
        width = float(histograms[forward_indices[0]].bin_width)
        layout = ForwardBackwardLayout(
            common_t0_time_us(
                histograms,
                grouping,
                crop.common_t0_bin,
                selected_indices=sorted(set(forward_indices + backward_indices)),
            ),
            width,
            crop.window(window_grouping),
            crop,
            alpha,
            beta,
            forward_gid,
            backward_gid,
        )
    except ValueError as exc:
        raise InvalidGroupingError(str(exc), grouping) from exc
    return layout


def group_forward_backward(
    histograms: list[Histogram],
    grouping: dict,
    *,
    resolved_indices: tuple[list[int], list[int]] | None = None,
) -> GroupedForwardBackward:
    """Resolve the selected pair's layout, then sum its observed counts."""
    pair = independent_pair_indices(
        grouping, n_histograms=len(histograms), resolved_indices=resolved_indices
    )
    layout = resolve_forward_backward_layout(histograms, grouping, resolved_indices=pair)
    groups = [
        group_counts_aligned(histograms, list(indices), alignment=layout.alignment)
        for indices in pair
    ]
    if any(group.variance is not None for group in groups):
        groups = [GroupCounts(group.counts, group.independent_variance) for group in groups]
    return GroupedForwardBackward(
        counts=ForwardBackwardCounts(
            *groups, layout.t0_time_us, layout.bin_width_us, layout.window, layout.alignment
        ),
        alpha=layout.alpha,
        beta=layout.beta,
        forward_gid=layout.forward_gid,
        backward_gid=layout.backward_gid,
    )
