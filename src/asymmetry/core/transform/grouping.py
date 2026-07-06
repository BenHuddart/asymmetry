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


#: Grouping-dict key carrying the T0Policy-resolved effective per-detector t0
#: bins (0-based, one per histogram). Distinct from the file-derived
#: ``detector_t0_bins`` per-run fact so a *manual* policy can shift alignment
#: without touching the file values. Absent for the ``from_file`` default.
EFFECTIVE_DETECTOR_T0_KEY = "effective_detector_t0_bins"


def detector_t0_overrides(grouping: dict | None, n_histograms: int) -> list[int] | None:
    """Extract policy-resolved per-detector t0 overrides from a grouping dict.

    Reads :data:`EFFECTIVE_DETECTOR_T0_KEY` (a *manual* T0Policy writes it) and
    returns it as an int list when it lines up with the histogram count, else
    ``None`` (the ``from_file`` default aligns on ``Histogram.t0_bin``).
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    raw = grouping.get(EFFECTIVE_DETECTOR_T0_KEY)
    if not isinstance(raw, (list, tuple)) or len(raw) != n_histograms:
        return None
    try:
        return [int(v) for v in raw]
    except (TypeError, ValueError):
        return None


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
) -> NDArray[np.float64]:
    """Sum detector counts after aligning each detector's own ``t0_bin``.

    ISIS NeXus files normally provide one common ``t0`` for every detector, so
    :func:`apply_grouping` is sufficient. PSI BIN/MDU data can carry different
    ``t0`` values per detector. This helper shifts each selected detector so
    that its local ``t0_bin`` lands on ``common_t0_bin`` before summing.

    ``detector_t0_bins`` optionally overrides each detector's alignment t0
    (indexed 0-based across the full histogram list) — the non-destructive
    route a *manual* T0Policy uses to shift the effective t0 without rewriting
    ``Histogram.t0_bin``. When ``None``, each histogram's own ``t0_bin`` is used.
    """
    present = _present_indices(group_indices, len(histograms))
    selected = [(i, histograms[i]) for i in present]
    if not selected:
        return np.array([], dtype=np.float64)

    if common_t0_bin is None:
        common_t0_bin = max(
            0, max(_detector_t0(histograms, i, detector_t0_bins) for i, _ in selected)
        )
    common_t0_bin = max(0, int(common_t0_bin))

    shifted: list[NDArray[np.float64]] = []
    for i, hist in selected:
        counts = np.asarray(hist.counts, dtype=np.float64)
        offset = common_t0_bin - _detector_t0(histograms, i, detector_t0_bins)
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
    detector_t0_bins: list[int] | None = None,
) -> int:
    """Return a common t0 suitable for comparing multiple detector groups.

    ``detector_t0_bins`` optionally overrides the per-detector t0 (see
    :func:`apply_grouping_aligned`) so a *manual* T0Policy can shift the common
    t0 without mutating ``Histogram.t0_bin``.
    """
    indices = sorted({idx for group in group_indices for idx in group})
    indices = _present_indices(indices, len(histograms))
    if not indices:
        if not histograms:
            return 0
        return _detector_t0(histograms, 0, detector_t0_bins)
    return max(0, max(_detector_t0(histograms, idx, detector_t0_bins) for idx in indices))


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

    Mirrors WiMDA's logbook "good events" (``LogbookUnit.pas``): the sum of
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

    def _safe_int(raw: object) -> int | None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    first_good = _safe_int(grouping.get("first_good_bin"))
    last_good = _safe_int(grouping.get("last_good_bin"))
    if first_good is None or last_good is None:
        return None
    lo = max(0, min(first_good, last_good))
    hi = max(first_good, last_good)

    groups_raw = grouping.get("groups")
    if not isinstance(groups_raw, dict):
        return None
    # Coerce group ids to int so a JSON-restored grouping (string keys) matches
    # the int forward/backward ids just as an in-memory grouping does.
    groups_by_id: dict[int, object] = {}
    for key, vals in groups_raw.items():
        gid = _safe_int(key)
        if gid is not None:
            groups_by_id[gid] = vals
    f_gid = _safe_int(grouping.get("forward_group"))
    b_gid = _safe_int(grouping.get("backward_group"))
    selected: list[object] = []
    if f_gid is not None and f_gid in groups_by_id:
        selected.extend(groups_by_id[f_gid])
    if b_gid is not None and b_gid in groups_by_id:
        selected.extend(groups_by_id[b_gid])
    if not selected:
        return None

    n_hist = len(histograms)
    total = 0.0
    for det in selected:
        det_idx = _safe_int(det)
        if det_idx is None:
            continue
        hist_idx = det_idx - 1
        if hist_idx < 0 or hist_idx >= n_hist:
            continue
        counts = np.asarray(histograms[hist_idx].counts, dtype=float)
        if counts.size == 0:
            continue
        hi_clamped = min(hi, counts.size - 1)
        if hi_clamped >= lo:
            total += float(np.sum(counts[lo : hi_clamped + 1]))
    return total if total > 0 else None


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


def group_detectors_outside_run(groups: dict, group_id: int, n_histograms: int) -> list[int]:
    """Return the 1-based detector numbers in *group_id* absent from the run.

    A grouping (typically an instrument preset or a saved profile) may name
    detectors a particular run does not contain — e.g. a full HAL-9500 preset
    referencing the backward ring (detectors 10-17) applied to a forward-ring-only
    ``.mdu`` file with only nine histograms. Reduction rejects such a group
    (:func:`effective_group_indices` would resolve indices past the last
    histogram), so this helper names the offending detectors for a clear,
    provenance-preserving diagnostic instead of a silent skip.

    Returns the sorted detector numbers whose 0-based index is ``>= n_histograms``
    (or ``< 1``), empty when every referenced detector is present.
    """
    if n_histograms is None or n_histograms < 0:
        return []
    entries = groups.get(group_id)
    if entries is None:
        entries = groups.get(str(group_id))
    if not isinstance(entries, list):
        return []
    missing: set[int] = set()
    for value in entries:
        detector = value[0] if isinstance(value, (list, tuple)) and value else value
        try:
            number = int(detector)
        except (TypeError, ValueError):
            continue
        if number < 1 or number - 1 >= n_histograms:
            missing.add(number)
    return sorted(missing)


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

    detector_t0_bins = detector_t0_overrides(grouping, len(histograms))
    common_t0 = common_t0_for_groups(
        histograms, forward_indices, backward_indices, detector_t0_bins=detector_t0_bins
    )
    forward = apply_grouping_aligned(
        histograms, forward_indices, common_t0_bin=common_t0, detector_t0_bins=detector_t0_bins
    )
    backward = apply_grouping_aligned(
        histograms, backward_indices, common_t0_bin=common_t0, detector_t0_bins=detector_t0_bins
    )
    return GroupedForwardBackward(
        forward=forward,
        backward=backward,
        common_t0=int(common_t0),
        alpha=alpha,
        forward_gid=forward_gid,
        backward_gid=backward_gid,
    )
