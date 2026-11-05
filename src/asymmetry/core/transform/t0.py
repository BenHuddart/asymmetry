"""Per-channel prompt-maximum and leading half-height timing features.

These features provide candidate integer anchors, not a physical pulse-centre
certification. Selected automatic requests require each channel's own estimate.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram

_PULSED_TOKENS = ("isis", "ral", "rutherford", "j-parc", "jparc", "kek", "riken")
_CONTINUOUS_TOKENS = ("psi", "triumf", "lem")

#: How far the detected t0 may sit from the file's before it is worth a warning,
#: per source family (D8, from the ISIS/PSI header survey). Continuous sources
#: resolve the prompt peak to a bin or two; a pulsed source's edge midpoint is
#: broader, so it gets one more.
T0_TOLERANCE_BINS: dict[str, int] = {"continuous": 2, "pulsed": 3}

#: Which family each :func:`find_t0` strategy belongs to.
_STRATEGY_FAMILY = {"prompt_peak": "continuous", "pulse_edge": "pulsed"}

#: Grouping-dict key carrying the T0Policy-resolved effective per-detector t0
#: bins (0-based, one per histogram). Distinct from the file-derived
#: ``detector_t0_bins`` per-run fact so a *manual* policy can shift alignment
#: without touching the file values. Absent for the ``from_file`` default.
EFFECTIVE_DETECTOR_T0_KEY = "effective_detector_t0_bins"


def detector_t0_overrides(grouping: dict | None, n_histograms: int) -> list[int] | None:
    """Read an explicit integer table without changing an invalid request."""
    grouping = grouping if isinstance(grouping, dict) else {}
    if EFFECTIVE_DETECTOR_T0_KEY not in grouping:
        return None
    raw = grouping[EFFECTIVE_DETECTOR_T0_KEY]
    if not isinstance(raw, (list, tuple)) or len(raw) != n_histograms:
        raise ValueError("Effective detector t0 overrides must contain one integer per histogram.")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in raw):
        raise ValueError(
            "Effective detector t0 overrides must be non-negative integer bin indices."
        )
    return [int(v) for v in raw]


def effective_detector_t0_bins(histograms: list[Histogram], grouping: dict | None) -> list[int]:
    """The per-detector t0 bins every alignment in the app must use (D10).

    The single resolver: a policy-resolved override from the grouping
    (:data:`EFFECTIVE_DETECTOR_T0_KEY`, written by a *manual* or *auto_detect*
    :class:`~asymmetry.core.project.profiles.T0Policy`) when present and of
    matching length; malformed explicit tables raise. When absent, use each
    histogram's own file-derived ``t0_bin``. The
    override is non-destructive — ``Histogram.t0_bin`` is never rewritten — so
    every consumer that re-derives alignment from the histograms alone silently
    ignores the user's t0 choice. Pass the result as ``detector_t0_bins=`` to
    :func:`~asymmetry.core.transform.grouping.common_t0_for_groups` and
    :func:`~asymmetry.core.transform.grouping.apply_grouping_aligned`; the
    structural harness enforces that.
    """
    override = detector_t0_overrides(grouping, len(histograms))
    if override is not None:
        return override
    return [int(hist.t0_bin) for hist in histograms]


def good_window_for_groups(
    group_indices: Sequence[int],
    detector_t0_bins: Sequence[int],
    detector_first_good_bins: Sequence[int],
    detector_last_good_bins: Sequence[int],
    *,
    common_t0_bin: int,
    n_bins: int,
) -> tuple[int, int]:
    """The good window of the *aligned* group sums, in grouped-bin indices (F13).

    Alignment shifts each detector by ``common_t0_bin − t0_i``, so its own good
    window moves with it; the run's window is the *intersection* over the
    analysis (forward + backward) detectors only — a spectator detector (a veto
    counter, a ring this grouping does not use) must not narrow the analysed
    range. A detector whose header puts its good window before its own t0 keeps
    that negative offset (the window is the header's, not a derived quantity);
    only the resulting bounds are clamped into ``[0, n_bins)``.

    The single rule shared by the PSI, ROOT and NeXus loaders and by
    :func:`~asymmetry.core.project.profiles.resolve_effective_grouping`, which
    must re-derive it whenever a profile's analysis groups differ from the
    loader's default pair (the loader's window belongs to the loader's pair and
    to the common t0 *that* pair produced).

    ``n_bins`` is the length of the aligned group sums, the last index the
    window may name. ``group_indices`` are 0-based; an empty list yields the
    full range.
    """
    if n_bins < 1:
        raise ValueError("Selected channels have empty observed support.")
    last_bin = int(n_bins) - 1
    firsts = [int(detector_first_good_bins[i]) - int(detector_t0_bins[i]) for i in group_indices]
    lasts = [int(detector_last_good_bins[i]) - int(detector_t0_bins[i]) for i in group_indices]
    first_good = max(0, int(common_t0_bin) + max(firsts, default=-common_t0_bin))
    last_good = min(last_bin, int(common_t0_bin) + min(lasts, default=last_bin - common_t0_bin))
    if first_good > last_good:
        raise ValueError("Selected channels have empty good-bin support.")
    return first_good, last_good


def run_t0_time_us(histograms: list[Histogram], common_t0_bin: int) -> float | None:
    """Mean exact origin of arithmetic members at a shared integer anchor."""
    values = [
        hist.t0_time_us
        for hist in histograms
        if hist.t0_bin == common_t0_bin and hist.t0_time_us is not None
    ]
    return float(np.mean(values)) if values else None


def selected_t0_reference(
    histograms: list[Histogram],
    grouping: dict | None,
    *,
    selected_indices: Sequence[int] | None = None,
) -> int:
    """Selected maximum effective anchor, with acquired ordinal breaking ties."""
    from asymmetry.core.transform.grouping import effective_group_indices

    grouping = grouping if isinstance(grouping, dict) else {}
    if selected_indices is None:
        if "groups" in grouping:
            selected_indices = sorted(
                {
                    i
                    for key, default in (("forward_group", 1), ("backward_group", 2))
                    for i in effective_group_indices(
                        grouping, grouping.get(key, default), n_histograms=len(histograms)
                    )
                }
            )
        else:
            selected_indices = list(range(len(histograms)))
    bins = effective_detector_t0_bins(histograms, grouping)
    return min(selected_indices, key=lambda i: (-bins[i], i))


def common_t0_time_us(
    histograms: list[Histogram],
    grouping: dict | None,
    common_t0_bin: int,
    *,
    selected_indices: Sequence[int] | None = None,
) -> float:
    """Exact selected reference on the output grid, with integer policy shifts.

    The largest effective anchor wins; ties use acquired-channel ordinal.
    An output anchor may be reindexed by a leading crop. Stored run scalars
    cannot supply the origin for a different selected pair.
    """
    reference = selected_t0_reference(histograms, grouping, selected_indices=selected_indices)
    hist = histograms[reference]
    width = float(hist.bin_width)
    if hist.t0_time_us is None:
        return (common_t0_bin + 0.5) * width
    return hist.t0_time_us_effective + (common_t0_bin - hist.t0_bin) * width


def t0_stamp_residual_us(
    histograms: list[Histogram],
    grouping: dict | None,
    common_t0_bin: int,
) -> float:
    r"""The sub-bin offset of the exact t0 from the centre of ``common_t0_bin`` (D4).

    Every aligned time axis in the app is stamped as the integer-bin axis plus
    this residual: ``t_k = (k − common_t0_bin)·w + residual``. Folds the
    repeated ``(common_t0_bin + 0.5)·w − common_t0_time_us(...)`` expression
    into one place. Exactly ``0.0`` when the run carries no exact t0 (
    :func:`common_t0_time_us` then returns the bin centre itself), which keeps
    the integer-bin axis unchanged to the last bit.
    """
    bin_width = float(histograms[0].bin_width)
    return (float(common_t0_bin) + 0.5) * bin_width - common_t0_time_us(
        histograms, grouping, common_t0_bin
    )


@dataclass(frozen=True)
class T0Estimate:
    """Time-zero estimate for one histogram.

    ``width_bins`` is how many bins the feature the estimate was read off
    actually spans — the prompt peak's FWHM, or the pulse's 10 %→90 % rise —
    as a descriptive diagnostic, not a confidence interval: the same PSI prompt peak is 1 bin wide at
    1 ns binning and 10 at 98 ps. ``0`` when the estimate failed.
    """

    t0_bin: int
    strategy: str  # "prompt_peak" | "pulse_edge"
    peak_bin: int
    ok: bool
    message: str = ""
    width_bins: int = 0


@dataclass(frozen=True)
class RunT0Search:
    """Per-detector t0 estimates plus their consensus for one run.

    ``consensus_t0_bin`` / ``spread_bins`` describe the *raw* estimates: the
    median and full range of the detected bins themselves. They only say
    something about the run's time zero when every detector's header t0 is the
    same, which is exactly what PSI per-detector headers are not — two GPS
    detectors legitimately sitting 170 bins earlier than the rest make the raw
    median and the header maximum incomparable.

    The ``shift_*`` fields describe the *shifts* ``est_i − file_i`` instead, which
    are comparable across a staggered run: ``shift_median_bins`` is how far the
    detected time zero sits from the file's (the Δ the grouping window reports)
    and ``shift_spread_bins`` is how much the detectors disagree about that
    shift. A run whose detectors are each within a bin of their own header has a
    small shift spread however wide its raw spread is.

    ``width_bins`` is the median of the detectors' :attr:`T0Estimate.width_bins`
    — how many bins the feature t0 was read off spans — and sets the tolerance
    every check is judged against (:func:`tolerance_bins`).
    """

    estimates: list[T0Estimate]
    consensus_t0_bin: int
    spread_bins: int
    strategy: str
    ok: bool
    message: str = ""
    shift_median_bins: int = 0
    shift_spread_bins: int = 0
    width_bins: int = 0


def detector_t0_shifts(
    histograms: list[Histogram],
    estimates: list[T0Estimate],
) -> list[int | None]:
    """``est_i − file_i`` per detector; ``None`` where the estimate failed.

    Measured against each histogram's own **file** ``t0_bin``, never a resolved
    override — the shift is what the detection says about the header, so a policy
    that already moved the alignment must not move the baseline it is judged by.
    """
    return [
        int(estimate.t0_bin) - int(hist.t0_bin) if estimate.ok else None
        for hist, estimate in zip(histograms, estimates, strict=True)
    ]


def shift_statistics(shifts: Sequence[int | None]) -> tuple[int, int]:
    """The median and the full range of the resolved per-detector shifts.

    ``(0, 0)`` when no detector resolved. The median is the run's shift — one
    detector that missed cannot drag it — and the range is how far the detectors
    disagree, which is the spread worth warning about (a staggered run's raw
    estimate spread is not).
    """
    values = [shift for shift in shifts if shift is not None]
    if not values:
        return 0, 0
    return int(round(float(np.median(values)))), int(max(values) - min(values))


def source_is_pulsed(metadata: dict[str, Any] | None) -> bool | None:
    """Established source hints, or None when no supported strategy is known."""
    metadata = metadata if isinstance(metadata, dict) else {}
    if isinstance(metadata.get("source_pulsed"), bool):
        return metadata["source_pulsed"]
    text = " ".join(
        str(metadata.get(key, "")) for key in ("facility", "instrument", "source", "area")
    ).lower()
    matches = [
        pulsed
        for pulsed, tokens in ((False, _CONTINUOUS_TOKENS), (True, _PULSED_TOKENS))
        if re.search(r"(?<![a-z0-9])(?:" + "|".join(tokens) + r")(?![a-z0-9])", text)
    ]
    return matches[0] if len(matches) == 1 else None


def _peak_fwhm_bins(c: NDArray[np.float64], peak: int) -> int:
    """The prompt peak's full width at half maximum, in bins (at least 1).

    Walks out from the argmax while the counts hold at or above half the peak.
    On a continuous source the prompt spike stands an order of magnitude over
    the decay tail, so the walk stops on the peak's own flanks.
    """
    half = c[peak] / 2.0
    lo = peak
    while lo > 0 and c[lo - 1] >= half:
        lo -= 1
    hi = peak
    while hi < c.size - 1 and c[hi + 1] >= half:
        hi += 1
    return hi - lo + 1


def _rise_width_bins(c: NDArray[np.float64], peak: int, crossing: int) -> int:
    """The pulse's 10 %→90 % rise width in bins, on the walk-back to *crossing*.

    Measured on the same leading edge the half-maximum crossing was read off,
    so an early flash before the pulse cannot widen it.
    """
    below_low = np.flatnonzero(c[: peak + 1] < 0.1 * c[peak])
    start = int(below_low[-1]) if below_low.size else 0
    above_high = np.flatnonzero(c[crossing : peak + 1] >= 0.9 * c[peak])
    end = crossing + int(above_high[0]) if above_high.size else peak
    return max(0, end - start)


def find_t0(
    counts: NDArray[np.float64],
    *,
    pulsed: bool,
) -> T0Estimate:
    """Estimate t0 for one histogram.

    Continuous: the maximum-count bin (prompt peak; first occurrence wins on
    ties). Pulsed: the
    half-maximum crossing of the leading edge, linearly interpolated and
    rounded to the nearest bin.

    Each estimate also carries the width of the feature it was read off
    (:attr:`T0Estimate.width_bins`), used only for descriptive display thresholds.
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.size == 0 or not np.any(c > 0.0):
        return T0Estimate(
            0,
            "pulse_edge" if pulsed else "prompt_peak",
            0,
            ok=False,
            message="Histogram has no counts",
        )
    peak = int(np.argmax(c))
    if not pulsed:
        return T0Estimate(peak, "prompt_peak", peak, ok=True, width_bins=_peak_fwhm_bins(c, peak))

    half = c[peak] / 2.0
    # Walk back from the peak: the edge is the crossing adjacent to the
    # pulse, i.e. just after the LAST sub-half-maximum bin before the peak.
    # Taking the first above-half bin anywhere would instead lock onto an
    # early prompt flash or noise spike that tops half-maximum.
    below_half = np.flatnonzero(c[: peak + 1] < half)
    if below_half.size == 0:
        return T0Estimate(
            0,
            "pulse_edge",
            peak,
            ok=False,
            message="Histogram starts above half-maximum — no leading edge in range",
        )
    crossing = int(below_half[-1]) + 1
    below = c[crossing - 1]
    above = c[crossing]
    fraction = 0.5 if above == below else (half - below) / (above - below)
    t0_bin = int(round(crossing - 1 + fraction))
    return T0Estimate(
        t0_bin,
        "pulse_edge",
        peak,
        ok=True,
        width_bins=_rise_width_bins(c, peak, crossing),
    )


def find_t0_for_run(
    histograms: list[Histogram],
    metadata: dict[str, Any] | None = None,
    *,
    pulsed: bool | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> RunT0Search:
    """Estimate t0 for every histogram of a run, with a consensus.

    The consensus is the median of the per-detector estimates (rounded);
    ``spread_bins`` is their full range. Both describe the raw estimates, so on
    per-detector-t0 data they mix the detection with the detectors' real
    stagger; :attr:`RunT0Search.shift_median_bins` /
    :attr:`RunT0Search.shift_spread_bins` are the health indicators to read
    there — a spread of a few bins in the *shifts* is normal detector-to-detector
    variation, a large one means dead detectors or a wrong strategy.
    """
    if pulsed is not None and not isinstance(pulsed, bool):
        raise ValueError("pulsed must be True, False, or None for metadata inference.")
    if pulsed is None:
        pulsed = source_is_pulsed(metadata)
    metadata = metadata or {}
    if pulsed is None or metadata.get("raw_period_histograms_available") is False:
        strategy = "unknown" if pulsed is None else "pulse_edge" if pulsed else "prompt_peak"
        message = (
            "Automatic t0 unavailable: selected period has no acquired histograms."
            if metadata.get("raw_period_histograms_available") is False
            else "Automatic t0 unavailable: requires a known source or an explicit pulsed strategy."
        )
        return RunT0Search(
            [T0Estimate(0, strategy, 0, False, message) for _ in histograms],
            0,
            0,
            strategy,
            False,
            message,
        )
    strategy = "pulse_edge" if pulsed else "prompt_peak"
    estimates = []
    for hist in histograms:
        if cancel_check is not None:
            cancel_check()
        estimates.append(find_t0(hist.counts, pulsed=pulsed))
    good = [est.t0_bin for est in estimates if est.ok]
    if not good:
        return RunT0Search(
            estimates=estimates,
            consensus_t0_bin=0,
            spread_bins=0,
            strategy=strategy,
            ok=False,
            message="No histogram produced a t0 estimate",
        )
    consensus = int(round(float(np.median(good))))
    spread = int(max(good) - min(good))
    shift_median, shift_spread = shift_statistics(detector_t0_shifts(histograms, estimates))
    widths = [int(est.width_bins) for est in estimates if est.ok]
    return RunT0Search(
        estimates=estimates,
        consensus_t0_bin=consensus,
        spread_bins=spread,
        strategy=strategy,
        ok=True,
        shift_median_bins=shift_median,
        shift_spread_bins=shift_spread,
        width_bins=int(round(float(np.median(widths)))),
    )


def selected_t0_diagnostics(
    histograms: list[Histogram],
    grouping: dict,
    search: RunT0Search,
    *,
    resolved_indices: tuple[list[int], list[int]] | None = None,
) -> tuple[int, int, int]:
    """Selected median shift, shift range, and median feature width; diagnostics only."""
    from asymmetry.core.transform.grouping import independent_pair_indices

    forward, backward = independent_pair_indices(
        grouping, n_histograms=len(histograms), resolved_indices=resolved_indices
    )
    indices = sorted(set(forward) | set(backward))
    shifts = detector_t0_shifts(histograms, search.estimates)
    delta, spread = shift_statistics([shifts[i] for i in indices])
    widths = [search.estimates[i].width_bins for i in indices if search.estimates[i].ok]
    return delta, spread, int(round(float(np.median(widths)))) if widths else 0


def acquired_t0_sources(grouping: dict, n_histograms: int) -> list[str]:
    """Per-channel file facts; older projects retain their recorded scalar fact."""
    sources = grouping.get("detector_t0_sources")
    if sources is not None:
        if (
            not isinstance(sources, (list, tuple))
            or len(sources) != n_histograms
            or any(source not in ("file", "missing", "conflict") for source in sources)
        ):
            raise ValueError(
                "Detector timing sources must name one file/missing status per acquired histogram."
            )
        return list(sources)
    source = "missing" if grouping.get("t0_source") == "missing" else "file"
    return [source] * n_histograms


def effective_t0_sources(grouping: dict, n_histograms: int) -> list[str]:
    """Effective outcomes, including an explicitly supplied absolute timing table."""
    if "effective_t0_sources" in grouping:
        sources = grouping["effective_t0_sources"]
        if (
            not isinstance(sources, (list, tuple))
            or len(sources) != n_histograms
            or any(
                source not in ("file", "manual", "estimated", "unresolved", "missing")
                for source in sources
            )
        ):
            raise ValueError(
                "Effective timing sources must name one outcome per acquired histogram."
            )
        return list(sources)
    if detector_t0_overrides(grouping, n_histograms) is not None:
        return ["manual"] * n_histograms
    return acquired_t0_sources(grouping, n_histograms)


def require_selected_timing(grouping: dict, indices: Sequence[int], n_histograms: int) -> None:
    """Reject unresolved required anchors without rejecting the acquired run."""
    sources = effective_t0_sources(grouping, n_histograms)
    unresolved = [i + 1 for i in indices if sources[i] in ("missing", "unresolved")]
    if unresolved:
        raise ValueError(
            "Automatic t0 unavailable for selected channels " + ", ".join(map(str, unresolved))
        )


def resolve_selected_timing(
    histograms: list[Histogram],
    grouping: dict,
    metadata: dict | None = None,
    *,
    mode: str = "from_file",
    resolved_indices: tuple[list[int], list[int]] | None = None,
    offset_bins: int = 0,
    search: RunT0Search | None = None,
    pulsed: bool | None = None,
    cancel_check: Callable[[], None] | None = None,
    search_provider: Callable | None = None,
) -> None:
    """Resolve only the selected acquired channels, retaining file facts unchanged.

    Explicit detection replaces required anchors. File/manual policies search
    only when required file timing is absent; no failed estimate borrows a delay.
    A supplied search is reusable across selections of the same raw acquisition.
    """
    from asymmetry.core.transform.grouping import independent_pair_indices

    forward, backward = independent_pair_indices(
        grouping, n_histograms=len(histograms), resolved_indices=resolved_indices
    )
    indices = sorted(set(forward) | set(backward))
    for key in (
        EFFECTIVE_DETECTOR_T0_KEY,
        "effective_t0_sources",
        "t0_search_strategy",
        "t0_search_spread_bins",
    ):
        grouping.pop(key, None)
    file_sources = acquired_t0_sources(grouping, len(histograms))
    bins = [int(hist.t0_bin) for hist in histograms]
    sources = [
        "unresolved" if mode == "auto_detect" or source == "missing" else "file"
        for source in file_sources
    ]
    needed = [i for i in indices if mode == "auto_detect" or file_sources[i] == "missing"]
    for i in indices:
        if i not in needed:
            sources[i] = "file"
    if needed:
        if metadata is not None and metadata.get("raw_period_histograms_available") is False:
            grouping["effective_t0_sources"] = sources
            raise ValueError(
                "Automatic t0 unavailable: selected period has no acquired histograms."
            )
        if search is None:
            search = (search_provider or find_t0_for_run)(
                histograms, metadata, pulsed=pulsed, cancel_check=cancel_check
            )
        grouping["t0_search_strategy"] = search.strategy
        grouping["t0_search_spread_bins"] = (
            selected_t0_diagnostics(
                histograms, grouping, search, resolved_indices=resolved_indices
            )[1]
            if search.strategy != "unknown"
            else 0
        )
        for i in needed:
            if search.strategy != "unknown" and search.estimates[i].ok:
                bins[i] = search.estimates[i].t0_bin
                sources[i] = "estimated"
    grouping["effective_t0_sources"] = sources
    from asymmetry.core.io.periods import selected_period_index

    if "period_histograms" in grouping:
        grouping["effective_t0_period"] = selected_period_index(grouping) or 0
    grouping["t0_policy"] = {
        "mode": mode,
        **({"offset_bins": offset_bins} if mode == "manual" else {}),
    }
    if needed and search.strategy == "unknown":
        raise ValueError(search.message)
    require_selected_timing(grouping, indices, len(histograms))
    if mode == "manual":
        for i in indices:
            bins[i] = max(0, bins[i] + offset_bins)
            sources[i] = "manual"
    if mode == "auto_detect" or offset_bins or needed:
        grouping[EFFECTIVE_DETECTOR_T0_KEY] = bins
    grouping["t0_bin"] = max(bins[i] for i in indices)
    if grouping.get("t0_time_us") is not None:
        grouping["t0_time_us"] = common_t0_time_us(
            histograms, grouping, grouping["t0_bin"], selected_indices=indices
        )


def tolerance_bins(strategy: str, width_bins: int = 0) -> int:
    """Descriptive display threshold: the larger of a family floor and feature width.

    This is neither a confidence interval nor a physical calibration test.
    """
    return max(T0_TOLERANCE_BINS[_STRATEGY_FAMILY[strategy]], int(width_bins))


@dataclass(frozen=True)
class T0Assessment:
    """The verdict on a run's time zero: what to show and how loudly (D8).

    ``level`` is ``"ok"``, ``"warn"`` or ``"error"``. An *error* is a notice, not
    a block (D9): the reduction always proceeds with the chosen mode, so the
    levels only drive the colour and the Apply summary. ``messages`` are plain
    sentences rendered verbatim by the GUI; ``delta_bins`` is the run's shift —
    the median of the per-detector ``est_i − file_i`` — and
    ``outlier_detectors`` are the 1-based detector numbers whose own shift
    disagrees with that median, i.e. the detectors that disagree with the
    *others* about where time zero moved. A detector that legitimately sits 170
    bins before the rest and whose header says so is not an outlier.

    Every check is judged against :func:`tolerance_bins` for the run's strategy
    *and* the measured width of the feature t0 was read off, so the same physical
    disagreement is judged the same way at any binning.
    """

    level: str
    delta_bins: int | None
    messages: tuple[str, ...]
    outlier_detectors: tuple[int, ...]


def _good_window_messages(
    histograms: list[Histogram],
    grouping: dict,
    search: RunT0Search,
    *,
    detected_common: int,
    file_common: int,
    base: int,
) -> list[str]:
    """Report overlap with a detected count feature, without certifying arrival time."""
    first_good = grouping.get("first_good_bin")
    if first_good is None:
        return []
    first_good = int(first_good)
    messages: list[str] = []
    if first_good <= detected_common:
        messages.append(
            f"First good bin {first_good + base} is at or before the detected t0 "
            f"(bin {detected_common + base})"
        )
    if search.strategy != "pulse_edge":
        return messages
    # Each detector's peak lives on its own bin axis; alignment moves it by
    # ``file_common − t0_i``, so compare on the common axis the window uses.
    from asymmetry.core.transform.grouping import effective_group_indices

    selected = set(
        effective_group_indices(
            grouping, grouping.get("forward_group", 1), n_histograms=len(histograms)
        )
    ) | set(
        effective_group_indices(
            grouping, grouping.get("backward_group", 2), n_histograms=len(histograms)
        )
    )
    peaks = [
        int(estimate.peak_bin) + file_common - int(hist.t0_bin)
        for i, (hist, estimate) in enumerate(zip(histograms, search.estimates, strict=True))
        if i in selected and estimate.ok
    ]
    if peaks and first_good <= max(peaks):
        messages.append(
            f"First good bin {first_good + base} is inside the muon pulse "
            f"(peak at bin {max(peaks) + base})"
        )
    return messages


def assess_t0(
    histograms: list[Histogram],
    grouping: dict | None,
    search: RunT0Search | None,
) -> T0Assessment:
    """Compare a run's file t0 with the detected one and return the verdict (D8).

    Checks run in severity order and the first *error* wins outright; warnings
    accumulate. ``search`` is the detection for this run
    (:func:`find_t0_for_run`) or ``None`` while one is still pending, in which
    case only the file-side checks can fire.

    Every bin number inside a message is written in the run's own display base
    (``grouping["bin_index_base"]``, 1 for ISIS), because the messages are shown
    beside bin numbers the GUI already displays that way — a message quoting the
    internal 0-based index would contradict the line it sits on. Comparisons and
    :attr:`T0Assessment.delta_bins` stay internal: a difference is base-free.
    """
    # Imported here, not at module scope: grouping.py imports this module for
    # the resolver, so a top-level import would close the cycle.
    from asymmetry.core.transform.grouping import common_t0_for_groups, independent_pair_indices

    grouping = grouping if isinstance(grouping, dict) else {}
    n_hist = len(histograms)
    base = int(grouping.get("bin_index_base", 0))
    detector_t0_bins = effective_detector_t0_bins(histograms, grouping)
    try:
        forward_idx, backward_idx = independent_pair_indices(grouping, n_histograms=n_hist)
    except ValueError as exc:
        return T0Assessment("error", None, (str(exc),), ())
    file_common = common_t0_for_groups(
        histograms, forward_idx, backward_idx, detector_t0_bins=detector_t0_bins
    )

    if grouping.get("t0_source") == "missing":
        return T0Assessment(
            level="error",
            delta_bins=None,
            messages=(
                "No time zero in the file header; selected channels require their own estimates",
            ),
            outlier_detectors=(),
        )

    n_bins = int(histograms[0].n_bins)
    if not 0 <= file_common < n_bins:
        return T0Assessment(
            level="error",
            delta_bins=None,
            messages=(f"Time zero is bin {file_common + base}, outside the run's {n_bins} bins",),
            outlier_detectors=(),
        )

    messages: list[str] = []
    if grouping.get("t0_source") == "conflict":
        messages.append("Header time_zero disagrees with t0_bin; using t0_bin")

    delta: int | None = None
    outliers: tuple[int, ...] = ()
    if search is not None and search.ok:
        delta, spread, width = selected_t0_diagnostics(histograms, grouping, search)
        tol = tolerance_bins(search.strategy, width)
        selected = set(forward_idx) | set(backward_idx)
        shifts = [
            shift if i in selected else None
            for i, shift in enumerate(detector_t0_shifts(histograms, search.estimates))
        ]
        detected_common = max(
            (
                est.t0_bin + file_common - histograms[i].t0_bin
                for i, est in enumerate(search.estimates)
                if i in selected and est.ok
            ),
            default=file_common,
        )
        if abs(delta) > tol:
            messages.append(
                f"Detected t0 is bin {detected_common + base}, file t0 is bin "
                f"{file_common + base} — further apart than the {tol}-bin tolerance"
            )
        outliers = tuple(
            index + 1
            for index, shift in enumerate(shifts)
            if shift is not None and abs(shift - delta) > tol
        )
        if outliers:
            listed = ", ".join(str(number) for number in outliers)
            messages.append(
                f"Detectors {listed} disagree with the other detectors' t0 shift "
                f"by more than {tol} bins"
            )
        if spread > 4 * tol:
            messages.append(f"Detector spread {spread} bins — check the source type")
        messages.extend(
            _good_window_messages(
                histograms,
                grouping,
                search,
                detected_common=detected_common,
                file_common=file_common,
                base=base,
            )
        )

    return T0Assessment(
        level="warn" if messages else "ok",
        delta_bins=delta,
        messages=tuple(messages),
        outlier_detectors=outliers,
    )
