"""Automatic time-zero search from the prompt peak or pulse rising edge.

Two strategies, selected by source type (textbook §14.2/§15.3):

- **continuous** (PSI, TRIUMF) — t0 is the sharp prompt peak from beam
  positrons triggering both counters: the maximum-count bin (the
  WiMDA ``SearchT0ButtonClick`` / musrfit ``musrt0`` convention; ties
  resolve to the earliest bin).
- **pulsed** (ISIS, J-PARC) — t0 is the *centre* of the muon pulse, found
  in practice from the midpoint of the histogram's rising edge. WiMDA uses
  the maximum bin for pulsed data too, which lands at the pulse peak
  rather than its centre of rise (study divergence D9).

Results are estimates for the user to confirm — a Find t0 action fills the
grouping override controls and never silently overwrites loader values
("never rely on information stored in the data file, if you have not
recorded it yourself" — textbook §15.3, p. 223).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
    """Extract policy-resolved per-detector t0 overrides from a grouping dict.

    Reads :data:`EFFECTIVE_DETECTOR_T0_KEY` (a *manual* T0Policy writes it) and
    returns it as an int list when it lines up with the histogram count, else
    ``None``. Callers that want the alignment values themselves should use
    :func:`effective_detector_t0_bins`; this raw accessor exists for the cache
    digests that must distinguish "no override" from "override equal to file".
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    raw = grouping.get(EFFECTIVE_DETECTOR_T0_KEY)
    if not isinstance(raw, (list, tuple)) or len(raw) != n_histograms:
        return None
    try:
        return [int(v) for v in raw]
    except (TypeError, ValueError):
        return None


def effective_detector_t0_bins(histograms: list[Histogram], grouping: dict | None) -> list[int]:
    """The per-detector t0 bins every alignment in the app must use (D10).

    The single resolver: a policy-resolved override from the grouping
    (:data:`EFFECTIVE_DETECTOR_T0_KEY`, written by a *manual* or *auto_detect*
    :class:`~asymmetry.core.project.profiles.T0Policy`) when present and of
    matching length, else each histogram's own file-derived ``t0_bin``. The
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
    last_bin = max(0, int(n_bins) - 1)
    firsts = [int(detector_first_good_bins[i]) - int(detector_t0_bins[i]) for i in group_indices]
    lasts = [int(detector_last_good_bins[i]) - int(detector_t0_bins[i]) for i in group_indices]
    first_good = min(last_bin, max(0, int(common_t0_bin) + max(firsts, default=0)))
    last_good = min(last_bin, max(0, int(common_t0_bin) + min(lasts, default=last_bin)))
    return first_good, max(first_good, last_good)


def run_t0_time_us(histograms: list[Histogram], common_t0_bin: int) -> float | None:
    """The run's exact t0 in µs, or ``None`` when no detector carries one (D4).

    Per-detector data can carry a different exact t0 per detector while the run
    reduces onto one common bin. The run's value is the mean exact t0 over the
    detectors whose integer ``t0_bin`` *is* the common bin — the detectors whose
    counts are not shifted by alignment. The others' sub-bin residuals (≤ ½ bin)
    are dropped. ``None`` when none of those detectors has an exact t0, in which
    case consumers fall back to
    :attr:`~asymmetry.core.data.dataset.Histogram.t0_time_us_effective`.
    """
    values = [
        float(hist.t0_time_us)
        for hist in histograms
        if hist.t0_time_us is not None and int(hist.t0_bin) == int(common_t0_bin)
    ]
    if not values:
        return None
    return float(np.mean(values))


def common_t0_time_us(
    histograms: list[Histogram],
    grouping: dict | None,
    common_t0_bin: int,
) -> float:
    r"""The run's exact common t0 in µs — the origin of every time stamp (D4).

    Time stamps are bin *centres* measured from this value:
    ``t_k = (k + 0.5)·w − T0``. Three sources, in order:

    1. ``grouping["t0_time_us"]`` — the per-run fact a loader recorded, already
       moved by ``delta·w`` if a :class:`~asymmetry.core.project.profiles.T0Policy`
       shifted the alignment. It belongs to the *effective* common bin, so pass
       the bin the alignment actually used.
    2. :func:`run_t0_time_us` over ``histograms`` at ``common_t0_bin`` — the mean
       exact t0 of the detectors that sit on that bin.
    3. The bin centre ``(common_t0_bin + 0.5)·w``. This fallback makes
       ``t_k = (k − common_t0_bin)·w``, the integer-bin axis, to the last bit.
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    stored = grouping.get("t0_time_us")
    if stored is not None:
        return float(stored)
    measured = run_t0_time_us(histograms, common_t0_bin)
    if measured is not None:
        return measured
    return (float(common_t0_bin) + 0.5) * float(histograms[0].bin_width)


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
    and so how precisely a bin index can name it. It is the resolution of the
    measurement, not of the file: the same PSI prompt peak is 1 bin wide at
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


def source_is_pulsed(metadata: dict[str, Any] | None) -> bool:
    """Infer pulsed vs continuous source from run metadata.

    Unknown facilities default to pulsed (the prompt-peak maximum and the
    pulse peak coincide on continuous data anyway; the edge-midpoint only
    differs when there is a wide pulse to have a midpoint of).
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    text = " ".join(
        str(metadata.get(key, "")) for key in ("facility", "instrument", "source", "area")
    ).lower()
    if any(token in text for token in _CONTINUOUS_TOKENS):
        return False
    if any(token in text for token in _PULSED_TOKENS):
        return True
    return True


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
    ties, matching WiMDA's descending strict-comparison scan). Pulsed: the
    half-maximum crossing of the leading edge, linearly interpolated and
    rounded to the nearest bin — the pulse-centre convention.

    Each estimate also carries the width of the feature it was read off
    (:attr:`T0Estimate.width_bins`), which is how precisely a bin index can name
    that feature and therefore the tolerance every later comparison deserves.
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.size == 0 or not np.any(c > 0.0):
        return T0Estimate(0, "prompt_peak", 0, ok=False, message="Histogram has no counts")
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
    if pulsed is None:
        pulsed = source_is_pulsed(metadata)
    strategy = "pulse_edge" if pulsed else "prompt_peak"
    estimates = [find_t0(hist.counts, pulsed=pulsed) for hist in histograms]
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


def detected_detector_t0_bins(
    search: RunT0Search,
    file_bins: Sequence[int],
    *,
    missing: bool,
) -> list[int]:
    """Each detector's **own** detected t0 — the ``musrt0 -g`` model.

    A run whose detectors genuinely sit at different times (PSI per-detector
    headers) keeps that stagger: collapsing the estimates into one median and
    shifting every detector by its distance from the group's header maximum
    compares two incomparable numbers and moves the whole run.

    A detector whose search failed (no counts, no leading edge) keeps its file
    t0 moved by the median of the shifts that did resolve, so it stays aligned
    with its neighbours. On a run with no header t0 at all (*missing*) the file
    values carry no information, so it takes the median *estimate* instead.
    """
    bins: list[int] = []
    for file_bin, estimate in zip(file_bins, search.estimates, strict=True):
        if estimate.ok:
            resolved = int(estimate.t0_bin)
        elif missing:
            resolved = int(search.consensus_t0_bin)
        else:
            resolved = int(file_bin) + int(search.shift_median_bins)
        bins.append(max(0, resolved))
    return bins


def tolerance_bins(strategy: str, width_bins: int = 0) -> int:
    """The divergence tolerance in bins for a :func:`find_t0` *strategy* (D8).

    The larger of the source family's floor and the measured width of the
    feature t0 was read off (``width_bins``; ``0`` when nothing was measured, so
    the floor applies alone). A bin index cannot name the centre of a prompt peak
    more precisely than the peak is wide, and how wide that is in *bins* depends
    on the binning: the D8 floors were calibrated on ~1 ns data, where a PSI
    prompt peak is a bin or two, but the same peak spans 4–11 bins at the 98 ps
    binning a modern GPS run uses. A fixed floor there reports jitter of a few
    tenths of a nanosecond as detectors disagreeing about time zero.
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
    """Warn when the analysed window opens on top of the muon arrival.

    The failure this catches is silent and expensive: a good window that starts
    at or before time zero includes the prompt peak (or, at a pulsed source, part
    of the pulse itself) in the asymmetry, which reads as a spuriously large
    early asymmetry rather than as an error. The window is judged against the
    *detected* t0, because that is where the muons actually arrived.

    Nothing to say when the grouping carries no good window (a payload that has
    not resolved one yet).
    """
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
    peaks = [
        int(estimate.peak_bin) + file_common - int(hist.t0_bin)
        for hist, estimate in zip(histograms, search.estimates, strict=True)
        if estimate.ok
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
    from asymmetry.core.transform.grouping import common_t0_for_groups, effective_group_indices

    grouping = grouping if isinstance(grouping, dict) else {}
    n_hist = len(histograms)
    base = int(grouping.get("bin_index_base", 0))
    detector_t0_bins = effective_detector_t0_bins(histograms, grouping)
    forward_idx = effective_group_indices(
        grouping, int(grouping.get("forward_group", 1)), n_histograms=n_hist
    )
    backward_idx = effective_group_indices(
        grouping, int(grouping.get("backward_group", 2)), n_histograms=n_hist
    )
    file_common = common_t0_for_groups(
        histograms, forward_idx, backward_idx, detector_t0_bins=detector_t0_bins
    )

    if grouping.get("t0_source") == "missing":
        return T0Assessment(
            level="error",
            delta_bins=None,
            messages=("No time zero in the file header; using the detected value",),
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
        tol = tolerance_bins(search.strategy, search.width_bins)
        shifts = detector_t0_shifts(histograms, search.estimates)
        delta, spread = shift_statistics(shifts)
        detected_common = int(file_common) + delta
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
