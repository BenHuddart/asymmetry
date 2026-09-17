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
    """Time-zero estimate for one histogram."""

    t0_bin: int
    strategy: str  # "prompt_peak" | "pulse_edge"
    peak_bin: int
    ok: bool
    message: str = ""


@dataclass(frozen=True)
class RunT0Search:
    """Per-detector t0 estimates plus their consensus for one run."""

    estimates: list[T0Estimate]
    consensus_t0_bin: int
    spread_bins: int
    strategy: str
    ok: bool
    message: str = ""


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
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.size == 0 or not np.any(c > 0.0):
        return T0Estimate(0, "prompt_peak", 0, ok=False, message="Histogram has no counts")
    peak = int(np.argmax(c))
    if not pulsed:
        return T0Estimate(peak, "prompt_peak", peak, ok=True)

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
    return T0Estimate(t0_bin, "pulse_edge", peak, ok=True)


def find_t0_for_run(
    histograms: list[Histogram],
    metadata: dict[str, Any] | None = None,
    *,
    pulsed: bool | None = None,
) -> RunT0Search:
    """Estimate t0 for every histogram of a run, with a consensus.

    The consensus is the median of the per-detector estimates (rounded);
    ``spread_bins`` is their full range, a quick health indicator — a spread
    of a few bins is normal detector-to-detector variation, a large one
    means dead detectors or a wrong strategy.
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
    return RunT0Search(
        estimates=estimates,
        consensus_t0_bin=consensus,
        spread_bins=spread,
        strategy=strategy,
        ok=True,
    )


def tolerance_bins(strategy: str) -> int:
    """The divergence tolerance in bins for a :func:`find_t0` *strategy* (D8)."""
    return T0_TOLERANCE_BINS[_STRATEGY_FAMILY[strategy]]


@dataclass(frozen=True)
class T0Assessment:
    """The verdict on a run's time zero: what to show and how loudly (D8).

    ``level`` is ``"ok"``, ``"warn"`` or ``"error"``. An *error* is a notice, not
    a block (D9): the reduction always proceeds with the chosen mode, so the
    levels only drive the colour and the Apply summary. ``messages`` are plain
    sentences rendered verbatim by the GUI; ``outlier_detectors`` are the
    1-based detector numbers whose own estimate diverges from their file t0.
    """

    level: str
    delta_bins: int | None
    messages: tuple[str, ...]
    outlier_detectors: tuple[int, ...]


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
        tol = tolerance_bins(search.strategy)
        delta = int(search.consensus_t0_bin) - int(file_common)
        if abs(delta) > tol:
            messages.append(
                f"Detected t0 is bin {int(search.consensus_t0_bin) + base}, file t0 is bin "
                f"{file_common + base} — further apart than the {tol}-bin tolerance"
            )
        outliers = tuple(
            index + 1
            for index, (estimate, effective) in enumerate(
                zip(search.estimates, detector_t0_bins, strict=True)
            )
            if estimate.ok and abs(int(estimate.t0_bin) - int(effective)) > tol
        )
        if outliers:
            listed = ", ".join(str(number) for number in outliers)
            messages.append(
                f"Detectors {listed} disagree with their file t0 by more than {tol} bins"
            )
        if int(search.spread_bins) > 4 * tol:
            messages.append(f"Detector spread {search.spread_bins} bins — check the source type")

    return T0Assessment(
        level="warn" if messages else "ok",
        delta_bins=delta,
        messages=tuple(messages),
        outlier_detectors=outliers,
    )
