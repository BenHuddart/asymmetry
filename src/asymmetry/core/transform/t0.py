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
    rising = np.flatnonzero(c[: peak + 1] >= half)
    if rising.size == 0:  # cannot happen (peak >= half), defensive
        return T0Estimate(peak, "pulse_edge", peak, ok=True)
    crossing = int(rising[0])
    if crossing == 0:
        return T0Estimate(
            0,
            "pulse_edge",
            peak,
            ok=False,
            message="Histogram starts above half-maximum — no leading edge in range",
        )
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
