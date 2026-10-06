"""Exact rotation of the transverse projection pair into a rotating frame.

With both transverse projections measured, the frame turning at ν about B₀ ∥ z
is reached without a filter (docs/plans/rotating-frame-projection.md):

    z_phys = (P_x − b_x) + i·(−s)·(P_y − b_y)/g,    z′ = z_phys · e^{+i(2πνt + φ)}

- s is the measured sense of the labels: −1 when (P_x, P_y, P_z) are
  right-handed about B₀, so that P_y is the physical y.
- g = a_y/a_x is the transverse gain.
- b_x and b_y are the baselines left after alpha, one pair per period: on RF
  data the red and green baselines differ by up to 0.1 % in a way that follows
  the detuning (docs/plans/rotating-frame-projection.md, D9). A period
  combination's curve holds the same combination of them.
- The muon precesses negatively about +z, so e^{+iθ} undoes the precession.
- With φ the RF phase at t0, the drive's co-rotating field sits on the chosen
  lab axis of the frame, and a spin leaving +z nutates into +y′ (B₁ ∥ x′) or
  −x′ (B₁ ∥ y′).

P′_x = Re z′, P′_y = Im z′. Each bin is transformed alone, so bins stay
independent and may be rebinned afterwards — unlike the filtered single-signal
demodulation of :mod:`asymmetry.core.transform.rrf`.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from asymmetry.core.data.dataset import MuonDataset

__all__ = [
    "FRAME_FIELDS",
    "MIN_CONTRAST",
    "PeriodBaseline",
    "ROTATED_LABELS",
    "RUN_FIELDS",
    "SETUP_FIELDS",
    "B1Axis",
    "FrameEstimate",
    "Provenance",
    "RotatingFrame",
    "RunEstimate",
    "estimate_frame",
    "period_weights",
    "rotate_transverse",
]

#: The rotated projections' labels, by the lab projection each replaces.
ROTATED_LABELS: dict[str, str] = {"P_x": "P′_x", "P_y": "P′_y"}

#: How a two-period run's display combines its periods (red is period 1).
_TWO_PERIOD_WEIGHTS: dict[str, tuple[float, float]] = {
    "red": (1.0, 0.0),
    "green": (0.0, 1.0),
    "green_minus_red": (-1.0, 1.0),
    "green_plus_red": (1.0, 1.0),
}

#: Fields shared by the runs of one setup, then a run's scalar field; its
#: baselines are per period (:class:`PeriodBaseline`).
SETUP_FIELDS: tuple[str, ...] = ("frequency_mhz", "b1_axis", "rf_phase_deg", "sense")
RUN_FIELDS: tuple[str, ...] = ("gain",)
FRAME_FIELDS: tuple[str, ...] = SETUP_FIELDS + RUN_FIELDS

#: Direction (°) a spin leaving +z nutates to in the frame, by B₁ axis: ẑ × B̂₁.
_NUTATION_TARGET_DEG = {"x": 90.0, "y": 180.0}


class B1Axis(StrEnum):
    """The lab axis of the linear RF field; its sign is absorbed by the phase."""

    X = "x"
    Y = "y"


class Provenance(StrEnum):
    """Where a frame field's value came from — Auto-detect never pre-ticks TYPED."""

    DEFAULT = "default"
    ESTIMATED = "estimated"
    TYPED = "typed"


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


def period_weights(period_mode: str, periods: int) -> tuple[float, ...]:
    """How the displayed curve combines a run's periods: one weight per period."""
    if periods == 1:
        return (1.0,)
    if periods == 2:
        return _TWO_PERIOD_WEIGHTS[str(period_mode)]
    raise ValueError(f"A rotating frame covers one or two periods, not {periods}.")


@dataclass(frozen=True)
class PeriodBaseline:
    """One period's transverse baselines (%) and where they came from."""

    x: float
    y: float
    provenance: Provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite("b_x (%)", self.x))
        object.__setattr__(self, "y", _finite("b_y (%)", self.y))
        object.__setattr__(self, "provenance", Provenance(self.provenance))


@dataclass(frozen=True)
class RotatingFrame:
    """One run's rotating frame: setup fields, gain, per-period baselines, provenance.

    Build a new one with :meth:`typed_frequency`; change scalar fields with
    :meth:`with_values` and a period's baselines with :meth:`with_baseline`,
    which record where the new values came from.
    """

    frequency_mhz: float
    b1_axis: B1Axis
    rf_phase_deg: float
    sense: int
    gain: float
    baselines: tuple[PeriodBaseline, ...]
    provenance: Mapping[str, Provenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frequency = _finite("ν_RF (MHz)", self.frequency_mhz)
        if frequency <= 0.0:
            raise ValueError(f"ν_RF (MHz) must be > 0, got {frequency!r}.")
        object.__setattr__(self, "frequency_mhz", frequency)
        object.__setattr__(self, "b1_axis", B1Axis(self.b1_axis))
        object.__setattr__(self, "rf_phase_deg", _finite("φ_RF (°)", self.rf_phase_deg))
        if self.sense not in (-1, 1):
            raise ValueError(f"Rotation sense must be ±1, got {self.sense!r}.")
        gain = _finite("Gain a_y/a_x", self.gain)
        if gain <= 0.0:
            raise ValueError(f"Gain a_y/a_x must be > 0, got {gain!r}.")
        object.__setattr__(self, "gain", gain)
        if not self.baselines:
            raise ValueError("A rotating frame needs the baselines of at least one period.")
        object.__setattr__(self, "baselines", tuple(self.baselines))
        if set(self.provenance) != set(FRAME_FIELDS):
            raise ValueError(f"Frame provenance must name exactly {FRAME_FIELDS}.")
        object.__setattr__(
            self, "provenance", {name: Provenance(self.provenance[name]) for name in FRAME_FIELDS}
        )

    @classmethod
    def typed_frequency(cls, frequency_mhz: float, periods: int) -> RotatingFrame:
        """The frame a user starts by typing ν_RF; every other field a default."""
        provenance = {name: Provenance.DEFAULT for name in FRAME_FIELDS}
        provenance["frequency_mhz"] = Provenance.TYPED
        baselines = tuple(PeriodBaseline(0.0, 0.0, Provenance.DEFAULT) for _ in range(periods))
        return cls(frequency_mhz, B1Axis.X, 0.0, -1, 1.0, baselines, provenance)

    def with_values(self, provenance: Provenance, **values: Any) -> RotatingFrame:
        """These scalar fields set to ``values``, recorded as coming from ``provenance``."""
        unknown = set(values) - set(FRAME_FIELDS)
        if unknown:
            raise ValueError(f"Unknown frame fields {sorted(unknown)}.")
        sources = dict(self.provenance) | {name: provenance for name in values}
        return replace(self, **values, provenance=sources)

    def with_baseline(
        self, period: int, provenance: Provenance, x: float, y: float
    ) -> RotatingFrame:
        """Period ``period``'s baselines set to (x, y), recorded as from ``provenance``."""
        baselines = list(self.baselines)
        baselines[period] = PeriodBaseline(x, y, provenance)
        return replace(self, baselines=tuple(baselines))

    def combined_baseline(self, weights: Sequence[float]) -> tuple[float, float]:
        """The baselines (b_x, b_y) a curve combining the periods by ``weights`` holds."""
        if len(weights) != len(self.baselines):
            raise ValueError(
                f"{len(weights)} period weights for a frame of {len(self.baselines)} periods."
            )
        return (
            float(sum(w * b.x for w, b in zip(weights, self.baselines, strict=True))),
            float(sum(w * b.y for w, b in zip(weights, self.baselines, strict=True))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FRAME_FIELDS} | {
            "b1_axis": str(self.b1_axis),
            "baselines": [
                {"x": b.x, "y": b.y, "provenance": str(b.provenance)} for b in self.baselines
            ],
            "provenance": {name: str(source) for name, source in self.provenance.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RotatingFrame:
        """Inverse of :meth:`to_dict`; raises ``ValueError`` naming the bad field."""
        return cls(
            **{name: data[name] for name in FRAME_FIELDS},
            baselines=tuple(
                PeriodBaseline(b["x"], b["y"], b["provenance"]) for b in data["baselines"]
            ),
            provenance=data["provenance"],
        )


def rotate_transverse(
    px: MuonDataset,
    py: MuonDataset,
    frame: RotatingFrame,
    *,
    weights: Sequence[float],
) -> tuple[MuonDataset, MuonDataset]:
    """The P′_x and P′_y datasets of one run's transverse pair, errors propagated.

    ``weights`` say how the curves combine the run's periods
    (:func:`period_weights`), so the same combination of the period baselines
    is removed. The rotated components share each bin's noise (their
    covariance is (σ_y² − σ_x²)·sinθ·cosθ, zero at unit gain and equal errors);
    each dataset carries its own per-bin error.
    """
    if px.time.shape != py.time.shape or not np.array_equal(px.time, py.time):
        raise ValueError("P_x and P_y must share one time axis to be rotated together.")
    t = px.time
    bx, by = frame.combined_baseline(weights)
    x = px.asymmetry - bx
    y = (py.asymmetry - by) / frame.gain
    var_x = np.square(px.error)
    var_y = np.square(py.error / frame.gain)
    theta = 2.0 * math.pi * frame.frequency_mhz * t + math.radians(frame.rf_phase_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    s = frame.sense
    rotated = {
        "P′_x": (x * cos + s * y * sin, np.sqrt(var_x * cos**2 + var_y * sin**2)),
        "P′_y": (x * sin - s * y * cos, np.sqrt(var_x * sin**2 + var_y * cos**2)),
    }
    pair = tuple(
        MuonDataset(
            time=t.copy(),
            asymmetry=values,
            error=errors,
            metadata={**source.metadata, "projection": label},
            run=source.run,
        )
        for (label, (values, errors)), source in zip(rotated.items(), (px, py), strict=True)
    )
    return pair[0], pair[1]


@dataclass(frozen=True)
class RunEstimate:
    """One run's proposed baselines, per period, and gain, with how strongly it turns at ν."""

    run_key: Hashable
    #: (b_x, b_y) of each period, in the order the periods were given.
    baselines: tuple[tuple[float, float], ...]
    gain: float
    #: |amplitude at s·ν| / |amplitude at −s·ν| for this run alone.
    contrast: float


@dataclass(frozen=True)
class FrameEstimate:
    """Proposed setup fields from the runs together, and each run's own fields.

    ν_RF is the user's and is not estimated: under RF the transverse spectrum is
    a sideband pair at ν ± ν₁, which can lie far from ν when B₁ is large.
    """

    sense: int
    rf_phase_deg: float
    contrast: float
    runs: tuple[RunEstimate, ...]


#: Half-width of the band about ±ν, as a fraction of ν, that holds the signal.
_BAND = 0.25

#: Below this contrast the data carry too little transverse signal for an estimate.
MIN_CONTRAST = 3.0


#: Knot spacing of the envelope splines, in turns of ν: the envelope may vary up
#: to about 0.6 ν (a nutation's ν₁ well below ν) while the carrier stays unresolved.
_KNOT_TURNS = 0.8


def _carrier_baseline(curve: MuonDataset, nu: float) -> float:
    """A curve's baseline, fitted beside a signal turning near ν.

    Weighted least squares of b + c(t)·cos 2πνt + d(t)·sin 2πνt with c, d cubic
    splines: any slowly varying amplitude and phase — the ν ± ν₁ pair of a
    nutation, a decay, a detuning — is absorbed by c and d, so the signal
    cannot leak into b as it does into a mean of early, strong bins.
    """
    t = curve.time
    lo, hi = float(t[0]), float(t[-1])
    intervals = max(4, math.ceil((hi - lo) * nu / _KNOT_TURNS))
    knots = np.concatenate(([lo] * 3, np.linspace(lo, hi, intervals + 1), [hi] * 3))
    envelope = BSpline.design_matrix(t, knots, 3).toarray()
    phase = 2.0 * math.pi * nu * t
    design = np.hstack(
        (
            np.ones((t.size, 1)),
            envelope * np.cos(phase)[:, None],
            envelope * np.sin(phase)[:, None],
        )
    )
    weight = 1.0 / curve.error
    solution, *_ = np.linalg.lstsq(design * weight[:, None], curve.asymmetry * weight, rcond=None)
    return float(solution[0])


def estimate_frame(
    runs: Sequence[tuple[Hashable, Sequence[tuple[MuonDataset, MuonDataset]]]],
    *,
    frequency_mhz: float,
    b1_axis: B1Axis,
) -> FrameEstimate:
    """Estimate a frame from each run's periods, as (key, [(P_x, P_y), …]), at the typed ν_RF.

    Each period is a single DAE period, never a combination: the baselines
    are a period's, and a green − red curve would turn the nutation over.
    Every sum is weighted by inverse variance, so the noisy late bins of a
    long window do not drown the signal.

    - Each period's baselines are fitted beside its signal (a period's own:
      red and green differ on RF data).
    - The gain is the ratio of the RMS transverse swings, because the lab-frame
      transverse polarisation is circular.
    - The sense is the side, ±ν, whose band holds more power over every period;
      the contrast is the ratio of the two bands.
    - φ_RF puts the transverse polarisation on the nutation axis ẑ × B̂₁. The
      axis is the principal direction, known modulo 180°. The sign comes from
      early times, because a spin leaving +z first moves towards ẑ × B̂₁.
    """
    if not runs:
        raise ValueError("Auto-detect needs at least one run with P_x and P_y.")
    nu = float(frequency_mhz)
    too_short = [
        key for key, periods in runs for px, _py in periods if (px.time[-1] - px.time[0]) * nu < 2.0
    ]
    if too_short:
        raise ValueError(
            f"Auto-detect needs a window of at least two turns of ν_RF; runs {too_short} have less."
        )

    weighted_curves = []
    per_run = []
    for key, periods in runs:
        baselines = tuple(
            (_carrier_baseline(px, nu), _carrier_baseline(py, nu)) for px, py in periods
        )
        swing = np.zeros(4)
        for (px, py), (bx, by) in zip(periods, baselines, strict=True):
            wx, wy = 1.0 / np.square(px.error), 1.0 / np.square(py.error)
            swing += (
                np.sum(wx * (px.asymmetry - bx) ** 2),
                np.sum(wx),
                np.sum(wy * (py.asymmetry - by) ** 2),
                np.sum(wy),
            )
        gain = float(np.sqrt((swing[2] / swing[3]) / (swing[0] / swing[1])))
        curves = []
        for (px, py), (bx, by) in zip(periods, baselines, strict=True):
            weight = 2.0 / (np.square(px.error) + np.square(py.error / gain))
            z = (px.asymmetry - bx) + 1j * (py.asymmetry - by) / gain
            curves.append((px.time, weight * z))
        weighted_curves.extend(curves)
        per_run.append((key, baselines, gain, curves))

    span = max(t[-1] - t[0] for t, _z in weighted_curves)
    band = nu + _BAND * nu * np.linspace(-1.0, 1.0, max(9, int(16.0 * _BAND * nu * span)))

    def band_power(t: NDArray[np.float64], z: NDArray[np.complex128], sign: int) -> float:
        return float(
            np.sum(np.square(np.abs(np.exp(-2j * math.pi * sign * np.outer(band, t)) @ z)))
        )

    run_powers = [
        np.array([sum(band_power(t, z, sign) for t, z in curves) for sign in (1, -1)])
        for _key, _baselines, _gain, curves in per_run
    ]
    totals = np.sum(run_powers, axis=0)
    side = 0 if totals[0] >= totals[1] else 1
    sense = 1 if side == 0 else -1

    second_moment = 0j
    early = 0j
    for t, z in weighted_curves:
        z_frame = (z.real - 1j * sense * z.imag) * np.exp(2j * math.pi * nu * t)
        second_moment += np.sum(z_frame**2)
        early += np.sum((1.0 - (t - t[0]) / (t[-1] - t[0])) * z_frame)
    axis_angle = 0.5 * math.atan2(second_moment.imag, second_moment.real)
    along = math.cos(axis_angle) * early.real + math.sin(axis_angle) * early.imag
    direction = math.degrees(axis_angle) + (0.0 if along >= 0.0 else 180.0)
    phase = _NUTATION_TARGET_DEG[str(b1_axis)] - direction
    return FrameEstimate(
        sense=sense,
        rf_phase_deg=(phase + 180.0) % 360.0 - 180.0,
        contrast=float(np.sqrt(totals[side] / totals[1 - side])),
        runs=tuple(
            RunEstimate(key, baselines, gain, float(np.sqrt(power[side] / power[1 - side])))
            for (key, baselines, gain, _curves), power in zip(per_run, run_powers, strict=True)
        ),
    )
