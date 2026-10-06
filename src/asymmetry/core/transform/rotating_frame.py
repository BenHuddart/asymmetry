"""Exact rotation of the transverse projection pair into a rotating frame.

With both transverse projections measured, the frame turning at ν about B₀ ∥ z
is reached without a filter (docs/plans/rotating-frame-projection.md):

    z_phys = (P_x − b_x) + i·(−s)·(P_y − b_y)/g,    z′ = z_phys · e^{+i(2πνt + φ)}

- s is the measured sense of the labels: −1 when (P_x, P_y, P_z) are
  right-handed about B₀, so that P_y is the physical y.
- g = a_y/a_x is the transverse gain; b_x and b_y are the baselines left after
  alpha.
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

from asymmetry.core.data.dataset import MuonDataset

__all__ = [
    "FRAME_FIELDS",
    "ROTATED_LABELS",
    "RUN_FIELDS",
    "SETUP_FIELDS",
    "B1Axis",
    "FrameEstimate",
    "Provenance",
    "RotatingFrame",
    "RunEstimate",
    "estimate_frame",
    "rotate_transverse",
]

#: The rotated projections' labels, by the lab projection each replaces.
ROTATED_LABELS: dict[str, str] = {"P_x": "P′_x", "P_y": "P′_y"}

#: Fields shared by the runs of one setup, then the fields of each run.
SETUP_FIELDS: tuple[str, ...] = ("frequency_mhz", "b1_axis", "rf_phase_deg", "sense")
RUN_FIELDS: tuple[str, ...] = ("baseline_x", "baseline_y", "gain")
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


@dataclass(frozen=True)
class RotatingFrame:
    """One run's rotating frame: setup fields, run fields and their provenance.

    Build a new one with :meth:`typed_frequency`; change fields with
    :meth:`with_values`, which records where the new values came from.
    """

    frequency_mhz: float
    b1_axis: B1Axis
    rf_phase_deg: float
    sense: int
    baseline_x: float
    baseline_y: float
    gain: float
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
        object.__setattr__(self, "baseline_x", _finite("b_x (%)", self.baseline_x))
        object.__setattr__(self, "baseline_y", _finite("b_y (%)", self.baseline_y))
        gain = _finite("Gain a_y/a_x", self.gain)
        if gain <= 0.0:
            raise ValueError(f"Gain a_y/a_x must be > 0, got {gain!r}.")
        object.__setattr__(self, "gain", gain)
        if set(self.provenance) != set(FRAME_FIELDS):
            raise ValueError(f"Frame provenance must name exactly {FRAME_FIELDS}.")
        object.__setattr__(
            self, "provenance", {name: Provenance(self.provenance[name]) for name in FRAME_FIELDS}
        )

    @classmethod
    def typed_frequency(cls, frequency_mhz: float) -> RotatingFrame:
        """The frame a user starts by typing ν_RF; every other field a default."""
        provenance = {name: Provenance.DEFAULT for name in FRAME_FIELDS}
        provenance["frequency_mhz"] = Provenance.TYPED
        return cls(frequency_mhz, B1Axis.X, 0.0, -1, 0.0, 0.0, 1.0, provenance)

    def with_values(self, provenance: Provenance, **values: Any) -> RotatingFrame:
        """These fields set to ``values``, recorded as coming from ``provenance``."""
        unknown = set(values) - set(FRAME_FIELDS)
        if unknown:
            raise ValueError(f"Unknown frame fields {sorted(unknown)}.")
        sources = dict(self.provenance) | {name: provenance for name in values}
        return replace(self, **values, provenance=sources)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FRAME_FIELDS} | {
            "b1_axis": str(self.b1_axis),
            "provenance": {name: str(source) for name, source in self.provenance.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RotatingFrame:
        """Inverse of :meth:`to_dict`; raises ``ValueError`` naming the bad field."""
        return cls(**{name: data[name] for name in FRAME_FIELDS}, provenance=data["provenance"])


def rotate_transverse(
    px: MuonDataset, py: MuonDataset, frame: RotatingFrame
) -> tuple[MuonDataset, MuonDataset]:
    """The P′_x and P′_y datasets of one run's transverse pair, errors propagated.

    The rotated components share each bin's noise (their covariance is
    (σ_y² − σ_x²)·sinθ·cosθ, zero at unit gain and equal errors); each dataset
    carries its own per-bin error.
    """
    if px.time.shape != py.time.shape or not np.array_equal(px.time, py.time):
        raise ValueError("P_x and P_y must share one time axis to be rotated together.")
    t = px.time
    x = px.asymmetry - frame.baseline_x
    y = (py.asymmetry - frame.baseline_y) / frame.gain
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
    """One run's proposed baselines and gain, with how strongly it turns at ν."""

    run_key: Hashable
    baseline_x: float
    baseline_y: float
    gain: float
    #: |amplitude at s·ν| / |amplitude at −s·ν| for this run alone.
    contrast: float


@dataclass(frozen=True)
class FrameEstimate:
    """Proposed setup fields from the runs together, and each run's own fields.

    ``frequency_mhz`` is a check on the typed ν_RF: the power centroid of the
    band around it, which sits between a nutation's two sidebands at ν ± ν₁.
    """

    frequency_mhz: float
    sense: int
    rf_phase_deg: float
    contrast: float
    runs: tuple[RunEstimate, ...]


#: Half-width of the band about ±ν, as a fraction of ν, that holds the signal.
_BAND = 0.25


def _weighted_mean(values: NDArray[np.float64], errors: NDArray[np.float64]) -> float:
    weights = 1.0 / np.square(errors)
    return float(np.sum(weights * values) / np.sum(weights))


def estimate_frame(
    pairs: Sequence[tuple[Hashable, MuonDataset, MuonDataset]],
    *,
    frequency_mhz: float,
    b1_axis: B1Axis,
) -> FrameEstimate:
    """Estimate a frame from runs' (key, P_x, P_y) curves at the typed ν_RF.

    - Baselines are inverse-variance means: whole turns average to them.
    - The gain is the ratio of the RMS transverse swings, because the lab-frame
      transverse polarisation is circular.
    - The sense is the side, ±ν, whose band holds more power; the contrast is the
      ratio of the two bands.
    - φ_RF puts the transverse polarisation on the nutation axis ẑ × B̂₁. The
      axis is the principal direction, known modulo 180°. The sign comes from
      early times, because a spin leaving +z first moves towards ẑ × B̂₁.
    """
    if not pairs:
        raise ValueError("Auto-detect needs at least one run with P_x and P_y.")
    nu = float(frequency_mhz)
    baselined = []
    for key, px, py in pairs:
        bx = _weighted_mean(px.asymmetry, px.error)
        by = _weighted_mean(py.asymmetry, py.error)
        x, y = px.asymmetry - bx, py.asymmetry - by
        gain = float(np.sqrt(np.mean(np.square(y)) / np.mean(np.square(x))))
        baselined.append((key, px.time, x + 1j * y / gain, bx, by, gain))

    span = max(t[-1] - t[0] for _, t, *_ in baselined)
    band = nu + _BAND * nu * np.linspace(-1.0, 1.0, max(9, int(16.0 * _BAND * nu * span)))

    def band_power(t: NDArray[np.float64], z: NDArray[np.complex128], sign: int):
        return np.square(np.abs(np.exp(-2j * math.pi * sign * np.outer(band, t)) @ z))

    runs, totals, spectrum = [], np.zeros(2), np.zeros((2, band.size))
    for key, t, z, bx, by, gain in baselined:
        power = np.stack([band_power(t, z, 1), band_power(t, z, -1)])
        spectrum += power
        sums = power.sum(axis=1)
        totals += sums
        runs.append((key, bx, by, gain, sums))
    side = 0 if totals[0] >= totals[1] else 1
    sense = 1 if side == 0 else -1
    floor = np.median(spectrum[1 - side])
    excess = np.clip(spectrum[side] - floor, 0.0, None)
    centroid = float(np.sum(band * excess) / np.sum(excess)) if excess.any() else nu

    second_moment = 0j
    early = 0j
    for _key, t, z, *_ in baselined:
        z_frame = (z.real - 1j * sense * z.imag) * np.exp(2j * math.pi * nu * t)
        second_moment += np.sum(z_frame**2)
        early += np.sum((1.0 - (t - t[0]) / (t[-1] - t[0])) * z_frame)
    axis_angle = 0.5 * math.atan2(second_moment.imag, second_moment.real)
    along = math.cos(axis_angle) * early.real + math.sin(axis_angle) * early.imag
    direction = math.degrees(axis_angle) + (0.0 if along >= 0.0 else 180.0)
    phase = _NUTATION_TARGET_DEG[str(b1_axis)] - direction
    return FrameEstimate(
        frequency_mhz=centroid,
        sense=sense,
        rf_phase_deg=(phase + 180.0) % 360.0 - 180.0,
        contrast=float(np.sqrt(totals[side] / totals[1 - side])),
        runs=tuple(
            RunEstimate(key, bx, by, gain, float(np.sqrt(sums[side] / sums[1 - side])))
            for key, bx, by, gain, sums in runs
        ),
    )
