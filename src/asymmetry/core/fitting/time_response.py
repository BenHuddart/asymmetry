"""Normalised instantaneous-source response on the resolved observation clock."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import ismethod, unwrap

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.time_window import time_window_mask
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def live_source_ages(time: NDArray, separation_us: float):
    """Yield live masks, ages, births and equal incident weights in µs.

    Separation is validated by the consuming request boundary. Age zero is
    live; ordinary ratios and absolute counts share these causal populations.
    """
    half = separation_us / 2
    for birth in (-half, half):
        live = time >= birth
        if np.any(live):
            yield live, time[live] - birth, birth, 0.5


@dataclass(frozen=True)
class TimeResponse:
    """Fixed equal-incident double-delta response, with midpoint origin in µs.

    ``off`` preserves the supplied callable's full domain. ``double_delta``
    evaluates only live ages, including age zero. Separation is supplied,
    never a fit parameter. Dormant off inputs are unused and canonicalised.
    """

    mode: str = "off"
    separation_us: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("off", "double_delta"):
            raise ValueError("Time response must be off or double_delta.")
        if self.mode == "off":
            object.__setattr__(self, "separation_us", 0.0)
        else:
            try:
                separation = float(self.separation_us)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Supply a finite nonnegative fixed pulse separation in µs."
                ) from exc
            if not np.isfinite(separation) or separation < 0:
                raise ValueError("Supply a finite nonnegative fixed pulse separation in µs.")
            object.__setattr__(self, "separation_us", separation)

    def to_dict(self) -> dict:
        """Canonical consumed settings; off has no active separation."""
        return {
            "mode": self.mode,
            **({"separation_us": self.separation_us} if self.mode != "off" else {}),
        }

    @property
    def first_arrival_us(self) -> float | None:
        """The earliest defined prediction time for an active response."""
        return None if self.mode == "off" else -self.separation_us / 2

    @classmethod
    def from_dict(cls, payload: Mapping | None) -> TimeResponse:
        """Restore optional authored settings; absent historical settings are off."""
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping) or "mode" not in payload:
            raise ValueError("Time response settings need an explicit off or double_delta mode.")
        mode = payload["mode"]
        if mode == "double_delta" and "separation_us" not in payload:
            raise ValueError("Supply a fixed pulse separation in µs for double_delta.")
        return cls(mode=mode, separation_us=payload.get("separation_us", 0.0))

    def evaluate(self, source: Callable, time: NDArray, **parameters: float) -> NDArray:
        """Evaluate the complete source at each live age and normalise survival.

        A point before the first birth has no defined normalised prediction.
        Subtracting the latest live birth in the exponent cancels the common
        survival factor without underflow at late observation times.
        """
        time = np.asarray(time, dtype=float)
        first_arrival = self.first_arrival_us
        if first_arrival is None:
            return source(time, **parameters)
        if np.any(time < first_arrival):
            raise ValueError(
                "Double-pulse model has no source population before the first arrival "
                f"(t = {first_arrival:g} µs); choose a supported fit window or turn Time response off."
            )
        latest_birth = np.where(time >= -first_arrival, -first_arrival, first_arrival)
        numerator = np.zeros_like(time)
        denominator = np.zeros_like(time)
        for live, age, birth, incident in live_source_ages(time, self.separation_us):
            weight = incident * np.exp((birth - latest_birth[live]) / MUON_LIFETIME_US)
            numerator[live] += weight * source(age, **parameters)
            denominator[live] += weight
        return numerator / denominator

    def curve_times(self, t_min: float, t_max: float, points: int) -> NDArray:
        """Sample only the defined prediction interval for a display overlay.

        Literal window edges need not coincide with retained bin centres.
        Clipping extrapolation to the first birth does not omit observations;
        :meth:`evaluate` still diagnoses unsupported requested model points.
        """
        first_arrival = self.first_arrival_us
        if first_arrival is not None:
            if (
                t_max < first_arrival
                and time_window_mask(np.array([first_arrival]), t_min, t_max).item()
            ):
                t_max = first_arrival
            t_min = max(t_min, first_arrival)
            if t_max < t_min:
                raise ValueError(
                    "Double-pulse curve window has no source population; choose a supported fit window."
                )
        return np.linspace(t_min, t_max, points)

    def compose(self, source: Callable) -> Callable:
        """Wrap a fractional or percent source callable without changing scale."""
        if self.mode == "off":
            return source

        return TimeResponseFunction(source, self)


@dataclass(frozen=True)
class TimeResponseFunction:
    """Callable source with its authored time response available to selectors."""

    source: Callable
    time_response: TimeResponse

    @property
    def __wrapped__(self):
        return self.source

    def __call__(self, time, **parameters):
        return self.time_response.evaluate(self.source, time, **parameters)


def first_model_arrival(model_fn: Callable) -> float | None:
    """Return the exact first arrival for known model callables and wrappers."""
    from asymmetry.core.fitting.composite import CompositeModel

    fn = unwrap(model_fn, stop=lambda value: isinstance(value, TimeResponseFunction))
    if isinstance(fn, TimeResponseFunction):
        return fn.time_response.first_arrival_us
    if ismethod(fn) and isinstance(fn.__self__, CompositeModel):
        return fn.__self__.time_response.first_arrival_us
    return None
