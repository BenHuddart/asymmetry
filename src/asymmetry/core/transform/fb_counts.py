"""Validated grouped counts and their aligned acquisition-bin coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _bin_index(value: object, name: str, *, signed: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer bin index; got {value!r}.")
    if not signed and value < 0:
        raise ValueError(f"{name} must be non-negative; got {value}.")
    return int(value)


@dataclass(frozen=True)
class BinWindow:
    """Inclusive, zero-based window in aligned grouped-bin coordinates."""

    first: int
    last: int

    def __post_init__(self) -> None:
        first = _bin_index(self.first, "good-bin window first")
        last = _bin_index(self.last, "good-bin window last")
        if first > last:
            raise ValueError(f"good-bin window [{first}, {last}] is reversed.")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "last", last)

    def validate_length(self, n_bins: int) -> None:
        if n_bins < 1 or self.last >= n_bins:
            raise ValueError(
                f"good-bin window [{self.first}, {self.last}] is outside {n_bins} bins."
            )

    @property
    def slice(self) -> slice:
        return slice(self.first, self.last + 1)


def good_window(grouping: dict[str, Any] | None, n_bins: int) -> BinWindow:
    """Read a declared window; absent endpoints select the full aligned record."""
    grouping = {} if grouping is None else grouping
    window = BinWindow(grouping.get("first_good_bin", 0), grouping.get("last_good_bin", n_bins - 1))
    window.validate_length(n_bins)
    return window


def validated_count_arrays(*arrays: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Validate a public count-array boundary without truncation or broadcasting."""
    values = tuple(np.asarray(array) for array in arrays)
    if any(value.ndim != 1 for value in values):
        raise ValueError("count arrays must have one-dimensional shape.")
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("count arrays must have equal shape.")
    if any(value.dtype.kind not in "biuf" for value in values):
        raise ValueError("count arrays must contain real numeric values.")
    converted = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if any(not np.all(np.isfinite(value)) for value in converted):
        raise ValueError("count arrays must contain finite values.")
    return converted


def count_timing(
    bin_width_us: float, common_t0_bin: int, t0_time_us: float | None
) -> tuple[float, float]:
    """Validate a uniform acquisition grid, returning its width and exact t0 in μs."""
    if (
        isinstance(bin_width_us, bool)
        or not isinstance(bin_width_us, Real)
        or not np.isfinite(bin_width_us)
        or bin_width_us <= 0
    ):
        raise ValueError("bin width must be finite and positive in μs.")
    common_t0_bin = _bin_index(common_t0_bin, "common t0 bin")
    width = float(bin_width_us)
    if t0_time_us is None:
        return width, (common_t0_bin + 0.5) * width
    if (
        isinstance(t0_time_us, bool)
        or not isinstance(t0_time_us, Real)
        or not np.isfinite(t0_time_us)
    ):
        raise ValueError("t0 time must be finite in μs.")
    return width, float(t0_time_us)


@dataclass(frozen=True)
class GroupCounts:
    """One grouped count record with optional independent per-bin variance.

    ``variance=None`` denotes the Poisson count model. Arrays are copied and
    read-only, so changing an input buffer cannot invalidate the checked shape.
    """

    counts: NDArray[np.float64]
    variance: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        arrays = (self.counts,) if self.variance is None else (self.counts, self.variance)
        values = validated_count_arrays(*arrays)
        if self.variance is not None and np.any(values[1] < 0):
            raise ValueError("count variance must be non-negative.")
        for name, value in zip(("counts", "variance"), values):
            owned = np.array(value, dtype=np.float64, copy=True, order="C")
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


@dataclass(frozen=True)
class AlignmentCrop:
    """Detector shifts and the chosen terminal crop, all in acquisition bins."""

    common_t0_bin: int
    detector_offsets: tuple[tuple[int, int], ...]
    group_lengths: tuple[int, int]
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "common_t0_bin", _bin_index(self.common_t0_bin, "common t0 bin"))
        lengths = tuple(_bin_index(value, "group length") for value in self.group_lengths)
        length = _bin_index(self.length, "aligned length")
        if len(lengths) != 2 or length < 1 or length > min(lengths):
            raise ValueError("alignment crop requires two non-empty groups covering its length.")
        offsets = tuple(
            (
                _bin_index(index, "detector index"),
                _bin_index(offset, "detector offset", signed=True),
            )
            for index, offset in self.detector_offsets
        )
        if len({index for index, _offset in offsets}) != len(offsets):
            raise ValueError("alignment crop must record each detector once.")
        object.__setattr__(self, "detector_offsets", offsets)
        object.__setattr__(self, "group_lengths", lengths)
        object.__setattr__(self, "length", length)


@dataclass(frozen=True)
class SubtractedBackground:
    """Subtracted constant count levels and their standard errors per raw bin."""

    forward: float
    backward: float
    forward_error: float
    backward_error: float


@dataclass(frozen=True)
class ForwardBackwardCounts:
    """Aligned forward/backward counts with one validated window and time grid.

    ``alignment`` owns the common t0 bin; ``t0_time_us`` preserves any sub-bin
    residual. Background provenance is a snapshot, with its constant levels and
    standard errors stored once under ``values`` and ``value_errors``.
    """

    forward: GroupCounts
    backward: GroupCounts
    t0_time_us: float
    bin_width_us: float
    window: BinWindow
    alignment: AlignmentCrop
    deadtime_applied: bool = False
    background_state: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.forward, GroupCounts) or not isinstance(self.backward, GroupCounts):
            raise ValueError("forward and backward counts require GroupCounts values.")
        if self.forward.counts.shape != self.backward.counts.shape:
            raise ValueError("forward and backward counts must have equal shape.")
        n = self.forward.counts.size
        if n < 1:
            raise ValueError("forward/backward counts require at least one bin.")
        if (self.forward.variance is None) != (self.backward.variance is None):
            raise ValueError("count variance must be present for both groups or neither.")
        if not isinstance(self.window, BinWindow) or not isinstance(self.alignment, AlignmentCrop):
            raise ValueError("counts require a BinWindow and AlignmentCrop.")
        self.window.validate_length(n)
        if self.alignment.length != n:
            raise ValueError("alignment crop length must equal the count shape.")
        width, t0 = count_timing(self.bin_width_us, self.common_t0_bin, self.t0_time_us)
        object.__setattr__(self, "bin_width_us", width)
        object.__setattr__(self, "t0_time_us", t0)
        state = None if self.background_state is None else deepcopy(dict(self.background_state))
        if state is not None:
            if ("values" in state) != ("value_errors" in state):
                raise ValueError("constant background levels and errors must be present together.")
            if "values" in state:
                levels, uncertainty = validated_count_arrays(state["values"], state["value_errors"])
                if levels.size != 2 or np.any(uncertainty < 0):
                    raise ValueError(
                        "background provenance requires two levels and non-negative errors."
                    )
                state["values"] = tuple(float(value) for value in levels)
                state["value_errors"] = tuple(float(value) for value in uncertainty)
        object.__setattr__(
            self, "background_state", None if state is None else MappingProxyType(state)
        )

    def background_metadata(self) -> dict[str, object] | None:
        """Export independent JSON-ready provenance with list-valued count levels."""
        if self.background_state is None:
            return None
        state = deepcopy(dict(self.background_state))
        if "values" in state:
            state["values"] = list(state["values"])
            state["value_errors"] = list(state["value_errors"])
        return state

    @property
    def common_t0_bin(self) -> int:
        return self.alignment.common_t0_bin

    @property
    def background_level(self) -> tuple[float, float] | None:
        if self.background_state is None or "values" not in self.background_state:
            return None
        forward, backward = self.background_state["values"]
        return float(forward), float(backward)

    def subtracted_background(self) -> SubtractedBackground | None:
        """Read the constant background needed by integral and α uncertainty propagation."""
        if self.background_level is None:
            return None
        forward, backward = self.background_level
        forward_error, backward_error = self.background_state["value_errors"]
        return SubtractedBackground(forward, backward, float(forward_error), float(backward_error))
