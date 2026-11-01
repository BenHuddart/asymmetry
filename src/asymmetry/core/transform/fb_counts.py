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


def _count_snapshot(value: NDArray[np.float64]) -> NDArray[np.float64]:
    owned = np.array(value, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class SharedComponent:
    """One shared parameter's variance, sensitivity and raw-count covariance.

    For corrected counts c = r + g(θ̂ − θ), ``sensitivity`` is g and
    ``count_covariance`` is cov(r, θ̂). Components in a group must describe
    mutually independent parameters; all arrays use its acquisition bins.
    """

    name: str
    variance: float
    sensitivity: NDArray[np.float64]
    count_covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("shared count components require a name.")
        if (
            isinstance(self.variance, bool)
            or not isinstance(self.variance, Real)
            or not np.isfinite(self.variance)
            or self.variance < 0
        ):
            raise ValueError("shared parameter variance must be finite and non-negative.")
        sensitivity, covariance = validated_count_arrays(self.sensitivity, self.count_covariance)
        object.__setattr__(self, "variance", float(self.variance))
        object.__setattr__(self, "sensitivity", _count_snapshot(sensitivity))
        object.__setattr__(self, "count_covariance", _count_snapshot(covariance))


@dataclass(frozen=True)
class GroupCounts:
    """Grouped counts with independent variance and shared parameter components.

    ``variance=None`` denotes the Poisson count model. Arrays are copied and
    read-only, so changing an input buffer cannot invalidate the checked shape.
    """

    counts: NDArray[np.float64]
    variance: NDArray[np.float64] | None = None
    shared: tuple[SharedComponent, ...] = ()

    def __post_init__(self) -> None:
        arrays = (self.counts,) if self.variance is None else (self.counts, self.variance)
        values = validated_count_arrays(*arrays)
        if self.variance is not None and np.any(values[1] < 0):
            raise ValueError("count variance must be non-negative.")
        shared = tuple(self.shared)
        if self.variance is None and shared:
            raise ValueError("shared corrections require explicit independent count variance.")
        if any(not isinstance(component, SharedComponent) for component in shared):
            raise ValueError("shared count corrections require SharedComponent values.")
        if any(component.sensitivity.shape != values[0].shape for component in shared):
            raise ValueError("shared count components must have the count shape.")
        if len({component.name for component in shared}) != len(shared):
            raise ValueError("shared count components must have distinct names.")
        if shared:
            positive = values[1] > 0
            normalized = []
            for component in shared:
                covariance = component.count_covariance
                if np.any(covariance[~positive] != 0) or (
                    component.variance == 0 and np.any(covariance != 0)
                ):
                    raise ValueError("shared count covariance requires non-zero source variance.")
                if component.variance > 0:
                    normalized.append(
                        covariance[positive]
                        / np.sqrt(values[1][positive])
                        / np.sqrt(component.variance)
                    )
            if normalized:
                influence = np.column_stack(normalized)
                # Normalising by each parameter's σ makes the joint covariance
                # test invariant to the units chosen for counts or parameters.
                residual = np.eye(len(normalized)) - influence.T @ influence
                if not np.all(np.isfinite(residual)) or np.linalg.eigvalsh(residual)[0] < (
                    -64 * np.finfo(float).eps * len(normalized)
                ):
                    raise ValueError(
                        "shared components require a positive-semidefinite joint covariance."
                    )
        for name, value in zip(("counts", "variance"), values):
            object.__setattr__(self, name, _count_snapshot(value))
        object.__setattr__(self, "shared", shared)


def linear_combination_variance(
    group: GroupCounts, edges: ArrayLike, weights: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Variance of each weighted bin sum, including estimation-window overlap.

    For coefficients a, each shared parameter adds (Σag)² var(θ̂) +
    2(Σag)(Σa cov(r, θ̂)) to the independent Poisson term Σa²v. Edges
    select half-open acquisition-bin intervals; weights cover the whole group.
    """
    bounds = np.asarray(edges)
    if (
        bounds.ndim != 1
        or bounds.dtype.kind not in "iu"
        or bounds.size < 2
        or bounds[0] < 0
        or bounds[-1] > group.counts.size
        or np.any(bounds[1:] <= bounds[:-1])
    ):
        raise ValueError("variance edges must be increasing integer bounds within the count shape.")
    coefficients = np.ones(group.counts.size) if weights is None else np.asarray(weights)
    if coefficients.shape != group.counts.shape:
        raise ValueError("variance weights must have the count shape.")
    (coefficients,) = validated_count_arrays(coefficients)
    bounds = np.asarray(bounds, dtype=np.intp)
    window = slice(int(bounds[0]), int(bounds[-1]))
    starts = bounds[:-1] - bounds[0]
    v = group.counts if group.variance is None else group.variance
    if np.any(v < 0):
        raise ValueError(
            "Poisson count variance requires non-negative counts or explicit variance."
        )
    result = np.add.reduceat((coefficients**2 * v)[window], starts)
    magnitude = result.copy()
    for component in group.shared:
        sensitivity = np.add.reduceat((coefficients * component.sensitivity)[window], starts)
        covariance = np.add.reduceat((coefficients * component.count_covariance)[window], starts)
        shared_variance = sensitivity**2 * component.variance
        overlap = 2 * sensitivity * covariance
        result += shared_variance + overlap
        magnitude += shared_variance + np.abs(overlap)
    tolerance = 32 * np.finfo(np.float64).eps * magnitude
    if not np.all(np.isfinite(result)):
        raise ValueError("count linear-combination variance is not finite.")
    if np.any(result < -tolerance):
        raise ValueError("count covariance gives a negative linear-combination variance.")
    return np.maximum(result, 0.0)


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
class ForwardBackwardCounts:
    """Aligned forward/backward counts with one validated window and time grid.

    ``alignment`` owns the common t0 bin; ``t0_time_us`` preserves any sub-bin
    residual. Background provenance stores its constant levels; standard errors
    are exported from the shared count components.
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
            if "value_errors" in state:
                raise ValueError("background standard errors derive from shared count components.")
            if "values" in state:
                (levels,) = validated_count_arrays(state["values"])
                if levels.size != 2:
                    raise ValueError("background provenance requires two constant levels.")
                state["values"] = tuple(float(value) for value in levels)
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
            state["value_errors"] = [
                float(
                    np.sqrt(
                        sum(
                            component.variance
                            for component in group.shared
                            if component.name == "background"
                        )
                    )
                )
                for group in (self.forward, self.backward)
            ]
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
