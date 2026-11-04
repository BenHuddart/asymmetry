"""Decode one acquisition grid from redundant muon NeXus timing declarations.

Conventions follow S. Cottrell, NeXus Instrument Definitions for Muon Data,
version 2 revision 8 (2011), pp. 14 and 49–51 (including the v1 definition).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from asymmetry.core.io.physical_units import (
    UnitConvention,
    canonical,
    storage_precision_us,
    uniform_bin_width_us,
)

logger = logging.getLogger(__name__)
_REFERENCE = "Cottrell, muon NeXus definition v2 revision 8 (2011), pp. 14, 49–51"


@dataclass(frozen=True)
class T0Decode:
    """Acquired integer anchors and coherent exact origins for one period."""

    t0_bins: np.ndarray
    t0_time_us: list[float | None]
    source: str


@dataclass(frozen=True)
class _TimeField:
    """Canonical values with the precision of the stored declaration."""

    values: np.ndarray
    dtype: np.dtype
    scale: float
    context: str

    @property
    def error(self) -> np.ndarray:
        """Four storage ULPs plus float64 conversion round-off, in μs."""
        return storage_precision_us(self.values, stored_dtype=self.dtype, storage_scale=self.scale)


def _read_time(group: Any, name: str, *, default_unit: str, context: str) -> _TimeField | None:
    node = group.get(name)
    if node is None:
        return None
    raw = np.asarray(node)
    field_context = f"{context}/{name}"
    values, _ = canonical(
        raw,
        node.attrs.get("units"),
        "time",
        context=field_context,
        convention=UnitConvention(default_unit, _REFERENCE),
    )
    scale, _ = canonical(
        1,
        node.attrs.get("units"),
        "time",
        context=field_context,
        convention=UnitConvention(default_unit, _REFERENCE),
    )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{field_context}: timing must contain finite, nonempty values.")
    return _TimeField(values, raw.dtype, float(scale), field_context)


def _timing_table(values: np.ndarray, n_periods: int, n_detectors: int, context: str) -> np.ndarray:
    """Accept only scalar, detector-vector, or period-by-detector timing."""
    if values.size == 1:
        return np.full((n_periods, n_detectors), float(values.reshape(-1)[0]))
    if values.shape == (n_detectors,):
        return np.broadcast_to(values, (n_periods, n_detectors))
    if values.shape == (n_periods, n_detectors):
        return values
    raise ValueError(
        f"{context}: expected scalar, {n_detectors} detectors, or "
        f"({n_periods}, {n_detectors}) period/detector values; got {values.shape}."
    )


def read_nexus_timing(
    group: Any, *, version: str, n_periods: int, n_detectors: int, n_bins: int, context: str
) -> tuple[float, list[T0Decode], np.ndarray]:
    """Validate file timing once, before constructing zero-origin histograms."""
    resolution = _read_time(
        group, "resolution", default_unit="ps" if version == "v1" else "ns", context=context
    )
    raw = _read_time(group, "raw_time", default_unit="us", context=context)
    if raw is None:
        raw = _read_time(group, "time_of_flight", default_unit="us", context=context)
    corrected = _read_time(group, "corrected_time", default_unit="us", context=context)
    exact = _read_time(group, "time_zero", default_unit="us", context=context)
    if exact is not None and int(group["time_zero"].attrs.get("available", 1)) == 0:
        exact = None

    width = None
    width_error = 0.0
    if resolution is not None:
        if resolution.values.size != 1 or float(resolution.values.reshape(-1)[0]) < 0:
            raise ValueError(f"{resolution.context}: resolution must be a nonnegative scalar.")
        candidate = float(resolution.values.reshape(-1)[0])
        if candidate > 0:
            width = candidate
            width_error = float(resolution.error.reshape(-1)[0])

    axes = []
    for field in (raw, corrected):
        if field is None:
            continue
        if field.values.ndim != 1:
            raise ValueError(f"{field.context}: expected a shared one-dimensional time axis.")
        kind = "edges" if field.values.size == n_bins + 1 else "centres"
        if n_bins == 1 and field.values.size == 1 and width is not None:
            candidate, span_error = width, 0.0
        else:
            candidate = uniform_bin_width_us(
                field.values,
                kind=kind,
                n_bins=n_bins,
                stored_dtype=field.dtype,
                storage_scale=field.scale,
                context=field.context,
            )
            span_error = float((field.error[0] + field.error[-1]) / (field.values.size - 1))
        if width is None:
            width, width_error = candidate, span_error
        # Compare the full span, not a noisy adjacent-bin median. Endpoint
        # quantisation bounds the uncertainty of the reconstructed width.
        if abs(candidate - width) > span_error + width_error:
            raise ValueError(f"{field.context}: time axis conflicts with acquisition resolution.")
        indices = np.arange(field.values.size)
        expected = field.values[0] + indices * width
        coordinate_error = field.error + field.error[0] + indices * width_error
        if np.any(np.abs(field.values - expected) > coordinate_error):
            raise ValueError(
                f"{field.context}: coordinates conflict with uniform acquisition grid."
            )
        axes.append((field, kind))
    if width is None:
        raise ValueError(f"{context}: no usable time axis or positive resolution.")

    raw_origin = 0.0
    origin_error = 0.0
    for field, kind in axes:
        if field is raw:
            raw_origin = float(field.values[0]) - (width / 2 if kind == "centres" else 0)
            origin_error = float(field.error[0]) + (width_error / 2 if kind == "centres" else 0)
    exact_values = None
    exact_errors = None
    if exact is not None:
        exact_values = (
            _timing_table(exact.values, n_periods, n_detectors, exact.context) - raw_origin
        )
        exact_errors = (
            _timing_table(exact.error, n_periods, n_detectors, exact.context) + origin_error
        )
    attr = group["counts"].attrs.get("t0_bin")
    anchors = None
    if attr is not None:
        values = np.asarray(attr, dtype=float)
        if (
            not np.all(np.isfinite(values))
            or np.any(values != np.floor(values))
            or np.any(values < 1)
        ):
            raise ValueError(f"{context}/counts:t0_bin: expected positive integer anchors.")
        anchors = (
            _timing_table(values, n_periods, n_detectors, f"{context}/counts:t0_bin").astype(
                np.int64
            )
            - 1
        )

    decodes = []
    for period in range(n_periods):
        times = None if exact_values is None else exact_values[period]
        if anchors is None and times is None:
            decodes.append(
                T0Decode(np.zeros(n_detectors, dtype=np.int64), [None] * n_detectors, "missing")
            )
            continue
        if anchors is None:
            # Only values indistinguishable from an edge at stored precision
            # snap to that edge for integer anchoring; exact times stay intact.
            q = times / width
            nearest = np.rint(q)
            errors = exact_errors[period] + np.abs(nearest) * width_error
            bins = np.floor(np.where(np.abs(times - nearest * width) <= errors, nearest, q)).astype(
                np.int64
            )
            bins = np.maximum(0, bins)
        else:
            bins = anchors[period]
        conflict = False
        if times is not None and anchors is not None:
            errors = exact_errors[period] + (bins + 1) * width_error
            conflict = bool(
                np.any((times < bins * width - errors) | (times > (bins + 1) * width + errors))
            )
        for field, kind in axes:
            if field is not corrected:
                continue
            axis_t0 = (width / 2 if kind == "centres" else 0) - float(field.values[0])
            axis_error = float(field.error[0]) + (width_error / 2 if kind == "centres" else 0)
            # A shared corrected axis can describe a detector or the common
            # aligned origin. Per-detector exact values remain authoritative.
            if times is not None:
                conflict |= not bool(
                    np.any(np.abs(times - axis_t0) <= exact_errors[period] + axis_error)
                )
            else:
                conflict |= not bool(
                    np.any(
                        (axis_t0 >= bins * width - axis_error - bins * width_error)
                        & (axis_t0 <= (bins + 1) * width + axis_error + (bins + 1) * width_error)
                    )
                )
        if conflict:
            logger.warning(
                "%s: NeXus t0_bin/time_zero/corrected_time declarations conflict; keeping integer anchors and dropping exact t0.",
                context,
            )
        decodes.append(
            T0Decode(
                bins,
                [None] * n_detectors if times is None or conflict else times.tolist(),
                "conflict" if conflict else "file",
            )
        )
    return width, decodes, np.asarray([]) if corrected is None else corrected.values
