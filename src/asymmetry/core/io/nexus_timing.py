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
    assess_uniform_bin_width_us,
    canonical,
    physical_unit,
    storage_precision_us,
    unit_text,
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
    if name == "time_zero" and int(node.attrs.get("available", 1)) == 0:
        return None
    raw = np.asarray(node)
    field_context = f"{context}/{name}"
    token = unit_text(node.attrs.get("units")) or default_unit
    conversion = physical_unit(token, dimension="time")
    if (
        raw.dtype.kind not in "iuf"
        or raw.size == 0
        or not np.all(np.isfinite(raw))
        or conversion is None
    ):
        logger.warning(
            "%s: unusable timing declaration; using other file timing when available.",
            field_context,
        )
        return None
    values, _ = canonical(
        raw,
        node.attrs.get("units"),
        "time",
        context=field_context,
        convention=UnitConvention(default_unit, _REFERENCE),
    )
    return _TimeField(values, raw.dtype, conversion.factor, field_context)


def _timing_table(
    values: np.ndarray, n_periods: int, n_detectors: int, context: str
) -> np.ndarray | None:
    """Accept only scalar, detector-vector, or period-by-detector timing."""
    if values.size == 1:
        return np.full((n_periods, n_detectors), float(values.reshape(-1)[0]))
    if values.shape == (n_detectors,):
        return np.broadcast_to(values, (n_periods, n_detectors))
    if values.shape == (n_periods, n_detectors):
        return values
    logger.warning(
        "%s: unusable timing shape %s; expected scalar, detector-vector, or period/detector table.",
        context,
        values.shape,
    )
    return None


def read_nexus_timing(
    group: Any, *, version: str, n_periods: int, n_detectors: int, n_bins: int, context: str
) -> tuple[float, list[T0Decode], str]:
    """Validate file timing once, before constructing zero-origin histograms."""
    resolution = _read_time(
        group, "resolution", default_unit="ps" if version == "v1" else "ns", context=context
    )
    width = None
    width_error = 0.0
    source = "resolution"
    if resolution is not None:
        if resolution.values.size == 1 and float(resolution.values.reshape(-1)[0]) > 0:
            width = float(resolution.values.reshape(-1)[0])
            width_error = float(resolution.error.reshape(-1)[0])
        else:
            logger.info(
                "%s: resolution is not a positive scalar; using an axis when available.",
                resolution.context,
            )
    if width is None:
        for source in ("raw_time", "time_of_flight", "corrected_time"):
            axis = _read_time(group, source, default_unit="us", context=context)
            if axis is None:
                continue
            kind = "edges" if axis.values.size == n_bins + 1 else "centres"
            candidate = assess_uniform_bin_width_us(
                axis.values,
                kind=kind,
                n_bins=n_bins,
                stored_dtype=axis.dtype,
                storage_scale=axis.scale,
                context=axis.context,
            )
            if isinstance(candidate, str):
                logger.warning("%s; trying the next timing axis.", candidate)
                continue
            width = candidate
            width_error = float((axis.error[0] + axis.error[-1]) / (axis.values.size - 1))
            break
    if width is None:
        raise ValueError(f"{context}: no usable time axis or positive resolution.")

    exact = _read_time(group, "time_zero", default_unit="us", context=context)
    exact_values = (
        None
        if exact is None
        else _timing_table(exact.values, n_periods, n_detectors, exact.context)
    )
    exact_errors = (
        None
        if exact_values is None
        else storage_precision_us(exact_values, stored_dtype=exact.dtype, storage_scale=exact.scale)
    )
    attr = group["counts"].attrs.get("t0_bin")
    anchors = None
    if attr is not None:
        values = np.asarray(attr)
        if (
            values.dtype.kind in "iuf"
            and np.all(np.isfinite(values))
            and np.all(values == np.floor(values))
            and np.all(values >= 1)
        ):
            table = _timing_table(values, n_periods, n_detectors, f"{context}/counts:t0_bin")
            if table is not None:
                anchors = table.astype(np.int64) - 1
        else:
            logger.warning(
                "%s/counts:t0_bin: unusable integer anchors; using exact time_zero when available.",
                context,
            )
    if (
        anchors is None
        and exact_values is None
        and (
            attr is not None
            or (exact is not None)
            or (
                group.get("time_zero") is not None
                and int(group["time_zero"].attrs.get("available", 1)) != 0
            )
        )
    ):
        raise ValueError(f"{context}: no usable t0_bin or time_zero declaration.")

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
        if conflict:
            logger.warning(
                "%s: NeXus t0_bin/time_zero declarations conflict; keeping integer anchors and dropping exact t0.",
                context,
            )
        decodes.append(
            T0Decode(
                bins,
                [None] * n_detectors if times is None or conflict else times.tolist(),
                "conflict" if conflict else "file",
            )
        )
    return width, decodes, source
