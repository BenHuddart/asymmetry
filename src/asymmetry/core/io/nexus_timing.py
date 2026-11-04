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
    raw_unit = node.attrs.get("units")
    conversion = (
        physical_unit(unit_text(raw_unit) or default_unit, dimension="time")
        if not isinstance(raw_unit, np.ndarray) or raw_unit.size == 1
        else None
    )
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
        raw_unit,
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
) -> tuple[float, list[T0Decode], str, np.ndarray]:
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
        conflicts = np.zeros(n_detectors, dtype=bool)
        if times is not None and anchors is not None:
            errors = exact_errors[period] + (bins + 1) * width_error
            conflicts = (times < bins * width - errors) | (times > (bins + 1) * width + errors)
        if np.any(conflicts):
            logger.warning(
                "%s: NeXus t0_bin/time_zero declarations conflict; keeping integer anchors and dropping exact t0 only for conflicting detectors.",
                context,
            )
        decodes.append(
            T0Decode(
                bins,
                [None] * n_detectors
                if times is None
                else [None if conflicts[i] else float(value) for i, value in enumerate(times)],
                "conflict" if np.any(conflicts) else "file",
            )
        )
    windows = _raw_good_windows(
        group,
        version=version,
        n_periods=n_periods,
        n_detectors=n_detectors,
        n_bins=n_bins,
        width=width,
        width_error=width_error,
        context=context,
    )
    return width, decodes, source, windows


def _good_bin_declaration(
    values: Any, version: str, n_periods: int, n_detectors: int, context: str
) -> np.ndarray | None:
    """Decode the supported ISIS integer spelling, including numeric text attributes."""
    if values is None:
        return None
    try:
        raw = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        logger.warning("%s: unusable good-bin declaration; trying other file bounds.", context)
        return None
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
        logger.warning("%s: unusable good-bin declaration; trying other file bounds.", context)
        return None
    if raw.size != 1 and not (version == "v2" and n_periods == 1 and raw.shape == (n_detectors,)):
        logger.warning(
            "%s: unsupported good-bin shape %s; trying other file bounds.", context, raw.shape
        )
        return None
    return _timing_table(raw, n_periods, n_detectors, context)


def _raw_good_windows(
    group: Any,
    *,
    version: str,
    n_periods: int,
    n_detectors: int,
    n_bins: int,
    width: float,
    width_error: float,
    context: str,
) -> np.ndarray:
    """Inclusive centre membership on zero-origin acquisition bins, before t0.

    Physical bounds reference raw_time (Cottrell p. 51). Centre membership is
    Asymmetry's whole-bin policy; the definition does not prescribe rounding.
    """
    bounds = np.empty((n_periods, n_detectors, 2), dtype=np.int64)
    centres = (np.arange(n_bins) + 0.5) * width
    for endpoint, name in enumerate(("first_good", "last_good")):
        integer = _good_bin_declaration(
            group["counts"].attrs.get(f"{name}_bin"),
            version,
            n_periods,
            n_detectors,
            f"{context}/counts:{name}_bin",
        )
        if integer is None and version == "v1":
            node = group.get(f"{name}_bin")
            integer = _good_bin_declaration(
                None if node is None else np.asarray(node),
                version,
                n_periods,
                n_detectors,
                f"{context}/{name}_bin",
            )
        if integer is not None:
            bounds[:, :, endpoint] = np.maximum(0, integer - 1).astype(np.int64)
            continue
        bounds[:, :, endpoint] = 0 if endpoint == 0 else n_bins - 1
        if version != "v2":
            continue
        field = _read_time(group, f"{name}_time", default_unit="us", context=context)
        if field is None:
            continue
        # V2 defines scalar or [period,spectrum], not a period-only vector.
        if (
            field.values.size != 1
            and field.values.shape != (n_periods, n_detectors)
            and not (n_periods == 1 and field.values.shape == (n_detectors,))
        ):
            logger.warning(
                "%s: unsupported good-time shape %s; keeping acquired extent.",
                field.context,
                field.values.shape,
            )
            continue
        table = _timing_table(field.values, n_periods, n_detectors, field.context)
        errors = storage_precision_us(table, stored_dtype=field.dtype, storage_scale=field.scale)
        for period in range(n_periods):
            for detector in range(n_detectors):
                delta = centres - table[period, detector]
                tolerance = errors[period, detector] + (np.arange(n_bins) + 0.5) * width_error
                membership = delta >= -tolerance if endpoint == 0 else delta <= tolerance
                indices = np.flatnonzero(membership)
                bounds[period, detector, endpoint] = (
                    int(indices[0 if endpoint == 0 else -1])
                    if indices.size
                    else (n_bins if endpoint == 0 else -1)
                )
    # Negative ends mean full range in Histogram; encode emptiness explicitly.
    empty = bounds[:, :, 0] > bounds[:, :, 1]
    bounds[empty] = (n_bins, n_bins - 1)
    return bounds
