"""Declared physical units at file boundaries, with exact SI conversions.

SI prefixes and Celsius follow the BIPM SI Brochure, 9th edition, sections
2.3.1 and 3: https://www.bipm.org/en/publications/si-brochure .
Magnetic conversion uses the canonical gauss-to-tesla constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from asymmetry.core.utils.constants import CELSIUS_ZERO_K, GAUSS_TO_TESLA

Dimension = Literal["time", "field", "temperature"]
CANONICAL_UNITS: dict[Dimension, str] = {"time": "us", "field": "G", "temperature": "K"}


@dataclass(frozen=True)
class UnitDeclaration:
    """The file declaration or sourced format convention used for a conversion."""

    unit: str
    source: Literal["file", "format"]
    reference: str = ""


@dataclass(frozen=True)
class UnitConvention:
    """A missing-unit convention from a documented format, rather than an instrument guess."""

    unit: str
    reference: str


@dataclass(frozen=True)
class PhysicalUnit:
    """Affine conversion into the dimension's canonical unit."""

    dimension: Dimension
    symbol: str
    factor: float
    offset: float = 0.0
    words: tuple[str, ...] = ()


_UNITS = (
    PhysicalUnit("time", "s", 1e6, words=("second", "seconds")),
    PhysicalUnit("time", "ms", 1e3, words=("millisecond", "milliseconds", "milli.second")),
    PhysicalUnit("time", "us", 1, words=("microsecond", "microseconds", "micro.second")),
    PhysicalUnit("time", "ns", 1e-3, words=("nanosecond", "nanoseconds", "nano.second")),
    PhysicalUnit("time", "ps", 1e-6, words=("picosecond", "picoseconds", "pico.second")),
    PhysicalUnit("field", "G", 1, words=("gauss",)),
    PhysicalUnit("field", "kG", 1e3, words=("kilogauss",)),
    PhysicalUnit("field", "T", 1 / GAUSS_TO_TESLA, words=("tesla", "teslas")),
    PhysicalUnit("field", "mT", 1e-3 / GAUSS_TO_TESLA, words=("millitesla", "milliteslas")),
    PhysicalUnit("field", "uT", 1e-6 / GAUSS_TO_TESLA, words=("microtesla", "microteslas")),
    PhysicalUnit("temperature", "K", 1, words=("kelvin", "kelvins")),
    PhysicalUnit("temperature", "mK", 1e-3, words=("millikelvin", "millikelvins")),
    PhysicalUnit(
        "temperature",
        "°C",
        1,
        CELSIUS_ZERO_K,
        ("degC", "celsius", "degree celsius", "degrees celsius", "centigrade"),
    ),
)


def unit_text(raw: object, *, context: str = "unit attribute") -> str:
    """Decode a scalar unit attribute, including Latin-1 and both micro glyphs."""
    if raw is None:
        return ""
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            raise ValueError(f"{context}: a unit attribute must contain exactly one token.")
        raw = raw.reshape(-1)[0]
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="surrogateescape")
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw.decode("latin-1")
    return str(raw).strip("\x00 \t\r\n").replace("µ", "u").replace("μ", "u")


def physical_unit(raw: object) -> PhysicalUnit | None:
    """Match case-sensitive symbols and case-insensitive spelled-out unit names."""
    token = unit_text(raw)
    for unit in _UNITS:
        if token == unit.symbol or token.casefold() in tuple(
            word.casefold() for word in unit.words
        ):
            return unit
    return None


def canonical(
    value: object,
    raw_unit: object,
    dimension: Dimension,
    *,
    context: str,
    convention: UnitConvention | None = None,
) -> tuple[np.ndarray, UnitDeclaration]:
    """Convert a declared value or array; unsupported or missing units raise."""
    token = unit_text(raw_unit, context=context)
    declaration = UnitDeclaration(token, "file")
    if not token and convention is not None:
        token = convention.unit
        declaration = UnitDeclaration(token, "format", convention.reference)
    unit = physical_unit(token)
    if unit is None or unit.dimension != dimension:
        supported = ", ".join(u.symbol for u in _UNITS if u.dimension == dimension)
        raise ValueError(
            f"{context}: {dimension} unit {token or '<missing>'!r}; supported units: {supported}."
        )
    return np.asarray(value, dtype=float) * unit.factor + unit.offset, declaration


def uniform_bin_width_us(
    axis: np.ndarray,
    *,
    kind: Literal["edges", "centres"],
    n_bins: int,
    stored_dtype: np.dtype,
    context: str,
    storage_scale: float = 1.0,
) -> float:
    """Validate a canonical axis using four ULPs of its original storage precision."""
    values = np.asarray(axis, dtype=float)
    expected = n_bins + 1 if kind == "edges" else n_bins
    if values.ndim != 1 or values.size != expected or values.size < 2:
        raise ValueError(
            f"{context}: time axis requires {expected} {kind} and at least two points."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context}: time axis must contain only finite values.")
    steps = np.diff(values)
    if np.any(steps <= 0):
        raise ValueError(f"{context}: time axis must be strictly increasing.")
    width = float((values[-1] - values[0]) / (values.size - 1))
    tolerance = 4 * float(np.spacing(np.max(np.abs(values))))
    if np.issubdtype(stored_dtype, np.floating):
        magnitude = np.asarray(np.max(np.abs(values)) / storage_scale, dtype=stored_dtype)
        tolerance = max(tolerance, 4 * float(np.spacing(magnitude)) * storage_scale)
    if np.any(np.abs(steps - width) > tolerance):
        raise ValueError(f"{context}: time axis must have uniform bin widths.")
    return width
