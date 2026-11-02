"""Dimensioned NeXus datasets and their recorded unit declarations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from asymmetry.core.io.physical_units import (
    CANONICAL_UNITS,
    Dimension,
    UnitConvention,
    canonical,
    physical_unit,
    uniform_bin_width_us,
    unit_text,
)

DEFINITION = "https://nmi3.eu/nexus_instrument_definitions_for_muon_data-get_file=nexus_instrument_definitions_for_muon_data.pdf.pdf"
# Cottrell, muon NeXus v2 rev. 8, pp. 11, 14, 49–52: timing conventions.
MISSING_UNITS = {
    name: UnitConvention("us", f"{DEFINITION} pp. 11, 14, 49–51")
    for name in (
        "raw_time",
        "corrected_time",
        "time_zero",
        "first_good_time",
        "last_good_time",
        "dead_time",
        "deadtime",
        "deadtimes",
    )
}
MISSING_RESOLUTION = {
    "v1": UnitConvention("ps", f"{DEFINITION} p. 14"),
    "v2": UnitConvention("ns", f"{DEFINITION} p. 49"),
}


@dataclass
class NexusUnits:
    """Per-file conversion boundary; declarations contain no duplicate measurement values."""

    entry: Any
    entry_name: str
    source_file: str
    version: str
    declarations: dict[str, Any] = field(default_factory=dict)

    def node(self, path: str):
        """Resolve paths in both h5py and the HDF4 group adapter."""
        node = self.entry
        for part in path.split("/"):
            node = node.get(part)
            if node is None:
                return None
        return node

    def context(self, path: str) -> str:
        """Identify the original file and dataset in boundary errors."""
        return f"{self.source_file}: /{self.entry_name}/{path}"

    def read(self, path: str, dimension: Dimension = "time") -> np.ndarray:
        """Read a dimensioned dataset, or return an empty array when absent."""
        node = self.node(path)
        if node is None:
            return np.array([], dtype=float)
        name = path.rsplit("/", 1)[-1]
        convention = (
            MISSING_RESOLUTION[self.version] if name == "resolution" else MISSING_UNITS.get(name)
        )
        value, declaration = canonical(
            node[()],
            node.attrs.get("units"),
            dimension,
            context=self.context(path),
            convention=convention,
        )
        self.declarations.setdefault(dimension, {})[f"/{self.entry_name}/{path}"] = asdict(
            declaration
        )
        if dimension == "time" and not np.all(np.isfinite(value)):
            raise ValueError(f"{self.context(path)}: time values must be finite.")
        return value

    def scalar(self, path: str, dimension: Dimension, default: float | None = None) -> float | None:
        """Read a scalar quantity without inventing a unit for a present dataset."""
        value = self.read(path, dimension)
        if self.node(path) is None:
            return default
        if value.size != 1:
            raise ValueError(f"{self.context(path)}: expected one {dimension} value.")
        return float(value.reshape(-1)[0])

    def bin_width(self, prefix: str, n_bins: int) -> float:
        """Validate every provided axis and cross-check the declared resolution."""
        widths = []
        tolerance = 0.0
        for name in ("corrected_time", "raw_time", "time"):
            path = f"{prefix}/{name}"
            node = self.node(path)
            if node is None:
                continue
            axis = self.read(path)
            unit = physical_unit(self.declarations["time"][f"/{self.entry_name}/{path}"]["unit"])
            raw = np.asarray(node[()])
            if n_bins == 1 and axis.shape == (1,):
                continue
            kind = "edges" if axis.size == n_bins + 1 else "centres"
            width = uniform_bin_width_us(
                axis,
                kind=kind,
                n_bins=n_bins,
                stored_dtype=raw.dtype,
                context=self.context(path),
                storage_scale=unit.factor,
            )
            widths.append(width)
            if np.issubdtype(raw.dtype, np.floating):
                tolerance = max(tolerance, 4 * float(np.spacing(np.max(np.abs(raw)))) * unit.factor)
        if widths and any(abs(width - widths[0]) > tolerance for width in widths[1:]):
            raise ValueError(f"{self.context(prefix)}: time axes disagree on bin width.")
        path = f"{prefix}/resolution"
        resolution = self.scalar(path, "time")
        if resolution is not None:
            if resolution < 0:
                raise ValueError(
                    f"{self.context(path)}: histogram resolution must be positive or zero (not applicable)."
                )
            if resolution > 0:
                declared = self.declarations["time"][f"/{self.entry_name}/{path}"]
                tolerance += physical_unit(declared["unit"]).factor / 2
                if widths and abs(widths[0] - resolution) > tolerance:
                    raise ValueError(f"{self.context(path)}: resolution disagrees with time axis.")
                if not widths:
                    return resolution
        if not widths:
            raise ValueError(f"{self.context(prefix)}: no usable time axis or resolution.")
        return widths[0]

    def clock_seconds(self, path: str, node: Any) -> np.ndarray:
        """Read declared NXlog times and their optional clock scaling factor.

        https://manual.nexusformat.org/classes/base_classes/NXlog.html specifies
        that scaling_factor applies before interpreting the declared time unit.
        """
        clock_path = f"{path}/time"
        time_values, declaration = canonical(
            node["time"][()],
            node["time"].attrs.get("units"),
            "time",
            context=self.context(clock_path),
        )
        scale = np.asarray(node["time"].attrs.get("scaling_factor", 1.0))
        if scale.size != 1 or scale.dtype.kind not in "fiu" or not np.isfinite(scale).all():
            raise ValueError(f"{self.context(clock_path)}: scaling_factor must be a finite scalar.")
        if not np.isfinite(time_values).all():
            raise ValueError(f"{self.context(clock_path)}: NXlog times must be finite.")
        factor = float(scale.reshape(-1)[0])
        recorded = asdict(declaration)
        if "scaling_factor" in node["time"].attrs:
            recorded["scaling_factor"] = factor
        self.declarations.setdefault("time", {})[f"/{self.entry_name}/{clock_path}"] = recorded
        return time_values * factor / 1e6

    def log_values(
        self, path: str, node: Any, value_name: str, label: str
    ) -> tuple[np.ndarray, str]:
        """Canonicalize physical NXlogs while retaining unrelated engineering logs."""
        values = np.asarray(node[value_name][()])
        token = unit_text(node[value_name].attrs.get("units"))
        unit = physical_unit(token)
        text = f"{path} {label}".casefold()
        dimension = unit.dimension if unit is not None else None
        if "temp" in text:
            dimension = "temperature"
        elif path in {"Field_Main", "Field_X", "Field_Y", "Field_Z"}:
            dimension = "field"
        if dimension not in {"field", "temperature"}:
            return values, token
        converted = self.read(f"{path}/{value_name}", dimension)
        return converted, CANONICAL_UNITS[dimension]
