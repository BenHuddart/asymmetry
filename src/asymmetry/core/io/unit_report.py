"""Inspection of retained NeXus declarations, without rereading source files."""

from dataclasses import asdict, dataclass
from typing import Any

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io.physical_units import CANONICAL_UNITS, nexus_physical_unit, nexus_quantity


@dataclass(frozen=True)
class UnitRecord:
    """A source declaration and the interpretation available for that source."""

    quantity: str
    path: str
    original_value: Any
    declared_unit: Any
    declaration: str
    interpreted_value: Any
    interpreted_unit: str | None
    interpretation: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe inspection record."""
        return asdict(self)

    def describe(self) -> str:
        """Compact source information shared by the GUI and terminal."""
        declaration = (
            str(self.declared_unit) if self.declaration == "declared" else self.declaration
        )
        interpreted = (
            self.interpreted_value[0]
            if isinstance(self.interpreted_value, list) and len(self.interpreted_value) == 1
            else self.interpreted_value
        )
        result = (
            f" → {interpreted} {self.interpreted_unit}"
            if self.interpreted_value is not None
            else ""
        )
        original = (
            self.original_value[0]
            if isinstance(self.original_value, list) and len(self.original_value) == 1
            else self.original_value
        )
        return f"{original} {declaration}{result}; {self.explanation} [{self.path}]"


def _bounded_value(value):
    """Keep array inspection bounded independently of histogram length."""
    if isinstance(value, list) and len(value) > 8:
        return {"length": len(value), "preview": value[:8]}
    return value


def _converted_summary(value, raw, dimension):
    """Affine physical-unit conversion preserves extrema and arithmetic means."""
    if isinstance(value, dict):
        return {
            "shape": value["shape"],
            **{
                key: float(nexus_quantity(value[key], raw, dimension))
                for key in ("min", "max", "mean")
            },
        }
    return _bounded_value(nexus_quantity(value, raw, dimension).tolist())


def unit_report(
    dataset: MuonDataset, *, interpretations: dict[str, str] | None = None
) -> list[UnitRecord]:
    """Describe original declarations and recorded NeXus interpretation policy.

    Raw engineering logs are not assigned a physical dimension by their names.
    Historical projects without the policy marker retain declarations without
    claiming that current conversion rules produced their saved quantities.
    """
    metadata = dataset.metadata
    tree = metadata.get("nexus_fields", {})
    if not tree:
        return []
    current = metadata.get("nexus_unit_policy") == "permissive-1"
    records = []
    timing_names = {
        "raw_time",
        "corrected_time",
        "time_of_flight",
        "resolution",
        "time_zero",
        "first_good_time",
        "last_good_time",
        "dead_time",
        "deadtime",
        "deadtimes",
    }

    def walk(node, path):
        if (
            isinstance(node, dict)
            and not (set(node) <= {"value", "@attrs"} and "value" in node)
            and "kind" not in node
        ):
            for name, child in sorted(node.items()):
                if name != "@attrs":
                    walk(child, f"{path}/{name}")
            return
        name = path.rsplit("/", 1)[-1]
        attrs = node.get("@attrs", {}) if isinstance(node, dict) else {}
        value = node.get("value", node) if isinstance(node, dict) else node
        raw = attrs.get("units")
        token = raw[0] if isinstance(raw, list) and len(raw) == 1 else raw
        parent = path.rsplit("/", 1)[0].lstrip("/")
        logs = metadata.get("nexus_time_series", {})
        is_log = parent in logs and name in {"time", "value", "values"}
        dimension = {"/sample/temperature": "temperature", "/sample/magnetic_field": "field"}.get(
            path
        )
        if dimension is None and not (is_log or name in timing_names or "units" in attrs):
            return
        if "units" not in attrs:
            declaration = "Units not declared"
        elif token is None or isinstance(token, str) and not token.strip():
            declaration = "Unit declaration blank"
        else:
            declaration = "declared"
        interpreted = None
        interpreted_unit = None
        interpretation = "raw"
        explanation = "Raw data; not assigned a physical interpretation"
        quantity = dimension or ("log clock" if is_log and name == "time" else name)
        if is_log and name != "time":
            if parent in {"Field_Main", "Field_Z", "Field_X", "Field_Y"}:
                dimension = "field"
            elif parent == metadata.get("sample_temperature_log_source"):
                dimension = "temperature"
        dimension = (interpretations or {}).get(path, dimension)
        numeric = (
            isinstance(value, dict)
            and all(key in value for key in ("shape", "min", "max", "mean"))
            or isinstance(value, (int, float))
            or isinstance(value, list)
            and len(value) <= 64
            and all(isinstance(v, (int, float)) for v in value)
        )
        if not current:
            interpretation = "unrecorded"
            explanation = "Interpretation not recorded"
        elif dimension is not None:
            unit = nexus_physical_unit(raw, dimension)
            interpreted_unit = CANONICAL_UNITS[dimension]
            if numeric:
                interpreted = _converted_summary(value, raw, dimension)
            interpretation = "declared" if unit else "compatibility"
            explanation = (
                f"Converted to {interpreted_unit}"
                if unit
                else f"Numeric value retained; compatibility assumption {interpreted_unit}"
            )
            if unit is None and declaration == "declared":
                incompatible = any(
                    nexus_physical_unit(raw, other) is not None
                    for other in CANONICAL_UNITS
                    if other != dimension
                )
                explanation = (
                    f"Unit incompatible with {dimension}" if incompatible else "Unit unrecognised"
                ) + f"; numeric value retained as {interpreted_unit}"
            if not numeric:
                explanation = "Stored value is not numeric; no conversion shown"
            if path == "/sample/magnetic_field" and metadata.get("field_source"):
                explanation += f"; effective field source: {metadata['field_source']}"
            if metadata.get("temperature_selection") and dimension == "temperature":
                explanation += "; explicit temperature selection recorded separately"
        elif name in timing_names:
            interpretation = "timing"
            explanation = "Timing declaration; unusable declarations may be ignored in favour of other file timing"
            convention = (
                "ps"
                if name == "resolution" and metadata.get("nexus_version") == "v1"
                else "ns"
                if name == "resolution"
                else "us"
            )
            if declaration != "declared":
                explanation += f"; documented format convention {convention} (Cottrell v2 rev. 8, pp. 14, 49–51)"
            if name == metadata.get("nexus_timing_source"):
                explanation += "; selected bin-width source"
                interpreted_unit = "us"
                if dataset.run is not None and dataset.run.histograms:
                    interpreted = dataset.run.histograms[0].bin_width
            if name in {
                "raw_time",
                "corrected_time",
                "time_of_flight",
                "resolution",
            } and name != metadata.get("nexus_timing_source"):
                interpretation = "ignored"
                explanation = "Not selected as the acquisition bin-width source"
            if name in {"dead_time", "deadtime", "deadtimes"}:
                unit = nexus_physical_unit(raw, "time")
                explicit = (
                    metadata.get("nexus_version") == "v2"
                    and unit is not None
                    and unit.symbol in {"us", "ns"}
                )
                interpreted_unit = "us"
                if numeric:
                    interpreted = (
                        _converted_summary(value, raw, "time")
                        if explicit
                        else _bounded_value(value)
                    )
                interpretation = "declared" if explicit else "compatibility"
                explanation = (
                    "Explicit v2 deadtime converted to us"
                    if explicit
                    else "Numeric value retained; compatibility assumption us"
                )
        elif is_log and name == "time":
            explanation = "Clock retained as stored; run-active samples use time >= 0"
        records.append(
            UnitRecord(
                quantity,
                path,
                _bounded_value(value),
                raw,
                declaration,
                interpreted,
                interpreted_unit,
                interpretation,
                explanation,
            )
        )

    walk(tree, "")
    return records
