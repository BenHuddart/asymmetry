"""Physical quantities in PSI BIN/MDU text header fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from asymmetry.core.io.physical_units import UnitConvention, parse_quantity

# Header text conventions require a sourced PSI profile review.
MISSING_UNITS = {
    "temperature": UnitConvention("K", "unverified: PSI header temperature convention"),
    "field": UnitConvention("G", "unverified: PSI header field convention"),
}


@dataclass(frozen=True)
class HeaderQuantities:
    """Canonical header values and the declarations used to read them."""

    temperature: float
    field: float
    declarations: dict[str, Any]


def read_header_quantities(
    source_file: str, temperature_text: str, field_text: str, *, mean_temperature: str = ""
) -> HeaderQuantities:
    """Read adjacent units; MDU's nonzero mean temperature takes precedence."""
    values, declarations = {}, {}
    for dimension, text in (("temperature", temperature_text), ("field", field_text)):
        value, declaration = parse_quantity(
            text,
            dimension,
            context=f"{source_file}: header/{dimension}",
            convention=MISSING_UNITS[dimension],
            default=0.0,
        )
        values[dimension] = value
        if declaration is not None:
            declarations[dimension] = {f"header/{dimension}": asdict(declaration)}
    if mean_temperature:
        mean, declaration = parse_quantity(
            mean_temperature,
            "temperature",
            context=f"{source_file}: header/mean_temp",
            convention=MISSING_UNITS["temperature"],
            default=0.0,
        )
        if mean != 0 or values["temperature"] == 0:
            values["temperature"] = mean
            declarations.pop("temperature", None)
            if declaration is not None:
                declarations["temperature"] = {"header/mean_temp": asdict(declaration)}
    return HeaderQuantities(values["temperature"], values["field"], declarations)
