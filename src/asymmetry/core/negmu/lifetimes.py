"""Element-keyed total disappearance lifetimes for synthetic capture analysis.

Suzuki, Measday and Roalsvig, Phys. Rev. C 35, 2212 (1987), Tables III-IV,
separate mean life from capture rate. Their Eq. (4) includes bound decay:
tau^-1 = capture rate + Q/tau_free. Selected measurements retain their quoted
uncertainties and element/isotope context; source labels alone do not verify
other transcriptions. Unconfirmed and provisional values remain unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ElementLifetime:
    """Total disappearance lifetime in microseconds and its selection context.

    ``unverified`` retains a source-labelled transcription without certifying
    its measurement selection. ``unconfirmed`` identifies a known source gap;
    ``provisional`` retains an explicitly provisional value. Only
    ``selected_measurement`` identifies a directly checked published row.
    """

    symbol: str
    z: int
    name: str
    tau_us: float
    sigma_us: float | None
    source: str
    status: Literal["unverified", "unconfirmed", "provisional", "selected_measurement"] = (
        "unverified"
    )
    source_location: str | None = None
    selection_note: str | None = None


#: Reserved label for the free-μ⁻ decay-background component (τ = MUON_LIFETIME_US).
DECAY_BACKGROUND_LABEL: str = "decayBG"

_S = "SuzukiMeasdayRoalsvig1987"
_W = "WiMDA-provisional"

_REF49_LOCATION = (
    "Suzuki Table IV, p. 2219, ref. 49: M. Eckhause, R. T. Siegel, R. E. Welsh "
    "and T. A. Filippas, Nucl. Phys. 81, 575 (1966)."
)
_ELEMENT_SELECTION = (
    "Selected element-labelled measurement as printed; target isotope composition "
    "and detection channel are unspecified here. No aggregation is applied."
)

#: Element symbol to the single lifetime authority; unselected values remain unresolved.
ELEMENT_LIFETIMES: dict[str, ElementLifetime] = {
    # Source-labelled transcriptions are not independently selected measurements.
    "H": ElementLifetime("H", 1, "Hydrogen", 2.19480, 0.00006, _S),
    "He": ElementLifetime("He", 2, "Helium", 2.188, None, _W, status="provisional"),
    "Li": ElementLifetime("Li", 3, "Lithium", 2.1869, 0.0004, _S),
    "Be": ElementLifetime("Be", 4, "Beryllium", 2.16747, 0.00089, _S),
    "B": ElementLifetime("B", 5, "Boron", 2.097, 0.003, _S),
    "C": ElementLifetime("C", 6, "Carbon", 2.030, 0.001, _S),
    "N": ElementLifetime("N", 7, "Nitrogen", 1.920, 0.002, _S),
    "O": ElementLifetime("O", 8, "Oxygen", 1.795, 0.002, _S),
    "F": ElementLifetime("F", 9, "Fluorine", 1.461, 0.005, _S),
    "Ne": ElementLifetime("Ne", 10, "Neon", 1.461, 0.009, _S),
    "Na": ElementLifetime("Na", 11, "Sodium", 1.204, 0.002, _S),
    "Mg": ElementLifetime("Mg", 12, "Magnesium", 1.069, 0.002, _S),
    "Al": ElementLifetime("Al", 13, "Aluminium", 0.864, 0.001, _S),
    "Si": ElementLifetime("Si", 14, "Silicon", 0.759, 0.001, _S),
    "P": ElementLifetime("P", 15, "Phosphorus", 0.616, 0.001, _S),
    "S": ElementLifetime("S", 16, "Sulfur", 0.555, 0.001, _S),
    "Cl": ElementLifetime("Cl", 17, "Chlorine", 0.561, 0.002, _S),
    "Ar": ElementLifetime("Ar", 18, "Argon", 0.537, 0.032, _S),
    "K": ElementLifetime("K", 19, "Potassium", 0.435, 0.001, _S),
    "Ca": ElementLifetime("Ca", 20, "Calcium", 0.336, 0.001, _S),
    "Sc": ElementLifetime("Sc", 21, "Scandium", 0.317, 0.003, _S),
    "Ti": ElementLifetime("Ti", 22, "Titanium", 0.329, 0.001, _S),
    "V": ElementLifetime("V", 23, "Vanadium", 0.280, 0.002, _S),
    "Cr": ElementLifetime("Cr", 24, "Chromium", 0.259, 0.002, _S),
    "Mn": ElementLifetime("Mn", 25, "Manganese", 0.231, 0.001, _S),
    "Fe": ElementLifetime("Fe", 26, "Iron", 0.206, 0.001, _S),
    "Co": ElementLifetime("Co", 27, "Cobalt", 0.186, 0.001, _S),
    "Ni": ElementLifetime("Ni", 28, "Nickel", 0.157, 0.001, _S),
    "Cu": ElementLifetime("Cu", 29, "Copper", 0.164, 0.001, _S),
    "Zn": ElementLifetime("Zn", 30, "Zinc", 0.161, 0.001, _S),
    "Ga": ElementLifetime("Ga", 31, "Gallium", 0.163, 0.002, _S),
    "Ge": ElementLifetime("Ge", 32, "Germanium", 0.167, 0.001, _S),
    "As": ElementLifetime("As", 33, "Arsenic", 0.153, 0.001, _S),
    "Se": ElementLifetime("Se", 34, "Selenium", 0.163, 0.001, _S),
    "Br": ElementLifetime("Br", 35, "Bromine", 0.133, 0.001, _S),
    "Kr": ElementLifetime("Kr", 36, "Krypton", 0.136, None, _W, status="provisional"),
    "Rb": ElementLifetime("Rb", 37, "Rubidium", 0.137, 0.003, _S),
    "Sr": ElementLifetime("Sr", 38, "Strontium", 0.132, 0.002, _S),
    "Y": ElementLifetime("Y", 39, "Yttrium", 0.120, 0.001, _S),
    "Zr": ElementLifetime("Zr", 40, "Zirconium", 0.110, 0.001, _S),
    "Nb": ElementLifetime("Nb", 41, "Niobium", 0.092, 0.001, _S),
    "Mo": ElementLifetime("Mo", 42, "Molybdenum", 0.104, 0.001, _S),
    "Tc": ElementLifetime("Tc", 43, "Technetium", 0.095, None, _W, status="provisional"),
    # Source selection remains unresolved except for the explicitly selected rows.
    "Ru": ElementLifetime("Ru", 44, "Ruthenium", 0.0958, 0.0006, _S, status="unconfirmed"),
    "Rh": ElementLifetime(
        "Rh",
        45,
        "Rhodium",
        0.0958,
        0.0006,
        _S,
        status="selected_measurement",
        source_location=_REF49_LOCATION,
        selection_note=_ELEMENT_SELECTION,
    ),
    "Pd": ElementLifetime(
        "Pd",
        46,
        "Palladium",
        0.0960,
        0.0006,
        _S,
        status="selected_measurement",
        source_location=_REF49_LOCATION,
        selection_note=_ELEMENT_SELECTION,
    ),
    "Ag": ElementLifetime(
        "Ag",
        47,
        "Silver",
        0.0870,
        0.0015,
        _S,
        status="selected_measurement",
        source_location="Suzuki Table IV, p. 2219, ref. a (this experiment).",
        selection_note=(
            "Selected for directly documented elemental self-supporting target and decay-electron "
            "timing in Section II; isotope composition is unspecified. Other Table IV measurements "
            "(refs. 40, 54, 49 and 76) are alternatives, not averaged or ranked by uncertainty."
        ),
    ),
    "Cd": ElementLifetime("Cd", 48, "Cadmium", 0.0906, 0.0007, _S, status="unconfirmed"),
    "Sn": ElementLifetime("Sn", 50, "Tin", 0.0907, 0.0008, _S, status="unconfirmed"),
    "Sb": ElementLifetime("Sb", 51, "Antimony", 0.0924, 0.0009, _S, status="unconfirmed"),
    "Te": ElementLifetime("Te", 52, "Tellurium", 0.104, 0.001, _S, status="unconfirmed"),
    "I": ElementLifetime("I", 53, "Iodine", 0.0856, 0.0006, _S, status="unconfirmed"),
    "Cs": ElementLifetime("Cs", 55, "Cesium", 0.088, 0.002, _S, status="unconfirmed"),
    "Ba": ElementLifetime("Ba", 56, "Barium", 0.0949, 0.0006, _S, status="unconfirmed"),
    "La": ElementLifetime("La", 57, "Lanthanum", 0.0899, 0.0007, _S, status="unconfirmed"),
    "Ce": ElementLifetime("Ce", 58, "Cerium", 0.0840, 0.0006, _S, status="unconfirmed"),
    "Pr": ElementLifetime("Pr", 59, "Praseodymium", 0.0721, 0.0006, _S, status="unconfirmed"),
    "Nd": ElementLifetime("Nd", 60, "Neodymium", 0.0784, 0.0007, _S, status="unconfirmed"),
    "Sm": ElementLifetime("Sm", 62, "Samarium", 0.079, 0.001, _S, status="unconfirmed"),
    "Gd": ElementLifetime("Gd", 64, "Gadolinium", 0.0806, 0.0008, _S, status="unconfirmed"),
    "Tb": ElementLifetime("Tb", 65, "Terbium", 0.0762, 0.0007, _S, status="unconfirmed"),
    "Ho": ElementLifetime(
        "Ho",
        67,
        "Holmium",
        0.0749,
        0.0006,
        _S,
        status="selected_measurement",
        source_location=(
            "Suzuki Table IV, p. 2220, ref. 54: T. A. Filippas et al., Phys. Lett. 6, 118 (1963)."
        ),
        selection_note=_ELEMENT_SELECTION,
    ),
    "Er": ElementLifetime(
        "Er",
        68,
        "Erbium",
        0.0744,
        0.0015,
        _S,
        status="selected_measurement",
        source_location="Suzuki Table IV, p. 2220, ref. a (this experiment).",
        selection_note=(
            "Selected authors' measurement with an elemental self-supporting target and "
            "decay-electron timing described in Section II; isotope composition is unspecified."
        ),
    ),
    "Tm": ElementLifetime("Tm", 69, "Thulium", 0.074, 0.002, _S, status="unconfirmed"),
    # Further source-labelled transcriptions and provisional entries.
    "Hf": ElementLifetime("Hf", 72, "Hafnium", 0.075, 0.001, _S),
    "Ta": ElementLifetime("Ta", 73, "Tantalum", 0.0755, 0.0006, _S),
    "W": ElementLifetime("W", 74, "Tungsten", 0.0765, 0.0008, _S),
    "Re": ElementLifetime("Re", 75, "Rhenium", 0.076, None, _W, status="provisional"),
    "Os": ElementLifetime("Os", 76, "Osmium", 0.078, None, _W, status="provisional"),
    "Ir": ElementLifetime("Ir", 77, "Iridium", 0.074, None, _W, status="provisional"),
    "Pt": ElementLifetime("Pt", 78, "Platinum", 0.074, None, _W, status="provisional"),
    "Au": ElementLifetime("Au", 79, "Gold", 0.0728, 0.0005, _S),
    "Hg": ElementLifetime("Hg", 80, "Mercury", 0.076, 0.001, _S),
    "Tl": ElementLifetime("Tl", 81, "Thallium", 0.0704, 0.0008, _S),
    "Pb": ElementLifetime("Pb", 82, "Lead", 0.0747, 0.0004, _S),
    "Bi": ElementLifetime("Bi", 83, "Bismuth", 0.0735, 0.0004, _S),
    # Element labels do not resolve the source's actinide isotope/channel alternatives.
    "Th": ElementLifetime("Th", 90, "Thorium", 0.0780, 0.0003, _S, status="unconfirmed"),
    "U": ElementLifetime("U", 92, "Uranium", 0.0775, 0.0002, _S, status="unconfirmed"),
    "Np": ElementLifetime("Np", 93, "Neptunium", 0.0720, 0.0007, _S, status="unconfirmed"),
}


def lifetime(symbol: str) -> ElementLifetime:
    """Return the table entry for ``symbol`` (raises :class:`KeyError` if absent)."""
    try:
        return ELEMENT_LIFETIMES[symbol]
    except KeyError:
        raise KeyError(
            f"No capture lifetime entry for element {symbol!r}. "
            f"Available: {sorted(ELEMENT_LIFETIMES, key=lambda s: ELEMENT_LIFETIMES[s].z)}"
        ) from None


def tau_us(symbol: str) -> float:
    """Return the total disappearance lifetime (µs) for ``symbol``."""
    return lifetime(symbol).tau_us


def has_element(symbol: str) -> bool:
    """Return True if ``symbol`` is in the table."""
    return symbol in ELEMENT_LIFETIMES


def elements() -> list[str]:
    """Element symbols present, ordered by atomic number Z."""
    return sorted(ELEMENT_LIFETIMES, key=lambda s: ELEMENT_LIFETIMES[s].z)
