"""EXPERIMENTAL — WORK IN PROGRESS. Negative-muon (μ⁻) capture-lifetime analysis.

This API is UNVALIDATED against real μ⁻ elemental-analysis data. No μ⁻ corpus
exists in this project; every result here has been exercised only against
synthetic histograms. The element lifetime values are literature-anchored
(Suzuki, Measday & Roalsvig, Phys. Rev. C 35, 2212 (1987), via Blundell et al.,
Muon Spectroscopy: An Introduction, OUP 2022, Table C.1), but the fitting,
capture-ratio, and background machinery have NOT been checked against an
established tool (WiMDA, Mantid) on measured data. The API, parameter names, and
return shapes MAY CHANGE without notice. Do not rely on results for publication
without independent verification. This feature is deliberately NOT exposed in the
GUI fit builders. Promotion trigger for a GUI: real ISIS μ⁻ data AND a user.

Negative-muon capture lifetimes: an element-keyed table of muonic-atom total
disappearance lifetimes τ(Z) = 1/(Λ_capture + Λ_bound-decay), the seeds for the
multi-exponential capture fit.

Source: Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An
Introduction* (OUP, 2022), Appendix C, Table C.1 — values combined from
T. Suzuki, D. F. Measday & J. P. Roalsvig, Phys. Rev. C 35, 2212 (1987).
A few entries marked WiMDA-provisional are from WiMDA's older table where
Table C.1 has no value. WiMDA's table is NOT trusted for values (see the
porting study, comparison.md §5).

Provenance note: WiMDA's ``mystrings`` table (``NegMuAnalyse.pas:104–120``) was
the workflow reference, **not** the value source. It has a 69-vs-67 length
mismatch, the ``'Ti'``→``'Tl'`` symbol bug, and several value divergences
(Ne 1.520→1.461, Zn 0.169→0.161, Sr 0.142→0.132, Ba 0.072→0.0949).
Adopt Table C.1; cite Suzuki/Measday/Roalsvig 1987.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementLifetime:
    """Literature capture lifetime for one element."""

    symbol: str
    z: int
    name: str
    tau_us: float
    sigma_us: float | None
    source: str  # "SuzukiMeasdayRoalsvig1987" | "WiMDA-provisional"


#: Reserved label for the free-μ⁻ decay-background component (τ = MUON_LIFETIME_US).
DECAY_BACKGROUND_LABEL: str = "decayBG"

_S = "SuzukiMeasdayRoalsvig1987"
_W = "WiMDA-provisional"

#: Element symbol → ElementLifetime. Transcribed from plan.md §2.
#: ⚠confirm rows (period-5 transition cluster and lanthanides/actinides) are
#: included at the plan's stated values; they have not been independently
#: verified against the textbook table layout and are NOT test-spot-check anchors.
ELEMENT_LIFETIMES: dict[str, ElementLifetime] = {
    # --- Confident rows (cross-validated vs WiMDA to ≤ rounding) ---
    "H": ElementLifetime("H", 1, "Hydrogen", 2.19480, 0.00006, _S),
    "He": ElementLifetime("He", 2, "Helium", 2.188, None, _W),
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
    "Kr": ElementLifetime("Kr", 36, "Krypton", 0.136, None, _W),
    "Rb": ElementLifetime("Rb", 37, "Rubidium", 0.137, 0.003, _S),
    "Sr": ElementLifetime("Sr", 38, "Strontium", 0.132, 0.002, _S),
    "Y": ElementLifetime("Y", 39, "Yttrium", 0.120, 0.001, _S),
    "Zr": ElementLifetime("Zr", 40, "Zirconium", 0.110, 0.001, _S),
    "Nb": ElementLifetime("Nb", 41, "Niobium", 0.092, 0.001, _S),
    "Mo": ElementLifetime("Mo", 42, "Molybdenum", 0.104, 0.001, _S),
    "Tc": ElementLifetime("Tc", 43, "Technetium", 0.095, None, _W),
    # --- ⚠confirm rows (period-5 transition + lanthanides/actinides) ---
    "Ru": ElementLifetime("Ru", 44, "Ruthenium", 0.0958, 0.0006, _S),
    "Rh": ElementLifetime("Rh", 45, "Rhodium", 0.0960, 0.0006, _S),
    "Pd": ElementLifetime("Pd", 46, "Palladium", 0.0885, 0.0006, _S),
    "Ag": ElementLifetime("Ag", 47, "Silver", 0.0906, 0.0007, _S),
    "Cd": ElementLifetime("Cd", 48, "Cadmium", 0.0906, 0.0007, _S),
    "Sn": ElementLifetime("Sn", 50, "Tin", 0.0907, 0.0008, _S),
    "Sb": ElementLifetime("Sb", 51, "Antimony", 0.0924, 0.0009, _S),
    "Te": ElementLifetime("Te", 52, "Tellurium", 0.104, 0.001, _S),
    "I": ElementLifetime("I", 53, "Iodine", 0.0856, 0.0006, _S),
    "Cs": ElementLifetime("Cs", 55, "Cesium", 0.088, 0.002, _S),
    "Ba": ElementLifetime("Ba", 56, "Barium", 0.0949, 0.0006, _S),
    "La": ElementLifetime("La", 57, "Lanthanum", 0.0899, 0.0007, _S),
    "Ce": ElementLifetime("Ce", 58, "Cerium", 0.0840, 0.0006, _S),
    "Pr": ElementLifetime("Pr", 59, "Praseodymium", 0.0721, 0.0006, _S),
    "Nd": ElementLifetime("Nd", 60, "Neodymium", 0.0784, 0.0007, _S),
    "Sm": ElementLifetime("Sm", 62, "Samarium", 0.079, 0.001, _S),
    "Gd": ElementLifetime("Gd", 64, "Gadolinium", 0.0806, 0.0008, _S),
    "Tb": ElementLifetime("Tb", 65, "Terbium", 0.0762, 0.0007, _S),
    "Ho": ElementLifetime("Ho", 67, "Holmium", 0.079, 0.001, _S),
    "Er": ElementLifetime("Er", 68, "Erbium", 0.0749, 0.0006, _S),
    "Tm": ElementLifetime("Tm", 69, "Thulium", 0.074, 0.002, _S),
    # --- Confident rows (continued) ---
    "Hf": ElementLifetime("Hf", 72, "Hafnium", 0.075, 0.001, _S),
    "Ta": ElementLifetime("Ta", 73, "Tantalum", 0.0755, 0.0006, _S),
    "W": ElementLifetime("W", 74, "Tungsten", 0.0765, 0.0008, _S),
    "Re": ElementLifetime("Re", 75, "Rhenium", 0.076, None, _W),
    "Os": ElementLifetime("Os", 76, "Osmium", 0.078, None, _W),
    "Ir": ElementLifetime("Ir", 77, "Iridium", 0.074, None, _W),
    "Pt": ElementLifetime("Pt", 78, "Platinum", 0.074, None, _W),
    "Au": ElementLifetime("Au", 79, "Gold", 0.0728, 0.0005, _S),
    "Hg": ElementLifetime("Hg", 80, "Mercury", 0.076, 0.001, _S),
    "Tl": ElementLifetime("Tl", 81, "Thallium", 0.0704, 0.0008, _S),
    "Pb": ElementLifetime("Pb", 82, "Lead", 0.0747, 0.0004, _S),
    "Bi": ElementLifetime("Bi", 83, "Bismuth", 0.0735, 0.0004, _S),
    # --- ⚠confirm rows (actinides) ---
    "Th": ElementLifetime("Th", 90, "Thorium", 0.0780, 0.0003, _S),
    "U": ElementLifetime("U", 92, "Uranium", 0.0775, 0.0002, _S),
    "Np": ElementLifetime("Np", 93, "Neptunium", 0.0720, 0.0007, _S),
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
    """Return the capture lifetime (μs) for ``symbol``."""
    return lifetime(symbol).tau_us


def has_element(symbol: str) -> bool:
    """Return True if ``symbol`` is in the table."""
    return symbol in ELEMENT_LIFETIMES


def elements() -> list[str]:
    """Element symbols present, ordered by atomic number Z."""
    return sorted(ELEMENT_LIFETIMES, key=lambda s: ELEMENT_LIFETIMES[s].z)
