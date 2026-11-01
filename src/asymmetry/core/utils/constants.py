"""Physical constants and period-mode identifiers relevant to μSR.

The exact defining values follow the revised SI, effective in 2019. Measured
magnetic constants use the CODATA 2018 adjustment, independent of the installed
SciPy version. Uncertainties below are standard uncertainties; unit conversions
are exact. Explicit approximations retain their stated precision.

Sources
-------
* E. Tiesinga et al., Rev. Mod. Phys. **93**, 025010 (2021), Table XXXI
  (`CODATA 2018 <https://doi.org/10.1103/RevModPhys.93.025010>`_).
* S. Navas et al. (Particle Data Group), Phys. Rev. D **110**, 030001 (2024):
  `muon <https://pdg.lbl.gov/2024/listings/rpp2024-list-muon.pdf>`_ and
  `pion <https://pdg.lbl.gov/2024/download/db2024.pdf>`_ lifetimes.
* W. Liu et al., Phys. Rev. Lett. **82**, 711 (1999),
  `muonium hyperfine interval <https://doi.org/10.1103/PhysRevLett.82.711>`_.
"""

from __future__ import annotations

from enum import StrEnum
from math import pi


class PeriodMode(StrEnum):
    """Supported combination modes for two-period muon data."""

    RED = "red"
    GREEN = "green"
    GREEN_MINUS_RED = "green_minus_red"
    GREEN_PLUS_RED = "green_plus_red"


#: Allowed ordering keys for run series and field scans — the per-run variable
#: an x-axis is sorted by. Shared by :class:`asymmetry.core.representation.series.FitSeries`
#: and :func:`asymmetry.core.transform.build_field_scan` so the two cannot drift.
ORDER_KEYS = ("field", "temperature", "run")


#: Planck constant h in J s; revised SI (2019), exact.
PLANCK_CONSTANT_J_S = 6.62607015e-34

#: Elementary charge e in C; revised SI (2019), exact.
ELEMENTARY_CHARGE_C = 1.602176634e-19

#: Boltzmann constant k_B in J/K; revised SI (2019), exact.
BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23

#: Avogadro constant N_A in mol⁻¹; revised SI (2019), exact.
AVOGADRO_CONSTANT_PER_MOL = 6.02214076e23

#: Reduced Planck constant ħ = h/(2π) in J s; exact SI relation.
REDUCED_PLANCK_CONSTANT_J_S = PLANCK_CONSTANT_J_S / (2 * pi)

#: Magnetic flux quantum Φ₀ = h/(2e) in Wb; exact SI relation.
MAGNETIC_FLUX_QUANTUM_WB = PLANCK_CONSTANT_J_S / (2 * ELEMENTARY_CHARGE_C)

#: Boltzmann constant k_B in meV/K; exact SI relation, 1 eV = e J/C.
BOLTZMANN_CONSTANT_MEV_PER_K = BOLTZMANN_CONSTANT_J_PER_K / ELEMENTARY_CHARGE_C * 1e3

#: Molar gas constant R = N_A k_B in J/(mol K); exact SI relation.
MOLAR_GAS_CONSTANT_J_PER_MOL_K = AVOGADRO_CONSTANT_PER_MOL * BOLTZMANN_CONSTANT_J_PER_K

#: Negative-muon magnetic moment in J/T; CODATA 2018, uncertainty 1.0e-33 J/T.
MUON_MAGNETIC_MOMENT_J_PER_T = -4.49044830e-26

#: Muon γ_μ/(2π) in MHz/T; source unverified, approximation within 6e-8
#: relative of CODATA 2018 2|μ_μ|/h. This value does not include a sign.
MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = 135.538817

#: Electron gyromagnetic-ratio magnitude in rad/(s T); CODATA 2018, uncertainty 53 rad/(s T).
ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_PER_T = 1.76085963023e11

#: Proton γ_p/(2π) in MHz/T; truncated CODATA 2018 value 42.577478518(18).
PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T = 42.577478

#: Vacuum μ₀/(4π) in H/m; CODATA 2018, uncertainty 1.5e-17 H/m, not exact.
VACUUM_PERMEABILITY_OVER_4PI_H_PER_M = 1.00000000055e-7

#: Muon lifetime in μs; PDG 2024, uncertainty 0.0000022 μs.
MUON_LIFETIME_US = 2.1969811

#: Charged-pion lifetime in μs; rounded from PDG 2024 0.026033(5) μs.
#: Sets the surface-muon pulse-arrival time scale in the MaxEnt response.
PION_LIFETIME_US = 0.026

#: Fluorine-19 γ_F/(2π) in MHz/T; source and shielding convention unverified.
FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T = 40.053

#: Vacuum muonium hyperfine interval in MHz; truncated from Liu et al. (1999)
#: 4463.302765(53) MHz. Used as a fit starting value.
MUONIUM_VACUUM_HYPERFINE_MHZ = 4463.302

#: Tesla per gauss; exact unit definition, 1 G = 10⁻⁴ T.
GAUSS_TO_TESLA = 1.0e-4

#: Electron gyromagnetic-ratio magnitude in rad/(μs G); CODATA 2018, exact conversion of the SI value.
ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G = (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_PER_T * GAUSS_TO_TESLA * 1e-6
)
