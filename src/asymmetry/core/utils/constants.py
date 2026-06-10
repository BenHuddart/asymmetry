"""Physical constants and period-mode identifiers relevant to μSR."""

from __future__ import annotations

from enum import StrEnum


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


# Muon gyromagnetic ratio  γ_μ / (2π)  in MHz/T
MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = 135.538817

# Fluorine-19 gyromagnetic ratio  γ_F / (2π)  in MHz/T
FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T = 40.053

# Proton (1H) gyromagnetic ratio  γ_p / (2π)  in MHz/T (CODATA)
PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T = 42.577478

# Muon lifetime in microseconds
MUON_LIFETIME_US = 2.1969811

# Pion lifetime in microseconds — sets the intrinsic short-time limit of the
# muon-pulse arrival distribution at a pulsed source (surface muons come from
# pion decay), so it enters the MaxEnt pulse-shape response.
PION_LIFETIME_US = 0.026

# Conversion: 1 Gauss = 1e-4 Tesla
GAUSS_TO_TESLA = 1.0e-4

# Electron gyromagnetic ratio in rad / (microsecond * Gauss)
# Derived from 1.76085963023e11 rad / (s * T) using:
#   T -> G: multiply by 1e-4
#   s -> us: multiply by 1e-6
ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G = 17.6085963023
