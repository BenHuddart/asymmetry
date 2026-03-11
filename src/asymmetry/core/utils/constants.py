"""Physical constants relevant to μSR."""

# Muon gyromagnetic ratio  γ_μ / (2π)  in MHz/T
MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = 135.538817

# Muon lifetime in microseconds
MUON_LIFETIME_US = 2.1969811

# Conversion: 1 Gauss = 1e-4 Tesla
GAUSS_TO_TESLA = 1.0e-4

# Electron gyromagnetic ratio in rad / (microsecond * Gauss)
# Derived from 1.76085963023e11 rad / (s * T) using:
#   T -> G: multiply by 1e-4
#   s -> us: multiply by 1e-6
ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G = 17.6085963023
