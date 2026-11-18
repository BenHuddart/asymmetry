"""Textual limits of the existing displayed-spectrum uncertainty approximations."""

SPECTRAL_UNCERTAINTY_LIMITS = (
    "Spectral correlations, estimated phase, upstream calibration/background and "
    "MaxEnt reconstruction uncertainty are not fully propagated. Scalar padding "
    "rescaling can understate or overstate errors; it does not establish coverage."
)

MOMENT_UNCERTAINTY_SCOPE = (
    "Heuristic approximate errors from independent spectral-bin noise and frozen "
    "processing, with scalar padding rescaling. " + SPECTRAL_UNCERTAINTY_LIMITS
)

FIT_UNCERTAINTY_SCOPE = "Approximate spectral fit errors/statistics. " + SPECTRAL_UNCERTAINTY_LIMITS
