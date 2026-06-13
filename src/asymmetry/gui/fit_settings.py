"""User-configurable fit-display settings, backed by ``QSettings``.

These are presentation preferences (not project data): they tune how fit
diagnostics are *shown*, never how a fit is computed, so they live with the GUI
rather than in a saved ``.asymp``. Kept Qt-side so the core stays headless.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

from asymmetry.core.fitting.result_summary import FIT_QUALITY_CONFIDENCE

#: QSettings key for the two-sided χ² quality-band confidence level R.
FIT_QUALITY_CONFIDENCE_SETTINGS_KEY = "fit/quality_confidence"

#: Bounds match ``assess_fit_quality``'s own clamp, so the setting can never push
#: the band outside the range the core will honour.
_CONFIDENCE_MIN = 0.5
_CONFIDENCE_MAX = 0.999


def fit_quality_confidence(settings: QSettings | None = None) -> float:
    """Return the configured χ² quality-band confidence R, clamped to [0.5, 0.999].

    Falls back to WiMDA's ``Rgoodfit`` default (0.95) when the setting is unset or
    unparseable. The band widens toward χ²ᵣ = 1 as R → 1, so a higher confidence
    means a more forgiving good/poor/overdone verdict.
    """
    settings = settings or QSettings()
    raw = settings.value(FIT_QUALITY_CONFIDENCE_SETTINGS_KEY, FIT_QUALITY_CONFIDENCE)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = FIT_QUALITY_CONFIDENCE
    return min(max(value, _CONFIDENCE_MIN), _CONFIDENCE_MAX)


def set_fit_quality_confidence(value: float, settings: QSettings | None = None) -> float:
    """Persist *value* (clamped to [0.5, 0.999]) and return the stored figure."""
    settings = settings or QSettings()
    clamped = min(max(float(value), _CONFIDENCE_MIN), _CONFIDENCE_MAX)
    settings.setValue(FIT_QUALITY_CONFIDENCE_SETTINGS_KEY, clamped)
    return clamped
