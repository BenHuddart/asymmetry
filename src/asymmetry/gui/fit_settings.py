"""User-configurable fit-display and project settings, backed by ``QSettings``.

These are presentation/session preferences (not project data): the fit-quality
settings tune how fit diagnostics are *shown*, never how a fit is computed, and
the autosave interval tunes session crash-recovery cadence, never a saved
project's contents — so both live with the GUI rather than in a saved
``.asymp``. Kept Qt-side so the core stays headless.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QSettings

from asymmetry.core.fitting.result_summary import FIT_QUALITY_CONFIDENCE

#: QSettings key for the two-sided χ² quality-band confidence level R.
FIT_QUALITY_CONFIDENCE_SETTINGS_KEY = "fit/quality_confidence"

#: Bounds match ``assess_fit_quality``'s own clamp, so the setting can never push
#: the band outside the range the core will honour.
_CONFIDENCE_MIN = 0.5
_CONFIDENCE_MAX = 0.999

#: QSettings key for the autosave interval (minutes); ``0`` disables autosave.
AUTOSAVE_INTERVAL_SETTINGS_KEY = "project/autosave_interval_minutes"

#: Default autosave cadence (D9): frequent enough to bound crash loss, rare
#: enough not to contend with a background project save.
_AUTOSAVE_INTERVAL_DEFAULT = 5


def fit_quality_confidence(settings: QSettings | None = None) -> float:
    """Return the configured χ² quality-band confidence R, clamped to [0.5, 0.999].

    Falls back to the muon-tuned default :data:`FIT_QUALITY_CONFIDENCE` (0.999)
    when the setting is unset or unparseable — high-statistics muon fits live at
    large ν where WiMDA's 0.95 band is too tight (a routine χ²ᵣ ≈ 1.2 read "poor").
    The band widens toward χ²ᵣ = 1 as R → 1, so a higher confidence means a more
    forgiving good/poor/overdone verdict.
    """
    settings = settings or QSettings()
    raw = settings.value(FIT_QUALITY_CONFIDENCE_SETTINGS_KEY, FIT_QUALITY_CONFIDENCE)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = FIT_QUALITY_CONFIDENCE
    return _clamp_confidence(value)


def set_fit_quality_confidence(value: float, settings: QSettings | None = None) -> float:
    """Persist *value* (clamped to [0.5, 0.999]) and return the stored figure."""
    settings = settings or QSettings()
    clamped = _clamp_confidence(value)
    settings.setValue(FIT_QUALITY_CONFIDENCE_SETTINGS_KEY, clamped)
    return clamped


def autosave_interval_minutes(settings: QSettings | None = None) -> int:
    """Return the configured autosave interval in minutes; ``0`` disables it.

    Falls back to the default (5 minutes) when the setting is unset or
    unparseable, and clamps a negative stored value to 0 (disabled) rather
    than arming a timer with a negative interval.
    """
    settings = settings or QSettings()
    raw = settings.value(AUTOSAVE_INTERVAL_SETTINGS_KEY, _AUTOSAVE_INTERVAL_DEFAULT)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _AUTOSAVE_INTERVAL_DEFAULT
    return max(value, 0)


def set_autosave_interval_minutes(value: int, settings: QSettings | None = None) -> int:
    """Persist *value* (clamped to >= 0) and return the stored figure."""
    settings = settings or QSettings()
    clamped = max(int(value), 0)
    settings.setValue(AUTOSAVE_INTERVAL_SETTINGS_KEY, clamped)
    return clamped


def _clamp_confidence(value: float) -> float:
    """Clamp to [0.5, 0.999]; non-finite input (NaN/inf, e.g. a corrupt stored
    string like ``"nan"``) falls back to the default rather than propagating —
    ``min``/``max`` pass NaN straight through, which would silently void the χ²
    verdict downstream."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return FIT_QUALITY_CONFIDENCE
    if not math.isfinite(value):
        return FIT_QUALITY_CONFIDENCE
    return min(max(value, _CONFIDENCE_MIN), _CONFIDENCE_MAX)
