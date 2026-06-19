"""Shared scoring of NXlog time-series paths against browser summary fields.

Both the Data Browser ("Use temperature / field from log") and the Run Info
dialog need to pick *the same* sensor for a given run from its
``nexus_time_series`` map. They used to carry near-duplicate scorers that drifted
apart; this module is the single implementation both call so they stay
consistent.

Two correctness rules the temperature scorer encodes:

* a **detector-electronics** thermometer (``DetectorTemp1`` sits near 298 K) is
  never the sample temperature — it is excluded so the alphabetical tie-break
  cannot select it over ``Temp_Cryostat`` (the HiFi "room temperature for every
  run" bug);
* **cryostat / VTI / He / dilution** sensors track the sample closely and are
  preferred over a bare, unspecified ``*Temp*`` log.

Matching uses the series path *and* its NXlog ``name`` (the sensor label),
because native HDF4 v1 logs name the Vgroup generically and carry the real
sensor name only in a ``name`` child (the converted HDF5 twin bakes it into the
selog path). See :meth:`asymmetry.core.io.nexus.NexusLoader._extract_time_series`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Sample-environment thermometer name fragments (matched against the
#: space-stripped label). A temperature log naming one of these tracks the
#: sample and outranks a bare/unspecified ``*Temp*`` log.
_CRYOSTAT_TEMP_TOKENS = (
    "cryostat",
    "cryo",
    "vti",
    "variox",
    "dilution",
    "dilt",
    "still",
    "mixing",
    "he3",
    "he4",
)


def _series_text(series_path: str, info: Any) -> str:
    """Normalised lower-case search text for a series: its path plus NXlog name."""
    name = ""
    if isinstance(info, Mapping):
        name = str(info.get("name", "") or "")
    raw = f"{series_path} {name}"
    return " ".join(raw.replace("_", " ").replace("/", " ").lower().split())


def _is_detector_temperature(words: set[str]) -> bool:
    """True when a single label token names detector electronics, e.g.
    ``DetectorTemp1`` — never the sample temperature. Checked per token so an
    ancestor path like ``instrument/detector/Temp_Sample`` is not excluded."""
    return any("detector" in word and "temp" in word for word in words)


def score_series_path(field_key: str, series_path: str, info: Any = None) -> int:
    """Score how well a log series matches a browser summary field (0 = no match).

    ``info`` is the series entry from ``nexus_time_series`` (a mapping that may
    carry ``role`` / ``primary`` / ``name``); it may be ``None``.
    """
    if not isinstance(info, Mapping):
        info = {}
    role = str(info.get("role", "")).strip().lower()
    primary = bool(info.get("primary", False))
    if field_key == "temperature" and role == "sample_temperature":
        return 100 if primary else 70
    if field_key == "field" and role == "sample_field":
        return 80 if primary else 60

    normalized = _series_text(series_path, info)
    compact = normalized.replace(" ", "")
    words = set(normalized.split())

    if field_key == "temperature":
        if not (
            "temp" in compact
            or "samtsvalue" in compact
            or "dilt" in compact
            or "variox" in compact
            or "(k)" in normalized
        ):
            return 0
        # Detector electronics temperature is never the sample temperature.
        if _is_detector_temperature(words):
            return 0
        score = 10
        if "sample" in normalized:
            score += 20
        if "sam ts value" in normalized:
            score += 30
        if "sample temperature" in normalized or "sampletemp" in compact:
            score += 20
        if "moderator" in normalized:
            score -= 5
        # Prefer cryostat / VTI / He / dilution sensors over a bare *Temp* log.
        if "he" in words or any(token in compact for token in _CRYOSTAT_TEMP_TOKENS):
            score += 15
        return score
    if field_key == "field":
        if "field" not in normalized and "magnet" not in normalized:
            return 0
        score = 10
        if "sample" in normalized:
            score += 10
        return score
    if field_key == "field_direction":
        return 10 if "direction" in normalized else 0
    return 0
