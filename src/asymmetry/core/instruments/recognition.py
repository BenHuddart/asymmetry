"""Positive file-name and acquired-label conventions at the recognition boundary."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from asymmetry.core.instruments.catalog import profile_by_key
from asymmetry.core.instruments.schema import matches_name

EMU_PROFILE = profile_by_key("isis.emu")
HIFI_PROFILE = profile_by_key("isis.hifi")
MUSR_PROFILE = profile_by_key("isis.musr")
FLAME_PROFILE = profile_by_key("psi.flame")
GPS_PROFILE = profile_by_key("psi.gps")
HAL_PROFILE = profile_by_key("psi.hal9500")


def canonical_layout_name(raw: object, *, filename: bool = False) -> str | None:
    """Normalize free-form instrument text to a canonical instrument name."""
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None

    for profile in (FLAME_PROFILE, GPS_PROFILE, HIFI_PROFILE, MUSR_PROFILE, EMU_PROFILE):
        if matches_name(profile.recognition, raw, filename=filename):
            return profile.identity.layout_key
    return None


def _metadata_labels(metadata: dict) -> list[str]:
    """Return detector-label hints from metadata-like dictionaries."""
    raw = metadata.get("histogram_labels") or metadata.get("detector_labels")
    labels: list[str] = []
    if isinstance(raw, (list, tuple)):
        labels.extend(str(label) for label in raw)
    group_names = metadata.get("group_names")
    if isinstance(group_names, dict):
        labels.extend(str(label) for label in group_names.values())
    return labels


def _labels_match_flame(labels: list[str], n_histograms: int) -> bool:
    """Match the eight-channel split-corner FLAME label signature."""
    if n_histograms != 8:
        return False
    compact = {re.sub(r"[^a-z0-9]+", "", label.lower()) for label in labels}
    if not compact:
        return False
    # The split-plate corners are the FLAME signature; without all four this is
    # not FLAME regardless of what the main-axis labels say.
    has_corners = {"rf", "rb", "lf", "lb"}.issubset(compact)
    if not has_corners:
        return False
    has_main = (
        bool({"forw", "forward"} & compact)
        and bool({"back", "backward"} & compact)
        and "left" in compact
        and bool({"right", "righ"} & compact)
    )
    return has_main


def _is_psi_hal(metadata: dict, source_file: str | None) -> bool:
    """Match established HAL header tokens within the caller's PSI namespace."""
    return any(
        matches_name(HAL_PROFILE.recognition, metadata.get(key))
        for key in ("instrument", "instrument_name", "spectrometer", "name")
    ) or bool(
        source_file and matches_name(HAL_PROFILE.recognition, Path(source_file).stem, filename=True)
    )


def _labels_match_gps(labels: list[str], n_histograms: int) -> bool:
    """Match the six-channel combined GPS label signature."""
    if n_histograms != 6:
        return False
    compact = {re.sub(r"[^a-z0-9]+", "", label.lower()) for label in labels}
    if not compact:
        return False
    return (
        bool({"forw", "forward"} & compact)
        and bool({"back", "backward"} & compact)
        and "up" in compact
        and "down" in compact
        and bool({"righ", "right"} & compact)
        and "left" in compact
    )


def _labels_match_gps_classic(labels: list[str], n_histograms: int) -> bool:
    """Match combined GPS labels with both transverse axes and partial acquisition."""
    if not 4 <= n_histograms <= 6:
        return False
    compact = {re.sub(r"[^a-z0-9]+", "", label.lower()) for label in labels}
    if not compact:
        return False
    return (
        bool({"forw", "forward"} & compact)
        and bool({"back", "backward"} & compact)
        and "up" in compact
        and "down" in compact
        and bool({"righ", "right", "left"} & compact)
    )


#: The eight ``_B``/``_F`` half-counter labels of the GPS ROOT exports, compacted.
_GPS_HALF_COUNTER_TOKENS: Final[frozenset[str]] = frozenset(
    {"upb", "upf", "downb", "downf", "rightb", "rightf", "leftb", "leftf"}
)


def _gps_label_tokens(labels: list[str]) -> set[str]:
    """Normalise acquired GPS labels for the declared signatures."""
    return {re.sub(r"[^a-z0-9]+", "", label.lower()) for label in labels}


def _has_gps_forward_backward(compact: set[str]) -> bool:
    """Require both named beam-axis counters."""
    return bool({"forw", "forward"} & compact) and bool({"back", "backward"} & compact)


def _has_gps_combined_transverse(compact: set[str]) -> bool:
    """Require all four combined transverse counters."""
    return (
        "up" in compact
        and "down" in compact
        and bool({"righ", "right"} & compact)
        and "left" in compact
    )


def _labels_match_gps_subdetectors(labels: list[str]) -> bool:
    """Match split-only GPS labels without the combined transverse readout."""
    compact = _gps_label_tokens(labels)
    if not compact:
        return False
    if _has_gps_combined_transverse(compact):
        return False
    has_mobile = any("mob" in token for token in compact)
    return (
        _has_gps_forward_backward(compact)
        and _GPS_HALF_COUNTER_TOKENS.issubset(compact)
        and has_mobile
    )


def _labels_match_gps_combined_subdetectors(labels: list[str]) -> bool:
    """Match GPS labels containing both combined and split transverse readouts."""
    compact = _gps_label_tokens(labels)
    if not compact:
        return False
    has_mobile = any("mob" in token for token in compact)
    return (
        _has_gps_forward_backward(compact)
        and _has_gps_combined_transverse(compact)
        and _GPS_HALF_COUNTER_TOKENS.issubset(compact)
        and has_mobile
    )


def _is_psi_gps(metadata: dict, source_file: str | None, n_histograms: int) -> bool:
    """Match GPS name or acquired-label evidence within the caller's PSI namespace."""
    if any(
        matches_name(GPS_PROFILE.recognition, metadata.get(key))
        for key in ("instrument", "instrument_name", "beamline", "spectrometer", "name")
    ) or (
        source_file and matches_name(GPS_PROFILE.recognition, Path(source_file).stem, filename=True)
    ):
        return True
    labels = _metadata_labels(metadata)
    return (
        _labels_match_gps(labels, n_histograms)
        or _labels_match_gps_classic(labels, n_histograms)
        or _labels_match_gps_subdetectors(labels)
        or _labels_match_gps_combined_subdetectors(labels)
    )


def _gps_variant(metadata: dict, n_histograms: int) -> str:
    """Select the established GPS export from its count and acquired labels."""
    if n_histograms == 15:
        return "GPS-RD15"
    if n_histograms == 11:
        return "GPS-RD"
    labels = _metadata_labels(metadata)
    if _labels_match_gps_combined_subdetectors(labels):
        return "GPS-RD15"
    if _labels_match_gps_subdetectors(labels):
        return "GPS-RD"
    return "GPS"


def recognize_layout(
    n_histograms: int,
    *,
    metadata: dict | None = None,
    source_file: str | None = None,
) -> str | None:
    """Classify positive header/label/filename hints; count alone stays unidentified."""
    psi_data = False
    if isinstance(metadata, dict):
        facility = str(metadata.get("facility", "")).strip().lower()
        psi_data = facility == "psi" or bool(metadata.get("psi_format"))
        # Facility qualifies the shared HIFI alias.
        if psi_data and _is_psi_hal(metadata, source_file):
            return "HAL"
        if psi_data and _is_psi_gps(metadata, source_file, n_histograms):
            return _gps_variant(metadata, n_histograms)
        for key in (
            "instrument",
            "instrument_name",
            "beamline",
            "spectrometer",
            "facility_instrument",
            "name",
        ):
            resolved = canonical_layout_name(metadata.get(key))
            if resolved is not None:
                if psi_data and resolved not in {"FLAME", "GPS"}:
                    return None
                return resolved
        if _labels_match_flame(_metadata_labels(metadata), n_histograms):
            return "FLAME"

    if source_file:
        path_token = canonical_layout_name(Path(source_file).stem, filename=True)
        if path_token is not None:
            if psi_data and path_token not in {"FLAME", "GPS"}:
                return None
            return path_token

    return None
