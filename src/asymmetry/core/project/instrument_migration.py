"""Saved layout choices retain their historical namespace and scientific payloads."""

from dataclasses import dataclass

from asymmetry.core.instruments.resolution import (
    GenericInstrumentIdentity,
    InstrumentSelection,
    RecordedInstrumentChoice,
    UnmappedAcquisition,
    UnresolvedInstrumentIdentity,
)

# Stored layout choices use the registry namespace, independently of raw file aliases.
_LAYOUT_CHOICES = {
    "emu": ("isis.emu", "nexus96", 96),
    "hifi": ("isis.hifi", "nexus64", 64),
    "musr": ("isis.musr", "nexus64", 64),
    "flame": ("psi.flame", "eight_channel", 8),
    "hal": ("psi.hal9500", "full17", 17),
    "gps": ("psi.gps", "bin_combined", 6),
    "gps-rd": ("psi.gps", "root_split", 11),
    "gps-rd15": ("psi.gps", "root_combined_split", 15),
    "ltf": ("psi.ltf", None, None),
    "gpd": ("psi.gpd", None, None),
    "dolly": ("psi.dolly", None, None),
    "alc": ("psi.alc", None, None),
    "lem": ("psi.lem", None, None),
}
_GPS_CONFIGURATIONS = {6: "bin_combined", 11: "root_split", 15: "root_combined_split"}


@dataclass(frozen=True)
class LegacyProfileFingerprint:
    """Data-only representation of a released two-field grouping fingerprint."""

    instrument: str
    histogram_count: int

    def to_dict(self):
        return {"instrument": self.instrument, "histogram_count": self.histogram_count}


def legacy_histogram_count(overrides: dict) -> int | None:
    """Only saved acquired-channel tables establish the count, never group membership."""
    for key in ("detector_t0_bins", "detector_first_good_bins", "histogram_labels"):
        value = overrides.get(key)
        if isinstance(value, (list, tuple)) and value:
            return len(value)
    return None


def _stored_layout_selection(label: str, count: int | None):
    normalized = label.strip().casefold()
    if normalized in {"generic", "generic_layout"}:
        return GenericInstrumentIdentity()
    if normalized not in _LAYOUT_CHOICES:
        return UnresolvedInstrumentIdentity(label)
    key, acquisition, expected = _LAYOUT_CHOICES[normalized]
    if acquisition is None:
        acquisition = UnmappedAcquisition("undeclared")
    elif count:
        if normalized == "gps":
            acquisition = _GPS_CONFIGURATIONS.get(count, UnmappedAcquisition("incompatible"))
        elif count != expected:
            acquisition = UnmappedAcquisition("incompatible")
    return InstrumentSelection(key, acquisition, "1", "legacy")


def _fingerprint_count(fingerprint: dict) -> int | None:
    count = fingerprint.get("histogram_count", 0)
    return count if type(count) is int and count >= 0 else None


def _migrated_fingerprint(fingerprint: dict) -> dict:
    result = dict(fingerprint)
    label = fingerprint.get("instrument", "")
    count = _fingerprint_count(fingerprint)
    if not isinstance(label, str) or count is None:
        return result
    selected = _stored_layout_selection(label, count) if label.strip() else None
    result.update(instrument="", histogram_count=count, acquisition=None)
    if isinstance(selected, InstrumentSelection):
        result["instrument"] = selected.profile_key
        result["acquisition"] = (
            selected.acquisition if isinstance(selected.acquisition, str) else None
        )
        result.pop("unresolved_identity", None)
    elif isinstance(selected, UnresolvedInstrumentIdentity):
        result["unresolved_identity"] = selected.label
    elif isinstance(selected, GenericInstrumentIdentity):
        result.pop("unresolved_identity", None)
    return result


def _profile_saved_selection(profile: dict):
    fingerprint = profile.get("fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    label = fingerprint.get("instrument", "")
    count = _fingerprint_count(fingerprint)
    if not isinstance(label, str) or not label.strip() or count is None:
        return None
    return _stored_layout_selection(label, count)


def _dataset_saved_choice(entry: dict, profiles: list):
    overrides = entry.get("grouping_overrides")
    count = legacy_histogram_count(overrides) if isinstance(overrides, dict) else None
    label = overrides.get("instrument") if isinstance(overrides, dict) else None
    if isinstance(label, str) and label.strip():
        return RecordedInstrumentChoice(
            _stored_layout_selection(label, count),
            (("grouping_overrides.instrument", label),),
        )
    name = entry.get("profile")
    if not isinstance(name, str) or not name:
        return None
    candidates = [
        profile for profile in profiles if isinstance(profile, dict) and profile.get("name") == name
    ]
    if not isinstance(overrides, dict):
        if len(candidates) == 1:
            selected = _profile_saved_selection(candidates[0])
            if selected is not None:
                return RecordedInstrumentChoice(selected, (("profile", name),))
    return RecordedInstrumentChoice(
        UnresolvedInstrumentIdentity(f"profile:{name}"),
        (
            ("profile", name),
            ("identity", "released" if isinstance(overrides, dict) else "ambiguous"),
        ),
    )


def migrate_instrument_identities(data: dict) -> dict:
    """Convert active saved choices only; raw metadata and analysis history stay untouched."""
    result = dict(data)
    profiles = data.get("grouping_profiles")
    if isinstance(profiles, list):
        result["grouping_profiles"] = [
            dict(profile, fingerprint=_migrated_fingerprint(profile["fingerprint"]))
            if isinstance(profile, dict) and isinstance(profile.get("fingerprint"), dict)
            else profile
            for profile in profiles
        ]
    else:
        profiles = []
    for field in ("datasets", "combined_datasets"):
        if not isinstance(data.get(field), list):
            continue
        entries = []
        for entry in data[field]:
            if not isinstance(entry, dict):
                entries.append(entry)
                continue
            updated = dict(entry)
            if "instrument_choice" not in entry:
                choice = _dataset_saved_choice(entry, profiles)
                if choice is not None:
                    updated["instrument_choice"] = choice.to_dict()
            overrides = entry.get("grouping_overrides")
            if isinstance(overrides, dict) and "instrument" in overrides:
                updated["grouping_overrides"] = {
                    key: value for key, value in overrides.items() if key != "instrument"
                }
            entries.append(updated)
        result[field] = entries
    return result
