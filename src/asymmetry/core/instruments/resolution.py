"""Canonical instrument decisions, independent of rendering and numerical defaults."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from asymmetry.core.instruments.catalog import PROFILES, profile_by_key
from asymmetry.core.instruments.recognition import ACQUIRED_LABEL_KEYS, recognize_layout
from asymmetry.core.instruments.schema import InstrumentProfile, matches_name
from asymmetry.core.instruments.source import SourceAssessment, assess_source, source_for_run

__all__ = [
    "SourceAssessment",
    "assess_source",
    "source_for_run",
    "InstrumentSelection",
    "InstrumentResolution",
    "UnmappedAcquisition",
    "assess_instrument",
]


@dataclass(frozen=True)
class UnmappedAcquisition:
    """Known identity with no declared map, or incompatible acquired channels."""

    reason: Literal["undeclared", "incompatible"]

    def __post_init__(self):
        if self.reason not in {"undeclared", "incompatible"}:
            raise ValueError("Unmapped acquisition requires its availability reason")


@dataclass(frozen=True)
class InstrumentSelection:
    """Stable identity/configuration and the provenance of this selected version."""

    profile_key: str
    acquisition: str | UnmappedAcquisition
    profile_revision: str
    origin: str

    def __post_init__(self):
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.profile_key, self.profile_revision, self.origin)
        ):
            raise ValueError("Instrument selection requires stable identity, revision and origin")
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", self.profile_key):
            raise ValueError("Instrument selection requires a canonical facility-qualified key")
        if not isinstance(self.acquisition, UnmappedAcquisition) and not (
            isinstance(self.acquisition, str) and self.acquisition.strip()
        ):
            raise ValueError("Instrument selection requires a configuration or unavailable map")

    @property
    def identity(self):
        return self.profile_key, self.acquisition

    def to_dict(self):
        return {
            "profile_key": self.profile_key,
            "acquisition": {"unavailable": self.acquisition.reason}
            if isinstance(self.acquisition, UnmappedAcquisition)
            else self.acquisition,
            "profile_revision": self.profile_revision,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {
            "profile_key",
            "acquisition",
            "profile_revision",
            "origin",
        }:
            raise ValueError("Instrument selection requires its canonical fields")
        acquisition = data["acquisition"]
        if isinstance(acquisition, dict):
            if set(acquisition) != {"unavailable"}:
                raise ValueError("Unavailable acquisition requires its reason")
            acquisition = UnmappedAcquisition(acquisition["unavailable"])
        return cls(data["profile_key"], acquisition, data["profile_revision"], data["origin"])


@dataclass(frozen=True)
class InstrumentResolution:
    """Candidates plus original evidence; unresolved evidence cannot select a default."""

    candidates: tuple[InstrumentSelection, ...]
    evidence: tuple[tuple[str, str], ...]
    unresolved_reason: Literal["absent", "unsupported", "conflict"] | None = None

    def __post_init__(self):
        if len({candidate.identity for candidate in self.candidates}) != len(self.candidates):
            raise ValueError(
                "Instrument candidates must have distinct stable identities/configurations"
            )
        if self.unresolved_reason not in {None, "absent", "unsupported", "conflict"}:
            raise ValueError("Instrument resolution requires a known outcome")
        if self.unresolved_reason is None and len(self.candidates) != 1:
            raise ValueError("A selected instrument requires exactly one candidate")
        if self.unresolved_reason in {"absent", "unsupported"} and self.candidates:
            raise ValueError("An unidentified instrument cannot have selected candidates")
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in self.evidence
        ):
            raise ValueError("Instrument evidence must preserve named raw text")

    @property
    def selection(self) -> InstrumentSelection | None:
        return self.candidates[0] if self.unresolved_reason is None else None

    def to_dict(self):
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "evidence": [list(item) for item in self.evidence],
            "unresolved_reason": self.unresolved_reason,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {
            "candidates",
            "evidence",
            "unresolved_reason",
        }:
            raise ValueError("Instrument resolution requires candidates, evidence and outcome")
        if (
            not isinstance(data["candidates"], list)
            or not isinstance(data["evidence"], list)
            or any(not isinstance(item, list) or len(item) != 2 for item in data["evidence"])
        ):
            raise ValueError("Instrument resolution requires ordered candidates and evidence")
        return cls(
            tuple(InstrumentSelection.from_dict(item) for item in data["candidates"]),
            tuple(tuple(item) for item in data["evidence"]),
            data["unresolved_reason"],
        )


def selection_for_profile(
    profile: InstrumentProfile, count: int, origin: str, *, labels: tuple[str, ...] = ()
) -> InstrumentSelection:
    matches = [
        config.key
        for config in profile.detectors.acquisitions
        if len(config.channels) == count
        and (
            not config.require_file_labels
            or (
                len(labels) == count
                and all(
                    labels[channel.ordinal - 1] in channel.file_labels
                    for channel in config.channels
                )
            )
        )
    ]
    acquisition = (
        matches[0]
        if len(matches) == 1
        else UnmappedAcquisition("incompatible" if profile.detectors.acquisitions else "undeclared")
    )
    return InstrumentSelection(profile.identity.key, acquisition, profile.identity.revision, origin)


def assess_instrument(
    n_histograms: int, *, metadata: dict | None = None, source_file: str | None = None
) -> InstrumentResolution:
    """Collect independent identity evidence without choosing a conflicting hint."""
    metadata = metadata if isinstance(metadata, dict) else {}
    count = int(n_histograms)
    keys = (
        "instrument",
        "instrument_name",
        "beamline",
        "spectrometer",
        "facility_instrument",
        "name",
    )
    evidence = tuple(
        (key, str(metadata[key]))
        for key in (
            "facility",
            "psi_format",
            *keys,
            "histogram_labels",
            "detector_labels",
            "group_names",
        )
        if metadata.get(key) is not None and str(metadata[key]).strip()
    )
    if source_file:
        evidence += (("filename", Path(source_file).name),)
    evidence += (("histogram_count", str(count)),)
    facility = str(metadata.get("facility", "")).strip().casefold()
    facilities = {
        profile.identity.facility
        for profile in PROFILES
        if re.search(
            r"(?<![a-z0-9])" + re.escape(profile.identity.facility.casefold()) + r"(?![a-z0-9])",
            facility,
        )
    }
    if metadata.get("psi_format"):
        facilities.add("PSI")
    acquired_labels = next((metadata[key] for key in ACQUIRED_LABEL_KEYS if metadata.get(key)), ())
    labels = tuple(acquired_labels) if isinstance(acquired_labels, (list, tuple)) else ()
    found = []
    hints = [
        (key, metadata[key], False)
        for key in keys
        if metadata.get(key) is not None and str(metadata[key]).strip()
    ]
    if source_file:
        hints.append(("filename", Path(source_file).stem, True))
    for key, raw, filename in hints:
        compact = re.sub(r"[^a-z0-9]+", "", str(raw).casefold())
        matches = [
            profile
            for profile in PROFILES
            if matches_name(profile.recognition, raw, filename=filename)
            or compact == re.sub(r"[^a-z0-9]+", "", profile.identity.layout_key.casefold())
        ]
        scoped = [
            profile
            for profile in matches
            if not facilities or profile.identity.facility in facilities
        ]
        matches = scoped or matches
        exact_identity = [
            profile
            for profile in matches
            if compact
            in {
                re.sub(r"[^a-z0-9]+", "", value.casefold())
                for value in (profile.identity.name, profile.identity.layout_key)
            }
        ]
        exact_alias = [
            profile
            for profile in matches
            if compact
            in {
                re.sub(r"[^a-z0-9]+", "", value.casefold()) for value in profile.recognition.aliases
            }
        ]
        if exact_identity or exact_alias:
            matches = exact_identity or exact_alias
        else:
            # A borrowed alias needs its facility namespace when another instrument owns that name.
            matches = [
                profile
                for profile in matches
                if any(
                    re.sub(r"[^a-z0-9]+", "", alias.casefold()) in compact
                    and not any(
                        other.identity.key != profile.identity.key
                        and re.sub(r"[^a-z0-9]+", "", other.identity.name.casefold())
                        == re.sub(r"[^a-z0-9]+", "", alias.casefold())
                        for other in matches
                    )
                    for alias in (
                        profile.recognition.filename_tokens
                        if filename
                        else profile.recognition.aliases
                    )
                )
            ]
        for profile in matches:
            found.append(
                selection_for_profile(
                    profile, count, "filename" if filename else "metadata", labels=labels
                )
            )
    label_metadata = {
        key: metadata[key]
        for key in ("facility", "psi_format", "histogram_labels", "detector_labels", "group_names")
        if key in metadata
    }
    label_key = recognize_layout(count, metadata=label_metadata)
    if label_key:
        for profile in PROFILES:
            if any(
                (config.layout_key or profile.identity.layout_key) == label_key
                for config in profile.detectors.acquisitions
            ):
                found.append(selection_for_profile(profile, count, "labels", labels=labels))
    candidates = tuple({candidate.identity: candidate for candidate in found}.values())
    contrary = bool(
        facilities
        and any(
            profile_by_key(candidate.profile_key).identity.facility not in facilities
            for candidate in candidates
        )
    )
    if len(candidates) > 1 or len(facilities) > 1 or contrary:
        return InstrumentResolution(candidates, evidence, "conflict")
    if candidates:
        return InstrumentResolution(candidates, evidence)
    return InstrumentResolution((), evidence, "unsupported" if hints else "absent")


@dataclass(frozen=True)
class UnresolvedInstrumentIdentity:
    """Retained opaque identity whose mapping is not established."""

    label: str

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Unresolved recorded identity requires its original nonempty label")


@dataclass(frozen=True)
class GenericInstrumentIdentity:
    """Explicit selection of the preset-free acquired-channel display."""


@dataclass(frozen=True)
class RecordedInstrumentChoice:
    """A selected identity, including a retained unknown, and its evidence context."""

    selection: InstrumentSelection | UnresolvedInstrumentIdentity | GenericInstrumentIdentity
    evidence: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(
            self.selection,
            (InstrumentSelection, UnresolvedInstrumentIdentity, GenericInstrumentIdentity),
        ):
            raise ValueError(
                "Recorded instrument choice requires a known, unresolved or generic identity"
            )
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) for value in item)
            for item in self.evidence
        ):
            raise ValueError("Recorded instrument choice requires named original evidence")

    def to_dict(self):
        selected = self.selection
        value = (
            selected.to_dict()
            if isinstance(selected, InstrumentSelection)
            else {"unresolved": selected.label}
            if isinstance(selected, UnresolvedInstrumentIdentity)
            else {"generic": True}
        )
        return {"selection": value, "evidence": [list(item) for item in self.evidence]}

    @classmethod
    def from_dict(cls, data):
        if (
            not isinstance(data, dict)
            or set(data) != {"selection", "evidence"}
            or not isinstance(data["selection"], dict)
            or not isinstance(data["evidence"], list)
        ):
            raise ValueError("Recorded instrument choice requires selection and evidence")
        value = data["selection"]
        if set(value) == {"unresolved"}:
            selected = UnresolvedInstrumentIdentity(value["unresolved"])
        elif set(value) == {"generic"} and value["generic"] is True:
            selected = GenericInstrumentIdentity()
        else:
            selected = InstrumentSelection.from_dict(value)
        if any(not isinstance(item, list) or len(item) != 2 for item in data["evidence"]):
            raise ValueError("Recorded instrument evidence requires ordered field/value pairs")
        return cls(selected, tuple(tuple(item) for item in data["evidence"]))


def normalized_evidence(evidence):
    """Case and whitespace changes do not create a different identity conflict."""
    return tuple(sorted((key, value.strip().casefold()) for key, value in evidence))


@dataclass(frozen=True)
class InstrumentContext:
    """Current acquired evidence and the independent recorded identity decision."""

    observed: InstrumentResolution
    recorded: RecordedInstrumentChoice | None

    @property
    def requires_choice(self) -> bool:
        if self.recorded is None:
            return self.observed.unresolved_reason == "conflict"
        if normalized_evidence(self.recorded.evidence) == normalized_evidence(
            self.observed.evidence
        ):
            return False
        if self.observed.unresolved_reason in {"absent", "unsupported"}:
            return False
        selected = self.recorded.selection
        current = self.observed.selection
        if not isinstance(selected, InstrumentSelection) or current is None:
            return True
        if selected.identity == current.identity:
            return False
        return not self.has_new_acquisition_map

    @property
    def has_new_acquisition_map(self) -> bool:
        """The current profile explicitly maps a readout unavailable at capture."""
        selected = self.recorded.selection if self.recorded is not None else None
        current = self.observed.selection
        if (
            not isinstance(selected, InstrumentSelection)
            or current is None
            or selected.profile_key != current.profile_key
            or selected.acquisition != UnmappedAcquisition("incompatible")
        ):
            return False
        recorded = dict(normalized_evidence(self.recorded.evidence))
        observed = dict(normalized_evidence(self.observed.evidence))
        recorded_labels = {recorded[key] for key in ACQUIRED_LABEL_KEYS if key in recorded}
        observed_labels = {observed[key] for key in ACQUIRED_LABEL_KEYS if key in observed}
        recorded_count = recorded.get("histogram_count")
        if (
            (recorded_count is not None and recorded_count != observed.get("histogram_count"))
            or (recorded_labels and recorded_labels != observed_labels)
            or (not recorded_labels and selected.origin != "legacy")
        ):
            return False
        return any(
            config.key == current.acquisition
            and selected.profile_revision in config.previously_unmapped_revisions
            for config in profile_by_key(current.profile_key).detectors.acquisitions
        )

    @property
    def selection(self) -> InstrumentSelection | None:
        if self.requires_choice:
            return None
        if self.recorded is not None:
            return (
                self.recorded.selection
                if isinstance(self.recorded.selection, InstrumentSelection)
                else None
            )
        return self.observed.selection


def resolve_instrument(run, recorded: RecordedInstrumentChoice | None = None) -> InstrumentContext:
    """Prepare current evidence without mutating a recorded scientific request."""
    metadata = dict(run.metadata)
    for key in ACQUIRED_LABEL_KEYS:
        if key not in metadata and key in run.grouping:
            metadata[key] = run.grouping[key]
    observed = assess_instrument(
        len(run.histograms), metadata=metadata, source_file=run.source_file
    )
    return InstrumentContext(observed, recorded if recorded is not None else run.instrument_choice)


class InstrumentConflictError(ValueError):
    """A recorded choice and contrary acquired evidence require an explicit decision."""

    def __init__(self, context: InstrumentContext):
        super().__init__(
            "Instrument identity conflicts with the acquired evidence; choose the retained "
            "identity or an observed instrument explicitly. Grouping and calibrations are retained."
        )
        self.context = context


def layout_key_for_selection(selection: InstrumentSelection, count: int) -> str | None:
    """A historical selected map is usable only with the actual acquired channel count."""
    if isinstance(selection.acquisition, UnmappedAcquisition):
        return None
    profile = next((item for item in PROFILES if item.identity.key == selection.profile_key), None)
    if profile is None:
        return None
    config = next(
        (item for item in profile.detectors.acquisitions if item.key == selection.acquisition), None
    )
    if config is None or len(config.channels) != count:
        return None
    return config.layout_key or profile.identity.layout_key


def choice_for_layout(run, layout_key: str) -> RecordedInstrumentChoice:
    """Validate a user's public layout selection without applying its grouping preset."""
    observed = resolve_instrument(run).observed
    if layout_key == "Generic":
        return RecordedInstrumentChoice(GenericInstrumentIdentity(), observed.evidence)
    for profile in PROFILES:
        for config in profile.detectors.acquisitions:
            if layout_key == (config.layout_key or profile.identity.layout_key):
                selected = InstrumentSelection(
                    profile.identity.key, config.key, profile.identity.revision, "manual"
                )
                return RecordedInstrumentChoice(selected, observed.evidence)
    raise ValueError(f"Unknown instrument layout selection: {layout_key!r}")


def accept_instrument_choice(run, selected) -> RecordedInstrumentChoice:
    """Capture the evidence accepted by an explicit identity decision."""
    return RecordedInstrumentChoice(selected, resolve_instrument(run).observed.evidence)
