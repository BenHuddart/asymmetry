"""Established source hints, kept distinct from an absent or contradictory hint."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.instruments.resolution import RecordedInstrumentChoice

_PULSED_TOKENS = ("isis", "ral", "rutherford", "j-parc", "jparc", "kek", "riken")
_CONTINUOUS_TOKENS = ("psi", "triumf", "lem")


@dataclass(frozen=True)
class SourceAssessment:
    """A selected source value and its authority, including an explicit unknown."""

    pulsed: bool | None
    reason: Literal["known", "absent", "unsupported", "conflict"]
    origin: str
    profile_key: str | None = None
    profile_revision: str | None = None

    def __post_init__(self):
        if self.pulsed is not None and not isinstance(self.pulsed, bool):
            raise ValueError("Source value must be pulsed, continuous or unknown")
        if not isinstance(self.reason, str) or self.reason not in {
            "known",
            "absent",
            "unsupported",
            "conflict",
        }:
            raise ValueError("Source assessment requires an established reason")
        if (self.reason == "known") != (self.pulsed is not None):
            raise ValueError("Only a known source assessment has a Boolean value")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ValueError("Source assessment requires its resolution origin")
        has_profile = self.profile_key is not None or self.profile_revision is not None
        if (self.origin == "profile") != has_profile or (
            has_profile
            and not all(
                isinstance(value, str) and value.strip()
                for value in (self.profile_key, self.profile_revision)
            )
        ):
            raise ValueError("Profile source origin requires its key and revision together")

    def to_dict(self) -> dict:
        return {
            "pulsed": self.pulsed,
            "reason": self.reason,
            "origin": self.origin,
            **(
                {"profile_key": self.profile_key, "profile_revision": self.profile_revision}
                if self.profile_key is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SourceAssessment:
        required = {"pulsed", "reason", "origin"}
        if (
            not isinstance(data, dict)
            or not required <= data.keys()
            or data.keys() - required - {"profile_key", "profile_revision"}
        ):
            raise ValueError("Captured source requires value, reason and origin")
        return cls(**data)


def assess_source(metadata: dict | None) -> SourceAssessment:
    """Preserve established Boolean/token precedence without profile defaults."""
    metadata = metadata if isinstance(metadata, dict) else {}
    if isinstance(metadata.get("source_pulsed"), bool):
        return SourceAssessment(metadata["source_pulsed"], "known", "explicit")
    text = " ".join(
        str(metadata.get(key, "")) for key in ("facility", "instrument", "source", "area")
    ).lower()
    matches = [
        pulsed
        for pulsed, tokens in ((False, _CONTINUOUS_TOKENS), (True, _PULSED_TOKENS))
        if re.search(r"(?<![a-z0-9])(?:" + "|".join(tokens) + r")(?![a-z0-9])", text)
    ]
    if len(matches) == 1:
        return SourceAssessment(matches[0], "known", "metadata")
    if matches:
        return SourceAssessment(None, "conflict", "metadata")
    unsupported = any(
        metadata.get(key) is not None and bool(str(metadata[key]).strip())
        for key in ("source_pulsed", "source", "facility")
    )
    return SourceAssessment(None, "unsupported" if unsupported else "absent", "metadata")


def source_for_run(run: Run) -> SourceAssessment:
    """Retain captured authority; only fresh absence can use a documented profile source."""
    if run.source_assessment is not None:
        return run.source_assessment
    assessment = assess_source(run.metadata)
    if assessment.reason != "absent":
        return assessment
    from asymmetry.core.instruments.catalog import PROFILES
    from asymmetry.core.instruments.resolution import resolve_instrument
    from asymmetry.core.instruments.schema import EvidenceStatus

    context = resolve_instrument(run)
    selected = context.selection
    if selected is None:
        return assessment
    profile = next((item for item in PROFILES if item.identity.key == selected.profile_key), None)
    if profile is None or profile.source.pulsed is None:
        return assessment
    evidence = next(item for item in profile.sources.evidence if item.fact == "source")
    if evidence.status is not EvidenceStatus.SOURCE:
        return assessment
    return SourceAssessment(
        profile.source.pulsed,
        "known",
        "profile",
        profile.identity.key,
        profile.identity.revision,
    )


# Established automatic-range conventions, retained without new physical validation.
# They are separate from the source-family classification and pulse-response requests.
_RF_PERIOD_US = {"psi": 0.01975, "triumf": 0.04337, "ral": 0.0, "isis": 0.0}


def beam_period_us(facility: str) -> float:
    """Retain the established RF-period label arithmetic, including unknown zero."""
    label = facility.lower()
    return next((period for name, period in _RF_PERIOD_US.items() if name in label), 0.0)


def resolve_facility(
    *,
    metadata: dict | None = None,
    grouping: dict | None = None,
    instrument_choice: RecordedInstrumentChoice | None = None,
) -> str:
    """Resolve the established RF-period hint without deriving it from canonical identity."""
    from asymmetry.core.instruments.resolution import UnresolvedInstrumentIdentity

    metadata = metadata if isinstance(metadata, dict) else {}
    grouping = grouping if isinstance(grouping, dict) else {}
    facility = str(metadata.get("facility", "") or "").strip()
    if facility:
        return facility
    if metadata.get("psi_format"):
        return "PSI"
    instrument = str(metadata.get("instrument", "") or "").strip()
    if instrument:
        return instrument
    legacy = str(grouping.get("instrument", "") or "").strip()
    if legacy:
        return legacy
    if instrument_choice is not None and isinstance(
        instrument_choice.selection, UnresolvedInstrumentIdentity
    ):
        return instrument_choice.selection.label
    return str(metadata.get("background_facility", "") or "").strip()


def capture_source_authority(run: Run) -> None:
    """Capture each acquired owner's decision once, preserving known and unknown records."""
    run.source_assessment = source_for_run(run)
    for original in run.grouping.get("period_inputs", ()):
        capture_source_authority(original)
