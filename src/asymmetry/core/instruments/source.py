"""Established source hints, kept distinct from an absent or contradictory hint."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import Run

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
    """Prepare raw input only when it carries no selected source record."""
    return (
        run.source_assessment if run.source_assessment is not None else assess_source(run.metadata)
    )
