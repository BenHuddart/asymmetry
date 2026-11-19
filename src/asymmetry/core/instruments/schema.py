"""Construction-validated instrument facts, channel maps and review provenance."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from asymmetry.core.instruments.geometry import View, expand_view
from asymmetry.core.instruments.layout import AsymmetryProjection, ReferenceArrow


class EvidenceStatus(str, Enum):
    SOURCE = "Source-backed"
    INFERRED = "Diagram-inferred"
    SCHEMATIC = "Schematic"
    AUTHORED = "Asymmetry-defined"
    UNVERIFIED = "Unverified"


@dataclass(frozen=True)
class Identity:
    key: str
    facility: str
    name: str
    revision: str
    layout_key: str
    display_name: str
    beamline: str | None = None


@dataclass(frozen=True)
class Source:
    kind: str
    description: str

    def __post_init__(self):
        if self.kind not in {"pulsed", "continuous", "unknown"}:
            raise ValueError("Source kind must be pulsed, continuous or unknown")

    @property
    def pulsed(self) -> bool | None:
        """Declared source family; no automatic analysis-default application."""
        return {"pulsed": True, "continuous": False, "unknown": None}[self.kind]


@dataclass(frozen=True)
class Recognition:
    aliases: tuple[str, ...]
    filename_tokens: tuple[str, ...] = ()


def matches_name(recognition: Recognition, raw: object, *, filename: bool = False) -> bool:
    """Match declared compact name tokens; facility precedence remains with the caller."""
    compact = re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())
    tokens = recognition.filename_tokens if filename else recognition.aliases
    return any(re.sub(r"[^a-z0-9]+", "", token.lower()) in compact for token in tokens)


@dataclass(frozen=True)
class Dimension:
    """A physical annotation, separate from schematic coordinates."""

    name: str
    value: float
    units: str

    def __post_init__(self):
        if not self.name or not self.units or not math.isfinite(self.value):
            raise ValueError("Physical dimensions need a name, finite value and units")


@dataclass(frozen=True)
class PhysicalBank:
    key: str
    label: str
    description: str
    dimensions: tuple[Dimension, ...] = ()


@dataclass(frozen=True)
class PhysicalElement:
    key: str
    label: str
    bank: str | None = None
    layer: str | None = None


@dataclass(frozen=True)
class Channel:
    key: str
    ordinal: int
    physical: tuple[str, ...] = ()
    label: str | None = None
    file_labels: tuple[str, ...] = ()
    relationship: str = ""
    selectable: bool = True


@dataclass(frozen=True)
class Acquisition:
    key: str
    label: str
    channels: tuple[Channel, ...]
    views: tuple[View, ...]
    layout_key: str | None = None
    reference_arrows: tuple[ReferenceArrow, ...] = ()

    def __post_init__(self):
        if self.layout_key is not None and not self.layout_key.strip():
            raise ValueError("Acquisition layout key cannot be blank")
        for arrow in self.reference_arrows:
            if not arrow.label or not all(
                math.isfinite(value) for value in (*arrow.start, *arrow.end)
            ):
                raise ValueError("Reference arrows need a label and finite endpoints")


@dataclass(frozen=True)
class Detectors:
    banks: tuple[PhysicalBank, ...]
    elements: tuple[PhysicalElement, ...]
    acquisitions: tuple[Acquisition, ...]


@dataclass(frozen=True)
class ChannelGroup:
    id: int
    name: str
    channels: tuple[str, ...]


@dataclass(frozen=True)
class Preset:
    name: str
    groups: tuple[ChannelGroup, ...]
    forward_group: int
    backward_group: int
    projections: tuple[AsymmetryProjection, ...] = ()


@dataclass(frozen=True)
class ConfigurationPresets:
    acquisition: str
    presets: tuple[Preset, ...]


@dataclass(frozen=True)
class Defaults:
    apply_default_preset_on_load: bool = False


@dataclass(frozen=True)
class Reference:
    key: str
    title: str
    url: str


@dataclass(frozen=True)
class Evidence:
    fact: str
    status: EvidenceStatus
    detail: str
    reference: str | None = None
    locator: str = ""


@dataclass(frozen=True)
class Sources:
    references: tuple[Reference, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Review:
    revision: str
    scope: str
    reviewed: bool = False
    reviewer: str | None = None
    name_consent: bool = False

    @property
    def public_name(self) -> str | None:
        return self.reviewer if self.reviewed and self.name_consent else None


def _unique(values: tuple, description: str) -> set:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {description}")
    return set(values)


@dataclass(frozen=True)
class InstrumentProfile:
    """One instrument's immutable declaration, in the contributor-guide section order.

    Acquired ordinals remain the analysis identity. Hardware affiliations and
    repeated drawing occurrences do not establish independent observations,
    exact count sums, calibrated directions or a detector noise model.
    """

    identity: Identity
    source: Source
    recognition: Recognition
    detectors: Detectors
    presets: tuple[ConfigurationPresets, ...]
    quirks: tuple[str, ...]
    defaults: Defaults
    sources: Sources
    review: Review

    @property
    def fact_keys(self) -> tuple[str, ...]:
        return (
            "identity",
            "source",
            "recognition",
            "physical.banks",
            "physical.elements",
            "quirks",
            "defaults",
            *(
                f"dimension.{bank.key}.{dimension.name}"
                for bank in self.detectors.banks
                for dimension in bank.dimensions
            ),
            *(f"acquisition.{config.key}" for config in self.detectors.acquisitions),
            *(
                f"view.{config.key}.{view.key}"
                for config in self.detectors.acquisitions
                for view in config.views
            ),
            *(
                f"preset.{entry.acquisition}.{preset.name}"
                for entry in self.presets
                for preset in entry.presets
            ),
        )

    def __post_init__(self):
        identity = self.identity
        if (
            not all(
                (
                    identity.key,
                    identity.facility,
                    identity.name,
                    identity.revision,
                    identity.layout_key,
                    identity.display_name,
                )
            )
            or "." not in identity.key
        ):
            raise ValueError("Profile needs facility-qualified identity and revision")
        if self.review.revision != identity.revision or not self.review.scope:
            raise ValueError("Review scope must identify this profile revision")
        if self.review.name_consent and not self.review.reviewer:
            raise ValueError("Name consent requires a reviewer name")
        if any(
            not token.strip()
            for token in (
                *self.recognition.aliases,
                *self.recognition.filename_tokens,
            )
        ):
            raise ValueError("Recognition tokens cannot be blank")
        banks = _unique(tuple(bank.key for bank in self.detectors.banks), "physical bank IDs")
        hardware = _unique(
            tuple(element.key for element in self.detectors.elements), "physical element IDs"
        )
        if any(
            element.bank is not None and element.bank not in banks
            for element in self.detectors.elements
        ):
            raise ValueError("Physical element references an unknown bank")
        configurations = _unique(
            tuple(config.key for config in self.detectors.acquisitions), "acquisition keys"
        )
        _unique(
            tuple(
                config.layout_key or identity.layout_key for config in self.detectors.acquisitions
            ),
            "effective layout keys",
        )
        preset_configs = _unique(
            tuple(entry.acquisition for entry in self.presets), "preset configurations"
        )
        if preset_configs != configurations:
            raise ValueError("Preset configurations must match acquisition configurations")
        for config in self.detectors.acquisitions:
            keys = _unique(tuple(channel.key for channel in config.channels), "channel keys")
            _unique(tuple(channel.ordinal for channel in config.channels), "channel ordinals")
            if not keys or any(
                not channel.key or type(channel.ordinal) is not int or channel.ordinal < 1
                for channel in config.channels
            ):
                raise ValueError(
                    "Channels need nonempty keys and positive integer histogram numbers"
                )
            if any(not set(channel.physical) <= hardware for channel in config.channels):
                raise ValueError("Channel references an unknown physical element")
            _unique(tuple(view.key for view in config.views), "view IDs")
            if not config.views or len({view.mode for view in config.views}) != 1:
                raise ValueError("An acquisition needs views sharing one drawing mode")
            editable = []
            for view in config.views:
                for element in expand_view(view):
                    target = element.target
                    if target.channel is not None and target.channel not in keys:
                        raise ValueError("View references an unknown acquired channel")
                    if target.hardware is not None and target.hardware not in hardware:
                        raise ValueError("View references unknown hardware context")
                    if target.editable:
                        editable.append(target.channel)
            expected = {channel.key for channel in config.channels if channel.selectable}
            if set(editable) != expected or len(editable) != len(expected):
                raise ValueError("Each selectable channel needs exactly one editable occurrence")
            entry = next(entry for entry in self.presets if entry.acquisition == config.key)
            _unique(tuple(preset.name for preset in entry.presets), "preset names")
            for preset in entry.presets:
                groups = _unique(tuple(group.id for group in preset.groups), "preset group IDs")
                if not {preset.forward_group, preset.backward_group} <= groups:
                    raise ValueError("Preset asymmetry legs reference unknown groups")
                if preset.forward_group == preset.backward_group:
                    raise ValueError("Preset asymmetry legs must be different groups")
                for group in preset.groups:
                    if group.id < 1 or not group.channels or not set(group.channels) <= expected:
                        raise ValueError("Preset membership references unavailable channels")
                    _unique(group.channels, "channels within one group")
                for projection in preset.projections:
                    if (
                        not {projection.forward_group, projection.backward_group} <= groups
                        or projection.forward_group == projection.backward_group
                    ):
                        raise ValueError("Projection legs must name distinct existing groups")
                    if not math.isfinite(projection.alpha) or projection.alpha <= 0:
                        raise ValueError("Projection alpha must be finite and positive")
        references = _unique(
            tuple(reference.key for reference in self.sources.references), "source IDs"
        )
        evidence_keys = _unique(
            tuple(evidence.fact for evidence in self.sources.evidence), "evidence fact keys"
        )
        if evidence_keys != set(self.fact_keys):
            raise ValueError(
                f"Evidence must cover exactly the declared facts: missing={set(self.fact_keys) - evidence_keys}, extra={evidence_keys - set(self.fact_keys)}"
            )
        for evidence in self.sources.evidence:
            if not evidence.detail or (
                evidence.reference is not None and evidence.reference not in references
            ):
                raise ValueError("Evidence requires an explanation and valid source reference")
            if evidence.status in {EvidenceStatus.SOURCE, EvidenceStatus.INFERRED} and (
                evidence.reference is None or not evidence.locator
            ):
                raise ValueError(
                    "Source-backed or diagram-inferred facts need a reference and locator"
                )
