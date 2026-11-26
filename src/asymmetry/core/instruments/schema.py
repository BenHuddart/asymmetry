"""Construction-validated instrument facts, channel maps and review provenance."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from asymmetry.core.instruments.geometry import BeamDirection, Occurrence, View, expand_view
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
    """Match metadata aliases or filename tokens followed by separators or run digits."""
    text = str(raw or "").lower()
    if filename:
        text = Path(text).name
        return any(
            re.search(
                r"(?<![a-z0-9])"
                + r"[^a-z0-9]*".join(re.findall(r"[a-z0-9]+", token.lower()))
                + r"(?![a-z])",
                text,
            )
            for token in recognition.filename_tokens
        )
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return any(re.sub(r"[^a-z0-9]+", "", token.lower()) in compact for token in recognition.aliases)


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


def _readout_members(keys: tuple[str, ...], minimum: int = 1) -> None:
    if (
        not isinstance(keys, tuple)
        or len(keys) < minimum
        or any(not isinstance(key, str) or not key.strip() for key in keys)
    ):
        raise ValueError("Readout relationships require immutable nonempty channel keys")
    if len(set(keys)) != len(keys):
        raise ValueError("Readout relationships require distinct channel keys")


@dataclass(frozen=True)
class ReadoutSet:
    """Allowed acquired channels; the first set is the default for new settings."""

    key: str
    label: str
    channels: tuple[str, ...]
    default_preset: str

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.key, self.label, self.default_preset)
        ):
            raise ValueError("Readout sets require a key, label and default preset")
        _readout_members(self.channels)


@dataclass(frozen=True)
class CombinedReadout:
    """Shared event content, without an exact-bin sum or covariance assertion."""

    channel: str
    constituents: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("Combined readouts require a channel key")
        _readout_members(self.constituents, 2)


@dataclass(frozen=True)
class ConditionalContribution:
    """One contributor may enter one of the mutually exclusive destinations."""

    channel: str
    destinations: tuple[str, ...]
    label: str

    def __post_init__(self):
        if (
            not isinstance(self.channel, str)
            or not self.channel.strip()
            or not isinstance(self.label, str)
            or not self.label.strip()
        ):
            raise ValueError("Conditional contributions require a label")
        _readout_members(self.destinations)


@dataclass(frozen=True)
class Acquisition:
    key: str
    label: str
    channels: tuple[Channel, ...]
    views: tuple[View, ...]
    layout_key: str | None = None
    reference_arrows: tuple[ReferenceArrow, ...] = ()
    beam: BeamDirection | None = None
    auxiliary: tuple[Occurrence, ...] = ()
    readout_sets: tuple[ReadoutSet, ...] = ()
    combined_readouts: tuple[CombinedReadout, ...] = ()
    conditional_contributions: tuple[ConditionalContribution, ...] = ()

    def __post_init__(self):
        if any(
            not isinstance(values, tuple)
            for values in (
                self.readout_sets,
                self.combined_readouts,
                self.conditional_contributions,
            )
        ):
            raise ValueError("Acquisition readout declarations require immutable tuples")
        if self.beam is not None and not isinstance(self.beam, BeamDirection):
            raise ValueError("Acquisition beam requires a typed BeamDirection")
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
class OrientationGroupingDefault:
    """A file condition may use this preset for the declared detector orientation."""

    acquisition: str
    detector_orientation: str
    preset: str

    def __post_init__(self):
        if (
            not self.acquisition.strip()
            or not self.preset.strip()
            or self.detector_orientation not in {"Longitudinal", "Transverse"}
        ):
            raise ValueError(
                "File grouping default requires a map, canonical orientation and preset"
            )


@dataclass(frozen=True)
class Defaults:
    apply_default_preset_on_load: bool = False
    absent_file_grouping: OrientationGroupingDefault | None = None
    legacy_v1_zero_grouping: OrientationGroupingDefault | None = None


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
                f"readout_set.{c.key}.{r.key}"
                for c in self.detectors.acquisitions
                for r in c.readout_sets
            ),
            *(
                f"combined_readout.{c.key}.{r.channel}"
                for c in self.detectors.acquisitions
                for r in c.combined_readouts
            ),
            *(
                f"conditional_contribution.{c.key}.{r.channel}"
                for c in self.detectors.acquisitions
                for r in c.conditional_contributions
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
        for rule in (self.defaults.absent_file_grouping, self.defaults.legacy_v1_zero_grouping):
            if rule is not None and not any(
                entry.acquisition == rule.acquisition
                and any(preset.name == rule.preset for preset in entry.presets)
                for entry in self.presets
            ):
                raise ValueError(
                    "File grouping default requires an existing acquisition and preset"
                )
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
            for target in config.auxiliary:
                if target.channel not in keys or not target.editable:
                    raise ValueError("Auxiliary readouts must name editable acquired channels")
                editable.append(target.channel)
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
            _unique(tuple(r.key for r in config.readout_sets), "readout-set keys")
            combined = _unique(
                tuple(r.channel for r in config.combined_readouts), "combined readouts"
            )
            _unique(
                tuple(r.channel for r in config.conditional_contributions),
                "conditional contributors",
            )
            if (
                config.readout_sets
                and set().union(*(set(r.channels) for r in config.readout_sets)) != expected
            ):
                raise ValueError("Readout sets must cover exactly the selectable acquired channels")
            for road in config.combined_readouts:
                if (
                    road.channel not in expected
                    or not set(road.constituents) <= expected
                    or combined.intersection(road.constituents)
                ):
                    raise ValueError("Combined readouts require acquired non-combined constituents")
            for contribution in config.conditional_contributions:
                if (
                    contribution.channel not in expected
                    or not set(contribution.destinations) <= combined
                    or contribution.channel in contribution.destinations
                ):
                    raise ValueError(
                        "Conditional contributions require acquired channels and combined destinations"
                    )
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
            for readouts in config.readout_sets:
                default = next(
                    (p for p in entry.presets if p.name == readouts.default_preset), None
                )
                if default is None or not {
                    key for group in default.groups for key in group.channels
                } <= set(readouts.channels):
                    raise ValueError(
                        "Readout-set default preset must use channels available in that set"
                    )
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
