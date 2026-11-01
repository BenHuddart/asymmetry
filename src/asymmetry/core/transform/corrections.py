"""Requested and effective correction records shared by reduction entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from asymmetry.core.data.dataset import Histogram, MuonDataset


class CorrectionStatus(StrEnum):
    OFF = "off"
    APPLIED = "applied"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class CorrectionUnavailableError(ValueError):
    """A requested correction has no usable result for final analysis."""


def _freeze_effective(value):
    """Own a read-only tree of JSON correction settings and numerical results."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("correction provenance keys must be strings")
        return MappingProxyType({key: _freeze_effective(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_effective(item) for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real) and isfinite(value):
        return int(value) if isinstance(value, Integral) else float(value)
    raise ValueError("correction provenance requires finite JSON values")


def _export_effective(value):
    """Copy correction provenance into independent JSON objects and arrays."""
    if isinstance(value, Mapping):
        return {key: _export_effective(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_export_effective(item) for item in value]
    return value


@dataclass(frozen=True)
class CorrectionOutcome:
    """One correction's request and effective values, with an explicit status."""

    name: str
    requested: str
    status: CorrectionStatus
    reason: str | None = None
    effective: Mapping[str, Any] | None = None
    uncertainty: str | None = None

    def __post_init__(self):
        status = CorrectionStatus(self.status)
        if not isinstance(self.requested, str) or not self.requested:
            raise ValueError("a correction request requires a nonempty mode name")
        if self.name not in {"deadtime", "background"}:
            raise ValueError("a correction outcome must name deadtime or background")
        if (status == CorrectionStatus.OFF) != (self.requested in {"off", "none"}):
            raise ValueError("correction status is off exactly when the request is off or none")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("correction reasons must be strings")
        if self.uncertainty is not None and not isinstance(self.uncertainty, str):
            raise ValueError("correction uncertainty status must be a string")
        if self.effective is not None and not isinstance(self.effective, Mapping):
            raise ValueError("effective correction values require a mapping")
        incomplete = status in {CorrectionStatus.UNAVAILABLE, CorrectionStatus.FAILED}
        if (incomplete and not self.reason) or (not incomplete and self.reason is not None):
            raise ValueError(
                "an incomplete correction requires a reason; other outcomes forbid one"
            )
        applied = status == CorrectionStatus.APPLIED
        if (
            applied != (self.effective is not None)
            or (applied and not self.uncertainty)
            or (not applied and self.uncertainty is not None)
        ):
            raise ValueError("effective values and uncertainty belong only to applied corrections")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "effective", _freeze_effective(self.effective))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requested": self.requested,
            "status": self.status.value,
            "reason": self.reason,
            "effective": _export_effective(self.effective),
            "uncertainty": self.uncertainty,
        }

    def __reduce__(self):
        """Rebuild the owned read-only tree when crossing process-worker boundaries."""
        return type(self), (
            self.name,
            self.requested,
            self.status,
            self.reason,
            _export_effective(self.effective),
            self.uncertainty,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorrectionOutcome:
        return cls(**value)


@dataclass(frozen=True)
class CorrectionOutcomes:
    """Both correction decisions for a single reduction, including explicit off modes."""

    deadtime: CorrectionOutcome = CorrectionOutcome("deadtime", "off", CorrectionStatus.OFF)
    background: CorrectionOutcome = CorrectionOutcome("background", "none", CorrectionStatus.OFF)

    def __post_init__(self):
        if self.deadtime.name != "deadtime" or self.background.name != "background":
            raise ValueError("correction outcomes must occupy their named slots")

    @property
    def complete(self) -> bool:
        return all(
            outcome.status in {CorrectionStatus.OFF, CorrectionStatus.APPLIED}
            for outcome in (self.deadtime, self.background)
        )

    def require_complete(self, *, context: str = "") -> None:
        for outcome in (self.deadtime, self.background):
            if outcome.status in {CorrectionStatus.UNAVAILABLE, CorrectionStatus.FAILED}:
                raise CorrectionUnavailableError(
                    (f"{context}: " if context else "")
                    + f"{outcome.name} correction requested ({outcome.requested}) "
                    f"but not applied: {outcome.reason}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"deadtime": self.deadtime.to_dict(), "background": self.background.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorrectionOutcomes:
        return cls(
            CorrectionOutcome.from_dict(value["deadtime"]),
            CorrectionOutcome.from_dict(value["background"]),
        )


@dataclass(frozen=True)
class BackgroundReference:
    """Resolved reference counts and the exposure actually used in subtraction."""

    histograms: list[Histogram]
    scale: float
    run_number: int | None = None
    source_file: str = ""
    good_frames_sample: float | None = None
    good_frames_reference: float | None = None
    scale_source: str = "explicit_scale"

    def __post_init__(self):
        object.__setattr__(self, "source_file", Path(self.source_file).name)
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("reference background scale must be finite and positive")


def require_complete_dataset(dataset: MuonDataset) -> None:
    """Reject arrays whose requested correction could not be restored or applied."""
    reason = dataset.metadata.get("correction_error")
    if reason is not None:
        raise CorrectionUnavailableError(
            f"Run {dataset.run_number}: Corrections incomplete: {reason}"
        )


def background_grouping_metadata(outcomes: Mapping[str, CorrectionOutcomes]) -> dict[str, Any]:
    """Derive legacy background display values without changing request settings."""
    applied = {
        period: item.background.to_dict()["effective"]
        for period, item in outcomes.items()
        if item.background.status == CorrectionStatus.APPLIED
    }
    if not applied:
        return {}
    methods = {item["method"] for item in applied.values()}
    result = {"background_method": next(iter(methods)) if len(methods) == 1 else "per_period"}
    records = list(applied.values())
    levels = records[0].get("values")
    if levels is not None and all(item.get("values") == levels for item in records):
        result["background_values"] = levels
    if len(records) == 1:
        details = records[0].get("details", {})
        if details:
            result["background_details"] = details
    else:
        result["background_details"] = {"periods": applied}
    return result
