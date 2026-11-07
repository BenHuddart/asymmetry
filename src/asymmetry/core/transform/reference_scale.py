"""Reference-count scale intent, measured exposure, and fixed-scale provenance."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class ReferenceExposure:
    """An evidenced acquisition exposure; ``None`` means exposure is unknown.

    The deadtime denominator is a separate acquisition fact. In particular,
    whole-run deadtime normalisation does not measure a pulse-frame exposure.
    """

    value: float | None = None
    origin: str = "unknown"
    unit: str = "good_frames"
    period_number: int = 1

    def __post_init__(self):
        if (
            not isinstance(self.origin, str)
            or not self.origin
            or not isinstance(self.unit, str)
            or not self.unit
        ):
            raise ValueError("Reference exposure requires its origin and unit")
        if (
            not isinstance(self.period_number, int)
            or isinstance(self.period_number, bool)
            or self.period_number < 1
        ):
            raise ValueError("Reference exposure requires an acquired period")
        if self.value is not None and (
            isinstance(self.value, bool)
            or not isfinite(self.value)
            or self.value <= 0
            or self.origin.startswith("unknown")
            or self.origin == "whole_run_deadtime"
        ):
            raise ValueError("Reference exposure requires a finite positive value and its origin")

    @classmethod
    def from_dict(cls, value: dict | None):
        """Read an acquisition fact; untagged frame numbers are not exposure evidence."""
        if value is None:
            return cls()
        if not isinstance(value, dict):
            return cls(origin="unknown:unusable declaration")
        try:
            return cls(**value)
        except (TypeError, ValueError, OverflowError):
            return cls(origin="unknown:unusable declaration")

    @classmethod
    def from_grouping(cls, grouping: dict):
        """Read subtraction exposure independently of the DT normaliser."""
        return cls.from_dict(grouping.get("reference_exposure"))

    def to_dict(self):
        """Return JSON-compatible acquisition provenance."""
        return asdict(self)


@dataclass(frozen=True)
class ReferenceScaleRequest:
    """Automatic exposure scaling or an unconditional authored Manual coefficient."""

    mode: Literal["automatic", "manual"] = "automatic"
    coefficient: float | None = None

    def __post_init__(self):
        if self.mode not in ("automatic", "manual"):
            raise ValueError("Reference scale mode must be Automatic or Manual")
        if self.mode == "manual":
            if (
                self.coefficient is None
                or isinstance(self.coefficient, bool)
                or not isfinite(self.coefficient)
                or self.coefficient <= 0
            ):
                raise ValueError("Manual reference coefficient must be finite and positive")
        elif self.coefficient is not None:
            raise ValueError("Automatic reference scale has no authored coefficient")

    @classmethod
    def from_payload(cls, payload: dict):
        """Validate authored settings at the user/project request boundary."""
        if {"scale", "good_frames_sample", "good_frames_reference"} & payload.keys():
            raise ValueError(
                "Use scale_request for Automatic or Manual reference scaling; historical snapshots require project migration"
            )
        value = payload.get("scale_request", {"mode": "automatic"})
        if not isinstance(value, dict):
            raise ValueError("Reference scale requires an Automatic or Manual request")
        try:
            coefficient = value.get("coefficient")
            if coefficient is not None and not isinstance(coefficient, bool):
                coefficient = float(coefficient)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Manual reference coefficient must be finite and positive") from None
        return cls(value.get("mode"), coefficient)

    def to_dict(self):
        """Return only authored intent, without an applied scale snapshot."""
        return {"mode": self.mode} | (
            {"coefficient": self.coefficient} if self.mode == "manual" else {}
        )

    def resolve(
        self,
        sample: ReferenceExposure = ReferenceExposure(),
        reference: ReferenceExposure = ReferenceExposure(),
    ):
        """Resolve compatible facts from the selected acquisitions, or assume unity."""
        if self.mode == "manual":
            return ResolvedReferenceScale(self, self.coefficient, "manual", sample, reference)
        if (
            sample.value is not None
            and reference.value is not None
            and sample.unit == reference.unit
        ):
            coefficient = sample.value / reference.value
            if isfinite(coefficient) and coefficient > 0:
                return ResolvedReferenceScale(self, coefficient, "live_exposure", sample, reference)
        return ResolvedReferenceScale(self, 1.0, "exposure_unknown", sample, reference)


@dataclass(frozen=True)
class ResolvedReferenceScale:
    """Applied fixed coefficient and exposure evidence, separate from authored intent."""

    request: ReferenceScaleRequest
    coefficient: float
    source: str
    sample: ReferenceExposure
    reference: ReferenceExposure

    def __post_init__(self):
        if not isfinite(self.coefficient) or self.coefficient <= 0:
            raise ValueError("Resolved reference coefficient must be finite and positive")

    def provenance(self):
        """Record the exact-scale assumption and the facts consumed."""
        return {
            "request": self.request.to_dict(),
            "coefficient": self.coefficient,
            "source": self.source,
            "sample": self.sample.to_dict(),
            "reference": self.reference.to_dict(),
            "uncertainty": "fixed scale; independent sample/reference variances",
            "assumption": "Exposure unknown; scale 1 assumed"
            if self.source == "exposure_unknown"
            else "",
        }

    def summary(self):
        """Human-readable result provenance shared by previews and exports."""
        if self.source == "exposure_unknown":
            return "Exposure unknown; scale 1 assumed"
        label = "Manual" if self.request.mode == "manual" else "Automatic exposure ratio"
        return f"{label}: {self.coefficient:.6g}"
