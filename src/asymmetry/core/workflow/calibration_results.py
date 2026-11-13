"""Workflow views of shared calibration results and immutable quick context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite

from asymmetry.core.data.acquisition import ContentIdentity
from asymmetry.core.fitting.tf_calibration import TF_CALIBRATION_METHOD, TFCalibrationSelection
from asymmetry.core.project.calibration import CalibrationRecord, calibration_payload
from asymmetry.core.project.json_values import encode_non_finite
from asymmetry.core.transform.asymmetry import AlphaEstimate
from asymmetry.core.transform.corrections import CorrectionOutcomes
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow


@dataclass(frozen=True)
class QuickCalibrationResult:
    """One shared estimate, plus the source facts needed to interpret/replay it."""

    estimate: AlphaEstimate
    run_number: int
    forward_group: int
    backward_group: int
    period_number: int | None
    source_file: str
    corrections: CorrectionOutcomes
    alignment: AlignmentCrop
    window: BinWindow
    bin_width_us: float
    t0_time_us: float
    period_mapping: tuple[tuple[int, str], ...] = ()
    mapped_period_number: int | None = None
    computation_revision: str | None = None
    content_identity: ContentIdentity | None = None

    def to_dict(self) -> dict:
        """Export shared result fields and source context without histogram arrays."""
        return {
            **asdict(self.estimate),
            **{
                f.name: getattr(self, f.name)
                for f in fields(self)
                if f.name
                not in {"estimate", "corrections", "alignment", "window", "content_identity"}
            },
            "content_identity": self.content_identity.to_dict()
            if self.content_identity is not None
            else None,
            "corrections": self.corrections.to_dict(),
            "alignment": asdict(self.alignment),
            "window": asdict(self.window),
        }

    @classmethod
    def from_dict(cls, data: dict) -> QuickCalibrationResult:
        """Restore the shared typed estimate and existing correction/timing owners."""
        alignment = data["alignment"]
        return cls(
            estimate=AlphaEstimate(**{f.name: data[f.name] for f in fields(AlphaEstimate)}),
            **{
                f.name: data[f.name]
                for f in fields(cls)
                if f.name
                not in {
                    "estimate",
                    "corrections",
                    "alignment",
                    "window",
                    "period_mapping",
                    "mapped_period_number",
                    "computation_revision",
                    "content_identity",
                }
            },
            period_mapping=tuple((int(n), target) for n, target in data["period_mapping"]),
            mapped_period_number=data.get("mapped_period_number"),
            computation_revision=data.get("computation_revision"),
            content_identity=ContentIdentity.from_dict(data.get("content_identity")),
            corrections=CorrectionOutcomes.from_dict(data["corrections"]),
            alignment=AlignmentCrop(
                **{
                    **alignment,
                    **{
                        name: tuple(tuple(v) for v in alignment[name])
                        for name in ("timing_residuals_us", "timing_sources")
                    },
                }
            ),
            window=BinWindow(**data["window"]),
        )


@dataclass(frozen=True)
class CalibrationOutcome:
    """A shared typed result with source/pair context, without copied coefficients."""

    result: QuickCalibrationResult | CalibrationRecord

    @property
    def record(self) -> CalibrationRecord | None:
        return self.result if isinstance(self.result, CalibrationRecord) else None

    @property
    def selected(self):
        return self.record.selected if self.record is not None else self.result.estimate

    @property
    def alpha(self) -> float | None:
        selected = self.selected
        if selected is None or not selected.ok:
            return None
        return selected.alpha if isfinite(selected.alpha) and selected.alpha > 0 else None

    @property
    def message(self) -> str:
        selected = self.selected
        if selected is None:
            return "No usable TF candidate"
        if self.alpha is None and selected.ok:
            return "Calibration requires a finite positive alpha"
        return selected.message

    @property
    def method(self) -> str:
        return TF_CALIBRATION_METHOD if self.record is not None else self.result.estimate.method

    @property
    def source(self):
        """The first candidate owns common source facts even without a selection."""
        if self.record is None:
            return self.result
        result = self.record.result
        return result.candidates[0] if isinstance(result, TFCalibrationSelection) else result

    @property
    def run_number(self) -> int:
        return self.source.source_run_number if self.record is not None else self.source.run_number

    @property
    def period_number(self) -> int | None:
        return self.source.period_number

    @property
    def forward_group(self) -> int:
        return (
            self.source.request.forward_group
            if self.record is not None
            else self.source.forward_group
        )

    @property
    def backward_group(self) -> int:
        return (
            self.source.request.backward_group
            if self.record is not None
            else self.source.backward_group
        )

    def to_dict(self) -> dict:
        """Derive scalar views; preserve the canonical block before JSON sanitising."""
        selected = self.selected
        data = {
            "run_number": self.run_number,
            "period_number": self.period_number,
            "forward_group": self.forward_group,
            "backward_group": self.backward_group,
            "method": self.method,
            "alpha": self.alpha,
        }
        if self.record is not None:
            data["calibration"] = encode_non_finite(calibration_payload(self.record))
            data.update(
                alpha_error=None if selected is None else selected.alpha_error,
                beta=1.0
                if not self.source.request.fit_beta
                else None
                if selected is None
                else selected.beta,
                beta_error=None if selected is None else selected.beta_error,
                beta_fixed=not self.source.request.fit_beta,
                alpha_beta_correlation=None
                if selected is None
                else selected.alpha_beta_correlation,
                model=None if selected is None else selected.model,
                uncertainty_status=None if selected is None else selected.fit.uncertainty_status,
            )
        else:
            data.update(
                self.result.to_dict(), alpha=self.alpha, beta=1.0, beta_fixed=True, beta_error=None
            )
        data.update(ok=self.alpha is not None, message=self.message)
        return data
