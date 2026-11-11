"""Frozen applied TF calibration and its project-boundary codec.

Profiles retain history; coefficient policies point into the same frozen record.
No source file is opened, fitted or reselected by persistence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite, isnan
from pathlib import Path
from typing import TYPE_CHECKING

from asymmetry.core.fitting.count_results import CountFitDomain, CountFitParameter, JointCountFit
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.tf_calibration import (
    TFCalibrationRequest,
    TFCalibrationResult,
    TFCalibrationSelection,
    TFEnvelope,
)
from asymmetry.core.transform.corrections import CorrectionOutcome
from asymmetry.core.transform.fb_counts import AlignmentCrop, BinWindow
from asymmetry.core.transform.projections import _AXIS_ALPHA_KEYS

if TYPE_CHECKING:
    from asymmetry.core.workflow.calibration_results import QuickCalibrationResult


CALIBRATION_PAYLOAD_KEYS = (
    "calibration_records",
    "alpha_policy",
    "beta_policy",
    "projection_alpha_policies",
)


@dataclass(frozen=True)
class CalibrationRecord:
    """Stable identity of a captured fit or selection, including failed diagnostics."""

    calibration_id: str
    result: TFCalibrationResult | TFCalibrationSelection

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_id, str) or not self.calibration_id:
            raise ValueError("Calibration requires a nonempty identity")

    @property
    def selected(self) -> TFCalibrationResult | None:
        return (
            self.result.selected if isinstance(self.result, TFCalibrationSelection) else self.result
        )

    def require_source(self) -> Path:
        """Resolve the recorded input for an explicit rerun, never during loading."""
        result = (
            self.result.candidates[0]
            if isinstance(self.result, TFCalibrationSelection)
            else self.result
        )
        path = Path(result.source_file)
        if not result.source_file or not path.is_file():
            raise ValueError("Calibration rerun requires a resolvable source acquisition file")
        return path


@dataclass(frozen=True)
class CalibratedCoefficient:
    """Read-only scalar compatibility view of one canonical applied candidate."""

    record: CalibrationRecord
    envelope: TFEnvelope
    coefficient: str

    @property
    def result(self) -> TFCalibrationResult:
        result = self.record.result
        if isinstance(result, TFCalibrationSelection):
            return next(c for c in result.candidates if c.envelope == self.envelope)
        return result

    @property
    def mode(self) -> str:
        return "calibrated"

    @property
    def value(self) -> float:
        return self.result.alpha if self.coefficient == "alpha" else self.result.beta

    @property
    def error(self) -> float | None:
        return self.result.fit.standard_error(self.coefficient)

    @property
    def method(self) -> str:
        return self.result.method

    @property
    def source_run(self) -> int:
        return self.result.source_run_number

    def to_dict(self) -> dict:
        return {
            "calibration_id": self.record.calibration_id,
            "envelope": self.envelope.value,
            "coefficient": self.coefficient,
        }


@dataclass(frozen=True)
class QuickCalibratedCoefficient:
    """Scalar policy view backed by the workflow's full captured quick result."""

    result: QuickCalibrationResult

    @property
    def mode(self):
        return "calibrated"

    @property
    def value(self):
        return self.result.estimate.alpha

    @property
    def error(self):
        return self.result.estimate.alpha_error

    @property
    def method(self):
        return self.result.estimate.method

    @property
    def source_run(self):
        return self.result.run_number

    def to_dict(self):
        return {"quick_calibration": self.result.to_dict()}


def restore_coefficient(data, records):
    """Resolve one stored policy through the existing canonical result codecs."""
    if "quick_calibration" in data:
        from asymmetry.core.workflow.calibration_results import (
            CalibrationOutcome,
            QuickCalibrationResult,
        )

        result = QuickCalibrationResult.from_dict(data["quick_calibration"])
        if CalibrationOutcome(result).alpha is None:
            raise ValueError("Applied quick calibration requires a usable positive alpha")
        return QuickCalibratedCoefficient(result)
    by_id = {r.calibration_id: r for r in records}
    if data["calibration_id"] not in by_id:
        raise ValueError("Applied coefficient must reference its canonical calibration")
    record = by_id[data["calibration_id"]]
    view = CalibratedCoefficient(record, TFEnvelope(data["envelope"]), data["coefficient"])
    candidates = (
        record.result.candidates
        if isinstance(record.result, TFCalibrationSelection)
        else (record.result,)
    )
    if not any(c.envelope == view.envelope and c.ok for c in candidates):
        raise ValueError("Applied calibration must reference a usable captured candidate")
    return view


def apply_calibration_outcome(profile, outcome, projection=None):
    """Commit a current workflow outcome without inventing provenance from numbers."""
    from asymmetry.core.project.profiles import BetaPolicy

    if outcome.alpha is None:
        raise ValueError("Cannot apply calibration without a usable result")
    if outcome.record is None:
        coefficient = QuickCalibratedCoefficient(outcome.result)
    else:
        record = outcome.record
        if projection is None:
            apply_calibration(profile, record)
            return
        if record.selected.request.fit_beta:
            raise ValueError("Projection calibration requires beta fixed at one")
        _capture_calibration(profile, record)
        coefficient = CalibratedCoefficient(record, record.selected.envelope, "alpha")
    if projection is None:
        profile.alpha_policy = coefficient
        profile.beta_policy = BetaPolicy(value=1.0)
    else:
        profile.projection_alpha_policies[projection] = coefficient
        key = _AXIS_ALPHA_KEYS.get(projection)
        if key:
            profile.extra.pop(key, None)
        for spec in profile.projections:
            if spec["label"] == projection:
                spec.pop("alpha", None)


def resolve_projection_calibrations(policies, grouping):
    """Derive projection values and attribution from their coefficient owners."""
    for label, policy in policies.items():
        targets = [
            (spec, "alpha") for spec in grouping.get("projections", ()) if spec["label"] == label
        ]
        key = _AXIS_ALPHA_KEYS.get(label)
        if key:
            targets.append((grouping, key))
        for target, prefix in targets:
            for suffix, value in (
                ("", policy.value),
                ("_error", policy.error),
                ("_reference_run", policy.source_run),
                ("_method", policy.method or None),
            ):
                if value is None:
                    target.pop(prefix + suffix, None)
                else:
                    target[prefix + suffix] = value


def calibration_values(payload):
    """Derive runtime coefficient views from canonical override transport at its boundary."""
    if not any(key in payload for key in CALIBRATION_PAYLOAD_KEYS):
        return payload
    policies = restore_profile_calibrations(payload)
    result = {**payload, "projections": [dict(p) for p in (payload.get("projections") or ())]}
    for name in ("alpha", "beta"):
        if f"{name}_policy" in policies and policies[f"{name}_policy"].mode != "per_run_estimate":
            result[name] = policies[f"{name}_policy"].value
    resolve_projection_calibrations(policies["projection_alpha_policies"], result)
    return result


def apply_calibration(profile, record: CalibrationRecord) -> None:
    """Apply a usable captured result to the existing profile policy owner."""
    selected = record.selected
    if selected is None or not selected.ok:
        raise ValueError("Cannot apply a calibration without a usable candidate")
    _capture_calibration(profile, record)
    for name in ("alpha", "beta"):
        setattr(profile, f"{name}_policy", CalibratedCoefficient(record, selected.envelope, name))


def _capture_calibration(profile, record):
    """One frozen identity belongs to exactly one captured result in history."""
    existing = {r.calibration_id: r for r in profile.calibration_records}
    if record.calibration_id in existing and existing[record.calibration_id] != record:
        raise ValueError("Calibration identity must name exactly one frozen record")
    existing[record.calibration_id] = record
    profile.calibration_records = tuple(existing.values())


def _domain_payload(domain: CountFitDomain) -> dict:
    return {
        **{
            f.name: getattr(domain, f.name)
            for f in fields(domain)
            if f.name not in {"alignment", "deadtime", "background_preflight", "raw_windows"}
        },
        "alignment": asdict(domain.alignment),
        "raw_windows": [asdict(w) for w in domain.raw_windows],
        "deadtime": domain.deadtime.to_dict(),
        "background_preflight": [
            (label, outcome.to_dict()) for label, outcome in domain.background_preflight
        ],
    }


def _restore_domain(data: dict) -> CountFitDomain:
    return CountFitDomain(
        **{key: data[key] for key in ("group_id", "bin_width_us")},
        acquired_sources=tuple(tuple(item) for item in data.get("acquired_sources", ())),
        detector_indices=tuple(data["detector_indices"]),
        time_us=tuple(data["time_us"]),
        raw_windows=tuple(BinWindow(**w) for w in data["raw_windows"]),
        alignment=AlignmentCrop(
            **{
                **data["alignment"],
                **{
                    key: tuple(tuple(v) for v in data["alignment"][key])
                    for key in ("timing_residuals_us", "timing_sources")
                },
            }
        ),
        deadtime=CorrectionOutcome.from_dict(data["deadtime"]),
        background_preflight=tuple(
            (label, CorrectionOutcome.from_dict(outcome))
            for label, outcome in data["background_preflight"]
        ),
    )


def calibration_payload(record: CalibrationRecord) -> dict:
    """Encode candidates with shared acquisition/domain facts stored once."""
    selection = isinstance(record.result, TFCalibrationSelection)
    candidates = record.result.candidates if selection else (record.result,)
    first = candidates[0]
    domains = first.fit.domains
    source = (first.source_run_number, first.period_number, first.source_file)
    payloads = []
    for c in candidates:
        if (
            c.fit.domains != domains
            or (c.source_run_number, c.period_number, c.source_file) != source
        ):
            raise ValueError("Calibration candidates must share acquisition and count domains")
        fit = {
            f.name: getattr(c.fit, f.name)
            for f in fields(c.fit)
            if f.name not in {"parameters", "domains", "objective"}
        }
        fit.update(
            parameters=[asdict(p) for p in c.fit.parameters], objective=c.fit.objective.value
        )
        payloads.append(
            {
                "request": asdict(c.request),
                "fit": fit,
                "ok": c.ok,
                "message": c.message,
                "envelope": c.envelope.value,
            }
        )
    return {
        "calibration_id": record.calibration_id,
        "selection": selection,
        "criterion": "minimum_total_cost" if selection else None,
        "source_run_number": source[0],
        "period_number": source[1],
        "source_file": source[2],
        "domains": [_domain_payload(d) for d in domains],
        "candidates": payloads,
    }


def restore_calibration(data: dict) -> CalibrationRecord:
    """Validate canonical identities, constraints and covariance at the project boundary."""
    if data["criterion"] != ("minimum_total_cost" if data["selection"] else None):
        raise ValueError("Unknown calibration selection criterion")
    domains = tuple(_restore_domain(d) for d in data["domains"])
    candidates = []
    for c in data["candidates"]:
        request = TFCalibrationRequest(
            **{
                **c["request"],
                "parameters": tuple(CountFitParameter(**p) for p in c["request"]["parameters"]),
            }
        )
        f = c["fit"]
        parameters = tuple(CountFitParameter(**p) for p in f["parameters"])
        names = [p.name for p in parameters]
        # Free request seeds may be clipped; the full fit owns effective values.
        for effective, specs in ((False, request.parameters), (True, parameters)):
            for p in specs:
                if (
                    isnan(p.min)
                    or isnan(p.max)
                    or p.min > p.max
                    or (not p.fixed and p.min == p.max)
                ):
                    raise ValueError("Calibration parameter bounds must be ordered")
                if p.fixed and (not isfinite(p.value) or not p.min <= p.value <= p.max):
                    raise ValueError(
                        "Fixed calibration parameters must lie in their explicit bounds"
                    )
                if p.name in {"alpha", "beta", "stretch"} and (
                    (isfinite(p.min) and p.min <= 0)
                    or (isfinite(p.max) and p.max <= 0)
                    or (
                        (p.fixed or (effective and c["ok"]))
                        and (not isfinite(p.value) or p.value <= 0)
                    )
                ):
                    raise ValueError(
                        "Calibration balances and stretch require positive physical domains"
                    )
        beta = next((p for p in parameters if p.name == "beta"), None)
        if (
            beta is None
            or beta.fixed == request.fit_beta
            or (not request.fit_beta and beta.value != 1)
        ):
            raise ValueError("Calibration beta constraints must agree with the captured request")
        if c["ok"] and (not f["success"] or any(not isfinite(p.value) for p in parameters)):
            raise ValueError("A usable calibration requires a successful finite fit")
        order = tuple(f["covariance_parameters"])
        if len(set(names)) != len(names) or not {"alpha", "beta"} <= set(names):
            raise ValueError("Calibration parameters require unique names including alpha and beta")
        if len(set(order)) != len(order) or set(order) != {
            p.name for p in parameters if not p.fixed
        }:
            raise ValueError("Calibration covariance order must contain each free parameter once")
        covariance = (
            None if f["covariance"] is None else tuple(tuple(row) for row in f["covariance"])
        )
        if covariance is not None and (
            len(covariance) != len(order) or any(len(row) != len(order) for row in covariance)
        ):
            raise ValueError("Calibration covariance dimensions must match its parameter order")
        fit = JointCountFit(
            **{
                **f,
                "parameters": parameters,
                "covariance_parameters": order,
                "covariance": covariance,
                "objective": Objective(f["objective"]),
                "domains": domains,
                "minos_intervals": tuple(tuple(v) for v in f["minos_intervals"]),
            }
        )
        candidates.append(
            TFCalibrationResult(
                request,
                fit,
                data["source_run_number"],
                data["period_number"],
                data["source_file"],
                c["ok"],
                c["message"],
                TFEnvelope(c["envelope"]),
            )
        )
    envelopes = [c.envelope for c in candidates]
    if (data["selection"] and envelopes != list(TFEnvelope)) or (
        not data["selection"] and len(candidates) != 1
    ):
        raise ValueError("Calibration candidate scope must match the captured operation")
    result = TFCalibrationSelection(tuple(candidates)) if data["selection"] else candidates[0]
    return CalibrationRecord(data["calibration_id"], result)


def profile_calibration_payload(profile, data: dict) -> dict:
    """Capture history and references without storing a second scalar authority."""
    if profile.calibration_records:
        data["calibration_records"] = [calibration_payload(r) for r in profile.calibration_records]
    for name in ("alpha", "beta"):
        policy = getattr(profile, f"{name}_policy")
        if isinstance(policy, CalibratedCoefficient) and not any(
            r is policy.record for r in profile.calibration_records
        ):
            raise ValueError("Applied calibration must belong to profile history")
        if (
            isinstance(policy, (CalibratedCoefficient, QuickCalibratedCoefficient))
            or profile.projections
        ):
            data[f"{name}_policy"] = policy.to_dict()
            for key in (name, f"{name}_method", f"{name}_error", f"{name}_reference_run"):
                data.pop(key, None)
    if profile.projection_alpha_policies:
        data["projection_alpha_policies"] = {
            label: policy.to_dict() for label, policy in profile.projection_alpha_policies.items()
        }
    for label in profile.projection_alpha_policies:
        key = _AXIS_ALPHA_KEYS.get(label)
        if key is not None:
            for suffix in ("", "_error", "_reference_run", "_method"):
                data.pop(key + suffix, None)
                data.get("extra", {}).pop(key + suffix, None)
        for spec in data.get("projections", ()):
            if spec["label"] == label:
                for field in ("alpha", "alpha_error", "alpha_reference_run", "alpha_method"):
                    spec.pop(field, None)
    return data


def restore_profile_calibrations(data: dict) -> dict:
    records = tuple(restore_calibration(r) for r in data.get("calibration_records", ()))
    by_id = {r.calibration_id: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Profile calibration identities must be unique")
    result = {"calibration_records": records}
    for name in ("alpha", "beta"):
        policy = data.get(f"{name}_policy", {})
        if "quick_calibration" in policy:
            if name != "alpha":
                raise ValueError("Quick calibration supplies alpha, with beta fixed at one")
            result[f"{name}_policy"] = restore_coefficient(policy, records)
        elif "calibration_id" in policy:
            if policy["calibration_id"] not in by_id or policy["coefficient"] != name:
                raise ValueError("Applied coefficient must reference its canonical calibration")
            view = restore_coefficient(policy, records)
            result[f"{name}_policy"] = view
        elif "mode" in policy:
            from asymmetry.core.project.profiles import AlphaPolicy, BetaPolicy

            result[f"{name}_policy"] = (AlphaPolicy if name == "alpha" else BetaPolicy).from_dict(
                policy
            )
    projections = {
        label: restore_coefficient(policy, records)
        for label, policy in data.get("projection_alpha_policies", {}).items()
    }
    for policy in projections.values():
        if isinstance(policy, CalibratedCoefficient) and (
            policy.coefficient != "alpha" or not policy.result.beta_fixed
        ):
            raise ValueError("Projection calibration requires alpha with beta fixed at one")
    result["projection_alpha_policies"] = projections
    return result
