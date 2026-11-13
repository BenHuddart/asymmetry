"""Captured scalar-scan executions, independent of current optimizers and sources."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np

from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.fitting.field_scan import ScanBaselineResult
from asymmetry.core.fitting.parameter_models import ParameterCompositeModel, ParameterModelFitResult
from asymmetry.core.fitting.result_payload import (
    _deserialize_parameter_set,
    _serialize_parameter_set,
)
from asymmetry.core.project.json_values import encode_non_finite
from asymmetry.core.transform.corrections import CorrectionOutcomes
from asymmetry.core.transform.integral import FieldScan
from asymmetry.core.transform.scalar_observation import ScalarObservation
from asymmetry.core.transform.units import AsymmetryUnit


def scan_support(scan: FieldScan) -> dict:
    """Capture actual coordinates, observations, units and recipe, without raw acquisitions."""
    return {
        "x": scan.x.tolist(),
        "value": scan.value.tolist(),
        "error": scan.error.tolist(),
        "run_numbers": list(scan.run_numbers),
        "order_key": scan.order_key,
        "method": scan.method,
        "derivative": scan.derivative,
        "x_label": scan.x_label,
        "y_label": scan.y_label,
        "units": scan.units.value,
        "source_assumption": scan.source_assumption,
        "computation_revision": scan.computation_revision,
        "observations": [o.to_dict() if o is not None else None for o in scan.observations],
        "correlated_pairs": [list(pair) for pair in scan.correlated_pairs],
        "excluded": [list(item) for item in scan.excluded],
        "correction_outcomes": {
            str(run): {key: value.to_dict() for key, value in outcomes.items()}
            for run, outcomes in scan.correction_outcomes.items()
        },
        "effective_recipe": scan.effective_recipe.to_dict() if scan.effective_recipe else None,
    }


def restore_scan(data: dict) -> FieldScan:
    """Restore fitted support, retaining unknown execution attribution."""
    values = deepcopy(data)
    for name in ("x", "value", "error"):
        values[name] = np.asarray(values[name], dtype=float)
        values[name].setflags(write=False)
    values["observations"] = tuple(
        ScalarObservation.from_dict(o) if o is not None else None for o in values["observations"]
    )
    values["correlated_pairs"] = tuple(tuple(pair) for pair in values["correlated_pairs"])
    values["units"] = AsymmetryUnit(values["units"])
    values["correction_outcomes"] = {
        int(run): {key: CorrectionOutcomes.from_dict(value) for key, value in outcomes.items()}
        for run, outcomes in values["correction_outcomes"].items()
    }
    recipe = values["effective_recipe"]
    values["effective_recipe"] = EffectiveReductionRecipe.from_dict(recipe) if recipe else None
    return FieldScan(**values)


def _captured_covariance(covariance, expected):
    """Covariance axes must identify exactly the fitted free parameters."""
    if covariance is None:
        raise ValueError("Scan history requires captured fit covariance.")
    names, values = covariance
    matrix = np.asarray(values, dtype=float)
    if (
        len(names) != len(expected)
        or set(names) != expected
        or matrix.shape != (len(names), len(names))
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T)
        or np.any(np.diag(matrix) < 0)
    ):
        raise ValueError("Scan covariance axes and matrix must match fitted free parameters.")


@dataclass(frozen=True)
class ScanFitRecord:
    """One completed solve and its sampled prediction on captured display support.

    Peak covariance is the baseline-propagated result, separate from the
    conditional optimizer covariance. A peak references its exact baseline
    execution by index in the owning history. Predictions are retained outputs,
    so opening does not evaluate a potentially changed model implementation.
    """

    kind: str
    support: dict
    settings: dict
    model: dict
    result: dict
    prediction: list[float]
    request: dict = field(default_factory=dict)
    baseline_index: int | None = None
    covariance: tuple[list[str], list[list[float]]] | None = None
    effective_dof: float | None = None

    @classmethod
    def capture(cls, kind, scan, settings, model, fit, prediction, **kwargs):
        """Freeze the completed result using the shared parameter codec."""
        result = asdict(fit)
        result["parameters"] = _serialize_parameter_set(fit.parameters)
        return cls(
            kind,
            scan_support(scan),
            deepcopy(settings),
            model.to_dict(),
            result,
            np.asarray(prediction, dtype=float).tolist(),
            **deepcopy(kwargs),
        )

    def fit_result(self) -> ParameterModelFitResult:
        """Reconstitute retained parameters/statistics without a solve."""
        data = deepcopy(self.result)
        for key in ("chi_squared", "reduced_chi_squared"):
            data[key] = float(data[key])
        data["uncertainties"] = {
            name: float(value) for name, value in data["uncertainties"].items()
        }
        data["parameters"] = _deserialize_parameter_set(data["parameters"])
        if data["covariance"] is not None:
            names, matrix = data["covariance"]
            data["covariance"] = (names, matrix)
        data["params_at_bound"] = tuple(data["params_at_bound"])
        return ParameterModelFitResult(**data)

    def baseline_result(self) -> ScanBaselineResult:
        """Restore the baseline dependency for an explicitly requested peak fit."""
        if self.kind != "baseline":
            raise ValueError("A peak fit requires a captured baseline execution.")
        return ScanBaselineResult(
            restore_scan(self.support),
            ParameterCompositeModel.from_dict(self.model),
            tuple(tuple(region) for region in self.settings["regions"]),
            self.fit_result(),
            captured_baseline=tuple(self.prediction),
        )

    def matches(self, scan, settings):
        """Execution revision is historical; applicability concerns inputs and settings."""
        current = scan_support(scan)
        return (
            all(
                encode_non_finite(current[key]) == encode_non_finite(value)
                for key, value in self.support.items()
                if key != "computation_revision"
            )
            and self.settings == settings
        )


@dataclass
class ScanFitHistory:
    """The scan owns executions; active indices alone control their applicability."""

    records: list[ScanFitRecord] = field(default_factory=list)
    active: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    legacy_settings: dict = field(default_factory=dict)

    def capture(self, record: ScanFitRecord):
        """Append an explicit execution, retaining older numerical history."""
        if record.kind == "baseline":
            self.invalidate("peaks", "rf")
        else:
            self.invalidate("rf")
        if record.kind == "rf":
            self.invalidate("baseline", "peaks")
        self.active[record.kind] = len(self.records)
        self.records.append(record)
        self.missing = [kind for kind in self.missing if kind != record.kind]

    def invalidate(self, *kinds):
        """Detach stale overlays without deleting their historical executions."""
        for kind in kinds:
            self.active.pop(kind, None)

    def to_dict(self):
        """Return detached JSON-compatible records and active references."""
        return {
            "records": [asdict(record) for record in self.records],
            "active": dict(self.active),
            "missing": list(self.missing),
            "legacy_settings": deepcopy(self.legacy_settings),
        }

    @classmethod
    def from_dict(cls, data):
        """Decode at the project boundary; malformed references are never guessed."""
        records = [ScanFitRecord(**deepcopy(record)) for record in data["records"]]
        active = dict(data["active"])
        for kind, index in active.items():
            if (
                type(index) is not int
                or not 0 <= index < len(records)
                or records[index].kind != kind
            ):
                raise ValueError("Scan fit history has an invalid active execution reference.")
        for index, record in enumerate(records):
            scan = restore_scan(record.support)
            if np.asarray(record.prediction, dtype=float).shape != scan.x.shape:
                raise ValueError("Scan fit prediction must match captured coordinates.")
            if record.kind not in {"baseline", "peaks", "rf"}:
                raise ValueError("Scan history has an unknown fit kind.")
            if not isinstance(record.settings, dict) or not isinstance(record.request, dict):
                raise ValueError("Scan history requires captured settings and request mappings.")
            if (
                scan.value.shape != scan.x.shape
                or scan.error.shape != scan.x.shape
                or len(scan.run_numbers) != scan.n_points
                or (scan.observations and len(scan.observations) != scan.n_points)
            ):
                raise ValueError("Scan history requires parallel captured observations.")
            for parameter in record.result["parameters"]:
                if "value" not in parameter:
                    raise ValueError("Scan history requires captured parameter values.")
            fit = record.fit_result()
            model = ParameterCompositeModel.from_dict(record.model)
            if not fit.success or set(fit.parameters.names) != set(model.param_names):
                raise ValueError("Scan history requires completed model parameters.")
            if record.kind == "rf" and model.component_names != ["RFResonanceMuP"]:
                raise ValueError("RF scan history requires the RF resonance model.")
            if record.kind == "peaks" and not set(model.component_names) <= {
                "GaussianLCR",
                "LorentzianLCR",
            }:
                raise ValueError("Scan peak history requires supported resonance components.")
            free = {parameter.name for parameter in fit.parameters.free_parameters}
            if fit.covariance is not None or record.kind in {"baseline", "peaks"}:
                _captured_covariance(fit.covariance, free)
            if record.kind == "baseline":
                record.baseline_result()
            if record.kind == "peaks":
                if record.covariance is None or record.effective_dof is None:
                    raise ValueError("Scan peak history requires baseline-propagated covariance.")
                _captured_covariance(record.covariance, free)
            if record.kind == "peaks" and (
                record.baseline_index is None
                or not 0 <= record.baseline_index < index
                or records[record.baseline_index].kind != "baseline"
            ):
                raise ValueError("Scan peak history requires its preceding baseline execution.")
        if "rf" in active and ({"baseline", "peaks"} & active.keys()):
            raise ValueError("RF and baseline/peak fits cannot share an active overlay.")
        if "peaks" in active and records[active["peaks"]].baseline_index != active.get("baseline"):
            raise ValueError("Active scan peaks require their captured baseline.")
        return cls(
            records, active, list(data["missing"]), deepcopy(data.get("legacy_settings", {}))
        )
