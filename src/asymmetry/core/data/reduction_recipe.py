"""Immutable, JSON-only interpretation of an effective scientific reduction.

Recipes describe resolved operations, not raw replay buffers or cache validity.
Source locators and diagnostic prose are retained but do not identify numerical
requests. Unknown acquisition content identities remain explicitly unknown.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from asymmetry.core import computation
from asymmetry.core.data.acquisition import ContentIdentity, SourceDeclaration

if TYPE_CHECKING:
    from asymmetry.core.transform.constituent_counts import ConstituentCounts
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.fb_counts import ForwardBackwardCounts
    from asymmetry.core.transform.grouping import ForwardBackwardLayout
    from asymmetry.core.transform.integral import FieldScanPoint
    from asymmetry.core.transform.rebin import BinnedRatio, BinningSpec

_PRESENTATION = frozenset(
    {
        "source_file",
        "file_name",
        "title",
        "run_label",
        "message",
        "projection_label",
        "scale_summary",
    }
)


def numerical_recipe(value: Any, *, include_execution: bool = False) -> Any:
    """Separate locators and prose from numerical facts, retaining source references."""
    if isinstance(value, dict):
        # CountFitDomain descriptors put the locator at position 2; ordinals stay numerical.
        return {
            k: (
                [list(source[:2]) + [None] + list(source[3:]) for source in v]
                if k == "acquired_sources"
                else numerical_recipe(v, include_execution=include_execution)
            )
            for k, v in (
                value if include_execution else computation.without_revision(value)
            ).items()
            if k not in _PRESENTATION and not (k == "reason" and isinstance(v, str))
        }
    if isinstance(value, list):
        return [numerical_recipe(v, include_execution=include_execution) for v in value]
    return value


@dataclass(frozen=True)
class EffectiveReductionRecipe:
    """Versioned canonical JSON with no mutable values or executable objects.

    ``identity`` identifies the recorded interpretation and available source
    references; it is not a whole-file content fingerprint. ``to_dict`` returns
    a detached view. This record alone never grants histogram replay, covariance,
    or likelihood eligibility.
    """

    canonical_json: str

    def __post_init__(self) -> None:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("Effective reduction recipe requires version 1")
        object.__setattr__(
            self,
            "canonical_json",
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EffectiveReductionRecipe:
        """Capture a detached JSON record; reject arrays and non-finite numbers."""
        return cls(json.dumps(dict(value), allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        """Return an independent serializable view of the captured authority."""
        return json.loads(self.canonical_json)

    def publishable(self) -> dict[str, Any]:
        """Disclose numerical facts without nested source locators or diagnostic prose.

        The detached projection retains numerical identity and never mutates the
        private calibration, reference, or constituent history. Existing export
        titles and comments have their own disclosure policy.
        """
        return numerical_recipe(self.to_dict(), include_execution=True)

    @property
    def identity(self) -> str:
        """SHA-256 of canonical numerical facts, excluding locators and prose."""
        payload = json.dumps(
            numerical_recipe(self.to_dict()), sort_keys=True, separators=(",", ":")
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def sources(self) -> tuple[SourceDeclaration, ...]:
        """Inventory captured result inputs without looking up or certifying files.

        Constituent order, signed coefficients, and acquired period selectors
        survive even when several terms share identical physical file bytes.
        """
        sources = []

        def visit(value, role="sample", coefficient=1.0, input_number=None):
            if isinstance(value, dict):
                coefficient = value.get("coefficient", coefficient)
                input_number = value.get("input_number", input_number)
                if "source_file" in value:
                    sources.append(
                        SourceDeclaration(
                            role,
                            value["source_file"],
                            ContentIdentity.from_dict(value.get("content_identity")),
                            value.get("run_number", value.get("source_run_number")),
                            value.get("data_period_number", value.get("period_number")),
                            coefficient,
                            input_number,
                            value.get("dae_period_number"),
                        )
                    )
                for key, child in value.items():
                    visit(
                        child,
                        key if key in {"reference", "calibration", "constituents"} else role,
                        coefficient,
                        input_number,
                    )
            elif isinstance(value, list):
                for child in value:
                    visit(child, role, coefficient, input_number)

        visit(self.to_dict())
        return tuple(sources)

    def with_facts(self, **facts: Any) -> EffectiveReductionRecipe:
        """Complete a captured operation with its actual output facts."""
        return self.from_dict(self.to_dict() | facts)

    def scalar_units(self, *, units: str, factor: float) -> EffectiveReductionRecipe:
        """Describe a scalar conversion with its captured native recipe as the input."""
        return self.from_dict(
            {
                "version": 1,
                "operation": "convert_scalar_units",
                "units": units,
                "factor": factor,
                "input": self.to_dict(),
            }
        )

    def scan_selection(self, indices: list[int], *, derivative: bool) -> EffectiveReductionRecipe:
        """Keep selected point recipes with scalar scans and source context with derivatives."""
        selected = self.curve_operation("select_scan_points", indices=indices)
        if derivative:
            return selected
        value = self.to_dict()
        positions = {old: new for new, old in enumerate(indices)}
        return selected.with_facts(
            correlated_pairs=[
                (positions[a], positions[b])
                for a, b in value.get("correlated_pairs", [])
                if a in positions and b in positions
            ],
            point_recipes=[value["point_recipes"][i] for i in indices],
            coordinates=[value["coordinates"][i] for i in indices],
        )

    def curve_operation(self, operation: str, **parameters: Any) -> EffectiveReductionRecipe:
        """Append an ordered curve selection or transform without implying raw replay."""
        value = self.to_dict()
        value.setdefault("curve_operations", []).append({"operation": operation, **parameters})
        return self.from_dict(value)


def source_reference(metadata: dict | None) -> dict:
    """Retain available acquisition references without manufacturing content identity."""
    metadata = metadata or {}
    return {
        "run_number": metadata.get("source_run_number", metadata.get("run_number")),
        "source_file": metadata.get("source_file", ""),
        "data_period_number": metadata.get("data_period_number", metadata.get("period_number")),
        "dae_period_number": metadata.get("dae_period_number"),
        "content_identity": metadata.get("content_identity"),
    }


def preparation_recipe(
    observation: ConstituentCounts,
    grouping: dict[str, Any],
    layout: ForwardBackwardLayout,
    metadata: dict[str, Any] | None,
    reference: BackgroundReference | None = None,
) -> EffectiveReductionRecipe:
    """Capture the actual resolved constituent/layout owners, without resolving again."""
    from asymmetry.core.io.period_counts import run_arithmetic_policy
    from asymmetry.core.project.json_values import encode_non_finite
    from asymmetry.core.transform.constituent_counts import CONSTITUENT_PREPARATION_POLICY

    offsets = dict(layout.alignment.detector_offsets)
    timing_sources = dict(layout.alignment.timing_sources)
    offset = 0
    members = []
    for member in observation.members:
        size = len(member.histograms)
        selected = [
            i - offset
            for indices in observation.pair
            for i in indices
            if offset <= i < offset + size
        ]
        labels = (member.metadata or {}).get("histogram_labels") or member.grouping.get(
            "histogram_labels", []
        )
        members.append(
            {
                "source": source_reference(member.metadata),
                "input_number": member.input_number,
                "period_number": member.period_number,
                "coefficient": member.coefficient,
                "acquired_arithmetic": member.arithmetic,
                "raw_period_histograms_available": (member.metadata or {}).get(
                    "raw_period_histograms_available"
                ),
                "detectors": [
                    {
                        "index": i,
                        "bin_width_us": member.histograms[i].bin_width,
                        "n_bins": member.histograms[i].n_bins,
                        "t0_bin": layout.common_t0_bin - offsets[offset + i],
                        "t0_source": timing_sources.get(offset + i),
                        "file_t0_bin": member.histograms[i].t0_bin,
                        "file_t0_time_us": member.histograms[i].t0_time_us,
                        "file_good_bins": [
                            member.histograms[i].good_bin_start,
                            member.histograms[i].good_bin_end,
                        ],
                        "declared_postpileup": "postpileup"
                        in str(labels[i]).casefold().replace(" ", "")
                        if i < len(labels)
                        else None,
                        "variance": "poisson"
                        if member.histograms[i].count_variance is None
                        else "supplied",
                    }
                    for i in selected
                ],
            }
        )
        offset += size
    calibration = calibration_attribution(grouping)
    return EffectiveReductionRecipe.from_dict(
        {
            "version": 1,
            "operation": "prepared_counts",
            "source": source_reference(metadata),
            "constituent_policy": CONSTITUENT_PREPARATION_POLICY,
            **run_arithmetic_policy(metadata or {}),
            "constituents": members,
            "groups": {"forward": list(observation.pair[0]), "backward": list(observation.pair[1])},
            "excluded_detectors": sorted(grouping.get("excluded_detectors", [])),
            "alignment": layout.alignment.to_dict(),
            "good_window": asdict(layout.window),
            "bin_width_us": layout.bin_width_us,
            "t0_time_us": layout.t0_time_us,
            "calibration": encode_non_finite(calibration),
            "reference": reference.provenance() if reference is not None else None,
        }
    )


def calibration_attribution(grouping: dict[str, Any]) -> dict[str, Any]:
    """Retain the authorities that actually supply the selected pair's coefficients."""
    from asymmetry.core.transform.grouping import selected_projection_labels
    from asymmetry.core.transform.projections import _AXIS_ALPHA_KEYS

    labels = selected_projection_labels(grouping)
    projections = {p["label"]: p for p in grouping.get("projections", ())}
    projection_policies = grouping.get("projection_alpha_policies", {})
    applied, base_alpha = [], not labels
    for label in labels:
        if label in projection_policies:
            policy = projection_policies[label]
        else:
            key = _AXIS_ALPHA_KEYS.get(label, "alpha")
            target = grouping if label in _AXIS_ALPHA_KEYS else projections.get(label, {})
            if key not in target:
                base_alpha = True
                continue
            policy = {
                name: target[key + suffix]
                for name, suffix in (
                    ("value", ""),
                    ("error", "_error"),
                    ("method", "_method"),
                    ("source_run", "_reference_run"),
                )
                if key + suffix in target
            }
        applied.append({"projection_label": label, "policy": policy})
    coefficients = (["alpha"] if base_alpha else []) + ([] if labels else ["beta"])
    policies = {
        name: grouping[f"{name}_policy"] for name in coefficients if f"{name}_policy" in grouping
    }
    record_ids = {
        p["calibration_id"]
        for p in (*policies.values(), *(v["policy"] for v in applied))
        if "calibration_id" in p
    }
    legacy = {
        name + suffix: grouping[name + suffix]
        for name in coefficients
        for suffix in ("_method", "_error", "_reference_run")
        if name + suffix in grouping
    }
    return legacy | {
        "policies": policies,
        "projection_policies": sorted(
            applied, key=lambda v: json.dumps(numerical_recipe(v["policy"]), sort_keys=True)
        ),
        "records": sorted(
            (
                r
                for r in grouping.get("calibration_records", ())
                if r["calibration_id"] in record_ids
            ),
            key=lambda r: r["calibration_id"],
        ),
    }


def field_scan_recipe(
    points: Sequence[FieldScanPoint], order_key: str, method: str
) -> EffectiveReductionRecipe:
    """Capture an ordered scalar series, retaining explicit unknown point provenance."""
    return EffectiveReductionRecipe.from_dict(
        {
            "version": 1,
            "operation": "field_scan",
            "observable": "asymmetry",
            "units": "fraction",
            "order_key": order_key,
            "method": method,
            "coordinates": [p.x for p in points],
            "point_recipes": [
                p.effective_recipe.to_dict() if p.effective_recipe is not None else None
                for p in points
            ],
        }
    )


def ratio_recipe(
    counts: ForwardBackwardCounts,
    binning: BinningSpec,
    alpha: float,
    beta: float,
    time_range: tuple[float | None, float | None] | None,
    result: BinnedRatio,
    *,
    units: str,
) -> EffectiveReductionRecipe | None:
    """Complete a resolved preparation with the actual ratio grid and conditional-noise contract."""
    from asymmetry.core.data.ratio_validity import RATIO_OBSERVATION_POLICY
    from asymmetry.core.transform.rebin import BINNING_POLICY

    time, _, _, _, noise, edges = result
    if counts.effective_recipe is None:
        return None
    return counts.effective_recipe.with_facts(
        operation="fb_asymmetry",
        observable="asymmetry",
        units=units,
        time_units="microsecond",
        alpha=alpha,
        beta=beta,
        corrections=counts.corrections.to_dict(),
        binning=asdict(binning),
        binning_policy=BINNING_POLICY,
        aligned_bin_edges=edges.tolist(),
        time_range=time_range,
        n_points=int(time.size),
        uncertainty={
            "background_estimator_counting": "propagated_when_estimated",
            "calibration": "conditional",
            "fixed_correction_parameters": "conditional",
            "counting_noise": "retained" if noise is not None else "marginal",
            "ratio_policy": RATIO_OBSERVATION_POLICY,
        },
    )
