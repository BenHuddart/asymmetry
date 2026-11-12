"""Carry scalar observation facts through the percent-valued scan surface."""

import numpy as np

from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe, field_scan_recipe
from asymmetry.core.transform.corrections import CorrectionOutcomes
from asymmetry.core.transform.integral import FieldScan, FieldScanPoint
from asymmetry.core.transform.scalar_observation import ScalarObservation
from asymmetry.core.transform.units import ASYMMETRY_PERCENT


def display_scan(window, x_key, *, include_excluded=False):
    """Sort the actual requested points with their mathematical fit support."""
    selected = [
        p
        for p in window._alc_scan_points
        if (include_excluded or p["run"] not in window._alc_excluded_runs)
        and (x_key == "run" or p[x_key] is not None)
    ]
    selected.sort(key=lambda p: (p["run"] if x_key == "run" else p[x_key], p["run"]))
    if len(selected) < 2:
        return None
    series = window._project_model.batch(window._alc_scan_series_id)
    positions = {p.get("scan_index"): i for i, p in enumerate(selected)}
    points = [
        FieldScanPoint(
            p["run"],
            p["run"] if x_key == "run" else p[x_key],
            p["value"] * 100.0,
            p["error"] * 100.0,
            effective_recipe=EffectiveReductionRecipe.from_dict(p["effective_recipe"]).scalar_units(
                units="percent", factor=100.0
            )
            if p.get("effective_recipe") is not None
            else None,
        )
        for p in selected
    ]
    scan = FieldScan(
        x=np.array([p["run"] if x_key == "run" else p[x_key] for p in selected], dtype=float),
        value=np.array([p["value"] * 100.0 for p in selected]),
        error=np.array([p["error"] * 100.0 for p in selected]),
        run_numbers=[p["run"] for p in selected],
        order_key=x_key,
        method=window._scan_controller.estimator(),
        x_label=window._ALC_X_LABELS[x_key],
        units=ASYMMETRY_PERCENT,
        correction_outcomes={
            p["run"]: {
                label: CorrectionOutcomes.from_dict(record)
                for label, record in p.get("correction_outcomes", {}).items()
            }
            for p in selected
        },
        observations=tuple(
            ScalarObservation.from_dict(p["observation"])
            if p.get("observation") is not None
            else None
            for p in selected
        ),
        correlated_pairs=tuple(
            (positions[a], positions[b])
            for a, b in series.extra.get("correlated_pairs", ())
            if a in positions and b in positions
        )
        if series is not None
        else (),
    )
    scan.effective_recipe = (
        field_scan_recipe(points, x_key, scan.method)
        .with_facts(
            units="percent",
            correlated_pairs=[list(pair) for pair in scan.correlated_pairs],
            source_assumption=scan.source_assumption,
        )
        .curve_operation(
            "convert_scalar_units", factor=100.0, input_units="fraction", output_units="percent"
        )
    )
    return scan


def observation_status(scan):
    """Visible mathematical status, independent of numeric display errors."""
    observations = [o for o in scan.observations if o is not None]
    omitted = sum(o.reason != 0 for o in observations)
    bins = sum(o.report.omitted for o in observations)
    parts = []
    if omitted:
        parts.append(f"{omitted} without Gaussian uncertainty")
    if bins:
        parts.append(f"{bins} omitted observations")
    if scan.correlated_pairs:
        parts.append("Shared counting observations: independent-error fit unavailable")
    return parts, [o.report.summary() for o in observations if o.report.omitted] + [
        scan.source_assumption
    ]
