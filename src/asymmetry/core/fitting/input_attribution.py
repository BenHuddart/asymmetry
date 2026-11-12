"""Durable attribution of fitted observations, separate from model structure."""

import json
from hashlib import sha256

import numpy as np

from asymmetry.core.data.reduction_recipe import numerical_recipe
from asymmetry.core.fitting.likelihood import GaussianObservations, dataset_observations


def fit_input_attribution(observations, dataset=None, *, window=None, error_oversampling=1.0):
    """Capture sampled values, errors and omission support without archiving arrays.

    Labels and model-screening metadata do not identify observations. The recipe
    identifies interpretation only; the digest also includes actual sampled bytes.
    This record neither verifies external files nor restores counting covariance.
    """
    gaussian = isinstance(observations, GaussianObservations)
    arrays = (
        (observations.time, observations.values, observations.sigma)
        if gaussian
        else (
            observations.time,
            observations.counts,
        )
    )
    recipe = dataset.effective_recipe if dataset is not None else None
    interpretation = numerical_recipe(recipe.to_dict()) if recipe is not None else None
    if interpretation is not None:
        # Exact sampled coordinates below own centre selection; keep value transforms.
        operations = interpretation.pop("curve_operations", [])
        operations = [op for op in operations if op["operation"] != "select_time_centres"]
        if operations:
            interpretation["curve_operations"] = operations
        if interpretation.get("operation") == "rotate_transverse":
            from asymmetry.core.transform.rotating_frame import FRAME_FIELDS, RotatingFrame

            frame = RotatingFrame.from_dict(interpretation["frame"])
            # Only the selected baseline and frame values interpret these observations.
            interpretation["frame"] = {
                key: interpretation["frame"][key] for key in FRAME_FIELDS
            } | {"combined_baseline": frame.combined_baseline(interpretation["weights"])}
    context = {
        "version": 1,
        "error_oversampling": error_oversampling,
        "objective": "gaussian" if gaussian else "poisson",
        "report": observations.report.to_dict()
        if gaussian and observations.report is not None
        else None,
        "interpretation": interpretation,
    }
    digest = sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode())
    for array in arrays:
        values = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    support = dataset.time if dataset is not None else observations.time
    inside = np.ones(support.size, dtype=bool)
    if window is not None:
        if window[0] is not None:
            inside &= support >= window[0]
        if window[1] is not None:
            inside &= support <= window[1]
    support = support[inside]
    digest.update(np.ascontiguousarray(support, dtype="<f8").tobytes())
    if dataset is not None and dataset.ratio_validity is not None:
        digest.update(dataset.ratio_validity.reasons[inside].tobytes())
    return {
        "identity": digest.hexdigest(),
        "error_oversampling": error_oversampling,
        "n": int(observations.time.size),
        "min": float(support.min()),
        "max": float(support.max()),
        "effective_recipe": recipe.to_dict() if recipe is not None else None,
    }


def saved_fit_matches(slot, dataset):
    """Verify current observations before drawing historical residuals."""
    recorded = (slot.result or {}).get("input_attribution")
    if recorded is None or (slot.result or {}).get("objective") != "gaussian":
        return False
    from asymmetry.core.fitting.likelihood import gaussian_selection

    window = {"min": recorded["min"], "max": recorded["max"]}
    selection = gaussian_selection(dataset, window["min"], window["max"])
    if (
        not selection.used
        or not all(
            np.all(np.isfinite(a[selection.mask]))
            for a in (dataset.time, dataset.asymmetry, dataset.error)
        )
        or np.any(dataset.error[selection.mask] <= 0)
    ):
        return False
    current = fit_input_attribution(
        dataset_observations(dataset, window["min"], window["max"]),
        dataset,
        window=(window["min"], window["max"]),
        error_oversampling=max(float(dataset.metadata.get("fourier_padding", 1) or 1), 1.0),
    )
    return current["identity"] == recorded["identity"]
