"""Durable attribution of fitted observations, separate from model structure."""

import json
from hashlib import sha256

import numpy as np

from asymmetry.core import computation
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
        "computation_revision": dataset.computation_revision if dataset is not None else None,
        "error_oversampling": error_oversampling,
        "n": int(observations.time.size),
        "min": float(support.min()),
        "max": float(support.max()),
        "effective_recipe": recipe.to_dict() if recipe is not None else None,
    }


def saved_fit_matches(slot, dataset):
    """Verify current observations before drawing historical residuals."""
    recorded = (slot.result or {}).get("input_attribution")
    if (
        recorded is None
        or not computation.is_current((slot.result or {}).get("computation_revision"))
        or (slot.result or {}).get("objective") != "gaussian"
    ):
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
    return current["identity"] == recorded["identity"] and same_source_context(current, recorded)


def same_source_context(first, second):
    """Keep equal sampled observations from overwriting a different source history.

    This comparison does not redefine observation identity. Unknown legacy sources
    stay unknown and do not prevent an explicitly requested new fit.
    """
    return source_context_key(first) == source_context_key(second)


def source_context_key(attribution):
    """Hashable ordered source context, separate from the sampled-observation digest."""
    from asymmetry.core.data.reduction_recipe import recipe_source_declarations

    recipe = (attribution or {}).get("effective_recipe")
    return _source_signature(recipe_source_declarations(recipe) if recipe else ())


def _source_signature(sources):
    """Ordered source content and selectors, independent of local locators."""
    return tuple(
        (
            s.role,
            s.expected,
            s.run_number,
            s.period_number,
            s.coefficient,
            s.input_number,
            s.dae_period_number,
        )
        for s in sources
    )


def source_context_status(attribution, dataset):
    """Describe source currentness from captured in-memory facts without acquisition I/O."""
    from asymmetry.core.data.reduction_recipe import recipe_source_declarations
    from asymmetry.core.io.periods import source_run_of

    recipe = (attribution or {}).get("effective_recipe")
    sources = (
        tuple(
            s
            for s in recipe_source_declarations(recipe)
            if s.role != "calibration" and s.coefficient != 0
        )
        if recipe
        else ()
    )
    if (
        dataset is not None
        and not (
            dataset.run.source_file
            if dataset.run is not None
            else dataset.metadata.get("source_file")
        )
        and not any(s.source_file or s.expected is not None for s in sources)
    ):
        return "in_memory"
    if not sources or any(s.expected is None for s in sources):
        return "unverified"
    if dataset is None:
        return "unavailable"
    if dataset.effective_recipe is not None:
        current = tuple(
            s
            for s in dataset.effective_recipe.sources()
            if s.role != "calibration" and s.coefficient != 0
        )
        # Locators do not identify the acquired counts; ordered selections do.
        if current and all(s.expected is not None for s in current):
            return (
                "matched" if _source_signature(sources) == _source_signature(current) else "changed"
            )
    run = dataset.run
    if (
        run is None
        or run.source_binding.observed is None
        or any(
            s.role not in {"sample", "constituents"} or s.run_number != source_run_of(run)
            for s in sources
        )
    ):
        return "unverified"
    return (
        "matched" if all(s.expected == run.source_binding.observed for s in sources) else "changed"
    )
