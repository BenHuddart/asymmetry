"""Adapt applied vector requests to the authoritative core projection cohort."""

from dataclasses import replace
from functools import partial

from asymmetry.core import computation
from asymmetry.core.instrument import derive_projection_pairs
from asymmetry.core.transform.background import resolve_background_mode, resolve_facility
from asymmetry.core.transform.preparation import observation_histograms
from asymmetry.core.transform.projections import (
    reduce_selected_projections,
    rotate_selected_projections,
)
from asymmetry.core.transform.reduce import (
    asymmetry_reduction_metadata,
    correction_flags_from_grouping,
)
from asymmetry.gui.utils.reduction_cache import prepare_histogram_operation
from asymmetry.gui.widgets.rotating_frame_bar import FRAME_METADATA_KEY


def vector_axis_datasets(datasets, labels, memberships, reference_resolver, parent):
    """Build each run's declared projections together, retaining shared noise."""
    axis_labels = list(labels) if labels else list(memberships)
    axis_map = {label: [] for label in axis_labels}
    for index, dataset in enumerate(datasets):
        selected = [label for label in axis_labels if index in memberships.get(label, ())]
        if not selected:
            continue
        run = dataset.run
        grouping = run.grouping
        pairs = derive_projection_pairs(
            grouping["groups"], grouping.get("group_names"), grouping.get("projections")
        )
        if len(pairs) < 2:
            axis_map[selected[0]].append(dataset)
            continue
        cohort = _prepare_vector(dataset, reference_resolver, parent, reduce_selected_projections)
        for label in selected:
            curve = cohort[label]
            axis_map[label].append(curve)
    return axis_map


def fit_projection_dataset(dataset, projection, cache, reference_resolver, parent):
    """Prepare a lab fit projection from the run's captured histogram observation."""
    observation = dataset.histogram_observation
    if observation is None:
        return _prepare_vector(dataset, reference_resolver, parent, reduce_selected_projections)[
            projection
        ]

    def prepare():
        snapshot = replace(dataset, run=observation.source)
        return _prepare_vector(
            snapshot, lambda _: observation.reference, parent, reduce_selected_projections
        )[projection]

    key = (observation.identity, projection, computation.COMPUTATION_REVISION)
    return cache.get_or_compute(dataset.run, "lab_fit_projection", key, prepare, _projection_bytes)


def bind_projection_conditions(curve, live):
    """Hand out the prepared projection with the live run's condition metadata."""
    return replace(
        curve,
        run=replace(curve.run, metadata=dict(live.run.metadata)),
        metadata=_projection_metadata(live, curve),
    )


def _projection_bytes(curve):
    """Count retained histogram buffers and sparse noise responses once."""
    observation = curve.histogram_observation
    histograms = observation_histograms(curve.run) + observation_histograms(observation.source)
    if observation.reference is not None:
        histograms.extend(observation.reference.histograms)
    buffers = [curve.time, curve.asymmetry, curve.error, curve.ratio_validity.reasons]
    buffers.extend(array for hist in histograms for array in (hist.counts, hist.count_variance))
    buffers.extend(
        hist.counting_observation.variance
        for hist in histograms
        if hist.counting_observation is not None
    )
    if curve.counting_noise is not None:
        for term in curve.counting_noise.terms:
            buffers.extend(
                (
                    term.observation.variance,
                    term.direct.data,
                    term.direct.indices,
                    term.direct.indptr,
                )
            )
            buffers.extend(array for estimator in term.estimators for array in estimator)
    return sum(array.nbytes for array in {id(a): a for a in buffers if a is not None}.values())


def _projection_metadata(dataset, curve):
    """Keep the projection, numerical metadata and selected window with the curve."""
    labels = {
        key: curve.metadata[key]
        for key in ("projection", "period_index", "fit_range")
        if key in curve.metadata
    }
    return (
        asymmetry_reduction_metadata(dataset.run.metadata | dataset.metadata, curve.metadata)
        | labels
    )


def _prepare_vector(dataset, reference_resolver, parent, operation):
    run = dataset.run
    grouping = run.grouping
    flags = correction_flags_from_grouping(grouping)
    reference = (
        reference_resolver(grouping)
        if flags.use_background and resolve_background_mode(grouping) == "reference_run"
        else None
    )
    result = prepare_histogram_operation(
        run,
        reference=reference,
        parent=parent,
        prepare=partial(
            operation,
            run,
            facility=resolve_facility(
                metadata=run.metadata, grouping=grouping, instrument_choice=run.instrument_choice
            ),
            reference=reference,
        ),
        label="Preparing vector projections…",
        context="Vector projection",
    )
    for curve in result.values():
        curve.metadata = _projection_metadata(dataset, curve)
    return result


def rotated_vector_datasets(dataset, snapshot, reference_resolver, parent):
    """Obtain rotated display and fit observations from the same worker operation."""
    result = _prepare_vector(
        dataset,
        reference_resolver,
        parent,
        partial(rotate_selected_projections, frame=snapshot.frame, weights=snapshot.weights),
    )
    for curve in result.values():
        curve.metadata[FRAME_METADATA_KEY] = snapshot
    return result
