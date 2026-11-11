"""Adapt applied vector requests to the authoritative core projection cohort."""

from functools import partial

from asymmetry.core.instrument import derive_projection_pairs
from asymmetry.core.transform.background import resolve_background_mode, resolve_facility
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
            facility=resolve_facility(metadata=run.metadata, grouping=grouping),
            reference=reference,
        ),
        label="Preparing vector projections…",
        context="Vector projection",
    )
    for curve in result.values():
        labels = {key: curve.metadata[key] for key in ("projection", "period_index")}
        curve.metadata = (
            asymmetry_reduction_metadata(run.metadata | dataset.metadata, curve.metadata) | labels
        )
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
