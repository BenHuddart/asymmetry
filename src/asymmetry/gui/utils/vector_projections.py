"""Adapt applied vector requests to the authoritative core projection cohort."""

from asymmetry.core.instrument import derive_projection_pairs
from asymmetry.core.transform.background import resolve_background_mode, resolve_facility
from asymmetry.core.transform.projections import reduce_selected_projections
from asymmetry.core.transform.reduce import correction_flags_from_grouping


def vector_axis_datasets(datasets, labels, memberships, reference_resolver):
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
        flags = correction_flags_from_grouping(grouping)
        reference = (
            reference_resolver(grouping)
            if flags.use_background and resolve_background_mode(grouping) == "reference_run"
            else None
        )
        cohort = reduce_selected_projections(
            run,
            facility=resolve_facility(metadata=run.metadata, grouping=grouping),
            reference=reference,
        )
        for label in selected:
            curve = cohort[label]
            curve.metadata.update(
                {key: value for key, value in dataset.metadata.items() if key not in curve.metadata}
            )
            axis_map[label].append(curve)
    return axis_map
