"""Explicit instrument preset preparation for scripted histogram analysis."""

from dataclasses import replace
from functools import partial

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.instrument import (
    derive_projection_pairs,
    get_instrument_layout,
    preset_grouping_for_run,
)
from asymmetry.core.instruments.readouts import readout_presets
from asymmetry.core.instruments.resolution import (
    InstrumentConflictError,
    layout_key_for_selection,
    resolve_instrument,
)
from asymmetry.core.io import background_reference_for_grouping
from asymmetry.core.project.profiles import (
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.transform.preparation import (
    histogram_observation,
    prepare_histogram_asymmetry,
)


def apply_grouping_preset(
    dataset: MuonDataset, preset_name: str, *, readout_set: str | None = None
) -> MuonDataset:
    """Return a newly reduced dataset with an explicitly selected instrument preset.

    Recognition and histogram-count variant resolution select the layout; the
    caller selects the exact preset name. ``readout_set`` selects a declared
    acquired-channel alternative; omission retains an applied choice or resolves
    the first compatible set for this explicit preset request. All preset groups must have acquired,
    non-excluded members. Existing exclusions, timing, binning, and correction
    policies are retained. Preset projections replace the previous projections
    and their manually authored balances; scalar balance policies are retained.

    Active captured calibration policies cannot survive a change to membership,
    the selected pair, or projection pairs: such requests raise ``ValueError``.
    Apply the preset to the original uncalibrated dataset, then calibrate it.
    Identical grouping retains the canonical coefficient policies and records.

    The source dataset is never modified. The returned run shares the acquired
    histograms, while grouping and curve provenance describe the new reduction.
    Cached period curves are discarded; all held periods must have raw histograms.
    Curve-only or rotating-frame inputs, unknown instruments/presets, incompatible
    labels, and unavailable channels raise explicit errors. No loading defaults are changed.
    """
    run = dataset.run
    if run is None or not run.histograms:
        raise ValueError("Applying a grouping preset requires acquired histograms.")
    if "rotating_frame" in dataset.metadata:
        raise ValueError("Apply a grouping preset to the laboratory-frame source dataset.")
    if "period_reduced" in run.grouping and "period_histograms" not in run.grouping:
        raise ValueError("Applying a grouping preset requires raw histograms for all held periods.")
    context = resolve_instrument(run)
    if context.requires_choice:
        raise InstrumentConflictError(context)
    selected = context.selection
    if selected is None:
        raise ValueError("Cannot apply a grouping preset to an unrecognised instrument.")
    key = layout_key_for_selection(selected, len(run.histograms))
    if key is None:
        raise ValueError("The selected instrument has no compatible acquired channel map.")
    layout = get_instrument_layout(key)
    if preset_name not in layout.presets:
        raise ValueError(f"Unknown preset {preset_name!r}")
    chosen_set = readout_set if readout_set is not None else run.grouping.get("readout_set")
    if layout.readout_sets:
        if chosen_set is None:
            chosen_set = next(
                (
                    r.key
                    for r in layout.readout_sets
                    if preset_name in readout_presets(layout, r.key)
                ),
                None,
            )
        if chosen_set is None or preset_name not in readout_presets(layout, chosen_set):
            raise ValueError("Preset is unavailable for the selected readout set")
    elif chosen_set is not None:
        raise ValueError("This acquisition has no declared readout-set choice")
    groups = preset_grouping_for_run(
        layout,
        preset_name,
        n_histograms=len(run.histograms),
        labels=run.grouping.get("histogram_labels", run.metadata.get("histogram_labels")),
    )
    preset = layout.presets[preset_name]
    fingerprint = profile_fingerprint_for_run(run)
    previous = profile_from_payload(run.grouping, "preset", fingerprint)
    projections = [p.to_payload() for p in preset.projections]
    profile = replace(
        previous,
        groups=groups,
        group_names={gid: preset.groups[gid].name for gid in groups},
        included_groups={gid: True for gid in groups},
        forward_group=preset.forward_group,
        backward_group=preset.backward_group,
        projections=projections,
        extra={
            "grouping_preset": preset_name,
            **({"readout_set": chosen_set} if chosen_set is not None else {}),
        },
    )
    changed = (
        previous.groups != profile.groups
        or (previous.forward_group, previous.backward_group)
        != (profile.forward_group, profile.backward_group)
        or derive_projection_pairs(previous.groups, previous.group_names, previous.projections)
        != derive_projection_pairs(profile.groups, profile.group_names, profile.projections)
    )
    policies = (previous.alpha_policy, previous.beta_policy)
    if changed and (
        previous.projection_alpha_policies or any(p.mode == "calibrated" for p in policies)
    ):
        raise ValueError(
            "Preset changes the grouping of an active captured calibration; "
            "apply it to the uncalibrated source and recalibrate."
        )
    observation = dataset.histogram_observation
    reference = observation.reference if observation is not None else None
    grouping = resolve_effective_grouping(
        profile,
        run,
        reference_resolver=partial(background_reference_for_grouping, reference),
    )
    grouping.pop("period_reduced", None)
    for gid in groups:
        effective_group_indices(grouping, gid, n_histograms=len(run.histograms))
    prepared_run = replace(
        run,
        grouping=grouping,
        metadata={k: v for k, v in run.metadata.items() if k != "projection"},
    )
    facility = run.metadata.get("facility")
    reduced = prepare_histogram_asymmetry(
        prepared_run, merge=1, facility=facility, reference=reference
    )
    return MuonDataset(
        time=reduced.time,
        asymmetry=reduced.asymmetry,
        error=reduced.error,
        metadata=reduced.metadata_with_alignment(
            {k: v for k, v in dataset.metadata.items() if k != "projection"}
        ),
        run=prepared_run,
        histogram_observation=histogram_observation(
            prepared_run, facility=facility, reference=reference
        ),
        ratio_validity=reduced.ratio_validity,
        counting_noise=reduced.counting_noise,
        effective_recipe=reduced.effective_recipe,
        computation_revision=reduced.computation_revision,
    )


__all__ = ["apply_grouping_preset"]
