"""Acquired readout choices and shared-event eligibility, independent of count values."""

from collections.abc import Sequence
from dataclasses import replace

from asymmetry.core.instruments.catalog import profile_by_key
from asymmetry.core.instruments.layout import InstrumentLayout, ReadoutSetDefinition
from asymmetry.core.instruments.resolution import (
    InstrumentConflictError,
    InstrumentContext,
    UnmappedAcquisition,
    assess_instrument,
    resolve_instrument,
)
from asymmetry.core.instruments.schema import Acquisition


def readout_acquisition(
    n_histograms: int, grouping: dict, metadata: dict | None = None
) -> Acquisition | None:
    """Use captured canonical authority when present, including explicit unknown."""
    metadata = {} if metadata is None else metadata
    if "instrument_context" in metadata:
        context = metadata["instrument_context"]
        if context is not None and not isinstance(context, InstrumentContext):
            raise ValueError("Readout authority requires a typed canonical instrument context")
    else:
        evidence = dict(metadata)
        for key in ("instrument", "facility", "histogram_labels"):
            if key not in evidence and key in grouping:
                evidence[key] = grouping[key]
        context = InstrumentContext(
            assess_instrument(
                n_histograms, metadata=evidence, source_file=metadata.get("source_file")
            ),
            None,
        )
    if context is not None and context.requires_choice:
        raise InstrumentConflictError(context)
    selection = context.selection if context is not None else None
    if selection is None or isinstance(selection.acquisition, UnmappedAcquisition):
        return None
    profile = profile_by_key(selection.profile_key)
    return next(
        config for config in profile.detectors.acquisitions if config.key == selection.acquisition
    )


def require_independent_readouts(
    n_histograms: int, grouping: dict, groups: Sequence[Sequence[int]], metadata: dict | None = None
) -> None:
    """Check only requested effective zero-based selections, never diagnostic groups.

    Known constituent roads share events. Conditional contributors conflict with
    either possible destination, but exclusive destinations do not thereby share
    events with each other. No histogram arithmetic or covariance is inferred.
    """
    selected = [index + 1 for group in groups for index in group]
    config = readout_acquisition(n_histograms, grouping, metadata)
    chosen = grouping.get("readout_set")
    if config is None:
        if chosen is not None:
            raise ValueError("Readout set is unavailable without a resolved acquired channel map")
        return
    ordinals = {channel.key: channel.ordinal for channel in config.channels}
    keys = {key for key, ordinal in ordinals.items() if ordinal in selected}
    if chosen is not None:
        readouts = next((r for r in config.readout_sets if r.key == chosen), None)
        if readouts is None:
            raise ValueError(f"Readout set {chosen!r} is unavailable for this acquisition")
        outside = keys - set(readouts.channels)
        if outside:
            raise ValueError(
                f"Channels {', '.join(sorted(outside))} are outside readout set {readouts.label}; edit grouping or choose another set"
            )
    if not config.combined_readouts and not config.conditional_contributions:
        return
    footprints = {key: {key} for key in keys}
    for road in config.combined_readouts:
        if road.channel in footprints:
            footprints[road.channel] = set(road.constituents)
    for first_index, first in enumerate(sorted(keys)):
        for second in sorted(keys)[first_index + 1 :]:
            if footprints[first].intersection(footprints[second]):
                raise ValueError(
                    f"Readouts {first} and {second} share events (known overlap); choose constituent or combined readouts, not both"
                )
    for contribution in config.conditional_contributions:
        destinations = set(contribution.destinations).intersection(keys)
        if contribution.channel in keys and destinations:
            raise ValueError(
                f"Readout {contribution.channel} may contribute to {', '.join(sorted(destinations))}: unresolved {contribution.label} overlap; choose readouts without shared events"
            )


def require_run_readouts(run, groups: Sequence[Sequence[int]]) -> None:
    """Check each actual acquisition before retained periods change ordinal meaning."""
    from asymmetry.core.transform.constituent_counts import (
        active_acquisitions,
        grouping_for_acquisition,
    )
    from asymmetry.core.transform.corrections import reference_observation_metadata

    context = resolve_instrument(run)
    if context.requires_choice:
        raise InstrumentConflictError(context)
    if "period_inputs" not in run.grouping:
        require_independent_readouts(
            len(run.histograms), run.grouping, groups, reference_observation_metadata(run)
        )
        return
    request = run.grouping
    if request.get("period_mode") == "green_minus_red":
        request = dict(request, period_mode="green_plus_red")
    for _, original, _ in active_acquisitions(request):
        context = resolve_instrument(original)
        if context.requires_choice:
            raise InstrumentConflictError(context)
        effective = grouping_for_acquisition(original, run.grouping)
        excluded = set(effective.get("excluded_detectors", ()))
        selected = tuple(tuple(i for i in group if i + 1 not in excluded) for group in groups)
        require_independent_readouts(
            len(original.histograms), effective, selected, reference_observation_metadata(original)
        )


def readout_set_definition(layout: InstrumentLayout, key: str) -> ReadoutSetDefinition:
    """Resolve an explicit choice without a fallback to another acquired set."""
    selected = next((s for s in layout.readout_sets if s.key == key), None)
    if selected is None:
        raise ValueError(f"Readout set {key!r} is unavailable for this acquisition")
    return selected


def readout_presets(layout: InstrumentLayout, key: str) -> dict:
    """Preset eligibility follows actual declared channel membership."""
    selected = set(readout_set_definition(layout, key).detector_ids)
    return {
        name: preset
        for name, preset in layout.presets.items()
        if {i for group in preset.groups.values() for i in group.detector_ids} <= selected
    }


def layout_for_readout_set(layout: InstrumentLayout, key: str) -> InstrumentLayout:
    """Filter the drawing only; the caller owns the applied analysis choice."""
    selected = set(readout_set_definition(layout, key).detector_ids)
    return replace(
        layout,
        banks=tuple(
            replace(
                bank,
                segments=tuple(
                    s for s in bank.segments if s.detector_id is None or s.detector_id in selected
                ),
            )
            for bank in layout.banks
        ),
        auxiliary=tuple(s for s in layout.auxiliary if s.detector_id in selected),
        presets=readout_presets(layout, key),
    )
