"""Relaxed-to-discrete proposal extraction."""

from __future__ import annotations

from asymmetry.core.fitting.global_search.adapters import (
    subset_additive_structure,
    with_parameter_roles,
)
from asymmetry.core.fitting.global_search.heuristics import (
    allows_rate_first_localization,
    is_background_parameter,
    localisation_threshold_scale,
)
from asymmetry.core.fitting.global_search.types import (
    DiscreteCandidate,
    ParameterTieMode,
    RelaxedFitResult,
)


def extract_discrete_candidates(
    result: RelaxedFitResult,
    structure,
    *,
    deviation_threshold: float = 0.05,
    ambiguity_band: float = 0.02,
    activity_threshold: float = 0.02,
    max_alternates: int = 2,
) -> tuple[DiscreteCandidate, ...]:
    """Convert a relaxed solution into one or more discrete candidate structures."""
    spec_map = structure.parameter_spec_map
    shared_names: list[str] = []
    local_names: list[str] = []
    ambiguous_names: list[str] = []
    notes: list[str] = []

    for tie in structure.parameter_ties:
        if tie.parameter_name in structure.fixed_param_names:
            continue
        if tie.mode == ParameterTieMode.LOCAL:
            local_names.append(tie.parameter_name)
            continue
        if tie.parameter_name in result.frozen_shared_names:
            shared_names.append(tie.parameter_name)
            notes.append(f"Kept {tie.parameter_name} shared by active-set freezing.")
            continue
        if is_background_parameter(tie.parameter_name):
            shared_names.append(tie.parameter_name)
            notes.append(f"Kept {tie.parameter_name} shared by staged background prior.")
            continue
        if not allows_rate_first_localization(tie.parameter_name):
            shared_names.append(tie.parameter_name)
            notes.append(f"Kept {tie.parameter_name} shared during rate-first staged search.")
            continue
        spec = spec_map[tie.parameter_name]
        scale = max(spec.bounds_width, 1e-6)
        deviations = [
            abs(run_values.get(tie.parameter_name, 0.0)) / scale
            for run_values in result.deviations_by_run.values()
        ]
        magnitude = float(max(deviations)) if deviations else 0.0
        scaled_threshold = deviation_threshold * localisation_threshold_scale(tie.parameter_name)
        scaled_band = ambiguity_band * localisation_threshold_scale(tie.parameter_name)
        if magnitude <= scaled_threshold:
            shared_names.append(tie.parameter_name)
        else:
            local_names.append(tie.parameter_name)
        if abs(magnitude - scaled_threshold) <= scaled_band:
            ambiguous_names.append(tie.parameter_name)

    candidate_structure = with_parameter_roles(
        structure,
        shared_names=tuple(sorted(shared_names)),
        local_names=tuple(sorted(local_names)),
    )

    inactive_ids = tuple(
        component_id
        for component_id, weight in sorted(result.activity_weights.items())
        if weight <= activity_threshold
    )
    if inactive_ids:
        remaining = tuple(
            component.instance_id
            for component in candidate_structure.components
            if component.instance_id not in set(inactive_ids)
        )
        pruned = subset_additive_structure(candidate_structure, active_component_ids=remaining)
        if pruned is not None:
            candidate_structure = pruned
            notes.append("Pruned inactive additive components: " + ", ".join(inactive_ids))

    primary = DiscreteCandidate(
        structure=candidate_structure,
        source_relaxed_signature=result.signature(),
        initial_params_by_run=result.seed_params_by_run,
        extraction_notes=tuple(notes),
        approx_score=float(result.objective_value) + (0.1 * len(local_names)),
        relaxed_local_names=tuple(sorted(local_names)),
        ambiguous_param_names=tuple(sorted(ambiguous_names)),
    )

    if not ambiguous_names:
        return (primary,)

    candidates: list[DiscreteCandidate] = [primary]
    for index, name in enumerate(ambiguous_names[: max(1, max_alternates - 1)], start=1):
        flipped_shared = set(shared_names)
        flipped_local = set(local_names)
        if name in flipped_shared:
            flipped_shared.remove(name)
            flipped_local.add(name)
        else:
            flipped_local.discard(name)
            flipped_shared.add(name)
        alternate_structure = with_parameter_roles(
            structure,
            shared_names=tuple(sorted(flipped_shared)),
            local_names=tuple(sorted(flipped_local)),
        )
        candidates.append(
            DiscreteCandidate(
                structure=alternate_structure,
                source_relaxed_signature=result.signature(),
                initial_params_by_run=result.seed_params_by_run,
                extraction_notes=primary.extraction_notes
                + (f"Ambiguous role threshold for {name}.",),
                approx_score=float(result.objective_value) + (0.35 * index),
                relaxed_local_names=primary.relaxed_local_names,
                ambiguous_param_names=primary.ambiguous_param_names,
            )
        )
    return tuple(candidates)
