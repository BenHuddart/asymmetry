"""Neighborhood generation for staged discrete refinement."""

from __future__ import annotations

from asymmetry.core.fitting.global_search.adapters import (
    candidate_with_structure,
    subset_additive_structure,
    with_parameter_roles,
)
from asymmetry.core.fitting.global_search.heuristics import allows_rate_first_localization
from asymmetry.core.fitting.global_search.types import (
    DiscreteCandidate,
    ModelStructure,
    SearchMove,
    SearchMoveType,
)


def generate_search_moves(
    candidate: DiscreteCandidate,
    *,
    full_structure: ModelStructure | None = None,
    allow_backward_moves: bool = False,
) -> tuple[SearchMove, ...]:
    """Return local refinement moves for one discrete candidate."""
    structure = candidate.structure
    shared_names, local_names, _fixed_names = structure.role_names()
    moves: list[SearchMove] = []
    ambiguous = set(candidate.ambiguous_param_names)

    for name in shared_names:
        if name in ambiguous and allows_rate_first_localization(name):
            moves.append(SearchMove(SearchMoveType.SPLIT_TO_LOCAL, name))
    for name in local_names:
        if name in ambiguous or allow_backward_moves:
            moves.append(SearchMove(SearchMoveType.MERGE_TO_SHARED, name))

    if structure.is_additive_mixture:
        active_components = [component for component in structure.components if component.active]
        if len(active_components) > 2:
            for component in active_components[:-1]:
                moves.append(SearchMove(SearchMoveType.REMOVE_COMPONENT, component.instance_id))
        if full_structure is not None:
            active_ids = {component.instance_id for component in active_components}
            for component in full_structure.components:
                if component.instance_id not in active_ids:
                    moves.append(SearchMove(SearchMoveType.ADD_COMPONENT, component.instance_id))

    return tuple(moves)


def apply_search_move(
    candidate: DiscreteCandidate,
    move: SearchMove,
    *,
    full_structure: ModelStructure | None = None,
) -> DiscreteCandidate | None:
    """Apply one refinement move to a candidate."""
    structure = candidate.structure
    shared_names, local_names, _fixed_names = structure.role_names()
    shared = set(shared_names)
    local = set(local_names)

    if move.move_type == SearchMoveType.SPLIT_TO_LOCAL:
        shared.discard(move.target)
        local.add(move.target)
        return candidate_with_structure(
            candidate,
            with_parameter_roles(
                structure,
                shared_names=tuple(sorted(shared)),
                local_names=tuple(sorted(local)),
            ),
        )
    if move.move_type == SearchMoveType.MERGE_TO_SHARED:
        local.discard(move.target)
        shared.add(move.target)
        return candidate_with_structure(
            candidate,
            with_parameter_roles(
                structure,
                shared_names=tuple(sorted(shared)),
                local_names=tuple(sorted(local)),
            ),
        )
    if move.move_type == SearchMoveType.REMOVE_COMPONENT and structure.is_additive_mixture:
        active_ids = tuple(
            component.instance_id
            for component in structure.components
            if component.instance_id != move.target and component.active
        )
        pruned = subset_additive_structure(structure, active_component_ids=active_ids)
        return candidate_with_structure(candidate, pruned) if pruned is not None else None
    if (
        move.move_type == SearchMoveType.ADD_COMPONENT
        and full_structure is not None
        and full_structure.is_additive_mixture
    ):
        active_ids = {
            component.instance_id for component in structure.components if component.active
        }
        active_ids.add(move.target)
        ordered = tuple(
            component.instance_id
            for component in full_structure.components
            if component.instance_id in active_ids
        )
        expanded = subset_additive_structure(full_structure, active_component_ids=ordered)
        return candidate_with_structure(candidate, expanded) if expanded is not None else None
    return None
