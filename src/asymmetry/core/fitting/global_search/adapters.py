"""Adapters between legacy wizard configuration and staged-search structures."""

from __future__ import annotations

from dataclasses import replace

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import CandidateTemplate
from asymmetry.core.fitting.global_search.types import (
    ComponentSpec,
    DiscreteCandidate,
    ModelStructure,
    ParameterSpec,
    ParameterTieMode,
    ParameterTieSpec,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def compile_legacy_structure(
    template: CandidateTemplate,
    *,
    current_parameter_types: dict[str, str] | None = None,
    current_values: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    treat_nonfixed_roles_as_hints: bool = False,
) -> ModelStructure:
    """Compile a legacy template plus role config into a staged-search structure."""
    current_parameter_types = current_parameter_types or {}
    current_values = current_values or {}
    parameter_bounds = parameter_bounds or {}

    components: list[ComponentSpec] = []
    for index, (component, mapping) in enumerate(
        zip(template.model.components, template.model._param_mappings, strict=True),  # noqa: SLF001
        start=1,
    ):
        components.append(
            ComponentSpec(
                instance_id=f"c{index}",
                family=component.name,
                source_index=index - 1,
                source_param_names=tuple(sorted((key, value) for key, value in mapping.items())),
                operator_before=template.model.operators[index - 2] if index > 1 else None,
                allow_activity_relaxation=bool(index < len(template.model.components) and template.model.operators[index - 1:index] == ["+"]),
            )
        )

    parameter_specs: list[ParameterSpec] = []
    parameter_ties: list[ParameterTieSpec] = []
    fixed_param_names: list[str] = []
    for name in template.model.param_names:
        bounds = parameter_bounds.get(name, (-float("inf"), float("inf")))
        role = str(current_parameter_types.get(name, "")).strip().lower()
        if role == "fixed":
            mode = ParameterTieMode.FIXED
            fixed_param_names.append(name)
        elif treat_nonfixed_roles_as_hints:
            mode = ParameterTieMode.RELAXED_SHARED
        elif role == "local":
            mode = ParameterTieMode.LOCAL
        elif role == "global":
            mode = ParameterTieMode.SHARED
        else:
            mode = ParameterTieMode.RELAXED_SHARED
        default_value = float(current_values.get(name, template.model.param_defaults.get(name, 0.0)))
        parameter_specs.append(
            ParameterSpec(
                name=name,
                source_name=name,
                min_value=float(bounds[0]),
                max_value=float(bounds[1]),
                default_value=default_value,
                fixed=mode == ParameterTieMode.FIXED,
            )
        )
        parameter_ties.append(
            ParameterTieSpec(
                parameter_name=name,
                source_name=name,
                mode=mode,
                deviation_penalty=1.0,
            )
        )

    return ModelStructure(
        template_key=template.key,
        title=template.title,
        model=template.model,
        components=tuple(components),
        parameter_specs=tuple(parameter_specs),
        parameter_ties=tuple(parameter_ties),
        fixed_param_names=tuple(sorted(set(fixed_param_names))),
        metadata=(("baseline", "1" if template.is_current_model_baseline else "0"),),
    )


def compile_structure_to_legacy_roles(
    structure: ModelStructure,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return legacy global/local/fixed role lists for one structure."""
    return structure.role_names()


def with_parameter_roles(
    structure: ModelStructure,
    *,
    shared_names: tuple[str, ...],
    local_names: tuple[str, ...],
) -> ModelStructure:
    """Return a structure with updated shared/local role assignments."""
    shared = set(shared_names)
    local = set(local_names)
    updated_ties = []
    for tie in structure.parameter_ties:
        if tie.parameter_name in structure.fixed_param_names or tie.mode == ParameterTieMode.FIXED:
            updated_ties.append(replace(tie, mode=ParameterTieMode.FIXED))
        elif tie.parameter_name in local:
            updated_ties.append(replace(tie, mode=ParameterTieMode.LOCAL))
        elif tie.parameter_name in shared:
            updated_ties.append(replace(tie, mode=ParameterTieMode.SHARED))
        else:
            updated_ties.append(replace(tie, mode=ParameterTieMode.RELAXED_SHARED))
    return replace(structure, parameter_ties=tuple(updated_ties))


def subset_additive_structure(
    structure: ModelStructure,
    *,
    active_component_ids: tuple[str, ...],
) -> ModelStructure | None:
    """Return a pruned additive-only structure using a subset of components."""
    if not structure.is_additive_mixture:
        return None
    component_by_id = {component.instance_id: component for component in structure.components}
    ordered = [component_by_id[component_id] for component_id in active_component_ids if component_id in component_by_id]
    if len(ordered) < 2:
        return None

    component_names = [component.family for component in ordered]
    operators = ["+"] * max(len(component_names) - 1, 0)
    new_model = CompositeModel(component_names, operators=operators)

    old_tie_by_source = {tie.source_name: tie for tie in structure.parameter_ties}
    old_spec_by_source = {spec.source_name: spec for spec in structure.parameter_specs}

    components: list[ComponentSpec] = []
    for index, (component_spec, mapping) in enumerate(
        zip(ordered, new_model._param_mappings, strict=True),  # noqa: SLF001
        start=1,
    ):
        components.append(
            replace(
                component_spec,
                operator_before="+" if index > 1 else None,
                source_param_names=tuple(sorted((key, value) for key, value in component_spec.source_param_names)),
                active=True,
            )
        )

    parameter_specs: list[ParameterSpec] = []
    parameter_ties: list[ParameterTieSpec] = []
    fixed_param_names: list[str] = []
    for component_spec, mapping in zip(ordered, new_model._param_mappings, strict=True):  # noqa: SLF001
        source_map = component_spec.source_param_map
        for base_name, new_name in mapping.items():
            if new_name == "__UNIT_AMPLITUDE__":
                continue
            source_name = source_map.get(base_name, new_name)
            old_spec = old_spec_by_source.get(source_name)
            old_tie = old_tie_by_source.get(source_name)
            if old_spec is None or old_tie is None:
                continue
            parameter_specs.append(
                ParameterSpec(
                    name=new_name,
                    source_name=source_name,
                    min_value=old_spec.min_value,
                    max_value=old_spec.max_value,
                    default_value=old_spec.default_value,
                    fixed=old_spec.fixed,
                )
            )
            parameter_ties.append(
                ParameterTieSpec(
                    parameter_name=new_name,
                    source_name=source_name,
                    mode=old_tie.mode,
                    deviation_penalty=old_tie.deviation_penalty,
                )
            )
            if old_tie.mode == ParameterTieMode.FIXED:
                fixed_param_names.append(new_name)

    return ModelStructure(
        template_key=structure.template_key,
        title=structure.title,
        model=new_model,
        components=tuple(components),
        parameter_specs=tuple(parameter_specs),
        parameter_ties=tuple(parameter_ties),
        fixed_param_names=tuple(sorted(set(fixed_param_names))),
        metadata=structure.metadata,
    )


def build_parameter_sets_for_structure(
    structure: ModelStructure,
    *,
    base_by_run: dict[int, ParameterSet],
    seed_by_run: dict[int, ParameterSet] | None = None,
) -> dict[int, ParameterSet]:
    """Build legacy per-run ParameterSets for one staged-search structure."""
    seed_by_run = seed_by_run or {}
    parameter_specs = {spec.name: spec for spec in structure.parameter_specs}

    parameter_sets: dict[int, ParameterSet] = {}
    for run_number, source_params in base_by_run.items():
        params = ParameterSet()
        seed_params = seed_by_run.get(int(run_number))
        source_by_name = {parameter.name: parameter for parameter in source_params}
        seed_by_name = {parameter.name: parameter for parameter in seed_params} if seed_params is not None else {}
        for spec in structure.parameter_specs:
            if spec.name in seed_by_name:
                seed_param = seed_by_name[spec.name]
                value = float(seed_param.value)
            elif spec.source_name in source_by_name:
                value = float(source_by_name[spec.source_name].value)
            else:
                value = float(spec.default_value)
            params.add(
                Parameter(
                    name=spec.name,
                    value=min(max(value, spec.min_value), spec.max_value),
                    min=spec.min_value,
                    max=spec.max_value,
                    fixed=spec.fixed or spec.name in structure.fixed_param_names,
                )
            )
        parameter_sets[int(run_number)] = params
    return parameter_sets


def candidate_with_structure(
    candidate: DiscreteCandidate,
    structure: ModelStructure,
) -> DiscreteCandidate:
    """Return a discrete candidate carrying a replacement structure."""
    return DiscreteCandidate(
        structure=structure,
        source_relaxed_signature=candidate.source_relaxed_signature,
        initial_params_by_run=candidate.initial_params_by_run,
        extraction_notes=candidate.extraction_notes,
        approx_score=candidate.approx_score,
        relaxed_local_names=candidate.relaxed_local_names,
        ambiguous_param_names=candidate.ambiguous_param_names,
    )
