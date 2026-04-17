"""Core staged-search domain types for the global fitting wizard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import ParameterSet


class ParameterTieMode(str, Enum):
    """Internal parameter-role modes for staged search."""

    FIXED = "fixed"
    SHARED = "shared"
    LOCAL = "local"
    RELAXED_SHARED = "relaxed_shared"


class SearchMoveType(str, Enum):
    """Discrete neighborhood move types for refinement."""

    SPLIT_TO_LOCAL = "split_to_local"
    MERGE_TO_SHARED = "merge_to_shared"
    REMOVE_COMPONENT = "remove_component"
    ADD_COMPONENT = "add_component"
    SWAP_COMPONENT_FAMILY = "swap_component_family"


@dataclass(frozen=True)
class ComponentSpec:
    """One component instance in a candidate structure."""

    instance_id: str
    family: str
    source_index: int
    source_param_names: tuple[tuple[str, str], ...]
    operator_before: str | None = None
    allow_activity_relaxation: bool = False
    active: bool = True

    @property
    def source_param_map(self) -> dict[str, str]:
        return dict(self.source_param_names)


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for one model parameter."""

    name: str
    source_name: str
    min_value: float
    max_value: float
    default_value: float
    fixed: bool = False

    @property
    def bounds_width(self) -> float:
        width = float(self.max_value - self.min_value)
        if width > 0.0 and width != float("inf"):
            return width
        return 1.0


@dataclass(frozen=True)
class ParameterTieSpec:
    """Role policy for one model parameter."""

    parameter_name: str
    source_name: str
    mode: ParameterTieMode
    deviation_penalty: float = 1.0


@dataclass(frozen=True)
class ModelStructure:
    """Canonical discrete model structure used for staged search."""

    template_key: str
    title: str
    model: CompositeModel
    components: tuple[ComponentSpec, ...]
    parameter_specs: tuple[ParameterSpec, ...]
    parameter_ties: tuple[ParameterTieSpec, ...]
    fixed_param_names: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def signature(self) -> tuple[object, ...]:
        return (
            self.template_key,
            tuple(component.instance_id for component in self.components if component.active),
            tuple(component.family for component in self.components if component.active),
            tuple(
                (tie.parameter_name, tie.source_name, tie.mode.value)
                for tie in sorted(self.parameter_ties, key=lambda item: item.parameter_name)
            ),
            tuple(sorted(self.fixed_param_names)),
            tuple(self.model.component_names),
            tuple(self.model.operators),
            tuple(self.model.open_parentheses),
            tuple(self.model.close_parentheses),
        )

    @property
    def parameter_spec_map(self) -> dict[str, ParameterSpec]:
        return {spec.name: spec for spec in self.parameter_specs}

    @property
    def parameter_tie_map(self) -> dict[str, ParameterTieSpec]:
        return {tie.parameter_name: tie for tie in self.parameter_ties}

    @property
    def is_additive_mixture(self) -> bool:
        return bool(self.model.operators) and all(op == "+" for op in self.model.operators)

    def role_names(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        shared: list[str] = []
        local: list[str] = []
        fixed: list[str] = list(self.fixed_param_names)
        for tie in self.parameter_ties:
            if tie.mode == ParameterTieMode.FIXED:
                fixed.append(tie.parameter_name)
            elif tie.mode == ParameterTieMode.LOCAL:
                local.append(tie.parameter_name)
            else:
                shared.append(tie.parameter_name)
        return tuple(shared), tuple(local), tuple(sorted(set(fixed)))


@dataclass(frozen=True)
class RelaxedFitProblem:
    """Input bundle for the relaxed shared-vs-local optimizer."""

    structure: ModelStructure
    datasets: tuple[object, ...]
    initial_params_by_run: dict[int, ParameterSet]
    penalty_weight: float
    bounds_scale_epsilon: float = 1e-6
    enable_activity_relaxation: bool = False
    penalty_schedule: tuple[float, ...] = ()
    active_set_threshold: float = 0.0

    def signature(self) -> tuple[object, ...]:
        return (
            self.structure.signature(),
            tuple(int(getattr(dataset, "run_number", 0)) for dataset in self.datasets),
            float(self.penalty_weight),
            bool(self.enable_activity_relaxation),
            tuple(float(value) for value in self.penalty_schedule),
            float(self.active_set_threshold),
        )


@dataclass(frozen=True)
class RelaxedFitResult:
    """Output from the relaxed shared-vs-local optimization stage."""

    success: bool
    objective_value: float
    base_values: dict[str, float]
    local_values_by_run: dict[int, dict[str, float]]
    deviations_by_run: dict[int, dict[str, float]]
    activity_weights: dict[str, float]
    fit_message: str = ""
    seed_params_by_run: dict[int, ParameterSet] = field(default_factory=dict)
    frozen_shared_names: tuple[str, ...] = ()
    continuation_penalties: tuple[float, ...] = ()
    continuation_objectives: tuple[float, ...] = ()

    def signature(self) -> tuple[object, ...]:
        return (
            tuple(sorted((name, float(value)) for name, value in self.base_values.items())),
            tuple(
                (
                    int(run_number),
                    tuple(sorted((name, float(value)) for name, value in values.items())),
                )
                for run_number, values in sorted(self.local_values_by_run.items())
            ),
            tuple(sorted((name, float(value)) for name, value in self.activity_weights.items())),
            tuple(sorted(self.frozen_shared_names)),
            tuple(float(value) for value in self.continuation_penalties),
        )


@dataclass(frozen=True)
class DiscreteCandidate:
    """One discrete candidate extracted from a relaxed result."""

    structure: ModelStructure
    source_relaxed_signature: tuple[object, ...]
    initial_params_by_run: dict[int, ParameterSet]
    extraction_notes: tuple[str, ...] = ()
    approx_score: float = float("inf")
    relaxed_local_names: tuple[str, ...] = ()
    ambiguous_param_names: tuple[str, ...] = ()

    def signature(self) -> tuple[object, ...]:
        return self.structure.signature()


@dataclass(frozen=True)
class SearchMove:
    """One local refinement mutation."""

    move_type: SearchMoveType
    target: str
    payload: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ModelScore:
    """Explicit model-selection scores for a fitted candidate."""

    chi_squared: float
    parameter_count: int
    sample_count: int
    bic: float
    aic: float
    aicc: float | None = None
    cv_score: float | None = None
    evidence_proxy: float | None = None
    primary_metric: str = "BIC"

    @property
    def primary_value(self) -> float:
        metric = self.primary_metric.upper()
        if metric == "AIC":
            return float(self.aic)
        if metric == "AICC":
            return float(self.aicc if self.aicc is not None else self.aic)
        return float(self.bic)


@dataclass
class SearchState:
    """Mutable state for greedy or beam-like local search."""

    incumbent_signature: tuple[object, ...] | None = None
    visited_signatures: set[tuple[object, ...]] = field(default_factory=set)
    history: list[str] = field(default_factory=list)
    exact_evaluations: int = 0
    approximate_candidates: int = 0
    beam_widths: list[int] = field(default_factory=list)
