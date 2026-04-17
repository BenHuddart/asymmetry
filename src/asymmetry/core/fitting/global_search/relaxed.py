"""Relaxed shared-vs-local optimization for staged global fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from asymmetry.core.fitting.global_search.types import (
    ModelStructure,
    ParameterTieMode,
    RelaxedFitProblem,
    RelaxedFitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


class RelaxedOptimizer(Protocol):
    """Protocol for relaxed staged-search optimizers."""

    def solve(self, problem: RelaxedFitProblem) -> RelaxedFitResult:
        """Solve one relaxed optimization problem."""


@dataclass(frozen=True)
class _VariableLayout:
    base_names: tuple[str, ...]
    local_names: tuple[str, ...]
    run_numbers: tuple[int, ...]


class SciPyRelaxedOptimizer:
    """Practical relaxed optimizer using smooth L1 penalties and SciPy."""

    def __init__(self, *, smoothing_epsilon: float = 1e-6, maxiter: int = 400) -> None:
        self._smoothing_epsilon = float(max(smoothing_epsilon, 1e-12))
        self._maxiter = int(max(maxiter, 50))

    def solve(self, problem: RelaxedFitProblem) -> RelaxedFitResult:
        structure = problem.structure
        run_numbers = tuple(sorted(int(run_number) for run_number in problem.initial_params_by_run))
        active_names = tuple(
            spec.name
            for spec in structure.parameter_specs
            if spec.name not in structure.fixed_param_names
            and structure.parameter_tie_map[spec.name].mode != ParameterTieMode.FIXED
        )
        parameter_specs = structure.parameter_spec_map
        penalty_schedule = tuple(
            float(value) for value in (problem.penalty_schedule or (problem.penalty_weight,))
        )
        current_base_values = {
            name: self._base_seed(problem, name)
            for name in active_names
        }
        current_local_values = {
            int(run_number): {
                name: float(problem.initial_params_by_run[run_number][name].value)
                for name in active_names
                if name in problem.initial_params_by_run[run_number]
            }
            for run_number in run_numbers
        }
        frozen_shared_names: set[str] = set()
        continuation_objectives: list[float] = []
        message = ""
        success = False

        def objective(
            vector: np.ndarray,
            layout: _VariableLayout,
            *,
            penalty_weight: float,
        ) -> float:
            base_values, partial_local_values = self._decode_vector(vector, layout)
            local_values = self._expand_local_values(
                base_values,
                partial_local_values,
                active_names=active_names,
                frozen_shared_names=frozen_shared_names,
                run_numbers=run_numbers,
            )
            objective_value = 0.0
            for dataset in problem.datasets:
                run_number = int(dataset.run_number)
                params = self._parameter_dict_for_run(
                    structure,
                    problem.initial_params_by_run[run_number],
                    local_values[run_number],
                )
                model_values = np.asarray(
                    structure.model.function(dataset.time, **params),
                    dtype=float,
                )
                if not np.all(np.isfinite(model_values)):
                    return 1.0e30
                errors = np.asarray(dataset.error, dtype=float)
                safe_errors = np.where(
                    np.isfinite(errors) & (errors > 0.0),
                    errors,
                    1e-12,
                )
                residual = (np.asarray(dataset.asymmetry, dtype=float) - model_values) / safe_errors
                objective_value += float(np.sum(residual * residual))

            for name in layout.base_names:
                spec = parameter_specs[name]
                tie = structure.parameter_tie_map[name]
                scale = max(spec.bounds_width, problem.bounds_scale_epsilon)
                if tie.mode == ParameterTieMode.LOCAL:
                    continue
                if tie.mode == ParameterTieMode.SHARED:
                    for run_number in layout.run_numbers:
                        diff = (local_values[run_number][name] - base_values[name]) / scale
                        objective_value += 1.0e4 * diff * diff
                    continue
                for run_number in layout.run_numbers:
                    diff = (local_values[run_number][name] - base_values[name]) / scale
                    objective_value += penalty_weight * math.sqrt(
                        diff * diff + self._smoothing_epsilon * self._smoothing_epsilon
                    )
            return float(objective_value)

        for penalty_weight in penalty_schedule:
            layout = _VariableLayout(
                base_names=active_names,
                local_names=tuple(
                    name for name in active_names if name not in frozen_shared_names
                ),
                run_numbers=run_numbers,
            )
            initial_values: list[float] = []
            bounds: list[tuple[float, float]] = []
            for name in layout.base_names:
                initial_values.append(float(current_base_values[name]))
                spec = parameter_specs[name]
                bounds.append((spec.min_value, spec.max_value))
            for run_number in layout.run_numbers:
                for name in layout.local_names:
                    initial_values.append(float(current_local_values[run_number].get(name, current_base_values[name])))
                    spec = parameter_specs[name]
                    bounds.append((spec.min_value, spec.max_value))

            initial = np.asarray(initial_values, dtype=float)
            try:
                result = minimize(
                    lambda vector: objective(vector, layout, penalty_weight=penalty_weight),
                    initial,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self._maxiter},
                )
                vector = np.asarray(result.x if result.success else initial, dtype=float)
                message = str(result.message)
                success = bool(result.success or np.isfinite(getattr(result, "fun", np.inf)))
            except Exception as exc:  # pragma: no cover - defensive fallback
                vector = initial
                message = str(exc)
                success = False

            current_base_values, partial_local_values = self._decode_vector(vector, layout)
            current_local_values = self._expand_local_values(
                current_base_values,
                partial_local_values,
                active_names=active_names,
                frozen_shared_names=frozen_shared_names,
                run_numbers=run_numbers,
            )
            continuation_objectives.append(
                float(objective(vector, layout, penalty_weight=penalty_weight))
            )
            newly_frozen = self._frozen_shared_names(
                structure,
                parameter_specs,
                base_values=current_base_values,
                local_values_by_run=current_local_values,
                active_set_threshold=problem.active_set_threshold,
            )
            if newly_frozen:
                frozen_shared_names.update(newly_frozen)

        base_values = current_base_values
        local_values = current_local_values
        deviations: dict[int, dict[str, float]] = {}
        seed_params_by_run: dict[int, ParameterSet] = {}
        for run_number in run_numbers:
            deviation_map = {}
            seeded = ParameterSet()
            source = problem.initial_params_by_run[run_number]
            for parameter in source:
                if parameter.name in local_values[run_number]:
                    value = float(local_values[run_number][parameter.name])
                else:
                    value = float(parameter.value)
                deviation_map[parameter.name] = float(value - base_values.get(parameter.name, value))
                seeded.add(
                    Parameter(
                        name=parameter.name,
                        value=value,
                        min=parameter.min,
                        max=parameter.max,
                        fixed=parameter.fixed,
                    )
                )
            deviations[run_number] = deviation_map
            seed_params_by_run[run_number] = seeded

        activity_weights = self._activity_weights(structure, local_values, parameter_specs)
        return RelaxedFitResult(
            success=success,
            objective_value=float(continuation_objectives[-1]) if continuation_objectives else 0.0,
            base_values=base_values,
            local_values_by_run=local_values,
            deviations_by_run=deviations,
            activity_weights=activity_weights,
            fit_message=message,
            seed_params_by_run=seed_params_by_run,
            frozen_shared_names=tuple(sorted(frozen_shared_names)),
            continuation_penalties=penalty_schedule,
            continuation_objectives=tuple(continuation_objectives),
        )

    def _expand_local_values(
        self,
        base_values: dict[str, float],
        partial_local_values: dict[int, dict[str, float]],
        *,
        active_names: tuple[str, ...],
        frozen_shared_names: set[str],
        run_numbers: tuple[int, ...],
    ) -> dict[int, dict[str, float]]:
        expanded: dict[int, dict[str, float]] = {}
        for run_number in run_numbers:
            run_map: dict[str, float] = {}
            for name in active_names:
                if name in frozen_shared_names:
                    run_map[name] = float(base_values[name])
                else:
                    run_map[name] = float(
                        partial_local_values.get(run_number, {}).get(name, base_values[name])
                    )
            expanded[run_number] = run_map
        return expanded

    def _frozen_shared_names(
        self,
        structure: ModelStructure,
        parameter_specs: dict[str, object],
        *,
        base_values: dict[str, float],
        local_values_by_run: dict[int, dict[str, float]],
        active_set_threshold: float,
    ) -> set[str]:
        if active_set_threshold <= 0.0:
            return set()
        frozen: set[str] = set()
        for name, tie in structure.parameter_tie_map.items():
            if tie.mode != ParameterTieMode.RELAXED_SHARED:
                continue
            if name not in base_values:
                continue
            spec = parameter_specs.get(name)
            scale = max(spec.bounds_width if spec is not None else 1.0, 1e-6)
            max_deviation = max(
                abs(local_values.get(name, base_values[name]) - base_values[name]) / scale
                for local_values in local_values_by_run.values()
            )
            if max_deviation <= active_set_threshold:
                frozen.add(name)
        return frozen

    def _base_seed(self, problem: RelaxedFitProblem, name: str) -> float:
        values = []
        for params in problem.initial_params_by_run.values():
            if name in params:
                values.append(float(params[name].value))
        if values:
            return float(np.mean(values))
        spec = problem.structure.parameter_spec_map[name]
        return float(spec.default_value)

    def _decode_vector(
        self,
        vector: np.ndarray,
        layout: _VariableLayout,
    ) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
        offset = 0
        base_values = {
            name: float(vector[offset + index])
            for index, name in enumerate(layout.base_names)
        }
        offset += len(layout.base_names)
        local_values: dict[int, dict[str, float]] = {}
        for run_number in layout.run_numbers:
            run_map = {}
            for name in layout.local_names:
                run_map[name] = float(vector[offset])
                offset += 1
            local_values[run_number] = run_map
        return base_values, local_values

    def _parameter_dict_for_run(
        self,
        structure: ModelStructure,
        source_params: ParameterSet,
        local_values: dict[str, float],
    ) -> dict[str, float]:
        values = {parameter.name: float(parameter.value) for parameter in source_params}
        values.update(local_values)
        return values

    def _activity_weights(
        self,
        structure: ModelStructure,
        local_values_by_run: dict[int, dict[str, float]],
        parameter_specs: dict[str, object],
    ) -> dict[str, float]:
        if not structure.is_additive_mixture:
            return {}
        weights: dict[str, float] = {}
        mappings = structure.model._param_mappings  # noqa: SLF001
        for component, mapping in zip(structure.components, mappings, strict=True):
            amplitude_name = mapping.get("A", mapping.get("A_bg"))
            if amplitude_name in {None, "__UNIT_AMPLITUDE__"}:
                continue
            amplitudes = [
                abs(local_values.get(amplitude_name, 0.0))
                for local_values in local_values_by_run.values()
            ]
            scale_spec = parameter_specs.get(amplitude_name)
            scale = scale_spec.bounds_width if scale_spec is not None else 1.0
            weights[component.instance_id] = float(np.mean(amplitudes) / max(scale, 1e-6))
        return weights
