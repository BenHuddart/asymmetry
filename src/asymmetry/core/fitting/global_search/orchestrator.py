"""Staged global-search orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.global_search.cache import ApproximateScoreCache, ExactStructureCache, WarmStartStore
from asymmetry.core.fitting.global_search.diagnostics import summarize_relaxed_fit
from asymmetry.core.fitting.global_search.moves import apply_search_move, generate_search_moves
from asymmetry.core.fitting.global_search.proposal import extract_discrete_candidates
from asymmetry.core.fitting.global_search.refine import SearchEvaluation, refine_candidate_search
from asymmetry.core.fitting.global_search.relaxed import RelaxedOptimizer, SciPyRelaxedOptimizer
from asymmetry.core.fitting.global_search.types import ModelStructure, RelaxedFitProblem
from asymmetry.core.fitting.parameters import ParameterSet


@dataclass(frozen=True)
class GlobalSearchConfig:
    """Configuration for staged search orchestration."""

    metric: SelectionMetric = SelectionMetric.BIC
    penalty_weight: float = 0.0
    deviation_threshold: float = 0.05
    ambiguity_band: float = 0.02
    activity_threshold: float = 0.02
    max_steps: int = 8
    max_neighbors: int = 6
    beam_width: int = 1
    max_exact_evaluations_per_step: int | None = None
    max_alternates: int = 2
    active_set_threshold: float = 0.0
    penalty_schedule: tuple[float, ...] = ()
    allow_backward_moves: bool = False
    instrumentation: dict[str, object] | None = None


class GlobalSearchOrchestrator:
    """Run relaxed proposal extraction plus greedy exact refinement."""

    def __init__(self, optimizer: RelaxedOptimizer | None = None) -> None:
        self._optimizer = optimizer or SciPyRelaxedOptimizer()
        self._exact_cache = ExactStructureCache()
        self._approx_cache = ApproximateScoreCache()
        self._warm_starts = WarmStartStore()

    def search(
        self,
        *,
        structure: ModelStructure,
        datasets: list[MuonDataset],
        base_by_run: dict[int, ParameterSet],
        evaluator: Callable[[object], SearchEvaluation],
        progress_callback: Callable[[str], None] | None = None,
        config: GlobalSearchConfig | None = None,
    ) -> tuple[SearchEvaluation, tuple[str, ...]]:
        config = config or GlobalSearchConfig()
        penalty_weight = float(config.penalty_weight)
        if penalty_weight <= 0.0:
            penalty_weight = max(2.0, math.log(max(sum(dataset.n_points for dataset in datasets), 2)))
        penalty_schedule = tuple(
            float(value) for value in config.penalty_schedule
        )
        if not penalty_schedule:
            if config.active_set_threshold > 0.0 or config.beam_width > 1:
                penalty_schedule = (0.5 * penalty_weight, penalty_weight, 1.75 * penalty_weight)
            else:
                penalty_schedule = (penalty_weight,)
        problem = RelaxedFitProblem(
            structure=structure,
            datasets=tuple(datasets),
            initial_params_by_run=base_by_run,
            penalty_weight=penalty_weight,
            enable_activity_relaxation=structure.is_additive_mixture,
            penalty_schedule=penalty_schedule,
            active_set_threshold=float(config.active_set_threshold),
        )
        relaxed_result = self._optimizer.solve(problem)
        diagnostics = list(summarize_relaxed_fit(relaxed_result))
        if relaxed_result.continuation_penalties:
            diagnostics.append(
                "Relaxed continuation penalties: "
                + ", ".join(f"{value:.3g}" for value in relaxed_result.continuation_penalties)
            )
        if relaxed_result.frozen_shared_names:
            diagnostics.append(
                "Active-set froze shared candidates: "
                + ", ".join(relaxed_result.frozen_shared_names)
            )
        instrumentation = config.instrumentation
        if instrumentation is not None:
            instrumentation["relaxed_penalties"] = list(relaxed_result.continuation_penalties)
            instrumentation["relaxed_objectives"] = list(relaxed_result.continuation_objectives)
            instrumentation["relaxed_frozen_shared_names"] = list(relaxed_result.frozen_shared_names)

        proposals = extract_discrete_candidates(
            relaxed_result,
            structure,
            deviation_threshold=config.deviation_threshold,
            ambiguity_band=config.ambiguity_band,
            activity_threshold=config.activity_threshold,
            max_alternates=config.max_alternates,
        )
        if instrumentation is not None:
            instrumentation["proposal_count"] = len(proposals)
        if progress_callback is not None:
            for index, candidate in enumerate(proposals, start=1):
                global_names, local_names, _fixed_names = candidate.structure.role_names()
                label = "primary" if index == 1 else f"alternate {index - 1}"
                progress_callback(
                    "Staged proposal "
                    f"{label}: Global[{', '.join(global_names) or 'none'}], "
                    f"Local[{', '.join(local_names) or 'none'}]."
                )
                for note in candidate.extraction_notes:
                    progress_callback(f"Staged proposal {label} note: {note}")
        evaluation, refine_diagnostics = refine_candidate_search(
            proposals,
            evaluator=evaluator,
            move_generator=lambda candidate: generate_search_moves(
                candidate,
                full_structure=structure,
                allow_backward_moves=config.allow_backward_moves,
            ),
            move_applier=lambda candidate, move: apply_search_move(candidate, move, full_structure=structure),
            approximate_scorer=lambda candidate: self._approximate_candidate_score(candidate),
            max_steps=config.max_steps,
            max_neighbors=config.max_neighbors,
            beam_width=config.beam_width,
            max_exact_evaluations_per_step=config.max_exact_evaluations_per_step,
            exact_cache=self._exact_cache,
            approx_cache=self._approx_cache,
            warm_start_store=self._warm_starts,
        )
        diagnostics.extend(refine_diagnostics)
        if instrumentation is not None:
            instrumentation["search_diagnostics"] = list(diagnostics)
        return evaluation, tuple(diagnostics)

    def _approximate_candidate_score(self, candidate) -> float:
        shared_names, local_names, _fixed_names = candidate.structure.role_names()
        relaxed_local_names = set(candidate.relaxed_local_names)
        actual_local_names = set(local_names)
        mismatch_count = len(actual_local_names.symmetric_difference(relaxed_local_names))
        ambiguous_overlap = len(set(candidate.ambiguous_param_names) & actual_local_names)
        return float(candidate.approx_score) + (0.35 * mismatch_count) + (0.1 * ambiguous_overlap)
