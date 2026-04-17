"""Greedy local refinement for staged discrete search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from asymmetry.core.fitting.global_search.cache import ApproximateScoreCache, ExactStructureCache, WarmStartStore
from asymmetry.core.fitting.global_search.types import DiscreteCandidate, ModelScore, SearchState


@dataclass(frozen=True)
class SearchEvaluation:
    """Result of evaluating one discrete candidate."""

    candidate: DiscreteCandidate
    score: ModelScore
    payload: object


def refine_candidate_search(
    initial_candidates: tuple[DiscreteCandidate, ...],
    *,
    evaluator: Callable[[DiscreteCandidate], SearchEvaluation],
    move_generator: Callable[[DiscreteCandidate], tuple[object, ...]],
    move_applier: Callable[[DiscreteCandidate, object], DiscreteCandidate | None],
    approximate_scorer: Callable[[DiscreteCandidate], float] | None = None,
    max_steps: int = 8,
    max_neighbors: int = 6,
    beam_width: int = 1,
    max_exact_evaluations_per_step: int | None = None,
    exact_cache: ExactStructureCache | None = None,
    approx_cache: ApproximateScoreCache | None = None,
    warm_start_store: WarmStartStore | None = None,
) -> tuple[SearchEvaluation, tuple[str, ...]]:
    """Run a bounded beam or greedy discrete refinement search."""
    exact_cache = exact_cache or ExactStructureCache()
    approx_cache = approx_cache or ApproximateScoreCache()
    warm_start_store = warm_start_store or WarmStartStore()
    state = SearchState()
    diagnostics: list[str] = []
    effective_beam_width = max(1, int(beam_width))
    exact_budget = max_exact_evaluations_per_step
    if exact_budget is None:
        exact_budget = max_neighbors
    exact_budget = max(1, int(exact_budget))

    def evaluate(candidate: DiscreteCandidate) -> SearchEvaluation:
        key = candidate.signature()
        cached = exact_cache.get(key)
        if isinstance(cached, SearchEvaluation):
            return cached
        result = evaluator(candidate)
        exact_cache.put(key, result)
        warm_start_store.put(key, candidate.initial_params_by_run)
        state.visited_signatures.add(key)
        state.exact_evaluations += 1
        return result

    ranked_initial = sorted(
        (evaluate(candidate) for candidate in initial_candidates),
        key=lambda item: item.score.primary_value,
    )
    beam = ranked_initial[:effective_beam_width]
    best = beam[0]
    state.incumbent_signature = best.candidate.signature()
    state.beam_widths.append(len(beam))
    diagnostics.append(
        f"Initialized refinement beam with {len(beam)} exact candidates."
    )

    for _step in range(max_steps):
        neighbors_by_signature: dict[tuple[object, ...], tuple[float, DiscreteCandidate]] = {}
        for incumbent in beam:
            for move in move_generator(incumbent.candidate):
                next_candidate = move_applier(incumbent.candidate, move)
                if next_candidate is None:
                    continue
                signature = next_candidate.signature()
                if signature in state.visited_signatures:
                    continue
                if approximate_scorer is None:
                    approx_score = next_candidate.approx_score
                else:
                    approx_score = approximate_scorer(next_candidate)
                state.approximate_candidates += 1
                approx_cache.put(signature, approx_score)
                current_best = neighbors_by_signature.get(signature)
                if current_best is None or approx_score < current_best[0]:
                    neighbors_by_signature[signature] = (approx_score, next_candidate)

        neighbors = sorted(neighbors_by_signature.values(), key=lambda item: item[0])
        if not neighbors:
            break
        diagnostics.append(
            f"Refinement step {_step + 1}: screened {len(neighbors)} approximate neighbors."
        )

        evaluated = [
            evaluate(next_candidate)
            for _approx_score, next_candidate in neighbors[:exact_budget]
        ]
        if not evaluated:
            break
        combined: dict[tuple[object, ...], SearchEvaluation] = {
            item.candidate.signature(): item for item in beam
        }
        for item in evaluated:
            combined[item.candidate.signature()] = item
        next_beam = sorted(
            combined.values(),
            key=lambda item: item.score.primary_value,
        )[:effective_beam_width]
        state.beam_widths.append(len(next_beam))
        diagnostics.append(
            f"Refinement step {_step + 1}: exact-evaluated {len(evaluated)} candidates, beam size {len(next_beam)}."
        )
        improved = any(
            item.score.primary_value + 1e-9 < best.score.primary_value
            for item in evaluated
        )
        beam = next_beam
        if beam[0].score.primary_value + 1e-9 < best.score.primary_value:
            best = beam[0]
            state.incumbent_signature = best.candidate.signature()
        if not improved:
            break

    diagnostics.append(
        f"Refinement completed after {state.exact_evaluations} exact evaluations and {state.approximate_candidates} approximate screenings."
    )
    return best, tuple(diagnostics)

