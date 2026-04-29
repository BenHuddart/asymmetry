"""Exact-fit adapter for staged-search candidates."""

from __future__ import annotations

from dataclasses import dataclass

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.global_search.adapters import (
    build_parameter_sets_for_structure,
    compile_structure_to_legacy_roles,
)
from asymmetry.core.fitting.global_search.score import score_exact_candidate
from asymmetry.core.fitting.global_search.types import DiscreteCandidate, ModelScore
from asymmetry.core.fitting.parameters import ParameterSet


@dataclass(frozen=True)
class ExactCandidateFit:
    """Exact-fit payload for one discrete candidate."""

    results_by_run: dict[int, FitResult]
    global_parameters: ParameterSet
    score: ModelScore
    initial_params_by_run: dict[int, ParameterSet]


class ExactStructureFitter:
    """Thin adapter over :class:`FitEngine` for staged search."""

    def __init__(self, fit_engine: FitEngine | None = None) -> None:
        self._fit_engine = fit_engine or FitEngine()

    def fit_candidate(
        self,
        datasets: list[MuonDataset],
        candidate: DiscreteCandidate,
        *,
        base_by_run: dict[int, ParameterSet],
        metric: SelectionMetric = SelectionMetric.BIC,
        max_calls: int = 1600,
    ) -> ExactCandidateFit:
        initial_params_by_run = build_parameter_sets_for_structure(
            candidate.structure,
            base_by_run=base_by_run,
            seed_by_run=candidate.initial_params_by_run,
        )
        global_names, local_names, _fixed_names = compile_structure_to_legacy_roles(
            candidate.structure
        )
        results_by_run, global_parameters = self._fit_engine.global_fit(
            datasets,
            candidate.structure.model.function,
            list(global_names),
            list(local_names),
            initial_params_by_run,
            max_calls=max_calls,
        )
        total_chi2 = float(sum(result.chi_squared for result in results_by_run.values()))
        sample_count = int(sum(dataset.n_points for dataset in datasets))
        parameter_count = len(global_names) + len(local_names) * len(datasets)
        score = score_exact_candidate(
            total_chi2,
            parameter_count,
            sample_count,
            primary_metric=metric.value,
        )
        return ExactCandidateFit(
            results_by_run=results_by_run,
            global_parameters=global_parameters,
            score=score,
            initial_params_by_run=initial_params_by_run,
        )
