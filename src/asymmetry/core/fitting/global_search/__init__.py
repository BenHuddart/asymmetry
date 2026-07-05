"""Staged-search helpers for the global fitting wizard."""

from asymmetry.core.fitting.global_search.adapters import (
    build_parameter_sets_for_structure,
    compile_legacy_structure,
    compile_structure_to_legacy_roles,
)
from asymmetry.core.fitting.global_search.exact import ExactCandidateFit, ExactStructureFitter
from asymmetry.core.fitting.global_search.homogeneity import (
    ParameterHomogeneity,
    chi2_sf,
    classify_parameter_homogeneity,
    homogeneity_statistic,
    wald_globalisation_cost,
    wald_subset_delta_chi2,
)
from asymmetry.core.fitting.global_search.orchestrator import (
    GlobalSearchConfig,
    GlobalSearchOrchestrator,
)
from asymmetry.core.fitting.global_search.proposal import extract_discrete_candidates
from asymmetry.core.fitting.global_search.relaxed import RelaxedOptimizer, SciPyRelaxedOptimizer
from asymmetry.core.fitting.global_search.score import score_exact_candidate
from asymmetry.core.fitting.global_search.types import (
    ComponentSpec,
    DiscreteCandidate,
    ModelScore,
    ModelStructure,
    ParameterSpec,
    ParameterTieMode,
    ParameterTieSpec,
    RelaxedFitProblem,
    RelaxedFitResult,
    SearchMove,
    SearchMoveType,
    SearchState,
)

__all__ = [
    "ComponentSpec",
    "DiscreteCandidate",
    "ExactCandidateFit",
    "ExactStructureFitter",
    "GlobalSearchConfig",
    "GlobalSearchOrchestrator",
    "ModelScore",
    "ModelStructure",
    "ParameterHomogeneity",
    "ParameterSpec",
    "ParameterTieMode",
    "ParameterTieSpec",
    "RelaxedFitProblem",
    "RelaxedFitResult",
    "RelaxedOptimizer",
    "SciPyRelaxedOptimizer",
    "SearchMove",
    "SearchMoveType",
    "SearchState",
    "build_parameter_sets_for_structure",
    "chi2_sf",
    "classify_parameter_homogeneity",
    "compile_legacy_structure",
    "compile_structure_to_legacy_roles",
    "extract_discrete_candidates",
    "homogeneity_statistic",
    "score_exact_candidate",
    "wald_globalisation_cost",
    "wald_subset_delta_chi2",
]
