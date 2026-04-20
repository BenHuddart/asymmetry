"""Fitting engine for μSR data."""

from asymmetry.core.fitting import muon_fluorine, sc
from asymmetry.core.fitting.composite import COMPONENTS, ComponentDefinition, CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
    build_candidate_templates,
    build_fit_wizard_recommendation,
    compute_information_criteria,
    fingerprint_spectrum,
    rerank_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardCandidatePortfolio,
    GlobalFitWizardRecommendation,
    GlobalParameterRecommendation,
    build_global_fit_wizard_candidate_portfolio,
    build_global_fit_wizard_screening_recommendation,
    build_global_fit_wizard_recommendation,
    merge_global_fit_wizard_recommendations,
    rerank_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelComponentDefinition,
    ParameterModelFit,
    ParameterModelFitResult,
    component_names_for_x,
    evaluate_parameter_model_fit,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import ParamInfo, Parameter, ParameterSet, get_param_info

__all__ = [
    "FitEngine",
    "SelectionMetric",
    "SpectrumFingerprint",
    "CandidateTemplate",
    "CandidateAssessment",
    "FitWizardRecommendation",
    "GlobalParameterRecommendation",
    "GlobalCandidateAssessment",
    "GlobalFitWizardCandidatePortfolio",
    "GlobalFitWizardRecommendation",
    "compute_information_criteria",
    "fingerprint_spectrum",
    "build_candidate_templates",
    "build_fit_wizard_recommendation",
    "rerank_fit_wizard_recommendation",
    "build_global_fit_wizard_candidate_portfolio",
    "build_global_fit_wizard_screening_recommendation",
    "build_global_fit_wizard_recommendation",
    "merge_global_fit_wizard_recommendations",
    "rerank_global_fit_wizard_recommendation",
    "sc",
    "muon_fluorine",
    "MODELS",
    "COMPONENTS",
    "ComponentDefinition",
    "CompositeModel",
    "ParameterModelComponentDefinition",
    "ParameterCompositeModel",
    "PARAMETER_MODEL_COMPONENTS",
    "ParameterModelFit",
    "ModelFitRange",
    "ParameterModelFitResult",
    "fit_parameter_model",
    "evaluate_parameter_model_fit",
    "component_names_for_x",
    "ParamInfo",
    "get_param_info",
    "Parameter",
    "ParameterSet",
]
