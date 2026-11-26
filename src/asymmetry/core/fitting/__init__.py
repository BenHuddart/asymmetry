"""Public fitting API for μSR data, resolved from each symbol’s owning module."""

from importlib import import_module

_EXPORTS_BY_MODULE = {
    "muon_fluorine": (),
    "sc": (),
    "asymmetry_global": ("GlobalFitResult", "fit_global"),
    "axis_transforms": ("AxisTransform", "validate_axis_expression"),
    "beta_calibration": ("BETA_ESTIMATION_METHODS", "BetaEstimate", "estimate_beta_detailed"),
    "component_tags": ("ComputationalCost", "FieldGeometry", "PhysicsClass"),
    "composite": ("COMPONENTS", "ComponentDefinition", "CompositeModel"),
    "composite_parameters": (
        "CompositeEvaluation",
        "CompositeExpression",
        "CompositeExpressionError",
        "CompositeParameterDefinition",
        "FunctionSpec",
        "validate_composite_expression",
    ),
    "cross_group_roles": (
        "CrossGroupCandidate",
        "CrossGroupParameterRecommendation",
        "CrossGroupRoleRecommendation",
        "suggest_cross_group_roles",
    ),
    "engine": (
        "AsymmetryScaleWarning",
        "FitCancelledError",
        "FitEngine",
        "FitResult",
        "FixedFrequencyFieldMismatchWarning",
        "drive_minuit",
    ),
    "field_scan": (
        "ScanBaselineResult",
        "ScanPeakFit",
        "as_composite_model",
        "fit_rf_resonance",
        "fit_scan_baseline",
        "fit_scan_model",
        "fit_scan_peaks",
        "parameter_set_for_model",
    ),
    "fit_log": ("FitLog", "enrich_summary_provenance"),
    "fit_quality": ("FitQuality", "assess_fit_quality"),
    "fit_wizard": (
        "CandidateAssessment",
        "CandidateTemplate",
        "FitWizardRecommendation",
        "SpectrumFingerprint",
        "assessment_with_curves",
        "build_candidate_templates",
        "build_fit_wizard_recommendation",
        "fingerprint_spectrum",
        "rerank_fit_wizard_recommendation",
    ),
    "global_fit_wizard": (
        "GlobalCandidateAssessment",
        "GlobalFitWizardCandidatePortfolio",
        "GlobalFitWizardRecommendation",
        "GlobalFitWizardScreeningTable",
        "GlobalParameterRecommendation",
        "build_global_fit_wizard_candidate_portfolio",
        "build_global_fit_wizard_recommendation",
        "build_global_fit_wizard_screening_recommendation",
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        "merge_global_fit_wizard_recommendations",
        "rerank_global_fit_wizard_recommendation",
        "series_rebin_factor",
        "series_template_alphabet",
    ),
    "global_search.trend_objective": ("SelectionObjective",),
    "grouped_time_domain": (
        "GROUP_NUISANCE_PARAMS",
        "GROUPED_SERIES_RELATIONSHIPS",
        "GROUPED_SERIES_SEEDING",
        "GroupedSeriesFitResult",
        "GroupedTimeDomainFitResult",
        "GroupedTimeDomainGroup",
        "SeedingRecommendation",
        "build_grouped_count_model",
        "build_grouped_time_domain_datasets",
        "build_grouped_time_domain_groups",
        "fit_grouped_series",
        "fit_grouped_time_domain",
        "grouped_time_domain_available",
        "recommend_grouped_series_seeding",
    ),
    "information_criteria": ("InformationCriteria", "SelectionMetric"),
    "joint": (
        "JointFitResult",
        "JointSeriesProblem",
        "SharedParameter",
        "SharedSuggestion",
        "fit_joint",
        "suggest_shared_parameters",
    ),
    "member_quality": (
        "MEMBER_QUALITY_FLAGS",
        "MemberQuality",
        "assess_member_quality",
        "member_quality_flags",
    ),
    "models": ("MODELS",),
    "mu_kinetics": (
        "ArrheniusResult",
        "BimolecularRateResult",
        "MuRelaxationSeriesResult",
        "fit_arrhenius",
        "fit_bimolecular_rate",
        "fit_mu_relaxation_series",
        "mu_relaxation_from_amplitude",
    ),
    "parameter_models": (
        "PARAMETER_MODEL_COMPONENTS",
        "ErrorMode",
        "ModelFitRange",
        "ParameterCompositeModel",
        "ParameterModelComponentDefinition",
        "ParameterModelFit",
        "ParameterModelFitResult",
        "apply_error_mode",
        "component_names_for_x",
        "effective_range_bounds",
        "evaluate_parameter_model_fit",
        "fit_parameter_model",
        "range_mask",
        "validate_fit_windows",
        "windows_mask",
    ),
    "parameters": ("AffineTie", "Parameter", "ParameterSet", "ParamInfo", "get_param_info"),
    "result_summary": ("fit_result_summary", "parameters_at_bound"),
    "rrf_offset": (
        "RRF_ROTATION_COMPONENTS",
        "UnsupportedRRFComponentError",
        "apply_rrf_offsets",
        "rrf_frequency_offsets",
        "rrf_offset_model",
    ),
    "series": ("AsymmetrySeriesResult", "fit_asymmetry_series"),
    "series_seeding": (
        "SeriesDiagnostics",
        "SeriesPoint",
        "detect_amplitude_collapse",
        "detect_frequency_outliers",
        "diagnose_series",
        "is_frequency_parameter",
        "recommend_series_seeding",
        "resolve_series_params",
        "suggest_series_seeds",
    ),
    "spectral": (
        "FREQUENCY_COMPONENT_NAMES",
        "append_frequency_field_derived_parameters",
        "default_frequency_model",
        "field_gauss_to_frequency_mhz",
        "frequency_mhz_to_field_gauss",
        "seed_peak_parameters_from_dataset",
    ),
    "time_response": ("TimeResponse",),
    "wizard_scope": (
        "ExcludedComponent",
        "ScopeQuery",
        "ScopeResolution",
        "ScopeView",
        "WizardScope",
        "describe_scope",
        "infer_run_geometries",
        "resolve_scope",
        "resolve_scope_for_dataset",
        "resolve_scope_for_datasets",
    ),
}
_EXPORT_OWNERS = {
    name: module for module, names in _EXPORTS_BY_MODULE.items() for name in names or (module,)
}
_OWNER_PACKAGES = {module.partition(".")[0] for module in _EXPORTS_BY_MODULE}

__all__ = list(_EXPORT_OWNERS)


def __getattr__(name: str):
    owner = _EXPORT_OWNERS.get(name)
    if owner is None:
        if name in _OWNER_PACKAGES:
            return import_module(f".{name}", __name__)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{owner}", __name__)
    value = module if owner == name else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_OWNER_PACKAGES})
