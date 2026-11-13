"""Project file persistence for Asymmetry.

Provides versioned JSON project files that capture the full GUI state
and can be reopened after package upgrades.
"""

from asymmetry.core.project.preparation import (
    PreparedProject,
    ProjectValidationError,
    load_prepared_project,
    prepare_project,
)
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    DeadtimePolicy,
    GoodWindowPolicy,
    GroupingProfile,
    ProfileFingerprint,
    aligned_n_bins,
    assigned_profile_for_run,
    default_profile_for_run,
    effective_grouping_for_loaded_run,
    named_profile_for_run,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
    resolve_good_window,
    run_file_good_window,
)
from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    load_project,
    save_project,
)
from asymmetry.core.project.sources import (
    ProjectSourceApplication,
    ProjectSourceVerification,
    apply_source_decisions,
    verify_project_sources,
)

__all__ = [
    "ProjectSourceApplication",
    "ProjectSourceVerification",
    "apply_source_decisions",
    "verify_project_sources",
    "PreparedProject",
    "ProjectValidationError",
    "load_prepared_project",
    "prepare_project",
    "CURRENT_SCHEMA_VERSION",
    "UnsupportedSchemaVersion",
    "load_project",
    "save_project",
    # Grouping profiles (schema v12).
    "AlphaPolicy",
    "BackgroundPolicy",
    "DeadtimePolicy",
    "GoodWindowPolicy",
    "GroupingProfile",
    "ProfileFingerprint",
    "aligned_n_bins",
    "assigned_profile_for_run",
    "default_profile_for_run",
    "effective_grouping_for_loaded_run",
    "named_profile_for_run",
    "profile_fingerprint_for_run",
    "profile_from_payload",
    "resolve_effective_grouping",
    "resolve_good_window",
    "run_file_good_window",
]
