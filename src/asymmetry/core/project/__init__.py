"""Project file persistence for Asymmetry.

Provides versioned JSON project files that capture the full GUI state
and can be reopened after package upgrades.
"""

from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    DeadtimePolicy,
    GroupingProfile,
    ProfileFingerprint,
    active_profile_for_run,
    effective_grouping_for_loaded_run,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    load_project,
    save_project,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "UnsupportedSchemaVersion",
    "load_project",
    "save_project",
    # Grouping profiles (schema v12).
    "AlphaPolicy",
    "BackgroundPolicy",
    "DeadtimePolicy",
    "GroupingProfile",
    "ProfileFingerprint",
    "active_profile_for_run",
    "effective_grouping_for_loaded_run",
    "profile_fingerprint_for_run",
    "profile_from_payload",
    "resolve_effective_grouping",
]
