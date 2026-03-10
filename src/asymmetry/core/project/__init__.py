"""Project file persistence for Asymmetry.

Provides versioned JSON project files that capture the full GUI state
and can be reopened after package upgrades.
"""

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
]
