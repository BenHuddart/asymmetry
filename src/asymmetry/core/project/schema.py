"""Project file schema, validation, and migration for Asymmetry project files.

Schema versioning is independent of the package version.  The ``schema_version``
field is the contract between project files and the reader.  The
``created_with_app_version`` field is stored for diagnostics only.

Compatibility policy
--------------------
* Releases must be able to read all schema versions within the supported window.
* Only the latest schema is written.
* Migration functions are one-per-step and retained for at least one major schema revision.
* Unknown top-level fields in a valid schema are preserved on load/save cycles.

Current schema (version 1)
--------------------------
::

    {
        "schema_version": 1,
        "created_with_app_version": "0.1.0",
        "datasets": [
            {
                "run_number": 3077,
                "source_file": "/abs/path/to/file.wim",
                "metadata_overrides": {"field": 150.0}
            }
        ],
        "combined_datasets": [
            {
                "combined_run_number": -1,
                "source_run_numbers": [3077, 3078]
            }
        ],
        "browser_state": {
            "sort_column": 0,
            "sort_order": "ascending",
            "filters": {"3": ["150.0"]},
            "selected_run_numbers": [3077]
        },
        "plot_state": {
            "current_run_number": 3077,
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "fit_curve": null,
            "fit_curves": {}
        },
        "single_fit_state": {
            "model_name": "ExponentialRelaxation",
            "parameters": [
                {"name": "A0", "value": 0.2, "fixed": false, "min": "-inf", "max": "inf"}
            ]
        },
        "global_fit_state": {
            "model_name": "ExponentialRelaxation",
            "parameters": [
                {"name": "A0", "value": 0.2, "type": "Global", "bounds": "-inf, inf"}
            ]
        },
        "fourier_state": {
            "window": "none",
            "padding": 1,
            "display": "Real"
        }
    }
"""

from __future__ import annotations

import json
from pathlib import Path

CURRENT_SCHEMA_VERSION: int = 1

_SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})


class UnsupportedSchemaVersion(ValueError):
    """Raised when a project file uses an unsupported schema version."""


def migrate_to_current(data: dict) -> dict:
    """Migrate a raw project dict to the current schema version.

    Version 1 is the initial version; no migration is needed.
    Future schemas will chain ``_migrate_v1_to_v2``, etc.

    Parameters
    ----------
    data : dict
        Raw JSON dict loaded from a project file.

    Returns
    -------
    dict
        Data at the current schema version.

    Raises
    ------
    UnsupportedSchemaVersion
        If the schema version is not in the supported window.
    """
    version = data.get("schema_version", 0)
    if version not in _SUPPORTED_VERSIONS:
        raise UnsupportedSchemaVersion(
            f"Project file uses schema version {version!r}. "
            f"Supported versions: {sorted(_SUPPORTED_VERSIONS)}. "
            "Upgrade the Asymmetry package to open this file, or check that "
            "the file is a valid Asymmetry project."
        )
    return data


def validate(data: dict) -> None:
    """Validate that required top-level keys are present.

    Parameters
    ----------
    data : dict
        Project dict at the current schema version.

    Raises
    ------
    ValueError
        If required keys are missing.
    """
    required = {"schema_version", "datasets"}
    missing = required - set(data)
    if missing:
        raise ValueError(
            f"Project file is missing required keys: {sorted(missing)}. "
            "The file may be corrupt or not a valid Asymmetry project."
        )


def load_project(path: str | Path) -> dict:
    """Load, migrate, and validate a project file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.asymp`` project file.

    Returns
    -------
    dict
        Validated project state dict at the current schema version.

    Raises
    ------
    UnsupportedSchemaVersion
        If the schema version is not supported.
    ValueError
        If required keys are missing or the file is not valid JSON.
    OSError
        If the file cannot be read.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    migrated = migrate_to_current(raw)
    validate(migrated)
    return migrated


def save_project(state: dict, path: str | Path) -> None:
    """Write a project state dict to a JSON file.

    Parameters
    ----------
    state : dict
        Project state as returned by ``MainWindow.collect_project_state()``.
    path : str or Path
        Destination ``.asymp`` file path.
    """
    Path(path).write_text(
        json.dumps(state, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: object) -> object:
    """JSON serializer for numpy array/scalar types."""
    try:
        import numpy as np  # local import to avoid hard dependency at module level
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serializable")


