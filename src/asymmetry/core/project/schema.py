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
from copy import deepcopy
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
    data = _normalise_legacy_top_level(data)

    version = data.get("schema_version", 0)
    if version not in _SUPPORTED_VERSIONS:
        raise UnsupportedSchemaVersion(
            f"Project file uses schema version {version!r}. "
            f"Supported versions: {sorted(_SUPPORTED_VERSIONS)}. "
            "Upgrade the Asymmetry package to open this file, or check that "
            "the file is a valid Asymmetry project."
        )
    migrated = _normalise_legacy_v1_payload(data)
    _ensure_optional_sections(migrated)
    return migrated


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


def _normalise_legacy_top_level(data: dict) -> dict:
    """Return a copy with tolerant handling for legacy top-level keys.

    Historical builds may omit ``schema_version`` or use ``app_version``.
    If the payload looks like an Asymmetry project, assume schema v1.
    """
    out = deepcopy(data)

    if "created_with_app_version" not in out and "app_version" in out:
        out["created_with_app_version"] = out["app_version"]

    if "schema_version" not in out and isinstance(out.get("datasets"), list):
        out["schema_version"] = 1

    return out


def _normalise_legacy_v1_payload(data: dict) -> dict:
    """Normalise legacy v1 aliases to the current canonical key names."""
    out = deepcopy(data)

    # Older docs/examples used nested fit_state.single/global blocks.
    fit_state = out.get("fit_state")
    if isinstance(fit_state, dict):
        if "single_fit_state" not in out and isinstance(fit_state.get("single"), dict):
            out["single_fit_state"] = fit_state["single"]
        if "global_fit_state" not in out and isinstance(fit_state.get("global"), dict):
            out["global_fit_state"] = fit_state["global"]

    # Fit-state aliases from older examples/builds.
    single_fit_state = out.get("single_fit_state")
    if isinstance(single_fit_state, dict):
        _normalise_single_fit_state(single_fit_state)

    global_fit_state = out.get("global_fit_state")
    if isinstance(global_fit_state, dict):
        _normalise_global_fit_state(global_fit_state)

    # Dataset-level aliases seen in legacy examples.
    datasets = out.get("datasets")
    if isinstance(datasets, list):
        normalised_datasets: list[dict] = []
        for entry in datasets:
            if not isinstance(entry, dict):
                continue
            ds = deepcopy(entry)
            if "source_file" not in ds and "source_path" in ds:
                ds["source_file"] = ds["source_path"]
            if "metadata_overrides" not in ds and isinstance(ds.get("metadata"), dict):
                ds["metadata_overrides"] = ds["metadata"]
            normalised_datasets.append(ds)
        out["datasets"] = normalised_datasets

    return out


def _ensure_optional_sections(data: dict) -> None:
    """Populate optional top-level sections with safe defaults.

    This keeps restore paths resilient when loading project files from older
    builds that did not persist every panel state section.
    """
    data.setdefault("combined_datasets", [])
    data.setdefault("browser_state", {})
    data.setdefault("plot_state", {})
    data.setdefault("single_fit_state", {})
    data.setdefault("global_fit_state", {})
    data.setdefault("fit_ui_state", {})
    data.setdefault("fit_parameters_state", {})
    data.setdefault("fourier_state", {})


def _normalise_single_fit_state(state: dict) -> None:
    """Normalise legacy single-fit key aliases in-place."""
    if "model_name" not in state and "model" in state:
        state["model_name"] = state["model"]


def _normalise_global_fit_state(state: dict) -> None:
    """Normalise legacy global-fit key aliases in-place."""
    if "model_name" not in state and "model" in state:
        state["model_name"] = state["model"]

    params = state.get("parameters")
    if not isinstance(params, list):
        return
    for p in params:
        if not isinstance(p, dict):
            continue
        if "type" not in p and "classification" in p:
            p["type"] = p["classification"]
