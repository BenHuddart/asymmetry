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

Current schema (version 9)
--------------------------
::

    {
        "schema_version": 1,
        "created_with_app_version": "0.1.0",
        "datasets": [
            {
                "run_number": 3077,
                "source_file": "/abs/path/to/file.nxs",
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
            "selected_run_numbers": [3077],
            "selected_group_ids": [],
            "data_groups": []
        },
        "plot_state": {
            "current_run_number": 3077,
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "workspace_state": {
                "active_domain": "time"
            },
            "frequency_plot_state": {
                "plot_panel_domain": "frequency",
                "x_min": 0.0,
                "x_max": 100.0,
                "y_min": -1.0,
                "y_max": 10.0,
                "frequency_x_unit": "frequency_mhz",
                "frequency_x_limits_by_unit": {}
            },
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
            "filter_start_us": 0.0,
            "filter_time_constant_us": 1.5,
            "padding": 1,
            "phase_degrees": 0.0,
            "t0_offset_us": 0.0,
            "display": "(Power)^1/2",
            "auto_phase": false,
            "auto_phase_method": "Peak",
            "use_phase_table": false,
            "estimate_average_error": false,
            "group_enabled_table": {},
            "group_phase_table": {}
        },
        "frequency_fit_state": {
            "domain": "frequency",
            "single_fit_state": {},
            "global_fit_state": {},
            "fit_ui_state": {}
        }
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CURRENT_SCHEMA_VERSION: int = 9

_SUPPORTED_VERSIONS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})

#: Fourier-state keys that describe the FFT generation recipe (recipe-only
#: persistence carries these into each dataset's ``freq_fft`` representation).
_FOURIER_RECIPE_KEYS: tuple[str, ...] = (
    "window",
    "padding",
    "phase_degrees",
    "t0_offset_us",
    "display",
    "filter_start_us",
    "filter_time_constant_us",
    "subtract_average_signal",
    "group_enabled_table",
    # Frequency-domain finishers (post-FFT conditioning); additive, defaulted.
    "pulse_compensation",
    "pulse_half_width_us",
    "pulse_max_gain",
    "baseline_mode",
    "baseline_kappa",
    "exclude_enabled",
    "diamag_exclusion",
    "diamag_half_width_mhz",
    "exclusion_ranges",
    "remove_diamag",
    "burg_order_min",
    "burg_order_max",
    # Muoniated-radical correlation spectrum; additive, defaulted.
    "correlation_reference_field_gauss",
    "correlation_order",
)


class UnsupportedSchemaVersion(ValueError):
    """Raised when a project file uses an unsupported schema version."""


def migrate_to_current(data: dict) -> dict:
    """Migrate a raw project dict to the current schema version.

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
    migrated = dict(data)
    if version == 1:
        migrated = _migrate_v1_to_v2(migrated)
        version = 2
    if version == 2:
        migrated = _migrate_v2_to_v3(migrated)
        version = 3
    if version == 3:
        migrated = _migrate_v3_to_v4(migrated)
        version = 4
    if version == 4:
        migrated = _migrate_v4_to_v5(migrated)
        version = 5
    if version == 5:
        migrated = _migrate_v5_to_v6(migrated)
        version = 6
    if version == 6:
        migrated = _migrate_v6_to_v7(migrated)
        version = 7
    if version == 7:
        migrated = _migrate_v7_to_v8(migrated)
        version = 8
    if version == 8:
        migrated = _migrate_v8_to_v9(migrated)
    return migrated


def _migrate_v1_to_v2(data: dict) -> dict:
    """Migrate schema v1 project state to v2.

    v2 adds browser group metadata used by grouped global-fitting workflows.
    """
    migrated = dict(data)
    migrated["schema_version"] = 2

    browser_state = dict(migrated.get("browser_state", {}))
    browser_state.setdefault("selected_group_ids", [])
    browser_state.setdefault("data_groups", [])
    migrated["browser_state"] = browser_state

    return migrated


def _migrate_v2_to_v3(data: dict) -> dict:
    """Migrate schema v2 project state to v3.

    v3 adds browser-state persistence for user-selected dynamic metadata
    columns (``extra_columns``).
    """
    migrated = dict(data)
    migrated["schema_version"] = 3

    browser_state = dict(migrated.get("browser_state", {}))
    browser_state.setdefault("extra_columns", [])
    migrated["browser_state"] = browser_state
    return migrated


def _migrate_v3_to_v4(data: dict) -> dict:
    """Migrate schema v3 project state to v4.

    v4 adds optional per-axis alpha values (``alpha_x``, ``alpha_y``,
    ``alpha_z``) to vector-polarization grouping payloads.
    """
    migrated = dict(data)
    migrated["schema_version"] = 4

    datasets = migrated.get("datasets")
    if not isinstance(datasets, list):
        return migrated

    updated_datasets: list[dict] = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        ds = dict(item)
        grouping = ds.get("grouping_overrides")
        if isinstance(grouping, dict) and _is_vector_grouping_payload(grouping):
            ds["grouping_overrides"] = _migrate_grouping_alpha_fields(grouping)
        updated_datasets.append(ds)

    migrated["datasets"] = updated_datasets
    return migrated


def _migrate_v4_to_v5(data: dict) -> dict:
    """Migrate schema v4 project state to v5.

    v5 adds a separate frequency-domain fit state namespace so spectral fits do
    not overwrite time-domain single/global fit settings.
    """
    migrated = dict(data)
    migrated["schema_version"] = 5
    migrated.setdefault(
        "frequency_fit_state",
        {
            "domain": "frequency",
            "single_fit_state": {},
            "global_fit_state": {},
            "fit_ui_state": {},
        },
    )
    return migrated


def _migrate_v5_to_v6(data: dict) -> dict:
    """Migrate schema v5 project state to v6.

    v6 introduces first-class, recipe-only representations.  Each dataset gains
    a ``representations`` map and the project gains a top-level ``batches`` list.

    This migration is **additive and lossless**: the existing global
    ``single_fit_state`` / ``global_fit_state`` / ``multi_group_fit_state`` /
    ``frequency_fit_state`` / ``fourier_state`` blobs are preserved so projects
    still open on the pre-redesign code paths during the transition.

    Population is faithful to what v5 actually persisted:

    * per-run time single fits  → ``time_fb_asymmetry.fit``
    * per-run frequency single fits → ``freq_fft.fit``
    * the (global) ``fourier_state`` → ``freq_fft.recipe.fourier_config``

    Global/grouped fits are *not* reconstructed into batches: v5 never persisted
    their member runs or per-run results, so there is nothing faithful to carry
    over.  ``batches`` is therefore initialised empty.
    """
    migrated = dict(data)
    migrated["schema_version"] = 6

    datasets = migrated.get("datasets")
    if not isinstance(datasets, list):
        migrated.setdefault("batches", [])
        return migrated

    single_state = migrated.get("single_fit_state")
    frequency_state = migrated.get("frequency_fit_state")
    frequency_single_state = (
        frequency_state.get("single_fit_state") if isinstance(frequency_state, dict) else None
    )
    fourier_config = _fourier_recipe_from_state(migrated.get("fourier_state"))
    # Only runs that actually had a generated spectrum get an FFT recipe, so a
    # migrated project does not start recomputing FFTs for every run.
    spectra_runs = _runs_with_cached_spectra(migrated.get("fourier_spectra_state"))

    updated: list[dict] = []
    for entry in datasets:
        if not isinstance(entry, dict):
            updated.append(entry)
            continue
        ds = dict(entry)
        run_number = _coerce_int(ds.get("run_number"))
        representations: dict[str, dict] = {}

        time_fit = _single_state_to_fit_slot(
            _resolve_single_state_for_run(single_state, run_number)
        )
        if time_fit is not None:
            representations["time_fb_asymmetry"] = {
                "recipe": {},
                "fit": time_fit,
                "trend_state": {},
            }

        freq_fit = _single_state_to_fit_slot(
            _resolve_single_state_for_run(frequency_single_state, run_number)
        )
        has_spectrum = run_number in spectra_runs
        if freq_fit is not None or has_spectrum:
            representations["freq_fft"] = {
                "recipe": (
                    {"fourier_config": dict(fourier_config)}
                    if (fourier_config and has_spectrum)
                    else {}
                ),
                "fit": freq_fit if freq_fit is not None else _empty_fit_slot(),
                "trend_state": {},
            }

        if representations:
            ds["representations"] = representations
        updated.append(ds)

    migrated["datasets"] = updated
    migrated.setdefault("batches", [])
    return migrated


def _migrate_v6_to_v7(data: dict) -> dict:
    """Migrate schema v6 project state to v7.

    v7 generalises the v6 ``Batch`` into a ``FitSeries`` whose members may be
    runs *or* detector groups, and formalises each representation's
    ``trend_state``.  The migration is **additive and lossless**:

    * every top-level series (``batches``) gains ``member_kind`` (defaulting to
      ``"runs"``), an empty ``nuisance_params`` list, and an empty
      ``member_source_run`` map;
    * every representation's ``trend_state`` is normalised through
      :class:`~asymmetry.core.representation.trend_state.TrendState`, which wraps
      any unrecognised keys under ``legacy`` rather than dropping them.

    Existing v6 projects carry an empty ``batches`` list and empty trend states,
    so for them this is effectively just the version bump.
    """
    from asymmetry.core.representation.trend_state import TrendState

    migrated = dict(data)
    migrated["schema_version"] = 7

    series_list = migrated.get("batches")
    if isinstance(series_list, list):
        updated_series: list[dict] = []
        for series in series_list:
            if not isinstance(series, dict):
                updated_series.append(series)
                continue
            entry = dict(series)
            entry.setdefault("member_kind", "runs")
            entry.setdefault("nuisance_params", [])
            entry.setdefault("member_source_run", {})
            updated_series.append(entry)
        migrated["batches"] = updated_series

    datasets = migrated.get("datasets")
    if isinstance(datasets, list):
        updated_datasets: list[dict] = []
        for entry in datasets:
            if not isinstance(entry, dict):
                updated_datasets.append(entry)
                continue
            ds = dict(entry)
            reps = ds.get("representations")
            if isinstance(reps, dict):
                normalised: dict[str, dict] = {}
                for rep_key, rep in reps.items():
                    if not isinstance(rep, dict):
                        normalised[rep_key] = rep
                        continue
                    rep_copy = dict(rep)
                    rep_copy["trend_state"] = TrendState.from_dict(
                        rep_copy.get("trend_state")
                    ).to_dict()
                    normalised[rep_key] = rep_copy
                ds["representations"] = normalised
            updated_datasets.append(ds)
        migrated["datasets"] = updated_datasets

    return migrated


def _migrate_v7_to_v8(data: dict) -> dict:
    """Migrate schema v7 project state to v8.

    v8 adds a freeform ``extra`` dict to each top-level series (``FitSeries``),
    carrying per-series state such as the ALC scan's baseline regions, peaks, and
    view options. The migration is **additive and lossless**: every series gains
    an empty ``extra`` map.
    """
    migrated = dict(data)
    migrated["schema_version"] = 8

    series_list = migrated.get("batches")
    if isinstance(series_list, list):
        updated_series: list[dict] = []
        for series in series_list:
            if isinstance(series, dict):
                entry = dict(series)
                entry.setdefault("extra", {})
                updated_series.append(entry)
            else:
                updated_series.append(series)
        migrated["batches"] = updated_series

    return migrated


def _migrate_v8_to_v9(data: dict) -> dict:
    """Migrate schema v8 project state to v9.

    v9 adds optional per-projection fit slots to each representation
    (``projection_fits``), so each polarization projection of a vector grouping
    can keep its own fit. Pre-v9 projects simply carry no per-projection slots;
    the migration is a lossless version bump — ``representation_from_dict``
    already treats a missing ``projection_fits`` as "no per-projection fits".
    """
    migrated = dict(data)
    migrated["schema_version"] = 9
    return migrated


def _empty_fit_slot() -> dict:
    """Return a serialised empty :class:`FitSlot`."""
    return {
        "model": None,
        "parameters": [],
        "result": None,
        "provenance": "none",
        "batch_id": None,
        "diverged": False,
        "include_in_trend": True,
    }


def _single_state_to_fit_slot(state: object) -> dict | None:
    """Convert a v5 single-fit-tab state into a serialised :class:`FitSlot`.

    Returns ``None`` when *state* carries no model (nothing to migrate).
    """
    if not isinstance(state, dict):
        return None
    model = state.get("composite_model")
    if not isinstance(model, dict):
        return None
    raw_params = state.get("parameters")
    parameters = (
        [dict(p) for p in raw_params if isinstance(p, dict)] if isinstance(raw_params, list) else []
    )
    result_html = state.get("result_html")
    result = (
        {"result_html": result_html}
        if isinstance(result_html, str) and result_html.strip()
        else None
    )
    return {
        "model": dict(model),
        "parameters": parameters,
        "result": result,
        "provenance": "single",
        "batch_id": None,
        "diverged": False,
        "include_in_trend": True,
    }


def _resolve_single_state_for_run(single_state: object, run_number: int) -> dict | None:
    """Return the per-run single-fit state for *run_number* from a v5 blob."""
    if not isinstance(single_state, dict):
        return None
    states_by_run = single_state.get("states_by_run")
    if isinstance(states_by_run, dict):
        per_run = states_by_run.get(str(run_number))
        if isinstance(per_run, dict):
            return per_run
    # Fall back to the bare active state only for its own run.
    active = single_state.get("active_run_number")
    if (
        active is not None
        and _coerce_int(active) == run_number
        and isinstance(single_state.get("composite_model"), dict)
    ):
        return single_state
    return None


def _runs_with_cached_spectra(spectra_state: object) -> set[int]:
    """Return the run numbers that had a generated FFT spectrum in v5."""
    runs: set[int] = set()
    if isinstance(spectra_state, dict):
        for run_key in spectra_state:
            try:
                runs.add(int(run_key))
            except (TypeError, ValueError):
                continue
    return runs


def _fourier_recipe_from_state(fourier_state: object) -> dict:
    """Extract the recipe-relevant subset of a v5 ``fourier_state`` blob."""
    if not isinstance(fourier_state, dict):
        return {}
    return {key: fourier_state[key] for key in _FOURIER_RECIPE_KEYS if key in fourier_state}


def _coerce_int(value: object, default: int = 0) -> int:
    """Return *value* as int, or *default* if conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _migrate_grouping_alpha_fields(grouping: dict) -> dict:
    """Populate per-axis alpha fields from a scalar alpha value."""
    result = dict(grouping)
    alpha = _coerce_float(result.get("alpha", 1.0), 1.0)
    result["alpha_x"] = _coerce_float(result.get("alpha_x", result.get("alpha_px", alpha)), alpha)
    result["alpha_y"] = _coerce_float(result.get("alpha_y", result.get("alpha_py", alpha)), alpha)
    result["alpha_z"] = _coerce_float(result.get("alpha_z", result.get("alpha_pz", alpha)), alpha)
    return result


def _is_vector_grouping_payload(grouping: dict) -> bool:
    """Return True when grouping payload matches vector-polarization naming."""
    axis_token = str(grouping.get("vector_axis", "")).strip().lower().replace("_", "")
    if axis_token in {"px", "py", "pz"}:
        return True

    names_raw = grouping.get("group_names")
    if not isinstance(names_raw, dict):
        return False
    names = {str(v).strip().lower() for v in names_raw.values()}

    has_pz = "pz forward" in names and "pz backward" in names
    has_py = ("py top" in names and "py bottom" in names) or (
        "py up" in names and "py down" in names
    )
    has_px = "px left" in names and "px right" in names
    return has_pz and has_py and has_px


def _coerce_float(value: object, default: float) -> float:
    """Return *value* as float, or *default* if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
    raw = _decode_non_finite(json.loads(Path(path).read_text(encoding="utf-8")))
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
        json.dumps(_encode_non_finite(state), indent=2, default=_json_default),
        encoding="utf-8",
    )


#: Wrapper key used to round-trip non-finite floats through strict JSON.
#: ``NaN``/``Infinity``/``-Infinity`` are not valid JSON tokens, so a bare
#: ``json.dumps`` (allow_nan=True) writes them as non-standard barewords that
#: external strict parsers reject. We encode them as a uniquely-keyed object
#: ``{"__nonfinite__": "NaN"}`` on save and reconstruct the float on load, so
#: the file is valid strict JSON while in-app load behaviour is byte-for-byte
#: unchanged (every consumer still sees a real ``inf``/``-inf``/``nan`` float).
_NONFINITE_KEY = "__nonfinite__"


def _encode_non_finite(obj: object) -> object:
    """Recursively replace non-finite floats with a strict-JSON wrapper.

    Reversed by :func:`_decode_non_finite`. Also normalises numpy scalars/arrays
    to plain Python so the result is fully JSON-serialisable.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {_NONFINITE_KEY: "NaN"}
        if math.isinf(obj):
            return {_NONFINITE_KEY: "Infinity" if obj > 0 else "-Infinity"}
        return obj
    if isinstance(obj, dict):
        return {k: _encode_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_non_finite(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.floating):
            return _encode_non_finite(float(obj))
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return _encode_non_finite(obj.tolist())
    except ImportError:
        pass
    return obj


def _decode_non_finite(obj: object) -> object:
    """Recursively reconstruct non-finite floats from the save-time wrapper."""
    if isinstance(obj, dict):
        if len(obj) == 1 and _NONFINITE_KEY in obj:
            try:
                return float(obj[_NONFINITE_KEY])
            except (TypeError, ValueError):
                return obj
        return {k: _decode_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_non_finite(v) for v in obj]
    return obj


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
