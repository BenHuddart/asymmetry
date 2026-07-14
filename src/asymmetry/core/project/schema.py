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

Current schema (version 15)
---------------------------

Version 15 unifies ``DataGroup`` and ``FitSeries`` (D1/D7/D9). Each top-level
``data_groups`` entry gains ``kind`` (``"user"``/``"auto"``, default
``"user"``). Each run-membered series in ``batches`` gains a structural
``group_id`` (resolved from its legacy ``source_group_id`` when that id names a
live group in the project's ``data_groups`` block, else ``null`` — a *frozen*
legacy analysis; old projects must not sprout groups the user never made),
``excluded_run_numbers`` (default empty), and ``last_fitted_members`` (seeded
from ``member_run_numbers`` so a loaded series is not spuriously stale).
Detector-group series (``member_kind == "groups"``) get only the additive
defaults and are never group-resolved. Additive and tolerant: a project with
no ``data_groups`` block, no ``batches``, or junk entries migrates cleanly. See
:func:`_migrate_v14_to_v15`.

Version 14 adds a per-plot-panel ``waterfall`` block — ``{"enabled": bool,
"offset": float | null}`` (null offset = automatic spacing) — inside
``plot_state`` and its nested ``frequency_plot_state``, recording the
single-axis overlay waterfall stack. Additive: absent/default (disabled, auto)
in older files; see :func:`_migrate_v13_to_v14`.

Version 13 adds a top-level ``global_fit_studies`` list: each entry is a
serialized :class:`~asymmetry.core.representation.global_fit_study.GlobalFitStudy`
(the persisted, named cross-group global parameter fit — "study" — that
replaces the single-slot ``last_cross_group_fit`` the trend panel used to
keep in its own panel state). Absent/empty in older files, mirroring the
pattern used for other optional top-level lists (e.g. ``grouping_profiles``
below); see :func:`_migrate_v12_to_v13`.

Version 12 adds project-level named **grouping profiles**. The project carries a
top-level ``grouping_profiles`` list (serialized
:class:`~asymmetry.core.project.profiles.GroupingProfile` dicts, each with
``active: true`` for its fingerprint). Each dataset either names a ``profile``
(inheriting that profile's settings), carries a per-run ``grouping_overrides``
payload (released from any profile, as before), or has neither (inheriting the
active profile for its fingerprint). See
:func:`_migrate_v11_to_v12` for the v11->v12 collapse rules and
:mod:`asymmetry.core.project.profiles` for resolution.

Version 11 schema
-----------------
::

    {
        "schema_version": 1,
        "created_with_app_version": "0.1.0",
        "datasets": [
            {
                "run_number": 3077,
                "source_file": "/abs/path/to/file.nxs",
                "metadata_overrides": {
                    "field": 150.0,
                    "custom_fields": {"custom:ab12cd34": "annealed"}
                }
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
            # GUI display state (name/members/collapsed) for the data browser's
            # group headers. Distinct from the top-level "data_groups" block
            # (Phase 7/D1, not shown in this pre-v6 illustrative example): that
            # one is the core DataGroup registry (no "collapsed"), mirrored from
            # this block at load/save so FitSeries.source_group_id provenance
            # can resolve without a GUI dependency.
            "data_groups": [],
            "extra_columns": [
                {"id": "nexus_fields.sample.shape", "label": "Orientation",
                 "kind": "metadata", "source_key": "nexus_fields.sample.shape"},
                {"id": "custom:ab12cd34", "label": "Anneal", "kind": "custom"}
            ]
        },
        "plot_state": {
            "current_run_number": 3077,
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
            "waterfall": {"enabled": false, "offset": null},
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
                "frequency_axis_mode": "absolute",
                "frequency_reference_mode": "run",
                "frequency_x_limits_by_unit": {},
                "waterfall": {"enabled": false, "offset": null}
            },
            "fit_curve": null,
            "fit_curves": {}
        },
        "fit_states": {
            "time": {
                "domain": "time",
                "single_fit_state": {
                    "model_name": "ExponentialRelaxation",
                    "parameters": [
                        {"name": "A0", "value": 0.2, "fixed": false,
                         "min": "-inf", "max": "inf"}
                    ]
                },
                "global_fit_state": {
                    "model_name": "ExponentialRelaxation",
                    "parameters": [
                        {"name": "A0", "value": 0.2, "type": "Global",
                         "bounds": "-inf, inf"}
                    ]
                },
                "fit_ui_state": {}
            },
            "frequency": {
                "domain": "frequency",
                "single_fit_state": {},
                "global_fit_state": {},
                "fit_ui_state": {}
            }
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
        }
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CURRENT_SCHEMA_VERSION: int = 16

_SUPPORTED_VERSIONS: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
)

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
        version = 9
    if version == 9:
        migrated = _migrate_v9_to_v10(migrated)
        version = 10
    if version == 10:
        migrated = _migrate_v10_to_v11(migrated)
        version = 11
    if version == 11:
        migrated = _migrate_v11_to_v12(migrated)
        version = 12
    if version == 12:
        migrated = _migrate_v12_to_v13(migrated)
        version = 13
    if version == 13:
        migrated = _migrate_v13_to_v14(migrated)
        version = 14
    if version == 14:
        migrated = _migrate_v14_to_v15(migrated)
        version = 15
    if version == 15:
        migrated = _migrate_v15_to_v16(migrated)
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
    migrated.setdefault("frequency_fit_state", _domain_fit_state("frequency", None))
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


def _migrate_v9_to_v10(data: dict) -> dict:
    """Migrate schema v9 project state to v10.

    v10 generalises the data browser's ``browser_state.extra_columns`` from a bare
    list of metadata keys (strings) into a list of column definitions
    ``{"id", "label", "kind", "source_key"}`` so the browser can carry both
    derived *metadata* columns and user-editable *custom* columns, and so any
    metadata column can be renamed while retaining its underlying source key. Each
    legacy string promotes to a ``metadata`` column whose id/source_key is that
    key; the gui-facing display label is resolved by the panel on load. Custom
    column *values* live per-run in each dataset's ``metadata_overrides`` (under
    ``custom_fields``) and need no migration here.
    """
    migrated = dict(data)
    migrated["schema_version"] = 10

    browser_state = dict(migrated.get("browser_state", {}))
    raw_columns = browser_state.get("extra_columns")
    if isinstance(raw_columns, list):
        upgraded: list[dict] = []
        for entry in raw_columns:
            if isinstance(entry, str):
                key = entry.strip()
                if key:
                    upgraded.append(
                        {"id": key, "label": key, "kind": "metadata", "source_key": key}
                    )
            elif isinstance(entry, dict):
                upgraded.append(entry)
        browser_state["extra_columns"] = upgraded
        migrated["browser_state"] = browser_state

    return migrated


def _migrate_v11_to_v12(data: dict) -> dict:
    """Migrate schema v11 project state to v12.

    v12 introduces project-level named grouping profiles
    (:class:`~asymmetry.core.project.profiles.GroupingProfile`). The migration
    groups datasets by *fingerprint* ``(instrument, histogram count)`` derived
    from each dataset's stored ``grouping_overrides``, then:

    * **Missing metadata** — if a dataset's ``grouping_overrides`` lacks the
      instrument name or a usable histogram count, it is migrated conservatively:
      the ``grouping_overrides`` payload is kept as-is and it joins no profile, so
      resolution falls back to the per-run payload exactly as in v11.
    * **All-identical collapse** — within a fingerprint, if every run's shareable
      grouping fields are identical, one active profile ``"Default (<instrument>)"``
      is created and each contributing dataset drops its ``grouping_overrides`` in
      favour of ``profile: "<name>"``.
    * **Divergence** — if the shareable fields differ, a profile is built from the
      **majority** payload (ties resolved by first occurrence); datasets matching
      the majority inherit it (``profile`` set, overrides dropped) while divergent
      datasets keep their ``grouping_overrides``.

    The migration is additive and lossless: nothing that could not be faithfully
    lifted into a profile is dropped — divergent and metadata-poor datasets retain
    their full per-run payload.
    """
    from asymmetry.core.project.profiles import (
        ProfileFingerprint,
        profile_from_payload,
    )

    migrated = dict(data)
    migrated["schema_version"] = 12
    migrated.setdefault("grouping_profiles", [])

    datasets = migrated.get("datasets")
    if not isinstance(datasets, list):
        return migrated

    # Bucket dataset indices by fingerprint. Datasets whose overrides lack the
    # metadata needed to fingerprint them are left untouched (conservative path).
    buckets: dict[tuple[str, int], list[int]] = {}
    fingerprints: dict[tuple[str, int], ProfileFingerprint] = {}
    for index, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            continue
        overrides = entry.get("grouping_overrides")
        if not isinstance(overrides, dict):
            continue
        instrument = str(overrides.get("instrument", "") or "").strip()
        histogram_count = _histogram_count_from_overrides(overrides)
        if not instrument or histogram_count is None:
            continue  # Conservative: keep grouping_overrides, no profile.
        key = (instrument.lower(), int(histogram_count))
        buckets.setdefault(key, []).append(index)
        fingerprints.setdefault(key, ProfileFingerprint(instrument, int(histogram_count)))

    updated = [dict(entry) if isinstance(entry, dict) else entry for entry in datasets]
    profiles: list[dict] = list(migrated.get("grouping_profiles") or [])

    for key, indices in buckets.items():
        fingerprint = fingerprints[key]
        payloads = [updated[i].get("grouping_overrides") for i in indices]
        majority_payload = _majority_payload(payloads)
        if majority_payload is None:
            continue
        profile_name = _unique_profile_name(
            f"Default ({fingerprint.instrument})", {p.get("name") for p in profiles}
        )
        profile = profile_from_payload(majority_payload, profile_name, fingerprint, active=True)
        profiles.append(profile.to_dict())

        majority_signature = _shareable_signature(majority_payload)
        for i in indices:
            payload = updated[i].get("grouping_overrides")
            if _shareable_signature(payload) == majority_signature:
                # Inherits the profile: swap the per-run copy for a reference.
                updated[i].pop("grouping_overrides", None)
                updated[i]["profile"] = profile_name
            # Divergent datasets keep their grouping_overrides unchanged.

    migrated["datasets"] = updated
    migrated["grouping_profiles"] = profiles
    return migrated


def _migrate_v12_to_v13(data: dict) -> dict:
    """Migrate schema v12 project state to v13.

    v13 adds a top-level ``global_fit_studies`` list — the persisted registry
    of named cross-group global parameter fits (see
    :class:`asymmetry.core.representation.global_fit_study.GlobalFitStudy`).
    This is purely additive: pre-v13 projects carried at most one such fit,
    single-slot, inside the trend panel's own serialized panel state (a key
    this migration does not touch); it is left for the GUI-side migration
    (a later phase) to lift into a study on first load, so this step is just
    the version bump plus an empty default for the new list.
    """
    migrated = dict(data)
    migrated["schema_version"] = 13
    migrated.setdefault("global_fit_studies", [])
    return migrated


def _migrate_v13_to_v14(data: dict) -> dict:
    """Migrate schema v13 project state to v14.

    v14 adds a per-plot-panel ``waterfall`` block ``{"enabled": bool, "offset":
    float | null}`` (null offset = automatic spacing) recording the single-axis
    overlay waterfall stack. It lives inside ``plot_state`` and its nested
    ``frequency_plot_state``. This is purely additive: pre-v14 projects had no
    such control, so the default is disabled/auto, matching what the GUI's plot
    panel restores when the key is absent.
    """
    migrated = dict(data)
    migrated["schema_version"] = 14
    default = {"enabled": False, "offset": None}
    plot_state = migrated.get("plot_state")
    if isinstance(plot_state, dict):
        plot_state = dict(plot_state)
        plot_state.setdefault("waterfall", dict(default))
        freq_state = plot_state.get("frequency_plot_state")
        if isinstance(freq_state, dict):
            freq_state = dict(freq_state)
            freq_state.setdefault("waterfall", dict(default))
            plot_state["frequency_plot_state"] = freq_state
        migrated["plot_state"] = plot_state
    return migrated


def _migrate_v14_to_v15(data: dict) -> dict:
    """Migrate schema v14 project state to v15.

    v15 unifies ``DataGroup`` and ``FitSeries`` (D1/D7/D9). The migration is
    additive and tolerant of every legacy shape found in the wild (pre-Phase-7
    saves with no ``data_groups`` block, projects with no ``batches``, junk
    entries):

    * Each ``data_groups`` entry gains ``kind`` (default ``"user"``).
    * Each run-membered series (``member_kind == "runs"``) gains:

      - ``group_id`` — set to its legacy ``source_group_id`` **iff** that id
        names a live group in this project's ``data_groups`` block, else
        ``None``. A group-less series migrates to a *frozen legacy* analysis
        (``group_id`` ``None``); old projects must not sprout groups the user
        never made (D9). A later re-run creates its auto-group then.
      - ``excluded_run_numbers`` — default empty.
      - ``last_fitted_members`` — a copy of ``member_run_numbers`` (so a loaded
        series is not spuriously stale).

    * Detector-group series (``member_kind == "groups"``, D8) get only the
      additive defaults (``group_id=None``, empty exclusions, ``last_fitted_members``
      copy) — never a group resolution.
    """
    migrated = dict(data)
    migrated["schema_version"] = 15

    # Collect the live group ids first so series can resolve their group link.
    known_group_ids: set[str] = set()
    groups = migrated.get("data_groups")
    if isinstance(groups, list):
        updated_groups: list = []
        for group in groups:
            if isinstance(group, dict):
                entry = dict(group)
                entry.setdefault("kind", "user")
                gid = entry.get("group_id")
                if gid is not None:
                    known_group_ids.add(str(gid))
                updated_groups.append(entry)
            else:
                updated_groups.append(group)
        migrated["data_groups"] = updated_groups

    batches = migrated.get("batches")
    if isinstance(batches, list):
        updated_batches: list = []
        for series in batches:
            if not isinstance(series, dict):
                updated_batches.append(series)
                continue
            entry = dict(series)
            member_kind = entry.get("member_kind", "runs")
            members = entry.get("member_run_numbers")
            members = list(members) if isinstance(members, list) else []
            if "group_id" not in entry:
                source_group_id = entry.get("source_group_id")
                if (
                    member_kind == "runs"
                    and source_group_id is not None
                    and str(source_group_id) in known_group_ids
                ):
                    entry["group_id"] = str(source_group_id)
                else:
                    entry["group_id"] = None
            entry.setdefault("excluded_run_numbers", [])
            if "last_fitted_members" not in entry:
                entry["last_fitted_members"] = members
            updated_batches.append(entry)
        migrated["batches"] = updated_batches

    return migrated


def _migrate_v15_to_v16(data: dict) -> dict:
    """Migrate schema v15 project state to v16.

    v16 replaces the frequency panel's boolean ``frequency_axis_relative_to_
    reference`` (an offset applied only to the x-limit boxes, never to the
    plotted data) with a real ``frequency_axis_mode`` transform
    (``"absolute"``/``"shift"``/``"relative_ppm"``) plus a
    ``frequency_reference_mode`` (``"run"``/``"common"``). A pre-v16 project that
    had the old flag ON maps to ``"shift"`` about a ``"common"`` reference — the
    closest match to the retired behaviour, where a single panel reference was
    applied. The flag OFF (or absent) maps to plain ``"absolute"``. The retired
    ``:relative`` per-mode x-limit stash keys are ephemeral view state and are
    dropped; new limits reframe cleanly. Lives inside ``plot_state`` and its
    nested ``frequency_plot_state``.
    """
    migrated = dict(data)
    migrated["schema_version"] = 16

    def _upgrade_freq_state(freq_state: dict) -> dict:
        freq_state = dict(freq_state)
        if "frequency_axis_mode" not in freq_state:
            legacy_relative = bool(freq_state.get("frequency_axis_relative_to_reference", False))
            freq_state["frequency_axis_mode"] = "shift" if legacy_relative else "absolute"
            freq_state.setdefault(
                "frequency_reference_mode", "common" if legacy_relative else "run"
            )
        freq_state.pop("frequency_axis_relative_to_reference", None)
        stash = freq_state.get("frequency_x_limits_by_unit")
        if isinstance(stash, dict):
            freq_state["frequency_x_limits_by_unit"] = {
                key: value for key, value in stash.items() if not str(key).endswith(":relative")
            }
        return freq_state

    plot_state = migrated.get("plot_state")
    if isinstance(plot_state, dict):
        plot_state = dict(plot_state)
        freq_state = plot_state.get("frequency_plot_state")
        if isinstance(freq_state, dict):
            plot_state["frequency_plot_state"] = _upgrade_freq_state(freq_state)
        migrated["plot_state"] = plot_state

    return migrated


def _histogram_count_from_overrides(overrides: dict) -> int | None:
    """Infer a run's histogram count from a stored grouping payload, or ``None``.

    Prefers an explicit per-detector list length (``detector_t0_bins``,
    ``detector_first_good_bins``, ``histogram_labels`` — stored by the PSI/ROOT
    loaders); falls back to the largest 1-based detector id named across the
    groups (a lower bound that still fingerprints NeXus payloads consistently).
    Returns ``None`` when neither is available.
    """
    for key in ("detector_t0_bins", "detector_first_good_bins", "histogram_labels"):
        value = overrides.get(key)
        if isinstance(value, (list, tuple)) and value:
            return len(value)
    groups = overrides.get("groups")
    if isinstance(groups, dict):
        max_det = 0
        for dets in groups.values():
            if not isinstance(dets, (list, tuple)):
                continue
            for entry in dets:
                raw = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
                try:
                    max_det = max(max_det, int(raw))
                except (TypeError, ValueError):
                    continue
        if max_det > 0:
            return max_det
    return None


#: Shareable grouping payload keys that define a profile's identity for the
#: v11->v12 collapse. Two payloads with the same signature over these keys yield
#: the same profile; per-run/file-derived keys are deliberately excluded.
_SHAREABLE_SIGNATURE_KEYS = (
    "groups",
    "group_names",
    "included_groups",
    "forward_group",
    "backward_group",
    "projections",
    "vector_axis",
    "excluded_detectors",
    "alpha",
    "alpha_x",
    "alpha_y",
    "alpha_z",
    "alpha_method",
    "grouping_preset",
    "binning_mode",
    "bin0_us",
    "bin10_us",
    "bunching_factor",
    "period_mode",
    "deadtime_correction",
    "deadtime_mode",
    "deadtime_manual_us",
    "deadtime_estimated_us",
    "dead_time_us",
    "background_correction",
    "background_mode",
    "background_fixed_values",
)


def _shareable_signature(payload: object) -> str:
    """Return a stable string signature over a payload's shareable keys.

    Only the file-format-independent (shareable) keys participate, so two runs
    that differ solely in per-run facts (t0, good-bin window, per-detector
    tables) collapse to the same profile.
    """
    if not isinstance(payload, dict):
        return ""
    subset = {k: payload[k] for k in _SHAREABLE_SIGNATURE_KEYS if k in payload}
    try:
        return json.dumps(subset, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(sorted(subset.items(), key=lambda kv: kv[0]))


def _majority_payload(payloads: list[object]) -> dict | None:
    """Return the payload with the most common shareable signature.

    Ties are resolved by first occurrence (the earliest run wins), matching the
    migration spec. Returns ``None`` when no dict payloads are present.
    """
    dict_payloads = [p for p in payloads if isinstance(p, dict)]
    if not dict_payloads:
        return None
    counts: dict[str, int] = {}
    first_for_signature: dict[str, dict] = {}
    order: list[str] = []
    for payload in dict_payloads:
        signature = _shareable_signature(payload)
        if signature not in counts:
            counts[signature] = 0
            first_for_signature[signature] = payload
            order.append(signature)
        counts[signature] += 1
    # Highest count, ties broken by first appearance (stable over ``order``).
    best_signature = max(order, key=lambda sig: counts[sig])
    return first_for_signature[best_signature]


def _unique_profile_name(base: str, existing: set) -> str:
    """Return *base*, suffixed with ``" (2)"``, ``" (3)"``… to stay unique."""
    names = {str(n) for n in existing if n}
    if base not in names:
        return base
    index = 2
    while f"{base} ({index})" in names:
        index += 1
    return f"{base} ({index})"


def _domain_fit_state(domain: str, source: object) -> dict:
    """Return a per-domain fit-state block, defaulting missing sub-keys to ``{}``.

    ``source`` is the legacy blob the block is folded from: the whole project
    dict for the time domain (whose fit keys sit at the top level) or the nested
    ``frequency_fit_state`` for frequency. Pass ``None`` for a canonical empty
    block. Single source of the ``{domain, single_fit_state, global_fit_state,
    fit_ui_state}`` shape used across migrations.
    """
    src = source if isinstance(source, dict) else {}
    return {
        "domain": domain,
        "single_fit_state": src.get("single_fit_state") or {},
        "global_fit_state": src.get("global_fit_state") or {},
        "fit_ui_state": src.get("fit_ui_state") or {},
    }


def _migrate_v10_to_v11(data: dict) -> dict:
    """Migrate schema v10 project state to v11.

    v11 consolidates the fit-panel state into a single representation-keyed
    ``fit_states`` block so the time- and frequency-domain fit forms round-trip
    symmetrically.  Earlier schemas stored the time-domain single/global/UI fit
    state as *un-keyed* top-level keys (``single_fit_state``,
    ``global_fit_state``, ``fit_ui_state``) while the frequency domain nested
    under ``frequency_fit_state`` -- an asymmetry that let a frequency fit form
    bleed into the time-domain form on restore (F21c).

    This migration folds the legacy keys into
    ``fit_states = {"time": ..., "frequency": ...}`` and drops the legacy
    top-level copies.  Each per-domain block carries a ``domain`` tag so a stale
    blob can never be applied to the wrong domain.  ``multi_group_fit_state`` is
    unrelated to the per-domain fit forms and is left at the top level.
    """
    migrated = dict(data)
    migrated["schema_version"] = 11

    # Time-domain fit keys sit at the top level; frequency nests under
    # ``frequency_fit_state``.
    migrated["fit_states"] = {
        "time": _domain_fit_state("time", migrated),
        "frequency": _domain_fit_state("frequency", migrated.get("frequency_fit_state")),
    }

    for legacy_key in (
        "single_fit_state",
        "global_fit_state",
        "fit_ui_state",
        "frequency_fit_state",
    ):
        migrated.pop(legacy_key, None)

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
