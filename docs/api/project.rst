Project Persistence
===================

.. currentmodule:: asymmetry.core.project

The ``asymmetry.core.project`` module provides versioned JSON project files
(``.asymp``) that capture the full GUI session — loaded datasets, panel
settings, fit configuration, and Fourier settings — so you can close the
application and resume exactly where you left off.

Raw data arrays are **not** embedded in the project file; instead, source
file paths are stored and the files are reloaded on open.  This keeps project
files small and avoids duplicating data.

The schema version embedded in every ``.asymp`` file is independent of the
package version, so project files survive package upgrades as long as the
schema version is supported.

Public API
----------

.. autofunction:: load_project

.. autofunction:: save_project

.. autodata:: CURRENT_SCHEMA_VERSION

   The integer schema version written into every new project file.  Currently
   ``7``.

.. autoexception:: UnsupportedSchemaVersion

Schema internals
----------------

These functions are used internally by :func:`load_project` and
:func:`save_project` but may be useful for testing or migration scripts.

.. autofunction:: asymmetry.core.project.schema.validate

.. autofunction:: asymmetry.core.project.schema.migrate_to_current

Project file format
-------------------

A project file is a UTF-8 JSON document with the following top-level
structure::

    {
        "schema_version": 7,
        "created_with_app_version": "0.1.0",
        "datasets": [
            {
                "source_file": "/absolute/path/to/file.nxs",
                "run_number": 1234,
                "metadata_overrides": { "field": 50.0 },
                "representations": {
                    "time_fb_asymmetry": {
                        "recipe": {},
                        "fit": {
                            "model": { "component_names": ["Exponential"], "operators": [] },
                            "parameters": [...],
                            "result": { "success": true, "reduced_chi_squared": 0.97, ... },
                            "provenance": "single",
                            "batch_id": null,
                            "diverged": false,
                            "include_in_trend": true
                        },
                        "trend_state": {}
                    },
                    "freq_fft": {
                        "recipe": { "fourier_config": { "window": "gaussian", ... } },
                        "fit": { "model": null, "provenance": "none", ... },
                        "trend_state": {}
                    }
                }
            }
        ],
        "combined_datasets": [
            { "source_run_numbers": [1234, 1235] }
        ],
        "batches": [
            {
                "batch_id": "batch-1",
                "rep_type": "time_fb_asymmetry",
                "member_kind": "runs",
                "member_run_numbers": [1234, 1235, 1236],
                "member_source_run": {},
                "order_key": "field",
                "canonical_model": { "component_names": ["Exponential"], "operators": [] },
                "param_roles": { "A": "local", "Lambda": "global" },
                "nuisance_params": [],
                "results_by_run": {
                    "1234": { "success": true, "parameters": { "Lambda": 0.31 }, ... }
                },
                "diverged_runs": []
            }
        ],
        "browser_state": {
            "sort_column": 0,
            "sort_order": "ascending",
            "filters": {},
            "selected_run_numbers": [1234],
            "selected_group_ids": [],
            "data_groups": [],
            "extra_columns": []
        },
        "plot_state": { ... },
        "single_fit_state": {
            "model_name": "ExponentialRelaxation",
            "parameters": [...],
            "result_html": "<b>χ² = ...</b>"
        },
        "global_fit_state": {
            "model_name": "ExponentialRelaxation",
            "parameters": [...],
            "result_html": "<b>χ² = ...</b>"
        },
        "fit_ui_state": { "active_tab_index": 0 },
        "fit_parameters_state": { "rows": [...], "x_axis": "Auto", ... },
        "fourier_state": {
            "window": "gaussian",
            "filter_start_us": 0.0,
            "filter_time_constant_us": 1.5,
            "padding": 4,
            "display": "Power"
        }
    }

**Schema versioning history:**

* v1–v4 — initial releases
* v5 — grouping overrides, wizard cache
* v6 — per-dataset ``representations`` map (recipe-only FFT, per-run ``FitSlot``,
  trend state); top-level ``batches`` list
* v7 — ``FitSeries`` gains ``member_kind``, ``nuisance_params``,
  ``member_source_run``; ``trend_state`` normalized to structured shape;
  group series recorded in ``batches``

Unknown top-level keys are silently preserved during migration, ensuring
forward compatibility when a newer version of the application adds new state.

Representation
--------------

Each dataset entry's ``representations`` map holds one entry per analysis
view the user has exercised.  Keys are ``RepresentationType`` values:
``"time_fb_asymmetry"``, ``"time_groups"``, ``"freq_fft"``, ``"freq_maxent"``.

Each representation stores:

``recipe``
    Generation parameters (for FFT spectra: the ``GroupSpectrumConfig``
    serialisation; for time-domain views: empty).  On project load the
    transient arrays are recomputed from the recipe rather than stored.

``fit``
    A single :class:`~asymmetry.core.representation.base.FitSlot` — the most
    recent fit for this (dataset, representation) pair.  Includes the model
    dict, fitted-parameter list, result summary, provenance marker
    (``"none"``, ``"single"``, ``"batch"``, ``"global"``), the ``batch_id`` of
    the owning :class:`~asymmetry.core.representation.series.FitSeries` when
    the fit was part of a series, and divergence flags used by the trending
    panel.

``trend_state``
    Opaque dict persisting the user's x-axis and y-parameter selections in the
    Fit Parameters panel for this representation.

FitSeries (``batches``)
-----------------------

The top-level ``batches`` list stores
:class:`~asymmetry.core.representation.series.FitSeries` objects that
accumulate results across multiple members.

Key fields:

``member_kind``
    ``"runs"`` for F-B asymmetry / FFT batch fits; ``"groups"`` for
    multi-run grouped time-domain fits.

``member_run_numbers``
    Ordered list of member keys.  For ``"runs"`` series these are real run
    numbers; for ``"groups"`` series they are synthetic negative keys of the
    form ``-(source_run * 1000 + group_index)``.

``member_source_run``
    Map from synthetic group key → source run number (``"groups"`` series only).

``param_roles``
    Per-parameter classification: ``"global"``, ``"local"``, or ``"fixed"``.

``nuisance_params``
    List of per-(run, group) nuisance parameter names excluded from trending
    (``"groups"`` series only).

``results_by_run``
    Per-member fit result summaries (parameter values, uncertainties, χ²)
    used to drive the Fit Parameters trending panel.
