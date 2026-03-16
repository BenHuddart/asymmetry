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
    ``3``.

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
        "schema_version": 3,
        "created_with_app_version": "0.1.0",
        "datasets": [
            {
                "source_file": "/absolute/path/to/file.wim",
                "run_number": 1234,
                "metadata_overrides": { "field": 50.0 }
            }
        ],
        "combined_datasets": [
            { "source_run_numbers": [1234, 1235] }
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
            "window": "Hanning",
            "padding": 4,
            "display": "Power"
        }
    }

Unknown top-level keys are silently preserved during migration, ensuring
forward compatibility when a newer version of the application adds new state.
