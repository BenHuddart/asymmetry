Project Files
=============

Asymmetry project files (``.asymp``) persist analysis state across sessions.

What Is Stored
--------------

Project files store:

* Loaded dataset references (source file paths)
* Browser state (sorting, filters, selected runs, dynamic columns)
* Plot state (ranges, selected run, bunch factor)
* Fit-panel and Fourier-panel state

Project files do not embed raw detector arrays.

Save and Load
-------------

.. code-block:: python

   from asymmetry.core.project.schema import load_project, save_project

   state = {
       "schema_version": 3,
       "created_with_app_version": "0.1.0",
       "datasets": [{"run_number": 3077, "source_file": "run3077.wim", "metadata_overrides": {}}],
       "browser_state": {
           "sort_column": 0,
           "sort_order": "ascending",
           "filters": {},
           "selected_run_numbers": [3077],
           "selected_group_ids": [],
           "data_groups": [],
           "extra_columns": [],
       },
   }

   save_project(state, "session.asymp")
   restored = load_project("session.asymp")
   print(restored["schema_version"])

Schema Migration
----------------

Use ``migrate_to_current`` to normalize older project files in scripts or tests.

.. code-block:: python

   from asymmetry.core.project.schema import migrate_to_current

   old_state = {
       "schema_version": 2,
       "datasets": [],
       "browser_state": {
           "filters": {},
           "selected_run_numbers": [],
           "selected_group_ids": [],
           "data_groups": [],
       },
   }

   migrated = migrate_to_current(old_state)
   print(migrated["schema_version"])  # current schema

Runnable Example
----------------

See ``examples/project_files.py`` for a complete executable script.
