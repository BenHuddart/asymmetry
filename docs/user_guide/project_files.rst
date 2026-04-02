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

For two-period NeXus runs, grouping metadata persisted with each dataset also
includes red/green period configuration such as ``period_mode`` and per-period
histogram metadata used by RG recomputation.

Project files do not embed raw detector arrays.

Grouping Overrides
------------------

Each dataset entry in the project JSON may contain a ``grouping_overrides``
block that stores the full custom grouping applied in the current session.
When present, it is re-applied to the dataset on project load so that the
exact detector grouping is preserved without re-running the Grouping dialog.

``grouping_overrides`` keys:

``groups``
    Mapping of group-slot index → list of detector channel indices.

``group_names``
    Mapping of group-slot index → display name string (e.g. ``"Pz Forward"``).

``forward_group``
    Group-slot index used as the forward detector sum.

``backward_group``
    Group-slot index used as the backward detector sum.

``alpha``
    Detector balance factor.

``first_good_bin``
    Integer index of the first bin included in asymmetry computation.

``last_good_bin``
    Integer index of the last bin included in asymmetry computation.

``bunching_factor``
    Time-bin rebunching factor applied to the dataset.

``deadtime_correction``
    Boolean; whether per-detector deadtime correction is active.

``grouping_preset``
    Name of the last applied preset (e.g. ``"Vector Polarization"``).

``instrument``
    Instrument name set in the Detector Layout Editor (e.g. ``"EMU"``).
    Saved so the correct detector schematic is shown on reopen without
    re-running instrument detection.

``vector_axis``
    Active polarization axis at save time (``P_x``, ``P_y``, ``P_z``, or ``ALL``).

``period_mode``
    Two-period RG mode (``Red``, ``Green``, ``G minus R``, ``G plus R``).


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
