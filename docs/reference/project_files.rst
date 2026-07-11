Project files
=============

The ``.asymp`` project file persists the analysis state of a session:
which datasets are loaded, how the Data Browser is sorted and filtered,
the per-run grouping (groups, alpha, bunching, deadtime, background),
the single- and global-fit model setups with their parameter tables and
bounds, separate frequency-domain fit state with spectral peak models,
the most recent fit overlays, the Fourier panel state including
per-run phase tables, any cached Fit Wizard or Global Fit Wizard
analyses, per-run *representation* fit slots (single and series fits
recorded into the domain representation model), and the *batches* (fit
series) that drive the Fit Parameters trending panel. Raw detector
arrays are *not* embedded — the file references source data by path and
reloads from disk on open. Fourier spectra are regenerated from their
stored recipe (window, padding, phase, group selection) rather than
embedded, so the file remains compact even after frequency-domain work.
This makes ``.asymp`` files small enough to share alongside the raw
data when sending an analysis to a collaborator, or to archive alongside
paper supplementary material so that readers can reproduce every fit
shown in the figures.

Asymmetry's project file is JSON with an integer schema version, which
is independent of the package version: opening a project written by an
older release triggers automatic schema migration, and the loader
refuses to silently accept a file written by a future schema it does
not understand. The format is therefore directly diffable in version
control, which is the main practical difference from musrfit's
hand-editable ``.msr`` text format (the import of ``.msr`` into
``.asymp`` is a roadmap candidate; see :doc:`/explanation/comparison`)
and from Mantid's binary HDF5 ``.mantid`` files.

What is stored
--------------

Project files store:

* Loaded dataset references (source file paths)
* Browser state (sorting, filters, selected runs, dynamic columns)
* Plot state (ranges, selected run, bunch factor, overlay mode, and waterfall
  stacking — see :ref:`waterfall stacking <waterfall-stacking>`)
* Fit-panel and Fourier-panel state
* Separate frequency-fit state for displayed Fourier spectra
* Per-run Fourier group-phase tables, included groups, and auto-estimated
  phase markers
* Cached single-fit and global-fit wizard analysis payloads when present
* **Per-dataset representations** — for each analysis domain (F-B asymmetry,
  detector groups, FFT, MaxEnt) that the user has exercised, the stored
  representation records a *recipe* (for FFT: the generation config) and a
  *FitSlot* (the most recent fit's model, parameters, result summary,
  provenance, and trending flags). Fourier spectra are re-generated from the
  recipe on load; time-domain asymmetry is re-computed from the raw data.
* **Fit series (batches)** — each batch or global fit over multiple runs (or
  multiple runs' detector groups) is recorded as a ``FitSeries`` that carries
  the member list, parameter roles, per-member result summaries, and divergence
  state. The Fit Parameters trending panel reads directly from these series,
  organised by the active representation.
* **Data groups** — the top-level ``data_groups`` registry is the canonical
  store of named run collections (:doc:`the Data Browser's groups
  <gui_usage>`). A run-membered ``FitSeries`` that was launched from a group
  (or auto-created for an ad-hoc batch selection) carries a structural
  ``group_id`` back to its owning group; see `Data groups and fit series`_
  below for the full field set this adds to both ``data_groups`` entries and
  ``batches`` entries.

.. _data-groups-and-fit-series:

Data groups and fit series
---------------------------

A group owns zero or more series: the same run collection can be fit with
several models side by side, and a series' *effective* membership is derived
live as the owning group's members minus that series' own exclusions, so
adding or removing a run from the group is reflected the next time the series
is re-run rather than requiring a fresh batch fit. Each top-level
``data_groups`` entry stores:

``kind``
    ``"user"`` for a group the user named explicitly, or ``"auto"`` for one
    minted automatically the first time an ad-hoc run selection is batch- or
    global-fitted (so every batch fit has an explicit owning group). Renaming
    an ``"auto"`` group promotes it to ``"user"``.

Each run-membered (``member_kind == "runs"``) entry in ``batches`` gains:

``group_id``
    The id of the ``data_groups`` entry that owns this series, or ``null`` for
    a **frozen** series — a legacy analysis, or one whose owning group has
    since been deleted with its fits kept (rather than deleted with the
    group). A frozen series' membership is a fixed snapshot, exactly the
    pre-v15 behaviour.

``excluded_run_numbers``
    Run numbers the user has dropped from *this* series without removing them
    from the owning group — for example a member run whose data turned out to
    be unusable, kept in the group for record-keeping but excluded from the
    fit. Effective membership is the owning group's ``member_run_numbers``
    minus this list.

``last_fitted_members``
    A snapshot of the members that were actually fit the last time this
    series ran. When the group's live membership (minus exclusions) no longer
    matches this snapshot, the series is *stale* — the Fit Parameters panel
    marks its trend pill with a ``⚠`` and a tooltip reading "Membership
    changed since last fit — re-run to refresh."; re-running the series
    updates the snapshot and clears the marker. Detector-group series
    (``member_kind == "groups"``) and frozen series are never stale.

The Data Browser's ``browser_state.data_groups`` block is a separate, smaller
structure: it is *view* state (the panel's own per-group ``collapsed`` flag)
rather than the group registry itself, and doubles as a self-contained
fallback for a standalone browser panel or a pre-registry project — the
top-level ``data_groups`` list is always the source of truth when both are
present. See :doc:`gui_usage` for the Data Browser's grouping UI (multi-group
membership, the auto/user colour distinction, and the "Fit this group…"
binding) and :doc:`parameter_trending` for how a stale series surfaces in the
trending panel.

For two-period NeXus runs, grouping metadata persisted with each dataset also
includes red/green period configuration such as ``period_mode`` and per-period
histogram metadata used by RG recomputation.

Project files do not embed raw detector arrays or computed Fourier spectra.

Wizard cache state
------------------

The single-fit and global-fit panels may each include an optional
``wizard_state`` block inside their saved UI state.

For single-fit runs this stores:

* the cached recommendation payload for the run
* the signature used to decide whether the cached result can be reused
* the wizard log text shown in the analysis log window

For the global-fit tab this stores the cached global recommendation together
with the run-set, parameter-role, value, and bounds signature that produced it.

These blocks are optional and backward-compatible. Older project files do not
need them, but when present they allow wizard results to reopen immediately
without rerunning the expensive analysis.

Grouping overrides
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
    Calibration constant α.

``first_good_bin``
    Integer index of the first bin included in asymmetry computation.

``last_good_bin``
    Integer index of the last bin included in asymmetry computation.

``bunching_factor``
    Time-bin rebunching factor applied to the dataset.

``deadtime_correction``
    Boolean; whether per-detector deadtime correction is active.

``deadtime_method``
    Optional string set after grouping is applied. ``"file"`` means
    per-detector deadtime values from the data file were used.

``dead_time_us``
    Optional per-detector deadtime values in microseconds, read from formats
    that provide NeXus-style deadtime metadata.

``background_correction``
    Boolean; whether grouped-count background subtraction is active.

``background_method``
    Optional string set after grouping is applied. ``"fixed"`` means explicit
    forward/backward background values were used. ``"estimated"`` means the
    values were calculated from background bin ranges. ``"invalid_range"``
    means background correction was requested but not applied because the range
    was outside the grouped data.

``background_ranges``
    Optional inclusive background-bin ranges for forward and backward grouped
    histograms, stored as ``[[forward_start, forward_end],
    [backward_start, backward_end]]``. Shared ``background_range`` metadata may
    also be used as input.

``background_values``
    Optional forward/backward background values that were subtracted from the
    grouped histograms during the last grouping apply.

``detector_t0_bins``
    Optional per-detector time-zero bins, used by formats such as PSI BIN/MDU
    and MusrRoot/LEM ROOT where each detector can carry its own ``t0``.
    Grouping aligns detector histograms by these values before summing.

``root_histo_numbers``
    Optional list of original ROOT ``hDecay`` histogram numbers. Present for
    MusrRoot/LEM ROOT files so grouped detector slots can be traced back to
    the source ROOT objects used by musrfit.

``grouping_preset``
    Name of the last applied preset (e.g. ``"Vector Polarization"``).

``instrument``
    Instrument name set in the Detector Layout Editor (e.g. ``"EMU"``).
    Saved so the correct detector schematic is shown on reopen without
    re-running instrument detection.

``vector_axis``
    Active polarisation axis at save time (``P_x``, ``P_y``, ``P_z``, or ``ALL``).

``period_mode``
    Two-period RG mode (``Red``, ``Green``, ``G minus R``, ``G plus R``).


Save and load
-------------

.. code-block:: python

   from asymmetry.core.project.schema import load_project, save_project

   state = {
       "schema_version": 15,
       "created_with_app_version": "0.1.0",
       "datasets": [{"run_number": 3077, "source_file": "run3077.nxs", "metadata_overrides": {}}],
       "data_groups": [
           {"group_id": "g1", "name": "B = 60 G", "member_run_numbers": [3077],
            "order_key": "run", "kind": "user"},
       ],
       "browser_state": {
           "sort_column": 0,
           "sort_order": "ascending",
           "filters": {},
           "selected_run_numbers": [3077],
           "selected_group_ids": [],
           "data_groups": [
               {"group_id": "g1", "name": "B = 60 G", "member_run_numbers": [3077],
                "collapsed": False, "kind": "user"},
           ],
           "extra_columns": [],
       },
   }

   save_project(state, "session.asymp")
   restored = load_project("session.asymp")
   print(restored["schema_version"])

Schema migration
----------------

Use ``migrate_to_current`` to normalise older project files in scripts or tests.

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

Runnable example
----------------

See ``examples/project_files.py`` for a complete executable script.
