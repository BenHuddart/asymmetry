Project files
=============

The ``.asymp`` project file persists the analysis state of a session:
which datasets are loaded, how the Data Browser is sorted and filtered,
the per-run grouping (groups, alpha, bunching, deadtime, background),
the single- and global-fit model setups with their parameter tables and
bounds, separate frequency-domain fit state with spectral peak models,
the most recent fit overlays, the Fourier panel state including
per-run phase tables, any cached Fit Wizard or Global Fit Wizard
analyses, per-run *representation* fit slots (the Single tab's fit
alone — a batch, global, grouped or scan run never writes one), and the
*batches* (fit series, each carrying the recipe that produced it) that
drive the Fit Parameters trending panel and the plot's overlays. Raw
detector arrays are *not* embedded — the file references source data by
path and reloads from disk on open. Fourier spectra are regenerated from
their stored recipe (window, padding, phase, group selection) rather
than embedded, so the file remains compact even after frequency-domain
work. This makes ``.asymp`` files small enough to share alongside the
raw data when sending an analysis to a collaborator, or to archive
alongside paper supplementary material so that readers can reproduce
every fit shown in the figures.

Saving is crash-safe: the new content is written to a temporary file
next to the target and only swapped into place once it is fully flushed
to disk, so a crash or a full disk mid-write leaves the previous save
untouched. The file it replaces is kept alongside it as ``<name>.asymp.bak``
(one generation — a second save overwrites it). While a project has
unsaved changes, a background timer also writes a crash-recovery
snapshot to ``<name>.autosave.asymp`` (or, for a session that has never
been saved, a file under the platform's application-data directory);
see `Crash-safe save and autosave`_ below.

Because only paths are stored, a project whose data files have since
moved cannot reload them directly. On open, Asymmetry lists the files it
could not find in a **Data Files Not Found** prompt and offers to
**Locate Data Directory**: choose the folder that now holds the files
and every missing dataset whose file name is present there is loaded
from it. The look-up matches on file name alone, so it works across
operating systems — a project saved on Windows with ``C:\...`` paths
resolves against a folder chosen on macOS or Linux, and vice versa. The
stored paths are left as written; saving the project afterwards records
the new locations.

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
  *FitSlot* — the Single tab's own fit for that representation only: model,
  parameters, result summary, provenance (``"none"``/``"single"``/``"wizard"``)
  and, when the single-fit GUI produced it, a ``ui_state`` blob that restores
  the form verbatim. Fourier spectra are re-generated from the recipe on
  load; time-domain asymmetry is re-computed from the raw data.
* **Fit series (batches)** — each batch, global, grouped or scan fit over
  multiple runs (or multiple runs' detector groups) is recorded as a
  ``FitSeries`` that carries the member list, parameter roles, per-member
  result summaries, and the *recipe* that produced it (model, parameter
  rows, fit range, seeding, co-add) plus its own ``trend_excluded_runs``.
  The Fit Parameters trending panel reads directly from these series,
  organised by the active representation.
* **Active series** — the top-level ``active_series`` map names, per
  representation, the series the Batch tab, the Parameters chip rail and the
  plot's default overlay all currently point at; see `Fit series recipe and
  active series`_ below.
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

A group is a **phase** when its ``parent_group_id`` names another
``data_groups`` entry — the series group a transition partitions it out of
(see :ref:`phases-within-a-group` in :doc:`gui_usage`). Schema v19 adds six
additive phase fields to every ``data_groups`` entry, meaningless (and left at
their defaults) on an ordinary group:

``parent_group_id``
    The id of the series group this phase belongs to, or ``null`` for an
    ordinary group. A group *is* a phase exactly when this is set.

``phase_ordinal``
    1-based position of this phase along the sweep axis (``1`` for the
    coldest/lowest-field phase), or ``null``. Drives both the phase's display
    name (:doc:`gui_usage`'s ``Phase I``, ``Phase II``, …) and its identity
    colour, which cycle past five.

``phase_range``
    ``[first, last]`` axis value spanned by this phase's members, or ``null``.

``phase_boundaries``
    ``{"lower": [estimate, half_gap] | null, "upper": [estimate, half_gap] |
    null}``. Both keys are always present; a ``null`` side means this phase
    sits at a series end, where there is no break to estimate.

``phase_color``
    The swatch colour (a hex string) assigned when the wizard created this
    phase, or ``null`` to fall back to the ordinal's slot in the identity
    palette.

``phase_provenance``
    A plain, JSON-able record of how this phase was found and last fitted:
    ``found_at`` (ISO timestamp), ``selected_breaks``, ``gains``,
    ``axis_key``, ``model_title``, ``confidence``, ``shared_parameters``,
    ``fit_state``, and ``reduced_chi_squared`` — every key is genuinely
    optional, since a phase exists as soon as the partition is applied, which
    is before its own global fit has run. Defaults to ``{}``.

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

.. _fit-series-recipe-and-active-series:

Fit series recipe and active series
------------------------------------

Every ``batches`` entry (schema v20) carries the setup that produced it as a
``recipe`` dict, and its own trend gating:

``recipe``
    ``{"parameters": [...], "fit_range": {"min": ..., "max": ...}, "seeding":
    "auto"|..., "coadd": {"mode": "off"|..., "window": 2}}``. ``parameters``
    is the Batch tab's table at record time — one entry per physics
    parameter with ``name``, ``value``, ``type`` (``Global``/``Local``/
    ``Fixed``/``File``), ``bounds`` and ``seeded``; ``fit_range`` is the
    window the series was cropped to, in the representation's own unit,
    with either bound ``null`` for unbounded; ``seeding`` is the Batch tab's
    per-run seeding mode; ``coadd`` is the co-add mode and window. This is
    the whole recipe a re-run compares for identity (below), and what the
    Batch tab restores when you reopen the series.

``trend_excluded_runs``
    Member run numbers ticked out of *this* series' trend without touching
    its fit — the per-series successor to the old per-slot
    ``include_in_trend`` flag, toggled from the same **Exclude from trend**
    plot action.

Recording a series compares its recipe and effective member set against
what is already recorded — the series open in the Batch tab first, then the
representation's active series (below), then the newest series already
describing the same analysis — as a single canonical identity string
(``FitSeries.recipe_identity()``). An identical match **replaces** that
series' results in place, keeping its ``batch_id`` and label; any
difference — a wider window, a different model or classification, a
different member set — writes a **new** ``batches`` entry instead, with a
fresh id and a default label derived from the recipe
(``<model> · <fit-range>[ · <group>]``, suffixed `` (2)`` on a collision).
Nothing is ever superseded or deduplicated at load time: two series that
differ only in fit range are two entries, kept side by side.

The top-level ``active_series`` object maps each representation's value
(``"time_fb_asymmetry"``, ``"freq_fft"``, …) to the ``batch_id`` of its
*active* series — the one the Batch tab has open, the Parameters chip rail
highlights, and the plot draws by default on every run the series covers.
Deleting a series clears any ``active_series`` entry that pointed at it,
leaving that representation with none until another is opened or recorded.

Joint fits (schema v22)
------------------------

The top-level ``joint_fits`` list holds one entry per recorded
:doc:`joint fit <joint_fit>`, serialised from
:class:`~asymmetry.core.representation.joint_fit.JointFit`:

``joint_id``, ``label``, ``rep_type``
    The record's id, its user-assigned label (``null`` when it still tracks
    the members' own default name), and the representation type its members
    belong to.

``member_batch_ids``
    The member series' ``batch_id`` values, in tick order — also the order
    a shared row's seed is taken from (its *first* contributing member).

``shared``
    One entry per shared-table row: ``name``, ``members`` (``{batch_id:
    that series' own parameter name}``), ``value``, and ``min``/``max``
    (``null`` for an unbounded side).

``result``
    The last run's summary only — shared values, uncertainties, the shared
    covariance rows, combined and per-series χ², and a timestamp — never
    curves; the curves are each member's own ``results_by_run``, already
    persisted there under its own ``batch_id``.

A pre-v22 project has no joint fits: ``_migrate_v21_to_v22`` is purely
additive (version bump plus an empty ``joint_fits`` list). A malformed
``shared`` row is dropped individually on load rather than failing the whole
record, and a record naming fewer than two members is dropped entirely — a
joint fit of one series composes nothing.

Every entry in ``batches`` gains two optional, additive fields once a joint
run has recorded onto it: ``joint_fit_id`` (the owning record's id, or
``null`` for an unstamped series) and ``shared_params`` (``{this series' own
parameter name: shared name}``, empty when unstamped). Both are display
state read by the Parameters panel and the joint-fit window, never part of
what the series *is* — ``FitSeries.recipe_identity()`` ignores them, so
re-running a series' recipe unchanged still replaces the same entry in
place. A solo Batch-tab run of a stamped series always clears both fields,
which is what makes staleness (see :ref:`joint-fit-staleness-and-detaching`
in :doc:`joint_fit`) computable from live project state rather than stored.

.. _fit-slot-fields-v20:

Per-run fit slot fields (schema v20)
--------------------------------------

A representation's ``fit`` (and each entry of ``projection_fits``) is a
*FitSlot* holding the Single tab's own fit alone:

``model``, ``parameters``, ``result``
    The composite model, its parameter table and the fit-result summary —
    ``null``/empty for a representation that has never been singly fitted.

``provenance``
    ``"none"``, ``"single"`` or ``"wizard"``. ``"batch"`` and ``"global"``
    are recognised only while reading an older file — schema v19 and
    earlier wrote them for a run that belonged to a batch or global fit;
    the v19→v20 migration drops such a slot entirely, since its result
    already lives on the series that recorded it.

``ui_state``
    The single-fit form payload (composite model, parameters, result HTML,
    wizard cache) needed to restore the editor verbatim; present only when
    the single-fit GUI itself produced the slot, omitted (not written)
    otherwise.

A slot no longer carries ``batch_id``, ``diverged`` or ``include_in_trend``:
per-run state cannot diverge when it only ever holds one fit, and trend
gating moved to the series' own ``trend_excluded_runs`` above. A run with no
single fit but a series result still shows fitted state on the Single tab —
reconstructed from the active series' recorded result for that run, not
stored on the slot itself (:doc:`gui_usage`, "Carrying a model forward
between runs").

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

The stored recommendation is a **compact** form of the in-memory one. A live
recommendation keeps, for every candidate it assessed, a fitted curve, its
component curves and the fit residuals at the resolution of the analysed
record — hundreds of megabytes on a fine-resolution run. What is written to
the file instead is:

* each curve sampled on a uniform stride down to at most 512 points (all the
  curves of one candidate share a stride, so they stay aligned), with each
  sample rounded to seven significant digits — these arrays are only ever
  drawn, and the wizard's own plots decimate more aggressively than that;
* no residual *series* — the numbers derived from it (residual RMS, runs
  z-score, autocorrelation, FFT peak SNR and the gate reasons) are stored on
  the assessment itself, so only the residual panel of a *cached* redisplay
  is affected: it is blank until the wizard is re-run for that run.

Everything else — the ranking, scores, fitted parameters, uncertainties,
diagnostics, peak analysis and narrative — is stored in full, so a reopened
project still shows the cached recommendation, its comparison table and its
answer card without re-running the analysis. Re-running the wizard always
restores full-resolution curves and residuals.

Payloads written before this change (full-resolution curves, residuals
included) still load unchanged; no schema bump is involved, since the block is
optional and both shapes deserialise.

Grouping profiles and assignments
---------------------------------

The project's named grouping profiles (see :doc:`detector_grouping`) are
stored in a top-level ``grouping_profiles`` list. Each entry is a serialized
profile — its ``name``, ``fingerprint`` (``{"instrument": ...,
"histogram_count": ...}``), the grouping structure, and the alpha, deadtime,
background, and t0 policies. The ``active`` flag marks each fingerprint's
**default-for-new-runs** profile (exactly one per fingerprint); several
profiles of the same fingerprint may be in concurrent use. An optional
``color`` (a ``#rrggbb`` string, assigned from the GUI palette when a
non-default profile is first saved; the ★ default stays colourless) is the
profile's identity colour, worn by its runs' numbers in the Data Browser and
throughout the Grouping window; profiles saved before colours existed are
tolerated and recoloured on next save.

Each dataset entry records which profile it follows (schema v17):

``profile``
    The name of the dataset's assigned grouping profile. A dataset carrying
    only ``profile`` follows that profile — its grouping is re-resolved
    against the profile on load. A dataset carrying **both** ``profile`` and
    ``grouping_overrides`` is *released*: the override payload below is
    authoritative for resolution, and ``profile`` names the base profile the
    run stays assigned to (the reattach target). A dataset with neither key
    follows its fingerprint's default profile, as does one whose assigned
    name no longer resolves.

.. _t0-policy-schema:

t0 policy (schema v21)
~~~~~~~~~~~~~~~~~~~~~~

Each profile's ``t0_policy`` block stores which of the three t0 modes
(:doc:`detector_grouping` § Time-zero (t0) modes) it uses — it is always an
explicit, stored choice, never inferred from comparing values:

``mode``
    ``"from_file"`` (the default), ``"manual"`` or ``"auto_detect"``.

``offset_bins``
    *Manual* only — the signed offset, in bins, from each run's own file t0
    (not an absolute bin: one profile shifts every run in scope by the same
    amount, however their individual headers differ).

``legacy_value``
    A pre-v21 absolute-bin manual t0, carried over unconverted from schema
    v20 and older. It is not itself resolvable: only project *open*, once the
    runs are loaded, can convert it into the equivalent ``offset_bins``
    against the run the value was originally typed against (``source_run``
    when recorded, else the first loaded run of the profile's fingerprint).
    Resolving a policy that still carries ``legacy_value`` raises. Schema
    v20's absolute ``value`` is renamed to ``legacy_value`` on migration
    (``_migrate_v20_to_v21``); nothing is converted at that point.

``strategy`` / ``spread_bins`` / ``source_run``
    *Auto-detect* only — provenance from the last resolution's search
    (strategy name and detector spread), plus which run the values came from.

On project open, ``heal_t0_policies`` repairs two states that can only be
recognised once the runs are known: it converts any surviving
``legacy_value`` (see above), and it heals a ``manual`` policy whose
``offset_bins`` resolves to zero against every run in its scope back to
``from_file`` — a leftover of the pre-v21 comparison-based inference that
mislabelled a plain From file profile as Manual. Each repair is logged (see
:doc:`detector_grouping`); the project file itself is not rewritten until
the next save.

Good-window policy
~~~~~~~~~~~~~~~~~~~

Each profile's ``good_window_policy`` block stores which of the two
good-window modes (:doc:`detector_grouping` § Good-window modes) it uses —
an explicit, stored choice, never inferred from a run's resolved values.
It is written only when the mode is not ``from_file``, so a project without
a Manual good window round-trips with no trace of the key and needs no
schema bump.

``mode``
    ``"from_file"`` (the default, and the only value ever written before
    this feature) or ``"manual"``.

``first_offset_bins`` / ``last_offset_bins``
    *Manual* only — each end's signed offset, in bins, from the run's own
    **effective** t0 (the t0 the profile's ``t0_policy`` resolves to, not
    necessarily the file t0): one profile therefore gives every run it
    covers the same window relative to its own t0, and the window rides
    along automatically when a Manual or Auto-detect t0 shift moves that
    t0. An end left unset keeps the run's file value for that end.

Grouping overrides
------------------

Each dataset entry in the project JSON may contain a ``grouping_overrides``
block that stores the full custom grouping of a run *released* from its
profile (or of a run whose fingerprint has no profile). When present, it is
re-applied to the dataset on project load so that the exact detector grouping
is preserved without re-running the Grouping dialog.

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

``t0_bin``
    The run's common analysis time-zero: an absolute, 0-based bin index (the
    max of ``detector_t0_bins`` over the forward/backward analysis groups,
    unless a t0 policy shifted it — see :ref:`t0-policy-schema` below).

``t0_time_us``
    Optional exact time-zero in microseconds, kept alongside ``t0_bin`` for
    formats that record a continuous value (ISIS ``time_zero``, MusrRoot's
    ``Double_t`` ``Time Zero Bin``). Absent when the file carries only an
    integer bin; see :doc:`detector_grouping` § Time-zero (t0) modes for the
    bin-centre time-stamp convention this feeds.

``t0_source``
    How the run's t0 was determined: ``"file"`` (the header value, the
    default), ``"conflict"`` (an ISIS file whose ``time_zero`` and ``t0_bin``
    disagreed — the attribute won), ``"missing"`` (no usable header t0 at
    all, not yet resolved) or ``"detected"`` (a ``"missing"`` run resolved
    against the automatic search's consensus). See :doc:`data_reduction/t0_search`.

``detector_t0_bins``
    Optional per-detector time-zero bins, used by formats such as PSI BIN/MDU
    and MusrRoot/LEM ROOT where each detector can carry its own ``t0``.
    Grouping aligns detector histograms by these values before summing.

``effective_detector_t0_bins``
    Optional per-detector t0 bins actually used for alignment, written only
    when a *manual* or *auto-detect* t0 policy resolves to a shift from the
    file values. Every alignment consumer in the app reads this (when
    present) in preference to ``detector_t0_bins`` through one resolver,
    ``effective_detector_t0_bins()`` in ``core/transform/t0.py`` — the run's
    loaded histograms and their file-derived ``detector_t0_bins`` are never
    mutated.

``t0_search_strategy`` / ``t0_search_spread_bins``
    Provenance recorded when a t0 policy runs the automatic search (Auto-detect,
    or a From file run whose header t0 was missing): the strategy used
    (``"prompt_peak"`` or ``"pulse_edge"``) and the detector-to-detector
    spread of the estimates, in bins.

``t0_method`` / ``t0_reference_run``
    Provenance written by **Promote t₀** (:doc:`count_domain_fitting`):
    ``t0_method`` is always ``"count_fit"``, and ``t0_reference_run`` is the
    run number the fitted offset came from — the same pattern as
    ``alpha_method`` / ``alpha_reference_run``.

``bin_index_base``
    Display-only numbering base for the bin indices above (1 for ISIS,
    0 otherwise). The internal, zero-based bin indices are unaffected; this
    only tells the GUI which number to add back when showing a bin index to
    the user.

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

.. _crash-safe-save-and-autosave:

Crash-safe save and autosave
-----------------------------

``save_project`` never writes the target file directly: the new JSON is
serialised to a temporary file in the same directory, flushed and
``fsync``\ ed, and only then swapped over the target with an atomic
``os.replace``. A crash, a full disk or a raised exception at any point up
to that swap leaves the previous file exactly as it was and simply drops
the temporary file, so a save can never leave a project half-written. The
file the swap replaces is kept alongside it as ``<name>.asymp.bak`` — one
generation, overwritten by the next save — as a manual fallback (open it by
renaming away the ``.bak``); this is skipped for the autosave snapshot
below, which is itself a backup and keeps none of its own. The ``.bak`` is
copied from the existing project file rather than the file being renamed to
it, so the project file itself stays in place right up to the swap: there is
no moment at which a crash could leave no project file at all.

While a project has unsaved changes, a timer (default every 5 minutes,
``0`` disables it — the ``QSettings`` key ``project/autosave_interval_minutes``)
writes a crash-recovery snapshot to ``<name>.autosave.asymp`` beside a saved
project, or to an ``untitled.autosave.asymp`` file under the platform's
application-data directory for a session that has never been saved. A
successful save or a clean window close deletes it — including the
``untitled`` snapshot a session wrote before its first **Save As**. If you
keep working while a save is writing, that work is not in the saved file:
the session stays marked as modified and its snapshot is kept. Opening a
project whose autosave sibling is newer than the file itself asks "An autosave from
*<time>* is newer than this project. Load the autosave instead?", with
**Load autosave** (reads the snapshot but still treats the session as
unsaved against the original path), **Open saved file** (reads the project
normally and leaves the autosave in place) and **Cancel**.

Save and load
-------------

.. code-block:: python

   from asymmetry.core.project.schema import load_project, save_project

   state = {
       "schema_version": 17,
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
