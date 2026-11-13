Project persistence
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

.. autofunction:: prepare_project

.. autofunction:: load_prepared_project

.. autoclass:: PreparedProject
   :members:

.. autoexception:: ProjectValidationError

.. autoexception:: UnsupportedSchemaVersion

.. autodata:: CURRENT_SCHEMA_VERSION

   The storage schema is ``40``. Wizard payload, workdir, effective-recipe, and
   scientific computation versions are separate authorities.

Preparation and application
---------------------------

``load_project`` returns canonical JSON-compatible state. ``prepare_project``
accepts a complete versioned state dictionary; ``load_prepared_project`` reads a
file and returns the same prepared form. Each performs the owning migrations and
pending-model retry, then constructs existing grouping profiles, representations,
series, joint fits, and parameter studies. Caller input remains unchanged. A
``checkpoint`` callback can raise cancellation between declarations.

``PreparedProject`` holds the detached canonical dictionary and existing domain
owners. Application transfers these objects to the receiving session; it is not
a reusable widget snapshot. No runtime object is inserted into the serialised
JSON. ``ProjectModel.from_project_state`` uses this same project boundary.
Standalone ``ProjectModel.from_dict`` reads the model-only codec rather than a
complete project file.

Malformed owned containers, duplicate identities, and required live links raise
``ProjectValidationError`` with a field location before GUI state changes. External
file/model availability, stale historical membership, draft fit forms, and original
constraint/uncertainty archives remain separate from structural validity. No load,
save, or preparation assigns current execution to an old or unknown result.

The GUI prepares before replacing the active session. Acquisition restoration can
still be cancelled after application begins, retaining a marked partial load with
a save confirmation. See :doc:`../reference/project_files` for the exact UI contract.

Canonical file contents
-----------------------

A UTF-8 JSON object contains ``schema_version`` and a ``datasets`` array. Dataset
rows have integer ``run_number`` identities and string source locators. Optional
``combined_datasets`` rows name their integer identity and ordered declared
sources. Project-level ``grouping_profiles`` retain grouping and calibration policy
owners; runs name a profile or retain released per-run overrides.

The optional registries are ``batches`` (fit series), ``data_groups``,
``joint_fits``, ``global_fit_studies``, and ``active_series``. A dataset's
``representations`` map is keyed by ``RepresentationType`` values, including
``time_fb_asymmetry``, ``time_groups``, ``time_maxent_recon``, ``freq_fft``, and
``freq_maxent``. Each representation retains its recipe, result metadata, trend
state, and ``single_fits`` sets keyed by projection (``""`` for the default).
Each set contains ``fits`` and an ``open_id`` selecting a saved fit.

``fit_states`` holds the time and frequency editor forms separately. Browser,
plot, Fourier, MaxEnt, and analysis-window containers retain presentation state.
Unknown user metadata is preserved. A future storage schema is rejected, rather
than interpreted as a known format.

Schema utilities
----------------

.. autofunction:: asymmetry.core.project.schema.validate

.. autofunction:: asymmetry.core.project.schema.migrate_to_current

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

Acquisition declarations
------------------------

``PreparedProject.sources`` is an ordered immutable inventory assembled at the
same preparation boundary as dataset, grouping, profile, calibration, and result
owners. It does not read acquisitions. Optional ``source_binding`` fields preserve
expected and observed content facts independently; their absence is unknown.
Core ``verify_sources`` consumes these declarations and caller-resolved candidates
without applying state. See :doc:`io` and :doc:`/reference/project_files` for
outcomes and the limits of current project reopening.


Verified source application
---------------------------

.. autofunction:: verify_project_sources

.. autofunction:: apply_source_decisions

.. autoclass:: ProjectSourceApplication
   :members:

``SourceDeclaration.owner`` carries its prepared address and whether it represents
an active input. Verification remains read-only. Explicit replacements name source
indices from that complete request, not filenames or equal digest values.
Historical owners cannot accept replacements. Unselected changed inputs remain
unavailable, while matched and legitimate legacy-unknown inputs continue normally.
The operation returns detached project owners; it does not mutate the supplied
project, fit a model, or write a file.

.. code-block:: python

   from asymmetry.core.project import (
       load_prepared_project, verify_project_sources,
       apply_source_decisions, save_project,
   )

   project = "session.asymp"
   prepared = load_prepared_project(project)
   verified = verify_project_sources(prepared, project)
   # Inspect verified.facts and explicitly choose the current owner indices to replace.
   application = apply_source_decisions(verified, replacements=[0])
   print(application.report())
   save_project(application.prepared.state, "accepted.asymp")

``report()`` omits source/candidate locators and private diagnostic text. Local
verification facts retain those fields for meaningful source selection. A caller
that consumes acquisitions must read ``candidate_file`` and use ``check_capture``
to compare the actual loader capture with the accepted observation. GUI restoration
uses this guard before applying acquired timing or reductions. Number-only legacy
reference requests retain their existing live-input resolution.

The API/CLI action updates source expectations; recomputation is performed through
the receiving operation's existing reduction workflow. It is not a general project
fitting engine, an old-code replay, or a raw-data archive.

``verify_project_sources`` returns ``ProjectSourceVerification`` with the rebound
``prepared`` owners and ordered ``facts``. Following-profile reference requests
are classified using the actual instrument and histogram count through the normal
loader on the cancellable verification worker. Authoritative per-run overrides
leave their base profiles historical. Unused or disabled reference settings do not
request replacement. Capture disagreement aborts before project application.
