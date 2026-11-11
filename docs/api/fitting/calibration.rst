TF calibration
==============

Fit acquired F/B histograms through one count model with beta fixed at one
or free. Applied deadtime correction selects weighted corrected counts with
retained measured/supplied variance; off/skipped correction uses raw Poisson.
See :doc:`/reference/data_reduction/alpha_calibration` for assumptions,
units, safeguards, conditional error meaning, and reference comparisons. The
supplied dataset owns acquisition/grouping/correction facts; the request owns
intent. Explicit periods use the existing acquired-period selector. Returned
native domains retain detector membership, alignment, raw-bin windows, timing,
and correction preflight outcomes without copying a mutable grouping table.

.. automodule:: asymmetry.core.fitting.tf_calibration
   :members: TFCalibrationRequest, TFCalibrationResult, TFEnvelope, TFCalibrationSelection, fit_tf_calibration, select_tf_calibration

.. automodule:: asymmetry.core.fitting.count_results
   :members: CountFitParameter, CountFitDomain, JointCountFit

Applying and saving a captured result
-------------------------------------

Use the existing grouping-profile owner to apply a completed fit or selection::

   from asymmetry.core.project.calibration import CalibrationRecord, apply_calibration
   from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, save_project

   record = CalibrationRecord("calibration-1", result)
   apply_calibration(profile, record)
   state = {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [],
            "grouping_profiles": [profile.to_dict()]}
   save_project(state, "calibration.asymp")

Restore the profile with ``GroupingProfile.from_dict`` after ``load_project``.
Calibration identities must be nonempty strings and are checked when a record
is constructed, including during project restore.
Both coefficient policies reference one frozen record; their scalar values and
errors derive from its full fit. An unsuccessful selection may be retained in
``profile.calibration_records`` for diagnostics but cannot be applied. See
:doc:`/reference/project_files` for source availability and history semantics.

.. automodule:: asymmetry.core.project.calibration
   :members: CalibrationRecord, CalibratedCoefficient, apply_calibration

Workflow calibration
--------------------

Scripted calibration and CLI requests use the same core operations::

   from asymmetry.core.io import load
   from asymmetry.core.workflow import CalibrationOptions, ReductionSettings, calibrate_run, reduce_run

   source = load("calibrant.nxs")
   settings = ReductionSettings(deadtime="from_file")
   outcome = calibrate_run(source, settings,
       CalibrationOptions("tf_calibration_fit", fit_beta=True, period=1))
   if outcome.alpha is not None:
       applied = ReductionSettings(alpha=outcome.record, deadtime="from_file")
       signal = reduce_run(target.run, applied)

``ReductionSettings.alpha`` accepts an authored scalar or a captured
``CalibrationRecord``. ``effective_alpha`` and ``effective_alpha_source`` derive
the applied scalar/identity; ``calibration`` exposes the record. The record is
the authority for both alpha and beta. ``to_dict`` derives scalar output views
and encodes the canonical block with the project nonfinite codec;
``from_dict`` restores it without fitting, source-file access, or a new identity.
Old scalar/source-only settings do not acquire inferred joint records.

``ReductionSettings.alpha`` also accepts a ``QuickCalibrationResult``, preserving
its shared typed estimate and source/pair/period/correction/timing context,
including the contributor mapping for a mapped quick calculation. Such mapped
results have no acquired-period ordinal; ``mapped_period_number`` retains the
selected red/green set's 1-based ordinal separately. A missing mapped-set
identifier remains unknown; reading does not guess it from the coefficient. Combined acquired sources default to
red for quick methods and record period 1. All methods require that source
period's histograms, rather than a cached curve backed by another period's counts.
``calibration_source`` retains a matching selected cached curve when available;
a source with held histograms alone has empty curve arrays. Period extraction
retains the source ``Run`` identity without requiring duplicate run-number metadata.
The ``quick_calibration`` view and JSON block retain that explicit operation;
automatic legacy ratio settings remain scalar/source-only. Fixed beta one is
applied authoritatively even when a target carries a historical nonunity beta.

Omitting the options argument in ``calibrate_run(source, settings)`` selects
TF calibration fit with automatic four-envelope selection and beta fixed at one.
Joint-beta requests require a single-pair source layout; projection layouts
retain beta one and reject ``fit_beta=True`` at the shared workflow boundary.
``explicit_calibration_options`` owns this fresh explicit default. Passing
``CalibrationOptions()`` deliberately retains the automatic integral ratio; explicit
``diamagnetic``, ``general``, and ``ratio`` return the shared typed quick result
with source/count context. TF with ``envelope=None`` selects all four; an explicit
``TFEnvelope`` constrains the actual fit. ``fit_tf_calibration`` also accepts
``envelope=`` directly and retains exponential as its default. Historical
``count_fit`` consumers retain their identity and exponential model.
Parameter overrides and cancellation pass into the shared TF operation.
A failed/no-selection result can be emitted for diagnostics but cannot become
applied settings. :doc:`/reference/agent_workflow` documents CLI flags and
independent source-period semantics.

.. automodule:: asymmetry.core.workflow.calibration
   :members: CalibrationOptions, calibrate_run, calibrate_source, calibration_source, explicit_calibration_options

.. automodule:: asymmetry.core.workflow.calibration_results
   :members: CalibrationOutcome, QuickCalibrationResult

.. automodule:: asymmetry.core.workflow.reduction
   :members: ReductionSettings, estimate_alpha_for_run, reduce_run

``calibrate_source`` accepts an identified source with its fully resolved grouping.
The GUI uses this seam for manual/estimated deadtime and fixed/reference background
policies that the CLI settings vocabulary cannot express. It shares preparation,
results, cancellation, and scientific support with ``calibrate_run``.

GUI application uses the same captured outcomes. Quick policies retain the full
``QuickCalibrationResult`` context through ``QuickCalibratedCoefficient``.
``GroupingProfile.projection_alpha_policies`` associates each projection label
with its own captured coefficient; TF views reference the profile's canonical
history. Each projection keeps beta fixed at one. Runtime scalar views are derived
from these owners, including released-run override application and persistence.
Projection value, error, method and source views update together; a manual
override removes the previous measurement's attribution rather than retaining
its error or reference run.

``reduce_run(run, settings)`` honours the authored count merge without spectral
analysis. TF count preparation honours the fixed grouping factor; a selection
with no complete fit bin raises the existing count-domain diagnostic before
unused correction preflight. The raw run and stored historical results remain
unchanged. See :doc:`/reference/count_domain_fitting`.

For workflow reduction, implicit settings inherit a selected projection's alpha.
An explicit calibration result or authored scalar overrides the requested pair;
other projections retain their own policies and beta one. Use
``ReductionSettings(alpha=1.0, alpha_source="user")`` for authored unity;
nondefault scalar values are explicit. A joint-beta record cannot be applied
to a declared projection. A scalar pair outside the projections retains both
coefficients, including the captured beta. Reduction cache identity includes
the pair-balance policy so cached curves and reported coefficients agree.

Fixed count-pulse input
-----------------------

``fit_single_histogram`` and ``fit_fb_alpha`` accept ``Parameter("dpsep", value,
fixed=True)`` in µs. The supplied separation must be finite and nonnegative;
a free parameter or one with a following link/constraint raises
``PulseRequestError`` with fixed-input guidance.
The double-delta model consumes no finite-width input. See
:doc:`/reference/count_domain_fitting` for its origin, survival weighting,
conditional errors, and onset limitations.
