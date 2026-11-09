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
results have no acquired-period ordinal. Combined acquired sources default to
red for quick methods and record period 1. All methods require that source
period's histograms, rather than a cached curve backed by another period's counts.
``calibration_source`` retains a matching selected cached curve when available;
a source with held histograms alone has empty curve arrays. Period extraction
retains the source ``Run`` identity without requiring duplicate run-number metadata.
The ``quick_calibration`` view and JSON block retain that explicit operation;
automatic legacy ratio settings remain scalar/source-only. Fixed beta one is
applied authoritatively even when a target carries a historical nonunity beta.

``CalibrationOptions()`` retains the automatic integral ratio; explicit
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
   :members: CalibrationOptions, calibrate_run, calibration_source

.. automodule:: asymmetry.core.workflow.calibration_results
   :members: CalibrationOutcome, QuickCalibrationResult

.. automodule:: asymmetry.core.workflow.reduction
   :members: ReductionSettings, estimate_alpha_for_run, reduce_run
