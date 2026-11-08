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
