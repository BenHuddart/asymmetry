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
