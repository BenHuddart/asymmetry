TF calibration
==============

Fit acquired raw F/B histograms through one count model with beta fixed at one
or free. See :doc:`/reference/data_reduction/alpha_calibration` for assumptions,
units, safeguards, conditional error meaning, and reference comparisons. The
supplied dataset owns acquisition/grouping/correction facts; the request owns
intent. Explicit periods use the existing acquired-period selector. Returned
native domains retain detector membership, alignment, raw-bin windows, timing,
and correction preflight outcomes without copying a mutable grouping table.

.. automodule:: asymmetry.core.fitting.tf_calibration
   :members: TFCalibrationRequest, TFCalibrationResult, fit_tf_calibration

.. automodule:: asymmetry.core.fitting.count_results
   :members: CountFitParameter, CountFitDomain, JointCountFit
