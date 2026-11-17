Fit wizards
===========

.. py:module:: asymmetry.core.fitting.global_search.trend_search


.. currentmodule:: asymmetry.core.fitting


Single/global automatic merging uses
:func:`~asymmetry.core.transform.preparation.prepare_wizard_dataset` before
``fit_record`` and candidate fan-out. Recorded histogram context selects counts
before the ratio; unknown and rotated curve observations use independent means.
Current reconstruction and numerical reuse identify the same fitted observation.
Known histogram Gaussian fits omit undefined ratios and exactly zero propagated
variance, carrying the core observation report through worker records, scoring,
completion, and rebuilding. Historical scores retain their recorded meaning;
compact caches omit residual arrays and current reuse requires the current policy.

Internal spectral diagnostics use
:func:`~asymmetry.core.fourier.diagnostic.prepare_diagnostic_observation`.
Uniform input curves retain their samples. Nonuniform captured histogram
observations use fixed native bins (factor 1, merge 1) over the actual retained
native edge support, after all recorded selections. No Run/recipe-only replay
or curve interpolation is inferred. The resulting lightweight owned arrays
cross existing process tasks without detector/reference replay buffers.

``PeakAnalysis.diagnostic_attribution`` and
``SpectrumFingerprint.diagnostic_attribution`` retain policy, source observation,
recipe and actual window/grid. ``FitResult.spectral_residual_attribution`` labels
a separate fixed count residual view, evaluated with the same model/parameters.
Original residual arrays, objective and statistics remain attached to the fit.
Absent historical attribution remains ``None``; loading never regenerates it.
``analyze_dataset_peaks(..., detrend_model=callable)`` evaluates a trend in percent
at the diagnostic times; sampled-only ``detrend_curve`` requires the same grid.

Single-spectrum fit-wizard helpers:

.. autoclass:: asymmetry.core.fitting.CandidateTemplate
   :members:

.. autoclass:: asymmetry.core.fitting.CandidateAssessment
   :members:

.. autoclass:: asymmetry.core.fitting.FitWizardRecommendation
   :members:

.. autofunction:: asymmetry.core.fitting.build_fit_wizard_recommendation

.. autofunction:: asymmetry.core.fitting.rerank_fit_wizard_recommendation

Global fit-wizard helpers:

.. autoclass:: asymmetry.core.fitting.GlobalParameterRecommendation
   :members:

.. autoclass:: asymmetry.core.fitting.GlobalCandidateAssessment
   :members:

.. autoclass:: asymmetry.core.fitting.GlobalFitWizardRecommendation
   :members:

.. autoclass:: asymmetry.core.fitting.SelectionObjective
   :members:

.. autoclass:: asymmetry.core.fitting.global_search.trend_objective.CandidateRung
   :members:
   :inherited-members:

.. autofunction:: asymmetry.core.fitting.build_global_fit_wizard_recommendation

.. autofunction:: asymmetry.core.fitting.rerank_global_fit_wizard_recommendation

Spectral peak analysis
----------------------

.. autoclass:: asymmetry.core.fitting.peak_detection.PeakAnalysis
   :members:

.. autofunction:: asymmetry.core.fitting.peak_detection.analyze_dataset_peaks

Composite
---------

.. py:data:: COMPONENTS
   :module: asymmetry.core.fitting.composite

   Registry of composite-model components.

.. autofunction:: asymmetry.core.fitting.composite.placeholder_component_definition

Damped line scan
----------------

.. autofunction:: asymmetry.core.fitting.damped_line_scan.detect_damped_lines

.. autoclass:: asymmetry.core.fitting.damped_line_scan.DampedLineAnalysis
   :no-members:
   :no-special-members:

.. autoclass:: asymmetry.core.fitting.damped_line_scan.DampedLine
   :no-members:
   :no-special-members:

.. py:attribute:: DampedLineAnalysis.threshold_delta_chi_squared
   :module: asymmetry.core.fitting.damped_line_scan

Fit wizard
----------

.. autodata:: asymmetry.core.fitting.fit_wizard.MIN_CYCLES_IN_EFFECTIVE_WINDOW
   :no-value:

.. autofunction:: asymmetry.core.fitting.fit_wizard.analysis_rebin_factor

.. autofunction:: asymmetry.core.fitting.fit_wizard.build_fit_wizard_recommendation

.. autofunction:: asymmetry.core.fitting.fit_wizard.fingerprint_spectrum

.. autofunction:: asymmetry.core.fitting.fit_wizard.single_fit_build_signature

.. py:attribute:: CandidateAssessment.refinement_delta_chi_squared
   :module: asymmetry.core.fitting.fit_wizard

.. py:attribute:: CandidateAssessment.under_converged
   :module: asymmetry.core.fitting.fit_wizard

.. py:attribute:: FitWizardRecommendation.rebin_factor
   :module: asymmetry.core.fitting.fit_wizard

.. py:attribute:: FitWizardRecommendation.analysed_points
   :module: asymmetry.core.fitting.fit_wizard

Global fit wizard
-----------------

.. autofunction:: asymmetry.core.fitting.global_fit_wizard.build_global_fit_wizard_screening_recommendation

.. autofunction:: asymmetry.core.fitting.global_fit_wizard.build_or_complete_single_fit_wizard_recommendations_for_global_portfolio

.. autofunction:: asymmetry.core.fitting.global_fit_wizard.series_template_alphabet

.. autofunction:: asymmetry.core.fitting.global_fit_wizard.alphabet_bound_dropped_keys

.. autofunction:: asymmetry.core.fitting.global_fit_wizard.series_rebin_factor

.. autofunction:: asymmetry.core.fitting.global_fit_wizard.screening_templates_for_effort_tier

Trend objective
---------------

.. autofunction:: asymmetry.core.fitting.global_search.trend_objective.trend_sort_key

.. autofunction:: asymmetry.core.fitting.global_search.trend_objective.trend_contenders

.. py:data:: TREND_TIE
   :module: asymmetry.core.fitting.global_search.trend_objective

Peak detection
--------------

.. autofunction:: asymmetry.core.fitting.peak_detection.analyze_damped_scan_peaks

.. autofunction:: asymmetry.core.fitting.peak_detection.analyze_early_window_peaks

.. autofunction:: asymmetry.core.fitting.peak_detection.detect_peaks_in_spectrum

.. autofunction:: asymmetry.core.fitting.peak_detection.merge_damped_scan_peaks

.. autofunction:: asymmetry.core.fitting.peak_detection.merge_early_peaks

.. autofunction:: asymmetry.core.fitting.peak_detection.merge_user_peaks

.. autoclass:: asymmetry.core.fitting.peak_detection.DetectedPeak
   :no-members:
   :no-special-members:

.. autofunction:: asymmetry.core.fitting.peak_detection.early_window_crops

Process pool
------------

.. py:module:: asymmetry.core.fitting.process_pool

.. autofunction:: asymmetry.core.fitting.process_pool.terminate_spawn_pool

Wizard timing
-------------

.. py:module:: asymmetry.core.fitting.wizard_timing

.. autoclass:: asymmetry.core.fitting.wizard_timing.WizardStageProgress
   :no-members:
   :no-special-members:
