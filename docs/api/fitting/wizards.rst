Fit wizards
===========

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
---------------------

.. autoclass:: asymmetry.core.fitting.peak_detection.PeakAnalysis
   :members:

.. autofunction:: asymmetry.core.fitting.peak_detection.analyze_dataset_peaks
