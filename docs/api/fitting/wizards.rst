Fit wizards
===========

.. currentmodule:: asymmetry.core.fitting


Single/global automatic merging uses
:func:`~asymmetry.core.transform.preparation.prepare_wizard_dataset` before
``fit_record`` and candidate fan-out. Recorded histogram context selects counts
before the ratio; unknown and rotated curve observations use independent means.
Reconstruction and numerical cache reuse identify the same fitted observation.

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
