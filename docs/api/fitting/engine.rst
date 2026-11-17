Fit engine
==========

.. currentmodule:: asymmetry.core.fitting


.. autoclass:: asymmetry.core.fitting.engine.FitEngine
   :members:
   :undoc-members:
   :show-inheritance:


Known histogram datasets carry :class:`~asymmetry.core.data.ratio_validity.RatioValidity`
independently of their full display arrays. Gaussian dataset fitting selects
usable observations within the fit window; array inputs keep strict validation.
``observation_report`` records exclusive used/undefined/zero-variance counts.
``residual_time`` and ``residual_sigma`` describe exactly the residual support
and stated errors used by the objective, including after counts-first merging.

.. autoclass:: asymmetry.core.data.ratio_validity.ObservationReport
   :members:

Asymmetry global
----------------

.. autoclass:: asymmetry.core.fitting.asymmetry_global.GlobalFitResult
   :no-members:
   :no-special-members:

.. autofunction:: asymmetry.core.fitting.asymmetry_global.fit_global

Engine
------

.. autoclass:: asymmetry.core.fitting.engine.AsymmetryScaleWarning
   :no-members:
   :no-special-members:

.. autoclass:: asymmetry.core.fitting.engine.FitCancelledError
   :no-members:
   :no-special-members:

.. autoclass:: asymmetry.core.fitting.engine.FixedFrequencyFieldMismatchWarning
   :no-members:
   :no-special-members:

.. autodata:: asymmetry.core.fitting.engine.POISSON_COST
   :no-value:

.. autoclass:: asymmetry.core.fitting.engine.SharedColumn
   :no-members:
   :no-special-members:

.. autofunction:: asymmetry.core.fitting.engine.drive_minuit

Fit quality
-----------

.. autofunction:: asymmetry.core.fitting.fit_quality.assess_fit_quality

Information criteria
--------------------

.. autofunction:: asymmetry.core.fitting.information_criteria.information_weights

Likelihood
----------

.. autoclass:: asymmetry.core.fitting.likelihood.FitInputError
   :no-members:
   :no-special-members:

.. autoclass:: asymmetry.core.fitting.likelihood.PoissonCounts
   :no-members:
   :no-special-members:
