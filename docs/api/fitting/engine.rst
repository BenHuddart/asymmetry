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
