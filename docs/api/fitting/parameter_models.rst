Parameter trend models
======================

.. currentmodule:: asymmetry.core.fitting


.. autoclass:: asymmetry.core.fitting.parameter_models.ParameterModelComponentDefinition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameter_models.ParameterCompositeModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameter_models.ParameterModelFitResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameter_models.ModelFitRange
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameter_models.ParameterModelFit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameter_models.ParameterModelFitExecution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameter_models.ParameterGroupData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameter_models.CrossGroupFitResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: asymmetry.core.fitting.parameter_models.component_names_for_x

.. autofunction:: asymmetry.core.fitting.parameter_models.fit_parameter_model

.. autofunction:: asymmetry.core.fitting.parameter_models.global_fit_parameter_model

.. autofunction:: asymmetry.core.fitting.parameter_models.evaluate_parameter_model_fit

.. note::

   The available parameter-trend component registry is exposed as
   ``asymmetry.core.fitting.parameter_models.PARAMETER_MODEL_COMPONENTS``.

   The angle-only :math:`K(\theta)` basis models ``KnightAnisotropy``,
   ``AngularCos2``, and ``AngularFourier2`` are registered alongside the
   others (``scope="angle"``).


Field-scan baseline and peak inference
--------------------------------------

These adapters fit original scan observations. Derivative scans support
inspection/export only: their marginal errors do not supply an independent
likelihood. The common fit-input boundary also checks the actual selected
window's scalar validity and positional source sharing before seeding.
See :doc:`/reference/alc_mode` for baseline propagation and scan units.

.. autofunction:: asymmetry.core.fitting.fit_scan_baseline

.. autoclass:: asymmetry.core.fitting.ScanBaselineResult
   :members:

.. autofunction:: asymmetry.core.fitting.fit_scan_peaks

.. autoclass:: asymmetry.core.fitting.ScanPeakFit
   :members:

Captured scalar-scan executions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``asymmetry.core.fitting.scan_history`` stores completed baseline, peak, and RF
executions separately from mutable form settings. Records retain fitted support,
parameters, available covariance/statistics, computation attribution, and sampled
predictions; decoding invokes no optimiser. A peak references the baseline
execution used for its first-order covariance propagation. Unknown historical
revisions remain unknown. Explicit refitting creates a new execution; persistence
does not certify acquisition replay or add uncertainty information.
