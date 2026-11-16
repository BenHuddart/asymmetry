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

Scalar fit adapters and ``fit_parameter_model`` accept an optional raising
``cancel_check``. Use an established cancellation exception such as
``asymmetry.core.fitting.engine.FitCancelledError``; it propagates through
solver and seed paths rather than returning a failed fit or partial covariance.
The default leaves successful numerical methods and results unchanged.
``ScanBaselineResult.evaluate_baseline``, ``baseline_correction_operator``, and
``baseline_corrected_covariance`` accept the same poll when these derived
operations are needed independently of a peak fit. Their property counterparts
retain the ordinary no-callback API. Work-unit limits are described in
:doc:`/reference/alc_mode`.

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

Captured Gaussian-width inputs
------------------------------

``fit_parameter_model`` accepts optional ``width_context`` alongside its
already transformed arrays. It checks selected known Gaussian sources at the
Brandt second-moment consumers connected to the output by additive, subtractive,
or quadrature branches, without applying another scale factor. Product and ratio
branches remain unclassified.
``ParameterGroupData.width_context`` supplies the same contract per group for
``global_fit_parameter_model``. Unknown context remains usable; it does not
certify a convention.

The workflow ``fit_trend`` accepts ``y_transform=AxisTransform.custom("sqrt(2)*x")``
and applies that existing transform to both values and errors. The GUI supplies
the actual transformed arrays and captured lens instead. Recorded results retain
the context of the observations that entered the fit, so later model or lens
edits do not relabel history. See :ref:`trend-width-inputs` for the refusal and
legacy-data policy.

.. autoclass:: asymmetry.core.fitting.sc.trend_width.WidthSource

.. autoclass:: asymmetry.core.fitting.sc.trend_width.TrendWidthContext
