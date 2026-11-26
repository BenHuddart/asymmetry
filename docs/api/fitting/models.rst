Models
======

.. currentmodule:: asymmetry.core.fitting

.. _python-damped-precession-fit:

Fit damped precession from a builder expression
-----------------------------------------------

Pass an expression from the GUI model builder to
:meth:`~asymmetry.core.fitting.composite.CompositeModel.from_expression`.
Read ``model.param_names`` before seeding the fit. The builder component
``Oscillatory`` takes ``A``, ``frequency``, and ``phase`` in that order.
In this expression the builder names the amplitude ``A_1``, uses
``Exponential`` as a unit-amplitude envelope with rate ``Lambda``, and
adds the ``Constant`` background ``A_bg``.

The example generates a noiseless curve independently of the model callable.
The time grid is in μs, the asymmetry is in percent, and the illustrative
standard errors are in percentage points. The fitted curve recovers the
generating curve to within 0.002 percentage points at every time.

.. code-block:: python
   :name: python-damped-precession-example

   import numpy as np

   from asymmetry.core.fitting import CompositeModel, FitEngine, Parameter, ParameterSet

   model = CompositeModel.from_expression("Oscillatory * Exponential + Constant")
   print(model.param_names)  # ['A_1', 'frequency', 'phase', 'Lambda', 'A_bg']

   time = np.linspace(0.0, 8.0, 161)
   asymmetry = 20.0 * np.cos(2 * np.pi * 1.2 * time + 0.25) * np.exp(-0.35 * time) + 3.0
   error = np.full_like(time, 0.2)
   params = ParameterSet([
       Parameter("A_1", value=18.0, min=0.0),
       Parameter("frequency", value=1.18, min=0.0),
       Parameter("phase", value=0.0),
       Parameter("Lambda", value=0.25, min=0.0),
       Parameter("A_bg", value=1.0),
   ])

   result = FitEngine().fit_arrays(time, asymmetry, error, model.function, params)
   assert result.success
   best_fit = model.function(time, **{p.name: p.value for p in result.parameters})
   np.testing.assert_allclose(best_fit, asymmetry, atol=0.002, rtol=0)

For a loaded dataset, pass the dataset, model callable, and parameters to
:meth:`~asymmetry.core.fitting.engine.FitEngine.fit`. See the
:doc:`/reference/cookbook` for other scripting recipes and the
:doc:`/reference/fit_functions/index` guide for model selection.

Models
------

Use ``MODELS["Oscillatory"].function`` when the damped sinusoid with a
constant baseline is the whole model. Its parameters after ``t`` are
``A0``, ``frequency``, ``phase``, ``Lambda``, and ``baseline`` in that
order. Damping and background are both part of this callable. See the
:ref:`standalone and builder name map <components-vs-models>` for other
equivalent expressions.

.. py:data:: MODELS
   :module: asymmetry.core.fitting.models

   Registry of built-in time-domain models. Each entry includes its callable,
   ordered parameter names, and default seeds.

.. automodule:: asymmetry.core.fitting.models
   :exclude-members: LINEAR_PARAM_ROLE_NAMES, default_linear_params, ModelDefinition, MODELS, LARMOR_RAD_PER_US_PER_GAUSS, field_decoupling_threshold_gauss

Mu kinetics
-----------

.. autofunction:: asymmetry.core.fitting.mu_kinetics.fit_arrhenius

.. autofunction:: asymmetry.core.fitting.mu_kinetics.fit_mu_relaxation_series

.. autofunction:: asymmetry.core.fitting.mu_kinetics.mu_relaxation_from_amplitude

.. py:attribute:: MuRelaxationSeriesResult.shared_amplitude
   :module: asymmetry.core.fitting.mu_kinetics

Model metadata and solver support
---------------------------------

.. autoclass:: asymmetry.core.fitting.models.ModelDefinition

.. autofunction:: asymmetry.core.fitting.models.default_linear_params

.. autodata:: asymmetry.core.fitting.models.LINEAR_PARAM_ROLE_NAMES

.. autodata:: asymmetry.core.fitting.models.LARMOR_RAD_PER_US_PER_GAUSS

.. autofunction:: asymmetry.core.fitting.models.field_decoupling_threshold_gauss
