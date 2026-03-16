Fitting
=======

.. currentmodule:: asymmetry.core.fitting

Fit Engine
----------

.. autoclass:: asymmetry.core.fitting.engine.FitEngine
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

.. automodule:: asymmetry.core.fitting.models
   :members:
   :undoc-members:

Composite Models
----------------

.. autoclass:: asymmetry.core.fitting.composite.ComponentDefinition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.composite.CompositeModel
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   The available time-domain component registry is exposed as
   ``asymmetry.core.fitting.composite.COMPONENTS``.

.. automodule:: asymmetry.core.fitting.diffusion
   :members:
   :undoc-members:

Parameter Trend Models
----------------------

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

Parameters
----------

.. autoclass:: asymmetry.core.fitting.parameters.Parameter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameters.ParameterSet
   :members:
   :undoc-members:
   :show-inheritance:

Fit Results
-----------

.. autoclass:: asymmetry.core.fitting.engine.FitResult
   :members:
   :undoc-members:
   :show-inheritance:
