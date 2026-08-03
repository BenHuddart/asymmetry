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

