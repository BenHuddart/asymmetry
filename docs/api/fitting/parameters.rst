Parameters
==========

.. currentmodule:: asymmetry.core.fitting


.. autoclass:: asymmetry.core.fitting.parameters.Parameter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.parameters.ParameterSet
   :members:
   :undoc-members:
   :show-inheritance:

Fit results
-----------

.. autoclass:: asymmetry.core.fitting.engine.FitResult
   :members:

Parameter carry
---------------

.. py:module:: asymmetry.core.fitting.parameter_carry

.. autoclass:: asymmetry.core.fitting.parameter_carry.GroupAmplitude
   :no-members:
   :no-special-members:

.. autofunction:: asymmetry.core.fitting.parameter_carry.carry_parameter_set

Seeding
-------

.. autofunction:: asymmetry.core.fitting.seeding.seed_parameters

.. autofunction:: asymmetry.core.fitting.seeding.seed_trend_parameters

.. autoclass:: asymmetry.core.fitting.seeding.Seed
   :no-members:
   :no-special-members:

Parameters
----------

.. autoclass:: asymmetry.core.fitting.parameters.AffineTie
   :no-members:
   :no-special-members:

.. py:data:: PARAM_INFO_REGISTRY
   :module: asymmetry.core.fitting.parameters

   Parameter symbols, units, and physical descriptions.

.. autofunction:: asymmetry.core.fitting.parameters.get_param_info
