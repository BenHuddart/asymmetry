Composite models
================

.. currentmodule:: asymmetry.core.fitting


.. autoclass:: asymmetry.core.fitting.composite.ComponentDefinition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: asymmetry.core.fitting.composite.CompositeModel
   :exclude-members: __weakref__, missing_component_names
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   The available time-domain component registry is exposed as
   ``asymmetry.core.fitting.composite.COMPONENTS``.

.. automodule:: asymmetry.core.fitting.diffusion
   :members:
   :undoc-members:

Muon-fluorine models
--------------------

.. automodule:: asymmetry.core.fitting.muon_fluorine.dipolar
   :members:
   :undoc-members:

.. automodule:: asymmetry.core.fitting.muon_fluorine.polarization
   :members:
   :undoc-members:


Fixed time response
-------------------

.. autoclass:: asymmetry.core.fitting.time_response.TimeResponse
   :members:

The optional ``CompositeModel.time_response`` setting round-trips with the
model. :meth:`asymmetry.core.fitting.composite.CompositeModel.with_time_response` creates a new snapshot. See
:ref:`ordinary-time-response` for normalisation, origin, and conditional errors.

Composite
---------

.. py:attribute:: CompositeModel.components
   :module: asymmetry.core.fitting.composite

.. py:attribute:: CompositeModel.param_names
   :module: asymmetry.core.fitting.composite

.. py:attribute:: CompositeModel.fraction_groups
   :module: asymmetry.core.fitting.composite

Expression tree
---------------

.. autofunction:: asymmetry.core.fitting.expression_tree.build_expression_tree

User functions
--------------

.. py:module:: asymmetry.core.fitting.user_functions

.. autofunction:: asymmetry.core.fitting.user_functions.register_parameter_component

.. autoclass:: asymmetry.core.fitting.user_functions.UserFunctionError
   :no-members:
   :no-special-members:

.. py:attribute:: CompositeModel.missing_component_names
   :module: asymmetry.core.fitting.composite
