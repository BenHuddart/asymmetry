Parameter Trending
==================

Use parameter models to fit extracted fit parameters as a function of field
or temperature.

Available Basis Components
--------------------------

.. code-block:: python

   from asymmetry.core.fitting import component_names_for_x

   print(component_names_for_x("field"))
   print(component_names_for_x("temperature"))

Build a Parameter Composite Model
---------------------------------

.. code-block:: python

   from asymmetry.core.fitting import ParameterCompositeModel

   model = ParameterCompositeModel(
       component_names=["DiffusionLF_2D", "Redfield", "Lambda_bg"],
       operators=["+", "+"],
   )
   print(model.formula_string())

Single-Series Fit
-----------------

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import Parameter, ParameterSet, fit_parameter_model

   field = np.array([20.0, 50.0, 100.0, 200.0, 400.0, 800.0])
   values = np.array([2.7, 2.2, 1.6, 1.1, 0.7, 0.5])
   errors = np.full_like(values, 0.1)

   params = ParameterSet([
       Parameter("A", value=3.0, min=0.0),
       Parameter("D_2D", value=0.7, min=0.0),
       Parameter("D_perp", value=0.0, min=0.0),
       Parameter("D", value=10.0, min=0.0),
       Parameter("nu", value=10.0, min=0.0),
       Parameter("m", value=2.0, min=0.0),
       Parameter("lambda_BG", value=0.0, min=0.0),
   ])

   result = fit_parameter_model(field, values, errors, model, params)
   print(result.success, result.reduced_chi_squared)

Cross-Group Fitting
-------------------

For jointly fitting multiple groups with shared and local parameters, use
``global_fit_parameter_model`` with ``ParameterGroupData`` entries.

.. code-block:: python

   from asymmetry.core.fitting.parameter_models import (
       ParameterGroupData,
       global_fit_parameter_model,
   )

   groups = [
       ParameterGroupData(
           group_id="g1",
           group_name="Sample A",
           x=field,
           y=values,
           yerr=errors,
           group_variable_value=5.0,
       ),
       ParameterGroupData(
           group_id="g2",
           group_name="Sample B",
           x=field,
           y=values * 1.1,
           yerr=errors,
           group_variable_value=15.0,
       ),
   ]

   result = global_fit_parameter_model(
       groups=groups,
       model=model,
       global_params=["D_2D", "D", "nu", "m"],
       local_params=["A", "lambda_BG"],
       fixed_params={"D_perp": 0.0},
   )
   print(result.success)

Runnable Example
----------------

See ``examples/parameter_trending.py`` for a complete executable script.

Superconducting Gap Models
--------------------------

For TF-muSR superconducting penetration-depth analysis via
temperature-dependent :math:`\sigma(T)`, see
:doc:`sc_penetration_depth`.
