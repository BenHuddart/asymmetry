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

Superconductivity Models
------------------------

The superconductivity API is organized around normalized superfluid density

.. math::

   \rho_s(T)=\left[\frac{\lambda(0)}{\lambda(T)}\right]^2,

with measured :math:`\sigma(T)` provided in additive and quadrature forms.

Kernel and Gap Helpers
^^^^^^^^^^^^^^^^^^^^^^

Thermal kernel and angular averaging:

.. autofunction:: asymmetry.core.fitting.sc.kernel.energy_integral

.. autofunction:: asymmetry.core.fitting.sc.kernel.superfluid_density_2d

.. autofunction:: asymmetry.core.fitting.sc.kernel.superfluid_density_3d

Gap-amplitude approximations and convention helpers:

.. autofunction:: asymmetry.core.fitting.sc.bcs.delta_bcs

.. autofunction:: asymmetry.core.fitting.sc.bcs.gap_ratio_from_mev

.. autofunction:: asymmetry.core.fitting.sc.bcs.resolve_gap_ratio

Angular gap form factors:

.. autofunction:: asymmetry.core.fitting.sc.gaps.isotropic_s

.. autofunction:: asymmetry.core.fitting.sc.gaps.d_wave

.. autofunction:: asymmetry.core.fitting.sc.gaps.anisotropic_s_cos4

.. autofunction:: asymmetry.core.fitting.sc.gaps.nonmonotonic_d_wave

.. autofunction:: asymmetry.core.fitting.sc.gaps.s_plus_g

Single-Gap And Anisotropic Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isotropic and nodal baselines:

.. autofunction:: asymmetry.core.fitting.sc.models.rho_s_wave

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_wave

.. autofunction:: asymmetry.core.fitting.sc.models.rho_d_wave

.. autofunction:: asymmetry.core.fitting.sc.models.sc_d_wave

Anisotropic variants:

.. autofunction:: asymmetry.core.fitting.sc.models.rho_anisotropic_s_cos4

.. autofunction:: asymmetry.core.fitting.sc.models.sc_anisotropic_s_cos4

.. autofunction:: asymmetry.core.fitting.sc.models.rho_nonmonotonic_d

.. autofunction:: asymmetry.core.fitting.sc.models.sc_nonmonotonic_d

.. autofunction:: asymmetry.core.fitting.sc.models.rho_s_plus_g

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_plus_g

Additional unconventional examples:

.. autofunction:: asymmetry.core.fitting.sc.models.rho_extended_s

.. autofunction:: asymmetry.core.fitting.sc.models.sc_extended_s

.. autofunction:: asymmetry.core.fitting.sc.models.rho_p_wave_axial

.. autofunction:: asymmetry.core.fitting.sc.models.sc_p_wave_axial

.. autofunction:: asymmetry.core.fitting.sc.models.rho_p_wave_polar_3d

Two-Gap And Alpha Models
^^^^^^^^^^^^^^^^^^^^^^^^

Weighted-sum two-gap models (including MgB2-style s+s decomposition):

.. autofunction:: asymmetry.core.fitting.sc.models.sc_two_gap_ss

.. autofunction:: asymmetry.core.fitting.sc.models.sc_two_gap_sd

Alpha-model scaling:

.. autofunction:: asymmetry.core.fitting.sc.models.sc_alpha_model

Quadrature Sigma Models
^^^^^^^^^^^^^^^^^^^^^^^

These variants are intended for workflows where superconducting and
non-superconducting linewidth channels are modeled as independent Gaussian
contributions added in quadrature.

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_wave_q

.. autofunction:: asymmetry.core.fitting.sc.models.sc_d_wave_q

.. autofunction:: asymmetry.core.fitting.sc.models.sc_s_plus_g_q

lambda and sigma conversion helpers:

.. autofunction:: asymmetry.core.fitting.sc.constants.sigma_to_lambda_nm

.. autofunction:: asymmetry.core.fitting.sc.constants.lambda_nm_to_sigma_us

.. autofunction:: asymmetry.core.fitting.sc.models.rho_to_lambda_inv_sq

.. autofunction:: asymmetry.core.fitting.sc.models.rho_to_lambda

.. autofunction:: asymmetry.core.fitting.sc.models.lambda_inv_sq_from_model

.. autofunction:: asymmetry.core.fitting.sc.models.lambda_from_model

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
