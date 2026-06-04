Parameter Trending
==================

.. image:: /_generated/screenshots/parameter_trending_mgb2.png
   :alt: σ(T) plot of synthetic MgB₂ data with the two-gap SC fit overlaid
   :width: 100%

*Synthetic MgB₂ σ(T) data (Tc = 36 K) with the SC_TwoGap_SS parametric*
*model fit overlaid (small and large gap ratios 1.1 and 2.3 with weight*
*0.55, the canonical MgB₂ alpha-model decomposition; Niedermayer et al.*
*Phys. Rev. B 65, 094512, 2002). The σ(T) → λ(T) inversion that follows*
*from this fit is discussed in* :doc:`sc_penetration_depth`.

Parameter trending is the second stage of any temperature or field scan
analysis. Once a series of runs has been individually (or globally) fitted
in the time domain, the extracted parameters themselves carry the
physically interesting information: an order parameter as a function of
temperature, a relaxation rate as a function of field, a precession
frequency tracking magnetic ordering. The parameter-trending framework
fits those *derived* curves with parametric models — power laws, Arrhenius
expressions, Redfield denominators, superconducting gap models, transport
forms — and reports the resulting physical quantities (critical
temperature, activation energy, gap ratios, dimensionality of motion)
with uncertainties.

Frequency-domain global fits use the same panel.  Peak centres and widths from
Fourier spectra appear as ``nu0`` and ``fwhm`` in MHz, with derived field
equivalents ``B0`` and ``Bwid`` for plotting or fitting spectral shifts and
broadening directly against field, temperature, or run number.

Representation-Aware Trending
------------------------------

The Fit Parameters panel automatically tracks which *representation* is active
in the central workspace.  The **Showing:** row at the top of the panel
indicates the current representation — ``F-B Asymmetry``, ``Detector Groups``,
``FFT``, or ``MaxEnt`` — and the panel's series buttons below show only the fit
series that belong to that representation.

Switching the central workspace view (e.g. from **F-B asymmetry** to
**Individual Groups**) refreshes the panel instantly to show the grouped-fit
series instead of the asymmetry ones.  Switching back restores the asymmetry
series.  Each representation maintains its own independent set of series, x-axis
selection, and derived composite parameters, so there is no risk of cross-
contamination between a time-domain and a frequency-domain analysis.

When a series button is selected, the **Data Browser** highlights (in amber) the
runs that belong to that series, making it easy to trace which runs contributed
to the current trend.

Typical use cases include locating :math:`T_c` and a critical exponent by
fitting :math:`\nu(T)` from a ZF series with a ``CriticalDivergence`` form
(:doc:`workflows/temperature_scan_magnetism`); inverting the TF
second-moment :math:`\sigma(T)` into a London penetration depth using one
of the ``SC_*`` gap models (:doc:`workflows/superconductor_penetration_depth`
and :doc:`sc_penetration_depth`); fitting an Arrhenius form to
:math:`\lambda(T)` across a temperature scan for motional narrowing;
and field sweeps in either LF or TF geometry, where the trending panel
handles parameter-vs-:math:`B` identically to parameter-vs-:math:`T`. The
GUI panel is the right tool for interactive exploration and quick model
selection; exporting to a Python script (the snippets below) is the right
tool for reproducible, publication-grade analyses and for unusual
parametric models that are not in the built-in registry.

Available Basis Components
--------------------------

.. code-block:: python

   from asymmetry.core.fitting import component_names_for_x

   print(component_names_for_x("field"))
   print(component_names_for_x("temperature"))

Build a Parameter Composite Model
---------------------------------

In the GUI parameter-trending workflow, the **Edit Model...** action opens the
same expression-oriented builder used for time-domain composite functions.
Each basis model is inserted as a single function token, then combined with
``+``, ``-``, ``*``, ``/``, and parentheses. The **Info** button reports the
documentation for the currently selected basis model, including superconducting
gap-model details when relevant.

The builder validates the expression in real time and shows the expanded
``y(x)`` preview before you accept it.

.. code-block:: python

   from asymmetry.core.fitting import ParameterCompositeModel

   model = ParameterCompositeModel(
       component_names=["DiffusionLF_2D", "Redfield", "Lambda_bg"],
       operators=["+", "+"],
   )
   print(model.formula_string())

Grouped expressions are also supported programmatically and in the GUI, for
example ``Linear + (Arrhenius * Constant)``.

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

Composite Parameters in the Fit Parameters Panel
------------------------------------------------

In the GUI Fit Parameters panel, you can define a derived parameter from
existing fitted parameters using **Create Composite Parameter**.

After selecting a derived parameter in the Y-parameter list, you can also use
**Edit Selected Composite** and **Remove Selected Composite** to manage saved
definitions.

The expression builder supports:

- Arithmetic: ``+``, ``-``, ``*``, ``/``, ``^``
- Parentheses and numeric constants
- Built-in constants: ``pi``, ``e``
- Built-in functions: ``sin``, ``cos``, ``tan``, ``asin``, ``acos``,
  ``atan``, ``sinh``, ``cosh``, ``tanh``, ``exp``, ``log``, ``log10``,
  ``sqrt``, ``abs``

Expressions are parsed safely (no arbitrary code execution) and validated in
real time against available fitted parameter names.

Derived parameters are integrated into the same workflow as ordinary fit
parameters:

- They appear in the Y-parameter selector and fitted-parameter table.
- They can be plotted and used with parameter-model fitting.
- They are saved/restored with project state.
- They are recomputed automatically when source fit parameters change.

Uncertainties are propagated using first-order error propagation:

.. math::

    \sigma_f^2 = J\,\Sigma\,J^T

where :math:`J` is the gradient of the expression with respect to source
parameters and :math:`\Sigma` is the parameter covariance matrix when
available (falling back to diagonal variances otherwise).

Runnable Example
----------------

See ``examples/parameter_trending.py`` for a complete executable script.

Superconducting Gap Models
--------------------------

For TF-muSR superconducting penetration-depth analysis via
temperature-dependent :math:`\sigma(T)`, see
:doc:`sc_penetration_depth`.
