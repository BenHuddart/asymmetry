Fitting
=======

Fit models to your μSR data using the integrated iminuit framework.

When fitting through the GUI, the plot panel's current bunch factor is applied
before the dataset is handed to the fitter. In other words, if the plot is
showing a rebinned dataset, the next GUI fit will use that rebinned dataset.
If you want to fit the full-resolution data and only simplify the view
afterwards, run the fit with ``Bunch = 1`` and then increase the bunch factor;
the fit curve remains overlaid on the plot.

Basic Fitting Workflow
-----------------------

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet
   from asymmetry.core.io import load
   
   # Load data
   dataset = load("data.wim")
   
   # Create fit engine
   engine = FitEngine()
   
   # Select model
   model = MODELS["ExponentialRelaxation"]
   
   # Set up parameters
   params = ParameterSet()
   params.add(Parameter(name="A0", value=0.2, min=0, max=1))
   params.add(Parameter(name="Lambda", value=0.5, min=0))
   params.add(Parameter(name="baseline", value=0.0, min=-0.1, max=0.1))
   
   # Perform fit
   result = engine.fit(dataset, model.function, params)
   
   # Print results
   print(f"Success: {result.success}")
   print(f"χ²: {result.chi_squared:.4f}")
   print(f"χ²ᵣ: {result.reduced_chi_squared:.4f}")
   for param in result.parameters:
       error = result.uncertainties.get(param.name, 0)
       print(f"{param.name}: {param.value} ± {error:.6f}")

Mathematical Notation in Documentation
--------------------------------------

Yes. The documentation supports LaTeX math notation via MathJax.

Use inline math for symbols and short expressions, for example
:math:`A_0`, :math:`\lambda`, :math:`\Delta`, :math:`\chi_r^2`.

Use display blocks for full equations:

.. math::

   A(t) = A_0 e^{-\lambda t} + A_{\mathrm{bg}}

In prose, keep the code-facing parameter names in monospaced form where needed,
for example ``Lambda`` and ``baseline``, and pair them with their physical
symbols :math:`\lambda` and :math:`A_{\mathrm{bg}}`.

Available Models
----------------

The following built-in models are available in the MODELS registry:

ExponentialRelaxation
~~~~~~~~~~~~~~~~~~~~~

Simple exponential relaxation:

.. math::

   A(t) = A_0 e^{-\lambda t} + B

Parameters: ``A0`` (initial asymmetry), ``Lambda`` (:math:`\lambda`, relaxation rate), ``baseline`` (:math:`A_{\mathrm{bg}}`)

.. code-block:: python

   model = MODELS["ExponentialRelaxation"]

GaussianRelaxation
~~~~~~~~~~~~~~~~~~

Gaussian Kubo-Toyabe relaxation:

.. math::

   A(t) = A_0 e^{-(\sigma t)^2} + A_{\mathrm{bg}}

Parameters: ``A0``, ``sigma`` (:math:`\sigma`, Gaussian relaxation rate), ``baseline``

.. code-block:: python

   model = MODELS["GaussianRelaxation"]

Oscillatory
~~~~~~~~~~~

Damped oscillation with exponential decay:

.. math::

   A(t) = A_0 e^{-\lambda t} \cos(2\pi f t + \phi) + A_{\mathrm{bg}}

Parameters: ``A0``, ``frequency`` (:math:`f`, MHz), ``phase`` (:math:`\phi`), ``Lambda`` (:math:`\lambda`), ``baseline``

.. code-block:: python

   model = MODELS["Oscillatory"]

StretchedExponential
~~~~~~~~~~~~~~~~~~~~

Stretched exponential (Kohlrausch) relaxation:

.. math::

   A(t) = A_0 e^{-(\lambda t)^\beta} + A_{\mathrm{bg}}

Parameters: ``A0``, ``Lambda`` (:math:`\lambda`), ``beta`` (:math:`\beta`, stretching exponent), ``baseline``

.. code-block:: python

   model = MODELS["StretchedExponential"]

StaticGKT_ZF
~~~~~~~~~~~~

Static Gaussian Kubo-Toyabe function for zero field:

.. math::

   A(t) = A_0 \left[\frac{1}{3} + \frac{2}{3}(1-\Delta^2 t^2)e^{-\Delta^2 t^2/2}\right] + A_{\mathrm{bg}}

Parameters: ``A0``, ``Delta`` (:math:`\Delta`), ``baseline``

.. code-block:: python

   model = MODELS["StaticGKT_ZF"]

Custom Models
~~~~~~~~~~~~~

Define your own model function:

.. code-block:: python

   import numpy as np
   
   def my_model(t, amplitude, tau, frequency):
       return amplitude * np.exp(-t/tau) * np.sin(2*np.pi*frequency*t)
   
   params = ParameterSet()
   params.add(Parameter(name="amplitude", value=0.2))
   params.add(Parameter(name="tau", value=1.0, min=0))
   params.add(Parameter(name="frequency", value=5.0, min=0))
   
   result = engine.fit(dataset, my_model, params)

Parameter Control
-----------------

Setting Bounds
~~~~~~~~~~~~~~

Constrain parameter ranges:

.. code-block:: python

   from asymmetry.core.fitting.parameters import Parameter
   
      param = Parameter(
         name="Lambda",
         value=0.5,    # Initial value
         min=0.0,      # Lower bound
         max=10.0      # Upper bound
      )

Fixing Parameters
~~~~~~~~~~~~~~~~~

Hold parameters constant during fitting:

.. code-block:: python

   # Fix a parameter at a specific value
   param = Parameter(name="A0", value=0.2, fixed=True)
   
   # Or set after creation
   param = Parameter(name="A0", value=0.2)
   param.fixed = True

In the GUI, use the "Fix" checkbox in the parameter table to fix/unfix parameters.

Analyzing Fit Results
----------------------

Accessing Fit Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = engine.fit(dataset, model.function, params)
   
   print(f"χ²: {result.chi_squared:.4f}")
   print(f"χ²ᵣ: {result.reduced_chi_squared:.4f}")
   print(f"Success: {result.success}")
   print(f"Message: {result.message}")

A good fit typically has χ²ᵣ ≈ 1. Values much larger than 1 indicate poor fit 
or underestimated errors. Values much less than 1 may indicate overestimated errors.

Parameter Values and Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Best-fit parameters with uncertainties
   for name, param in result.parameters.items():
       error = result.uncertainties.get(name, 0)
       print(f"{name}: {param.value:.6f} ± {error:.6f}")

Plotting Results
~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Generate fine time grid for smooth fit curve
   t_fit = np.linspace(dataset.time.min(), dataset.time.max(), 500)
   
   # Compute model with best-fit parameters
   fit_params = {p.name: p.value for p in result.parameters}
   y_fit = model.function(t_fit, **fit_params)
   
   # Plot data and fit
   plt.errorbar(dataset.time, dataset.asymmetry, yerr=dataset.error, 
                fmt='o', label='Data', alpha=0.6)
   plt.plot(t_fit, y_fit, 'r-', linewidth=2, label='Fit')
   plt.xlabel("Time (μs)")
   plt.ylabel("Asymmetry")
   plt.title(f"χ²ᵣ = {result.reduced_chi_squared:.3f}")
   plt.legend()
   plt.show()

Residuals
~~~~~~~~~

.. code-block:: python

   # Evaluate model at data points
   fit_values = model.function(dataset.time, **fit_params)
   residuals = (dataset.asymmetry - fit_values) / dataset.error
   
   plt.figure(figsize=(10, 4))
   plt.plot(dataset.time, residuals, 'o', alpha=0.6)
   plt.axhline(0, color='k', linestyle='--')
   plt.axhline(2, color='r', linestyle=':', alpha=0.5)
   plt.axhline(-2, color='r', linestyle=':', alpha=0.5)
   plt.xlabel("Time (μs)")
   plt.ylabel("Normalized Residuals")
   plt.title("Fit Residuals (in units of σ)")
   plt.grid(alpha=0.3)
   plt.show()

Good residuals should be randomly scattered around zero with most points 
within ±2σ.

Advanced Topics
---------------

Restricting Fit Range
~~~~~~~~~~~~~~~~~~~~~

Fit only a portion of the data:

.. code-block:: python

   # Fit only data between 0.1 and 5.0 μs
   result = engine.fit(dataset, model.function, params, 
                      t_min=0.1, t_max=5.0)

Global Fitting
~~~~~~~~~~~~~~

Fit multiple datasets simultaneously with shared and per-dataset parameters.
This is essential for analysing field or temperature series where some parameters
(e.g., initial asymmetry) are common to all runs while others (e.g., relaxation
rate) vary.

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet
   from asymmetry.core.io import load

   datasets = [load(f"run_{i}.wim") for i in range(1, 4)]
   model = MODELS["ExponentialRelaxation"]
   engine = FitEngine()

   # Build initial parameter sets for each dataset
   initial_params = {}
   for ds in datasets:
       ps = ParameterSet()
       ps.add(Parameter(name="A0", value=0.2, min=0, max=1))
       ps.add(Parameter(name="Lambda", value=0.5, min=0))
       ps.add(Parameter(name="baseline", value=0.0, min=-0.1, max=0.1))
       initial_params[ds.run_number] = ps

   # A0 is shared; Lambda and baseline vary per dataset
   results_dict, global_result = engine.global_fit(
       datasets=datasets,
       model_fn=model.function,
       global_params=["A0"],
       local_params=["Lambda", "baseline"],
       initial_params=initial_params,
       t_min=0.1,
       t_max=10.0,
   )

   # Inspect global (shared) parameters
   for p in global_result:
       print(f"[global] {p.name} = {p.value:.6f}")

   # Inspect per-dataset results
   for run_number, result in results_dict.items():
       print(f"\nRun {run_number}: χ²ᵣ = {result.reduced_chi_squared:.3f}")
       for p in result.parameters:
           err = result.uncertainties.get(p.name, 0)
           print(f"  {p.name} = {p.value:.6f} ± {err:.6f}")

In the GUI, the "Global" tab automates this workflow. Select multiple datasets,
mark shared parameters with the "Global" checkbox, and click "Run Global Fit".
The results are displayed in the **Fitted Parameters** panel where you can
inspect trends, export to CSV, or generate publication-quality GLE figures.

Monte Carlo Error Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bootstrap resampling for error estimation:

.. code-block:: python

   n_bootstrap = 100
   bootstrap_results = []

   for i in range(n_bootstrap):
       # Resample data
       resampled_data = data + np.random.randn(len(data)) * error

       # Fit
       result = engine.fit(model, time, resampled_data, error, params)
       bootstrap_results.append(result)

   # Analyze distribution of parameters

Diffusive LF Relaxation (Field-Series Model)
--------------------------------------------

For field-dependent relaxation-rate analysis, the parameter-model workflow now
includes diffusion-based LF components:

- ``DiffusionLF_1D``
- ``DiffusionLF_2D``
- ``DiffusionLF_3D``

These models fit :math:`\lambda(B_{LF})` directly (not time-domain asymmetry).

Model equations
~~~~~~~~~~~~~~~

Following Phys. Rev. B 106, L060401 (2022):

.. math::

   \lambda_{2D}(B_{LF}) = \frac{A^2}{4} J(\omega_e),
   \quad \omega_e = \gamma_e B_{LF}

.. math::

   \lambda_{0D}(B_{LF}) = \frac{D^2}{4}\,\frac{2/\nu}{1 + (\omega_\mu/\nu)^m},
   \quad \omega_\mu = \gamma_\mu B_{LF}

and the total fitted field dependence is represented as

.. math::

   \lambda(B_{LF}) = \lambda_{2D}(B_{LF}) + \lambda_{0D}(B_{LF}) + \lambda_{BG} + \lambda_{LCR}(B_{LF})

with

.. math::

   \lambda_{LCR}(B_{LF}) = f\,G(B_{LF}; B_0; B_{wid})

In the parameter-model builder, these are represented as separate basis
functions: ``DiffusionLF_*`` (dynamic diffusion term), ``Redfield`` (0D
dynamic term), ``Lambda_bg`` (constant background), and ``GaussianLCR``
(LCR Gaussian term).

The autocorrelation for n-dimensional diffusion follows Pratt,
J. Phys.: Conf. Ser. 2462 012038 (2023):

.. math::

   S_{nD}(t) = [e^{-2D_{nD}t} I_0(2D_{nD}t)]^n
   [e^{-2D_{\perp}t} I_0(2D_{\perp}t)]^{3-n},\quad n\in\{1,2,3\}

The implementation uses the one-sided cosine-transform convention:

.. math::

   J(\omega) = 2\int_0^{\infty} S_{nD}(t)\cos(\omega t)\,dt

Units and parameters
~~~~~~~~~~~~~~~~~~~~

- ``B_LF``: Gauss (G)
- ``A``: MHz (numerically equivalent to :math:`\mu s^{-1}`)
- ``D_2D``: :math:`\mu s^{-1}`
- ``D_perp``: :math:`\mu s^{-1}`
- ``D`` (Redfield amplitude): MHz
- ``nu`` (Redfield fluctuation rate): MHz
- ``m`` (Redfield exponent): dimensionless
- ``lambda_BG`` (``Lambda_bg`` component): :math:`\mu s^{-1}`

Notes on model choice
~~~~~~~~~~~~~~~~~~~~~

- Use ``DiffusionLF_2D`` for quasi-2D diffusion.
- Use ``DiffusionLF_1D`` or ``DiffusionLF_3D`` when dimensionality is clear
  from physical context.
- ``D_perp`` enables anisotropic slow diffusion; for isotropic cases you can
  keep ``D_perp = 0`` (or fix it in the fit).
- ``lambda_BG`` adds a field-independent contribution.

Example usage
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

   b_lf = np.linspace(10.0, 3000.0, 60)  # Gauss
   model = PARAMETER_MODEL_COMPONENTS["DiffusionLF_2D"]

   lam_dyn = model.function(
       b_lf,
       A=0.8,
       D_2D=2.0,
       D_perp=0.0,
   )

   lam = lam_dyn + 0.05  # Add Lambda_bg contribution separately

   print(lam[:5])

Tips for Good Fits
------------------

1. **Start with reasonable initial guesses**: Use physical intuition
2. **Set appropriate bounds**: Prevent unphysical values
3. **Check residuals**: Should be randomly distributed around zero
4. **Compare models**: Use chi-squared and AIC/BIC for model selection
5. **Validate uncertainties**: Bootstrap if parameter errors seem too small

Parameter Model Fitting
-----------------------

Once you have run fits across a series of runs (e.g., a field sweep or
temperature scan), the extracted parameters can themselves be fitted as a
function of the scan variable using the **Parameter Model Fitting** framework.
This is exposed in the GUI via the **Fitted Parameters** panel and is also
fully scriptable.

Available basis functions
~~~~~~~~~~~~~~~~~~~~~~~~~

The registry ``PARAMETER_MODEL_COMPONENTS`` contains the following basis
functions, organised by the scan variable they are designed for.

Common (field, temperature, or run number)
``````````````````````````````````````````

``Constant``
   :math:`f(x) = c`

   Parameters: ``c``

``Linear``
   :math:`f(x) = m x + b`

   Parameters: ``m``, ``b``

``PowerLaw``
   :math:`f(x) = a |x|^n + c`

   Parameters: ``a``, ``n``, ``c``

``ExponentialDecay``
   :math:`f(x) = a \exp(-x/\tau) + c`

   Parameters: ``a``, ``tau``, ``c``

Temperature-specific
````````````````````

``Arrhenius``
   :math:`f(T) = a \exp\!\left(-\frac{E_a}{k_B T}\right)`

   Parameters: ``a``, ``Ea`` (activation energy in meV)

``CriticalDivergence``
   :math:`f(T) = a |T - T_c|^{-\nu} + c`

   Parameters: ``a``, ``Tc``, ``nu``, ``c``

Field-specific
``````````````

``Redfield``
   :math:`f(B) = \dfrac{D^2}{4}\,\dfrac{2/\nu}{1 + (\omega_\mu/\nu)^m}`
   with :math:`\omega_\mu = \gamma_\mu B`.

   Parameters: ``D``, ``nu``, ``m``

   Notes: ``m`` is dimensionless.

``Lambda_bg``
   Constant background term,
   :math:`f(B) = \lambda_{BG}`.

   Parameters: ``lambda_BG``

``GaussianLCR``
   Gaussian level-crossing resonance term in the notation of Eq. (4)
   of Phys. Rev. Lett. 135, 046704 (2025):
   :math:`\lambda_{LCR}(B) = f\,G(B; B_0; B_{wid})`.

   Parameters: ``f``, ``B0``, ``Bwid``

``Lorentzian``
   Simple empirical Lorentzian line-shape,
   :math:`f(B)=\dfrac{a}{1+(B/B_0)^2}+c`.
   This is retained for convenience and is distinct from the paper-form
   ``Redfield`` term above.

   Parameters: ``a``, ``B0``, ``c``

Querying available components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from asymmetry.core.fitting.parameter_models import component_names_for_x

   print(component_names_for_x("field"))        # includes Redfield, Lambda_bg, GaussianLCR, Lorentzian
   print(component_names_for_x("temperature"))  # includes Arrhenius, CriticalDivergence
   print(component_names_for_x("run"))          # common components only

In the GUI model builder, field-series component choices are also filtered by
the selected y-parameter to avoid redundant constant terms:

- For Lambda-like y-parameters, ``Lambda_bg`` is shown and ``Constant`` is hidden.
- For non-Lambda y-parameters, ``Constant`` is shown and ``Lambda_bg`` is hidden.

Building a composite parameter model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~asymmetry.core.fitting.parameter_models.ParameterCompositeModel`
accepts a list of component names and optional operators (``+``, ``-``,
``*``, ``/``) to combine them.  Operator precedence follows normal arithmetic
rules: ``*`` and ``/`` are evaluated before ``+`` and ``-``.

.. code-block:: python

   from asymmetry.core.fitting.parameter_models import ParameterCompositeModel

   # Single component
   model = ParameterCompositeModel(["Constant"])
   print(model.param_names)    # ['c']
   print(model.formula_string())

   # Additive model: Linear + Constant (redundant here, but demonstrates API)
   model = ParameterCompositeModel(["Linear", "Constant"], operators=["+"])
   print(model.formula_string())

   # Redfield dynamic term on a linear background
   model = ParameterCompositeModel(["Redfield", "Linear"], operators=["+"])
   print(model.param_names)
   print(model.formula_string())

When the same basis function is used more than once, parameter names are
disambiguated automatically by appending ``_1``, ``_2``, etc.

.. code-block:: python

   model = ParameterCompositeModel(["Redfield", "Redfield", "Lambda_bg"], operators=["+", "+"])
   print(model.param_names)  # ['D_1', 'nu_1', 'm_1', 'D_2', 'nu_2', 'm_2', 'lambda_BG']

Inspecting additive components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For diagnostics and plotting, you can evaluate each basis component
individually from a fitted parameter set:

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting.parameter_models import ParameterCompositeModel

   x = np.linspace(10.0, 3000.0, 100)
   model = ParameterCompositeModel(["DiffusionLF_2D", "Lambda_bg", "Lorentzian"], operators=["+", "*"])

   params = {
       "A": 0.8,
       "D_2D": 2.0,
       "D_perp": 0.0,
       "lambda_BG": 0.05,
       "a": 1.0,
       "B0": 200.0,
       "c": 0.0,
   }

   all_components = model.evaluate_components(x, **params)
   additive_only = model.evaluate_components(x, additive_only=True, **params)

   print(model.additive_component_indices())
   print([name for name, _curve in additive_only])

``additive_only=True`` includes the first term and any term connected with
``+``. Terms connected with ``-``, ``*``, or ``/`` are excluded.

Fitting extracted parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~asymmetry.core.fitting.parameter_models.fit_parameter_model` to
fit parameter-vs-x data retrieved from a series of :class:`~asymmetry.core.fitting.engine.FitResult` objects.

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting.parameter_models import (
       ParameterCompositeModel,
       fit_parameter_model,
   )
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   # Suppose these come from a temperature series of μSR fits
   temperatures = np.array([5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0])
   lambda_values = np.array([0.12, 0.14, 0.18, 0.30, 0.52, 0.88, 1.40])
   lambda_errors = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04])

   # Fit an Arrhenius model
   model = ParameterCompositeModel(["Arrhenius"])
   params = ParameterSet([
       Parameter("a", value=2.0, min=0.0),
       Parameter("Ea", value=10.0, min=0.0),
   ])

   result = fit_parameter_model(temperatures, lambda_values, lambda_errors, model, params)

   if result.success:
       print(f"χ²ᵣ = {result.reduced_chi_squared:.3f}")
       for p in result.parameters:
           err = result.uncertainties.get(p.name, 0.0)
           print(f"  {p.name} = {p.value:.4f} ± {err:.4f}")

The optional ``x_min`` / ``x_max`` keyword arguments restrict which data
points are included in the fit:

.. code-block:: python

   result = fit_parameter_model(
       temperatures, lambda_values, lambda_errors, model, params,
       x_min=50.0, x_max=300.0,
   )

Generating smooth fit curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a successful fit, use
:func:`~asymmetry.core.fitting.parameter_models.evaluate_parameter_model_fit`
to produce a densely sampled curve for plotting.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from asymmetry.core.fitting.parameter_models import (
       ModelFitRange,
       ParameterModelFit,
       evaluate_parameter_model_fit,
   )

   fit = ParameterModelFit(
       parameter_name="Lambda",
       x_key="temperature",
       ranges=[
           ModelFitRange(
               x_min=float(temperatures.min()),
               x_max=float(temperatures.max()),
               model=model,
               parameters=params,
               result=result,
           )
       ],
       active=True,
   )

   curves = evaluate_parameter_model_fit(fit, num_points=300)

   plt.errorbar(temperatures, lambda_values, yerr=lambda_errors, fmt="o", label="Data")
   for curve in curves:
       plt.plot(curve.x, curve.y, "r-", label="Arrhenius fit")
   plt.xlabel("Temperature (K)")
   plt.ylabel("λ (μs⁻¹)")
   plt.legend()
   plt.show()

GUI workflow
~~~~~~~~~~~~~

In the **Fitted Parameters** panel:

1. Run single or global fits across your dataset series.
2. Select a parameter row in the table to open the model-fit toolbar.
3. Choose an x-variable (field or temperature) and add one or more basis
   functions using the **Add Component** button.
4. Set initial parameter values and click **Fit**.
   The fit runs in the background. The dialog shows a
   "Fit in progress..." status and temporarily disables fit-edit controls
   until the run completes.
5. The fitted curve is overlaid on the parameter-vs-x plot automatically.
6. Use **Export CSV** to save the fit results alongside the raw parameter
   values for further analysis.

Cross-group parameter model fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When two or more groups are selected in the **Fitted Parameters** panel,
Asymmetry opens the cross-group model-fit dialog. This dialog shares one
model across groups and lets each parameter be marked as:

* **Global**: one value shared by all groups
* **Local**: one independent value per group
* **Fixed**: held constant

Click **Run Fit** to execute the joint fit. As with single parameter-model
fits, the run is non-blocking: the dialog remains responsive, shows a
"Cross-group fit in progress..." status, and re-enables controls after
completion.

GUI Composite Function Builder
-------------------------------

As of the current GUI workflow, fitting functions are edited as composite
expressions for :math:`A(t)` using a component builder.

In the GUI fit panel (Single and Global tabs), click **Edit Function...** to
build a composite function for :math:`A(t)`.

Default function
~~~~~~~~~~~~~~~~

New fit tabs start with:

.. math::

   A(t) = A_1 e^{-\Lambda t} + A_{bg}

where ``A_bg`` is an explicit constant background component.

Component list
~~~~~~~~~~~~~~

The builder supports these basis components:

* ``Exponential``: :math:`A e^{-\Lambda t}`
* ``Gaussian``: :math:`A e^{-(\sigma t)^2}`
* ``Oscillatory``: :math:`A\cos(2\pi f t + \phi)`
* ``StretchedExponential``: :math:`A e^{-(|\Lambda|t)^\beta}`
* ``StaticGKT_ZF``
* ``Constant``: :math:`A_{bg}`

Use ``+``, ``-``, ``*``, ``/`` operators to combine components.

.. note::

   In the builder, oscillatory is pure cosine by default. To include damping,
   multiply by an exponential component.

Parameter naming rules
~~~~~~~~~~~~~~~~~~~~~~

Composite-parameter names are generated automatically:

* Amplitudes are always indexed by component order: ``A_1``, ``A_2``, ...
* ``A_bg`` is used for a unique constant background term
* Non-amplitude symbols are only indexed when duplicates are present in the
  same expression (for example ``Lambda`` vs ``Lambda_1`` and ``Lambda_2``)

This keeps names readable while guaranteeing uniqueness.

Programmatic Composite Models
------------------------------

You can also build the same composite models directly from Python:

.. code-block:: python

   from asymmetry.core.fitting.composite import CompositeModel
   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet

   model = CompositeModel(["Exponential", "Oscillatory", "Constant"], operators=["+", "*"])

   params = ParameterSet([
       Parameter("A_1", 25.0),
       Parameter("Lambda", 0.5),
       Parameter("A_2", 10.0),
       Parameter("frequency", 1.0),
       Parameter("phase", 0.0),
       Parameter("A_bg", 0.0),
   ])

   engine = FitEngine()
   result = engine.fit(dataset, model.function, params)

   print(model.formula_string())
   print(result.reduced_chi_squared)
