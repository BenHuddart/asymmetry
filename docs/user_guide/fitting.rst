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
   params.add(Parameter(name="lambda", value=0.5, min=0))
   params.add(Parameter(name="B", value=0.0, min=-0.1, max=0.1))
   
   # Perform fit
   result = engine.fit(dataset, model.function, params)
   
   # Print results
   print(f"Success: {result.success}")
   print(f"χ²: {result.chi_squared:.4f}")
   print(f"χ²ᵣ: {result.reduced_chi_squared:.4f}")
   for name, value in result.parameters.items():
       error = result.uncertainties.get(name, 0)
       print(f"{name}: {value} ± {error:.6f}")

Available Models
----------------

The following built-in models are available in the MODELS registry:

ExponentialRelaxation
~~~~~~~~~~~~~~~~~~~~

Simple exponential relaxation:

.. math::

   A(t) = A_0 e^{-\lambda t} + B

Parameters: ``A0`` (initial asymmetry), ``lambda`` (relaxation rate), ``B`` (baseline)

.. code-block:: python

   model = MODELS["ExponentialRelaxation"]

GaussianRelaxation
~~~~~~~~~~~~~~~~~~

Gaussian Kubo-Toyabe relaxation:

.. math::

   A(t) = A_0 e^{-\frac{1}{2}(\sigma t)^2} + B

Parameters: ``A0``, ``sigma`` (Gaussian relaxation rate), ``B``

.. code-block:: python

   model = MODELS["GaussianRelaxation"]

Oscillatory
~~~~~~~~~~~

Damped oscillation with exponential decay:

.. math::

   A(t) = A_0 e^{-\lambda t} \cos(2\pi f t + \phi) + B

Parameters: ``A0``, ``lambda``, ``freq`` (frequency in MHz), ``phi`` (phase), ``B``

.. code-block:: python

   model = MODELS["Oscillatory"]

StretchedExponential
~~~~~~~~~~~~~~~~~~~~

Stretched exponential (Kohlrausch) relaxation:

.. math::

   A(t) = A_0 e^{-(\lambda t)^\beta} + B

Parameters: ``A0``, ``lambda``, ``beta`` (stretching exponent, 0 < β ≤ 1), ``B``

.. code-block:: python

   model = MODELS["StretchedExponential"]

StaticGKT_ZF
~~~~~~~~~~~~

Static Gaussian Kubo-Toyabe function for zero field:

.. math::

   A(t) = A_0 \left[\frac{1}{3} + \frac{2}{3}(1-\sigma^2 t^2)e^{-\frac{1}{2}\sigma^2 t^2}\right] + B

Parameters: ``A0``, ``sigma``, ``B``

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
       name="lambda",
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

Tips for Good Fits
------------------

1. **Start with reasonable initial guesses**: Use physical intuition
2. **Set appropriate bounds**: Prevent unphysical values
3. **Check residuals**: Should be randomly distributed around zero
4. **Compare models**: Use chi-squared and AIC/BIC for model selection
5. **Validate uncertainties**: Bootstrap if parameter errors seem too small
