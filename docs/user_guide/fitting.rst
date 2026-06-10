Fitting Engine
==============

Asymmetry fits μSR data with an iminuit-based engine that exposes the same
machinery through the GUI fit panels and through the Python API. This
chapter documents the engine itself — the workflow, the parameter and
results objects, the statistics, the global-fit API, and the conventions
that apply across every model. The physical-model catalogue is split out
into dedicated chapters: see :doc:`composite_models` for the expression
grammar that drives the GUI builder, :doc:`fit_wizard` and
:doc:`global_fit_wizard` for the guided model-selection workflows, and the
per-function chapters for the depolarisation forms themselves
(:doc:`fit_functions/relaxation`, :doc:`fit_functions/relaxation`,
:doc:`fit_functions/oscillation`, :doc:`fit_functions/relaxation`, :doc:`fit_functions/kubo_toyabe`,
:doc:`fit_functions/kubo_toyabe`, :doc:`fit_functions/nuclear_dipolar`, :doc:`diffusion_ballistic_lf`,
:doc:`sc_penetration_depth`).

When fitting through the GUI, the plot panel's current bunch factor is
applied before the dataset is handed to the fitter. If the plot is showing
a rebinned dataset, the next GUI fit will use that rebinned dataset. To
fit the full-resolution data and only simplify the view afterwards, run
the fit with ``Bunch = 1`` and then increase the bunch factor; the fit
curve remains overlaid on the plot.

Basic Workflow
--------------

The minimal end-to-end fit is a model from the ``MODELS`` registry, a
``ParameterSet`` of initial values and bounds, and a
``FitEngine`` call:

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet
   from asymmetry.core.io import load

   dataset = load("data.nxs")

   model = MODELS["ExponentialRelaxation"]
   params = ParameterSet([
       Parameter("A0", value=25.0, min=0.0),
       Parameter("Lambda", value=0.5, min=0.0),
       Parameter("baseline", value=0.0),
   ])

   result = FitEngine().fit(dataset, model.function, params)

   print(f"χ²ᵣ = {result.reduced_chi_squared:.3f}")
   for p in result.parameters:
       err = result.uncertainties.get(p.name, 0.0)
       print(f"  {p.name} = {p.value:.6f} ± {err:.6f}")

For anything richer than a single-component model — multiple amplitudes,
shared baselines across detector groups, multiplicative envelopes,
fraction-constrained additive groups — use the composite-model path in
:doc:`composite_models`. The ``MODELS`` registry is intentionally
short and contains only the standalone variants with explicit baselines;
composite expressions are the canonical way to assemble realistic muSR
models.

Parameter Control
-----------------

Bounds, fixing, and initial values are set on individual
``Parameter`` objects:

.. code-block:: python

   from asymmetry.core.fitting.parameters import Parameter

   p = Parameter(name="Lambda", value=0.5, min=0.0, max=10.0)

   # Hold a parameter constant during the fit
   p_fixed = Parameter(name="A0", value=25.0, fixed=True)

   # Or toggle later
   p_fixed.fixed = True

In the GUI parameter table, the **Fix** checkbox does the same job; bounds
are entered directly in the ``min`` and ``max`` columns. Parameter
metadata — symbol, unit, default lower bound, physical description — comes
from
``PARAM_INFO_REGISTRY``; symbols
and units displayed in the GUI, in fit reports, and in exported GLE
figures all derive from this registry, so they are guaranteed to be
consistent with what the model function actually expects.

Fit Statistics
--------------

The ``FitResult`` exposes the standard goodness-of-fit numbers:

.. code-block:: python

   print(f"χ²  = {result.chi_squared:.4f}")
   print(f"χ²ᵣ = {result.reduced_chi_squared:.4f}")
   print(f"converged: {result.success}")
   print(f"message:   {result.message}")

A reduced :math:`\chi^2` near 1 is the usual target. Values much greater
than 1 indicate that either the model is missing physics, the parameter
errors on the input asymmetry are underestimated, or both. Values much
less than 1 generally mean the input errors are overestimated. Neither
case is automatically fatal — in the first, the next move is usually to
add or substitute a component motivated by the physics; in the second,
the parameter uncertainties on the fitted values will be inflated
correspondingly and should be re-derived against a calibrated error
model.

Parameter values and Hessian uncertainties come from
``result.parameters`` and ``result.uncertainties``:

.. code-block:: python

   for p in result.parameters:
       err = result.uncertainties.get(p.name, 0.0)
       print(f"{p.name}: {p.value:.6f} ± {err:.6f}")

Hessian errors assume the local quadratic approximation around the
minimum is a good description of the likelihood. For well-conditioned
fits this is fine. For ill-conditioned problems — strongly correlated
parameters, fits near a bound, multi-modal likelihoods — either bootstrap
(see :ref:`monte-carlo-errors` below) or compute MINOS errors externally.

Residuals
---------

Inspecting the residual time series is the most reliable check that the
model is structurally appropriate:

.. code-block:: python

   import matplotlib.pyplot as plt

   fit_values = model.function(
       dataset.time, **{p.name: p.value for p in result.parameters}
   )
   residuals = (dataset.asymmetry - fit_values) / dataset.error

   plt.plot(dataset.time, residuals, "o", alpha=0.6)
   plt.axhline(0, color="k", linestyle="--")
   plt.xlabel("Time (μs)")
   plt.ylabel("Normalised residual (σ)")

A good fit produces residuals that scatter randomly around zero with most
points within :math:`\pm 2\sigma` and no systematic trend over time.
Persistent structure — a slow oscillation, a turn-up at long times, a
visible knee — almost always points at a missing component rather than at
noise.

Restricting the Fit Range
-------------------------

The ``t_min`` and ``t_max`` keyword arguments to ``FitEngine.fit``
restrict which time bins enter the cost function:

.. code-block:: python

   result = engine.fit(
       dataset, model.function, params, t_min=0.1, t_max=5.0,
   )

The two common reasons to clip the range are (a) early-time bins
contaminated by deadtime or finite-pulse-width effects, and (b)
late-time bins where the signal has decayed into the background and adds
only noise to the fit. The same range is honoured by the fit wizard and
the global-fit workflow.

Global Fitting
--------------

The global-fit interface accepts a list of datasets, the model function,
and explicit lists of parameter names to treat as global (shared across
all runs) or local (independent per run):

.. code-block:: python

   from asymmetry.core.fitting.engine import FitEngine
   from asymmetry.core.fitting.models import MODELS
   from asymmetry.core.fitting.parameters import Parameter, ParameterSet
   from asymmetry.core.io import load

   datasets = [load(f"run_{i}.nxs") for i in range(1, 4)]
   model = MODELS["ExponentialRelaxation"]
   engine = FitEngine()

   initial_params = {}
   for ds in datasets:
       ps = ParameterSet([
           Parameter("A0", value=25.0, min=0.0),
           Parameter("Lambda", value=0.5, min=0.0),
           Parameter("baseline", value=0.0),
       ])
       initial_params[ds.run_number] = ps

   results_dict, global_result = engine.global_fit(
       datasets=datasets,
       model_fn=model.function,
       global_params=["A0"],
       local_params=["Lambda", "baseline"],
       initial_params=initial_params,
       t_min=0.1,
       t_max=10.0,
   )

   for p in global_result:
       print(f"[global] {p.name} = {p.value:.6f}")
   for run_number, run_result in results_dict.items():
       print(f"run {run_number}: χ²ᵣ = {run_result.reduced_chi_squared:.3f}")

The GUI **Batch** tab automates the same workflow: select multiple
datasets, mark parameters as **Global** (shared across runs) or **Local**
(per-run) in the parameter table, and click **Run Batch Fit**. A fit where
at least one parameter is **Global** is a global fit; otherwise each run is
fitted independently but the results are collected into one trendable series.
Results land in the **Fitted Parameters** panel where they can be browsed,
exported to CSV, or passed into the parameter-trending fit framework documented
in :doc:`parameter_trending`.

The same engine is also used in the Frequency workspace for displayed Fourier
spectra.  In that mode the Fit dock switches labels from ``A(t)`` to
``S(ν)``, uses MHz fit ranges, and offers peak/background components documented
in :doc:`frequency_domain_fitting`.

The :doc:`global_fit_wizard` automates model selection on ordered field or
temperature series; use that wizard before constructing a hand-built
global fit if you do not yet know which composite model the data prefer.

.. _monte-carlo-errors:

Monte Carlo Error Estimation
----------------------------

For parameters whose Hessian uncertainties are not trustworthy — heavy
correlations, fits sitting near a bound, multi-modal likelihoods —
bootstrap resampling gives an empirical distribution:

.. code-block:: python

   import numpy as np
   from asymmetry.core.data import MuonDataset

   n_bootstrap = 100
   bootstrap_results = []
   for _ in range(n_bootstrap):
       resampled = dataset.asymmetry + np.random.randn(len(dataset.asymmetry)) * dataset.error
       boot_dataset = MuonDataset(
           time=dataset.time, asymmetry=resampled, error=dataset.error,
       )
       bootstrap_results.append(engine.fit(boot_dataset, model.function, params))

   lambda_dist = np.array([
       next(p.value for p in r.parameters if p.name == "Lambda")
       for r in bootstrap_results
   ])
   print(f"Lambda = {lambda_dist.mean():.4f} ± {lambda_dist.std():.4f}")

Asymmetric MINOS errors are not yet built into the engine and are on the
roadmap.

Custom Models
-------------

Any callable with signature ``f(t, **params) -> array`` is a valid model
function:

.. code-block:: python

   import numpy as np

   def my_model(t, amplitude, tau, frequency):
       return amplitude * np.exp(-t / tau) * np.sin(2 * np.pi * frequency * t)

   params = ParameterSet([
       Parameter("amplitude", value=20.0),
       Parameter("tau", value=1.0, min=0.0),
       Parameter("frequency", value=5.0, min=0.0),
   ])

   result = engine.fit(dataset, my_model, params)

Custom callables do not get the symbol/unit metadata that registered
components carry, so the GUI parameter table will show raw names and no
units. For anything used more than once, a proper component or
``ModelDefinition`` registration is preferable.

See also
--------

- :doc:`composite_models` — composite-expression grammar, fraction groups,
  and the GUI **Build Fit Function** dialog.
- :doc:`fit_wizard` and :doc:`global_fit_wizard` — guided model selection
  for single and series-wide fits.
- :doc:`parameter_trending` — fitting extracted parameters as a function
  of field, temperature, or run number, including the field-dependent
  transport models in :doc:`diffusion_ballistic_lf` and the
  superconducting :math:`\sigma(T)` models in :doc:`sc_penetration_depth`.
- :doc:`grouped_time_domain_fitting` — multi-detector-group fits.
