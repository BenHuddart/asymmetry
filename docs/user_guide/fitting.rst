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

.. _fit-statistics:

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

.. _fit-advisory-warnings:

Advisory warnings (``result.warnings``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before it minimises, the engine runs a pair of *advisory guards* that flag the
two setup traps most likely to produce a converged-but-wrong fit:

- :class:`~asymmetry.core.fitting.AsymmetryScaleWarning` — the data and the
  seeded model sit on different asymmetry scales (the classic percent-vs-fraction
  mix-up; a loaded ``MuonDataset.asymmetry`` is on the **percent** scale, ``×100``).
- :class:`~asymmetry.core.fitting.FixedFrequencyFieldMismatchWarning` — a
  precession ``frequency`` is held **fixed** more than ~2 % from
  :math:`\gamma_\mu B` implied by the run's ``field``; pinning the line away from
  its true position leaks the misfit into the damping term and inflates the
  fitted Gaussian ``sigma`` (the vortex-state TF trap).

These are emitted through the Python :mod:`warnings` system (so they still reach
the log/stderr), and their messages are now **also carried on the result** as a
plain list, ``result.warnings``:

.. code-block:: python

   result = FitEngine().fit(ds, model.function, params)
   for note in result.warnings:
       print("⚠", note)

The guards never raise and never change the fit outcome — they only point you at
the fix. In the **GUI**, the fit panel surfaces these messages directly in the
result box alongside the converged line: a single fit shows them beneath its
``Fit converged`` summary, and a batch/global fit shows them (deduplicated, since
the same trap usually fires for every run) beneath ``Batch fit converged``. The
underlying science and the corrective recipe for each trap are in the cookbook —
see :ref:`the transverse-field frequency entry <cookbook-tf-frequency>` and
:ref:`the asymmetry-scale entry <cookbook-asymmetry-scale>`.

.. _fit-bg-amplitude-tf:

A stuck χ² ≈ 200? Free the background amplitude (TF data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A frequent cause of a huge, stubborn reduced :math:`\chi^2` (often
:math:`\chi^2_r \approx 200`) on **muSR transverse-field** data is a background
amplitude pinned at a non-physical lower bound. The constant background term
``A_bg`` (and the standalone ``baseline``) defaults to a **zero lower bound**,
but for TF data the fitted background is routinely **negative** — commonly around
:math:`-20\%` of the full asymmetry — because of detector-pair imbalance and the
asymmetry baseline convention. Clamped at zero, the minimiser cannot reach the
true minimum and the fit parks at a large :math:`\chi^2`.

The fix is to initialise the background **unbounded** (or with an explicitly
negative lower bound) so it can go negative:

.. code-block:: python

   from asymmetry.core.fitting.parameters import Parameter

   # WRONG for TF data: a zero floor traps A_bg → χ²ᵣ stuck near 200
   bad = Parameter("A_bg", value=0.0, min=0.0)

   # RIGHT: leave it unbounded so the background can settle negative
   bg = Parameter("A_bg", value=0.0)              # min=-inf, max=+inf by default
   # ...or pin a generous negative floor if you want some guard rails:
   bg = Parameter("A_bg", value=-0.05, min=-0.5, max=0.5)

In the GUI parameter table, clear the ``min`` cell for the background row (or set
it negative) before fitting TF data. If a TF fit converges with a glued-to-zero
background and a :math:`\chi^2_r` in the hundreds, this bound is the first thing
to check.

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

.. _fit-residuals:

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

.. _affine-ties:

Parameter ties (links and equal spacing)
----------------------------------------

A multi-line fit often shares a value across components, or constrains one
parameter to track another. Asymmetry offers two constraint kinds, both set on
the :class:`~asymmetry.core.fitting.parameters.Parameter` and both removing the
follower from the free-fit set:

**Equality link groups** (WiMDA "Ties") force ``follower == main``. Tag the
members of a group with the same ``link_group`` id; the group's main is fitted
and every follower inherits its value (and propagated uncertainty). Use these
to share an amplitude, phase, or relaxation rate across lines::

   Parameter("Lambda_2", value=0.3, link_group=1)
   Parameter("Lambda_4", value=0.3, link_group=1)   # follower == Lambda_2

**Affine ties** express an *offset* or *scaled* relationship that equality
cannot — ``follower = scale·main + offset_scale·offset + const`` — where
``offset`` may be a free **auxiliary** parameter the model itself never
consumes. The canonical use is keeping muonium satellites **equally spaced**
about a central line: a central frequency ``f_c`` and a free half-splitting
``delta`` drive both satellites, so the three lines stay symmetric while only
two frequencies are fitted. This stabilises the satellite amplitudes (the third
free frequency would otherwise trade against them), which is what makes the
shallow-donor ionisation energy extractable:

.. code-block:: python

   from asymmetry.core.fitting import AffineTie, Parameter, ParameterSet
   from asymmetry.core.fitting.composite import CompositeModel
   from asymmetry.core.fitting.engine import FitEngine

   model = CompositeModel.from_expression(
       "Oscillatory*Exponential + Oscillatory*Exponential "
       "+ Oscillatory*Exponential + Constant"
   )
   ps = ParameterSet([
       Parameter("A_1", 8.0, min=0), Parameter("frequency_1", 1.39),
       # ... phases / relaxation shared via link_group as above ...
       # Satellites derived from the centre and a free half-splitting:
       Parameter("frequency_3", 1.27,
                 tie=AffineTie(main="frequency_1", offset="delta", offset_scale=-1.0)),
       Parameter("frequency_5", 1.51,
                 tie=AffineTie(main="frequency_1", offset="delta", offset_scale=+1.0)),
       Parameter("delta", 0.12, min=0.0),   # free auxiliary half-splitting
   ])
   result = FitEngine().fit(dataset, model.function, ps, t_min=0.1, t_max=8.0)
   # The hyperfine constant is the satellite splitting, 2·delta, with its
   # own (delta-method) uncertainty:
   fitted = {p.name: p.value for p in result.parameters}
   a_mu = 2.0 * fitted["delta"]

A constant offset (no auxiliary parameter) pins a *known* splitting:
``AffineTie(main="frequency_1", const=+0.12)``. Tie references must be free,
fixed, or link-group parameters — ties may not chain to other ties, and a
parameter cannot be both link-grouped and affinely tied. Affine ties are a
deliberate capability beyond WiMDA (whose links are equality-only); see
``docs/porting/link-groups/`` for the design rationale. General *nonlinear*
expression constraints (``Parameter.expr``) remain reserved and are not yet
evaluated by the engine.

In the **GUI**, the single-fit parameter table has a **Tie** column: click a
row's button to open the tie editor and derive that parameter from the others
in the table (``main``, ``scale``, ``offset``, ``offset scale``, ``const``, with
a live formula preview). Setting a tie clears and disables that row's *Fix* and
*Link* controls. Because the editor references parameters that already have a
table row, equal spacing is expressed directly against existing lines — e.g. a
lower satellite ``f_lo = 2·f_c − f_hi`` (``main=f_c, scale=2, offset=f_hi,
offset_scale=-1``), which removes one free frequency exactly like the
auxiliary-``delta`` form. The free-auxiliary form above (a ``delta`` parameter
the model never consumes) is authored via the API; a project that uses it is
**preserved** intact when opened and re-saved in the GUI, even though the GUI
does not edit the auxiliary parameter directly.

Affine ties are honoured by the single-run engine (``FitEngine.fit``). Global,
count-domain, and grouped/series fits raise ``NotImplementedError`` when a tie is
present rather than silently ignoring it — fit each run individually.

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
exported to TSV, or passed into the parameter-trending fit framework documented
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
