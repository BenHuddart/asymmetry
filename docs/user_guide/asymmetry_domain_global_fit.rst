Asymmetry-Domain Global Fit (Scripting)
=======================================

:func:`asymmetry.core.fitting.fit_global` fits **one model across several
asymmetry traces at once**, sharing chosen *global* parameters across all
datasets while letting other *local* parameters vary per dataset. It works
directly on each dataset's :attr:`~asymmetry.core.data.dataset.MuonDataset.time`,
:attr:`~asymmetry.core.data.dataset.MuonDataset.asymmetry`, and
:attr:`~asymmetry.core.data.dataset.MuonDataset.error` arrays, minimising a
single combined weighted least-squares cost

.. math::

   \chi^2 = \sum_{d}\sum_{i}
            \left(\frac{A_{d,i} - \mu_{d,i}}{\sigma_{d,i}}\right)^2

with one shared value per global parameter and an independent value per dataset
for each local parameter.

.. note::

   This is the scriptable, core-API counterpart to the GUI Global Fit Wizard.
   It is the **asymmetry-domain, least-squares** sibling of the count-domain
   :func:`~asymmetry.core.fitting.fit_grouped_series` family. For the difference
   between the two — and which to reach for — see
   :ref:`asymmetry-vs-count-domain` below.

When to use it
--------------

Reach for ``fit_global`` whenever a physical parameter is genuinely shared
across several measurements while a nuisance parameter is not. The motivating
case is a global Keren fit that shares the dynamic field width
:math:`\Delta` and fluctuation rate :math:`\nu` across several longitudinal
fields at one temperature, while the amplitude floats per field; the shared rate
then yields an activation energy from its temperature dependence. Sharing the
physics pools the data and constrains the shared parameter far more tightly than
fitting each field on its own.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from asymmetry.core.fitting import fit_global, Parameter, ParameterSet

   # A normalized asymmetry/polarization model f(t, **params).
   def model(t, **p):
       return p["amp"] * np.exp(-p["lambda"] * np.asarray(t, dtype=float))

   # One seed structure, broadcast to every dataset.
   seed = ParameterSet([
       Parameter(name="amp", value=0.2, min=0.0, max=1.0),
       Parameter(name="lambda", value=0.5, min=0.0, max=10.0),
   ])

   result = fit_global(
       datasets,                 # a list (or dict) of asymmetry MuonDatasets
       model,
       global_params=["lambda"], # shared across all datasets
       local_params=["amp"],     # independent per dataset
       initial_params=seed,
   )

   print(result.global_parameters["lambda"].value,
         result.global_uncertainties["lambda"])
   print(result.reduced_chi_squared)
   for key, fit in result.dataset_results.items():
       print(key, fit.parameters["amp"].value)

Inputs and keying
-----------------

* ``datasets`` may be a **sequence** (keyed positionally ``0, 1, 2, …``) or a
  **mapping** (keyed by your own labels, e.g. field values). Results in
  ``GlobalFitResult.dataset_results`` use the same keys. The datasets need
  **not** carry unique run numbers — keying is handled for you.
* ``initial_params`` may be a single :class:`~asymmetry.core.fitting.ParameterSet`
  broadcast to every dataset, or a mapping ``key -> ParameterSet`` for per-dataset
  seeds. Each set must contain every name referenced by ``global_params`` and
  ``local_params``. Global seed values and bounds are taken from the first
  dataset's set.
* Fixed parameters (``Parameter(..., fixed=True)``) are held constant; bounds
  (``min`` / ``max``) are respected; a fixed global is not counted in the
  combined degrees of freedom.

The result
----------

:class:`~asymmetry.core.fitting.GlobalFitResult` carries:

* ``global_parameters`` — the shared fitted globals (a ``ParameterSet``);
* ``global_uncertainties`` — ``{name: sigma}`` 1σ HESSE errors on the free globals;
* ``dataset_results`` — per-dataset
  :class:`~asymmetry.core.fitting.FitResult` (each holding that dataset's globals
  *and* locals, with uncertainties and its own χ²/dof), keyed by your dataset key;
* ``chi_squared`` / ``dof`` / ``reduced_chi_squared`` — the **combined** statistic
  over all datasets, with
  :math:`\mathrm{dof} = \sum_d N_d - N_{\text{free global}} - \sum_d N_{\text{free local},d}`.

A single-dataset call behaves like an ordinary single fit: the combined reduced
χ² reduces exactly to that dataset's reduced χ².

.. _asymmetry-vs-count-domain:

Asymmetry-domain vs count-domain global fits
--------------------------------------------

Both paths drive the *same* simultaneous minimiser; they differ in what is
fitted and with which statistic:

* **Asymmetry-domain (this function)** — Gaussian weighted least squares on the
  ``.asymmetry`` traces you already hold, with their per-point :math:`\sigma_A`.
  This is the convenient default and matches the rest of the asymmetry-domain
  workflow.
* **Count-domain** (:func:`~asymmetry.core.fitting.fit_grouped_series`,
  :func:`~asymmetry.core.fitting.fit_grouped_time_domain`) — the Cash/Poisson
  statistic on lifetime-corrected grouped detector counts. This is the
  statistically faithful choice when counts are low, where the √N-Gaussian weight
  used in the asymmetry domain biases the fit. It requires the detector
  histograms, grouping, and the per-group nuisance block.

In short: use ``fit_global`` for the standard asymmetry-domain convenience
workflow; prefer the count-domain path for low-count data where Poisson
statistics matter.
