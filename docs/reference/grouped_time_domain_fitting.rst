Grouped time-domain fitting
===========================

.. image:: /_generated/screenshots/grouped_fit_ybco_knight.png
   :alt: MultiGroupFitWindow on YBCO TF above Tc, Individual Groups domain, 4 detector groups
   :width: 100%

*Grouped time-domain Knight-shift workflow on a synthetic YBa₂Cu₃O₇₋δ*
*run in the normal state (T = 100 K, B = 200 G). The central plot is in*
*the* **Individual Groups** *domain so the lifetime-corrected counts*
*N_d(t)·exp(t/τ_μ) for each of the four detectors are shown as separate*
*subplots, and the fit dock has auto-engaged the* **Multi-Group Fit**
*window. Per-group N₀ values fit as* **Local** *parameters; the local*
*field magnitude and phase are* **Shared** *— the canonical workflow for*
*extracting the muon Knight shift in the normal state of a*
*superconductor* (Sonier, Rev. Mod. Phys. **72**, 769 (2000)).

Grouped time-domain fitting is the right tool whenever the physically
meaningful quantity you are extracting depends on detector geometry, and
collapsing the raw counts onto a single forward/backward asymmetry would
average that geometry away. The canonical use is a paramagnetic
Knight-shift measurement, where a ppm-precision TF frequency shift is
recovered by sharing the Larmor frequency across every detector group
while letting per-group amplitudes and phases fit locally (Sonier,
Rev. Mod. Phys. **72**, 769 (2000)). Vortex-state second-moment analyses
benefit from the same machinery — per-group :math:`\sigma` values
expose calibration drifts in the vortex-lattice signal that a single
asymmetry would hide — and any time a shared-physics fit returns
inconsistent results across groups, the diagnosis is almost always a
calibration problem (alpha mismatch, per-detector time-zero drift) rather
than a model problem. For ordinary F-B asymmetry fits with no
geometry-sensitive observable, the regular single-fit panel is faster and
imposes no grouped overhead. For a full worked example on a rotated single
crystal, see :doc:`/workflows/knight_shift_angle`.

What this mode fits
-------------------

Grouped time-domain mode works on the detector groups defined in the
**Grouping** dialog for the active dataset. For each included group,
Asymmetry builds a lifetime-corrected grouped count trace of the form

.. math::

   N_{\mathrm{corr}}(t) = N(t) e^{t / \tau_\mu}

where :math:`\tau_\mu` is the fixed physical muon lifetime.

The fitted count-domain model is

.. math::

   N_{\mathrm{model}}(t) = N_0\left[1 + A\,P(t)\right] + B e^{t / \tau_\mu}

where:

* :math:`P(t)` is the shared physical polarisation function from the chosen fit function
* :math:`N_0` is the group normalisation
* :math:`A` is the group amplitude scale
* :math:`B` is the group background term
* the optional group ``relative_phase`` is added onto the model ``phase`` parameter when the model supports phase

Current GUI workflow
--------------------

Single-run grouped fit
~~~~~~~~~~~~~~~~~~~~~~

1. Open a raw dataset in **FB Asymmetry**.
2. Configure detector groups in **Grouping**.
3. Select the dataset you want to fit.
4. In the central workspace, switch to **Individual Groups**.
5. Launch **Fit** — the fit dock switches to the **Multi-Group Fit** window.
6. Ensure the **Single** tab is selected.
7. Adjust the fit function with **Edit Function...** if needed.
8. Configure the two parameter blocks described below.
9. Click **Run Grouped Fit**.

The grouped plot view shows stacked lifetime-corrected grouped traces for the
active dataset. Switching back to the **FB Asymmetry** tab restores the regular
fit dock content. After the fit completes, each subplot receives its fitted
grouped count curve, and the result is saved as a ``FitSeries`` entry for
parameter trending.

Multi-run grouped batch fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Select two or more runs in the Data Browser.
2. Switch to **Individual Groups** in the central workspace.
3. Launch **Fit** — the **Batch** tab in the Multi-Group Fit window accepts a
   multi-run member list fed from the current selection.
4. Adjust the fit function and classify physics parameters (see below).
5. Click **Run Grouped Fit**.

The Batch tab fits each run's detector groups with the same polarisation model
and records the results as a single ``FitSeries``, making parameter
trending available across the run series. Individual runs' results can be
piped back to the **Single** tab for inspection.

The **Grouping** dialog now includes an **Include** checkbox per group. Only
checked groups appear in the **Individual Groups** viewer and in grouped
time-domain fitting. At least two groups must remain included for grouped
plotting and grouped fitting to stay available.

Parameter blocks
----------------

The grouped GUI mode separates parameters into two blocks.

Per-group (nuisance) parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These nuisance parameters are attached to each detector group and are
always estimated **independently per (run, group)**:

* ``N0``
* ``background``
* ``amplitude``
* ``relative_phase``

They do not appear in the physics-role table and are not trended.

Physics (fit-function) parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the parameters of the selected composite polarisation function.
Each can be classified as:

* **Global**: one shared value across all runs in the batch (cross-run shared)
* **Local**: one fitted value per run, shared across that run's detector groups
* **Fixed**: held constant at the table value

A fit is called a **global grouped fit** when at least one physics parameter
carries the **Global** role; otherwise it is a **batch grouped fit** (N
independent per-run fits recorded together). The relationship is derived
automatically from the parameter-role table — there is no separate scope
selector.

.. note::

   The engine limit for grouped fits is that physics roles must be
   *homogeneous*: all physics parameters must be either all **Global** or all
   **Local/Fixed** for a given fit. Mixed cross-run-global + per-run-local
   physics is not yet supported and is rejected with a clear error message.

Interaction with existing plot controls
---------------------------------------

Grouped time-domain mode follows the same GUI data-preparation conventions as
the existing fit path where practical:

* grouping definitions come from the active dataset's **Grouping** payload
* detector deadtime correction is applied first when the grouping payload has it enabled
* grouped detector traces honour the grouping good-bin limits and bunching factor
* the active fit range still decides which part of the active dataset is sent to the grouped fitter

The grouped plot view itself shows the grouped lifetime-corrected traces built
from the current grouped analysis dataset, while the grouped fit uses the
active fit-range restricted dataset.

.. _grouped-cross-run-global-api:

Cross-run global fits from the API
----------------------------------

Sharing fit-function (physics) parameters across **several runs** — for example
a global Keren fit that shares :math:`\Delta` and :math:`\nu` across a set of LF
fields measured at one temperature — is done programmatically with
:func:`~asymmetry.core.fitting.grouped_time_domain.fit_grouped_series`. This is
the count-domain path: it operates on the raw, lifetime-corrected grouped
**counts** (Cash/Poisson cost by default), the same
:math:`N_0[1 + A\,P(t)] + B e^{t/\tau_\mu}` model documented above — it does
**not** take ``.asymmetry`` arrays.

The ``relationship`` argument selects the sharing mode. The valid values are
``GROUPED_SERIES_RELATIONSHIPS`` — ``("individual", "batch", "global")``:

* ``"individual"`` / ``"batch"`` — one independent grouped fit per run (recorded
  together for trending).
* ``"global"`` — shares the **physics** (fit-function) parameters in
  ``global_params`` across *all* runs at once, while ``local_params`` (the
  per-group nuisance parameters ``N0``, ``background``, ``amplitude``,
  ``relative_phase``) stay free per group.

Build each run's grouped count traces with
:func:`~asymmetry.core.fitting.grouped_time_domain.build_grouped_time_domain_groups`
(it returns a list of ``GroupedTimeDomainGroup`` objects per dataset), assemble
the ``members`` and ``initial_params`` maps keyed by a per-run id, then call
``fit_grouped_series``:

.. code-block:: python

   import numpy as np
   from asymmetry.core.io import load
   from asymmetry.core.fitting import Parameter, ParameterSet
   from asymmetry.core.fitting.grouped_time_domain import (
       build_grouped_time_domain_groups,
       fit_grouped_series,
       GROUP_NUISANCE_PARAMS,
   )

   # Shared physics polarization P(t; ...). First arg is time; the rest are the
   # fit-function parameters. The engine wraps it in N0[1 + A·P(t)] + B·e^(t/τ).
   def keren(t, Delta, nu):
       ...  # return the Keren polarization for these parameters

   # One LF field per run; the key is any per-run id (here the run number).
   runs = {44000: load("run_44000.nxs"), 44001: load("run_44001.nxs")}
   members = {
       run_id: build_grouped_time_domain_groups(ds, t_min=0.1, t_max=8.0)
       for run_id, ds in runs.items()
   }

   def seed():
       return ParameterSet([
           Parameter("N0", 1000.0, min=0.0),
           Parameter("background", 0.0),
           Parameter("amplitude", 0.2, min=0.0),
           Parameter("relative_phase", 0.0),
           Parameter("Delta", 0.3, min=0.0),   # physics — shared below
           Parameter("nu", 0.5, min=0.0),      # physics — shared below
       ])

   initial_params = {
       run_id: {group.group_id: seed() for group in groups}
       for run_id, groups in members.items()
   }

   result = fit_grouped_series(
       "global",
       members,
       keren,
       global_params=["Delta", "nu"],            # shared across ALL runs
       local_params=list(GROUP_NUISANCE_PARAMS),  # per-group nuisance
       initial_params=initial_params,
       cost="poisson",
   )
   print(result.success, result.shared_parameters)

The returned ``GroupedSeriesFitResult`` carries the cross-run
``shared_parameters`` plus per-member ``member_results``.

.. note::

   **This API shares parameters in the count domain.** It fits lifetime-corrected
   grouped detector counts with a Cash/Poisson statistic — the statistically
   faithful choice for low-count data. To share parameters across several
   ``.asymmetry`` traces in a single call instead, use the asymmetry-domain
   :func:`~asymmetry.core.fitting.fit_global` (Gaussian weighted least squares);
   see :doc:`asymmetry_domain_global_fit`. In the GUI, the asymmetry-domain
   shared-parameter workflow is the interactive :doc:`global_fit_wizard`.

Current limitations
-------------------

* The **Global Fit Wizard** is not available in grouped mode.
* Physics parameter roles must be homogeneous: all **Global** or all
  **Local/Fixed**. Mixed-role fits (some physics shared cross-run, others
  per-run) are not yet expressible in the engine.
* The grouped plot shows group traces, not detector-by-detector traces.
* Detector phase tables and detector quadrature workflows are out of scope.
* The count-domain lifetime term is fixed to the physical muon lifetime.

Practical notes
---------------

* Choose an oscillatory model with a ``phase`` parameter if you want non-zero
  ``relative_phase`` values to matter.
* For non-phase models, keep ``relative_phase`` at zero.
* If grouped mode reports that no valid grouped traces are available, check the
  active dataset has raw histograms plus at least two detector groups defined in
  **Grouping**.