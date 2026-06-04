Grouped Time-Domain Fitting
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
*superconductor (Sonier RMP 72, 769, 2000).*

Grouped time-domain fitting is the right tool whenever the physically
meaningful quantity you are extracting depends on detector geometry, and
collapsing the raw counts onto a single forward/backward asymmetry would
average that geometry away. The canonical use is a paramagnetic
Knight-shift measurement, where a ppm-precision TF frequency shift is
recovered by sharing the Larmor frequency across every detector group
while letting per-group amplitudes and phases fit locally (Sonier,
*Rev. Mod. Phys.* **72**, 769, 2000). Vortex-state second-moment analyses
benefit from the same machinery — per-group :math:`\sigma` values
expose calibration drifts in the vortex-lattice signal that a single
asymmetry would hide — and any time a shared-physics fit returns
inconsistent results across groups, the diagnosis is almost always a
calibration problem (alpha mismatch, per-detector time-zero drift) rather
than a model problem. For ordinary F-B asymmetry fits with no
geometry-sensitive observable, the regular single-fit panel is faster and
imposes no grouped overhead.

What This Mode Fits
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

* :math:`P(t)` is the shared physical polarization function from the chosen fit function
* :math:`N_0` is the group normalization
* :math:`A` is the group amplitude scale
* :math:`B` is the group background term
* the optional group ``relative_phase`` is added onto the model ``phase`` parameter when the model supports phase

Current GUI Workflow
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

The Batch tab fits each run's detector groups with the same polarization model
and records the results as a single ``FitSeries``, making parameter
trending available across the run series. Individual runs' results can be
piped back to the **Single** tab for inspection.

The **Grouping** dialog now includes an **Include** checkbox per group. Only
checked groups appear in the **Individual Groups** viewer and in grouped
time-domain fitting. At least two groups must remain included for grouped
plotting and grouped fitting to stay available.

Parameter Blocks
----------------

The grouped GUI mode separates parameters into two blocks.

Per-Group (Nuisance) Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These nuisance parameters are attached to each detector group and are
always estimated **independently per (run, group)**:

* ``N0``
* ``background``
* ``amplitude``
* ``relative_phase``

They do not appear in the physics-role table and are not trended.

Physics (Fit-Function) Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the parameters of the selected composite polarization function.
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

Interaction With Existing Plot Controls
---------------------------------------

Grouped time-domain mode follows the same GUI data-preparation conventions as
the existing fit path where practical:

* grouping definitions come from the active dataset's **Grouping** payload
* detector deadtime correction is applied first when the grouping payload has it enabled
* grouped detector traces honor the grouping good-bin limits and bunching factor
* the active fit range still decides which part of the active dataset is sent to the grouped fitter

The grouped plot view itself shows the grouped lifetime-corrected traces built
from the current grouped analysis dataset, while the grouped fit uses the
active fit-range restricted dataset.

Current Limitations
-------------------

* The **Global Fit Wizard** is not available in grouped mode.
* Physics parameter roles must be homogeneous: all **Global** or all
  **Local/Fixed**. Mixed-role fits (some physics shared cross-run, others
  per-run) are not yet expressible in the engine.
* The grouped plot shows group traces, not detector-by-detector traces.
* Detector phase tables and detector quadrature workflows are out of scope.
* The count-domain lifetime term is fixed to the physical muon lifetime.

Practical Notes
---------------

* Choose an oscillatory model with a ``phase`` parameter if you want non-zero
  ``relative_phase`` values to matter.
* For non-phase models, keep ``relative_phase`` at zero.
* If grouped mode reports that no valid grouped traces are available, check the
  active dataset has raw histograms plus at least two detector groups defined in
  **Grouping**.