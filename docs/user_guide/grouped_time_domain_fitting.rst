Grouped Time-Domain Fitting
===========================

Asymmetry now supports a first grouped time-domain fitting workflow in the
GUI. This mode is designed for fitting several detector groups from one active
run with one shared physical polarization function while keeping group-specific
normalization terms explicit.

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

1. Open a raw dataset in **FB Asymmetry**.
2. Configure detector groups in **Grouping**.
3. Select the dataset you want to fit.
4. In the central workspace, switch to **Individual Groups**.
5. Launch **Fit**.
6. In the **Multi-Group Fit** window, adjust the fit function with **Edit Function...** if needed.
7. Configure the two parameter blocks described below.
8. Click **Run Grouped Fit**.

The grouped plot view shows stacked lifetime-corrected grouped traces for the
active dataset. Launching **Fit** from that grouped view swaps the standard fit
dock over to the grouped-fit controls for the current dataset. Switching back
to the **FB Asymmetry** tab restores the regular fit dock content. After the fit
completes, each subplot receives its fitted grouped count curve.

The **Grouping** dialog now includes an **Include** checkbox per group. Only
checked groups appear in the **Individual Groups** viewer and in grouped
time-domain fitting. At least two groups must remain included for grouped
plotting and grouped fitting to stay available.

Parameter Blocks
----------------

The grouped GUI mode separates parameters into two blocks.

Per-Group Parameters
~~~~~~~~~~~~~~~~~~~~

These are the nuisance parameters attached to each detector group:

* ``N0``
* ``background``
* ``amplitude``
* ``relative_phase``

Each of these can be marked as:

* **Global**: one shared value across all included groups
* **Local**: one fitted value per group
* **Fixed**: held constant at the table value for all groups

Fit-Function Parameters
~~~~~~~~~~~~~~~~~~~~~~~

These are the parameters of the selected composite polarization function.

Each fit-function parameter can currently be:

* **Free**: one shared fitted value across all included groups
* **Fixed**: held constant at the table value

This keeps the first implementation slice aligned with the intended Asymmetry
style: explicit per-group nuisance controls plus one shared physical model.

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

This is the first GUI slice, not the final multi-group feature set.

Current limitations are:

* grouped mode works on one active dataset at a time, not across multiple runs
* the **Global Fit Wizard** is not used in grouped mode
* grouped fit-function parameters are only **Free** or **Fixed** in the GUI
* the grouped plot shows group traces, not detector-by-detector traces
* detector phase tables and detector quadrature workflows are still out of scope
* the count-domain lifetime term is fixed to the physical muon lifetime in this slice

Practical Notes
---------------

* Choose an oscillatory model with a ``phase`` parameter if you want non-zero
  ``relative_phase`` values to matter.
* For non-phase models, keep ``relative_phase`` at zero.
* If grouped mode reports that no valid grouped traces are available, check the
  active dataset has raw histograms plus at least two detector groups defined in
  **Grouping**.