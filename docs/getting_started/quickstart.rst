Quickstart
==========

Make a synthetic dataset, fit its relaxation, and keep a result you can reopen
and share. This walkthrough needs no measurement files. You will create three
files in a folder of your choice:

* ``first_run.nxs`` — the generated detector counts;
* ``first_analysis.asymp`` — the analysis settings and fitted result;
* ``first_fit.txt`` — a readable report of the fitted parameters.

Launch Asymmetry
----------------

Install Asymmetry using :doc:`installation`. Open **Asymmetry** from the Windows
Start menu or macOS Applications folder. For a Python installation, install the
``gui`` and ``hdf5`` extras described there, then run:

.. code-block:: bash

   asymmetry-gui

The window opens with an empty **Data Browser** on the left, a plot in the
centre, and an **Untitled** project tab above them.

Generate and save the data
-------------------------

#. Open **File → Generate Synthetic Run…**.
#. Keep **Template run** set to **Built-in: Ideal pulsed F/B (ISIS-style)**,
   **Generation** set to **Forward/backward asymmetry**, and the default
   **Model A(t)**, an exponential relaxation plus a constant.
#. Enter these values in the parameter table:

   .. list-table::
      :header-rows: 1
      :widths: 20 15 65

      * - Parameter
        - Value
        - Meaning
      * - ``A_1``
        - 24
        - Relaxing part of the asymmetry, in percent.
      * - ``Lambda``
        - 0.3
        - Relaxation rate, in μs⁻¹.
      * - ``A_bg``
        - 2
        - Constant part of the asymmetry, in percent.

#. Set **Total events** to 40 (40 million events). Leave the detector background
   at its default, keep **Fixed seed** checked, and set its number to 23. The
   seed makes this particular random draw repeatable in the same environment.
#. Click **Generate** and wait for it to finish. Click **Save as NeXus…**,
   save ``first_run.nxs`` in your chosen folder, then click **Close**.

The generated counts have random counting fluctuations around the chosen model.
The parameters are teaching choices, with no material attribution. The NeXus
file keeps the counts; the project you save later will refer to that file.

Open and inspect the saved run
------------------------------

#. Choose **File → New Project** to open an empty project tab.
#. Close the earlier **Untitled** tab with its **×**, and choose **Discard**
   when asked about unsaved changes. Its generated counts are already in
   ``first_run.nxs``; this discards only the unsaved session.
#. In the remaining tab, choose **File → Open Data File(s)…** and select
   ``first_run.nxs``. Select its row in the **Data Browser** if needed.

The central plot shows **asymmetry**: the normalised difference between the
forward and backward detector groups, displayed in percent. It starts near
26 % and relaxes towards the 2 % constant component. Error bars describe the
counting uncertainty of each point. The synthetic file already supplies the
grouping and calibration needed for this example.

For the route from counts to the plotted signal, explore
:doc:`/explanation/reduction_pipeline`. For real measurements, grouping and
calibration need checking against the instrument and experiment;
:doc:`/reference/grouping_calibration` explains those choices.

Fit and check the result
-----------------------

#. Open **Analysis → Fit**, then click **Wizard…** in the **Single** tab.
#. If no field direction is recorded, choose **Zero field**. Click **Analyze**
   and wait for the recommendation.
#. Check that its expression is an exponential plus a constant. The wizard
   can label this **Current model**, because it is also the initial model in
   the fit panel. The recommendation is a starting point for checking the data.
#. Check **Show residuals** before applying the fit. A residual is the
   difference between a data point and the fitted curve. Look for fluctuations
   around zero, rather than a persistent slope or wave: structure can reveal
   a missing contribution even when a summary statistic looks acceptable.
#. Click **Apply this fit**, close the wizard, and set the plot's upper **X:**
   limit to 10 μs to inspect the relaxation.

The fitted ``Lambda`` should be close to the generating value, 0.3 μs⁻¹.
Counting fluctuations mean the fitted values need not equal the inputs exactly.
The parameter table includes estimated uncertainties; these describe the fit
under its model and error assumptions, rather than every possible experimental
error. Click the fit's χ² chip for its result summary. A reduced χ² near one
is expected for this example, but neither convergence nor χ² alone establishes
that a model is physically appropriate. See :doc:`/reference/assessing_a_fit`
for the checks to use with your own measurements.

.. image:: /_generated/screenshots/quickstart_first_fit.png
   :alt: The saved synthetic first analysis, with an exponential-plus-constant fit
      over the reloaded data and fitted parameters in the Single tab.
   :width: 100%

*The walkthrough's generated file, fitted through the wizard and saved as a
project. Your fitted values should be close to the teaching inputs; the final
digits depend on the count draw and numerical environment.*

Save the analysis
-----------------

Choose **File → Save Project**, enter ``first_analysis.asymp``, and wait for
saving to finish. Keep ``first_run.nxs`` alongside it. The project records
settings and fits; it does not embed the detector counts. To return to the
analysis, choose **File → Open Project…** and select ``first_analysis.asymp``.
Keep both files when moving the analysis to another machine. Source relocation
and verification are explained in :doc:`/reference/project_files`.

Export a result
---------------

Choose **Analysis → Export fit report…**, save ``first_fit.txt``, and open it in
a text editor. It lists the model, fitted parameters with their uncertainties,
and fit statistics. Check that it includes ``Lambda`` close to 0.3 μs⁻¹. This
readable report complements the project and source file; it is not a reloadable
analysis on its own.

Where to go next
----------------

* :doc:`key_concepts` — runs, asymmetry, fits, and projects.
* :doc:`/reference/simulation` — one-click presets and other synthetic signals.
* :doc:`/reference/fitting` and :doc:`/reference/fit_wizard` — fit controls and
  model selection.
* :doc:`/workflows/index` — demonstrations to adapt to your own measurements.
