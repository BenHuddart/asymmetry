Quickstart
==========

Fit one zero-field μSR run from a copper sample on ISIS ARGUS, then save,
reopen, and export the result. The fit illustrates the characteristic dip and
recovery of a static Gaussian Kubo–Toyabe curve.

Download the run
----------------

Open the `ISIS DataGateway page for ARGUS run 76935
<https://data.isis.stfc.ac.uk/browse/instrument/24/facilityCycle/120415884/investigation/113613069/dataset/117743616/datafile/123013750>`__
and use its download control to save the single NeXus file
``ARGUS00076935.nxs`` in your working folder.

Open the data
-------------

Install Asymmetry using :doc:`installation`. Open **Asymmetry** from the
Windows Start menu or macOS Applications folder. For a Python installation,
include the ``gui,hdf5`` extras and run the following command.

.. code-block:: bash

   asymmetry-gui

The window opens with an empty **Data Browser** on the left, a plot in the
centre, and an **Untitled** project tab. Choose **File → Open Data File(s)…**,
open ``ARGUS00076935.nxs``, and select run **76935** in the **Data Browser**.

The plot shows asymmetry, the normalised difference between the forward and
backward detector groups, in percent. Error bars show counting uncertainty.
Use Asymmetry's default reduction for this run. The
:doc:`/explanation/reduction_pipeline` explains how the detector counts become
asymmetry, and :doc:`/reference/grouping_calibration` shows how to check the
settings for your own measurements.

Fit a model
-----------

#. Open **Analysis → Fit** and use the **Single** tab.
#. Click **Edit…**, then **Edit as text**. Replace the expression with
   ``StaticGKT_ZF + Constant`` and click **OK**.
#. Leave the initial parameter values and fit range at their defaults and
   click **Fit**.

``StaticGKT_ZF`` describes the zero-field relaxation expected for a static
Gaussian distribution of local magnetic fields. ``Constant`` allows a flat
asymmetry offset. The fitted width ``Delta`` (Δ) is about 0.394 μs⁻¹, with an
estimated standard uncertainty of about 0.0012 μs⁻¹. Click the **χ²ᵣ**
indicator beside **Fit** to see the parameter estimates and uncertainties.
The fit converges and Asymmetry rates it **poor** because the reduced χ² of
about 1.45 is above the expected band for this fit.

Inspect the fit
---------------

Set the plot's upper **X:** limit to 10 μs to see the dip near 4.4 μs and
the recovery towards the late-time level. The **X:** control changes the
display range. The **Fit range** in the **Single** tab controls which data
the fit uses.

A residual is the measured asymmetry minus the fitted value. Compare the
points with the curve around the dip and along the recovery. The vertical gap
is positive for a point above the curve and negative below it. Look for
stretches of points on the same side of the curve and compare the gaps with
the error bars. If you have two saved fits for the run, **Compare…** in
**Saved fits** plots their :ref:`normalised residuals <saved-single-fits>`.
See :doc:`/reference/assessing_a_fit` for more ways to assess a result.

.. figure:: /_generated/corpus_screenshots/corpus_quickstart_first_fit.png
   :alt: The fitted copper run with its dip and recovery, the Single fit tab,
      and the fit rating next to the reduced chi-squared.
   :width: 100%

   ARGUS run 76935 fitted with ``StaticGKT_ZF + Constant``. The plot shows the
   first 10 μs of the run, and the fit uses the default range.

Save and reopen the analysis
----------------------------

#. Choose **File → Save Project**, enter ``first_analysis.asymp``, and wait
   for saving to finish.
#. Close the project tab. Choose **File → Open Project…** and select
   ``first_analysis.asymp``. Select run **76935** if needed and check that
   **Saved fits** in the **Single** tab lists ``StaticGKT_ZF + Constant``.
   If Asymmetry shows **Historical fit**, the plot may hide the curve,
   while the fitted parameters and report export remain available.

The project records the analysis settings and fit and refers to the original
NeXus file for detector counts. Keep ``ARGUS00076935.nxs`` with the project
when moving the analysis. See :doc:`/reference/project_files` for source
relocation and verification.

Export a fit report
-------------------

In the reopened project, choose **Analysis → Export fit report…**, save
``first_fit.txt``, and open the report in a text editor. Check that the model
is ``StaticGKT_ZF + Constant`` and that ``A_1``, ``Delta``, and ``A_bg`` are
listed with their uncertainties. ``Delta`` is about 0.394 μs⁻¹, as in the
fit panel. The report is a readable record to share alongside the project
and run file.

Where to go next
----------------

* :doc:`key_concepts` introduces runs, asymmetry, fits, and projects.
* :doc:`/reference/fitting` explains fit controls and how to interpret the
  results.
* :doc:`/workflows/index` offers examples to adapt to your own measurements.
