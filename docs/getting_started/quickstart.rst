Quickstart
==========

This page takes you from a fresh install to a first fitted result in a few
minutes — and you need no data of your own, because Asymmetry can generate a
realistic demo run for you. The objects each step refers to are collected in
:doc:`key_concepts`, and complete real-world analyses follow in
:doc:`/workflows/index`.

Install and launch
------------------

Install Asymmetry (see :doc:`installation`). Open the desktop application from
the Windows Start menu or macOS Applications folder. With a Python installation
including the ``gui`` extra, start it from a terminal:

.. code-block:: bash

   asymmetry-gui

The main window opens with an empty Data Browser on the left and a plot in the
centre.

Make a demo dataset
-------------------

With your own data you would load a file from **File → Open…** and then set the
detector grouping and calibrate :math:`\alpha`
(see :doc:`/reference/grouping_calibration`). With nothing to hand, generate a
ready-made run instead:

#. Open **File → Simulate Preset**.
#. Choose **Ag — ZF Gaussian Kubo–Toyabe**.

A synthetic silver run appears in the Data Browser, badged ``SIM`` and tinted to
mark it as simulated, and the central plot shows its **asymmetry** — already
grouped and calibrated, because the preset carries a complete reduction. The
preset is a genuine Poisson-sampled dataset built from a known model, so it
behaves exactly as beamline data would.

Read the asymmetry
------------------

The curve is the zero-field **Kubo–Toyabe** relaxation of muons in silver: a
characteristic dip and partial recovery produced by the static distribution of
nuclear dipolar fields at the muon site, with a width set by the spread of those
fields. This time-domain signal is what every fit works on; for the physics
behind it see :doc:`/explanation/musr_primer` and
:doc:`/reference/fit_functions/kubo_toyabe`.

Fit it
------

Fit the relaxation and recover its parameters:

#. Open **Analysis → Fit**, then click **Wizard…** in the **Single** tab.
#. If the wizard shows no recorded field direction, choose **Zero field**.
#. Click **Analyze**. The wizard fingerprints the spectrum, fits a
   portfolio of candidate models, and ranks them by an information criterion.
#. It recommends **Static GKT + Constant** — the static Gaussian Kubo–Toyabe
   with a flat baseline (the expression ``StaticGKT_ZF + Constant``). Click
   **Apply this fit** to write the result back into the single-fit tab.
#. Set the plot's upper **X:** limit to 10 μs to inspect the dip and recovery.

The fitted field width comes out at :math:`\Delta \approx 0.39` μs⁻¹ — the value
the preset was generated from, recovered within the counting errors. That is a
complete analysis: data in, model fitted, parameter out.

.. image:: /_generated/screenshots/quickstart_first_fit.png
   :alt: The main window after the first fit, with the fitted Gaussian
         Kubo–Toyabe curve overlaid on the silver asymmetry data and the
         converged parameters shown in the fit panel.
   :width: 100%

*The main window right after the walkthrough's first fit converges. The red*
*fit curve traces the* ``StaticGKT_ZF + Constant`` *model over the silver*
*asymmetry, and the fit panel shows the applied wizard fit with*
*Δ = 0.39 μs⁻¹ (row* **Δ (μs⁻¹)** *) and a reduced chi-square near 1.*

Where to go next
----------------

* :doc:`key_concepts` — the objects an analysis is built from.
* :doc:`/workflows/index` — full, worked analyses on representative experiments.
* :doc:`/reference/fitting` and :doc:`/reference/fit_wizard` — the fitting tools
  in depth.
* :doc:`/reference/simulation` — more synthetic data, including whole temperature
  scans and the other preset materials.
