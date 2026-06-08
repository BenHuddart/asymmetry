ALC field scan in TCNQ
======================

This worked example builds an **avoided-level-crossing (ALC)** resonance
scan from a series of field-stepped runs and locates the resonance. ALC
mode plots one integral-asymmetry value per run against field, rather than
fitting a model in the time domain. For the full feature reference see
:doc:`/user_guide/alc_mode`; this chapter is the end-to-end walkthrough on
real data.

The runs
--------

The corpus *Chemistry → ALC resonance in TCNQ* set
(``Data_hdf5/``) is a field scan of a muoniated TCNQ radical at
:math:`T = 350\;\mathrm{K}`: **31 runs** stepped from **2000 to 5000 G**
in 100 G steps. Each run's swept field is read from its metadata, so the
Data Browser resolves a **B (G)** column automatically.

The GUI workflow
----------------

1. **Load the scan and select it.** Open the 31 runs and multi-select
   them in the Data Browser (click the first, then ``Shift``-click the
   last). The scan is built from the *selected* runs.
2. **Enter ALC mode.** With the **F-B asymmetry** view active (the
   default), the **ALC mode** toggle on the main toolbar is enabled —
   click it (it turns red/active). The Fit and Parameters docks are
   swapped for the bespoke ALC build panel and scan view.

   .. note::

      The toggle is available whenever the active representation is F-B
      asymmetry, regardless of how many runs are selected. If it is
      greyed out, you are in a frequency or groups view — its tooltip
      ("Switch to the F-B asymmetry view to use ALC mode") points the way.
      The two-or-more-runs requirement is checked when you build, not on
      the toggle.

3. **Build the scan.** Raise the **Fit** dock to reach the ALC build
   panel, set the **Integration window** — the time window each run's
   asymmetry is integrated over — to ``0.2``–``8`` µs, and click
   **Build Scan**. Each selected run collapses to one point.
4. **Read and fit the resonance.** The Parameters dock (scan view) plots
   integral asymmetry vs **B**. Use the x-axis selector (B / T / run), the
   **dA/dB** derivative toggle, the **Baseline** section (subtract the
   non-resonant background over chosen regions) and the **Peaks** section
   (add a Gaussian or Lorentzian and **Fit peaks**) to fit the dip.

Expected result
---------------

A clean **D1 ALC dip** appears at

.. math::

   B_0 \approx 3100\;\mathrm{G},

the muonium–proton :math:`\Delta_1` (:math:`\Delta M = \pm 1`) resonance.
The dip field corresponds to a muon–electron hyperfine coupling of
:math:`A_\mu \approx 84.5\;\mathrm{MHz}` — the headline number from a TCNQ
ALC measurement.

Scripting the scan
------------------

The GUI build panel calls the same core builder your scripts can use.
:func:`~asymmetry.core.transform.integral.build_field_scan` integrates
every run over the window and returns the sorted scan:

.. code-block:: python

   from glob import glob

   import numpy as np
   from asymmetry.core.io import load
   from asymmetry.core.transform.integral import build_field_scan

   scan_files = sorted(glob(".../ALC resonance in TCNQ/Data_hdf5/*.nxs"))
   runs = [load(p) for p in scan_files]       # the 31 field-stepped runs
   scan = build_field_scan(
       runs, t_min=0.2, t_max=8.0,
       method="integral", order_key="field",
   )

   b_dip = scan.x[int(np.argmin(scan.value))]
   print(round(float(b_dip)))                 # 3100  (G)

On the TCNQ scan the integral asymmetry dips to its minimum at the
:math:`3100\;\mathrm{G}` point, reproducing the GUI result.

See also
--------

- :doc:`/user_guide/alc_mode` — full ALC-mode reference: baseline and
  peak fitting, the differential view, and project persistence.
- :doc:`calibration_grouping_emu` — grouping and :math:`\alpha` setup
  shared by every run in the scan.
