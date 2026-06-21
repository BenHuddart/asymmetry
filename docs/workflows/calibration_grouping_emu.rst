EMU detector grouping and α calibration
=======================================

This is the recommended first walkthrough for new users. It covers the
everyday setup every time-domain analysis depends on: loading a run,
defining the forward/backward detector grouping, calibrating
:math:`\alpha` from a transverse-field (TF)
run, and turning on dead-time correction — all from the one **Grouping**
window.

It is the Asymmetry-worded counterpart to the WiMDA "Basics" exercise.
Where WiMDA asks you to type a detector list into a text box, Asymmetry
gives you a **visual Detector Layout Editor** with instrument presets and
collects every calibration field — :math:`t_0`, good-data window,
dead-time, grouping, and :math:`\alpha` — in a single dialog.

The runs
--------

The exercise uses two silver (Ag) runs from the corpus *Basics* set
(``Basics/data_hdf5/``):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Run
     - Type
     - Role
   * - ``EMU00044989``
     - TF (≈20 G)
     - Calibration run — its precession amplitude is used to estimate
       :math:`\alpha`.
   * - ``EMU00034998``
     - ZF
     - The measurement run the calibrated grouping is applied to.

Silver is the standard calibrant: it has a near-zero nuclear moment, so a
small transverse field produces a clean, slowly-relaxing precession whose
forward/backward amplitudes reveal any detector imbalance.

Step 1 — Load a run
-------------------

Open the TF calibration run from **File → Open** (or drag it onto the
window). The Data Browser lists one row per run; the central plot shows
the currently-selected run's asymmetry. Loading several files at once is
fine — the calibrated grouping you build below is applied to whichever
runs you select.

Step 2 — Open the Grouping window
---------------------------------

Click **Grouping** on the toolbar. This one dialog is the whole
calibration surface for a run:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - What it sets
   * - **Reference run**
     - The run whose values seed the dialog. Only the *checked* datasets
       are changed when you press **Apply**.
   * - **Forward Group / Backward Group**
     - Which detector groups form the :math:`F` and :math:`B` sums of the
       asymmetry.
   * - **Alpha** / **Estimate α**
     - The :math:`\alpha` calibration constant (see Step 3). **Estimate All α** does
       every checked run at once.
   * - **t0 Bin**
     - The time-zero bin (the muon-implantation prompt).
   * - **t_good Offset**
     - First good bin, measured as an offset after :math:`t_0`.
   * - **Last Good Bin**
     - End of the good-data window.
   * - **Bunching Factor**
     - Time-bin rebinning (also available on the toolbar).
   * - **Deadtime** / **Deadtime Mode**
     - Enable dead-time correction and choose File / Manual / Estimate
       (see Step 5).
   * - **Background**
     - Enable a constant-background subtraction.
   * - **Detector Layout…**
     - Opens the visual editor and instrument presets (see Step 4).
   * - **Load .grp / Save .grp**
     - Read or write a WiMDA-style grouping file.
   * - **Apply**
     - Recompute the asymmetry for the checked runs. The Log echoes what
       was applied (``F=1, B=2, alpha=…, deadtime=…``).

Step 3 — Estimate α from the TF run
-----------------------------------

With the TF calibration run (``EMU00044989``) as the reference run, click
**Estimate α** and then **Apply**. Asymmetry integrates the forward and
backward grouped counts over the good-data window and sets

.. math::

   \alpha = \frac{\sum_i F_i}{\sum_i B_i},

the same balance ratio Mantid's ``AlphaCalc`` uses. On the Ag TF run this
gives :math:`\alpha \approx 1.103`, and the TF precession becomes
symmetric about zero — the visual signature of a correct
:math:`\alpha`. A wrong :math:`\alpha` shows up as a precession riding on
a non-zero offset.

The same calculation is available from the Python API:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.transform import apply_grouping, estimate_alpha

   tf = load("Basics/data_hdf5/EMU00044989.nxs")
   g = tf.run.grouping
   groups = g["groups"]                             # {group_id: [1-based detector ids]}
   fwd = [int(d) - 1 for d in groups[g["forward_group"]]]
   bwd = [int(d) - 1 for d in groups[g["backward_group"]]]
   forward = apply_grouping(tf.run.histograms, fwd)
   backward = apply_grouping(tf.run.histograms, bwd)
   alpha = estimate_alpha(
       forward, backward,
       first_good_bin=g.get("first_good_bin"),
       last_good_bin=g.get("last_good_bin"),
   )
   print(round(alpha, 3))                           # 1.103

Carry the value onto the measurement run by selecting ``EMU00034998`` (and
any other same-sample runs), setting **Alpha** to the calibrated value,
and pressing **Apply**.

Step 4 — The visual Detector Layout Editor
------------------------------------------

Click **Detector Layout…** to open the visual editor — Asymmetry's
replacement for WiMDA's "edit detector list" text box. It draws the
instrument's physical forward and backward detector arrays as concentric
rings of segments:

- **Click a segment to toggle** its membership in the active group.
  Colours and names distinguish up to eight groups, and a detector may
  belong to more than one group.
- The **preset grouping** dropdown offers instrument-aware layouts. For
  EMU these are **Longitudinal** (the standard two-group F/B split used
  here) and **Vector Polarization** (the multi-axis layout used for
  :doc:`/reference/vector_polarization`).

For this Ag exercise the EMU **Longitudinal** preset is the correct
starting point; the editor is most useful when a detector is masked or
when you build a custom geometry. See
:doc:`/reference/detector_grouping` for the full reference.

Step 5 — Dead-time correction
-----------------------------

Tick **Enable Deadtime Correction** and choose a **Deadtime Mode**:

- **File** — use the per-detector dead-time values stored in the data
  file (the usual choice when the instrument provides them).
- **Manual** — type per-detector values (in ns), or paste a calibration.
- **Estimate** — fit the dead-time from the early-time count rate; **Cal**
  fits one value per detector from the reference run.

Asymmetry normalises the non-paralyzable correction by each run's own
per-run good-frame count, so it stays bounded run-to-run; the correction
is a sub-percent bump on a well-behaved EMU histogram.

Reusing the calibration
-----------------------

When you load further runs, Asymmetry auto-applies the existing grouping
to them (the Log shows *"Auto-applied existing project grouping"*) so a
whole series inherits the same :math:`F`/:math:`B` split, dead-time mode,
and good-data window. The measured per-run quantities — good-frame counts,
:math:`t_0` — are always re-read from each run, never inherited.

:math:`\alpha`, however, is a *sample* calibration, not an instrument
constant. The auto-applied grouping carries the reference run's
:math:`\alpha` across loads, so **re-estimate** :math:`\alpha` (or reset
it) whenever you switch to a different sample. Save a finished setup with
**Save .grp** to reuse it in a later session.

Where to go next
----------------

- :doc:`/reference/detector_grouping` — full grouping and layout-editor
  reference.
- :doc:`temperature_scan_magnetism` — applies one calibrated grouping
  across a temperature scan.
- :doc:`dynamic_kt_copper` — uses this setup as the front end to a
  zero-field dynamics fit.
