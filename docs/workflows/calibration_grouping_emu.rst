EMU detector grouping and α calibration
=======================================

This is the recommended first walkthrough for new users. It covers the
everyday setup every time-domain analysis depends on: loading a run,
defining the forward/backward detector grouping, calibrating
:math:`\alpha` from a transverse-field (TF)
run, and turning on dead-time correction — all built into one named
**grouping profile** for the instrument, edited from the **Grouping**
window.

It is the Asymmetry-worded counterpart to the WiMDA "Basics" exercise.
Where WiMDA asks you to type a detector list into a text box, Asymmetry
gives you a **visual Detector Layout Editor** with instrument presets, and
collects every calibration field — :math:`t_0`, good-data window,
dead-time, grouping, and :math:`\alpha` — under one profile, with each
correction calibrated in its own dedicated dialog.

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
--------------------

Open the TF calibration run from **File → Open** (or drag it onto the
window). The Data Browser lists one row per run; the central plot shows
the currently-selected run's asymmetry. Loading several files at once is
fine — any run of this instrument inherits the profile you build below
automatically, with no separate step to push it out to the rest of the
series.

Step 2 — Open the Grouping window
-----------------------------------

Click **Grouping** on the toolbar and create a profile for EMU (or pick an
existing one). The window is the whole calibration surface for that
profile:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - What it sets
   * - **Profile selector**
     - Which named profile you are editing — **New…** / **Duplicate…** to
       start fresh or branch from the current settings.
   * - **Preview run**
     - The run whose per-run facts (:math:`t_0`, good-bin window, file
       dead-time) seed the preview. Changing it never edits the profile.
   * - **Forward Group / Backward Group**
     - Which detector groups form the :math:`F` and :math:`B` sums of the
       asymmetry.
   * - **Alpha status row / Calibrate…**
     - The current :math:`\alpha` and its provenance, with a button that
       opens the alpha calibration dialog (see Step 3).
   * - **t0 Bin**
     - The time-zero bin (the muon-implantation prompt).
   * - **t_good Offset**
     - First good bin, measured as an offset after :math:`t_0`.
   * - **Last Good Bin**
     - End of the good-data window.
   * - **Bunching Factor**
     - Time-bin rebinning (also available on the toolbar).
   * - **Deadtime status row / Configure…**
     - Opens the deadtime dialog: mode (Off / File / Manual / Estimate),
       per-detector table, **Cal** fit (see Step 5).
   * - **Background status row / Configure…**
     - Opens the background dialog for a constant-background subtraction.
   * - **Live preview**
     - The forward/backward asymmetry the current draft would produce,
       updating as you edit.
   * - **Detector Layout…**
     - Opens the visual editor and instrument presets (see Step 4).
   * - **Scope panel**
     - Every run of this instrument, tagged *inherits* or *override*, with
       **Release** / **Reattach** to move a run between the two.
   * - **Apply**
     - Write the draft back to the profile. The Log echoes how many
       inheriting runs it reached.

Step 3 — Calibrate α from the TF run
--------------------------------------

With the TF calibration run (``EMU00044989``) as the preview run, click
**Calibrate…** beside the alpha status row. Because this run's field
geometry is transverse and its magnitude sits in the weak-TF window, the
calibration dialog highlights and pre-selects it in its own run dropdown.
Choose the **Diamagnetic** method: Asymmetry integrates the forward and
backward grouped counts over the good-data window and fits

.. math::

   \alpha = \frac{\sum_i F_i}{\sum_i B_i},

the same balance ratio Mantid's ``AlphaCalc`` uses (the **Ratio** method
computes exactly this integral form directly, with no oscillation fit).
On the Ag TF run this gives :math:`\alpha \approx 1.103`, and the dialog's
before/after preview shows the TF precession becoming symmetric about
zero — the visual signature of a correct :math:`\alpha`. A wrong
:math:`\alpha` shows up as a precession riding on a non-zero offset.
Accept the calibration: the status row now reads something like
"α = 1.103 · diamagnetic · run 44989".

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

Press **Apply** to write the calibrated :math:`\alpha` back to the
profile. Because ``EMU00034998`` is the same instrument, it will inherit
this profile — grouping and :math:`\alpha` included — the moment it is
loaded (see `Reusing the calibration`_ below); nothing further needs to be
selected or applied on it by hand.

Step 4 — The visual Detector Layout Editor
---------------------------------------------

Click **Detector Layout…** to open the visual editor — Asymmetry's
replacement for WiMDA's "edit detector list" text box. It draws the
instrument's physical forward and backward detector arrays as concentric
rings of segments:

- **Click a segment to toggle** its membership in the active group.
  Colours and names distinguish up to eight groups, each showing a live
  member count, and a detector may belong to more than one group — drawn
  as its own thin slice per membership.
- Hover a detector for its id, physical label, group memberships, and
  exclusion state.
- The **preset grouping** dropdown offers instrument-aware layouts. For
  EMU these are **Longitudinal** (the standard two-group F/B split used
  here) and **Vector Polarization** (the multi-axis layout used for
  :doc:`/reference/vector_polarization`).

For this Ag exercise the EMU **Longitudinal** preset is the correct
starting point; the editor is most useful when a detector is masked or
when you build a custom geometry. See
:doc:`/reference/detector_grouping` for the full reference.

Step 5 — Dead-time correction
--------------------------------

Click **Configure…** beside the deadtime status row and choose a mode:

- **File** — use the per-detector dead-time values stored in the data
  file (the usual choice when the instrument provides them).
- **Manual** — type per-detector values (in ns), or paste a calibration.
- **Estimate** — fit the dead-time from the early-time count rate; **Cal**
  fits one value per detector from the preview run.

The dialog also shows the maximum correction any detector receives at
:math:`t=0`, so an unreasonable value is visible immediately. Asymmetry
normalises the non-paralyzable correction by each run's own per-run
good-frame count, so it stays bounded run-to-run; the correction is a
sub-percent bump on a well-behaved EMU histogram.

Reusing the calibration
--------------------------

When you load further runs of the same instrument, Asymmetry auto-applies
the active profile to them (the Log shows *"Auto-applied existing project
grouping"*) so a whole series inherits the same :math:`F`/:math:`B` split,
dead-time mode, and good-data window with no per-run Apply step. The
measured per-run quantities — good-frame counts, :math:`t_0` — are always
re-read from each run, never inherited.

:math:`\alpha`, however, is a *sample* calibration, not an instrument
constant. Every run inheriting the profile carries its :math:`\alpha`, so
when you switch to a new sample, start a **new profile** for that sample
and **re-Calibrate α** from a reference run before trusting the asymmetry.
If one run in a series genuinely needs its own grouping — a masked
detector, say — use the scope panel's **Release** button rather than
starting a new profile for the whole series. A finished profile is saved
with the project, so it is ready to reuse the next time that project is
opened.

Where to go next
-----------------

- :doc:`/reference/detector_grouping` — full grouping-profile and
  layout-editor reference.
- :doc:`/reference/grouping_calibration` — a second worked calibration
  walkthrough, plus the scriptable API.
- :doc:`temperature_scan_magnetism` — applies one calibrated profile
  across a temperature scan.
- :doc:`dynamic_kt_copper` — uses this setup as the front end to a
  zero-field dynamics fit.
