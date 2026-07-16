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

Click **Grouping** on the toolbar (or **Analysis → Grouping…**) and create
a profile for EMU (or pick an existing one). The window is the whole
calibration surface for that profile:

.. figure:: /_generated/screenshots/grouping_window_profile_editor.png
   :width: 100%
   :alt: The Grouping window, with the scope panel on the left, the
      status rows and calibration buttons in the centre, and the live
      forward/backward asymmetry preview along the bottom.

   The Grouping window as you first meet it (shown here on a representative
   synthetic transverse-field run rather than the EMU silver run itself). The
   **Runs of this instrument** scope panel on the left is the selector: the run
   you pick there is the one the form previews and edits, and the live
   asymmetry along the bottom redraws as you change the grouping,
   :math:`\alpha`, binning, dead-time, or background.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - What it sets
   * - **Profile selector**
     - Which named profile you are editing — **New…** / **Duplicate…** to
       start fresh or branch from the current settings.
   * - **Scope panel** ("Runs of this instrument")
     - Every run of this instrument, tagged *inherits* or *override*. This is
       the **selector**: the run you pick here is the one the form previews and
       edits, and its per-run facts (:math:`t_0`, good-bin window, file
       dead-time) seed the status rows. **Release** / **Reattach** move a run
       between inheriting the profile and carrying its own override.
   * - **Editing-target strip**
     - A strip above the form that names what your edits currently apply to —
       "Editing profile '<name>' — applies to N runs" while an inheriting run
       is selected, or "Editing override for run N — this run only" while an
       overridden run is selected.
   * - **Forward Group / Backward Group**
     - Which detector groups form the :math:`F` and :math:`B` sums of the
       asymmetry.
   * - **Alpha status row / Calibrate…**
     - The current :math:`\alpha` and its provenance, with a button that
       opens the alpha calibration dialog (see Step 3).
   * - **t0 Bin**
     - The time-zero bin (the muon-implantation prompt), with a mode selector
       — **From file** (the default), **Manual**, or **Auto-detect**.
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
   * - **Apply**
     - Write the draft back to the profile — and any edited overrides to their
       own runs, in one pass. The Log echoes how many inheriting runs it
       reached.

Step 3 — Calibrate α from the TF run
--------------------------------------

In the **α (detector balance)** section of the Corrections panel, pick the TF
calibration run (``EMU00044989``) from the **Calibration run** dropdown.
Because this run's field geometry is transverse and its magnitude sits in the
weak-TF window, it is highlighted and pre-selected there. Choose a **Method**
and press **Estimate α**.

.. figure:: /_generated/screenshots/alpha_calibration_dialog.png
   :width: 80%
   :align: center
   :alt: The inline alpha calibration controls, with a highlighted
      transverse-field calibration run, a method combo, an Estimate α button,
      and the shared before (α = 1) / after (fitted α) asymmetry preview.

   The inline alpha calibration, shown here on a representative synthetic
   transverse-field run (your EMU silver run ``44989`` reports
   :math:`\alpha \approx 1.103`). The **Calibration run** dropdown highlights
   the likely TF calibration run, and the shared before/after preview shows the
   precession becoming symmetric about zero once the fitted :math:`\alpha` is
   applied.

Choose the **Diamagnetic (TF)** method. On a silver TF run the precession is a
clean, non-relaxing oscillation, and the correct :math:`\alpha` is the one
that makes the forward/backward asymmetry oscillate *symmetrically about
zero* — an imbalanced :math:`\alpha` leaves the precession riding on a
non-zero offset. Asymmetry finds it by minimising the weighted asymmetry
power over the good-data window,

.. math::

   \alpha = \arg\min_\alpha \sum_i \left(\frac{A_i}{\sigma_i}\right)^2,
   \qquad A_i = \frac{F_i - \alpha B_i}{F_i + \alpha B_i},

with :math:`\sigma_i` the Poisson-propagated per-bin asymmetry error — WiMDA's
diamagnetic estimate. (The **Count ratio ΣF/ΣB** method instead computes the
simple integral balance :math:`\alpha = \sum_i F_i / \sum_i B_i`, Mantid's
``AlphaCalc``, with no oscillation model; on a clean Ag run the two agree
closely. The **General (LF/ZF)** method accommodates a genuinely relaxing or
multi-component TF signal.) On the Ag TF run all of these give
:math:`\alpha \approx 1.103`, and the shared before/after preview shows the
balancing effect directly. The estimate applies immediately: the α section now
reads something like "α = 1.103(2) · Diamagnetic (TF) · run 44989".

The same calculation is available from the Python API. The diamagnetic fit
lives in :func:`~asymmetry.core.transform.estimate_alpha_detailed`; the
simpler integral ratio (equivalent to the **Count ratio ΣF/ΣB** method) is
:func:`~asymmetry.core.transform.estimate_alpha`:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.transform import apply_grouping, estimate_alpha_detailed

   tf = load("Basics/data_hdf5/EMU00044989.nxs")
   g = tf.run.grouping
   groups = g["groups"]                             # {group_id: [1-based detector ids]}
   fwd = [int(d) - 1 for d in groups[g["forward_group"]]]
   bwd = [int(d) - 1 for d in groups[g["backward_group"]]]
   forward = apply_grouping(tf.run.histograms, fwd)
   backward = apply_grouping(tf.run.histograms, bwd)
   est = estimate_alpha_detailed(
       forward, backward,
       method="diamagnetic",
       first_good_bin=g.get("first_good_bin"),
       last_good_bin=g.get("last_good_bin"),
   )
   print(round(est.alpha, 3))                        # 1.103

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

.. figure:: /_generated/screenshots/emu_longitudinal_layout.png
   :width: 100%
   :alt: The Detector Layout editor on EMU's Longitudinal preset, showing the
      forward and backward detector rings split into two groups.

   The Detector Layout editor on EMU's **Longitudinal** preset — the two-group
   forward/backward split this exercise uses. The preset dropdown seeds the
   arrangement; clicking a segment toggles its membership in the active group.


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
