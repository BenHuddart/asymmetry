EMU detector grouping and α calibration
=======================================

.. admonition:: Example data

   :ref:`Download the Rigi school data <rigi-school-data>` for the
   ISIS calibration and steering runs in the Basics folder.

   Use your own transverse-field calibration run, representative sample runs, and
   the detector timing, grouping, and dead-time information for your instrument.
   The EMU run numbers below identify the demonstration, not files included with
   Asymmetry. Replace them with your runs and check the instrument defaults before
   applying a profile to a series.

How do we tell detector imbalance and timing effects from a change in
the muon signal? Use the silver runs to see how **Grouping** combines
detector counts, how detector balance centres a clean transverse-field
oscillation, and how timing and dead-time settings affect the first bins.
Then use **Fit Parameters** to find the steering current that minimises
the silver-mask precession amplitude.

The runs
--------

The walkthrough uses HDF4 NeXus files in the Rigi school archive's
``Basics/data/`` folder. The table identifies the run for each calibration step.

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Run
     - Instrument / sample
     - Role
   * - ``MUSR00044989``
     - MuSR — Ag on candlestick
     - Detector grouping — 64 detectors summed into two groups (Step 2).
   * - ``EMU00018854``
     - EMU — silver, 100 G TF
     - :math:`\alpha` calibration from a clean transverse-field run (Step 3).
   * - ``emu00034998``
     - EMU — silver, high count rate
     - Dead-time correction, Off versus loaded (Step 5).
   * - ``EMU00018850``
     - EMU — silver, TF
     - :math:`t_0` and the good-data window (Step 6).
   * - ``EMU00044989``–``44997``
     - EMU — Ag mask on Fe₂O₃, 100 G TF
     - Steering-current trend across a run series (Step 7).

Silver is the calibrant for this example. In a small transverse field
it produces a clean, slowly relaxing oscillation. With the detector
groups assigned, a baseline offset indicates an imbalance. The
high-count-rate silver run lets you inspect the detector dead-time
correction separately.

Step 1 — Load a run
--------------------

Use **File → Open Data File(s)…** to load ``MUSR00044989`` in one project
tab. Choose **File → New Project** and load the EMU files together in the
new tab. The MuSR and EMU files numbered 44989 need separate tabs because
each project has one dataset per run number. See :ref:`run-number-conflicts`
for the message shown if both files are opened in the same tab.

Use the MuSR tab for the detector-grouping example in step 2. Switch to
the EMU tab for the calibration and steering-current exercises. Any EMU
run assigned to the profile follows the settings you build below.

Step 2 — Open the Grouping window
-----------------------------------

Click **Grouping** on the toolbar (or **Analysis → Grouping…**) and create
a profile for the instrument (or pick an existing one). The window is the
whole calibration surface for that profile:

.. figure:: /_generated/corpus_screenshots/corpus_basics_grouping.png
   :width: 100%
   :alt: The Grouping window on MUSR00044989, with the scope panel and
      group table on the left, the calibration form in the centre, and the
      live forward/backward asymmetry preview along the bottom.

   The Grouping window on the muon-school grouping run ``MUSR00044989``
   (Ag on candlestick), where the MuSR spectrometer's 64 detectors are the
   clearest demonstration of grouping. The **group table** on the left sums
   the detectors into two groups of 32 — one holding detectors 1–32, the
   other 33–64 — and the **Forward Group** / **Backward Group** selectors
   pair them into the :math:`F` and :math:`B` sums (here 33–64 forward,
   1–32 backward). The **α (detector balance)** card shows the manual value
   ``1.000000`` for this grouping example. The file supplies **t0 Bin**
   = 30, **t_good Offset** = 6, and **Last Good Bin** = 2048. **Deadtime**
   is **Off**. The preview redraws as you change the settings. The amber
   banner suggests a transverse grouping for this zero-field run. Keep
   the displayed grouping for the example.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - What it sets
   * - **Profile selector**
     - Which named profile you are editing, with **Rename…** to relabel it.
   * - **Scope panel** ("Runs of this instrument")
     - Every run of this instrument, tagged *follows <profile>* or *override*. This is
       the **selector**: the run you pick here is the one the form previews and
       edits, and its per-run facts (:math:`t_0`, good-bin window, file
       dead-time) seed the status rows. **Release from profile** /
       **Reattach to profile** move a run between inheriting the profile and
       carrying its own override.
   * - **Editing-target strip**
     - A strip above the form that names what your edits currently apply to —
       "Editing profile '<name>' — applies to N runs" while an inheriting run
       is selected, or the override wording while an overridden run is selected.
   * - **Forward Group / Backward Group**
     - Which detector groups form the :math:`F` and :math:`B` sums of the
       asymmetry.
   * - **α (detector balance)**
     - The coefficient and its provenance. Expand the card to choose the
       calibration run and method, then click **Estimate calibration**.
   * - **t0 Bin**
     - The time-zero bin (the muon-implantation prompt), with a mode selector
       — **From file** (the default), **Manual**, or **Auto-detect** — and a
       **Find t0** button for the automatic search.
   * - **t_good Offset**
     - First good bin, measured as an offset after :math:`t_0`.
   * - **Last Good Bin**
     - End of the good-data window.
   * - **Binning**
     - Time-bin rebinning; the **Fixed** mode merges a fixed number of raw
       bins (the bunching factor, also available on the toolbar).
   * - **Exclude Detectors**
     - A list of detector ids to drop from every group (masked or dead tubes).
   * - **Deadtime**
     - Expand the card for **Off**, **From file**, **Manual**, or
       **Estimate from run**, the per-detector table, and **Cal**.
   * - **Background**
     - Expand the card to select a constant-background subtraction.
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

In the **α (detector balance)** card of the Corrections column, pick the
silver TF calibration run (``EMU00018854``, silver in a 100 G transverse
field) from the **Calibration run** dropdown. Because this run's field
geometry is transverse and its magnitude sits in the weak-TF window, it is
highlighted and pre-selected there. For this quick cross-check choose **Diamagnetic** and press
**Estimate calibration**. The fresh explicit default is **TF calibration fit**;
these are different estimators. **Apply** commits the displayed result.

.. figure:: /_generated/corpus_screenshots/corpus_basics_alpha.png
   :width: 80%
   :align: center
   :alt: A completed Diamagnetic calibration in the draft alpha spinbox and balanced preview.


   The inline alpha calibration on the silver TF run ``EMU00018854``.
   With **Diamagnetic** and the good-bin window 21–1999, the quick result is
   :math:`\alpha = 0.88487`. The alpha spinbox and
   preview update immediately; **Apply** commits the draft to the run/project.


Choose the **Diamagnetic** method. On a silver TF run the precession
is a clean, non-relaxing oscillation, and the correct :math:`\alpha` is the
one that makes the forward/backward asymmetry oscillate *symmetrically
about zero* — an imbalanced :math:`\alpha` leaves the precession riding on
a non-zero offset. Asymmetry finds it by minimising the weighted asymmetry
power over the good-data window,

.. math::

   \alpha = \arg\min_\alpha \sum_i \left(\frac{A_i}{\sigma_i}\right)^2,
   \qquad A_i = \frac{F_i - \alpha B_i}{F_i + \alpha B_i},

with :math:`\sigma_i` the Poisson-propagated per-bin asymmetry error —
WiMDA's diamagnetic estimate. The **Count ratio ΣF/ΣB** method instead
computes the simple integral balance :math:`\alpha = \sum_i F_i / \sum_i
B_i` (Mantid's ``AlphaCalc``), with no oscillation model; on a clean silver
run the two agree closely. The **Decay balance** method balances
lifetime-corrected counts and accommodates a genuinely relaxing signal, but
needs visible relaxation. On this silver TF run the diamagnetic estimate
gives :math:`\alpha = 0.88487`. The alpha spinbox and preview update immediately,
with **Diamagnetic** beside the coefficient until it is edited by hand. **Apply**
commits the draft values and their captured context. The result reads
``α = 0.8849 · β = 1 (fixed) · Diamagnetic``.

The quick estimate reports no added uncertainty. The same calculation is
available from the Python API. The diamagnetic fit
lives in :func:`~asymmetry.core.transform.asymmetry.estimate_alpha_detailed`; the
simpler integral ratio (equivalent to the **Count ratio ΣF/ΣB** method) is
:func:`~asymmetry.core.transform.asymmetry.estimate_alpha`:

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.workflow import CalibrationOptions, ReductionSettings, calibrate_run

   # Replace with your TF calibration file; install its HDF5/HDF4 reader extra.
   tf = load("your_tf_calibration.nxs", period=1)
   outcome = calibrate_run(
       tf, ReductionSettings(), CalibrationOptions(method="diamagnetic")
   )
   print(round(outcome.result.estimate.alpha, 3))

Press **Apply** to write the calibrated :math:`\alpha` back to the profile.
Any run of the same instrument assigned to the profile follows it — grouping and
:math:`\alpha` included — the moment it is loaded (see `Reusing the
calibration`_ below), so nothing further needs to be selected or applied on
it by hand. Note, though, that :math:`\alpha` is a *sample* calibration:
carry it only to runs of the same sample and detector arrangement.

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
   (Shown here on EMU for its longitudinal geometry; the grouping window above
   is the same editor reached from any loaded instrument.)


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

For this silver exercise the **Longitudinal** preset is the correct
starting point; the editor is most useful when a detector is masked or
when you build a custom geometry. See
:doc:`/reference/detector_grouping` for the full reference.

Step 5 — Dead-time correction
--------------------------------

After a detector records a positron there is a short interval before it can
register the next; at the high instantaneous rates of a pulsed source a
positron arriving inside that interval is lost, and the count must be
statistically recovered. The per-detector dead-times are measured once from
a silver sample and stored in the data file (Kilcoyne, RAL-94-080).

Open the **Deadtime** card and choose the mode you want to inspect.

- **From file** — use the per-detector dead-time values stored in the data
  file (the usual choice, and WiMDA's "Auto Load").
- **Manual** — type per-detector values (in ns), or paste a calibration.
- **Estimate from run** — fit the dead-time from the early-time count rate;
  the **Cal** button fits one value per detector from the preview run.

The card also reports **Max correction at t=0**, the largest correction any
detector receives, so an unreasonable value is visible immediately.

.. figure:: /_generated/corpus_screenshots/corpus_basics_deadtime.png
   :width: 100%
   :alt: Forward/backward asymmetry of silver run emu00034998 with dead-time
      correction Off (grey) and From file (blue). The curves are furthest
      apart at the first plotted point.

   Forward/backward asymmetry of the high-rate silver run ``emu00034998``
   with dead-time **Off** (grey) and with the per-detector correction applied
   **From file** (blue). Both reductions use :math:`\alpha = 1`, background
   correction off, and 25 native bins per point (0.400 µs). Over the plotted
   interval through 12 µs, the largest difference is about 0.3 percentage
   points at the first point, where the count rate is highest.

Step 6 — Timing: t0 and the good-data window
----------------------------------------------

At a pulsed source the timing origin :math:`t_0` is when the *middle* of
the muon pulse reaches the sample. The good data do not begin until the
*whole* pulse has arrived — that time is :math:`t_\text{good}`, and the
difference :math:`t_\text{good} - t_0` is the **good-data offset**. Both are
normally read straight from the file (the instrument scientist sets them),
which is what **t0 Bin → From file** does; you rarely edit them, but it
helps to see what they mark.

.. figure:: /_generated/corpus_screenshots/corpus_basics_t0.png
   :width: 100%
   :alt: Summed detector counts near the muon pulse for EMU00018850, with
      t0 marked at the mid-pulse and tgood after the whole pulse has
      arrived, the interval between them shaded.

   The summed detector counts of the silver run ``EMU00018850`` through the
   muon pulse. The prompt rises over roughly 0.1 µs; the stored
   :math:`t_0 = 0.224\ \mu\text{s}` (bin 14) sits at the mid-pulse and the
   stored :math:`t_\text{good} = 0.336\ \mu\text{s}` (bin 21) sits after the
   whole pulse has passed, leaving a good-data offset of ≈ 112 ns — the
   finite pulse width of a pulsed source. On a continuous (DC) source, with
   no pulse broadening, that offset is very small.

If you ever need to *determine* :math:`t_0` rather than read it, the
muon-school method is field-independence of the TF phase: fit a damped TF
oscillation across a field series (silver at several transverse fields),
plot the fitted phase :math:`\varphi` against the Larmor angular frequency
:math:`\omega = 2\pi\nu` (with :math:`\gamma_\mu / 2\pi = 13.6` kHz per G),
and adjust :math:`t_0` until the gradient :math:`\mathrm{d}\varphi /
\mathrm{d}\omega` vanishes. A correct :math:`t_0` makes the phase
field-independent. For EMU the fitted value lands near the stored ≈ 0.24 µs.

Step 7 — A fit table with a manual column
-------------------------------------------

The last skill is turning a *series* of runs into one answer. The
muon-school steering exercise scans the horizontal beam-steering current
across nine runs (``EMU00044989``–``44997``, an Ag mask over rapidly
depolarising Fe₂O₃ in a 100 G transverse field) and asks which current
centres the beam. Each run is fitted with a single non-relaxing oscillating
component; the observable is the Ag-mask initial asymmetry :math:`a_0`, and
the trend to plot is :math:`a_0` against steering current.

The steering currents come from the school guide, ``Basics 2026.docx``.
In the Data Browser, use **+ → Custom column…** to add a
:ref:`logbook column <logbook-columns>` named ``Steering current (A)``.
Enter the current on each corresponding run. Open **View → Show Fit Parameters**, select the fit series, and use
that column on the **x** rail. Choose the amplitude on the **y** rail,
then select **Fit…** on its card and choose ``Cubic``.

.. figure:: /_generated/corpus_screenshots/corpus_basics_steering.png
   :width: 100%
   :alt: The Fit Parameters trending panel plotting Ag-mask initial
      asymmetry against steering current, a manually-entered column, with a
      fitted cubic overlaid whose minimum sits near zero current.

   The **Fit Parameters** panel plotting the Ag-mask initial asymmetry
   :math:`a_0` against **Steering current (A)** — the manual column, chosen
   on the x rail's abscissa picker. The nine per-run :math:`a_0` values rise
   from ≈ 7 % at ±1 A to a shallow minimum near zero, and a **Cubic** fit
   (the card's **Fit ✓** button) is overlaid on the points. The fitted
   minimum falls at
   :math:`I = -0.060` A: with the beam centred, muons pass *through* the Ag
   aperture onto the depolarising Fe₂O₃, so the Ag-mask signal is lowest at
   the centred current. This figure applies the **Cubic** trend fit to the
   WiMDA per-run fit results. Fit the public EMU runs to obtain your own
   per-run asymmetries and locate the minimum in Asymmetry.

Export the fitted trend with **⋯ → Export TSV…** or **⋯ → Export GLE (PDF)…**.

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
detector, say — use the scope panel's **Release from profile** button
rather than starting a new profile for the whole series. A finished profile
is saved with the project, so it is ready to reuse the next time that
project is opened.

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

.. rubric:: References

- Kilcoyne, RAL Report RAL-94-080 (1994) — per-detector dead-time
  determination from a silver sample.
