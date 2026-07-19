Grouping and calibration walkthrough
====================================

Before you fit anything, a raw run has to be *calibrated*: the time-zero and
good-bin range set, the detectors combined into forward and backward groups, the
calibration constant :math:`\alpha` fixed, and — for raw histogram formats — deadtime
correction applied. In Asymmetry this is a **grouping profile** — one named,
shareable calibration per instrument — edited in the **Grouping** window, with
the detector arrangement built visually in the **Detector Layout** editor and
:math:`\alpha`/deadtime/background each calibrated in its own dedicated dialog.

.. note::

   **Coming from WiMDA?** Asymmetry has no "edit detector list" text box. The
   visual **Detector Layout** editor — concentric detector rings you click to
   build groups, seeded by instrument presets — is its replacement. See
   `The Detector Layout editor`_ below.

This page is a practical walkthrough. For the grouping-profile model, the
stored payload, the deadtime/background correction maths, and the per-format
(PSI / ROOT) details, see the reference page :doc:`detector_grouping`; for the
same calibration carried out end-to-end on a real EMU run, see
:doc:`/workflows/calibration_grouping_emu`.

The Grouping window
--------------------

.. figure:: /_generated/screenshots/grouping_window_profile_editor.png
   :width: 100%
   :alt: The Grouping window profile editor, with its scope panel, status
      rows, and the live forward/backward asymmetry preview.

   The Grouping window: the scope panel (the selector) on the left, the
   :math:`\alpha`/deadtime/background status rows in the centre, and the live
   forward/backward asymmetry preview along the bottom. Two sample profiles
   are in concurrent use, with one run following each.

Open it with the **Grouping** button on the main toolbar (or
**Analysis → Grouping…**). It edits one **profile** at a time — the named,
shareable calibration for the loaded run's instrument — rather than pushing
settings out to whichever runs happen to be checked:

* **Profile selector** — which saved profile you are editing, with **New…**
  and **Duplicate…** to start fresh or branch from the current settings.
  For a project with more than one instrument loaded, an **Instrument**
  switcher beside it chooses which instrument's profile the window edits.
* **Scope panel — the selector.** Headed **Runs of this instrument**, it lists
  every run of the selected instrument, each tagged **follows <profile>** or
  **override**. The run you *select* here is the one the form previews and
  edits: selecting a run shows its effective settings, drives the live preview,
  and seeds the status rows with its own per-run facts (:math:`t_0`, good-bin
  window, file deadtime). **Release** / **Reattach** move a run between
  following the profile and carrying its own override (see
  `Scope: following, releasing, and assigning`_ below).
* **Editing-target strip** — a strip above the form that always names what your
  edits currently apply to: "Editing profile '<name>' — applies to N runs"
  while a following run is selected, or "Editing override for run N — this
  run only" while an overridden run is selected. The same tint highlights the
  selected scope row, so the two editing modes are never confused.
* **Forward Group** / **Backward Group** — the two groups that enter the
  asymmetry. Pick them from the detector groups defined in the layout editor.
* **Preset dropdown and chip** — an instrument-aware starting arrangement, with
  a chip that reads "Preset: <name>" until you edit a group by hand, at which
  point it switches to "Custom (edited from <name>)".
* **Alpha status row** — the current :math:`\alpha` and its provenance (a
  fixed value, or "diamagnetic · run 2923" for a calibrated one), with a
  **Calibrate…** button that opens the alpha calibration dialog (see the
  worked example below).
* **t0 Bin**, **t_good Offset**, **Last Good Bin**, **Bunching Factor** — the
  time-zero bin (with a **From file** / **Manual** / **Auto-detect** mode
  selector), the first-good-bin offset after t\ :sub:`0`, the last good bin,
  and the rebinning factor applied before analysis. These are per-run facts
  seeded from the selected run, not part of the profile.
* **Deadtime status row** — the current mode (off / from file / manual /
  estimated) with a **Configure…** button opening the deadtime dialog (mode,
  per-detector table, **Cal** fit, and a maximum-correction-at-t=0 summary).
* **Background status row** — the current mode with a **Configure…** button
  opening the background dialog (range / tail-fit / reference-run / fixed,
  with a shaded-window preview).
* **Live asymmetry preview** — updates automatically as you edit groups,
  :math:`\alpha`, binning, deadtime, or background, so the balancing effect of
  a change is visible immediately rather than only after Apply.

The editing target simply **follows the selected run**: pick a following run
and your edits go to the profile draft; pick a released run and they go to that
run's own override draft, which profile edits never touch. Override drafts
**accumulate** across the session — switching selection between the profile and
several overrides never prompts, and each keeps its own in-progress edits.

Press **Apply** to commit everything you have changed in one pass — the profile
to every run following the profile, plus each edited override to its own run. When an
override has pending edits the button names the blast radius, e.g.
**"Apply (profile + 2 overrides)"**, and the **LOG** reports how many
following runs the profile reached and which overrides were updated, for
example::

   Applied profile 'Silver TF' to 5 dataset(s); 1 override(s) untouched.

The only guard is closing the window with uncommitted changes, which prompts
and lists exactly what would be lost. For the full editing model — the
per-target draft accumulation and the close guard — see
:doc:`detector_grouping`.

Scope: following, releasing, and assigning
-------------------------------------------

Every run of a matching instrument either **follows** its assigned profile
(the default — a newly loaded run is assigned to the instrument's ★ default
profile) or carries its own **override** — a per-run grouping frozen at
the point it was released, which further profile edits do not touch. The
Data Browser marks an overridden run with a trailing **⊗** and a tooltip
naming the base profile it was released from.

Use the scope panel's **Release** button when one run in a series genuinely
needs a different grouping — a masked detector, say, or a one-off background
run — without pulling the rest of the series off the shared profile. Use
**Reattach** to drop that override once the run should go back to following
its profile. When the project holds several profiles for the instrument —
one per sample, typically — use **Assign to ▸** (or the Data Browser's
**Assign Grouping Profile** context menu) to move runs between them; see
:doc:`detector_grouping` for the full assignment model.

Once released, a run's override is **edited in place**: select it in the scope
panel and the editing-target strip switches to "Editing override for run N —
this run only", the form seeds from that run's own grouping, and your edits go
to a separate override draft. You can move freely between the profile and
several overrides in one session, editing each in turn, and a single **Apply**
commits them all — the profile to its following runs and every edited override
to its own run.

The Detector Layout editor
---------------------------

Click **Detector Layout…** in the Grouping window to open the visual editor.
It draws the instrument's physical forward and backward detector arrays as
**concentric rings**, and you build groups by clicking on them:

* define up to **8** groups (**Group 1** … **Group 8**), each with its own name
  and colour, with a live member count shown on each group's button;
* **click a detector segment to toggle** its membership in the active group. A
  single detector may belong to **more than one group** — which is exactly what
  transverse-field and vector-polarisation arrangements need, and the
  schematic draws every membership as its own thin slice so an overlapping
  arrangement stays visible;
* hover a detector for its id, physical label, group memberships, and
  exclusion state;
* the **Preset grouping** dropdown is instrument-aware and seeds a sensible
  starting arrangement. On EMU, for instance, the presets are **Longitudinal**
  (the usual two-group forward/backward split) and **Vector Polarization**;
* **Apply Grouping** returns the arrangement to the Grouping window, where you
  then choose the forward/backward groups and calibrate :math:`\alpha`.

Start from the preset that matches your measurement, refine it by clicking
segments, and only then calibrate :math:`\alpha`.

Worked example: EMU silver calibration
----------------------------------------

The standard EMU calibration is a transverse-field run on a silver sample, used
to fix :math:`\alpha` for the whole experiment.

#. **Load the TF calibration run** (EMU silver, run **44989**) and select it in
   the Data Browser.
#. **Open the Grouping window** from the toolbar and pick (or create) a
   profile for EMU. Its **Longitudinal** preset already gives the two-group
   forward/backward split, so the detector layout is ready.
#. **Click Calibrate…** beside the alpha status row. Because this run's field
   geometry is transverse and its field magnitude sits in the weak-TF window,
   it is highlighted and pre-selected in the calibration dialog's run dropdown.
   Choose the **Diamagnetic** method and accept: for this run it returns
   :math:`\alpha \approx 1.10` — the value that balances the forward and
   backward TF precession signals about zero. (The exact figure is
   run-specific; trust the fit, not a fixed number.) The dialog's before/after
   preview shows the precession becoming symmetric about zero as
   :math:`\alpha` is applied.
#. **Press Apply.** The LOG records the profile name and how many runs it
   reached, and the F–B asymmetry view now oscillates symmetrically about
   zero — the sign that the calibration is good.

To see calibration *propagate* across a series, load a partner run such as
**34998**: it follows the same profile — grouping, forward/backward choice,
and :math:`\alpha` — automatically (see the next section).

Inheritance across a series
------------------------------

Once a project has a default profile for an instrument, runs you load
afterwards are assigned to it automatically — no per-run Apply step is needed — and
the LOG reports::

   Auto-applied existing project grouping to 1 dataset(s); skipped 0.

This is exactly what you want for a temperature or field series measured on
one sample: calibrate once, then load the rest of the series and have every
run share the same profile. If a later edit changes the profile — a
recalibrated :math:`\alpha`, a different deadtime mode — every following run
picks up the change the next time it is displayed or reduced, with no
broadcast step required.

.. warning::

   A profile's :math:`\alpha` is carried onto **every** run of the same
   instrument assigned to it. Runs on a *different sample* are better kept
   on their own profile — assign each sample's runs to its own profile so
   the calibration never leaks across samples (see :doc:`detector_grouping`).
   :math:`\alpha` is a property of the sample-plus-geometry, not a universal
   constant, so when you switch to a new sample you should start (or switch
   to) a **different profile** and **re-Calibrate α** on a reference run for
   that sample before trusting the asymmetry. The inherited value is a
   convenience, not a measurement of the new sample.

Recomputing asymmetry from the API (custom grouping and α)
----------------------------------------------------------

Everything the Grouping window does is also reachable from the core API, so a
custom forward/backward grouping, an estimated :math:`\alpha`, and the
recomputed asymmetry can all be produced in a script without the GUI. The
pieces live in a few modules; this section is the map.

.. warning::

   **The default** :math:`\alpha = 1.0` **is wrong for most instruments.**
   Reading ``dataset.asymmetry`` straight off a freshly loaded run uses
   :math:`\alpha = 1.0` — every current loader (PSI-BIN, MusrRoot, and ISIS
   NeXus) defaults its own reduction this way by design, leaving calibration
   explicit — which only holds when the forward and backward groups
   are already perfectly balanced. On MUSR, EMU, and the PSI spectrometers it
   is not, so any *quantitative* asymmetry needs an explicit :math:`\alpha`
   estimate and a recompute. (For reference, the MUSR ``MUSR00044989``
   calibration groups detectors **1–32 as Back** and **33–64 as Fwd**, giving
   :math:`\alpha \approx 1.103`.)

Where each function lives
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`~asymmetry.core.transform.estimate_alpha` /
  :func:`~asymmetry.core.transform.estimate_alpha_detailed`
  (``asymmetry.core.transform``) — estimate :math:`\alpha` from grouped counts.
  The simple form is a count-ratio (:math:`\sum F / \sum B`, Mantid
  ``AlphaCalc``); the **detailed** form (``method="diamagnetic"`` by default)
  returns an :class:`~asymmetry.core.transform.AlphaEstimate` with an
  uncertainty and is the one to use for the diamagnetic (TF balance) method.
* ``apply_grouping(histograms, group_indices)``
  (``asymmetry.core.io.nexus``; also re-exported from
  ``asymmetry.core.transform``) — sums the listed detector histograms into one
  grouped counts array. ``group_indices`` are **0-based** indices into the
  run's histogram list.
* :func:`~asymmetry.core.transform.compute_asymmetry`
  (``asymmetry.core.transform``) — the standard fixed-width pair asymmetry
  :math:`(F - \alpha B)/(F + \alpha B)` with a chosen :math:`\alpha`; returns
  ``(asymmetry, error)``.
* ``binned_fb_asymmetry`` (``asymmetry.core.representation.time``) — the
  variable / constant-error binning path (it raises on the ``"fixed"`` binning
  mode; use ``compute_asymmetry`` for ordinary fixed-width bins).
* ``slice_to_good_window`` (``asymmetry.core.data.combine``) — trim a computed
  asymmetry/error pair to the good-bin window.
* ``prepare_histograms_with_deadtime`` (``asymmetry.core.fitting.grouped_time_domain``)
  — apply deadtime correction to the histograms before grouping.
* :func:`~asymmetry.core.transform.reduce.reduce_grouped_asymmetry`
  (``asymmetry.core.transform.reduce``) — the single reduction chokepoint the
  GUI and the Grouping window's live preview both call: deadtime, grouping,
  optional background, then the counts-then-ratio asymmetry, in one function.
* :func:`~asymmetry.core.project.profiles.resolve_effective_grouping`
  (``asymmetry.core.project.profiles``) — merge a
  :class:`~asymmetry.core.project.profiles.GroupingProfile` with a run to
  produce the same grouping payload shape the functions above consume; the
  scriptable equivalent of a run following a profile in the GUI.

Minimal workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from asymmetry.core.io import load
   from asymmetry.core.transform import (
       apply_grouping,
       estimate_alpha,
       compute_asymmetry,
   )

   ds = load("MUSR00044989.nxs")
   histograms = ds.run.histograms

   # MUSR F/B grouping: detectors 1–32 = Back, 33–64 = Fwd (1-based);
   # apply_grouping takes 0-based histogram indices.
   backward = apply_grouping(histograms, list(range(0, 32)))    # detectors 1–32
   forward = apply_grouping(histograms, list(range(32, 64)))    # detectors 33–64

   alpha = estimate_alpha(forward, backward)   # ≈ 1.103 for this run
   asymmetry, error = compute_asymmetry(forward, backward, alpha=alpha)

For the lower-level grouping/asymmetry APIs (deadtime-first ordering, the error
model, PSI background correction) see :doc:`data_processing`.

See also
--------

* :doc:`detector_grouping` — the grouping-profile model, the stored payload,
  deadtime/background maths, and per-format (PSI / ROOT) grouping details.
* :doc:`loading_data` — getting runs into the session in the first place.
* :doc:`data_processing` — the grouping and asymmetry APIs behind this window.
* :doc:`vector_polarization` — building :math:`P_x` / :math:`P_y` / :math:`P_z`
  groups in the layout editor.
* :doc:`project_files` — how a profile is saved and restored with a project.
