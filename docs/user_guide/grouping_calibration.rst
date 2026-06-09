Grouping and Calibration Walkthrough
====================================

Before you fit anything, a raw run has to be *calibrated*: the time-zero and
good-bin range set, the detectors combined into forward and backward groups, the
balance factor :math:`\alpha` fixed, and — for raw histogram formats — deadtime
correction applied. In Asymmetry this is all done in one place, the **Grouping**
window, with the detector arrangement edited visually in the **Detector Layout**
editor.

.. note::

   **Coming from WiMDA?** Asymmetry has no "edit detector list" text box. The
   visual **Detector Layout** editor — concentric detector rings you click to
   build groups, seeded by instrument presets — is its replacement. See
   `The Detector Layout editor`_ below.

This page is a practical walkthrough. For the stored grouping payload, the
deadtime/background correction maths, and the per-format (PSI / ROOT) details,
see the reference page :doc:`detector_grouping`.

The Grouping window
-------------------

Open it with the **Grouping** button on the main toolbar (or
**Analysis → Grouping…**). It operates on a *reference run* but can apply the
result to many runs at once:

* **Forward Group** / **Backward Group** — the two groups that enter the
  asymmetry. Pick them from the detector groups defined in the layout editor.
* **Alpha** and **Estimate α** — the forward/backward balance factor. Type a
  value, or click **Estimate α** to fit it from the reference run so the
  forward and backward precession signals balance about zero (see the worked
  example below).
* **t0 Bin**, **t_good Offset**, **Last Good Bin**, **Bunching Factor** — the
  time-zero bin, the first-good-bin offset after t\ :sub:`0`, the last good bin,
  and the rebinning factor applied before analysis.
* **Enable Deadtime Correction** — with three modes, **File** / **Manual** /
  **Estimate**, a per-detector nanosecond table (shown as ``H1: … ns``,
  ``H2: … ns`` …), and a **Cal** button that fits one deadtime value per
  detector from the reference run. Deadtime correction is a normal, working part
  of the pipeline; the maths is in :doc:`detector_grouping`.
* **Enable Background Correction** — an optional count-background subtraction for
  PSI-style raw histograms (off by default for ISIS/NeXus data).
* **Load .grp** / **Save .grp** — read or write a grouping definition as a
  ``.grp`` file, so a calibration can be reused across sessions.

Two controls govern *which* runs are affected:

* the **Reference run** dropdown selects the run that **Estimate α**, **Cal**,
  and the **Estimate** deadtime mode measure from;
* only the **checked** datasets in the list are changed when you press
  **Apply** — so you can calibrate on one run and push the result to a whole
  checked series in one click.

After **Apply**, the **LOG** echoes exactly what was set, for example::

   Applied grouping to 5 dataset(s); skipped 0. F=1, B=2, alpha=1.1,
   deadtime=on (applied=5, missing=0), background=off

so you can confirm the forward/backward groups, :math:`\alpha`, and the
correction states that actually landed.

The Detector Layout editor
--------------------------

Click **Detector Layout…** in the Grouping window to open the visual editor.
It draws the instrument's physical forward and backward detector arrays as
**concentric rings**, and you build groups by clicking on them:

* define up to **8** groups (**Group 1** … **Group 8**), each with its own name
  and colour;
* **click a detector segment to toggle** its membership in the active group. A
  single detector may belong to **more than one group** — which is exactly what
  transverse-field and vector-polarisation arrangements need;
* the **Preset grouping** dropdown is instrument-aware and seeds a sensible
  starting arrangement. On EMU, for instance, the presets are **Longitudinal**
  (the usual two-group forward/backward split) and **Vector Polarization**;
* **Apply Grouping** returns the arrangement to the Grouping window, where you
  then choose the forward/backward groups and :math:`\alpha`.

Start from the preset that matches your measurement, refine it by clicking
segments, and only then return to set :math:`\alpha`.

Worked example: EMU silver calibration
--------------------------------------

The standard EMU calibration is a transverse-field run on a silver sample, used
to fix :math:`\alpha` for the whole experiment.

#. **Load the TF calibration run** (EMU silver, run **44989**) and select it in
   the Data Browser.
#. **Open the Grouping window** from the toolbar. EMU's **Longitudinal** preset
   already gives the two-group forward/backward split, so the detector layout is
   ready.
#. **Click Estimate α.** For this run it returns :math:`\alpha \approx 1.10` —
   the value that balances the forward and backward TF precession signals about
   zero. (The exact figure is run-specific; trust the fit, not a fixed number.)
#. **Press Apply.** The LOG records the forward/backward groups and the fitted
   :math:`\alpha`, and the F–B asymmetry view now oscillates symmetrically about
   zero — the sign that the calibration is good.

To see calibration *propagate* across a series, load a partner run such as
**34998** after applying the calibration above: it inherits the same grouping
and :math:`\alpha` automatically (see the next section).

Auto-propagation across a series
--------------------------------

Once a project has an active grouping, runs you load afterwards inherit it
automatically — grouping, forward/backward choice, and :math:`\alpha` — and the
LOG reports::

   Auto-applied existing project grouping to 1 dataset(s); skipped 0.

This is exactly what you want for a temperature or field series measured on one
sample: calibrate once on the reference run, then load the rest of the series
and have every run share the same definition.

.. warning::

   Auto-propagation carries :math:`\alpha` across **every** subsequently-loaded
   run, including runs on a *different sample*. :math:`\alpha` is a property of
   the sample-plus-geometry, not a universal constant, so when you switch to a
   new sample you should **reset or re-Estimate α** on a calibration run for
   that sample before trusting the asymmetry. The auto-applied value is a
   convenience, not a measurement of the new sample.

See also
--------

* :doc:`detector_grouping` — the grouping payload, deadtime/background maths, and
  per-format (PSI / ROOT) grouping details.
* :doc:`loading_data` — getting runs into the session in the first place.
* :doc:`data_processing` — the grouping and asymmetry APIs behind this window.
* :doc:`vector_polarization` — building :math:`P_x` / :math:`P_y` / :math:`P_z`
  groups in the layout editor.
* :doc:`project_files` — how a calibration is saved and restored with a project.
