GUI usage
=========

This chapter is the comprehensive reference for the Asymmetry GUI — menus,
panels, dialogs, keyboard shortcuts. New users coming from a real
experiment should normally read :doc:`/workflows/index` first and dip into
this chapter as a lookup when the case studies reference a specific
dialog; the section headings below are organised for that kind of
targeted navigation rather than for cover-to-cover reading.

Launching the GUI
-----------------

From the command line:

.. code-block:: bash

   asymmetry-gui

Or from Python:

.. code-block:: python

   from asymmetry.gui.app import main
   main()

Main window layout
------------------

.. image:: /_generated/screenshots/main_window.png
   :alt: Asymmetry main window with an EuO temperature scan loaded
   :width: 100%

*Synthetic EuO ferromagnet temperature scan crossing the Curie point at*
*Tc=69 K — six zero-field runs from 30 K up to 90 K (cf. Blundell et al.,*
*Muon Spectroscopy: An Introduction, Oxford University Press, Oxford, 2022,*
*Fig. 6.6). The selected run at 65 K is just inside the ordered phase where*
*the spontaneous-field precession is at its slowest and the critical damping*
*is largest.*

The main window has four main areas:

1. **Data Browser** (left): Table of loaded datasets
2. **Plot Panel** (centre): Interactive plot display
3. **Analysis Panels** (right): Fit and Fourier controls
4. **Log Panel** (bottom): Status messages and command history

Loading data
------------

Multiple file selection
~~~~~~~~~~~~~~~~~~~~~~~

1. Click **File → Open** (or toolbar "Open" button)
2. Select one or multiple supported data files (use Ctrl+Click or Shift+Click)
3. All selected files will be loaded into the data browser

The data browser shows:

* Run number
* Title
* Temperature (K)
* Magnetic field (G)

Use **Options → Use temperature from log** to switch the fixed temperature
column from the scalar/header temperature to the average of the sample
temperature log when a supported log is available. The menu option applies the
same choice to every loaded run and clears any per-run overrides. Temperature
cells whose value comes from a log are shown with red text. Untick the option
to return the column to the scalar/header value.

The value shown here is also the one a batch parameter trend plots against and
exports to TSV: with the option on, a temperature trend uses each run's logged
temperature rather than its parked setpoint — see
:ref:`trend-abscissa-coordinate`.

Data Browser features
---------------------

Sorting
~~~~~~~

Click any column header to sort by that field:

* **Left-click**: Toggle between ascending and descending order
* Numeric columns (Run, T(K), B(G)) sort numerically
* Text columns (Title) sort alphabetically

The sorting is robust and works correctly even after filtering.

**Excel-style column filtering**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_generated/screenshots/data_browser_filter.png
   :alt: Data browser populated with an EuO scan plus the T(K) column filter dialog
   :width: 100%

*Right-click filter dialog applied to the T(K) column of the EuO ZF*
*temperature scan, with the three runs at or below Tc=69 K selected to*
*isolate the ordered ferromagnetic phase.*

Right-click any column header to open an Excel-style filter dialog:

1. **Right-click** a column header (Run, Title, T(K), or B(G))
2. A dialog appears showing all unique values in that column
3. **Check/uncheck** values to show/hide rows with those values
4. Use **Select All** / **Deselect All** for quick control
5. Click **OK** to apply the filter, or **Cancel** to abort

Features:

* Multiple columns can be filtered simultaneously
* Filters persist while you work with other data
* Clear filters by right-clicking and selecting all values again
* Works seamlessly with sorting

Multi-selection
~~~~~~~~~~~~~~~

* Click a row to select it
* Ctrl+Click to add/remove from selection
* Shift+Click to select a range

When more than one dataset is selected, the main plot overlays all selected
runs on the same axes using distinct colours when **Overlay** is enabled
(default).

If **Overlay** is disabled, only the most recently selected dataset is shown.
For a single selected Data Group, Asymmetry keeps the most recently displayed
dataset if it belongs to that group; otherwise it shows the first dataset in
the group.

Viewing data
~~~~~~~~~~~~

Click any dataset row to plot it in the main panel.

Context menu
~~~~~~~~~~~~

Right-click any dataset row to access options:

* **Co-add Selected** (appears when 2+ datasets selected): Combine selected datasets into one averaged dataset
* **Separate Combined** (appears when a combined dataset is selected): Break apart a combined dataset back into its source datasets
* **Remove Entry** or **Remove Selected Entries**: Delete the selected dataset(s)

These options appear contextually based on your current selection.

Data groups
~~~~~~~~~~~

.. image:: /_generated/screenshots/data_browser_groups.png
   :alt: Data browser with a blue user group and a red-grey auto group sharing a marked duplicate row
   :width: 100%

*A user group (blue, "T < Tc — EuO") and an auto-created group (red-grey,*
*"Runs 3003–3005", named the way an ad-hoc batch fit would mint one*
*automatically) sharing run 3003. The shared run renders under its primary*
*(user-group) row and again as a marked copy row under the auto group.*

A **data group** is a named, ordered collection of runs — a temperature scan,
a field sweep, or any set of runs you want to treat together. Groups organise
the Data Browser *and* drive batch and global fitting: a group is the vehicle
a batch fit runs over, and its owned fit series stay attached to it (see
:ref:`group-bound-series-staleness` in :doc:`parameter_trending`).

Select two or more loaded runs, right-click, and choose **Form Data Group** to
create one; a **Group name:** prompt asks for its display name. Groups appear
as headers in the browser with their member runs nested beneath, and a
group's context menu (right-click its header) offers:

* **Collapse Group** / **Expand Group** — hide or show its member rows.
* **Rename Group** — change the display name.
* **Ungroup** — dissolve the group. If it owns no fit series this happens
  silently; if it does, a dialog asks whether to **Keep fits** (the owned
  series become standalone, frozen analyses with their last-fitted membership
  as a fixed snapshot) or **Delete fits** (remove the group and its series
  together), with **Cancel** to back out.
* **Fit this group…** — bind the fit dock's Batch tab to this group so the
  recorded series stays attached to it; see :ref:`batch-tab-groups` below.
* **Show series from this group** — filter the Fit Parameters panel to this
  group's series.

Right-click a selection of ungrouped (or already-grouped) runs and choose
**Send to Group** to add them to an existing group — this always *adds* a
membership rather than moving the run out of any group it already belongs to,
so one run can belong to several groups at once (a run in both a field scan
and a temperature scan, say). A run with more than one membership renders
once per group: its entry in every group but the first (its *primary*
membership) carries a small circled-digit marker (①, ② …), and hovering the
marker shows an **Also in:** tooltip naming its other groups. Selecting any
of a run's rows reaches the same underlying dataset, so plotting, fitting, and
co-add never double-count it. **Remove from Group** (or **Remove from
Groups**, for a multi-row selection) removes just the clicked membership;
removing a run's primary membership promotes its earliest remaining copy to
primary.

Two further markers relate to detector grouping (see
:doc:`detector_grouping`): a run released from its grouping profile carries a
trailing **⊗**, and when a run's instrument has several grouping profiles in
the project (one per sample, say) a superscript index (¹, ², …) says which
profile the run follows, with the tooltip "Grouping profile: <name>" naming
it. Right-clicking a run selection then offers **Assign Grouping Profile**
to move the runs to another of the instrument's profiles.

Batch and global fits over an ad-hoc run selection (not bound to a group)
automatically create — or, for an identical run set, reuse — a group for the
fit, so every recorded batch series always has an owning group. These
**auto-created** groups are named from their run range (e.g. "Runs
1001–1010") and paint in a red-grey tint distinct from the blue used for
groups you name yourself, so the two are easy to tell apart at a glance;
renaming an auto-created group promotes it to an ordinary (blue) group.

Co-adding datasets
~~~~~~~~~~~~~~~~~~

To average multiple datasets:

1. Select 2 or more datasets (Ctrl+Click or Shift+Click)
2. Right-click any selected row and choose "Co-add Selected"
3. A new combined dataset appears at the position of the first selected dataset
4. Display shows run numbers combined: ``3077 + 3076``

The co-added dataset:

* Uses the first dataset's time grid
* Averages asymmetry values
* Propagates errors correctly: σ_combined = √(Σσ²) / N
* Requires identical grouping on every selected source dataset
* Mirrors the shared grouping state of its source datasets
* Can be plotted and analysed like any other dataset

If the selected datasets do not share the same grouping (groups, alpha,
good-bin limits, bunching, deadtime settings, and related grouped-data
controls), co-add is blocked and the browser leaves all source rows unchanged.
Align grouping first, then retry the co-add.

Separating combined data
~~~~~~~~~~~~~~~~~~~~~~~~

To remove a co-added dataset:

1. Select the combined dataset (display shows combined run numbers)
2. Right-click and choose "Separate Combined"
3. The combined entry is replaced by its source datasets at the same position
4. Source datasets are restored with the current grouping of the combined view

Grouping a combined dataset edits its hidden source datasets behind the scenes,
then rebuilds the combined row from those updated sources. Separating the
combined dataset therefore returns single runs with the same grouping that was
active on the combined entry.

Deleting datasets
~~~~~~~~~~~~~~~~~

To remove a dataset from the browser:

1. Select one or more datasets
2. Right-click and choose "Remove Entry" or "Remove Selected Entries"
3. Alternatively, press the **Delete** key

This removes the dataset(s) from the browser; they can be reloaded via **File → Open**.

Exporting the logbook
~~~~~~~~~~~~~~~~~~~~~

Use the **Export logbook** toolbar button (immediately after **Open**) to write
the current Data Browser contents to a file.

Export behaviour:

* Exports the currently active Data Browser columns, including dynamic columns
   added from **Run Info**.
* Preserves data-group organisation by writing a section header for each group.
* Includes datasets hidden by filters and collapsed groups.
* Writes section headers and data rows with a consistent column count so header
   labels and values align cleanly.

Formats:

* **TSV (recommended, default)**: Fastest export path and best choice for large
   logbooks. Opens directly in spreadsheet tools with clear column alignment.
* **RTF**: Rich text output with italicised **T** and **B** header labels.
   Useful for report-style sharing, but slower than TSV for large tables.

Default filename:

* If the current project has a name (for example ``My_Project.asymp``), the
   default export filename is ``My_Project_logbook.tsv``.
* If no project name is available, the default is ``logbook.tsv``.

Plot panel controls
-------------------

The plot panel displays the selected dataset(s) with error bars.

Axis limits
~~~~~~~~~~~

Control the plot range using the spinboxes at the top:

* **X min/max**: Time axis range (μs)
* **Y min/max**: Asymmetry axis range
* Press **Enter** after editing any limit value to apply immediately
* Click **Auto X** to auto-scale the X axis only
* Click **Auto Y** to auto-scale the Y axis only

Auto-Y uses points inside the currently selected X range and prefers reliable
foreground points (excluding undefined/low-confidence bins when available). On
the Frequency-domain plot, **Auto X** frames the spectrum sensibly — the
dominant line, or the field-derived Larmor window — rather than the full
Nyquist span.

**Auto X** and **Auto Y** stay active until you take manual control of the
view: typing a limit value turns off that axis's auto-scaling, and a **Zoom**
or **Pan** gesture turns off both so the framing you dragged to is kept
instead of snapping back to the data extent. Re-enable either at any time by
clicking its button again.

Default limits automatically adjust to fit the data including error bars,
with 5% padding.

Once you choose a window — by typing a limit, or by panning or zooming — that
window is held: recomputing a spectrum, browsing onto a run with no spectrum,
and switching runs all keep it, so you can compare the same window across a run
series. Toggling **Auto X** or **Auto Y** on is the explicit "follow the data"
escape hatch: it releases the held window for that axis and re-scales on every
redraw until you toggle it off.

Dense-data display (decimation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Very dense traces (e.g. high-resolution ROOT histograms or many-point
spectra) are *display-decimated*: only a bounded number of points is
rendered so panning, zooming, and switching runs stay responsive. A small
corner chip — for example ``4.0k of 1.2M pts`` — appears whenever the
current view is decimated, and disappears once you zoom in far enough that
every visible point is drawn.

Decimation affects the display only. Fits, transforms (FFT, MaxEnt,
moments) and exports always use the full-resolution data.

Time-domain scatter is decimated by uniform sampling, which is an unbiased
visual sample of noisy data. Frequency-domain spectra instead keep the
minimum and maximum of each display bucket, so a narrow spectral peak can
never vanish from the screen.

Legend label field
~~~~~~~~~~~~~~~~~~

Use the **Label** dropdown (left side of toolbar row 2) to choose how each selected run
is labelled in the legend:

* **Run** (default)
* **Field (G)**
* **Temperature (K)**
* **Comment**

The same **Label** dropdown is available on the frequency (FFT / MaxEnt)
toolbar and labels the overlaid spectra there, including custom columns such
as an **Angle (°)** field. The time-domain and frequency views keep their own
independent **Label** selection.

If the selected metadata value is unavailable for a run, Asymmetry
automatically falls back to the run label.

**Overlay** (to the right of **Label**) controls whether multi-selection and
group selections are drawn together on one set of axes, or reduced to one
dataset view as described above in **Multi-Selection**.

For single-dataset views, the main plot also shows the active grouping
``alpha`` value above the canvas when one is available. The overlay is hidden
for multi-dataset comparisons.

.. _waterfall-stacking:

Waterfall stacking
~~~~~~~~~~~~~~~~~~

.. image:: /_generated/screenshots/waterfall_overlay.png
   :alt: Waterfall-stacked time-domain overlay of an Ag LF decoupling series
   :width: 100%

*The same five-field Ag LF Kubo–Toyabe decoupling series as the*
:doc:`LF decoupling walkthrough </workflows/lf_decoupling_dynamics>`, *now with*
**Waterfall** *enabled: each field's trace is shifted onto its own baseline*
*instead of sharing one axis with the rest.*

An overlay of several runs is easiest to compare when every trace shares one
baseline, but that is exactly what makes closely-spaced curves hard to read
apart. **Waterfall** (next to **Overlay**, both on both the time-domain and
frequency-domain plot panels) resolves this by shifting each trace vertically
by a uniform per-trace offset :math:`i \cdot \Delta`, so an *n*-run overlay
reads as *n* cleanly separated curves stacked bottom to top in selection
order.

Waterfall only makes sense once there is more than one trace to separate, so
its checkbox tracks **Overlay**: it stays disabled and unchecked until Overlay
is on, and switching Overlay off unchecks it again. Stacked-subplot views —
Individual Groups and other multi-axis layouts — are unaffected; waterfall
only ever modifies the single-axis overlay.

The spacing Δ is automatic by default: 1.4× the median robust span (98th minus
2nd percentile of the finite samples, which ignores a handful of
saturation/outlier bins) across the traces about to be drawn, measured over
the displayed x-range rather than the full arrays — an FFT magnitude spectrum
keeps a long near-zero tail beyond the framed peak region that would otherwise
shrink Δ to a fraction of the visible spans. Neighbouring curves therefore
clear each other with a little breathing room without any tuning; zooming
afterwards does not re-space an already-drawn stack.
Type a value into the field beside the checkbox to fix Δ manually instead —
its placeholder reads ``Auto``, and clearing the field (leaving it blank)
returns to automatic spacing.

In the time domain, each stacked trace also gets a faint horizontal hairline
at its own shifted zero, so a curve's depolarisation or oscillation reads
against its own reference rather than the axis origin. Frequency-domain
waterfalls stack the same way but skip the hairline, since a spectrum already
sits on its own zero baseline.

**Export to GLE** and the plain-text data/fit exports both reproduce the
on-screen stack, but differ in *where* the offset lives. With ``gleplot`` >= 1.7
the GLE export applies each per-trace offset in the GLE script itself (a GLE
``let`` that shifts the trace at plot time), so the exported ``.dat``/``.fit``
sidecars keep their raw, unshifted values — the stack is a property of the
figure, not of the data, and the gleplot figure editor can retune it. Against
older ``gleplot`` (and always for the plain-text export, which has no script to
carry the offset) the offset is baked into the written column instead. Either
way each file records a ``waterfall offset:`` header line documenting the
applied shift.

The waterfall setting — on/off and the manual Δ (or automatic, if left
blank) — is saved per plot panel in the project file and restored on reopen.

Run Info and metadata columns
-----------------------------

Use **Get Info** (context menu on a run) to open the Run Info window.

Primary Run Info table
~~~~~~~~~~~~~~~~~~~~~~

The primary table shows key run parameters in four columns:

* **Include in Data Browser** (checkbox)
* **Field**
* **Value**
* **Log Plot**

Ticking a checkbox adds that field as a dynamic column in the Data Browser.
Unticking removes it.

Known summary fields use friendly Data Browser headers rather than raw metadata
paths. For example, ``run_info.points`` appears as **Points**, and
``nexus_fields.sample.shape`` appears as **Orientation**.

For parameters backed by time-series NeXus logs, a **Plot** button appears in
the right column and opens the full log trace. NeXus sample-temperature logs
follow the same Data Browser rule as PSI and ROOT logs: the fixed
**Temperature (K)** column uses the scalar/header temperature by default, and
switches to the log mean while **Options → Use temperature from log** is
enabled. The menu option applies to every loaded run. The **Temperature (K)**
include checkbox in Get Info overrides only the run whose Get Info window is
open, so it can be used to make per-dataset exceptions after setting the global
option. Temperature values shown in red are using log averages.

PSI-BIN ``.mon`` temperature logs are exposed through the same time-series
path. When a matching sidecar is loaded, the summary **Temperature (K)** row
gets a **Plot** button and the individual channels are listed in Advanced as
``nexus_time_series.psi_temperature/Temp_<channel>.mean`` rows.

MusrRoot ROOT slow-control histograms are handled in the same way. If the file
contains ``histos/SCAnaModule/hSampleTemperature``, Get Info shows the average
temperature from that histogram, provides a **Plot** button, and lets the
**Temperature (K)** checkbox replace the Data Browser's default header
temperature with the log mean for that run only. FLAME ROOT files can store the
same information under sensor names such as ``SAM_ts_value``; when the header
temperature references that sensor, the **Temperature (K)** row uses it for
the log plot.

Advanced subwindow
~~~~~~~~~~~~~~~~~~

Click **Advanced** to open a separate, scrollable subwindow containing a full
metadata table with the same include-checkbox and log-plot behaviour.

This allows promoting any advanced NeXus field into the Data Browser without
leaving the Run Info workflow.

For PSI-BIN temperature sidecars, Advanced also lists ``psi_temperature_log``
provenance fields, including the source ``.mon`` path and the
``Mantid LoadPSIMuonBin-compatible`` reader provenance.

For MusrRoot slow-control logs, Advanced lists ``musrroot_slow_control_log``
provenance fields and ``nexus_time_series.musrroot_slow_control/...`` summary
rows.

The Advanced window now includes a search box, so you can filter the metadata
table by field name or rendered value before adding a column or opening a log
trace.

Grouping and bunching
~~~~~~~~~~~~~~~~~~~~~

Grouping configuration is opened from **Grouping** (toolbar/menu). Supported
raw data runs use the full **Grouping** dialog, seeded with detector labels
and per-detector ``t0`` values from the file when those metadata are available.

This keeps fit inputs consistent with the run grouping settings and project
state.

When you open the Grouping dialog from a run selection, the currently selected
datasets are pre-ticked in the dialog. When loading new runs into an existing
project, Asymmetry reuses the grouping payload from the highest run number
currently present in the Data Browser when that payload is well-defined.

Full Grouping dialog
~~~~~~~~~~~~~~~~~~~~

The full Grouping dialog supports editing grouping assignment, alpha,
correction toggles, good-bin limits, and bunching. Deadtime correction is
enabled only when the source file provides deadtime constants; background
correction is enabled for PSI BIN/MDU and PSI/LEM ROOT data.

Changing bunching refreshes both the displayed curve and the dataset passed to
the fitting panel.

Detector Layout editor
~~~~~~~~~~~~~~~~~~~~~~

The full Grouping dialog includes a **Detector Layout...** button that opens a
visual grouping editor for ISIS instruments **HiFi**, **MuSR**, **EMU**, and the
PSI **FLAME** and **HAL-9500** spectrometers, for data loaded from NeXus, BIN,
MDU, or ROOT files.

The editor has three panels:

* **Detector schematic** (left): click detector sectors to toggle membership in
   the currently active group. FLAME detectors are shown as rectangular plates
   in a top-view diagram; detector plates that are not in any group are left
   unfilled.
* **Group slots** (centre): eight group buttons (**Group 1** to **Group 8**),
   each with an editable name field.
* **Presets** (right): instrument selector plus a preset dropdown with
   **Apply Grouping**.

Key behaviour:

* A detector can belong to multiple groups at the same time.
* Instrument is identified using the following priority when opening the editor:

  1. **Saved instrument** — if a previous session already set the instrument for
     this dataset, that choice is restored directly.
  2. **Run metadata** — fields such as ``instrument``, ``instrument_name``,
     ``beamline``, or ``spectrometer`` in the file's embedded metadata. PSI
     FLAME BIN/ROOT files are recognised when metadata identify ``FLAME``,
     including values such as ``LMU_BULKMUSR_FLAME``.
  3. **Source filename** — if the filename contains a recognisable instrument
     token (e.g. ``emu``, ``hifi``, ``musr``, ``flame``).
  4. **Histogram count** — 64 histograms → HiFi; 96 histograms → EMU;
     other counts → no automatic selection.

  The instrument can always be overridden from the instrument dropdown, and the
  chosen instrument is saved with the dataset so it survives project reloads.
* Group names are saved in project grouping payloads.

Preset highlights:

* **HiFi** and **MuSR** include longitudinal/transverse-style presets.
* **EMU** includes a **Vector Polarization** preset with six groups:
   ``Pz Forward``, ``Pz Backward``, ``Py Top``, ``Py Bottom``, ``Px Left``, and
   ``Px Right``.
* **FLAME** includes **Longitudinal** (Forward / Backward) and **Transverse**
   presets. The transverse preset groups ``Right``, ``R_B``, and ``R_F``
   together, and ``Left``, ``L_B``, and ``L_F`` together.
* PSI detector-layout names follow the PSI beam-direction convention. The main
   Grouping dialog uses spin-direction forward/backward for asymmetry
   calculation, so longitudinal PSI/FLAME analysis defaults are swapped:
   **Forward Group** = ``Group 2: Backward`` and **Backward Group** =
   ``Group 1: Forward``. ISIS datasets are unchanged.

.. note::

   Vector-polarisation mode is not exclusive to EMU. Any dataset whose six
   group names follow the canonical pattern
   (``Pz Forward`` / ``Pz Backward``, ``Py Top`` / ``Py Bottom``,
   ``Px Left`` / ``Px Right``) will activate vector-mode features in the main
   plot regardless of the instrument field.

When vector grouping is active, the main plot header shows a
**Polarization** dropdown near the alpha display. You can select:

* **x** — display the :math:`P_x` pair (Left / Right detectors).
* **y** — display the :math:`P_y` pair (Top / Bottom detectors).
* **z** — display the :math:`P_z` pair (Forward / Backward detectors).
* **All** — display all three polarisation components as stacked subplots
   sharing the same time axis.  Each subplot carries its own Y-axis label and
   colour-matched error bars.

When vector grouping is active, the Grouping dialog replaces the single alpha
control with per-axis alpha controls:

* ``alpha_x`` for ``P_x``
* ``alpha_y`` for ``P_y``
* ``alpha_z`` for ``P_z``

Each axis has its own **Estimate alpha** button, and an **Estimate All alpha**
button is available to update all three axis values in one step.

All of these are available without reopening the Grouping dialog.

The main-plot Y limits are remembered separately for each polarisation axis,
so changing between axes restores your previous Y-range for that axis.
In **All** mode the Y-axis controls are read-only; adjust the limits for each
component individually by switching to that axis first.

.. note::

    When a project is saved with **All** mode active, all three subplots are
    rendered immediately on the next project load — no manual axis switch is
    required.

.. note::

   The alpha value shown in the main plot header is axis-specific in ``x``,
   ``y``, and ``z`` modes, and hidden in **All** mode.

In the Detector Layout Editor, the preset panel shows a status line under the
preset dropdown, for example ``(Current: Vector Polarization)``. This remains
until the grouping is edited away from the applied preset, then changes to
``(Current: Custom)``.

Two-period RG mode
~~~~~~~~~~~~~~~~~~

When the reference run contains two periods in the full Grouping dialog, an
**RG Mode** row with WiMDA-style radio buttons:

* **Red**
* **Green**
* **G minus R**
* **G plus R**

The selected mode is applied during grouping recomputation:

* **Red** uses period-1 histograms.
* **Green** uses period-2 histograms.
* **G minus R** computes asymmetry for Green and Red separately, then subtracts
   them in asymmetry space: :math:`A_{G-R} = A_G - A_R`.
* **G plus R** computes asymmetry for Green and Red separately, then adds them
   in asymmetry space: :math:`A_{G+R} = A_G + A_R`.

For :math:`G \pm R`, the uncertainty is propagated from the two period
asymmetry uncertainties using quadrature:

.. math::

    \sigma_{G\pm R} = \sqrt{\sigma_G^2 + \sigma_R^2}

Colour behaviour in the main plot:

* Single-run views use the RG mode colour directly.
* Multi-run overlays keep the first trace in the selected RG mode colour and
   use contrasting colours for additional selected runs so traces remain
   distinguishable.

The selected RG mode is saved in the grouping payload as ``period_mode``.

Low-count tail display
~~~~~~~~~~~~~~~~~~~~~~

At late times, grouped counts can become very small. Asymmetry bins with a
non-positive denominator :math:`F + \alpha B` are treated as undefined and are
not drawn in the main plot.

Bins with very small but positive denominator may still show large
point-to-point variation (including values near :math:`\pm 100\%`), which is
expected for low-statistics tails.

Main plot labels and export
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main plot now supports interactive labels and direct GLE export for the
currently displayed dataset or dataset overlay.

For full detector schematics and vector grouping reference material, see
:doc:`detector_grouping` and :doc:`vector_polarization`.

**Interactive labels**

* Click **Add Annotation**, then click on the plot to place text at that data
   coordinate.
* Drag a label with the mouse to reposition it.
* Double-click a label to edit its text.
* Right-click a label to delete it.

**Export Plot(s) to GLE**

Main-plot export is now driven directly from the plot toolbar, using:

* **Export Plot(s) to GLE** button
* **Format** dropdown (PDF or EPS)

The export writes a self-contained ``<name>.gleplot`` folder by default. That
folder contains the ``.gle`` script plus the per-dataset sidecar files for the
currently displayed datasets. This works for:

* Single-dataset views
* Multi-selection overlays (all selected datasets currently plotted)

What is exported:

* Data as error bars (no connecting lines)
* Fit curves as lines when a fit overlay is available
* User plot labels/annotations (same text and coordinates)

Output files and naming inside ``<name>.gleplot``:

* ``<name>.gle`` for the exported script
* ``<label>.dat`` for each exported dataset (time, asymmetry, error)
* ``<label>.fit`` for each exported fit curve when present (time, fitted asymmetry)
* file stems are derived from the selected **Label** field value and sanitised
   for safe filenames

The ``.dat`` sidecar now includes structured comment headers describing the run
and grouping state used to produce the plotted asymmetry, including run number,
title/comment timestamps when available, histogram/binning information,
forward/backward grouping, alpha, good-bin limits, bunching, and deadtime
correction state.

The ``.fit`` header includes run metadata and fit metadata, including:

* fit function description
* :math:`\chi^2`
* reduced :math:`\chi^2`
* fitted parameter values and uncertainties (when available)

Compilation behaviour:

* If ``gle`` is available, Asymmetry compiles the ``.gle`` inside the export
  folder to the selected format from the **Format** dropdown.
* If ``gle`` is not available, Asymmetry still saves the ``.gle`` and sidecar
  files so you can compile later.
* If ``gleplot`` regenerates matching ``.dat`` files while saving, Asymmetry
  rewrites the metadata-rich sidecars afterwards so those headers are retained.

After the export is written, Asymmetry opens the exported ``.gle`` script
directly in the gleplot figure editor (an in-app window, styled to match the
rest of Asymmetry) so you can tweak the script and re-render without leaving
the app. The editor opens even when no GLE binary is installed — you can still
edit the script, and the editor reports "GLE: not found" in its status bar.
Its live preview compiles with the same GLE binary configured under
**Setup ▸ GLE Setup…**. This requires ``gleplot`` >= 1.6; against older
``gleplot`` installs Asymmetry falls back to the previous behaviour, a
read-only static preview dialog after successful compiles. The same editor is
also reachable directly from **Analysis ▸ GLE Figure Editor…**, which opens a
blank editor window at any time.

Fitting panel
-------------

.. image:: /_generated/screenshots/fit_wizard_gkt.png
   :alt: Single-dataset fit with a converged Gaussian Kubo–Toyabe model on Ag
   :width: 100%

*Converged static Gaussian Kubo–Toyabe fit on a synthetic ZF Ag polycrystal*
*dataset (Δ ≈ 0.39 μs⁻¹). Ag is the canonical nuclear-dipolar reference*
*sample at every μSR facility; the GKT function (Kubo & Toyabe, 1966) is*
*derived in Blundell et al. Ch 5.2.*

The fitting panel provides an interactive interface for fitting models to your data.
It is located in the right dock area next to the plot panel.

The fit function is shown as :math:`A(t)` and is built from components using
the **Edit Function...** button.

**Single dataset fitting**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "Single" tab fits one composite model to the currently selected dataset.
By default, the function is:

.. math::

   A(t) = A_1 e^{-\lambda t} + A_{\mathrm{bg}}

where ``A_bg`` is an explicit constant background term.

To edit the model:

1. Click **Edit Function...**
2. Add/remove components and choose operators (``+``, ``-``, ``*``, ``/``)
3. Confirm with **OK**

Available components in the builder:

* **Exponential**: :math:`A e^{-\lambda t}`
* **Gaussian**: :math:`A e^{-(\sigma t)^2}`
* **Oscillatory**: :math:`A\cos(2\pi f t + \phi)` — frequency :math:`f` in MHz
* **OscillatoryField**: :math:`A\cos(2\pi \gamma_\mu B t + \phi)` — field :math:`B` in Gauss;
  frequency is derived automatically from :math:`\gamma_\mu = 13.554\,\text{MHz/kG}`
* **StretchedExponential**: :math:`A e^{-(|\lambda|t)^\beta}`
* **StaticGKT_ZF**: Static Gaussian Kubo-Toyabe
* **LongitudinalFieldKT**: Hayano LF-KT; :math:`B_L` initialised from run field. See :ref:`fit-lf-kubo-toyabe`.
* **Nuclear dipolar / MuF**: Analytical single-``mu-F`` polarisation
* **Nuclear dipolar / FmuF_Linear**: Analytical collinear ``F-mu-F`` polarisation
* **Nuclear dipolar / FmuF_General**: Numerical powder-averaged ``F-mu-F`` polarisation
* **Constant**: :math:`A_{\mathrm{bg}}`

.. note::

   The oscillatory component in the builder is **pure cosine** by default.
   If you want damping, multiply it by an exponential component.

Parameter naming rules in the table:

* Additive terms get their own amplitudes: ``A_1``, ``A_2``, ...
* Multiplicative or divisive component chains share one amplitude parameter,
   so ``Exponential * Gaussian`` uses ``A_1`` rather than ``A_1`` and ``A_2``
* ``A_bg`` is present by default from the constant component
* Other symbols (for example ``Lambda``, ``sigma``, ``frequency``) are only
  indexed when duplicates exist in the same expression
  (for example ``Lambda_1``, ``Lambda_2``)

Fitting workflow:

1. **Select a dataset** in the data browser (it will be plotted)
2. **Adjust parameters** in the table:
   
   * **Value**: Initial guess for the parameter
   * **Fix**: Check to hold the parameter constant during fit
   * **Min/Max**: Set bounds (leave empty for no bounds)

3. **Click "Fit"** to execute the fitting

   The fit uses the dataset currently shown in the plot panel. Grouping and
   bunching are configured from the Grouping dialog and applied before fitting.
   Fits run asynchronously, and the dialog shows a "fit in progress" message
   while fit controls are temporarily disabled.

4. **Review results:**
   
   * Fit curve appears on the plot in red
   * Results box shows χ², χ²ᵣ (reduced chi-squared)
   * Best-fit parameter values with uncertainties
   * Log panel shows summary message

**Tips for Good Fits:**

* Start with reasonable initial parameter values
* Use bounds to prevent unphysical values (e.g., negative amplitudes)
* Fix parameters you know precisely
* Look for χ²ᵣ ≈ 1 (good fit); >> 1 (poor fit); << 1 (overestimated errors)
* If fit fails to converge, try different initial values or tighter bounds

**Carrying a model forward between runs**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selecting a different run in the Data Browser does not always blank the
Single tab's form. A run that already carries a **recorded fit result** —
its own single fit, or its role as a member of a batch/global fit — is
*protected*: selecting it always restores exactly that fitted state, so it is
never silently overwritten. Every other run is *refreshable*: selecting it
loads the composite model and parameter setup of the most recently fitted
function in the session (superseding anything it was showing before,
including a hand-edited form you never fitted — the protection trigger is
"did you commit by fitting", not "did you touch the form"), with
field-dependent parameters (a frequency seeded from the fitted peak, ``B_L``
from the run's applied field, and similar) reseeded for the newly-selected
run. A results box below the parameter table reads "Model carried from run
*N* — not fitted for this run" while a refreshed or carried form is showing,
so it is never mistaken for an actual fit of the displayed run. Only when
nothing has been fitted anywhere in the session yet does the form fall back
to carrying forward whatever was last *displayed*.

.. _batch-tab-groups:

**Batch fitting**
~~~~~~~~~~~~~~~~~

The **Batch** tab fits multiple datasets simultaneously with shared and
per-dataset parameters:

1. **Select multiple datasets** in the data browser (Ctrl+Click or Shift+Click)
2. Switch to the **Batch** tab in the fit panel
3. Optionally click **Edit Function...** to customise the composite :math:`A(t)`
4. **Set parameter type** in the table for each parameter:

   * **Global**: shared value across all selected datasets
   * **Local**: separate value per dataset
   * **Fixed**: held constant at the specified value
   * **File**: fixed at the per-dataset value read from the run metadata
     (for example ``B_L`` is set to the applied field in Gauss from the loaded
     file). Behaves like **Fixed** for the fit itself, but the value
     differs automatically for each selected dataset.

5. **Click "Run Batch Fit"**

   Batch/global fitting follows the same rule: current grouped/bunched
   dataset settings are applied to each selected dataset before fitting.

After a batch or global fit completes:

* Fit curves appear on the plot for all datasets
* The **Global Parameter Fit** window opens automatically when the fit has at
  least one **Global** parameter (see below)
* The log panel shows a summary with average χ²ᵣ
* The results are recorded as a group-bound fit series (see `Data groups`_
  above and :ref:`group-bound-series-staleness` in :doc:`parameter_trending`)

Fitting a group directly
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: /_generated/screenshots/batch_tab_group_binding.png
   :alt: Batch tab bound to a data group with one member unticked in the Batch members list
   :width: 100%

*The Batch tab bound to the "T scan — EuO" group via* **Fit this group…***,*
*showing the group-binding banner and the* **Batch members** *checklist with*
*the 69 K run unticked — excluded from this analysis without leaving the*
*group.*

Choosing **Fit this group…** from a data group's context menu (rather than
just selecting its runs) binds the Batch tab to that group: a **Fitting
group: <name>** banner appears above a new **Batch members** section listing
every member run with a checkbox. Untick a run to exclude it from *this*
analysis without removing it from the group — the exclusion is recorded on
the fit series, not the group, so the same group can still be fit a second
time with a different model over its full membership. An ordinary run
selection (rather than **Fit this group…**) clears any existing binding, so
the next batch fit auto-creates its own group instead of extending the
previous one.

**Grouped time-domain fitting**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asymmetry also supports WiMDA-style grouped count fitting inside one selected
run.

Use this mode when you want one shared physical function across several group
traces from the same dataset rather than one simultaneous fit across several
different runs.

The main plot workspace now has three top-level tabs:

* **FB Asymmetry**
* **Individual Groups**
* **Frequency Domain**

To launch grouped fitting:

* switch the central workspace to **Individual Groups**
* click **Fit** to replace the normal fit dock with the **Multi-Group Fit** controls

In that workflow:

* the plot switches to stacked grouped time-domain subplots for the active dataset
* the grouping definitions come from the current **Grouping** dialog payload
* the fit UI splits into **Per-Group Parameters** and **Fit-Function Parameters** blocks
* grouped runs are fitted in count space using lifetime-corrected grouped traces
* switching the central workspace back to **FB Asymmetry** restores the regular fit dock

See :doc:`grouped_time_domain_fitting` for the detailed workflow and current
limitations of this first implementation slice.

Global parameter fit window
---------------------------

.. image:: /_generated/screenshots/global_fit_lfkt.png
   :alt: Global fit setup for an Ag LF Kubo–Toyabe field-decoupling series
   :width: 100%

*Global-fit setup on a synthetic Ag LF-KT decoupling series, B_L = 0, 15,*
*50, and 100 G with shared Δ ≈ 0.39 μs⁻¹. The longitudinal field decouples*
*the nuclear dipolar broadening as γ_μB_L/Δ grows past unity (Hayano et al.,*
*Phys. Rev. B 20, 850 (1979); cf. Blundell et al. 2022, Fig. 5.6).*

The global-fit result window is split into two synchronised views:

* **Left: fit overlays per selected dataset**
* **Right: local-parameter trends vs the complementary sweep axis**

Left pane controls
~~~~~~~~~~~~~~~~~~

* **Show components**: stack additive fit components under the total fit curve
* **Log X / Log Y**: toggle axis scaling for dataset fit overlays
* **Share X Axis**: render one aligned x-axis across all group subplots
* **Add Label**: place draggable text labels directly on the plot

Right pane controls
~~~~~~~~~~~~~~~~~~~

* **Single Axes / Subplots**: choose one combined axis or one axis per local parameter
* **Log X**: log scaling for the local-parameter x-axis
* **Per-parameter Log Y**: enable from the selector table for individual local parameters
* **Add Label**: place draggable labels on local-parameter plots

Plot labels and annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Left-click and drag to reposition labels
* Double-click an existing label to edit text
* Right-click a label to remove it
* Automatic group labels can be hidden per subplot by right-clicking them

All label positions and text are saved in the project file and restored when the
project is reopened.

Fit Parameters panel
--------------------

After a global fit, the Fit Parameters panel shows how the varying
(per-dataset) parameters change across runs. It provides:

Parameter table
~~~~~~~~~~~~~~~

A tabular view of all fitted varying parameters, with columns for:

* Run number
* :math:`B` (G) — applied magnetic field
* :math:`T` (K) — sample temperature
* Each varying parameter with its uncertainty

Parameter trend plot
~~~~~~~~~~~~~~~~~~~~

A plot of one selected parameter versus a sweep variable. Controls include:

* **X axis**: Choose between Auto (inferred from data), :math:`B` (G), :math:`T` (K), or Run number.
  Auto mode detects whether field or temperature varies across the datasets.
* **Y parameter**: Select which varying parameter to plot
* **Scale**: Check **Log X** and/or **Log Y** to use logarithmic axes —
  useful for power-law behaviour or data spanning several orders of magnitude
* **Model components**: Enable **Show components** to stack additive parameter-model
   components under fitted overlays (for visual decomposition)
* **Plot labels**: Use **Add Label** and **Clear Labels** to annotate exported and
   on-screen parameter-trend plots

When **Show components** is enabled, y-axes are forced to linear scale with a
zero baseline so stacked component areas remain physically interpretable.

Exporting data
~~~~~~~~~~~~~~

**Export TSV**: Save the parameter table to a tab-separated (``.tsv``) file for
use in spreadsheets or other analysis software. A comment header records the
fitted model and the shared global-parameter values, and per-run reduced χ² and
χ² columns are appended after the parameter columns.

**Export to GLE**: Generate a publication-quality figure using the
`GLE Graphics Layout Engine <http://glx.sourceforge.io/>`_ via the
``gleplot`` Python library:

1. Click **"Export to GLE"**
2. Choose a name for the ``.gleplot`` export folder
3. Select the output format from the **Format** dropdown (PDF or EPS)
4. The export creates the named ``.gleplot`` folder and writes:

   * A ``<name>.gle`` script for the selected plot
   * A ``.dat`` data file with column headers and the globally shared
     parameters recorded as comments at the top of the file
   * Any optional ``.fit`` sidecars needed for active model overlays

5. If GLE is installed on your system, the script is compiled automatically
   to PDF or EPS and the exported ``.gle`` script opens in the gleplot figure
   editor (falling back to a read-only static preview dialog against
   ``gleplot`` < 1.6)
6. If GLE is not installed, the script and data files are still saved — and
   still open in the figure editor for editing — you can compile them later
   with ``gle -d pdf <name>.gle`` from inside the ``.gleplot`` folder

.. note::

   Install GLE from source or from http://glx.sourceforge.io/ to enable
   automatic compilation. The ``gleplot`` Python package is required for
   script generation (``pip install gleplot``).

Global-fit GLE exports
~~~~~~~~~~~~~~~~~~~~~~

The Global Parameter Fit window provides two dedicated GLE export buttons:

* **Export fits to GLE** (left pane): exports the per-dataset fit overlays
* **Export plot(s) to GLE** (right pane): exports local-parameter trend plots

Both buttons open the exported ``.gle`` script in the gleplot figure editor
after the export is written, the same as the main-plot and fit-parameters GLE
exports above.

For local-parameter exports, Asymmetry also writes:

* A ``*_local_parameters.dat`` table with units and an explicit column map
* Optional ``*_local_<parameter>.fit`` files for any active local model fit overlays

Both global-fit export buttons use the same foldered layout, so the exported
``.gle`` file, compiled output, and sidecars stay together in one bundle.

Both global and local export dialogs remember the last export directory and use
it as the default location next time.

Analysis workflows
------------------

Basic analysis
~~~~~~~~~~~~~~

1. Load data files
2. Sort by field or temperature
3. Select and view different runs
4. Adjust plot limits for detailed inspection
5. Apply bunching to reduce noise if needed

Comparing runs
~~~~~~~~~~~~~~

1. Filter by temperature or field
2. Click through filtered runs to compare
3. Use co-adding to average similar runs

Temperature series
~~~~~~~~~~~~~~~~~~

1. Sort by temperature column
2. Step through runs in order
3. Filter by field to isolate specific conditions

Field series
~~~~~~~~~~~~

1. Sort by field column  
2. Step through runs in order
3. Filter by temperature for isothermal scans

UI scale
--------

The interface density can be adjusted from **View → UI Scale**. Five preset
scales are available: 80 %, 90 %, 100 %, 110 %, and 120 %. The selection is
persisted between sessions. The default is 90 %.

The scale is *font-driven*: it multiplies the application font (and the derived
chrome font sizes) rather than resizing whole widgets. Everything sized from the
font metrics then follows automatically, so a scale change adjusts:

* base font size across panels and dialogs;
* table row heights and character-based column widths;
* dock minimum widths, keeping the fit table's columns readable at every scale.

Keyboard shortcuts
------------------

* **Ctrl+Shift+O**: Open data file(s)
* **Ctrl+N**: New project
* **Ctrl+O**: Open project
* **Ctrl+S**: Save project
* **Ctrl+Shift+S**: Save project as
* **Ctrl+Return**: Run the fit
* **Ctrl+W** / **Ctrl+Q**: Close the window (quit)

Tips and tricks
---------------

* Double-click column borders to auto-resize
* Use "All" filter with partial text to find specific runs
* Co-add replicate measurements to improve statistics
* Bunch data before Fourier analysis to reduce computation time
* Right-click on plot to save figure (via matplotlib toolbar)

Saving and reopening projects
------------------------------

A *project file* (``.asymp``) saves the complete state of your analysis
session.  You can close Asymmetry and resume exactly where you left off, or
maintain several independent analyses side-by-side.

Creating and saving
~~~~~~~~~~~~~~~~~~~~

* **File → New Project** (``Ctrl+N``) clears the current session.
* **File → Save Project** (``Ctrl+S``) saves to the current project file.
   If no project file is open yet, you will be prompted to choose a location.
* **File → Save Project As…** (``Ctrl+Shift+S``) always asks for a new filename.

.. _unsaved-changes-guard:

Unsaved-changes guard
~~~~~~~~~~~~~~~~~~~~~~

Hours of grouping, fitting, and trend work live in memory until you save, so
Asymmetry tracks whether the session holds unsaved work and refuses to drop it
silently. A ``*`` in the window title marks a modified session, and any action
that would clear it — closing the window, **New Project**, **Open Project**, or
opening a recent project — first raises a **Save / Discard / Cancel** prompt.
Choosing **Cancel** aborts the action and keeps your work; **Save** writes the
project (choosing a location if none is set yet) before proceeding. The flag is
set by every mutating action — data load and removal, grouping edits, each fit
completion, trend-series rename or delete, ALC scan builds, and custom-column
edits — and cleared on a successful save, open, or new project. Opening a
project or starting a fresh one therefore always begins from a clean state.

Opening a project
~~~~~~~~~~~~~~~~~~

* **File → Open Project…** opens a file dialog so you can locate an
   ``.asymp`` file.
* **File → Recent Projects** lists up to the 10 most recently opened projects
   for one-click access.

On open, every source data file referenced by the project is reloaded
from disk.  Asymmetry tries paths in this order:

1. The absolute path stored in the project file
2. A path relative to the ``.asymp`` file itself (useful when the whole
   analysis folder is moved together)
3. If files are still missing, a dialog offers to **locate a directory** —
   choose the folder where the data files now live, and Asymmetry will match
   each missing file by filename

This means you can move an entire analysis folder, or just redirect to a
new data location, without losing your session state.

If a source file cannot be found even after the search, Asymmetry logs a
warning and skips that dataset — the rest of the session is restored normally.

What is saved
~~~~~~~~~~~~~

* All loaded datasets (stored as file paths, not raw arrays)
* Co-added ("combined") dataset groups
* Data Browser sort column, sort order, column filters, selected rows, and
   dynamic metadata columns (including Run Info selections)
* Plot axis limits (X/Y), fit range, current run, overlay mode, and the
   waterfall stacking setting (on/off and manual Δ, or automatic)
* Grouping settings per run (groups, forward/backward selection, alpha,
   good-bin range, bunching, deadtime toggle)
* Most-recently-displayed fit overlay curve(s)
* Main plot labels/annotations
* Single-fit and global-fit model selection, parameter values, fixed/free
   flags, and bounds
* Fit results text (χ², χ²ᵣ, best-fit values with uncertainties)
* Active fit panel tab (Single or Global)
* Fitted Parameters panel rows, axis settings, plot mode, component-display
   toggle, and plot labels
* Fourier panel state, including apodisation settings, phase mode, per-run
  group phase tables, included groups, and phase-estimation settings

What is **not** saved
~~~~~~~~~~~~~~~~~~~~~

* Raw data arrays (always reloaded from the original files)
* Fourier transform output arrays
* Log panel messages

Schema versioning
~~~~~~~~~~~~~~~~~

The ``.asymp`` format uses an integer *schema version* that is independent of
the Asymmetry package version.  Project files from older releases will be
migrated automatically when opened in a newer release.  If a project file
requires a schema version that the installed Asymmetry version does not
understand, an error dialog is shown and the file is not loaded.
