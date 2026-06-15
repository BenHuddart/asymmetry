GUI Usage
=========

This chapter is the comprehensive reference for the Asymmetry GUI — menus,
panels, dialogs, keyboard shortcuts. New users coming from a real
experiment should normally read :doc:`workflows/index` first and dip into
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

Main Window Layout
------------------

.. image:: /_generated/screenshots/main_window.png
   :alt: Asymmetry main window with an EuO temperature scan loaded
   :width: 100%

*Synthetic EuO ferromagnet temperature scan crossing the Curie point at*
*Tc=69 K — six zero-field runs from 30 K up to 90 K (cf. Blundell et al.*
*Muon Spectroscopy, OUP 2022, Fig 6.6). The selected run at 65 K is just*
*inside the ordered phase where the spontaneous-field precession is at its*
*slowest and the critical damping is largest.*

The main window has four main areas:

1. **Data Browser** (left): Table of loaded datasets
2. **Plot Panel** (center): Interactive plot display
3. **Analysis Panels** (right): Fit and Fourier controls
4. **Log Panel** (bottom): Status messages and command history

Loading Data
------------

Multiple File Selection
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
exports to CSV: with the option on, a temperature trend uses each run's logged
temperature rather than its parked setpoint — see
:ref:`trend-abscissa-coordinate`.

Data Browser Features
---------------------

Sorting
~~~~~~~

Click any column header to sort by that field:

* **Left-click**: Toggle between ascending and descending order
* Numeric columns (Run, T(K), B(G)) sort numerically
* Text columns (Title) sort alphabetically

The sorting is robust and works correctly even after filtering.

**Excel-Style Column Filtering**
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

Multi-Selection
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

Viewing Data
~~~~~~~~~~~~

Click any dataset row to plot it in the main panel.

Context Menu
~~~~~~~~~~~~

Right-click any dataset row to access options:

* **Co-add Selected** (appears when 2+ datasets selected): Combine selected datasets into one averaged dataset
* **Separate Combined** (appears when a combined dataset is selected): Break apart a combined dataset back into its source datasets
* **Remove Entry** or **Remove Selected Entries**: Delete the selected dataset(s)

These options appear contextually based on your current selection.

Co-adding Datasets
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
* Can be plotted and analyzed like any other dataset

If the selected datasets do not share the same grouping (groups, alpha,
good-bin limits, bunching, deadtime settings, and related grouped-data
controls), co-add is blocked and the browser leaves all source rows unchanged.
Align grouping first, then retry the co-add.

Separating Combined Data
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

Deleting Datasets
~~~~~~~~~~~~~~~~~

To remove a dataset from the browser:

1. Select one or more datasets
2. Right-click and choose "Remove Entry" or "Remove Selected Entries"
3. Alternatively, press the **Delete** key

This removes the dataset(s) from the browser; they can be reloaded via **File → Open**.

Exporting the Logbook
~~~~~~~~~~~~~~~~~~~~~

Use the **Export logbook** toolbar button (immediately after **Open**) to write
the current Data Browser contents to a file.

Export behavior:

* Exports the currently active Data Browser columns, including dynamic columns
   added from **Run Info**.
* Preserves data-group organization by writing a section header for each group.
* Includes datasets hidden by filters and collapsed groups.
* Writes section headers and data rows with a consistent column count so header
   labels and values align cleanly.

Formats:

* **TSV (recommended, default)**: Fastest export path and best choice for large
   logbooks. Opens directly in spreadsheet tools with clear column alignment.
* **RTF**: Rich text output with italicized **T** and **B** header labels.
   Useful for report-style sharing, but slower than TSV for large tables.

Default filename:

* If the current project has a name (for example ``My_Project.asymp``), the
   default export filename is ``My_Project_logbook.tsv``.
* If no project name is available, the default is ``logbook.tsv``.

Plot Panel Controls
-------------------

The plot panel displays the selected dataset(s) with error bars.

Axis Limits
~~~~~~~~~~~

Control the plot range using the spinboxes at the top:

* **X min/max**: Time axis range (μs)
* **Y min/max**: Asymmetry axis range
* Press **Enter** after editing any limit value to apply immediately
* Click **Auto X** to auto-scale the X axis only
* Click **Auto Y** to auto-scale the Y axis only

Auto-Y uses points inside the currently selected X range and prefers reliable
foreground points (excluding undefined/low-confidence bins when available).

Default limits automatically adjust to fit the data including error bars, 
with 5% padding.

Dense-Data Display (Decimation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Very dense traces (e.g. high-resolution ROOT histograms or many-point
spectra) are *display-decimated*: only a bounded number of points is
rendered so panning, zooming and switching runs stay responsive. A small
corner chip — for example ``4.0k of 1.2M pts`` — appears whenever the
current view is decimated, and disappears once you zoom in far enough that
every visible point is drawn.

Decimation affects the display only. Fits, transforms (FFT, MaxEnt,
moments) and exports always use the full-resolution data.

Time-domain scatter is decimated by uniform sampling, which is an unbiased
visual sample of noisy data. Frequency-domain spectra instead keep the
minimum and maximum of each display bucket, so a narrow spectral peak can
never vanish from the screen.

Legend Label Field
~~~~~~~~~~~~~~~~~~

Use the **Label** dropdown (left side of toolbar row 2) to choose how each selected run
is labelled in the legend:

* **Run** (default)
* **Field (G)**
* **Temperature (K)**
* **Comment**

If the selected metadata value is unavailable for a run, Asymmetry
automatically falls back to the run label.

**Overlay** (to the right of **Label**) controls whether multi-selection and
group selections are drawn together on one set of axes, or reduced to one
dataset view as described above in **Multi-Selection**.

For single-dataset views, the main plot also shows the active grouping
``alpha`` value above the canvas when one is available. The overlay is hidden
for multi-dataset comparisons.

Run Info and Metadata Columns
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
metadata table with the same include-checkbox and log-plot behavior.

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

Grouping and Bunching
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

Detector Layout Editor
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

Key behavior:

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
* Group names are saved in project grouping payloads and in ``.grp`` files via
   ``group_name.N=...`` entries.

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

   Vector polarization mode is not exclusive to EMU.  Any dataset whose six
   group names follow the canonical pattern
   (``Pz Forward`` / ``Pz Backward``, ``Py Top`` / ``Py Bottom``,
   ``Px Left`` / ``Px Right``) will activate vector-mode features in the main
   plot regardless of the instrument field.

When vector grouping is active, the main plot header shows a
**Polarization** dropdown near the alpha display. You can select:

* **x** — display the :math:`P_x` pair (Left / Right detectors).
* **y** — display the :math:`P_y` pair (Top / Bottom detectors).
* **z** — display the :math:`P_z` pair (Forward / Backward detectors).
* **All** — display all three polarization components as stacked subplots
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

The main-plot Y limits are remembered separately for each polarization axis,
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

Two-Period RG Mode
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

Color behavior in the main plot:

* Single-run views use the RG mode color directly.
* Multi-run overlays keep the first trace in the selected RG mode color and
   use contrasting colors for additional selected runs so traces remain
   distinguishable.

The selected RG mode is saved in the grouping payload and in ``.grp`` files as
``period_mode``.

Low-count Tail Display
~~~~~~~~~~~~~~~~~~~~~~

At late times, grouped counts can become very small. Asymmetry bins with a
non-positive denominator :math:`F + \alpha B` are treated as undefined and are
not drawn in the main plot.

Bins with very small but positive denominator may still show large
point-to-point variation (including values near :math:`\pm 100\%`), which is
expected for low-statistics tails.

Main Plot Labels and Export
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
* file stems are derived from the selected **Label** field value and sanitized
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

Compilation behavior:

* If ``gle`` is available, Asymmetry compiles the ``.gle`` inside the export
  folder to the selected format from the **Format** dropdown.
* If ``gle`` is not available, Asymmetry still saves the ``.gle`` and sidecar
  files so you can compile later.
* If ``gleplot`` regenerates matching ``.dat`` files while saving, Asymmetry
  rewrites the metadata-rich sidecars afterwards so those headers are retained.

Fitting Panel
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

**Single Dataset Fitting**
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
* **Nuclear dipolar / MuF**: Analytical single-``mu-F`` polarization
* **Nuclear dipolar / FmuF_Linear**: Analytical collinear ``F-mu-F`` polarization
* **Nuclear dipolar / FmuF_General**: Numerical powder-averaged ``F-mu-F`` polarization
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

**Global Fitting**
~~~~~~~~~~~~~~~~~~

The "Global" tab fits multiple datasets simultaneously with shared and
per-dataset parameters:

1. **Select multiple datasets** in the data browser (Ctrl+Click or Shift+Click)
2. Switch to the **"Global"** tab in the fit panel
3. Optionally click **Edit Function...** to customize the composite :math:`A(t)`
4. **Set parameter type** in the table for each parameter:

   * **Global**: shared value across all selected datasets
   * **Local**: separate value per dataset
   * **Fixed**: held constant at the specified value
   * **File**: fixed at the per-dataset value read from the run metadata
     (for example ``B_L`` is set to the applied field in Gauss from the loaded
     file). Behaves like **Fixed** for the fit itself, but the value
     differs automatically for each selected dataset.

5. **Click "Run Global Fit"**

   Global fitting follows the same rule: current grouped/bunched dataset
   settings are applied to each selected dataset before fitting.

After a global fit completes:

* Fit curves appear on the plot for all datasets
* The **Global Parameter Fit** window opens automatically (see below)
* The log panel shows a summary with average χ²ᵣ

**Grouped Time-Domain Fitting**
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

Global Parameter Fit Window
---------------------------

.. image:: /_generated/screenshots/global_fit_lfkt.png
   :alt: Global fit setup for an Ag LF Kubo–Toyabe field-decoupling series
   :width: 100%

*Global-fit setup on a synthetic Ag LF-KT decoupling series, B_L = 0, 15,*
*50, and 100 G with shared Δ ≈ 0.39 μs⁻¹. The longitudinal field decouples*
*the nuclear dipolar broadening as γ_μB_L/Δ grows past unity (Hayano et al.*
*PRB 20, 850, 1979 — textbook Fig 5.6).*

The global-fit result window is split into two synchronized views:

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

Fitted Parameters Panel
-----------------------

After a global fit, the Fitted Parameters panel shows how the varying
(per-dataset) parameters change across runs. It provides:

Parameter Table
~~~~~~~~~~~~~~~

A tabular view of all fitted varying parameters, with columns for:

* Run number
* :math:`B` (G) — applied magnetic field
* :math:`T` (K) — sample temperature
* Each varying parameter with its uncertainty

Parameter Trend Plot
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

Exporting Data
~~~~~~~~~~~~~~

**Export CSV**: Save the parameter table to a CSV file for use in spreadsheets
or other analysis software.

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
   to PDF or EPS and a preview window is shown
6. If GLE is not installed, the script and data files are still saved —
   you can compile them later with ``gle -d pdf <name>.gle`` from inside the
   ``.gleplot`` folder

.. note::

   Install GLE from source or from http://glx.sourceforge.io/ to enable
   automatic compilation. The ``gleplot`` Python package is required for
   script generation (``pip install gleplot``).

Global-fit GLE exports
~~~~~~~~~~~~~~~~~~~~~~

The Global Parameter Fit window provides two dedicated GLE export buttons:

* **Export fits to GLE** (left pane): exports the per-dataset fit overlays
* **Export plot(s) to GLE** (right pane): exports local-parameter trend plots

For local-parameter exports, Asymmetry also writes:

* A ``*_local_parameters.dat`` table with units and an explicit column map
* Optional ``*_local_<parameter>.fit`` files for any active local model fit overlays

Both global-fit export buttons use the same foldered layout, so the exported
``.gle`` file, compiled output, and sidecars stay together in one bundle.

Both global and local export dialogs remember the last export directory and use
it as the default location next time.

Analysis Workflows
------------------

Basic Analysis
~~~~~~~~~~~~~~

1. Load data files
2. Sort by field or temperature
3. Select and view different runs
4. Adjust plot limits for detailed inspection
5. Apply bunching to reduce noise if needed

Comparing Runs
~~~~~~~~~~~~~~

1. Filter by temperature or field
2. Click through filtered runs to compare
3. Use co-adding to average similar runs

Temperature Series
~~~~~~~~~~~~~~~~~~

1. Sort by temperature column
2. Step through runs in order
3. Filter by field to isolate specific conditions

Field Series
~~~~~~~~~~~~

1. Sort by field column  
2. Step through runs in order
3. Filter by temperature for isothermal scans

UI Scale
--------

The interface density can be adjusted from **View → UI Scale**. Five preset
scales are available: 80 %, 90 %, 100 %, 110 %, and 120 %. The selection is
persisted between sessions. The default is 90 %.

Changing the scale adjusts:

* Base font size across panels and dialogs
* Table row heights
* Dock minimum widths
* Compact padding in parameter and data-browser tables

Keyboard Shortcuts
------------------

* **Ctrl+O**: Open file(s)
* **Up/Down arrows**: Navigate table selection
* **Ctrl+A**: Select all (in table)

Tips and Tricks
---------------

* Double-click column borders to auto-resize
* Use "All" filter with partial text to find specific runs
* Co-add replicate measurements to improve statistics
* Bunch data before Fourier analysis to reduce computation time
* Right-click on plot to save figure (via matplotlib toolbar)

Saving and Reopening Projects
------------------------------

A *project file* (``.asymp``) saves the complete state of your analysis
session.  You can close Asymmetry and resume exactly where you left off, or
maintain several independent analyses side-by-side.

Creating and saving
~~~~~~~~~~~~~~~~~~~~

* **File → New Project** clears the current session after asking for
   confirmation.  Any unsaved changes are discarded.
* **File → Save Project** (``Ctrl+S``) saves to the current project file.
   If no project file is open yet, you will be prompted to choose a location.
* **File → Save Project As…** always asks for a new filename.

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
* Plot axis limits (X/Y), fit range, and current run
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
