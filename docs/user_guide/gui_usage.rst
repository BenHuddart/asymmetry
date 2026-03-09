GUI Usage
=========

The Asymmetry GUI provides an interactive interface for data analysis.

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
2. Select one or multiple .wim files (use Ctrl+Click or Shift+Click)
3. All selected files will be loaded into the data browser

The data browser shows:

* Run number
* Title
* Temperature (K)
* Magnetic field (G)

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

**Search Filter**
~~~~~~~~~~~~~~~~~

Use the search filter at the top for quick text-based filtering:

1. Select a column from the dropdown (or "All" for global search)
2. Type filter text (case-insensitive, partial matches work)
3. Matching rows are shown, others hidden
4. Click "Clear" to reset the filter

Multi-Selection
~~~~~~~~~~~~~~~

* Click a row to select it
* Ctrl+Click to add/remove from selection
* Shift+Click to select a range

Viewing Data
~~~~~~~~~~~~

Click any dataset row to plot it in the main panel.

Co-adding Datasets
~~~~~~~~~~~~~~~~~~

To average multiple datasets:

1. Select 2 or more datasets (Ctrl+Click or Shift+Click)
2. Click "Co-add Selected"
3. A new combined dataset appears with negative run number
4. Title shows: ``[Combined: run1, run2, ...]``

The co-added dataset:

* Uses the first dataset's time grid
* Averages asymmetry values
* Propagates errors correctly: σ_combined = √(Σσ²) / N
* Can be plotted and analyzed like any other dataset

Separating Combined Data
~~~~~~~~~~~~~~~~~~~~~~~~

To remove a co-added dataset:

1. Select the combined dataset (has negative run number)
2. Click "Separate Combined"
3. The combined entry is removed (originals remain)

Plot Panel Controls
-------------------

The plot panel displays the currently selected dataset with error bars.

Axis Limits
~~~~~~~~~~~

Control the plot range using the spinboxes at the top:

* **X min/max**: Time axis range (μs)
* **Y min/max**: Asymmetry axis range
* Click **Apply** to update limits
* Click **Auto** to auto-scale to fit all data

Default limits automatically adjust to fit the data including error bars, 
with 5% padding.

Bunch Factor
~~~~~~~~~~~~

Reduce noise by combining adjacent points:

1. Set "Bunch: factor" spinbox to desired value (1 = no bunching)
2. Plot updates automatically
3. Original data is preserved; bunching is only for display/analysis

Example: factor=10 reduces 1000 points to 100 points with smaller error bars.

Fitting Panel
-------------

The fitting panel provides an interactive interface for fitting models to your data.
It is located in the right dock area next to the plot panel.

**Single Dataset Fitting**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "Single" tab fits one model to the currently selected dataset:

1. **Select a dataset** in the data browser (it will be plotted)
2. **Choose a model** from the dropdown:
   
   * ExponentialRelaxation: Simple exponential decay
   * GaussianRelaxation: Gaussian Kubo-Toyabe
   * Oscillatory: Damped oscillation
   * StretchedExponential: Kohlrausch relaxation
   * StaticGKT_ZF: Zero-field Gaussian Kubo-Toyabe

3. **Adjust parameters** in the table:
   
   * **Value**: Initial guess for the parameter
   * **Fix**: Check to hold the parameter constant during fit
   * **Min/Max**: Set bounds (leave empty for no bounds)

4. **Click "Run Fit"** to execute the fitting

5. **Review results:**
   
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
3. **Choose a model** from the dropdown
4. **Mark shared parameters**: Check "Global" for parameters that should have
   the same value across all datasets (e.g., ``A0``)
5. **Leave per-dataset parameters unchecked**: These vary independently for each
   dataset (e.g., ``lambda``, ``sigma``)
6. **Click "Run Global Fit"**

After a global fit completes:

* Fit curves appear on the plot for all datasets
* The **Fitted Parameters** panel opens automatically (see below)
* The log panel shows a summary with average χ²ᵣ

Fitted Parameters Panel
-----------------------

After a global fit, the Fitted Parameters panel shows how the varying
(per-dataset) parameters change across runs. It provides:

Parameter Table
~~~~~~~~~~~~~~~

A tabular view of all fitted varying parameters, with columns for:

* Run number
* 𝐵 (G) — applied magnetic field
* 𝑇 (K) — sample temperature
* Each varying parameter with its uncertainty

Parameter Trend Plot
~~~~~~~~~~~~~~~~~~~~

A plot of one selected parameter versus a sweep variable. Controls include:

* **X axis**: Choose between Auto (inferred from data), 𝐵 (G), 𝑇 (K), or Run number.
  Auto mode detects whether field or temperature varies across the datasets.
* **Y parameter**: Select which varying parameter to plot
* **Scale**: Check **Log X** and/or **Log Y** to use logarithmic axes —
  useful for power-law behaviour or data spanning several orders of magnitude

Exporting Data
~~~~~~~~~~~~~~

**Export CSV**: Save the parameter table to a CSV file for use in spreadsheets
or other analysis software.

**Export to GLE**: Generate a publication-quality figure using the
`GLE Graphics Layout Engine <http://glx.sourceforge.io/>`_ via the
``gleplot`` Python library:

1. Click **"Export to GLE"**
2. Choose a save location for the ``.gle`` script
3. Select the output format from the **Format** dropdown (PDF or EPS)
4. The export writes:

   * A ``.dat`` data file with column headers and the globally shared
     parameters recorded as comments at the top of the file
   * A ``.gle`` script that plots the currently selected parameter trend

5. If GLE is installed on your system, the script is compiled automatically
   to PDF or EPS and a preview window is shown
6. If GLE is not installed, the script and data files are still saved —
   you can compile them later with ``gle -d pdf fit_parameters.gle``

.. note::

   Install GLE from source or from http://glx.sourceforge.io/ to enable
   automatic compilation. The ``gleplot`` Python package is required for
   script generation (``pip install gleplot``).

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
