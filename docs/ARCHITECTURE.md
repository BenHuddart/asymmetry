# Asymmetry — μSR Data Analysis Library

## 1. Overview

**Asymmetry** is a Python library and application for the analysis of muon-spin spectroscopy (μSR) data. It provides a modular core engine for data reduction, fitting, and Fourier analysis, together with an interactive graphical front-end for visualization and experiment management.

The name reflects the fundamental observable in μSR: the time-dependent asymmetry of the muon-decay positron count rates.

---

## 2. Design Principles

| Principle | Detail |
|---|---|
| **Separation of concerns** | A pure-Python core library (`asymmetry.core`) with no GUI dependencies, and a distinct GUI package (`asymmetry.gui`). |
| **Scriptability** | Every operation available in the GUI must be accessible programmatically through the core API, enabling batch processing, Jupyter notebook workflows, and automated pipelines. |
| **Extensibility** | Plugin-style architecture for data loaders, fit functions, and Fourier methods so that new formats and models can be added without modifying the core. |
| **Reproducibility** | Analysis sessions should be serializable so that results can be reproduced and shared. |

---

## 3. Architecture

```
asymmetry/
├── core/               # Pure-Python analysis engine (no GUI deps)
│   ├── __init__.py
│   ├── data/           # Data model and container classes
│   │   ├── __init__.py
│   │   ├── dataset.py        # MuonDataset, Run, Histogram classes
│   │   └── logbook.py        # Logbook / run-table management
│   ├── io/             # File I/O loaders (plugin-based)
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract loader interface
│   │   ├── nexus.py          # NeXus / HDF5 loader
│   │   ├── psi.py            # PSI BIN/MDU raw histogram loader
│   │   ├── root.py           # ROOT file loader
│   │   ├── ascii.py          # Column-delimited ASCII / CSV
│   │   └── mud.py            # TRIUMF MUD format
│   ├── transform/      # Data transformations
│   │   ├── __init__.py
│   │   ├── asymmetry.py      # Asymmetry calculation from raw histograms
│   │   ├── grouping.py       # Detector grouping
│   │   ├── deadtime.py       # Dead-time correction
│   │   ├── rebin.py          # Rebinning utilities
│   │   └── background.py     # Background estimation & subtraction
│   ├── fitting/        # Fitting engine
│   │   ├── __init__.py
│   │   ├── engine.py         # Fit driver: single-run & global
│   │   ├── composite.py      # Composite A(t) builder primitives
│   │   ├── fit_wizard.py     # Single-spectrum fit fingerprinting and model comparison
│   │   ├── grouped_time_domain.py  # Grouped-series fit engine
│   │   ├── result_summary.py # Shared JSON-serialisable fit-result summary
│   │   ├── models.py         # Built-in μSR fit functions
│   │   ├── parameters.py     # Parameter objects with bounds & linking
│   │   ├── minimizers.py     # Minimizer backends (scipy, lmfit, iminuit)
│   │   └── results.py        # Fit-result container & statistics
│   ├── fourier/        # Frequency-domain analysis
│   │   ├── __init__.py
│   │   ├── spectrum.py       # GroupSpectrumConfig, compute_average_group_spectrum
│   │   ├── fft.py            # Standard FFT
│   │   ├── maxent.py         # Maximum-entropy transform
│   │   └── window.py         # Apodization / window functions
│   ├── representation/ # Domain representation model (recipe + fit + trend)
│   │   ├── __init__.py       # Public re-exports (FitSeries, FitSlot, RepresentationType, …)
│   │   ├── base.py           # Representation ABC, RepresentationType enum, FitSlot
│   │   ├── container.py      # DatasetRepresentations — per-run representation map
│   │   ├── factory.py        # from_rep_type() — constructs concrete Representation subclasses
│   │   ├── time.py           # TimeFBAsymmetry, TimeGroups representations
│   │   ├── frequency.py      # FrequencyFFT, FrequencyMaxEnt representations
│   │   ├── series.py         # FitSeries — ordered member series + divergence tracking
│   │   ├── trend_state.py    # TrendState dataclass for Fit Parameters panel state
│   │   └── project_model.py  # ProjectModel — in-memory owner of representations + batches
│   ├── project/        # Project persistence
│   │   ├── __init__.py
│   │   └── schema.py         # load_project, save_project, migrate_to_current (v1–v7)
│   └── utils/          # Shared utilities
│       ├── __init__.py
│       ├── constants.py      # Physical constants (μ⁺ gyromagnetic ratio, etc.)
│       └── units.py          # Unit handling / conversion
│
├── gui/                # Graphical front-end
│   ├── __init__.py
│   ├── app.py                # Application entry point
│   ├── mainwindow.py         # Main window shell
│   ├── panels/               # Dockable panels / views
│   │   ├── data_browser.py   # Run browser / logbook view
│   │   ├── plot_panel.py     # Interactive plotting canvas
│   │   ├── fit_panel.py      # Fit setup and results (SingleFitTab, GlobalFitTab, FitPanel)
│   │   ├── fit_parameters_panel.py  # Parameter trending panel (pull-based, representation-aware)
│   │   ├── fit_function_builder.py  # Composite fit-function dialog
│   │   ├── fourier_panel.py  # Fourier analysis controls
│   │   ├── initial_values_dialog.py # Per-member initial-values editor for batch fits
│   │   └── log_panel.py      # Message / command log
│   ├── dialogs/              # Modal dialogs
│   │   ├── load_data.py      # File-open with format detection
│   │   ├── preferences.py    # Application settings
│   │   └── export.py         # Export data / figures
│   ├── plotting/             # Plot helpers and renderers
│   │   ├── __init__.py
│   │   └── mpl_canvas.py     # Matplotlib canvas integration
│   ├── windows/              # Top-level analysis windows
│   │   ├── fit_wizard_window.py       # Guided single-spectrum fit wizard
│   │   ├── multi_group_fit_window.py  # Grouped fit surface (Single + Batch tabs)
│   │   └── ...
│   └── resources/            # Icons, stylesheets, etc.
│
├── cli.py              # Optional command-line interface
├── __init__.py
└── __main__.py         # `python -m asymmetry` entry point
```

### 3.1 Core (`asymmetry.core`)

The core has **zero** GUI dependencies. It depends only on the scientific Python stack and format-specific I/O libraries. This allows it to be used as a standalone library in scripts, notebooks, and CI pipelines.

### 3.2 GUI (`asymmetry.gui`)

The GUI is a separate, optional install target. It wraps the core API and provides interactive visualization, fitting dialogs, and logbook management.

### 3.3 Fit Wizard Boundary

The single-spectrum Fit Wizard is intentionally split across the core and GUI
layers.

- `asymmetry.core.fitting.fit_wizard` owns the deterministic, testable analysis
  pipeline: spectrum fingerprinting, curated candidate selection, multi-start
  fitting, AIC/AICc/BIC calculation, residual diagnostics, recommendation
  payloads, and serialization of cached wizard results for persistence.
- `asymmetry.gui.windows.fit_wizard_window` owns presentation: the four-step
  workflow, plots, metric explainers, residual warnings, and applying an
  accepted candidate back into the single-fit tab.

This separation keeps the recommendation logic usable outside Qt, makes the
feature straightforward to test with synthetic spectra, and leaves room to add
future comparison back-ends such as Bayesian evidence without redesigning the
window workflow.

### 3.4 Global Fit Wizard Boundary

The Global Fit Wizard follows the same split, but with one extra concern:
parameter sharing across an ordered series.

- `asymmetry.core.fitting.global_fit_wizard` owns the reusable analysis logic:
  ordered-axis inference, shared candidate-portfolio construction, the phase-1
  helper that completes missing per-run single-fit wizard tables for the shared
  portfolio, screening-table aggregation from those independent fits, coupled
  global optimization for user-selected candidate keys, information-criterion
  scoring, per-run residual diagnostics, continuity warnings, and reusable
  recommendation payloads that mix screening-only and globally optimized rows.
- `asymmetry.gui.windows.global_fit_wizard_window` owns the non-modal workflow:
  the screening-first interactive workflow, metric explainers, warning
  summaries, batch optimization controls, and applying a selected optimized
  recommendation back into the global-fit tab.
- `asymmetry.gui.panels.fit_panel.GlobalFitTab` remains the integration point
  for the real fit state. It opens the wizard, provides the current model and
  role configuration as context, persists any newly generated per-run
  single-fit wizard tables, and reuses the wizard's computed fit bundle to
  refresh plots and fitted-parameter views without rerunning the fit.

This keeps the recommendation engine scriptable and testable while avoiding a
second, UI-only implementation of the global-fit logic.

**Implemented toolkit:** PySide6 (Qt 6), chosen for its maturity, licensing flexibility (LGPL), and proven track record in large scientific applications (Mantid, SasView). The GUI layout draws inspiration from **WiMDA** — a data browser / logbook on the left, a central plot canvas, fit and Fourier analysis panels docked on the right, and a log/message panel at the bottom.

### 3.5 Domain Representation Model

The **representation model** (`asymmetry.core.representation`) is the spine of
the analysis session. It decouples *what was computed and fitted* from *how the
GUI currently looks*.

#### Representations

A `Representation` is a recipe-driven view of a `Run`. There are four kinds,
one per analysis domain:

| RepresentationType | Domain | Computes |
|---|---|---|
| `TIME_FB_ASYMMETRY` | Time | F-B asymmetry A(t) |
| `TIME_GROUPS` | Time | Lifetime-corrected detector-group traces |
| `FREQ_FFT` | Frequency | Averaged grouped FFT spectrum |
| `FREQ_MAXENT` | Frequency | MaxEnt spectrum (reserved) |

Each `Representation` stores:

- **recipe** — generation parameters (e.g. FFT window, padding, phase). Transient
  arrays are *not* persisted; they are recomputed from the recipe on project load.
- **fit** (`FitSlot`) — the most recent fit for this `(dataset, representation)`
  pair: model dict, fitted-parameter list, result summary, provenance
  (`"none"` / `"single"` / `"batch"` / `"global"`), and flags used by the
  trending panel (`diverged`, `include_in_trend`, `batch_id`).
- **trend_state** — opaque dict persisting the user's axis/parameter selections
  in the Fit Parameters panel.

#### FitSeries

A `FitSeries` (`asymmetry.core.representation.series`) collects multiple
member fits — either across runs (`member_kind="runs"`) or across a run's
detector groups (`member_kind="groups"`) — into one trendable unit. Members
are keyed by integer: real run numbers for run series, synthetic negative keys
`-(source_run * 1000 + group_index)` for group series.

Key attributes: `canonical_model`, `param_roles` (Global / Local / Fixed per
physics parameter), `nuisance_params` (group-only, always local),
`results_by_run` (per-member summary dicts that drive the trending panel),
`diverged_runs` (members whose stored model no longer matches the canonical).

#### ProjectModel

`ProjectModel` (`asymmetry.core.representation.project_model`) is the
in-memory owner of all representations and series for the active project. It
provides:

- `refresh_divergence()` — re-evaluates every batch member's model against its
  series canonical model; group series are evaluated at the source-run level.
- `trend_runs_for_batch(batch)` — returns ordered member keys where the
  source-run `FitSlot.include_in_trend` is True (group-aware).
- `set_member_trend_inclusion(batch_id, run_number, include)` — manually
  toggles a member's inclusion, routing synthetic keys to their source run.

#### Fit Parameters panel (pull model)

The `FitParametersPanel` operates on a **pull** model: whenever a fit
completes or the active representation changes, `MainWindow._refresh_trend_panel`
reads the relevant `FitSeries` from `ProjectModel`, builds per-member row dicts
(using `_frequency_spectra_by_run` for FFT series), and calls
`FitParametersPanel.load_representation_series`. The panel shows a **"Showing:"**
label indicating the active representation and a **Series** button row —
one button per `FitSeries` for the active representation. Selecting a series
button emits `series_selection_changed`, which the main window forwards to
`DataBrowserPanel.set_highlighted_runs`, tinting the member runs amber.

---

## 4. Functional Requirements

### 4.1 Data I/O

> **Implementation note:** ISIS muon NeXus, PSI BIN/MDU, and MusrRoot/LEM ROOT
> loaders are implemented in the shared loader registry.
> ASCII/CSV and TRIUMF MUD remain planned extension points.

| ID | Requirement |
|---|---|
| IO-1 | Load μSR histogram data from **NeXus/HDF5** files (ISIS, PSI, J-PARC NeXus conventions). *(Deferred — pending format confirmation.)* |
| IO-2 | Load **MusrRoot/LEM ROOT** files with `RunHeader` metadata and `hDecay` histograms, following musrfit's `PRunDataHandler::ReadRootFile`. |
| IO-3 | Load column-delimited **ASCII/CSV** files with user-defined column mappings. |
| IO-4 | Load **PSI binary** (`.bin`) and **PSI MDU** (`.mdu`) raw histogram formats. |
| IO-5 | Load **TRIUMF MUD** format. |
| IO-7 | Auto-detect file format where possible; fall back to manual selection. |
| IO-8 | Provide a **plugin interface** so third-party loaders can be registered without modifying the core. |
| IO-9 | Export processed asymmetry data and fit results to ASCII, HDF5, and JSON. |

### 4.2 Data Model

| ID | Requirement |
|---|---|
| DM-1 | Represent a single measurement run with metadata (run number, temperature, field, sample, comments, etc.). |
| DM-2 | Store per-detector raw histograms with timing information (t₀, bin width, good-bin range). |
| DM-3 | Support detector **grouping** definitions (forward, backward, custom groups). |
| DM-4 | Compute the **asymmetry** $A(t) = \frac{N_F(t) - \alpha\, N_B(t)}{N_F(t) + \alpha\, N_B(t)}$ from grouped histograms. |
| DM-5 | Apply **dead-time correction** to raw histograms using file-provided deadtime values when present. |
| DM-6 | Provide **rebinning** of time-domain data with variable bin widths. |
| DM-7 | Estimate and subtract **background** counts from grouped raw forward/backward histograms when enabled. |

### 4.2.1 PSI and Deadtime Provenance

The PSI support is deliberately tied to existing muon-analysis implementations:

- PSI BIN/MDU parsing follows musrfit's PSI raw-data reader, especially
  `PRunDataHandler::ReadPsiBinFile` and the PSI BIN/MDU structures used there.
  Mantid's `LoadPSIMuonBin` is used as an additional cross-check for PSI-BIN
  files.
- PSI-BIN `.mon` temperature sidecars follow Mantid's `LoadPSIMuonBin`
  behavior: search from the BIN directory up to three levels below it for a
  `.mon` filename containing the run number, parse the PSI title/date header
  and backslash-delimited rows, and expose each channel as a plottable run log
  in `nexus_time_series`. musrfit is still the reference for embedded PSI BIN
  temperature fields, but it was not found to implement `.mon` sidecar loading.
- Per-detector PSI `t0` values are preserved as detector metadata and used by
  grouping before histograms are summed.
- File-based deadtime correction follows the non-paralyzable correction used by
  musrfit `PRunBase::DeadTimeCorrection` and Mantid `ApplyDeadTimeCorr`, but
  it is applied only when the loaded file provides detector deadtime values
  and good-frame metadata. PSI BIN/MDU and MusrRoot/LEM ROOT files normally do
  not provide these NeXus-style deadtime constants, so they do not use a
  deadtime fallback.

### 4.2.2 Background Provenance

Raw-count background correction follows musrfit's `PRunAsymmetry` ordering:
group forward/backward histograms first, subtract fixed or estimated
background next, then calculate asymmetry. Estimated background values are the
mean counts in an inclusive background-bin range. When no range is supplied,
Asymmetry uses musrfit's fallback `0.1 * t0` to `0.6 * t0` range. PSI and
TRIUMF background ranges are adjusted toward complete accelerator periods using
the musrfit constants (`0.01975 us` and `0.04337 us` respectively).

The toggle is off by default and exposed only for PSI-style raw data
(`.bin`, `.mdu`, and PSI/LEM `.root`). Existing ISIS/NeXus raw grouping remains
on the file-deadtime path, and Asymmetry does not use background subtraction as
an ISIS deadtime fallback.

### 4.2.3 ROOT Provenance

ROOT support follows musrfit's `PRunDataHandler::ReadRootFile`. Asymmetry reads
MusrRoot `RunHeader` metadata and `hDecay%03d` histograms from both the
documented `TFolder` layout and the newer `TDirectory` layout. The musrfit
paths `histos/hDecay%03d` and `histos/DecayAnaModule/hDecay%03d` are both
searched. `RunInfo` metadata, `RedGreen Offsets`, and `DetectorInfo` fields are
used to preserve run metadata, detector labels, per-detector `t0`, and
good-bin ranges.

Legacy LEM ROOT files without a MusrRoot `RunHeader` are treated as a
best-effort compatibility path when `uproot` can expose their `RunInfo` and
histogram objects. The full TLemRunHeader/PyROOT object model is not
reimplemented.

### 4.3 Fitting — Time Domain

| ID | Requirement |
|---|---|
| FT-1 | Fit asymmetry data $A(t)$ to user-selected model functions. |
| FT-2 | Built-in models: exponential relaxation, Gaussian relaxation, oscillatory (cosine), stretched exponential, Kubo-Toyabe (static & dynamic, zero-field & longitudinal-field), Abragam, and combinations thereof. |
| FT-3 | Allow **user-defined fit functions** (Python callables or expression strings). |
| FT-4 | Support **composite models** built from sums/products of basis functions. |
| FT-5 | Support parameter **constraints**: fix, bound, tie (link) between parameters. |
| FT-6 | **Global (simultaneous) fitting** across multiple runs with shared and independent parameters. |
| FT-7 | Provide multiple minimizer back-ends: least-squares (Levenberg-Marquardt), Nelder-Mead, differential evolution, and (optionally) Minuit via `iminuit`. |
| FT-8 | Report fit statistics: $\chi^2$, $\chi^2_\text{red}$, parameter uncertainties, covariance matrix. |
| FT-9 | Visualize fit residuals. |
| FT-10 | Guide single-spectrum time-domain fitting with a wizard that fingerprints the active dataset, compares a curated portfolio of supported composite models, and applies the chosen result back into the fit panel. |
| FT-11 | Support grouped time-domain fitting: a **Single** tab fits one dataset's detector groups jointly; a **Batch** tab fits a multi-run series with the same grouped model, recording results as a ``FitSeries`` for parameter trending. Physics parameters are classified as ``Global`` (shared across runs), ``Local`` (per-run), or ``Fixed``; nuisance parameters (N₀, background, amplitude, relative_phase) are always per-(run, group). |

### 4.3.1 Grouped Time-Domain Boundary

The grouped time-domain feature follows the same core/GUI split as all other
fit workflows, with the additional concept of a **FitSeries** as the central
persisted object.

- `asymmetry.core.fitting.grouped_time_domain` owns grouped-domain preparation
  and grouped fitting. `fit_grouped_series` orchestrates individual, batch, and
  global grouped fits by calling `fit_grouped_time_domain` (the per-run
  building block) for each member and collecting results keyed by synthetic
  member keys of the form `-(source_run * 1000 + group_index)`.
- `asymmetry.core.representation.series.FitSeries` is the central persisted
  object. Each grouped fit (single-run or multi-run batch) records its results
  into a `FitSeries(member_kind="groups")` so parameter trending is available
  for grouped fits on the same footing as asymmetry fits.
- `asymmetry.core.representation.project_model.ProjectModel` owns the
  in-memory representation + series state and handles group-aware divergence
  detection.
- `asymmetry.gui.windows.multi_group_fit_window.MultiGroupFitWindow` owns the
  grouped-fit surface. It presents two `GlobalFitTab(member_kind="groups")`
  instances — a **Single** tab for fitting one run's detector groups and a
  **Batch** tab for multi-run grouped series — inside the main fit dock when
  the time-domain view is in grouped mode. Scope (member kind, member set) is
  derived from the active representation and data-browser selection rather than
  an explicit selector.
- `asymmetry.gui.mainwindow.MainWindow` is the integration point that ties
  grouped mode to the active dataset, the current time-domain plot, the
  grouping definitions, and the `ProjectModel`. It calls
  `_record_grouped_fit_series` (which persists a `FitSeries`) and
  `_refresh_trend_panel` (which pulls the series into the Fit Parameters panel).
- `asymmetry.gui.panels.plot_panel.PlotPanel` owns the stacked grouped trace
  display in the time-domain workspace.

This keeps grouped-count objective construction out of the GUI and leaves room
for later extensions such as detector-level fitting, explicit detector phase
tables, or grouped fit wizards without redesigning the current split.

### 4.4 Fourier Analysis — Frequency Domain

| ID | Requirement |
|---|---|
| FF-1 | Compute the **discrete Fourier transform** (real, imaginary, magnitude, phase) of asymmetry data. |
| FF-2 | Provide **apodization (window) functions**: Gaussian, Lorentzian, cosine, Hann, etc. |
| FF-3 | Support **Maximum Entropy** spectral reconstruction for enhanced frequency resolution. |
| FF-4 | Allow selection of the time range and padding prior to transformation. |
| FF-5 | Display frequency-domain spectra overlaid with time-domain fits where applicable. |

### 4.5 Logbook & Run Management

| ID | Requirement |
|---|---|
| LB-1 | Maintain a **logbook** (run table) listing all loaded runs with key metadata columns. |
| LB-2 | Allow **sorting, filtering, and searching** of the logbook by any metadata field. |
| LB-3 | Support **tagging** and user-defined annotations on runs. |
| LB-4 | Group runs into named **collections** for batch operations and global fits. |
| LB-5 | Persist logbook state across sessions (save/load). |
| LB-6 | Import/export logbook to CSV or spreadsheet formats. |

### 4.6 Plotting & Visualization

| ID | Requirement |
|---|---|
| PL-1 | Interactive time-domain plots of raw counts and asymmetry with error bars. |
| PL-2 | Overlay fit curves on data plots. |
| PL-3 | Frequency-domain spectral plots. |
| PL-4 | Parameter-vs-run-variable plots (e.g. relaxation rate vs. temperature). |
| PL-5 | Support for **multiple plot tabs/tiles** with linked or independent axes. |
| PL-6 | Export figures to PNG, SVG, PDF. |
| PL-7 | Matplotlib as the default rendering back-end, with the option to explore faster alternatives (e.g. PyQtGraph) later. |

### 4.6.1 Plot Responsiveness Policy

The current plot-responsiveness strategy keeps analysis fidelity separate from
display density.

- `asymmetry.gui.panels.plot_panel.PlotPanel` retains full-resolution
  `MuonDataset.time`, `asymmetry`, and `error` arrays in memory for the active
  view. These arrays remain the source of truth for fit inputs, auto-limit
  calculations, annotations, and export payloads.
- Dense matplotlib `errorbar()` artists are reduced only at render time. The
  panel applies a bounded per-trace sample cap before drawing visible points so
  large time-domain, grouped-time, and frequency-domain views do not emit every
  original sample into the canvas.
- Decimation is viewport-aware once limits have been established. The panel
  prefers points inside the current x-range and re-renders when the user pans,
  zooms, presses Auto X/Auto Y, or edits axis limits directly.
- Viewport-triggered redraws are deferred onto the Qt event loop and coalesced
  so paired `xlim_changed`/`ylim_changed` callbacks do not recursively trigger
  repeated redraws.
- Figure export intentionally bypasses this display-density policy. Export data
  is rebuilt from the full analysis arrays and fit payloads so saved `.dat`,
  `.fit`, and GLE/PDF/EPS outputs preserve the underlying resolution.
- This policy improves interaction cost without changing the core data model.
  Raw histograms, regrouping, and grouped Fourier preparation still use the
  full run payloads owned by `Run.histograms` and the existing grouping state.

Current limitations:

- The render cap is still a simple bounded sample strategy rather than a true
  min/max envelope reducer per pixel column.
- Y-only navigation still redraws the current view because the panel treats a
  limit change as a signal to rebuild the visible artists.
- Grouped Fourier compute cost is independent of this policy; only the plotted
  spectrum density changes here, not the FFT generation itself.

---

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-1 | **Python ≥ 3.10** (leverage modern typing, pattern matching). |
| NF-2 | Core dependencies: `numpy`, `scipy`, `h5py`, `lmfit`. Optional: `uproot` (ROOT I/O), `iminuit`, `matplotlib`. |
| NF-3 | GUI dependencies: `PySide6`, `matplotlib` (for embedded plots). |
| NF-4 | Cross-platform: Linux, macOS, Windows. |
| NF-5 | Installable from the Git repository via pip, with optional extras for GUI and I/O/export features. |
| NF-6 | Test suite with ≥ 80 % line coverage on the core; pytest as test runner. |
| NF-7 | Documentation: API reference (Sphinx/autodoc) and user guide (tutorials, examples). |
| NF-8 | Semantic versioning; changelog maintained per release. |
| NF-9 | Permissive open-source license (MIT or BSD-3-Clause recommended). |

---

## 6. Dependencies (initial)

### Core
| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | FFT, optimization, special functions (e.g. Kubo-Toyabe) |
| `h5py` | NeXus / HDF5 I/O |
| `lmfit` | High-level fitting with parameter objects |
| `uproot` | ROOT file I/O (optional) |
| `iminuit` | Minuit minimizer (optional) |

### GUI
| Package | Purpose |
|---|---|
| `PySide6` | Qt-based GUI toolkit |
| `matplotlib` | Plotting / figure export |

---

## 7. Milestones (suggested)

| Phase | Deliverable |
|---|---|
| **M0 — Skeleton** | Project scaffolding, packaging (`pyproject.toml`), CI, empty module structure. |
| **M1 — Data I/O** | NeXus and ASCII loaders; `MuonDataset` data model; asymmetry calculation; basic tests. |
| **M2 — Fitting** | Single-run fitting with built-in models; parameter management; result reporting. |
| **M3 — Fourier** | FFT and MaxEnt transforms; apodization; frequency-domain plotting. |
| **M4 — GUI shell** | Main window, data browser, plot panel, fit panel wired to the core. |
| **M5 — Global fitting** | Simultaneous multi-run fits with shared parameters. |
| **M6 — Logbook** | Logbook panel with persistence, filtering, and batch operations. |
| **M7 — Polish** | ROOT loader, MUD loader, export workflows, documentation, packaging. |

---

## 8. Open Questions

1. **NeXus conventions** — Which NXmuon application definition(s) should we target first? ISIS (`MUSR`, `EMU`, `HIFI`, `ARGUS`) and/or PSI (`GPS`, `LEM`)?
2. **MaxEnt algorithm** — Implement from scratch or wrap an existing library (e.g. `maxent` from Mantid)?
3. **Parallel computing** — Should the fitting engine support multiprocessing/threading for global fits from the start, or defer to a later milestone?
4. **Session serialization** — JSON project files, or HDF5-based session files, or both?
5. **Web front-end** — Is a secondary web-based interface (e.g. via Panel or Jupyter widgets) desired alongside the desktop GUI?

---

*This specification is a living document. It will be refined as design decisions are made and prototyping progresses.*
