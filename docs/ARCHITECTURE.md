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
│   │   ├── root.py           # ROOT file loader
│   │   ├── ascii.py          # Column-delimited ASCII / CSV
│   │   ├── psi_bin.py        # PSI binary format
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
│   │   ├── models.py         # Built-in μSR fit functions
│   │   ├── parameters.py     # Parameter objects with bounds & linking
│   │   ├── minimizers.py     # Minimizer backends (scipy, lmfit, iminuit)
│   │   └── results.py        # Fit-result container & statistics
│   ├── fourier/        # Frequency-domain analysis
│   │   ├── __init__.py
│   │   ├── fft.py            # Standard FFT
│   │   ├── maxent.py         # Maximum-entropy transform
│   │   └── window.py         # Apodization / window functions
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
│   │   ├── fit_panel.py      # Fit setup and results
│   │   ├── fit_function_builder.py  # Composite fit-function dialog
│   │   ├── fourier_panel.py  # Fourier analysis controls
│   │   └── log_panel.py      # Message / command log
│   ├── dialogs/              # Modal dialogs
│   │   ├── load_data.py      # File-open with format detection
│   │   ├── preferences.py    # Application settings
│   │   └── export.py         # Export data / figures
│   ├── plotting/             # Plot helpers and renderers
│   │   ├── __init__.py
│   │   └── mpl_canvas.py     # Matplotlib canvas integration
│   ├── windows/              # Top-level analysis windows
│   │   ├── fit_wizard_window.py      # Guided single-spectrum fit wizard
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

**Toolkit candidates** (to be evaluated in a prototype phase):

| Toolkit | Pros | Cons |
|---|---|---|
| **PySide6 (Qt 6)** | Mature widget set, dockable panels, excellent cross-platform support, widely used in scientific apps (Mantid, SasView). | Large dependency; licensing (LGPL) acceptable but needs tracking. |
| **PyQt6** | Functionally identical to PySide6. | GPL-licensed unless commercial; may conflict with permissive library licensing. |
| **Dear PyGui** | GPU-accelerated, modern look, easy plotting integration. | Smaller ecosystem; less mature for complex dock layouts. |
| **Streamlit / Panel (web-based)** | Zero-install for users; notebook-friendly. | Less responsive for heavy interactive fitting; harder to get desktop-app feel. |

**Initial recommendation:** PySide6, due to its maturity, licensing flexibility, and proven track record in large scientific applications.  The GUI layout draws inspiration from **WiMDA** — a data browser / logbook on the left, a central plot canvas, fit and Fourier analysis panels docked on the right, and a log/message panel at the bottom.

---

## 4. Functional Requirements

### 4.1 Data I/O

> **Implementation note:** The initial release targets **WiMDA `.wim` files**
> as the primary data format.  NeXus/HDF5 and ROOT loaders will be added once
> the exact muon-specific schemas at each facility are confirmed.  The existing
> `wim_parser.py` has been adapted into the `asymmetry.core.io.wim` module.

| ID | Requirement |
|---|---|
| IO-1 | Load μSR histogram data from **NeXus/HDF5** files (ISIS, PSI, J-PARC NeXus conventions). *(Deferred — pending format confirmation.)* |
| IO-2 | Load data from **ROOT** files (e.g. LEM/PSI, MEG-format). *(Deferred — pending format confirmation.)* |
| IO-3 | Load column-delimited **ASCII/CSV** files with user-defined column mappings. |
| IO-4 | Load **PSI binary** (`.bin`) format. |
| IO-5 | Load **TRIUMF MUD** format. |
| IO-6 | Load **WiMDA `.wim`** format (initial primary format). |
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
| DM-5 | Apply **dead-time correction** to raw histograms. |
| DM-6 | Provide **rebinning** of time-domain data with variable bin widths. |
| DM-7 | Estimate and subtract **background** counts. |

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
