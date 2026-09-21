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
│   │   ├── background.py     # Background estimation & subtraction
│   │   ├── integral.py       # Time-integral (ALC) asymmetry reduction
│   │   ├── peakfit.py        # ALC baseline/peak/RF fitting helpers
│   │   ├── promote.py        # Promote a MaxEnt/count-domain result to a dataset
│   │   ├── rrf.py            # Rotating-reference-frame transform
│   │   └── t0.py             # t0 (time-zero) estimation
│   ├── fitting/        # Fitting engine
│   │   ├── __init__.py
│   │   ├── engine.py         # Fit driver: single-run & global
│   │   ├── composite.py      # Composite A(t) builder primitives
│   │   ├── fit_wizard.py     # Single-spectrum fit fingerprinting and model comparison
│   │   ├── cross_group_roles.py    # AIC/AICc/BIC Global-vs-Local role suggestion for cross-group fits
│   │   ├── grouped_time_domain.py  # Grouped-series fit engine
│   │   ├── joint.py          # fit_joint / suggest_shared_parameters — several series, different models, shared columns
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
│   │   ├── series.py         # FitSeries — ordered member series, its recipe + identity
│   │   ├── trend_state.py    # TrendState dataclass for Fit Parameters panel state
│   │   ├── global_fit_study.py # GlobalFitStudy — persisted named cross-group fit + staleness digest
│   │   ├── joint_fit.py      # JointFit — persisted joint-fit record (members, shared table, last result); runtime-only staleness
│   │   └── project_model.py  # ProjectModel — in-memory owner of representations + batches
│   ├── project/        # Project persistence
│   │   ├── __init__.py
│   │   └── schema.py         # load_project, save_project, migrate_to_current (v1–v13)
│   ├── workflow/       # Scriptable session façade (drives the engine for the CLI/agents)
│   │   ├── survey.py         # survey_folder: per-run metadata, calibration candidates, scans
│   │   ├── reduction.py      # ReductionSettings, reduce_run, estimate_alpha_for_run
│   │   └── workdir.py        # WorkDir: the on-disk session (manifest, survey, reduced spectra)
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
│   │   ├── fit_panel.py      # Thin backward-compat shim re-exporting panels/fit/ (see below)
│   │   ├── fit/              # Fit setup and results — split from the former fit_panel.py
│   │   │   ├── __init__.py   # Public API re-exports; dependency order: seeding → tab_base → single_tab/global_tab → panel
│   │   │   ├── seeding.py     # Phase/background/N0 seed math (leaf, no intra-package deps)
│   │   │   ├── tab_base.py    # FitParameterTable, FitTabBase, tie dialog, shared delegates/helpers
│   │   │   ├── single_tab.py  # SingleFitTab
│   │   │   ├── global_tab.py  # GlobalFitTab + batch-seeding constants
│   │   │   ├── recipe_inputs.py # build_recipe_engine_inputs() — one series' recipe -> engine inputs, shared by the Batch tab and the joint-fit window
│   │   │   └── panel.py       # FitPanel container (hosts the two tabs)
│   │   ├── fit_parameters_panel.py  # Parameter trending panel (pull-based, representation-aware)
│   │   ├── cross_group_fit_dialog.py     # Cross-group Global/Local/Fixed role dialog (+ Suggest roles)
│   │   ├── cross_group_config.py         # Widget-free config→global_fit_parameter_model bridge (shared by dialog + refit)
│   │   ├── global_fit_setup_dialog.py    # Explicit setup dialog: parameter, series, x-axis, editable group variable
│   │   ├── fit_function_builder.py  # Composite fit-function dialog
│   │   ├── fourier_panel.py  # Fourier analysis controls
│   │   ├── initial_values_dialog.py # Per-member initial-values editor for batch fits
│   │   └── log_panel.py      # Message / command log
│   ├── widgets/               # Shared foundational widgets (reused across panels/windows)
│   │   ├── axis_limits.py     # FloatLimitField, AxisLimitControls, AxisLimitPolicy — shared numeric limit fields + per-axis Auto/Hold resolver
│   │   ├── mpl_canvas.py      # create_canvas() — shared Figure/FigureCanvasQTAgg construction
│   │   ├── fit_run_controls.py # FitRunControls — shared Stop/Cancel button + progress bar
│   │   └── ...                # collapsible_section.py, dock_header.py, projection_chip_bar.py, etc.
│   ├── utils/                 # Shared GUI-layer helpers (no Qt widget state)
│   │   ├── export.py          # compile_gle() — shared GLE subprocess-invocation wrapper
│   │   ├── formatting.py      # format_param_label() and other display-formatting helpers
│   │   └── ...                # latex_renderer.py, series_scoring.py
│   ├── windows/              # Top-level analysis windows (there is no separate `dialogs/` package —
│   │   │                     # all modal/secondary windows, including simple dialogs, live here)
│   │   ├── wizard_base.py             # WizardWindowBase — shared TaskRunner/progress/staleness/cancel skeleton
│   │   ├── fit_wizard_window.py       # Guided single-spectrum fit wizard
│   │   ├── global_fit_wizard_window.py # Guided batch/global fit wizard
│   │   ├── multi_group_fit_window.py  # Grouped fit surface (Single + Batch tabs)
│   │   ├── global_parameter_fit_window.py # Cross-group parameter-vs-x fit window (studies sidebar, grid, exports)
│   │   ├── global_fit_window_helpers.py   # Widget-free value/table-export builders + CorrelationMatrixDialog for the fit window
│   │   ├── global_fit_compare_dialog.py   # Read-only side-by-side comparison of two studies (Δχ²ᵣ/ΔAIC)
│   │   ├── joint_fit_window.py        # JointFitWindow — compose recorded series into one coupled fit with shared parameters
│   │   ├── grouping_dialog.py         # Shared detector-grouping editor
│   │   └── ...                        # detector_layout_dialog.py, run_info_dialog.py, simulate_dialog.py, etc.
│   └── styles/                # BENCH design tokens, palette, and stylesheet (see below)
│
├── cli/                # Command-line interface over `core.workflow` (survey/alpha/reduce/info)
├── __init__.py
└── __main__.py         # `python -m asymmetry` entry point
```

### 3.1 Core (`asymmetry.core`)

The core has **zero** GUI dependencies. It depends only on the scientific Python stack and format-specific I/O libraries. This allows it to be used as a standalone library in scripts, notebooks, and CI pipelines.

### 3.2 GUI (`asymmetry.gui`)

The GUI is a separate, optional install target. It wraps the core API and provides interactive visualization, fitting dialogs, and logbook management.

#### Shared foundations (`gui/widgets/`, `gui/utils/`, `gui/windows/wizard_base.py`)

Several small widget/helper pairs that grew up independently in two or more
panels have been converged into shared modules. New axis-limit fields,
Matplotlib canvases, GLE export code, formatted parameter labels, run-control
buttons, or wizard windows should reuse these rather than re-implementing
them:

- `gui.widgets.axis_limits.FloatLimitField` / `AxisLimitControls` — the
  numeric limit-field widget (fit-range min/max, plot axis min/max).
- `gui.widgets.mpl_canvas.create_canvas()` — shared
  `Figure` + `FigureCanvasQTAgg` (+ optional `NavigationToolbar2QT`)
  construction.
- `gui.widgets.fit_run_controls.FitRunControls` — the Stop/Cancel button plus
  optional progress bar shared by the single and global fit tabs.
- `gui.utils.export.compile_gle()` — the GLE subprocess-invocation wrapper
  (`subprocess.run([gle, "-d", fmt, file], ...)`) shared by every GLE
  export/preview call site.
- `gui.utils.formatting.format_param_label()` — shared parameter-name display
  formatting.
- `gui.windows.wizard_base.WizardWindowBase` — the shared `TaskRunner` +
  progress UI + request-id staleness + cancel/closeEvent skeleton for guided
  fit-wizard windows; a subclass supplies its own tabs, worker task, and
  result-tab population.
- `gui.widgets.section_overflow_indicator.SectionOverflowIndicator` — a
  viewport-overlay pill that names the scroll sections currently hidden below
  the fold and scrolls the first one into view on click; recomputes from cheap
  geometry reads on scroll/resize plus an explicit `refresh()`.
- `gui.widgets.elided_label.ElidedLabel` — a QLabel that elides right instead
  of imposing its text width as a layout minimum (zero minimum width; full
  text as a tooltip while elided). Use it for any long one-line status text
  inside a width-constrained column so the text can never force a horizontal
  scrollbar.
- `gui.panels.fit/` — the fit-setup-and-results package (`FitPanel`,
  `FitTabBase`, `SingleFitTab`, `GlobalFitTab`, `FitParameterTable`, seeding
  helpers), split out of the former monolithic `fit_panel.py` (now a thin
  re-export shim kept for backward compatibility until remaining private
  imports migrate — see `docs/audit/shared-foundations/FOLLOW-UPS.md`).

#### BENCH design tokens

Color, spacing, and widget style constants live in `asymmetry.gui.styles.tokens`
and `asymmetry.gui.styles.widgets`. All raw color literals belong in `tokens.py`;
no other file may hardcode hex strings. Current accent families:

| Token family | Usage |
|---|---|
| `ACCENT` / `ACCENT_SOFT` / `ACCENT_SOFT2` | Blue — primary interactive elements (most buttons, selection highlights) |
| `ACCENT_RED` / `ACCENT_RED_SOFT` / `ACCENT_RED_SOFT2` | Red — FitSeries identity: series buttons, data-browser membership tint |

`style_group_state_button(button, state, *, palette="blue")` in `widgets.py`
accepts `palette="red"` to apply the red family instead of blue — used for
the series buttons in the Fit Parameters panel.

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
- **fit** (`FitSlot`) — the *Single tab's own fit* for this `(dataset,
  representation)` pair, and nothing else: model dict, fitted-parameter
  list, result summary, provenance (`"none"` / `"single"` / `"wizard"`), and
  an optional `ui_state` blob that restores the single-fit form verbatim. A
  batch, global, grouped or scan run never writes this slot — its results
  live only on the `FitSeries` that recorded them (see below), so a run
  belonging to several series still carries one exploratory single fit
  rather than whichever series wrote last. `"batch"`/`"global"` remain valid
  `provenance` values only so an older file's slot parses on its way through
  the v19→v20 migration, which drops such a slot entirely.
- **trend_state** — opaque dict persisting the user's axis/parameter selections
  in the Fit Parameters panel.

#### DataGroup and FitSeries: the group owns its fits

A `DataGroup` (`asymmetry.core.representation.group`) is a named, ordered
collection of run numbers with a `kind` discriminator: `"user"` for a group
the user named explicitly, `"auto"` for one minted automatically the first
time an ad-hoc run selection is batch- or global-fitted, so every batch fit
has an explicit owning group (no bare "batch of runs with no group" state is
representable). Renaming an `"auto"` group promotes it to `"user"`
(`ProjectModel.rename_data_group`). A run may belong to any number of groups —
multi-membership is a core-layer property, not just a GUI presentation detail;
the Data Browser's one-row-per-membership rendering (primary row plus marked
copy rows, `docs/reference/gui_usage.rst` § "Data groups") is the GUI's own
concern on top of it.

**The group owns its fits**: a run-membered `FitSeries` (`member_kind="runs"`)
carries a structural `group_id` back to its owning group, and its *effective*
membership is derived live — the group's `member_run_numbers` minus the
series' own `excluded_run_numbers` (`FitSeries.effective_members`) — rather
than snapshot at record time. Results remain a snapshot of what was actually fit
(`FitSeries.last_fitted_members`); when the live effective membership no
longer matches that snapshot the series is **stale** (`FitSeries.is_stale`),
surfaced as a `⚠` on the series' chip, cleared by re-running. A group with
zero owned series behaves exactly as before this model existed; a group can
also own several series at once (the same run collection fit two different
ways, or fit twice over two different fit ranges).

`source_group_id` is retained on `FitSeries` as a **legacy provenance**
field only — read for backward compatibility (older saves, and the v14→v15
migration's resolution of `group_id`) but no longer written. A series is
**frozen** (`group_id=None`) when it predates this model, or when its owning
group was deleted and its fits kept rather than deleted with it
(`ProjectModel.remove_data_group(..., orphan_series=True)`); a frozen
series' membership is the old snapshot semantics, unconditionally.
Detector-group series (`member_kind="groups"`, § 4.3.1) are untouched by any
of this — they are not `DataGroup`-based and keep their original frozen
semantics.

A `FitSeries` (`asymmetry.core.representation.series`) collects multiple
member fits — either across runs (`member_kind="runs"`) or across a run's
detector groups (`member_kind="groups"`) — into one trendable unit. Members
are keyed by integer: real run numbers for run series, synthetic negative keys
`-(source_run * 1000 + group_index)` for group series.

Key attributes: `recipe` (the Batch tab's setup that produced this series —
`parameters`, `fit_range`, `seeding`, `coadd` — normalised by
`normalise_recipe` so every reader finds every key), `canonical_model`,
`param_roles` (Global / Local / Fixed per physics parameter, now derived
from `recipe["parameters"]` at record time), `nuisance_params` (group-only,
always local), `results_by_run` (per-member summary dicts that drive the
trending panel), `trend_excluded_runs` (member keys ticked out of *this*
series' trend, the per-series successor to the old per-slot
`include_in_trend`), `group_id` (structural owning-group link, `None` for
frozen/detector-group series), `excluded_run_numbers` (per-series exclusions
from the owning group's membership), `last_fitted_members` (last-fitted
snapshot, for staleness), `label` (optional user-given name, `None` when
unset — the GUI renders a default `"<model> · <fit-range>[ · <group>]"`
fallback, `naming.default_series_label`). `recipe_identity()` returns a
canonical string of the normalised recipe plus the sorted effective member
set — the single thing a re-run compares to decide whether it replaces this
series or starts a new one (below). Divergence — a member's stored model
disagreeing with the series' canonical one — cannot occur under this model,
since a member never stores the series' fit at all; the concept, and its
`⚠` glyph, no longer exist.

`display_name(fallback: str) -> str` returns `self.label` when set, otherwise
the caller-supplied fallback — used in series buttons, chooser dialogs, and log
messages so that user-assigned labels appear everywhere consistently.

**Recording and identity (D3).** `MainWindow._record_fit_series` is the
shared choke point for every recording path (batch, global, grouped, scan).
It never writes a member `FitSlot`. It resolves what the new result is a
re-run *of* — `MainWindow._open_series_for`, checking the Batch tab's open
series first, then the representation's `active_series`, then the newest
series of that representation with an identical `recipe_identity()` — and
either folds the new result into that series in place (same `batch_id`,
same label) or records a brand-new `FitSeries` that becomes the open and
active one, leaving the original untouched. There is no load-time
superseding or deduplication pass any more (the old `_series_signature`,
`remove_superseded_batches` and `dedupe_batches` are gone): two series that
differ only in fit range are two entries, kept side by side, forever, until
one is explicitly deleted.

#### ProjectModel

`ProjectModel` (`asymmetry.core.representation.project_model`) is the
in-memory owner of all representations, series, **and data groups** for the
active project — `ProjectModel.data_groups` is the single canonical registry
(keyed by `group_id`); there is no second, GUI-owned copy to keep in sync.
`DataBrowserPanel` holds a reference to it and is a view/controller over it —
the browser's own `collapsed` flag is the only group-related state it still
owns itself, persisted in `browser_state` (view state, not group identity).
`ProjectModel` provides:

- **Group mutation API** — `create_data_group` (mints or reuses a `group_id`,
  never strips a run from a group it already belongs to), `rename_data_group`
  (auto → user promotion), `set_data_group_members`, `find_auto_group`
  (identical-member-set reuse for D3), and `remove_data_group(group_id, *,
  orphan_series)` (D7's delete-fits/keep-fits choice — deletes or freezes the
  group's owned series). `series_for_group(group_id)` resolves the owned
  series by structural `group_id` with a legacy `source_group_id` fallback for
  frozen series that predate the structural link.
- **`active_series: dict[str, str]`** — representation value → `batch_id` of
  its active series (D5), persisted top-level as `active_series`.
  `set_active_series(rep_type, batch_id | None)` writes it; pressing a
  Parameters chip, opening a series in the Batch tab, or picking it from the
  plot's Fits menu all call it, so the three surfaces always agree on which
  series is active. `active_series_id(rep_type)` reads it back.
- `set_trend_excluded(batch_id, run_number, excluded)` — toggles one
  member's inclusion in *that series'* trend (`trend_excluded_runs`);
  exclusion is per series, so a run dropped from one series' trend still
  trends in every other series that contains it.
- `rename_batch(batch_id, label)` — sets `FitSeries.label` (empty/whitespace →
  `None`, reverting to the positional fallback); returns `True` on success.
- `remove_batch(batch_id)` — pops and returns the `FitSeries`, and clears any
  `active_series` entry pointing at it (D6). Touches nothing else: with no
  member `FitSlot`s to clear and no other series to re-evaluate, deleting a
  series only ever affects that series and its own trend. Unknown `batch_id`
  returns `None`.

Series identity resolution (`recipe_identity`, the open/active/newest-match
walk) lives in `MainWindow` rather than `ProjectModel` — it needs the Batch
tab's notion of "currently open", which is GUI state — so `ProjectModel`
itself no longer signature-matches or deduplicates series on its own.

#### Fit Parameters panel (pull model)

The `FitParametersPanel` operates on a **pull** model: whenever a fit
completes or the active representation changes, `MainWindow._refresh_trend_panel`
reads the relevant `FitSeries` from `ProjectModel`, builds per-member row dicts
(using `_frequency_spectra_by_run` for FFT series), and calls
`FitParametersPanel.load_representation_series`. The panel shows a **"Showing:"**
label indicating the active representation and a **chip rail** — one chip
per `FitSeries` for the active representation, styled with the red accent
(`ACCENT_RED`) so that FitSeries identity reads red consistently across the
UI, sectioned by the data group that owns each series (a swatch-and-name
header per group; a header-less "Standalone" section for group-less series;
no header row at all when every series is standalone).

Pressing a chip makes that series **active** (`ProjectModel.set_active_series`)
and emits `series_selection_changed(batch_id)`; while the Parameters dock is
**visible**, the main window forwards this to
`DataBrowserPanel.set_highlighted_runs`, tinting member dataset rows with a red
tint (`ACCENT_RED_SOFT`) — a **decorative** highlight that never touches the
real Qt selection, clearing when the dock hides and restoring when it shows
again (`QDockWidget.visibilityChanged`). Double-clicking a chip opens it in
the Batch tab (`MainWindow._open_series_in_batch_tab`), the same route the
chip menu's first entry takes.

**Context menu on a chip** (right-click), in order:

- **Open in Batch tab** — emits `series_open_requested(batch_id)`; the main
  window resolves the owning group's members, crops them to the series' own
  fit-range window, and hands the whole recipe to `FitPanel.open_series`,
  making it the open and active series.
- **Duplicate…** — emits `series_duplicate_requested(batch_id)`: opens the
  series, then calls `FitPanel.duplicate_open_series` to copy its recipe and
  members into an untouched draft beside it.
- **Rename…** — opens an input dialog prefilled with the current label; on
  accept emits `series_rename_requested(batch_id, new_label)`. The main window
  calls `rename_batch` and re-renders the panel.
- **Select members in browser** — emits `series_select_members_requested(batch_id)`.
  The main window resolves the member run numbers and calls
  `DataBrowserPanel.select_runs`, performing a true Qt selection that drives
  the normal `selection_changed` pathway. The decorative tint is unaffected.
- **Show fit overlay** — a checkable action, ticked and disabled when the
  chip is already active; otherwise it makes the series active exactly as
  pressing the chip would (the menu's route to the same call).
- **Delete series…** — confirms with a `QMessageBox` (`confirm_series_delete`
  in `gui/widgets/series_dialogs.py`, naming what survives), then emits
  `series_delete_requested(batch_id)`. The main window calls `remove_batch`
  and clears only that series' plot overlays (`PlotPanel.clear_fits_for_series`)
  — every other series' and every run's own single fit are untouched — and
  refreshes the panel; a surviving series retains its highlight, and the
  empty case clears the tint.

**Overlay keying and the plot toolbar's Fits menu.** `PlotPanel` keys every
fit curve by `(run, axis, fit_id)`, where `fit_id` is a `FitSeries.batch_id`
or the literal `"single"` for a run's own exploratory fit — never by run
alone — so a run belonging to several series keeps every one of their
curves simultaneously rather than the last writer overwriting the rest. The
active series (`PlotPanel.set_active_fit_id`, driven by the same
`ProjectModel.set_active_series` pointer) draws on every run it covers in
the fit accent colour; a compact **Fits** button at the right of the plot's
own toolbar row (reading `Fits · N` for a run holding N fits, hidden for a
run holding none so it never widens an empty panel) opens a menu over every fit id stored for the currently
displayed run. The menu is rebuilt on `aboutToShow`, never per event, and it
alone carries the fit names — the button's width is fixed, so a series' name
never enters the panel's minimum width. Ticking an entry toggles that curve
on or off (`shown_fits_by_run`, transient view state, never persisted or
recorded); the `●` marks the active series but the menu never sets it —
that pointer moves only from the Batch tab or a Parameters chip.

#### DataBrowserPanel — decorative highlight vs true selection

`set_highlighted_runs(run_numbers)` paints rows with `ACCENT_RED_SOFT` without
touching the Qt selection model. It is the sole mechanism used by the
visibility-gated FitSeries highlight.

`select_runs(run_numbers)` performs a real Qt multi-row selection (using the
existing `_restore_selection_by_keys` helper), scrolling to the first matched
row, and fires `selection_changed` through the normal pathway. It is used by
the "Select members in browser" context-menu action and never alters the tint.

### 3.6 Grouping Profiles

Detector grouping used to be a full payload copied onto every `Run` —
detector groups, `alpha`, deadtime/background modes, *and* per-run facts like
`t0` and the good-bin window all in one dict — so recalibrating `alpha` after
loading a fifth run of a series meant editing it a fifth time, with no
guardrail against two runs of the same series silently drifting apart.

**Model.** `asymmetry.core.project.profiles` splits the payload into
*shareable* settings (detector groups, forward/backward assignment, and
policy objects — `AlphaPolicy`, `BetaPolicy`, `DeadtimePolicy`,
`BackgroundPolicy`, `T0Policy`, each a mode plus whatever value the mode
needs) and *per-run* facts (`t0_bin`, `t0_time_us`, `t0_source`, good-bin
window, per-detector file deadtime, period tables) that always come from the
run itself. A `GroupingProfile` holds only the shareable half, is named, and
belongs to the project rather than to any one run.

`T0Policy.mode` (`"from_file"`, `"manual"`, `"auto_detect"`) is an explicit
stored field, never inferred by comparing a stored value against a
recomputed one — that inference (`t0_bin` vs. the max header t0 over *all*
detectors, rather than just the analysis groups) is what mislabelled fresh
drafts as Manual before `docs/plans/t0-determination.md`. Manual is stored
as a signed *offset* in bins from each run's own file t0, so one profile
shifts a whole set of runs consistently however their headers differ; the
per-detector bins every consumer aligns on — reduction, grouped Fourier,
MaxEnt, count-domain fits, the deadtime window, the plot mask — come from a
single resolver, `effective_detector_t0_bins()`
(`core/transform/t0.py`), enforced by a structural rule so a new consumer
cannot silently re-derive alignment from the file values alone. Where a
loader records a continuous t0 (ISIS `time_zero`, MusrRoot's `Double_t`
`Time Zero Bin`) alongside the integer `t0_bin`, every time axis is stamped
from the exact value — `(k + ½)·w − t0`, bin *k*'s centre minus t0 — rather
than the coarser integer-bin arithmetic; a file without one falls back to
the centre of `t0_bin` itself, so the two conventions agree exactly when
there is nothing to gain from the finer one.

**Resolution.** A profile applies to every run whose *fingerprint* —
`(instrument, histogram_count)` — matches. Each fingerprint has exactly one
*active* profile at a time (a project may hold several saved profiles per
fingerprint); a run inherits its fingerprint's active profile automatically
unless explicitly *released* into a per-run override.
`resolve_effective_grouping(profile, run)` merges the two halves back into
exactly the `run.grouping` payload shape used before profiles existed —
computing a policy-derived value where needed (a `per_run_estimate` alpha, or
`from_file` deadtime) — so nothing downstream of `run.grouping` (reduction,
fitting, export) needed to change.

**Schema v12.** The project file gained a top-level `grouping_profiles` list.
Each dataset either names a `profile` (inherits), carries its own
`grouping_overrides` (released), or has neither (inherits its fingerprint's
active profile). `_migrate_v11_to_v12` (`core/project/schema.py`) migrates
older projects automatically: datasets are bucketed by fingerprint, and
within each bucket, identical shareable settings collapse to one active
"Default (\<instrument\>)" profile while genuinely divergent runs keep their
own `grouping_overrides` untouched — the migration is additive and never
discards a setting it cannot faithfully represent as a profile.

**Reduction chokepoint.** `asymmetry.core.transform.reduce.reduce_grouped_asymmetry`
is the single function that turns a run's histograms into a forward/backward
asymmetry curve: deadtime correction, forward/backward grouping onto a common
`t0`, optional background subtraction, then the counts-then-ratio asymmetry.
It was lifted out of `MainWindow._reduce_grouped_histograms_to_asymmetry` so
that the GUI's live reduction and the Grouping window's live preview pane
share one numerics implementation instead of two that could drift apart; a
pinned test (`tests/gui/test_grouping_preview_pane.py`) keeps the preview
bit-identical to the original `MainWindow` output.

**GUI.** `gui/windows/grouping/` (`dialog.py` plus `alpha_section.py`,
`deadtime_section.py`, `background_section.py`, `correction_card.py`,
`scope_panel.py`, `preview_pane.py`, `profile_bridge.py`) replaced the old
single grouping dialog
with a profile editor: a draft `GroupingProfile` edited in place, a scope panel
for release/reattach, and a debounced live-preview pane whose reduction runs on a
`TaskRunner` worker thread per the threading invariant in `AGENTS.md`. The right
pane is tabless: a full-width pipeline strip over two side-by-side scroll columns
— "Grouping and timing" (`_grouping_scroll`) and "Corrections"
(`_corrections_scroll`) — with the compare pager and the single preview pane
pinned below both (the preview reduces from the draft's widget state, not from a
focused column). The deadtime, background and single-α controls are inline
`DeadtimeSectionWidget` / `BackgroundSectionWidget` / `AlphaSectionWidget` in the
Corrections column (all three standalone modals were retired), each wrapped in a
collapsible `CorrectionCard` (`correction_card.py` — grouping-specific, richer
than the shared `PanelSection`: its header carries the live status summary and
the compare indicator). Cards open expanded iff their stage is active, remember
user toggles only for the dialog's lifetime (no QSettings), and auto-expand when
a reseed activates their stage; a stale calibrated
α is flagged by a `" · stale"` suffix (and re-estimation tooltip) on the pipeline
α chip plus a warn-tint on the α card header. The shared α-estimate worker and
its run-combo / request builders live in
`alpha_section.py`; the per-projection *vector* α table (`dialog.py`) lives in the
same Corrections-column α area and drives its per-axis and "Estimate All α"
estimates inline through that same worker on the dialog's own `TaskRunner`,
serialised one axis at a time. Compare focus is driven by the pipeline chips and
the compare pager (the per-section checkboxes and the old compound "vs raw"
checkbox are both retired — `preview_pane.py` module docstring, D6 of
`docs/plans/grouping-preview.md`); focusing drives `_PreviewRequest.compare_stage`
(`"deadtime"`/`"background"`/`"alpha"`/`"beta"`), which `preview_pane._run_reduction`
renders as a **ghost** of that stage removed, drawn *on top* of the solid
full-pipeline curve in the stage's identity colour (`preview_pane._GHOST_COLORS`) —
the solid is never degraded, so the α compare's residual-⟨A⟩ acceptance number is
always read off the fully-corrected reduction, and the preview's y-axis follows
the solid alone (a fixed top-left caption names both curves, so an off-scale
ghost is still named without a legend). One reduction now feeds two views —
the asymmetry over the good window, and a `Counts` view of the corrected
forward/backward spectra over the full histogram with t0, the good window and
the background level marked (`GroupingPreviewPane.set_view`, `CountsCurves`) —
switching between them is a redraw, never a recompute. The compare is preview-only
(`compare_stage` never reaches `_current_grouping_payload`); the α compare
auto-focuses on calibration and is unavailable in vector mode (the
per-projection table owns α there). A
`Deadtime → group → Background → α` pipeline strip across the top of the right
pane shows each stage's live one-line summary and clicking a chip focuses that
stage's compare (the same `_compare_stage`), expands its card, and scrolls it
into view, making the reduction order visible.

### 3.7 Global Parameter Fit Studies

The cross-group global parameter fit — "fit a trend parameter such as
λ(B) at several temperatures with some parameters shared" (Wu *et al.*,
arXiv:2502.00130) — is a first-class, persisted, and named entity rather than a
single ephemeral result.

**Core.** `asymmetry.core.fitting.parameter_models.global_fit_parameter_model`
is the fitter: one `iminuit` problem over `N` `ParameterGroupData` snapshots,
each model parameter assigned a role of Global (one shared value), Local (one
value per group, named `{param}_{group}`), or Fixed. `CrossGroupFitResult`
carries the shared/local/fixed `ParameterSet`s plus per-group χ², n-points, and
the free-global correlation matrix, with canonical `to_dict()`/`from_dict()`.
`asymmetry.core.fitting.cross_group_roles.suggest_cross_group_roles` recommends
Global-vs-Local roles by a bounded (not `3^k`), deterministic AIC/AICc/BIC
search — an all-global baseline, single "flip to local" refits, then a greedy
beam accumulation — reusing the vocabulary of `global_fit_wizard`
(`score_delta`/`total_variation`/`roughness`/`rationale`).
`asymmetry.core.representation.global_fit_study.GlobalFitStudy` is the persisted
entity: a `study_id`, name, parameter/x/group-variable keys and labels, the
`ParameterGroupData` snapshot, the model, the dialog config (roles, bounds,
windows, error mode), the `CrossGroupFitResult`, and an `input_digest`
(`compute_group_input_digest`) used to detect that a study's inputs have drifted
from the live trend data (**staleness** is recomputed at load time, never
stored). Project schema **v13** adds a top-level `global_fit_studies` list;
`_migrate_v12_to_v13` is purely additive (version bump plus an empty default
for the new list). A legacy single-slot `last_cross_group_fit` payload is
lifted into one named study on the GUI side at project load
(`MainWindow._restore_global_fit_studies`, via
`study_from_legacy_cross_group_payload`), and old projects still open.

**GUI workflow.** Analysis ▸ *New global parameter fit…* (or the trend panel's
*Global fit (N groups)…* button when ≥2 group buttons are selected) opens
`gui/panels/global_fit_setup_dialog.py::GlobalFitSetupDialog` — parameter,
series checklist, x-axis, and an editable **group variable** with per-group
values (prefilled from inference but overridable, incl. custom columns).
*Continue* opens `gui/panels/cross_group_fit_dialog.py::CrossGroupFitDialog`,
which carries the per-parameter Global/Local/Fixed role table and a *Suggest
roles* button that runs `suggest_cross_group_roles` off-thread and applies the
recommendation. `gui/panels/cross_group_config.py` is the widget-free
config→`global_fit_parameter_model` bridge shared by the dialog and the
off-thread **Refit** path. On accept, `MainWindow._on_cross_group_fit_completed`
creates or updates (in place, by deterministic batch id) a `GlobalFitStudy` in
the MainWindow-owned registry (`_global_fit_studies`), persisted via schema v13,
and displays it in `gui/windows/global_parameter_fit_window.py`. That window
gained a studies sidebar (rename/duplicate/delete, stale badges), a grid Fig-3
pane with per-panel χ²ᵣ chips and a component legend, a quality bar, a
residual/pull toggle, a correlation-matrix view, table exports (TSV/CSV/LaTeX),
an *Edit fit…* re-entry, custom-x labels, and a stale banner with *Refit*.
`gui/windows/global_fit_window_helpers.py` holds its widget-free value/export
builders and the `CorrelationMatrixDialog`;
`gui/windows/global_fit_compare_dialog.py` is a read-only side-by-side
comparison of two same-parameter studies (overlaid curves, Δχ²ᵣ/ΔAIC/AICc/BIC).
The `_record_model_fit_results_series` bridge (each study result recorded as a
trendable results series) and the `modelfit-<digest>` decoration keying are
preserved.

### 3.8 Joint Fits

A **joint fit** (`docs/plans/joint-fit.md`) generalises a global fit from "one
model over several runs" to "several models over several series, coupled by
named shared columns". It introduces a third scope on top of the existing
Local (per run) / Global (per series) roles: **Shared**, held equal across two
or more series. "Linked" (equality link groups) and "Global Parameter Fit"
(§3.7) keep their existing, unrelated meanings.

**Core.** `core/fitting/engine.py`'s `_build_coupled_global_problem` is
generalised to several *blocks* (one per series: datasets, model function,
global/local names, initial parameter sets, fit range, local group key) plus a
shared column map; `global_fit` is now the one-block, no-shared caller through
the same builder, byte-for-byte unchanged in fitted values, χ² and result
packing. `core/fitting/joint.py` is the Qt-free, user-facing seam:
`JointSeriesProblem` (one series' contribution), `SharedParameter` (a shared
column's members, seed and bounds), `JointFitResult`, `fit_joint` (raises
rather than guards on overlap, a non-Global member, a tie, or
`strategy="profiled"`), and `suggest_shared_parameters` — two-tier
autodetection (`"exact"`: same full name, same unit, occurring once per model,
Global everywhere; `"candidate"`: same base name or component type, one
instance per model) built on `CompositeModel.parameter_identities`.
`core/representation/joint_fit.py::JointFit` is the persisted record (member
`batch_id`s in tick order, the shared table, the last result summary),
mirroring `GlobalFitStudy`'s tolerant `to_dict`/`from_dict` and runtime-only
(never stored) `stale_reason`. `FitSeries` gains optional `joint_fit_id` and
`shared_params` stamps, ignored by `recipe_identity()`; `ProjectModel` owns
`joint_fits`, cascading a member's removal (drop it; delete a joint fit left
with fewer than two members) and clearing stamps on `remove_joint_fit`.
Project schema **v22** adds the top-level `joint_fits` list, purely additive
(`_migrate_v21_to_v22`).

**GUI workflow.** `gui/panels/fit/recipe_inputs.py::build_recipe_engine_inputs`
extracts the recipe→engine-inputs step `GlobalFitTab._run_global_fit` already
did, so a joint run resolves a member's seeds, bounds and fit range exactly as
a solo Batch-tab run of that series would — the joint-fit window edits no
recipe. Analysis ▸ *New joint fit…* (and the *Joint fits* registry submenu,
mirroring *Global parameter fits*) opens `gui/windows/joint_fit_window.py::JointFitWindow`:
a **Series** picker (every eligible series of the active representation, an
overlapping or ineligible row disabled with its reason on the tooltip), a
**Shared parameters** table (`apply_param_table_style`, *Suggest*, *Add shared
parameter…*, *Remove*), and a footer (`FitRunControls`, a results card, a
stale banner with *Refit*, *Delete joint fit…*). The fit runs on `TaskRunner`;
a member's fitted curve is evaluated in the same worker call (never the GUI
thread) and carried back on the result. `MainWindow._on_joint_fit_completed`
writes each member's `results_by_run` in place under its existing `batch_id`
(no new series — the recipe did not change, only the constraint), stamps both
fields, and records or updates the `JointFit`; `_record_fit_series` clears a
series' stamps on every *solo* re-run, which is what makes a joint fit stale
the moment one member is re-run on its own. The Fit Parameters panel
(`fit_parameters_panel.py`) groups a joint fit's members into their own
chip-rail section ahead of the data-group sections (its header tooltip lists
every member's full name); a parameter the joint fit shares is, in every
member, still a Global one — no chip, no card, no flat line — and the
existing "held constant" hint's per-parameter entry gains a suffix naming
the joint fit it is shared across. The window's own default label
(`naming.joint_member_name`, `naming.default_joint_fit_label`) is built from
each member's short name (its own label, else its data group's name, else
its model label), falling back to that member's full fallback name on a
collision, so it stays short even for a joint fit of several members.

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

Binning follows the same counts-first order in every mode (fixed bunching
included): counts are summed onto the output bins and the asymmetry formed
per output bin (`core/transform/rebin.py::binned_fb_asymmetry`) — the
counts-then-ratio convention WiMDA, musrfit, and Mantid all use. Rebinning
an already-formed asymmetry curve (`rebin()`) is reserved for data without
raw histograms; on sparse data it lets one-sided raw bins' σ = 1
no-information sentinels inflate merged error bars. The audited
cross-program conventions are recorded in
[docs/porting/comparison-matrix.md](porting/comparison-matrix.md)
("Reduction numerics").

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

**Registry naming.** Fit functions live in three name-keyed registries —
`COMPONENTS` (composite time-domain basis functions), `MODELS` (single-channel
models), and `PARAMETER_MODEL_COMPONENTS` (parameter-trend components) — that
span different analysis domains, so a bare name does not identify which registry
(or domain) a function belongs to. The `python-user-functions` registration
facade (`core/fitting/user_functions.py`) formalises this: every registration
through it requires an explicit domain tag and a name unique across **all
three** registries, so a facade-registered name maps to exactly one
(registry, domain) pair. Code-path registrations share one insertion core
(`core/fitting/registration.py`, also used by `models._register`); the single
legacy cross-registry collision (`Constant`) stays grandfathered behind the
kind-aware documentation lookup. User-registered definitions carry a `user`
flag — provenance badges and docs-test exemptions key off the flag, never
name lists.

Authoring is not limited to hand-written files: `core/fitting/
user_function_authoring.py` turns a name/description/formula draft into the
same generated plugin-file shape and registers it through this facade, and
`gui/windows/new_user_function_dialog.py::NewUserFunctionDialog` is the
in-GUI front end both function builders launch, so a function created in the
GUI is indistinguishable from one dropped in `~/.asymmetry/user_functions/`
by hand.

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
  in-memory representation + series state, including these group series'
  staleness (`FitSeries.is_stale`, § 3.5) against their owning group's live
  membership.
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
- Decimation is viewport-aware: the x window is resolved through
  `asymmetry.gui.widgets.axis_limits.AxisLimitPolicy` before anything is
  drawn, so decimation and the y bounds are both computed for the window the
  canvas is about to show in one pass — there is no deferred viewport redraw
  and no "one view behind" refresh. The panel re-renders when the user pans,
  zooms, presses Auto X/Auto Y, or edits axis limits directly.
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
