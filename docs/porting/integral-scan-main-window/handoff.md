# Integral (ALC) scan → main window — exploration & handoff

Branch: `feat/integral-scan-main-window` (off `main`).

This document captures the current state of the integral (ALC) field-scan
representation and a handoff prompt for the implementing agent. **No code was
written in the exploration session** — the next step is a design discussion with
the user before implementation.

## Motivation

Today, selecting the "Integral scan" representation keeps the central plot on the
time-domain FB asymmetry, and the actual scan (integral asymmetry vs
field/temperature/run) is confined to the right-hand `ALCScanView` in the
Parameters dock. In scan mode the user does not interact with the per-run time
spectra — they are just integrated — so the scan itself should occupy the main
window. Baseline/Peaks/RF analysis controls should stay accessible (dock is fine).

## Current state

### Core layer (clean — should not need changes)

- `FieldScan` dataclass + `build_field_scan()` / `integrate_run()` /
  `integrate_asymmetry()` in `src/asymmetry/core/transform/integral.py` turn
  per-run time-domain spectra into one integrated point each
  (`A = (F − αB)/(F + αB)` summed over the window), then assemble/sort into a
  scan vs field/temperature/run.
- Fitting: `fit_scan_baseline()` (Linear/Cubic/… over non-resonant regions) →
  subtract → `fit_scan_model()` (GaussianLCR/LorentzianLCR/RFResonanceMuP) in
  `src/asymmetry/core/fitting/field_scan.py`.
- Persistence: the scan is a model-less `FitSeries`, with baseline/peaks/view
  state in `FitSeries.extra` (schema v10). No changes expected.

### GUI layer (where the move happens)

- `src/asymmetry/gui/panels/plot_workspace_panel.py` — the central
  `QStackedWidget` holds **exactly two** panels (`_time_panel`,
  `_frequency_panel`). `integral_scan` is registered as a *time* view
  (`_ALWAYS_ENABLED_VIEWS`, `_PRIMARY_TIME_VIEWS`).
- `src/asymmetry/gui/panels/plot_panel.py` (~1606) — `integral_scan` is
  normalized to the `fb_asymmetry` time-view mode, so the centre shows time
  spectra with the draggable fit-range acting as the integration window.
- `src/asymmetry/gui/panels/alc_panel.py` — `ALCFitPanel` (build controls →
  Fit dock) and `ALCScanView` (scan canvas **+** x/derivative controls **+**
  Baseline/Peaks/RF groups, all one widget → Parameters dock). The scan canvas
  hosts draggable baseline-region edges and peak centres.
- `src/asymmetry/gui/mainwindow.py` — `_alc_mode` (~4948),
  `_sync_fit_dock_mode` (~4920), `_INSPECTOR_DOMAIN_CONFIG` (~4962),
  `_on_plot_workspace_view_changed` (~8160), `_render_alc_scan` (~9199),
  `self._alc_scan_points`, and the fit-range↔integration-window sync
  (`fit_range_edit_committed`, `set_fit_range_display`, ~2206).

### Key wrinkle

`ALCScanView` is one widget bundling three things — the scan's matplotlib canvas
(with draggable baseline-region edges and peak centres), the x-axis/derivative
view controls, and the Baseline/Peaks/RF analysis groups (in a scroll area).
Moving "the plot" to the centre means splitting this widget.

### The real design tension

The integration window *is* the central time plot's draggable fit-range. If the
time spectra leave the centre entirely, "how does the user set/see the
integration window?" needs an answer (spinboxes only? a retained mini time-plot?
a split view?). This is the one interaction that currently happens on the time
plot.

## Handoff prompt for the implementing agent

> **Task: Move the integral (ALC) field-scan plot into the main window plot area
> (design + discuss before implementing).**
>
> You are working in the Asymmetry repo on branch `feat/integral-scan-main-window`
> (already created, based on `main`). Use the project venv `.venv/bin/python` and
> the harness (`python tools/harness.py ...`). Do **not** start coding until
> you've explored options and agreed a direction with the user via
> `AskUserQuestion`/discussion. Follow the repo's study-first ethos.
>
> **Goal.** Today, when the user selects the "Integral scan" representation, the
> central plot keeps showing the time-domain FB asymmetry, and the actual scan
> (integral asymmetry vs field/temperature/run) is confined to the right-hand
> `ALCScanView` in the Parameters dock. The user wants the scan itself to occupy
> the **main window**, because in scan mode they don't interact with the per-run
> time spectra — those are just integrated. Baseline/Peaks/RF analysis controls
> should stay accessible (dock is fine).
>
> **What you must NOT break:** the core layer is clean — `FieldScan`,
> `build_field_scan`, `fit_scan_baseline/fit_scan_model` (in
> `core/transform/integral.py` and `core/fitting/field_scan.py`) and the
> `FitSeries.extra` persistence (schema v10) should not need changes. This is a
> GUI restructuring.
>
> **Key files to study first:**
> - `src/asymmetry/gui/panels/plot_workspace_panel.py` — the central
>   `QStackedWidget` (only `_time_panel` + `_frequency_panel` today;
>   `integral_scan` is a *time* view via
>   `_ALWAYS_ENABLED_VIEWS`/`_PRIMARY_TIME_VIEWS`).
> - `src/asymmetry/gui/panels/plot_panel.py:~1606` — where `integral_scan` is
>   normalized to the `fb_asymmetry` time-view mode.
> - `src/asymmetry/gui/panels/alc_panel.py` — `ALCFitPanel` (build controls → Fit
>   dock) and `ALCScanView` (scan canvas **+** x/derivative controls **+**
>   Baseline/Peaks/RF groups, all one widget → Parameters dock). Note the canvas
>   hosts draggable baseline-region edges and peak centres.
> - `src/asymmetry/gui/mainwindow.py` — `_alc_mode` (~4948),
>   `_sync_fit_dock_mode` (~4920), `_INSPECTOR_DOMAIN_CONFIG` (~4962),
>   `_on_plot_workspace_view_changed` (~8160), `_render_alc_scan` (~9199),
>   `_alc_scan_points`, and the fit-range↔integration-window sync
>   (`fit_range_edit_committed`, `set_fit_range_display`, ~2206).
>
> **Design questions to resolve WITH the user before coding — present concrete
> options and trade-offs:**
> 1. **Fate of the time spectra in scan mode.** Fully replace the centre with the
>    scan plot, a split/secondary view, or a small retained time-domain preview?
>    (Couples to #2.)
> 2. **Integration-window selection.** The integration window currently *is* the
>    draggable fit-range on the central time plot. If the time plot leaves the
>    centre, how does the user set and visualize the window — spinboxes only, a
>    retained mini time-plot, drag handles elsewhere? This is the crux.
> 3. **Splitting `ALCScanView`.** The scan canvas and the Baseline/Peaks/RF
>    analysis controls are one widget; the plan likely splits the canvas (→
>    centre) from the controls (→ dock). Decide how the draggable
>    baseline-region/peak-centre interactions and the x-axis/derivative selectors
>    travel with the plot.
> 4. **Centre implementation.** Add a third panel (a dedicated ALC scan panel) to
>    the workspace `QStackedWidget`, vs. teaching `PlotPanel` to render scans as a
>    first-class time-view mode. Weigh reuse of the existing `ALCScanView`
>    rendering (`show_scan`/`_render_plot`) against a clean new panel.
> 5. **Dock layout in scan mode.** With the plot gone from the dock, what stays
>    there — just the build panel + analysis groups? Update
>    `_INSPECTOR_DOMAIN_CONFIG`/`_sync_fit_dock_mode` accordingly.
> 6. **Persistence & project round-trip.** Ensure saved ALC state
>    (`FitSeries.extra`) still restores correctly with the relocated widgets
>    (`_restore_alc_*` paths).
> 7. **Which spectra go into the scan (explore this thoroughly with the user).**
>    Today membership is *implicit*: the scan integrates exactly the data-browser
>    multi-selection. `MainWindow._update_selected_datasets` (~10726) pushes the
>    browser's `get_selected_datasets()` into the fit panel; **Build Scan**
>    (`_on_scan_requested`, ~9092) reads `self._fit_panel.batch_datasets()` and
>    `build_field_scan(order_key="run")` integrates every run in it. A run is
>    dropped only if it can't be integrated (no grouping) — or, in RF mode, isn't
>    two-period — and exclusions go to the log, not the UI. There is **no
>    scan-specific selection surface**: no per-point include/exclude, no
>    click-to-drop an outlier, no field/temperature range gate. Note
>    `build_field_scan` already accepts an unused `filter=` callback in core, which
>    could power any of these without touching the reduction. Explore more elegant
>    options with the user, e.g.: click a scan point to exclude/restore it;
>    a checkbox run-list within the scan panel decoupled from the browser; x-range
>    gating on the chosen axis; or simply surfacing which runs contribute (and why
>    others were dropped) in the UI rather than the log. Decide how membership
>    persists in `FitSeries` alongside the relocation.
>
> **Deliverable of your first pass:** a short options write-up + a
> recommendation, confirmed with the user via `AskUserQuestion`, then a concrete
> implementation plan. Only implement after the user picks a direction. Validate
> with `python tools/harness.py validate` and `gui-smoke`; add/adjust tests beside
> the behaviour.
>
> **Before pushing / opening a PR:** run a code review of the change and apply the
> fixes — invoke `/code-review` at `high` effort (or run the review workflow) and
> address the findings in the working tree, then re-validate. Do **not** push or
> open a PR until the review is clean and the user has confirmed the layout and
> interactions live in the GUI. Pushing burns the user's CI minutes, so only push
> when the user explicitly asks.
