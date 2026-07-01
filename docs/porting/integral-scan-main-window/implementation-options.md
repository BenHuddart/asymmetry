# Implementation options

The meat of this study. These are **to be resolved with the user** (via
`AskUserQuestion`) before implementation. Each question lists candidate approaches
and trade-offs.

## Q1 — Fate of the time spectra in scan mode

- **A. Fully replace the centre with the scan plot.** Cleanest; matches
  Mantid/WiMDA. Requires answering Q2 (window selection moves off the time plot).
- **B. Split view** (scan plot dominant + small time plot). Preserves drag-to-set
  window; busier layout.
- **C. Small retained time-domain preview** (e.g. a thumbnail strip). Compromise;
  more layout work.

## Q2 — Integration-window selection (the crux)

Today the window *is* the draggable fit-range on the central time plot.

- **A. Spinboxes only** (`ALCFitPanel` already has them). Simplest; loses direct
  manipulation.
- **B. Keep a small time plot** with the draggable range (couples to Q1-B/C).
- **C. Move drag handles elsewhere** (e.g. onto the scan or a dedicated strip).
  Least conventional.

## Q3 — Splitting `ALCScanView`

`ALCScanView` bundles the scan canvas **+** x-axis/derivative selectors **+**
Baseline/Peaks/RF groups in one widget. Moving the canvas to the centre means:

- Decide whether the x-axis/derivative selectors travel with the canvas (centre)
  or stay with the analysis controls (dock).
- Preserve the canvas's draggable baseline-region edges and peak centres
  (`_on_canvas_press/motion/release`, `_handle_artists`) after relocation.
- Keep `MainWindow._render_alc_scan()` → `show_scan()` feeding the relocated
  canvas from `self._alc_scan_points`.

## Q4 — Centre implementation strategy

- **A. Add a third panel** (dedicated ALC scan panel) to
  `PlotWorkspacePanel`'s `QStackedWidget`, routed when `active_view() ==
  "integral_scan"`. Clean separation; touches the view/stack plumbing and the
  `integral_scan → fb_asymmetry` normalization.
- **B. Teach `PlotPanel`** to render scans as a first-class time-view mode.
  Fewer new widgets; risks entangling scan rendering with time-domain code.

Reuse of the existing `ALCScanView` rendering (`show_scan`/`_render_plot`) vs a
clean new panel is the sub-decision here.

## Q5 — Dock layout in scan mode

With the plot gone from the dock, the right side holds the build panel + analysis
groups. Update `_INSPECTOR_DOMAIN_CONFIG` (mainwindow.py:~4962) and
`_sync_fit_dock_mode` (~4920) accordingly.

## Q6 — Persistence & project round-trip

Ensure saved ALC state (`FitSeries.extra`, schema v10) still restores with the
relocated widgets. Check the `_restore_alc_*` paths (mainwindow.py ~11574–12145)
and that baseline regions / peak centres re-draw on the new canvas.

## Q7 — Which spectra go into the scan (explore thoroughly with the user)

**Current mechanism (implicit).** Scan membership is exactly the data-browser
multi-selection. `MainWindow._update_selected_datasets` (~10726) pushes
`get_selected_datasets()` into the fit panel; **Build Scan** (`_on_scan_requested`,
~9092) reads `self._fit_panel.batch_datasets()` and
`build_field_scan(order_key="run")` integrates every run in it. A run is dropped
only if it can't be integrated (no grouping), or in RF mode isn't two-period;
exclusions go to the **log**, not the UI. There is no scan-specific selection
surface — no per-point include/exclude, no click-to-drop, no field/temperature
range gate. To change membership you reselect in the browser and rebuild, and you
cannot drop a single bad point.

`core.transform.integral.build_field_scan` already accepts an unused `filter=`
callback — a natural hook for any richer selection without touching the reduction.

**Candidate approaches (present trade-offs, get a decision):**

- **A. Per-point exclude/restore on the scan.** Click a scan point to drop/restore
  it (mirrors the existing draggable-handle interaction on the ALC canvas). Cheap,
  directly solves outlier removal. Membership stored in `FitSeries.extra`.
- **B. Checkbox run-list in the scan panel.** Explicit, decoupled from the browser
  selection; more UI, clearer provenance.
- **C. x-range gate.** Include only runs whose field/temperature falls in a chosen
  window. Good for trimming ends; interacts with the after-the-fact axis choice.
- **D. Keep browser-driven, make it visible.** Surface contributing runs and the
  dropped-with-reason list in the UI rather than the log. Smallest change.

Sub-decisions: how membership persists in `FitSeries`; how it interacts with the
axis choice (runs missing a field/temperature log already drop silently in
`_alc_display_scan`); whether exclusions survive a rebuild.

**Recommendation (draft — confirm):** A (per-point exclude) + D (surface
contributors/exclusions), both implementable via the core `filter=` hook and
`FitSeries.extra`, giving outlier control without a heavier run-list UI.

## Recommendation (draft — confirm with user)

Lean toward **Q1-A / Q4-A**: a dedicated third workspace panel showing the scan,
with the analysis controls staying docked (Q5). Resolve **Q2** first, since it
gates Q1 — the leading candidate is a slim retained time strip *or* spinbox-only
window entry, depending on how much the user values drag-to-set.
