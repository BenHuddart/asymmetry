# Implementation options

The meat of this study. Resolved with the user on 2026-07-01 — see
[Decisions](#decisions-2026-07-01) at the end. Each question lists the candidate
approaches and trade-offs that were considered.

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

## Decisions (2026-07-01)

Confirmed with the user via `AskUserQuestion`:

- **Q1/Q2 → scan-dominant centre + slim time strip.** The scan plot dominates the
  centre; a short, collapsible time-domain strip below it keeps the draggable
  integration-window handles (mirrored to the `ALCFitPanel` spinboxes as today).
  Drag-to-set is preserved without giving the time plot real estate it doesn't
  earn in scan mode.
- **Q4 → A, dedicated third workspace panel.** A new ALC scan panel joins
  `PlotWorkspacePanel`'s `QStackedWidget`, routed when
  `active_view() == "integral_scan"`. The scan canvas (with its draggable
  baseline-region edges and peak centres) is extracted from `ALCScanView` and
  reused; the `integral_scan → fb_asymmetry` normalization in `plot_panel.py` is
  removed.
- **Q3 → axis/derivative selectors travel with the plot.** A compact toolbar row
  on the central scan panel, matching how the frequency panel carries its own
  mode controls. Baseline/Peaks/RF groups stay in the dock (Q5).
- **Q7 → A + D.** Per-point exclude/restore by clicking a scan point (excluded
  points drawn greyed, membership persisted in `FitSeries.extra` as
  `excluded_runs`), plus surfacing contributing runs and the
  dropped-with-reason list in the UI (a provenance line under the scan) instead
  of only the log. Exclusions are keyed by run number so they survive a rebuild
  over the same selection. No checkbox run-list, no x-range gate.

  *Implementation note:* exclusions are applied at the display/fit layer
  (`_alc_display_scan(include_excluded=...)`), **not** via the core
  `build_field_scan(filter=)` hook — filtering at build time would remove the
  point from the arrays entirely, so it could not be drawn greyed or clicked to
  restore. The core hook stays available for period/run-type subsetting.
