# Comparison — integral-scan representation

No single reference program is being ported here; this is a GUI relocation within
Asymmetry. The comparison below is *current Asymmetry* vs *target*, with notes on
how established muon tools present ALC scans.

## Current Asymmetry behaviour

| Concern | Current |
|---|---|
| Central plot in `integral_scan` view | Time-domain FB asymmetry (`integral_scan` normalized to `fb_asymmetry` in `plot_panel.py:~1606`) |
| Scan plot (A vs B/T/run) | Right dock, inside `ALCScanView` (Parameters dock) |
| Build controls | Right dock, `ALCFitPanel` (Fit dock) |
| Baseline / Peaks / RF controls | Right dock, in `ALCScanView`'s scroll area |
| Integration window | The draggable fit-range shaded on the central time plot; mirrored to `ALCFitPanel` spinboxes |
| Which spectra are included | *Implicit* — exactly the data-browser multi-selection (`_update_selected_datasets` → fit panel → `batch_datasets()`); every selected run is integrated. Runs are dropped only if unintegrable (or non-two-period in RF mode); exclusions go to the log. No per-point exclude, no range gate. `build_field_scan` has an unused `filter=` hook. |
| View plumbing | `PlotWorkspacePanel` stack has only `_time_panel` + `_frequency_panel`; `integral_scan` is a *time* view |

## Target behaviour (as implemented)

| Concern | Target |
|---|---|
| Central plot in `integral_scan` view | The scan itself (A vs B/T/run), with draggable baseline regions / peak centres, in a dedicated third workspace panel (`IntegralScanPanel`) |
| Baseline / Peaks / RF controls | Remain in the right dock (the scan view's analysis section) |
| Time spectra | A slim, collapsible time strip (`IntegralTimeStrip`) under the scan carries the preview |
| Integration window | Draggable edges on the time strip, mirrored with the build panel's spinboxes; the time plot panel stays the canonical fit-range owner |
| Which spectra are included | Browser selection as before, plus click-a-point exclude/restore (greyed points; persisted as `excluded_runs`) and a provenance line showing contributors/drops |

## Reference notes (studied, not vendored)

- **Mantid ALC interface** presents the scan (integrated asymmetry vs field) as
  the primary plot, with baseline modelling and peak fitting as side panels — the
  layout this study is moving toward.
- **WiMDA** integral/ALC mode likewise foregrounds the scan; the per-run time
  spectra are a means to the integrated point, not an interaction surface.

_To be expanded during the design discussion: capture any specific Mantid/WiMDA
layout affordances the user wants to match or deliberately diverge from._
