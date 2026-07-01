# Integral (ALC) scan → main window

**Status:** implemented (2026-07-01; design decisions in
[implementation-options.md](implementation-options.md)); awaiting live-GUI
confirmation per [verification-plan.md](verification-plan.md)
**Branch:** `feat/integral-scan-main-window`
**Scope:** GUI restructuring only — no core or persistence changes expected.

## Problem

When the "Integral scan" representation is active, the central plot keeps showing
the time-domain FB asymmetry, while the actual scan (integral asymmetry vs
field/temperature/run) is confined to the right-hand `ALCScanView` in the
Parameters dock. In scan mode the user does not interact with the per-run time
spectra — they are just integrated — so the scan itself should occupy the main
window, with the Baseline/Peaks/RF analysis controls remaining in the dock.

## Why this is a study, not a straight edit

The integration window *is* the central time plot's draggable fit-range. Removing
the time spectra from the centre collides with "how does the user set/see the
integration window?" — the one interaction that currently lives on the time plot.
That tension needs a design decision with the user before any code is written.

A second design thread rides along: **which spectra are included in the scan** is
today implicit (the data-browser multi-selection), with no per-point exclude or
range gate. The user wants to explore a more elegant selection mechanism — see Q7
in [implementation-options.md](implementation-options.md).

## Artifacts

- [comparison.md](comparison.md) — current Asymmetry representation vs the target,
  with notes on how WiMDA/Mantid surface ALC scans.
- [implementation-options.md](implementation-options.md) — the open design
  questions and candidate approaches (**the meat**; to be resolved with the user).
- [test-data.md](test-data.md) — the ALC corpus and what to exercise.
- [verification-plan.md](verification-plan.md) — harness + live-GUI checklist.
- [handoff.md](handoff.md) — the ready-to-paste prompt for the implementing agent.

## How this study was produced

Two read-only exploration passes mapped the core (`FieldScan`,
`build_field_scan`, `fit_scan_baseline/fit_scan_model`, `FitSeries.extra`
persistence) and the GUI (`PlotWorkspacePanel` central stack, `ALCScanView` /
`ALCFitPanel`, and the `_alc_mode` / `_sync_fit_dock_mode` dock-swap machinery in
`mainwindow.py`). No code was changed. The remaining artifacts are seeded and are
to be completed during the design discussion with the user.
