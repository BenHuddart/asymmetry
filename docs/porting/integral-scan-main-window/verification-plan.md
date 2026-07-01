# Verification plan

## Harness ladder

```bash
python tools/harness.py structural                          # layout + boundaries
python tools/harness.py lint
python tools/harness.py test -- tests/test_field_scan_fitting.py   # focused
python tools/harness.py test --tier fast                    # inner loop (~40s)
python tools/harness.py gui-smoke                           # GUI startup/packaging
python tools/harness.py validate                            # standard tier, pre-push gate
```

Use the project venv `.venv/bin/python` (the harness re-execs itself with it).
GUI tests need `QT_QPA_PLATFORM=offscreen` in headless environments.

## GUI regression checks

- Switching into / out of the "Integral scan" representation leaves no orphaned
  or duplicated widgets (the dock-swap and stack routing stay consistent).
- The relocated scan canvas renders, and its draggable baseline regions / peak
  centres respond exactly as they did in the docked `ALCScanView`.
- Fit-range ↔ integration-window synchronization still holds under the chosen
  Q2 design (`fit_range_edit_committed` / `set_fit_range_display`).
- Project save → reopen restores the scan, baseline, and peaks (`FitSeries.extra`).

## Live-GUI confirmation (required before done)

Load the TCNQ ALC corpus (see test-data.md), enter Integral scan mode, Build
Scan, fit a Cubic baseline and a peak, read B0/Width/Amp, switch x-axis and toggle
the derivative — all with the scan in the main window and the controls docked.
The user confirms the layout and interactions before the change is considered
complete.

## Code review before push (required)

Before pushing or opening a PR, run a code review of the change and apply the
fixes: invoke `/code-review` at `high` effort (or the review workflow), address
the findings in the working tree, then re-run `validate` + `gui-smoke`. Only push
once the review is clean, the harness is green, and the user has confirmed the
live GUI. Pushing burns the user's CI minutes — push only when the user asks.

## Guardrails

- Core (`FieldScan`, `build_field_scan`, `fit_scan_baseline/fit_scan_model`) and
  the project schema must be untouched — if a change reaches into core, stop and
  reconsider; this is a GUI restructuring.
- Keep long work off the GUI thread (existing scan build already respects this).
