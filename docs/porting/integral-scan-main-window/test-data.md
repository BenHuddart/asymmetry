# Test data

## Corpus

- **TCNQ ALC / integral field scan** — EMU runs `emu00019489`–`emu00019519`
  (used in the earlier overnight GUI testing; see the ALC_TCNQ findings). A
  multi-run field scan with a D1 resonance, suitable for exercising Build Scan,
  Cubic baseline, and Gaussian/Lorentzian peak fits.
- Any multi-run set carrying a field (or temperature) log works for smoke-testing
  the relocated view; the scan needs ≥ 2 points with a valid x per
  `_alc_display_scan`.

## What to exercise

- Enter the "Integral scan" representation, select the runs, set an integration
  window, **Build Scan** — the scan must render in its new (central) location.
- Baseline fit (Linear / Cubic) over non-resonant regions; confirm the shaded
  regions and their draggable edges work on the relocated canvas.
- Peak fit (GaussianLCR / LorentzianLCR); confirm draggable peak centres and the
  overlaid total-fit curve.
- x-axis switch (field / temperature / run) and the dA/dB derivative toggle.
- Save → reopen the project; confirm the scan + baseline/peaks restore
  (`FitSeries.extra`).

## Automated coverage to add/adjust

- Existing GUI tests referencing `ALCScanView` placement will need updating once
  the widget is split/relocated (search `tests/` for `alc`, `ALCScanView`,
  `integral_scan`, `test_field_scan_fitting`).
- Add a test asserting the scan canvas is a descendant of the central workspace
  (mirror of the current dock-descendant assertion).

_Fill in exact run paths / any additional datasets during implementation._
