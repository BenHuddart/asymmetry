# Verification plan — RF-µSR resonance GUI surface

## Acceptance criteria
1. **Discoverable fit model.** `RFResonanceMuP` is selectable from a GUI fit picker
   on the RF field-scan path (dedicated RF-scan panel per option A2, or — fallback —
   the field-x parameter-trend Model-Fit picker per A1). Verified by a GUI smoke
   test asserting the model is listed.
2. **Scan acquisition.** The GUI can build the **(Red − Green) integral-asymmetry
   vs field** series from the benzene RF runs without hand-assembly (option B1/B2),
   exposed via a core builder with a headless unit test.
3. **End-to-end recovery.** Driving the GUI path on the benzene RF scan recovers
   **A_µ ≈ 514.78 MHz** (±a few MHz) and **A_p ≈ 124.6 MHz** (within the paper σ),
   resonances ≈ 865/773 G — matching the core port's verified result.

## Tests to add (with the implementation)
- `tests/test_rf_scan_builder.py` — core: build (Red − Green) field series (synthetic
  + corpus-conditional), fit `RFResonanceMuP`, assert A_µ within tolerance. Green in
  CI via skipif when corpus absent.
- `tests/test_rf_scan_panel_gui.py` — offscreen GUI: panel builds a scan and lists
  the RF model in its picker; A_µ/A_p read-outs populate after a fit.
- Picker-visibility regression: assert `RFResonanceMuP` is offered where intended
  (guards the Round-2 "absent from picker" finding from regressing).

## Manual GUI re-test (the Round-2 repro, inverted)
Reuse `_findings/windows-gui/Benzene_RF_gap.md` steps: load the benzene RF runs →
open the RF-scan path → confirm the model is now present and a scan is buildable →
fit → read A_µ/A_p. The finding's "absent" observations should all flip to present.

## Validation ladder
- `python tools/harness.py test -- tests/test_rf_scan_builder.py` (fast inner loop).
- `python tools/harness.py validate` (lint + structural + full suite; the porting
  policy already accepts this study via the `index.json` entry added in this branch).
- `python tools/harness.py gui-smoke` for the panel path.

## Out of scope / risks
- Core Hamiltonian numerics are frozen (verified in `rf-musr-resonance-fit`); do not
  re-derive. Risk areas are the Red/Green pairing semantics and `nu_RF` seeding — pin
  both in the study pass before coding (see `test-data.md` TODOs).
