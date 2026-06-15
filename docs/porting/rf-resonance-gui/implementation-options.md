# Implementation options — RF-µSR resonance GUI surface

Two decisions: **(A) how to expose the fit** and **(B) how to build the scan**.
Pick one per axis in the study pass; record the choice + rationale here.

## A. Exposing `RFResonanceMuP` to the user
- **A1 — Trend-picker only (smallest).** Ensure `RFResonanceMuP` is offered in the
  parameter-trend Model-Fit picker when the x-axis is **field** (it is registered
  field-scope; verify it actually lists in `model_fit_dialog`/Build Parameter Model
  and add it if filtered out). User builds a field series, then fits it. Lowest
  effort; still requires the user to assemble a (Red − Green) field series by hand.
- **A2 — Dedicated RF-scan panel (recommended).** A sibling of `alc_panel` that
  owns both the scan build (B) and the resonance fit (BG + 2 Lorentzians driven by
  `RFResonanceMuP`), with A_µ/A_p read-outs. Most discoverable; matches WiMDA's
  one-stop workshop. Largest effort.
- **A3 — Time-domain picker entry.** Add `RFResonanceMuP` to the Build-Fit-Function
  registry too. Rejected unless there is a genuine time-domain use — RF resonance is
  a field-scan observable, not a per-run time fit; listing it there would mislead.

## B. Building the (Red − Green) field scan
- **B1 — Extend integral-scan.** Add a "difference" mode to the existing integral
  scan (Red group − Green group, or RF-on − RF-off) so Build Scan yields the RF
  observable. Reuses `alc_panel` + `core/transform/integral.py`. Preferred if Red/
  Green are detector groups.
- **B2 — New core builder.** A `core/transform/` helper that pairs runs and returns
  the field-vs-(Red−Green) series, GUI-agnostic and unit-testable. Preferred if the
  pairing is run-based (RF-on/off run pairs) rather than group-based.

## Cross-cutting
- **Registry seam:** the model is in `parameter_models.py`; confirm whether the
  field-scope trend picker reads a registry that already includes it (then it is a
  *filter/visibility* fix) or needs an explicit add.
- **Seeding:** `nu_RF` is a known acquisition constant (e.g. 218.5 MHz for benzene)
  — surface it as a fixed/seeded input, not a free fit param by default.
- **Reuse, don't reimplement:** the Hamiltonian/numerics are done and verified;
  this work is GUI plumbing + (optionally) one core scan-builder.

## Recommendation (provisional, to confirm)
A2 + B1/B2 — a dedicated RF-scan panel mirroring `alc_panel`, with the scan builder
in core so it is testable headlessly. Fall back to A1 if a panel is out of budget.

## Chosen design (study pass — confirmed with maintainer 2026-06-15)

The study pass found the gap is **narrower than the skeleton assumed**: the core
already has the entire Green − Red period machinery (`PeriodMode`,
`combine_period_asymmetry`, `select_period`, `integrate_curve`, two-period loader)
and `RFResonanceMuP` already lists in the field-x parameter-trend Model-Fit picker.
The missing pieces are only (1) a Green − Red *integral-scan acquisition* and (2) a
way to *fit that scan with `RFResonanceMuP`* from the scan view. So a whole new
panel (A2) would duplicate the ALC scaffold for little gain.

**Chosen: extend the existing ALC integral-scan path (a blend of A2's discoverability
with B2), not a new panel.**

- **B2 — core scan-builder.** Add `build_rf_difference_scan(...)` (home:
  `core/io/periods.py`, which already owns red/green and may import
  `transform`; layering-clean since `io → transform`). Per run it pairs the two
  periods (red, green), forms the **Green − Red** reduced curve via the existing
  `combine_period_asymmetry(GREEN_MINUS_RED)` (single source of truth — no copied
  arithmetic), `integrate_curve`s it over the window, and returns a
  `FieldScan` in the **same fractional convention as `build_field_scan`** (÷100),
  ordered by field. Two-period-only runs survive; others land in `excluded` with a
  clear reason. GUI-agnostic and unit-testable headlessly.
- **A (fit exposure) — reuse the scan fitter.** `fit_scan_model` already accepts any
  component/expression, so it fits `RFResonanceMuP` with no new fitter. Surface it
  in `ALCScanView` as an **"RF resonance (A_µ, A_p)"** fit action with a ν_RF input
  (seeded 218.5 MHz, fixed by default) and **A_µ / A_p read-outs**. Reuses the
  baseline step and the existing percent display scale.

### Units (resolved)
`RFResonanceMuP` was verified on a **percent**-scale W-dip (`test_rf_musr_resonance.py`
seeds `ampl1=-18.0, BG=-1.5`). The ALC display path stores per-point values
**fractional** and multiplies ×100 for display *and* for the fit it drives. So the
builder returns fractional values (matching `build_field_scan`); the existing ×100
display feeds percent to the RF fit, where the registered seeds are valid. No
double-scaling, no special-casing of the display path.

### Rejected / deferred
- **A2 dedicated panel** — rejected: duplicates the ALC scaffold; the extend-path
  gives the same discoverability + A_µ/A_p read-outs for a far smaller diff.
- **B1 difference-mode on `build_field_scan`'s F/B path** — rejected: Green − Red is
  a *period-asymmetry difference*, not a forward/backward count difference, so the
  F/B `integrate_run` path cannot express it; a curve-level builder is correct.
- **A3 time-domain picker entry** — rejected (unchanged): RF resonance is a
  field-scan observable, not a per-run time fit.
- **ν_RF from metadata** — deferred: not present in the NeXus sample metadata;
  user-entered (seeded) is the contract.
