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
