# RF-µSR resonance — GUI surface (study)

**Status:** study (skeleton — Round-2 GUI finding). **Depends on:** the implemented
core port [`rf-musr-resonance-fit`](../rf-musr-resonance-fit/README.md) (closes
parity gap PC1).

## The gap (Round-2 GUI testing, Windows, v0.4.0 @ fbf8aae)

The core RF-resonance model **`RFResonanceMuP`** exists and is verified end-to-end
on the benzene corpus (recovers A_µ=516 MHz vs paper 514.78, A_p=125 MHz vs 124.6).
But it is registered **only as a field-scope parameter-trend component** in
`src/asymmetry/core/fitting/parameter_models.py` — there is **no GUI surface**:

1. **Absent from the time-domain / ALC fit-function picker.** Verified live: Build
   Fit Function → Functions → Muonium lists only `MuoniumHighTF / HighTFAniso /
   LFRelax / LowTF / TF / ZF`; `RFResonanceMuP` appears in none of the categories.
2. **No RF field-difference scan control.** The only field-domain surface is
   *Integral scan (ALC)*, which integrates asymmetry vs field. RF-µSR needs a
   **(Red − Green) integral-asymmetry vs swept field** scan (the W-shaped double
   dip), which has no entry point.

Finding detail: `_findings/windows-gui/Benzene_RF_gap.md`.

## What this study must produce

A design for surfacing PC1 in the GUI, in two parts:

- **(A) Fit-picker exposure** — make `RFResonanceMuP` selectable where a user fits
  an RF field scan (parameter-trend Model-Fit picker, and/or a dedicated RF-scan
  panel). Decide whether it stays trend-only (field x-axis) or also gains a
  scan-domain fit path.
- **(B) RF scan acquisition** — a control to build the (Red − Green) field-scan
  observable from the loaded RF runs (analogous to `alc_panel` / integral-scan),
  feeding the fit in (A).

## Entry points to study (Asymmetry)

- Core model: `core/fitting/parameter_models.py` (RFResonanceMuP, field-scope).
- Trend picker GUI: `gui/panels/model_fit_dialog.py` + `gui/panels/fit_parameters_panel.py`.
- Time-domain picker GUI: `gui/panels/fit_function_builder.py` + `widgets/function_expression_builder.py`.
- Scan acquisition pattern to mirror: `gui/panels/alc_panel.py` (integral scan / Build Scan).
- Reference (WiMDA): `RigiWorkshopFit` RF-resonance workflow — see the core study.

## Non-goals
- No change to the verified core RF Hamiltonian / `RFResonanceMuP` numerics.
- Not a new reference port (the port is done); this is GUI plumbing + a scan builder.
