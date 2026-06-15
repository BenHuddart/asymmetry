# Comparison — RF-µSR resonance GUI surface

How the RF-resonance workflow is exposed across programs, and where Asymmetry's
GUI stops short. (Core numerics comparison lives in
[`../rf-musr-resonance-fit/comparison.md`](../rf-musr-resonance-fit/comparison.md).)

## WiMDA (reference)
- `RigiWorkshopFit` provides the RF-resonance **fit model** (the two-Lorentzian
  exact-diagonalisation `RFresonanceMuPlusProtonExact`, ported as the core
  `RFResonanceMuP`). The **scan acquisition** is NOT in `RigiWorkshopfit.dpr`
  (that file holds only fit-function DLL entry points — verified by grepping it:
  the only "red"/"period" hits are `Redfield/BPP` and the `select` doc string).
  Acquisition lives in WiMDA's main data path / grouping ("RG box"), which the
  corpus teaching guide describes operationally.
- **Red/Green = two acquisition periods within each run, not detector groups and
  not run pairs** (resolved). From the benzene corpus ground truth
  (`Chemistry/Muon spectroscopy of benzene/GROUND_TRUTH.md` §3D): *"Red-Green
  mode, RF at 218 MHz; **Green = RF off, Red = RF on**; grouping window selects
  Red, Green, or **Green − Red** difference. Time-domain difference first
  (bunch 3), then time-integral (bunch 500), build fit table from the first bin,
  model integral asymmetry vs field as a sum of two Lorentzians."*
- WiMDA workflow, restated: per field-stepped run, form the **(Green − Red)**
  time-domain asymmetry, time-integrate it to one value, collect those vs swept
  static field (the W-shaped double dip), fit with the exact Hamiltonian model →
  A_µ (mean dip position) and A_p (dip splitting).

## Asymmetry — core (done)
- `RFResonanceMuP` (parameter_models.py): field-scope trend component, params
  `A_mu, A_p, nu_RF, ampl1, wid1, ampl2, wid2, BG`; verified on benzene corpus.
- Reachable **only** if the user already has a field-x parameter-trend series and
  types/selects the model in the Model-Fit dialog — i.e. not discoverable, and
  with no built-in way to produce the (Red − Green) scan it expects.

## Asymmetry — GUI (the gap)
| Capability | WiMDA | Asymmetry GUI (v0.4.0) |
|---|---|---|
| RF model in a fit picker | ✅ RigiWorkshopFit | ❌ absent from time-domain/ALC pickers |
| Build (Red − Green) field scan | ✅ | ❌ only plain integral-scan (ALC) exists |
| Fit scan → A_µ, A_p | ✅ | ⚠ core can, but no GUI entry to drive it |

## Adjacent Asymmetry surface to reuse
- `alc_panel.py` integral-scan: load field-scan runs → set integration window →
  Build Scan → field-domain Baseline+Peaks fit. The RF scan is the same shape with
  a **Green − Red period difference** instead of a single integral, then a
  2-Lorentzian + BG fit driven by `RFResonanceMuP`. Reuse this scaffold.
- **Green − Red machinery already exists in core** (resolved — no new arithmetic
  needed): `PeriodMode.GREEN_MINUS_RED` (`core/utils/constants.py`),
  `combine_period_asymmetry(..., mode)` and `select_period(...)`
  (`core/io/periods.py`), and `integrate_curve(time, asym, err, ...)` explicitly
  documented for *"a combined green∓red spectrum"* (`core/transform/integral.py`).
  The NeXus loader already loads two-period RF runs as one run carrying both
  `period_histograms` and a `period_reduced` cache, tagged `{1: red, 2: green}`.
- **What the existing ALC path does NOT do:** `build_field_scan → integrate_run →
  _reduce_run_to_fb` reduces from `run.histograms` (the **red** period only) and
  ignores `period_mode`. So the integral scan integrates Red, never Green − Red.
  This is the acquisition gap a small core scan-builder closes.
