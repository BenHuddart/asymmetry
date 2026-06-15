# Comparison — RF-µSR resonance GUI surface

How the RF-resonance workflow is exposed across programs, and where Asymmetry's
GUI stops short. (Core numerics comparison lives in
[`../rf-musr-resonance-fit/comparison.md`](../rf-musr-resonance-fit/comparison.md).)

## WiMDA (reference)
- `RigiWorkshopFit` provides an RF-resonance fitting workshop: load the RF field
  scan, form the (Red − Green) integral asymmetry vs swept static field, and fit
  the resulting W-shaped double dip to the exact muon+electron+proton Hamiltonian
  to extract A_µ and A_p. The scan acquisition and the model fit live in one path.
- **TODO (study pass):** trace the WiMDA Pascal entry points for (a) building the
  Red−Green scan from the run set and (b) seeding/fitting the resonance — cite
  unit/file/line under `$WIMDA_SRC` as the other RF study does.

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
  a **Red − Green difference** instead of a single integral, then a 2-Lorentzian +
  BG fit driven by `RFResonanceMuP`. Reuse this scaffold rather than inventing one.
- **TODO:** confirm whether "Red"/"Green" map to detector groups / RF-on vs RF-off
  runs in the corpus, and how the GUI should pair them.
