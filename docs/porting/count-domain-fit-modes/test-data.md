# Count-domain fit modes — test data

Corpus root: `~/Documents/WiMDA muon school`. Only formats Asymmetry loads are
used (HDF5 `.nxs`; PSI `.bin`/`.mdu`). Synthetic runs come from `core/simulate`,
which is on `main`.

## Real-data targets

| Mode | Run(s) | Why this run |
|---|---|---|
| α-free F+B fit | `Semiconductors/Shallow donor state in cadmium sulphide/Data_hdf5/EMU000207xx.nxs` (CdS, ISIS EMU, TF) | A transverse-field run with a clear diamagnetic precession and forward/backward detector banks — the canonical α-calibration shape. Already loadable HDF5; already used for link-group/muonium tests, so its grouping is understood. α from the F+B fit is cross-checked against `estimate_alpha` (diamagnetic / general / ratio) on the same run. |
| Single-histogram N₀·e^(−t/τ) | `Magnetism/Magnetic ordering in EuO/data/deltat_pta_gps_29xx.bin` (EuO, PSI GPS, continuous source) | A continuous-source run is the textbook single-histogram case: one detector's raw counts follow N₀·e^(−t/τ)·(1 + A·P) + bg with a flat random background. Loadable PSI `.bin`. Fitting one group recovers the decay envelope, N₀, and background. |
| Exclude range (real artefact-free baseline) | CdS TF run above | Fit the clean run, then fit again with a synthetic spike injected into an interior window and excluded — the two fits must agree (see verification-plan). |
| Deadtime-in-fit / promote | CdS / EuO high-statistics run | High count rate makes the deadtime term non-negligible; the fitted DT0 is compared to the grouping's calibrated τ_dead from `calibrate_deadtime_from_histograms`. |

WiMDA's own `wimda grouping/1276.fit`, `1277.fit` are WiMDA fit-result files
(grouping/calibration example). They are read as a **design** reference for the
expected grouping/α workflow, not parsed as oracles.

## Synthetic targets (`core/simulate`, ground-truth injection)

| Mode | Generator | Injected truth recovered |
|---|---|---|
| α-free F+B | `simulate_run(..., alpha=α₀)` with a TF precession signal, two detector groups | α₀ recovered within its fitted uncertainty; N₀_F/N₀_B ratio = α₀ |
| Single-histogram | `simulate_run` single group, known N₀, bg, λ | N₀, bg, λ (λ fixed at τ_μ; envelope amplitude recovered) |
| Exclude range | clean synthetic run + an injected artefact in [t_ex0, t_ex1] | masked fit == clean-data fit (parameters within tolerance) |
| Fittable t₀ | synthetic run generated with a known bin offset | t₀ recovered |
| Poisson vs Gaussian | low-count synthetic run (few counts/bin at late time) | Poisson cost recovers injected parameters with smaller bias than Gaussian √N — demonstrates the low-count divergence the modes exist to fix |
| Double pulse | new double-pulse option added to `simulate` (two time-shifted, exp(∓dpsep/2τ)-weighted copies) | dpsep and the pulse weighting round-trip |

Double-pulse synthesis is added to `core/simulate` (Phase 3) so the round-trip
test has a first-class generator rather than a handwritten array; this also
gives the GUI a way to produce demo double-pulse runs.

## Numerical oracles transcribed from WiMDA

Where parity is *claimed* (not just plausibility), the expected count value is
computed directly from the transcribed Pascal model in the test, e.g. for `fgFB`
at a sample (t, A, α, N₀, bg_F, bg_B):

- forward = N₀·√α·(1 + A) + bg_F·e^(t·λ_μ)
- backward = N₀·(1/√α)·(1 − A) + bg_B·e^(t·λ_μ)

and the model under test must reproduce it to round-off. These transcription
oracles live beside the fit tests so a future reader sees the WiMDA formula and
the Asymmetry result side by side.

## References

- *Muon spectroscopy* (muon-spectroscopy textbook) — continuous- vs pulsed-source
  count structure; the α-calibration TF measurement.
- Corpus provenance: ISIS Muon Training School datasets (CdS/EMU, EuO/GPS).
