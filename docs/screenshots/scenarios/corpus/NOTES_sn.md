# NOTES — Critical fields in Sn (Superconductivity)

Module: `tin_critical.py`. Example: `Superconductivity/Critical fields in Sn`.
Data: HIFI (ISIS) NeXus-v1 **HDF4** `.nxs`, runs 91488–91529 (42 runs); loader
verified (`file` reports HDF4; the normal loader reads all runs, T/B metadata
correct). Ground truth: the example's `GROUND_TRUTH.md` (audit-corrected).
Literature guidance (GT §10, Karl et al. PRB 99, 184515): T_c = 3.717(3) K,
B_c(0) = 30.578(6) mT ≈ 306 G, parabolic H_c(T) = B_c(0)[1 − (T/T_c)²].

## TL;DR

**The intermediate state is present and the H_c(T) mapping works — but only from
the raw detector counts, and only after realising the thermometer reads
~0.7–1.1 K low** (the guide's own "how much is the thermometer in error?"
question, answered from the data). Two obstacles had to be cleared:

1. **The standard loader grouping cancels the signal.** HIFI's 64 detectors form
   two *longitudinal* rings; the file's `grouping` array reduces them to a
   forward/backward pair, which cancels a *transverse* precession. The
   standard-loader F−B asymmetry is noise for every run (checked in the GUI:
   time-domain and Fourier-panel renders both empty — see "Rejected GUI
   scenarios"). The loaded `MuonDataset` retains no histograms, so regrouping is
   not reachable through the normal pipeline; the scenarios read the HDF4 counts
   directly (still via `corpus_path`) and re-group each ring into left/right
   halves (two quadrature pairs per ring → phase-insensitive line spectra).
2. **The nominal temperatures are wrong** (commissioning cryostat). The
   measured intermediate-state line pins the real temperatures (below).

## The physics found in the data (transverse-regrouped quadrature spectra)

- **40 G temperature scan (91516–91529):** a single line marches monotonically
  from ≈141 G at nominal 1.6 K down to the applied 40 G at nominal 2.8 K, every
  point 6–23σ above the noise floor. In the intermediate state the normal
  domains carry exactly B_c(T), so B_int = B_c(T_real) — this *is* the H_c(T)
  measurement. At nominal 2.8–2.9 K the line sits at the applied field with
  **doubled amplitude** (0.006→0.014): the boundary H_c(T) = 40 G has been
  crossed and the whole sample precesses at B_app.
- **Base-T field series (nominal 1 K):** 20 G → line at 138 G (6σ); 80 G →
  141 G (10σ) — the **same** field, independent of the applied field: the
  type-I domain-field signature, B_int = B_c(T_real) ≈ 140 G ⇒ T_real ≈ 2.73 K
  (the cryostat's true base). 160 G → line at 158 G (21σ), i.e. at the applied
  field: 160 G > B_c(T_real), the sample is fully **normal** in the 160 G run.
- **"Sensor-fault" block 91501–91515** (nominal 1.95–2.35 K, logged sensors
  ~8 K): strong lines exactly at the applied field at all three fields
  (18.6/38.8/78.6 G, 20–32σ, amplitudes ≈0.011–0.015 = the full normal-state
  amplitude). The sample was normal — **the ~8 K sensor readings were right and
  the nominal setpoints wrong.** Not a sensor fault: a setpoint-control fault.

### Boundary points vs guidance (the deliverable table)

Measured B_int per 40 G-scan run; T_real inferred by inverting the guidance
parabola (T_real = T_c·√(1 − B_int/B_c(0))); ΔT = thermometer error.

| Run | nominal T (K) | B_int (G) | σ | T_real (K) | ΔT (K) |
|---|---|---|---|---|---|
| 91516 | 1.6 | 140.9 | 7 | 2.73 | +1.13 |
| 91517 | 1.7 | 144.5 | 7 | 2.70 | +1.00 |
| 91518 | 1.8 | 129.9 | 6 | 2.82 | +1.02 |
| 91519 | 1.9 | 118.5 | 7 | 2.91 | +1.01 |
| 91520 | 2.0 | 108.8 | 6 | 2.98 | +0.98 |
| 91521 | 2.1 | 100.6 | 11 | 3.05 | +0.95 |
| 91522 | 2.2 | 91.9 | 9 | 3.11 | +0.91 |
| 91523 | 2.3 | 83.3 | 8 | 3.17 | +0.87 |
| 91524 | 2.4 | 76.9 | 11 | 3.22 | +0.82 |
| 91525 | 2.5 | 69.5 | 11 | 3.27 | +0.77 |
| 91526 | 2.6 | 59.0 | 10 | 3.34 | +0.74 |
| 91527 | 2.7 | 50.3 | 11 | 3.40 | +0.70 |
| 91528 | 2.8 | 39.0 | 20 | — (crossed; line at B_app) | ≲+0.67 |
| 91529 | 2.9 | 38.8 | 23 | — (crossed; line at B_app) | — |

Base-T field-series points: 91488 (20 G) → 138.1 G ⇒ T_real 2.75 K; 91490
(80 G) → 140.9 G ⇒ 2.73 K (nominal 1.0 K ⇒ ΔT ≈ +1.7 K: the cryostat never got
below ≈2.7 K). ΔT shrinks smoothly from ~1.1 to ~0.7 K as the setpoint rises —
consistent with a poorly anchored commissioning thermometer, and broadly with
the logged centre-stick sensor (which reads ~0.4–0.7 K above nominal).

Consistency checks: the boundary crossing at 40 G is observed between nominal
2.7 and 2.8 K vs the parabola's 3.47 K (ΔT ≈ +0.7 K, matching the trend); the
160 G run is normal exactly as required (160 > B_c(T_real) ≈ 140 G); and the
intermediate-state condition B_app > (1−N)·B_c with the measured onset at 20 G
implies an effective demagnetising factor N ≳ 0.86 — sensible for a thin foil
at 45° (perpendicular component dominates).

### The 3.85 MHz (B_c at a true 1 K) search — orchestrator follow-up

Requested check: a second line near B_c(1 K) ≈ 284 G ≈ 3.85 MHz in the
transverse-regrouped 160 G run. **Not present.** Searched run 91491 with the
quadrature periodogram over 3.2–4.6 MHz in windows t = 0.15–{1.5, 3, 5, 10} µs
(also 0.3 MHz-boxcar smoothing for a broadened line): nothing above 1.4σ per
ring (combined max 1.0σ at 3.47 MHz), vs 18–21σ for the 2.169 MHz applied-field
line; upper limit ≈ 0.002 amplitude ≈ 15 % of the observed line. **Explanation:
the sample was never at 1 K.** T_real ≈ 2.73 K ⇒ B_c ≈ 140 G < 160 G applied,
so the 160 G run is fully normal and no B_c line can exist in it. The B_c line
the coordinator predicted *does* exist — at ≈140 G (1.9 MHz) in the 20 G and
80 G base-T runs, and marching through the 40 G scan.

## Scenarios shipped (standalone Matplotlib, `bscco field_compare` pattern)

| name | what it shows | docs use |
|---|---|---|
| `corpus_sn_hc_dome` (**headline / top pick**) | Guidance dome (T_c, B_c(0), per-field crossings) + all 42 (nominal T, B_app) runs + the **measured B_int(T) points** from the 40 G scan computed from raw counts at capture time: red ◆ intermediate-state points falling systematically left of the parabola, grey ■ post-crossing points at B_app, thermometer-error arrow (observed crossing 2.8 K vs 3.47 K). | The phase-boundary mapping deliverable + the thermometer-error answer in one figure. |
| `corpus_sn_intermediate_lines` | (top) base-T spectra at 20/80/160 G: 20 & 80 G lines coincide at ≈140 G (domain field independent of B_app), 160 G at applied field (normal); (bottom) 40 G-scan spectra stacked: the line marching 141→39 G with the amplitude jump at the crossing. | The spectral evidence for the intermediate state; frequency-domain teaching figure. |
| `corpus_sn_transverse_recovery` | 160 G run 91491: standard F−B asymmetry (noise) vs L/R transverse regrouping of the same raw counts (clean 2.139 MHz ⇒ 158 G line, fitted). | The loader/grouping caveat; why the GUI shows nothing on this dataset. |

All deterministic; PNGs read back and reframed twice for caption/edge issues;
sizes 135/158/193 KB (budget 600 KB). No `requires_fit` (the only fit is scipy
`curve_fit` for an annotation; the spectra are direct linear projections).

## Intermediate-state observations (brief's checklist)

- Amplitude reduction + extra line: seen as predicted, but inverted from the
  naive reading — the *domain* line at B_c is the extra line (at ≈140 G), and
  the intermediate-state amplitude is ~⅓–½ of the normal-state amplitude
  (0.0035–0.0067 vs 0.011–0.015), growing as the normal-domain fraction grows
  with T, then jumping to full amplitude at the crossing.
- Amplitude ratio 20 G vs 80 G at base T (0.0035 vs 0.0050) is qualitatively
  consistent with flux conservation (normal fraction ∝ B_app/B_c), though a
  quantitative flux estimate (GT Q4) needs the sample/beam-spot geometry.
- No resolvable Meissner-fraction (B = 0) feature: it carries no precession and
  the detrended spectra suppress ν → 0.

## Rejected GUI scenarios (captured, inspected, dropped)

- `corpus_sn_tf_precession` (160 G, loader asymmetry, time domain): pure noise —
  no visible oscillation (the ring-sum cancellation, confirmed on screen).
- `corpus_sn_field_fft` (160 G, GUI Fourier panel): near-DC spike + flat noise;
  no line at the γ_μ·B marker. Dropped; the caveat figure carries the evidence.

## Problems / product notes

- **Loader retains no per-detector or per-group histograms**, so HIFI TF data
  reduced with the file's longitudinal grouping cannot be re-grouped in-app —
  and this dataset is *unusable* in the GUI as a result. Product opportunity: a
  reload-with-custom-grouping path (or keeping histograms on the dataset) would
  make Asymmetry able to run this whole workflow natively.
- The 40 G base-T run 91489 shows no clean line at either B_app or B_c —
  presumably an unlucky amplitude/damping for this field/geometry; excluded
  from figures (its scan twin 91516 is used).
- `scipy.optimize.curve_fit` used for one annotation fit; present in the venv.
- Thermometer error is quantified against the *guidance* parabola (GT §10 fence:
  same material, different sample); the ΔT numbers inherit that assumption.

## Top pick

`corpus_sn_hc_dome` — the measured intermediate-state B_int(T) points marching
down the (T, B) plane against the accepted dome, with the boundary crossing and
the ~0.7–1.1 K thermometer error annotated: the example's intended deliverable
(H_c(T) mapping + thermometer-error estimate) in a single render.
