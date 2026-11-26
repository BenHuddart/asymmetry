# NOTES — LiFeAs (pnictide superconductor) corpus scenarios

Module: `lifeas_pnictide.py` · Example: `Superconductivity/LiFeAs`
Spec: that example's `GROUND_TRUTH.md` (paper-graded — no teaching docx; the
reference is Pratt *et al.*, PRB **79**, 052508 (2009)).

The scenarios use the 40 mT temperature scan of LiFeAs Sample 1 (LFA,
PSI GPS runs 3366–3373). The low-temperature plateau in the vortex-lattice
field width B_rms gives the penetration depth λ_ab.

## Scenarios registered

| name | render | docs use |
|---|---|---|
| `corpus_lifeas_pair_select` | Matplotlib 2-panel, base-T run 3366: default Forward/Back (cancelled mush) vs Up/Down transverse pair (clean 5.44 MHz damped precession) | **Data-handling**: the essential detector-pairing choice for these spin-rotated (WED) runs — the loader's default pair cancels the signal |
| `corpus_lifeas_tf_fit` | GUI single-fit panel, two-Gaussian TF fit on 3366 (1.5 K, 40 mT, Up/Down): σ_2 = 1.189, σ_4 = 0.178, χ²ᵣ = 9.71, model string + param table | **Core fit**: the vortex signal σ_2 with the weakly-relaxing Ag background σ_4 → B_rms ≈ 1.96 mT (λ_ab ≈ 195 nm) |
| `corpus_lifeas_brms_t` | Fit Parameters trend panel showing Sample 1 two-Gaussian fits, with a 16 K transition marker | Field width from the raw GPS runs, with unresolved points ringed in grey |
| `corpus_lifeas_vortex_lineshape` | Matplotlib FFT overlay, S1 40 mT: 18 K narrow (nuclear) vs 1.5 K broad (vortex) p(B) line at 5.44 MHz | **Normal vs SC**: the field-distribution broadening below T_c that B_rms measures |

`requires_fit = True` on `tf_fit` and `brms_t` (real iminuit fits at capture).

## Workflow followed (GROUND_TRUTH §4 / §11)

- **Model** (§4): above T_c a single Gaussian (nuclear); below T_c the vortex
  broadening adds in quadrature, σ² = σ_VL² + σ_n² (Eq. 2), and B_rms = σ_VL/γ_µ
  (γ_µ = 0.8516 µs⁻¹ mT⁻¹). Powder London limit (Eq. 3):
  B_rms = √0.00371·φ₀/(3^{1/4}λ_ab)² → **195 nm ↔ 1.912 mT**
  (verified numerically — the √ prefactor matters).
- **Detector pairing (the key corpus fact).** These are spin-rotated "TF WED"
  GPS runs (5 histograms: Forw, Back, Up, Down, Righ). The loader's *default*
  Forward/Back pair sees the precession **in phase** and it **cancels** — the
  F/B FFT peaks at 2ν (≈ 11.8 MHz), ratio ≈ 2.5 (noise). The real 400 G
  precession (5.44 MHz, ratio ≈ 20) lives in the **Up/Down transverse pair**.
  Every quantitative render regroups onto Up/Down via `_regroup` (the same
  `apply_grouping_aligned` / `compute_asymmetry` primitives the loader and the
  grouping dialog use). This is documented as the deliberate reduction choice.
- **Gaussian convention.** Asymmetry's `Gaussian` component is A·exp(−(σt)²)
  (no ½), the paper's relaxation is exp(−σ²t²/2), so **σ_paper = √2·σ_Asymmetry**.
  B_rms[mT] = √2·σ_VL[µs⁻¹]/0.8516.
- **Weakly-relaxing background (§6, <20 %).** The base-T envelope decays fast
  then leaves a persistent ~2 % tail to 4 µs — muons in the Ag holder,
  precessing at the same ν with slow nuclear damping. A *single* Gaussian is
  diluted by that tail (base-T B_rms falls to ~1.1 mT). A **two-Gaussian
  signal+background** model recovers σ_VL: base-T B_rms ≈ 1.96 mT, matching the
  paper plateau. The two-component fit is used for the SC-state runs.
- **Nuclear reference σ_n.** The 18 K run (3373, above T_c = 16 K, 40 mT) has no
  vortex broadening, so σ_n is taken from a *single* Gaussian fit (σ_n = 0.156);
  the two-component split is degenerate there. σ_n is subtracted in quadrature.
- **Warm-start** σ downward in temperature; the two-component split degenerates
  as the vortex signal collapses near T_c, so the headline overlay keeps only
  the points where the broad component is well-defined (σ > 1.3 σ_n and signal
  fraction > 0.55): the low-T plateau + onset of collapse (1.5–10 K).

## Sample 1 trend

The fits give a base-temperature width of approximately 1.96 mT, or
λ_ab ≈ 193 nm by Eq. (3). Pratt et al. report 195(2) nm for Sample 1.
The panel rings the 12, 14, 16, and 18 K points in grey and excludes
them from the trend because the two-component fit does not resolve the
vortex contribution at those temperatures.

## Feature-demonstration opportunities

- **Detector grouping** shown as the headline data-reduction insight
  (`pair_select`) — the clearest example in the corpus of *why* pairing matters
  (default pairing produces pure noise). Could also be driven through the
  grouping dialog if a UI shot of that panel is wanted later.
- **Two-component (signal + background) time-domain fit** is shown converged in
  the GUI single-fit panel — a strong image of a physically-motivated composite
  model (vortex Gaussian + Ag-background Gaussian at a shared Larmor line).
- **Vortex line broadening** shown in the frequency domain (`vortex_lineshape`)
  as p(B) width — the direct picture of the second moment B_rms.
- Not captured: the **field-dependence** analysis (Fig. 2, σ_M ∝ B₀^n with
  n = 0.49(6)) — Sample 1's 100 G–6 kG and Sample 2's 7–200 G sweeps are in the
  corpus and could feed a B_rms-vs-field scenario, but that is the field-axis
  check, not the σ(T) trend the corpus targets (§11).

## Problems / honest caveats

1. **Default Forward/Back pairing is wrong for these WED runs** — it cancels the
   precession. The whole module regroups onto Up/Down. The in-app fix already
   exists, so this is a *discoverability* note, not a missing feature: the loader
   tags these runs `field_direction="Transverse"`, so the Grouping dialog raises
   the transverse-field nudge and `recommend_grouping_preset` returns GPS's
   `Spin-rotated (B+U/F+D)` preset (14.5σ on run 3366). The Detector Layout editor
   also offers `Transverse (Vector)` (matching musrfit WED(L); the Up/Down
   projection recovers the 5.41 MHz line at 17σ) and `WEP (spin-rotated)`. The
   correct native path is therefore Grouping dialog → transverse nudge → Detector
   Layout → apply the spin-rotated / vector preset. (An earlier version of this
   note flagged the lack of a "per-instrument WED preset" as a product gap — that
   was wrong; the presets exist and are wired to the transverse nudge, disproven
   against this branch's own `src`.)
2. **Absolute B_rms needs the two-Gaussian model + √2 convention.** A naïve
   single Gaussian on the raw pair gives ~1.1 mT (background dilution) and, in
   Asymmetry's exp(−(σt)²) convention, is a further √2 low — three quiet factors
   that must all be handled to land on the paper's 1.9 mT. Documented in the
   module docstring and `_brms_mT`.
3. **Two-component fit degenerates near T_c.** As the vortex signal collapses,
   the signal/background split becomes ill-conditioned (component swap, σ → bound
   ghosts). Only the well-defined low-temperature points contribute to the trend.
   χ²ᵣ ≈ 6–10 across the
   scan (raw error scaling on rebinned PSI data) — the fits converge and the
   parameters are physical, but this is not a χ²ᵣ ≈ 1 regime.
4. **MaxEnt does not cleanly separate the vortex/normal lineshape here — FFT
   stays the tool.** Re-tested on the PR-249 branch (2026-07-16): core `maxent()`
   on the Up/Down transverse pair (groups 3, 4), out-of-the-box auto steering,
   converges to **χ²/N ≈ 30–50** with **no** vortex-vs-normal contrast — the
   1.5 K and 18 K runs both reconstruct to an identical 5.40 MHz line. Cause is
   the field-derived auto window: on a 400 G run the fixed 300 G half-width spans
   ≈1.4–9.5 MHz, far wider than the ~5.4 MHz line, so the reconstruction is
   baseline-dominated (RMS width ≈1.2 MHz for both). With a **hand-tuned tight
   window (4–7 MHz, binning 4, 512 pts, 40 cycles)** the broadening does emerge —
   vortex RMS width ≈0.375 vs normal ≈0.255 MHz (FWHM 0.112 vs 0.088) — but
   χ²/N stays 8–13 (the same raw PSI error-scaling that keeps the time-domain
   fits at χ²ᵣ ≈ 6–10, caveat 3). So MaxEnt reproduces the physics only under
   manual steering and never at χ²/N ≈ 1; the frequency-domain
   `corpus_lifeas_vortex_lineshape` FFT overlay remains the cleaner, more
   compelling picture of the p(B) broadening. No MaxEnt scenario added.
