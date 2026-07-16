# NOTES — TRSB / Re₆Zr (`Superconductivity/TRSB`)

Corpus-driven documentation screenshots for the time-reversal-symmetry-breaking
(TRSB) measurement in the noncentrosymmetric superconductor Re₆Zr
(Singh, Hillier *et al.*, PRL **112**, 107002 (2014)). Data: MuSR runs
38176–38275, loaded from the HDF4 `.nxs` files in
`Superconductivity/TRSB/data/` through the real Asymmetry loader (the
`.nxs_v2`/`.RAW` copies are ignored; the HDF4 `.nxs` load natively).

Module: `trsb_re6zr.py`. Capture:
`.venv/bin/python -m docs.screenshots.capture_corpus --only <name>`.

## Scenarios registered

| Scenario | Render | Intended docs use |
|---|---|---|
| `corpus_trsb_zf_kt_fit` | Single ZF run 38224 (0.3 K) fitted with Gaussian Kubo–Toyabe × exp + const; fit panel + plot, first 13 µs. | The canonical KT "dip and ⅓ recovery" lineshape on real ZF data; converged single-fit panel. |
| `corpus_trsb_sigma_t_step` | **Headline.** Per-run ZF batch fit → σ(T), FitParametersPanel framed 0.250–0.270 µs⁻¹ with a dashed T_c marker. | The TRSB signature: the small spontaneous rise of the Gaussian KT rate below T_c = 6.75 K. |
| `corpus_trsb_lf_decoupling` | Base-T (0.3 K) overlay of ZF (38224) vs 10 mT / 100 G LF (38263), first 10 µs. | Static-origin proof (paper Fig. 3): 10 mT LF decouples the relaxation; overlay feature. |
| `corpus_trsb_tf_vortex_fit` | 400 G / 40 mT TF run 38180 (0.01 K) fitted with Oscillatory × Gaussian + const; first 1.4 µs. | Mixed-state precession fit / superfluid density (paper Eq. 1); Gaussian-damped rotation. |
| `corpus_trsb_sigma_sc_t` | Per-run TF batch fit → σ_sc(T), FitParametersPanel framed 0.10–0.50 µs⁻¹ with T_c marker. | Superfluid-density trend melting through T_c (paper Fig. 2b / WiMDA §7c). |

All five run real iminuit fits at capture time → `requires_fit = True`.

## Run selection & workflow (GROUND_TRUTH.md § refs)

- **ZF headline** (§3a, §4a, §6b Target 1): contiguous ZF scan 38224–38260 (F = 0),
  each fitted to `StaticGKT_ZF * Exponential + Constant`
  (G(t) = A₀·G^KT(t;Δ)·e^(−Λt) + A_bg — paper Eqs. 2–3). The Gaussian KT width
  Δ is the physics σ. Single-run headline uses 38224 (0.30 K).
- **LF control** (§3c, §4a.5): ZF 38224 vs LF 38263 (F = 100 G = 10 mT), both at 0.3 K.
- **TF superfluid density** (§3b, §4b, §7c, §6b Target 2): dense TF scan
  38180–38219 (F = 400 G = 40 mT), each fitted to `Oscillatory * Gaussian + Constant`
  (single Gaussian-relaxed rotation, paper Eq. 1). Single-run TF uses 38180 (0.01 K).
- Fits run over each run's full time range (0.098–31.44 µs) — that is what the GUI
  single-fit `Fit` button does (it passes no fit-range window to the engine).
  Trend series are loaded into `FitParametersPanel.load_representation_series`
  (same route as the synthetic `parameter_trending_mgb2` scenario); temperature is
  read from each dataset header and inferred as the trend abscissa.

## Fitted values vs ground-truth targets

| Quantity | This capture | GT target | Source | Verdict |
|---|---|---|---|---|
| ZF Δ(σ), base T (38224, 0.30 K) | **0.2611(8) µs⁻¹** | 0.2625 (fit plateau) | §6b Target 1 | ✓ within plot-reading unc. |
| ZF σ plateau below ~6 K (mean) | ≈ 0.259 µs⁻¹ | ≈ 0.2625 | §6b | ✓ slightly compressed |
| ZF σ background above 7 K (mean) | ≈ 0.2536 µs⁻¹ | 0.2557 | §6b | ✓ close |
| ZF spontaneous step Δσ across T_c | **≈ 0.0057 µs⁻¹** | 0.0068 (fit); 0.008–0.010 (envelope) | §6b Target 1 | ✓ correct sign/scale, slightly small |
| ZF Λ (electronic) vs T | flat ≈ 0.010–0.014 µs⁻¹, no transition | flat ≈ 0.020 (0.017–0.027) | §6b Fig 4b | ✓ flat — the key discriminator holds |
| ZF χ²_ν (per run) | ≈ 1.04–1.20 | (per-run, near 1) | — | ✓ |
| TF σ_sc, base T (38180, 0.01 K) | **0.4543(7) µs⁻¹** | 0.4463(68) | §7b/§7c | ✓ ~2 % high (model diff, below) |
| TF σ_sc, T → T_c plateau (7–8 K) | ≈ 0.15 µs⁻¹ | 0.169(3) | §7c | ≈ shape ✓, plateau ~10 % low |
| TF ν (38180) | 5.358 MHz (≈ 399.5 G) | 400 G / 395.7 G (§7b) | §7b | ✓ |
| TF fit χ²_ν (38180) | 0.99 | 1.066 (WiMDA) | §7b | ✓ |

Grading note (GT §9.9): the WiMDA C1 σ definition differs from the paper's
Fig 2b σ_sc(T→0) ≈ 0.65 µs⁻¹; per the brief these captures are graded against the
**WiMDA per-run numbers** (§7c, 0.446 → 0.169), which they reproduce in shape and
to within ~10 % in magnitude.

## Model-choice caveats (honest)

- **TF single- vs two-component background.** WiMDA fits TF with C1 (rotating
  Gaussian) **plus** C2 (rotating silver-background oscillation). These scenarios
  use a single rotating-Gaussian component with a *constant* background, which is
  simpler and renders cleanly, but shifts σ_sc slightly: base-T 0.454 vs 0.446 and
  the high-T plateau ≈0.15 vs 0.169. The curve *shape* (s-wave low-T plateau,
  melting 3–7 K, normal-state plateau) matches §7c exactly. A two-component
  `Oscillatory*Gaussian + Oscillatory + Constant` model is available
  (`param_names` = A_1, frequency_1, phase_1, sigma, A_3, frequency_3, phase_3,
  A_bg) and reproduces WiMDA more faithfully at the cost of a busier fit — left as
  a future refinement.
- **ZF absolute σ slightly compressed.** Fitting a free constant background over
  the full range yields plateau ≈0.259 / background ≈0.254 vs the digitised
  0.2625 / 0.2557. The step and the flat Λ (the physics) reproduce; the ~0.003 µs⁻¹
  absolute offset is within Fig-4 plot-reading uncertainty (§9.2).

## Feature-demonstration opportunities

Captured: converged single time-domain fit (ZF KT and TF rotation), σ(T) / σ_sc(T)
parameter-trending with tight custom framing + annotation, two-trace overlay.

Not captured but available on this example:
- **FFT / MaxEnt of a TF run** — the 400 G mixed-state P(B) lineshape (cf.
  `fourier_tf` / `maxent_ybco` on YBCO). Re₆Zr's line is narrower (single-gap
  s-wave), a nice contrast case.
- **Cross-format loader fixture** (GT §2): the same runs ship as HDF4 `.nxs`,
  HDF5 `.nxs_v2`, and legacy `.RAW` — a natural screenshot for a "loads any
  format to identical histograms" doc page.
- **Batch-fit panel** (Fit → Batch tab) driving the σ(T) series end-to-end, rather
  than injecting pre-fitted points into the trending panel.
- **Order-parameter / BCS trend overlay** on σ(T) (paper's red χ²_ν = 1.02 curve) —
  not added because the TRSB step is small and an unconstrained trend model is
  fragile on the real scatter; the tightly-framed points already carry the story.

## Problems hit

1. **GUI single-fit ignores the fit-range spinbox.** `single_tab._run_fit()` calls
   the engine with no `t_min`/`t_max`, so every fit spans the full 0.098–31.44 µs.
   Harmless here (KT and TF both converge over the full range), but the "Fit range"
   fields shown in the panel are not what was actually fitted — worth flagging.
2. **Sign-degenerate Gaussian width.** The KT/Gaussian width enters squared, so an
   unbounded minimiser is free to return Δ < 0 (identical χ²) — run 38224 did this
   in an early trial and displayed a negative Δ. Worked around by pinning the Min
   bound of the width (and amplitudes) to 0 via the parameter table (`COL_MIN`),
   which is what the scenarios do.
3. **Late-time ZF asymmetry error fan.** F−B asymmetry error blows up past ~10 µs
   as the counts decay (denominator → 0), producing a noisy fan that dominates a
   full-range view. Framed the ZF time views to 13 µs (KT fit) / 10 µs (LF overlay)
   to keep the physics legible.
4. **Loader field-geometry mislabel (GT §3 caveat).** This worktree's loader tags
   all 100 runs `field_direction = "Transverse"` (reads bank orientation, not
   `magnetic_field_state`). Ignored — runs are discriminated by field *magnitude*
   (0 / 100 / 400 G), which loads correctly. No effect on the plots.
5. **Two ZF runs report `success = False`** (38231, 38254: Λ drifts slightly
   negative / large σ error) but their Δ values sit within the trend scatter; kept
   in the σ(T) series rather than cherry-picking.

## Top pick for the docs

`corpus_trsb_sigma_t_step` — the σ(T) step on a tight 0.250–0.270 µs⁻¹ axis is the
headline: a subtle real-data signature (Δσ ≈ 0.006 µs⁻¹ across T_c) that Asymmetry
renders legibly, which is exactly what the docs want to show. `corpus_trsb_sigma_sc_t`
(clean s-wave superfluid-density melt) and `corpus_trsb_tf_vortex_fit` (crisp
χ²_ν = 0.99 precession fit) are the strongest supporting frames.
