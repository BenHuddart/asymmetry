# NOTES — Muonium reaction with maleic acid (Chemistry)

Module: `maleic_kinetics.py`. Corpus example:
`Chemistry/Muonium reaction with maleic acid` (EMU `.nxs`, runs 78251–78302,
ISIS/RAL 2018). Spec: the example's `GROUND_TRUTH.md`.

This is the corpus's **chemical-kinetics** example: muonium (Mu = μ⁺e⁻, a light
H-atom isotope) adds across the C=C bond of maleic acid in aqueous solution.
Pseudo-first-order in Mu ⇒ λ_Mu = λ₀ + k_Mu·[x]. Small TF-µSR: **2 G** to see
the triplet-Mu precession (ν_Mu ≈ 2.79 MHz), **100 G** for the diamagnetic
fraction.

## Scenarios registered

| Scenario | Render | Intended docs use | GT § |
|---|---|---|---|
| `corpus_maleic_mu_precession` | GUI fit panel (MainWindow): converged relaxing-Mu-oscillation fit on the deox-water 2 G run **78251**, rebinned ×3, zoomed to ~2.5 µs (~7 Mu cycles); red fit curve traces ν_Mu ≈ 2.79 MHz. | The core "muonium in water" fit — the fit-panel surface on a real chemistry run; teaching contrast that ν_Mu is **103×** the diamagnetic 2 G Larmor (0.027 MHz). | §4 (Mu model), §6 |
| `corpus_maleic_concentration` | Standalone matplotlib: bold fitted Mu component curves for deox water / half / full over faint binned data — same A_Mu/φ/ν, only λ_Mu differs, so they start together and damp apart. | Data-handling / physical-intuition step: relaxation visibly faster with [x] (the reaction consuming Mu). | §4, §5 |
| `corpus_maleic_kmu_trend` | **Headline** — FitParametersPanel: per-run λ_Mu vs a **manual concentration column** (`Maleic conc (rel. units)`), fitted line λ_Mu = λ₀ + k_Mu·[x]; slope = k_Mu. | The key feature demo (manual fit-table column — concentration is in **no** file metadata) + the headline kinetics result. | §3, §4 (trend 1), §6, §9.1 |
| `corpus_maleic_arrhenius` | **Reworked for PR 248** — real trending panel: full-conc. λ_Mu(T) over 278–358 K with Y→`ln` and X→`reciprocal` (1/T) axis transforms and a `Linear` model fit → E_a. Plain `ln` preset works directly (monotone rise, no plateau baseline — the clean-reproduction contrast to LLZ). | Distinctive extra: the overnight T-scan → activation energy; the *simple* axis-transform case. | §4 (trend 2), §6 |

All four run real iminuit fits at capture time (`requires_fit = True`).

## Workflow / model choices (GT §4)

- **Mu model:** `Oscillatory·Exponential + Exponential`. The Osc·Exp is the
  relaxing Mu precession (λ_Mu = the additive-free `Lambda_2`); the second
  additive Exponential is the strongly-**sloping diamagnetic baseline** (the 2 G
  asymmetry falls from ~18 % to ~5 % over 8 µs). A bare additive `Constant` on
  top of that slow Exp is **degenerate** at 2 G — MINUIT flags an invalid
  minimum (`success=False`) and the GUI then refuses to draw the fit curve.
  Dropping the Constant fixes it; the fits converge cleanly with identical λ_Mu.
- **Fixed ν_Mu = 2.788 MHz** (γ_Mu ≈ 1.394 MHz/G × 2.00 G, a physics constant,
  GT §4/§6). On this noisy single-run data a floating frequency latches onto
  per-bin noise for the fast-relaxing maleic runs → the fit fails with
  "call limit reached / invalid parameters". Fixing ν keeps it conditioned.
- **Anchored A_Mu and phase.** The Mu *fraction* (amplitude) that forms is set
  by the water and is common across the concentration series; only its
  relaxation changes. So A_Mu and φ are fitted once on the clean deox-water run
  (A_Mu ≈ 1.88 %, φ ≈ −0.31 rad) and **held** for every other run — only λ_Mu
  (and the baseline) float. Without this the fast-relaxing maleic runs collapse
  the Mu term into an unphysical early spike (A_Mu → 20 %, λ → ~9 µs⁻¹).
- Run selection (GT §3): concentration series at ~290 K = deox water **78251**
  ([x]=0), quarter **78279** (1), half **78277** (2), full **78257** (4)
  (ratio 1:2:4). Full-conc. T-scan (2 G Mu) = 78259/61/57/63/65/67/69/71/73/75
  at 278–358 K, warm-started in ascending T.

## Fitted values vs ground truth

| Quantity | This module | GT / literature (§6) | Note |
|---|---|---|---|
| ν_Mu at 2 G | 2.788 MHz (fixed) | ≈ 2.78 MHz | physics const, γ_Mu×2 G |
| λ₀ (deox water) | 0.46 µs⁻¹ | intercept, "small for deox" (deliverable) | clean, single-minimum fit |
| λ_Mu(quarter/half/full)@290 K | 1.63 / 2.13 / 2.80 µs⁻¹ | monotone ↑ with [x] (deliverable) | anchored fits |
| **k_Mu (slope)** | **0.70 ± 0.02 µs⁻¹ per rel-unit**, R² ≈ 0.83 | slope of λ_Mu vs [x] (deliverable) | **relative-conc units only — no molarity in any source file (GT §9.1)**; cannot convert to the ≈1.1×10¹⁰ M⁻¹s⁻¹ literature value |
| E_a (full-conc. λ(T)) | **7.3 ± 0.4 kJ/mol** (PR 248 transform panel: Y→ln, X→1/T, Linear; χ²ᵣ ≈ 0.9) — identical to the old standalone-matplotlib `ln λ vs 1/T` fit | ≈ 17.6 kJ/mol (diffusion-limited) | see problem 3 below |

## Feature-demonstration opportunities

- **Axis transforms — the *simple* case (PR 248).** `corpus_maleic_arrhenius`
  uses the `ln` Y-preset + `reciprocal` X-preset directly, no baseline surgery:
  λ_Mu(T) rises monotonically (2.30 → 4.84 µs⁻¹), so ln λ_Mu vs 1/T is a straight
  line out of the box and the Linear fit reproduces the old standalone figure's
  Eₐ = 7.3 kJ/mol exactly. This is the clean contrast to the LLZ ν(T) case, where
  a plateau baseline forced a *Custom* `log(x−c)` transform — a nice paired
  "presets suffice / Custom needed" story for the docs. The `ln` Y-transform also
  correctly greys out the per-parameter **log** axis-scale checkbox (guard works).
- **Manual fit-table column** (headline) — concentration is relative-only and in
  no metadata; the panel's custom-x route (`set_custom_x_fields`) is exactly the
  intended surface. Strongest single reason this example earns a doc page.
- **Parameter fixing in the GUI fit panel** — scenario 1 checks the `f (MHz)`
  Fix box programmatically (via the param-table `cellWidget(row, 2)` checkbox);
  a genuine demo of holding a physics constant during a fit.
- **Diamagnetic 100 G signal** (NOT captured): run **78252** shows a gorgeous
  clean 1.36 MHz diamagnetic Larmor precession (FFT peak ~6000, ampl ~15 %) —
  would make an excellent FFT / "muon fractions" render if a 5th scenario is
  wanted. The 100 G runs 78258/60/.../74 are the logbook-only ones (GT §3 note).

## Problems hit (honest)

1. **Weak, noisy Mu signal.** A_Mu ≈ 2 % sits under ~25 %/bin statistical noise
   at the raw 16 ns EMU base. Only the deox-water run fits cleanly with all
   params free; the maleic runs need the fixed-ν + anchored-A_Mu constraints
   above. The trend λ_Mu values are robust, but per-run χ²ᵣ is high (~20–60,
   small per-bin errors + approximate baseline) — cosmetic, not a wrong minimum.
2. **`success=False` with a good minimum.** The original Osc·Exp+Exp+Const model
   returned sensible λ_Mu but `result.success=False` (degenerate baseline), which
   the trend code tolerated but the GUI fit panel would not draw. Resolved by
   dropping the redundant Constant (see model note). Worth flagging as a
   product-side sharp edge: a converged-but-flagged fit is silently un-plotted.
3. **E_a ≈ 7 kJ/mol, ~half the ≈17.6 kJ/mol literature diffusion value.** Using
   the full-conc. λ_Mu(T) *directly* as the Arrhenius observable (rather than
   k_Mu(T) = [λ(T)−λ₀(T)]/[x], which needs a water T-scan we don't have — water
   is measured only at 290 K, GT §3) dilutes the temperature dependence with the
   ~T-independent λ₀, flattening the slope. The λ(T) trend itself is clean and
   monotone (2.3 → 4.9 µs⁻¹, 278 → 358 K); the render annotates the literature
   value and the plot is pedagogically correct, but the extracted number is a
   lower bound. Honest caveat, consistent with GT §9.
4. **Linearity R² ≈ 0.83.** The water→quarter step is steeper than quarter→full
   (mild saturation), so the λ_Mu vs [x] line is good but not perfect — real
   single-run data, and possibly the deox/aerated inconsistency GT §9.5 flags
   (the deox-water intercept vs possibly-aerated maleic solutions).

## Top pick

**`corpus_maleic_kmu_trend`** — the headline. It reproduces the deliverable
(k_Mu slope + λ₀ intercept) *and* is the cleanest demonstration in the whole
corpus of the **manual-column** feature, which this example forces because the
independent variable (concentration) exists in no data file. Runner-up:
`corpus_maleic_mu_precession` for a striking real "muonium in water" fit-panel
image.

## PR 248 round 2 (re-test, 2026-07-12)

Re-tested commit 4a91420 on the real `corpus_maleic_arrhenius` panel (10 real
λ_Mu(T) fits + Linear trend fit; the *simple* preset-transform case).

- **CONFIRMED FIXED — unit-aware transformed labels.** The re-captured render
  now reads:
  - X (reciprocal on T): **`1/T (K⁻¹)`**
  - Y (`ln` preset on λ): **`ln[λ (µs⁻¹)]`** — the dimensioned quantity is
    bracketed under the log. Section chip: `1/x · ln y`.
  This is the clean contrast to the LLZ Custom case: the `ln` preset linearises
  λ_Mu(T) directly (monotone rise, no plateau baseline), a straight line out of
  the box, and the per-parameter `log` axis-scale checkbox is correctly greyed.
- **Physics regression: none.** The clean monotone Linear fit is unchanged
  (E_a ≈ 7.3 kJ/mol — the lower-bound caveat of problem 3 still applies).
