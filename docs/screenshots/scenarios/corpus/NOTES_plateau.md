# NOTES — Dynamics in a magnetic plateau system (Ca₃Co₂O₆)

Corpus example: `Magnetism/Dynamics in a magnetic plateau system`
Module: `plateau_ca3co2o6.py` · Ground truth: that folder's `GROUND_TRUTH.md`.
Real HiFi HDF4-NeXus `.nxs` corpus files, runs **9023–9051**: a TF20 calibration
(9023, 300 K) plus a 15 K longitudinal-field decoupling scan 0 → 3.8 T
(9031–9051). Same data/campaign as Baker, Lord & Prabhakaran, *J. Phys.:
Condens. Matter* **23**, 306001 (2011), arXiv:1105.2200 (author-matched in the
NeXus headers, GT §11). Ca₃Co₂O₆ is a frustrated Ising-chain magnet with a
partial 1/3 magnetization plateau; the muon decouples static from **dynamic**
internal fields inside it.

## Scenarios registered

| name | render | intended docs use | requires_fit |
|---|---|---|---|
| `corpus_plateau_lf_overlay` | Raw LF spectra at ZF / 0.5 / 1.5 / 3.5 T (runs 9031/9039/9045/9050) overlaid, 0–12 µs. The 3.5 T trace is flat and high (~40 %); low-field traces relax fast and sit lower — the decoupling + asymmetry-recovery picture. | Data-handling / overlay feature; qualitative decoupling. | no |
| `corpus_plateau_exp_fit` | Converged `Exponential + Constant` fit on the 1.0 T run (9044): **A = 8.72 %, λ = 1.332 µs⁻¹, A_bg = 7.23 %, χ²ᵣ = 1.25**. Display bunched ×5, framed 0–10 µs so the decay + fit dominate. | Core analysis step — the per-run λ that feeds the trend. | **yes** |
| `corpus_plateau_lambda_field` | λ(µ₀H) trend in the parameter-trending panel (paper Fig. 2(a)): three-regime falloff, ~4.2 µs⁻¹ at 0.2 T → ~0.3 µs⁻¹ at 3.8 T. B on the panel's native "B (G)" axis. | Trend-building step; the λ(B) shape. | **yes** |
| `corpus_plateau_redfield` | **Headline.** λ⁻¹ vs µ₀²H² (paper Fig. 2(b)) as a matplotlib figure: plateau points on the linear Redfield fit, the 3.8 T saturated point excluded, annotated slope/intercept → Δ, τ. | Parameter-trending → Redfield linearity; the Δ/τ result. | **yes** |

## Run selection & workflow (GROUND_TRUTH refs)

- **Per-run model (GT §4 / eq.(3)).** `Exponential + Constant`
  = `A·exp(−λt) + A_bg`, fit 0–16 µs via the core `FitEngine` (the engine the
  single-fit panel drives). The additive `A_bg` absorbs the large field-growing
  baseline from positron spiralling (A_bg ≈ 7 % at 1 T rising to ~37 % at
  3.5 T; GT §4).
- **Warm-starting in field order (README lesson).** λ is seeded high (~9 µs⁻¹)
  and carried downward run-to-run. Essential on the low-field runs, which are
  degenerate (see Problems).
- **λ(B) trend runs.** 0.2 → 3.8 T (9036–9051). ZF (9031) and 0.1 T (9035)
  excluded: below 0.2 T the fast relaxation is unresolved at the ISIS pulse
  width and the asymmetry is suppressed, so the exponential is degenerate and λ
  unphysical (GT §4/§9). Runs 9024–9030 (thermally unsettled cooldown, GT §9)
  and 9032–9034 (redundant ZF) not used.
- **Redfield plateau fit (GT §4/§6).** λ⁻¹ = slope·(µ₀H)² + intercept,
  weighted, over **0.5–3.6 T** (runs 9039–9050). Solve
  `slope = τ/(2Δ²)`, `intercept = 1/(2γ_µ²Δ²τ)` with γ_µ = 2π×135.5 MHz/T for
  (Δ, τ). The 3.8 T point (9051, saturated) is plotted but excluded from the fit.

## Fitted values vs ground truth

### Per-run λ(µ₀H) (fitted here vs digitised Fig. 2(a), GT §11)

| µ₀H (T) | λ fitted (µs⁻¹) | λ GT (MHz) | | µ₀H (T) | λ fitted (µs⁻¹) | λ GT (MHz) |
|---:|---:|---:|---|---:|---:|---:|
| 0.2 | 4.17 | ~3.4 | | 1.0 | 1.33 | ~1.35 |
| 0.3 | 2.66 | ~2.4 | | 1.5 | 0.90 | ~0.85 |
| 0.4 | 2.50 | ~2.3 | | 2.0 | 0.69 | ~0.65 |
| 0.5 | 2.05 | ~1.95 | | 2.5 | 0.47 | ~0.5 |
| 0.6 | 1.86 | — | | 2.9 | 0.33 | ~0.4 |
| 0.7 | 1.78 | — | | 3.2 | 0.24 | ~0.3 |
| 0.8 | 1.62 | — | | 3.5 | 0.28 | ~0.3 |
| 0.9 | 1.44 | — | | 3.8 | 0.28 | ~0.3 |

Plateau points (0.5–3.8 T) sit right on the digitised targets; the 0.2–0.4 T
points run slightly high (steep-drop regime, large read-uncertainty in the GT).

### Redfield line → Δ, τ (headline)

| Quantity | Fitted here | GT target (paper §3/§6) | Note |
|---|---:|---:|---|
| slope d(λ⁻¹)/d(µ₀²H²) | 0.278(9) µs T⁻² | ≈ 0.267 µs T⁻² (GT §11) | ✓ |
| intercept (H→0) | 0.434(19) µs | ≈ 0.48 µs (GT §11) | ✓ (~9 % low) |
| **Δ (= σ_int)** | **41.1 mT** | **40.6(3) mT** | ✓ within ~0.5 mT |
| **τ** | **940 ps** | **880(30) ps** | ✓ ~7 % high (~2σ) |

Δ lands essentially on the paper value; τ is ~7 % high, traceable to the two
scattered high-field points (3.2 T low / 3.2 T² region — the paper notes these
"scatter around/below" the line, GT §11) pulling the slope up slightly. The fit
is, as the paper states, insensitive to small changes in the window; dropping
3.2 T brings τ closer to 880 ps but the full 0.5–3.6 T window is kept for
faithfulness to GT §4.

## Feature-demonstration opportunities

- **HiFi HDF4-NeXus loader** — real ISIS `.nxs` (HDF4-based) read natively;
  field/temperature metadata populate the browser (`Ca3Co2O6 T=15 F=…`). Good
  "we read HiFi format" evidence.
- **Redfield linearisation** — the λ⁻¹-vs-µ₀²H² transform is the distinctive
  physics of this example. The parameter-trending panel cannot square/invert its
  axes, so the headline is a matplotlib figure (mgb2 pattern) — a genuine
  candidate feature request for the trending panel (a "derived-axis" mode).
- **Not captured but available:** (a) the **TF20 α-calibration** on run 9023
  (the AlphaCalibrationDialog pattern from `ionic_motion_llz.py`) — GT §4 Q1
  data-prep step. (b) A **P_z(t) waterfall** across the full scan. (c) The
  **sub-plateau regime** (Δ ≃ 30 mT, τ ≃ 5 ns, GT §6) if the <0.5 T fits could
  be stabilised (they can't with a single exponential — see below).

## Problems / quirks hit

- **Low-field (<0.2 T) fits are degenerate.** At ZF and 0.1 T the true λ (~7.7,
  ~6 µs⁻¹) relaxes the signal within the first ~0.1 µs, unresolved at the ISIS
  pulse width; the observed asymmetry is also suppressed (GT §4). The remaining
  data is a flat baseline, so `A·exp(−λt)+bg` is under-determined — λ collapses
  to a spurious near-zero minimum (0.1 T fitted 0.005 µs⁻¹ from a high seed).
  These points are physically meaningless and are **excluded from the λ(B)
  trend**; this is exactly the GT §9 caveat, not a program bug. Only Δ and τ
  from the plateau are grading targets anyway (GT §9).
- **Baseline grows strongly with field.** A_bg rises from ~7 % (1 T) to ~37 %
  (3.5 T) via positron spiralling (GT §4), so the raw-spectra overlay separates
  the traces vertically by baseline as much as by relaxation. Framed Y 0–48 % to
  hold all four; the decoupling (flattening) still reads clearly.
- **Late-time F–B noise.** Past ~8 µs the F–B asymmetry fans out as counts
  vanish (±100 % bins). The fit runs on the unbinned 0–16 µs data (χ²ᵣ = 1.25);
  the `corpus_plateau_exp_fit` *display* is bunched ×5 and framed 0–10 µs so the
  decay is clean. The app tags χ²ᵣ = 1.25 as "poor" (its threshold wording) —
  the fit is in fact good.
- **Trend axis is gauss.** The panel labels the field axis "B (G)" (0–38000),
  not tesla; fine for Fig. 2(a), just note the unit in any caption.

## Top pick for the docs

`corpus_plateau_redfield` — the λ⁻¹-vs-µ₀²H² Redfield line is the headline result
and reproduces the paper's Δ = 40.6 mT / τ = 880 ps within uncertainty
(41.1 mT / 940 ps) with the derivation annotated on the figure. Pair it with
`corpus_plateau_exp_fit` (the per-run exponential that produces each λ point) as
the two-image story; `corpus_plateau_lambda_field` (Fig. 2(a)) is the natural
bridge between them.
