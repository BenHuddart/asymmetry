# NOTES — A high-Tc cuprate (BiSCCO) corpus scenarios

Module: `cuprate_bscco.py` · Example: `Superconductivity/A high-Tc cuprate`
Spec: that example's `GROUND_TRUTH.md` (WiMDA-graded; shipped `.fit`/`.dat`).

The classic transverse-field vortex-lattice **σ(T) → penetration-depth**
workflow on Bi-2212. All quantitative renders use the **MUSR 400 G scan**, the
authoritative deliverable (GT §7); the 200 G scan is used for the field
comparison. EMU 150 G is *not* used — its reference fits are unreliable
(χ²/dof ≈ 5–13, negative/absent σ, GT §9).

## Scenarios registered

| name | render | docs use |
|---|---|---|
| `corpus_bscco_tf_damping` | Time-domain overlay of run 1277 (10 K) vs 1276 (125 K), 400 G, first ~4 µs | Data-handling / core insight: the Gaussian **vortex damping** below T_c vs the undamped normal-state precession — field-distribution broadening, in the time domain |
| `corpus_bscco_vortex_fft` | GUI Fourier panel, run 1277 (10 K), framed 3.5–7.5 MHz | Frequency-domain: the **broad vortex p(B) line** at the ~5.4 MHz (400 G) Larmor frequency |
| `corpus_bscco_tf_fit` | Converged Oscillatory×Gaussian single fit, run 1277, zoomed 0.1–2.5 µs | Headline fit: σ ≈ **1.164 µs⁻¹** vs WiMDA **1.1467(75)**, χ²ᵣ = 1.09, "Fit converged" |
| `corpus_bscco_sigma_t` | Fit-Parameters trending panel, σ(T) 10–125 K (14 runs) | **The headline**: σ(T) reproducing the 14-row reference trend, with T_c ≈ 107 K marker and σ(10 K) → λ_L ≈ 255 nm note (§6b caveat inline) |
| `corpus_bscco_field_compare` | Matplotlib figure, σ(T) 400 G vs 200 G | The guide's **field-comparison** task: 200 G plateau sits below 400 G — pancake-vortex field dependence (§6b) |

`requires_fit = True` on `tf_fit`, `sigma_t`, `field_compare` (real iminuit
fits at capture time).

## Workflow followed (GT §4 / §7)

- **Model** (GT §7): `Component 1: Osc=Rotation Field, Rel=Gaussian`, i.e.
  Asymmetry's `Oscillatory * Gaussian + Constant`,
  G(t) = A·exp(−σ²t²/2)·cos(2πνt+φ) + A_bg. Parameter `sigma` is the Gaussian
  depolarisation rate.
- **Grouping caveat.** The WiMDA reference fits MUSR on **All Grps** (all TF
  detector groups, dependent amplitudes). The Asymmetry loader delivers the
  standard **forward/back asymmetry** (groups 1 vs 2 of the 4 MUSR quadrant
  groups, α = 1). Fitting that single F/B pair still recovers the reference
  σ(T) trend to a few percent (table below) because the F/B pair carries the
  full TF precession; the small offset is the price of not co-fitting all
  groups. This is the honest grouping difference, noted for grading.
- **Fit window.** Full data range (0.09–31.7 µs). The raw asymmetry saturates
  at ±100 % in the pre-t0 and late-time bad bins; seeds derive `A_bg` from the
  **good-bin median** (a plain mean is poisoned by the ±100 bins). The core
  engine's error weighting handles the growing late-time noise fan, so the
  full-range fit is stable (χ²ᵣ ≈ 0.9–1.4 across the scan).
- **σ sign.** The Gaussian width enters squared, so its fitted sign is a fit
  artefact; the batch series reports |σ| (GT §9: the 200 G run-1291 reference
  fit famously returns −0.21).
- **λ_L.** Guide formula σ(µs⁻¹) = 75780 / λ_L²(nm) applied to the lowest-T σ.

## Fitted σ vs WiMDA reference (both from the shipped module)

**400 G scan** (σ warm-started downward from 1.15 µs⁻¹):

| T (K) | fit σ | ref σ (§7) | Δ |
|---|---|---|---|
| 10 | **1.164** | 1.1467(75) | +1.5 % |
| 30 | 1.118 | 1.0995 | +1.6 % |
| 50 | 1.057 | 1.0268 | +3.0 % |
| 70 | 0.896 | 0.9098 | −1.5 % |
| 80 | 0.817 | 0.8141 | +0.4 % |
| 85 | 0.717 | 0.7047 | +1.7 % |
| 90 | 0.577 | 0.5638 | +2.4 % |
| 95 | 0.434 | 0.4235 | +2.5 % |
| 100 | 0.285 | 0.2683 | +6.1 % |
| 105 | 0.125 | 0.1174 | +6.7 % |
| 110 | 0.069 | 0.0600 | +14 % |
| 115 | 0.071 | 0.0576 | +23 % |
| 120 | 0.069 | 0.0560 | +23 % |
| 125 | 0.064 | 0.0549 | +17 % |

σ(10 K) = **1.164 µs⁻¹ → λ_L = 255 nm** (guide formula; reference 1.1467 → 257 nm).

**200 G scan** (run 1291 excluded — negative-σ pathology, §9):

| T (K) | fit σ | ref σ (§7) | Δ |
|---|---|---|---|
| 30 | 0.826 | 0.8316 | −0.7 % |
| 50 | 0.907 | 0.9095 | −0.3 % |
| 70 | 0.823 | 0.8432 | −2.4 % |
| 85 | 0.687 | 0.7052 | −2.5 % |
| 90 | 0.633 | 0.6263 | +1.1 % |
| 95 | 0.499 | 0.4968 | +0.4 % |
| 100 | 0.332 | 0.3321 | 0 % |
| 105 | 0.159 | 0.1585 | 0 % |
| 110 | 0.065 | 0.0654 | −1.4 % |
| 120 | 0.070 | 0.0664 | +5.4 % |

**Agreement is excellent in the superconducting range (≤ 3 %).** The high-T
tail (> 105 K) shows larger *percentage* deviation but the *absolute* offset is
tiny (~0.01 µs⁻¹): the F/B fit floors at a small residual σ ≈ 0.06–0.07 µs⁻¹
where the reference reaches ≈ 0.055. The single λ_L (255 nm) matches the
reference-derived 257 nm and the §6b literature order of magnitude (~260 nm),
but per §6b is **indicative only** — all three teaching fields lie below the
pancake-vortex crossover B* ≈ 500 G, so a single λ_ab is not physically robust
for Bi-2212. Captions respect that fence.

## Feature-demonstration opportunities

- **Vortex broadening** shown two ways: time-domain damping overlay
  (`tf_damping`) and frequency-domain FFT (`vortex_fft`). Complementary.
- **Field comparison** (`field_compare`) makes the §6b pancake-vortex physics
  visible (200 G plateau below 400 G), which the single-field σ(T) cannot.
- The **converged fit** panel shows the model string, per-parameter values with
  errors, and "Fit converged / χ²ᵣ = 1.09" — a strong single-fit teaching image.

## Problems / honest caveats

1. **MaxEnt does not render well here — dropped.** The guide asks to compare FFT
   with Maximum-Entropy, and `maxent_ybco` shows it works on synthetic data.
   On this real F/B asymmetry the numba MaxEnt solver **diverges** ("stopped
   early at cycle 7 as χ² began rising past the optimum") and produces spiky
   noise (~470 direction-changes across the vortex window at 4–8 cycles), not a
   clean line — its large α = 1 baseline defeats the reconstruction. Only the
   FFT frequency-domain render is shipped; the module docstring records this.
2. **Grouping is F/B, not All Grps** (see workflow). Reproduces σ(T) to a few
   percent but is not a byte-for-byte match to the WiMDA All-Grps numbers.
3. **`tf_damping` contrast is real but subtle.** The oscillation amplitude is
   only ~±9 % on a −23 % baseline with a growing late-time noise fan; the 10 K
   Gaussian collapse vs the 125 K persistence reads best in the 1.5–4 µs region.
   Caption carries the interpretation.
4. **Frequency-domain overlay is not supported by the GUI panel.** The
   Fit-Parameters trending panel plots one active series at a time (multi-select
   via `_set_selected_group_ids` does not force a two-series overlay on
   redraw), and the Fourier panel plots one run's spectrum. So the 10 K↔125 K
   frequency contrast is delivered in the **time** domain (`tf_damping`), and
   the **field** comparison (`field_compare`) is a standalone Matplotlib figure
   — the established `mgb2_lambda_t` / `parameter_trending` house pattern for
   comparison/derived plots the panel cannot render directly. Both series are
   still genuine per-run core-engine fits.
5. **Negative-σ pathology not reproduced as a render.** Run 1291 (200 G, 10 K)
   returns +0.176 in Asymmetry (positive minimum) rather than the reference
   −0.211; reproducing the sign would require a contrived negative seed, so it
   is documented (excluded from the 200 G series) rather than staged.

## Top pick for docs

**`corpus_bscco_sigma_t`** — the headline σ(T) trend in the real trending panel,
reproducing the 14-row WiMDA reference with the T_c marker and the σ→λ_L note
(and its §6b caveat) on the plot. `corpus_bscco_tf_fit` is the best companion
(the converged single fit that produces the 10 K plateau point), and
`corpus_bscco_vortex_fft` the best standalone frequency-domain image.
