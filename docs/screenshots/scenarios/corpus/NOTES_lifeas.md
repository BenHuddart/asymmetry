# NOTES — LiFeAs (pnictide superconductor) corpus scenarios

Module: `lifeas_pnictide.py` · Example: `Superconductivity/LiFeAs`
Spec: that example's `GROUND_TRUTH.md` (paper-graded — no teaching docx; the
reference is Pratt *et al.*, PRB **79**, 052508 (2009)).

The transverse-field vortex-lattice **B_rms(T) → penetration-depth** workflow on
the "111" iron-arsenide superconductor LiFeAs. Real PSI GPS `.bin` runs, two
samples: **Sample 1 "LFA"** (3366–3387, T_c = 16 K) and **Sample 2 "LFA_2"**
(3662–3697, T_c ≈ 12 K). The authoritative deliverable is the 40 mT (400 G)
temperature scan of Sample 1 (runs 3366–3373), which reproduces the paper's
Fig. 1 B_rms(T) plateau → λ_ab = 195 nm.

## Scenarios registered

| name | render | docs use |
|---|---|---|
| `corpus_lifeas_pair_select` | Matplotlib 2-panel, base-T run 3366: default Forward/Back (cancelled mush) vs Up/Down transverse pair (clean 5.44 MHz damped precession) | **Data-handling**: the essential detector-pairing choice for these spin-rotated (WED) runs — the loader's default pair cancels the signal |
| `corpus_lifeas_tf_fit` | GUI single-fit panel, two-Gaussian TF fit on 3366 (1.5 K, 40 mT, Up/Down): σ_2 = 1.189, σ_4 = 0.178, χ²ᵣ = 9.71, model string + param table | **Core fit**: the vortex signal σ_2 with the weakly-relaxing Ag background σ_4 → B_rms ≈ 1.96 mT (λ_ab ≈ 195 nm) |
| `corpus_lifeas_brms_t` | Matplotlib, two-sample B_rms(T) at 40 mT: digitised Fig. 1 (§11) both samples + real Asymmetry S1 fit overlay; λ_ab lines + T_c markers | **The headline**: plateaus ≈ 1.9 / ≈ 1.2 mT, T_c ≈ 16 / 12 K, λ_ab = 195/244 nm via Eq. (3) |
| `corpus_lifeas_vortex_lineshape` | Matplotlib FFT overlay, S1 40 mT: 18 K narrow (nuclear) vs 1.5 K broad (vortex) p(B) line at 5.44 MHz | **Normal vs SC**: the field-distribution broadening below T_c that B_rms measures |

`requires_fit = True` on `tf_fit` and `brms_t` (real iminuit fits at capture).

## Workflow followed (GROUND_TRUTH §4 / §11)

- **Model** (§4): above T_c a single Gaussian (nuclear); below T_c the vortex
  broadening adds in quadrature, σ² = σ_VL² + σ_n² (Eq. 2), and B_rms = σ_VL/γ_µ
  (γ_µ = 0.8516 µs⁻¹ mT⁻¹). Powder London limit (Eq. 3):
  B_rms = √0.00371·φ₀/(3^{1/4}λ_ab)² → **195 nm ↔ 1.912 mT, 244 nm ↔ 1.221 mT**
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

## Fitted B_rms vs paper (Sample 1, 40 mT, from the shipped module)

σ_n(18 K, single Gaussian) = 0.156 µs⁻¹ (Asymmetry conv.).

| T (K) | fit B_rms (mT) | Fig. 1 digitised §11 (mT) | Δ | kept in headline |
|---|---|---|---|---|
| 1.5  | **1.958** | 1.93 | +1.5 % | ✓ |
| 4.0  | 1.955 | 1.77 | +10 % | ✓ |
| 7.0  | 1.758 | 1.48 | +19 % | ✓ |
| 10.0 | 1.189 | 1.04 | +14 % | ✓ |
| 12.0 | (1.166) | 0.74 | — | ✗ (component swap) |
| 14.0 | (1.250) | 0.43 | — | ✗ |
| 16.0 | (0.733) | 0.11 | — | ✗ |
| 18.0 | (0.789) | 0.02 | — | ✗ (= σ_n, B_rms→0) |

**Base-T plateau is spot-on** (1.958 vs 1.93 mT → λ_ab ≈ 193 nm vs the printed
195(2) nm). The 4–10 K points run ~10–19 % high — the two-Gaussian split slightly
over-attributes width to the vortex component as the SC signal weakens; above
10 K the split degenerates (component swap) and points are dropped. Sample 2's
B_rms(T) comes entirely from the digitised Fig. 1 (§11): its supplied `.bin`
runs are 1.5/20 K field-dependence *pairs*, not a T-scan (see problems).

## B_rms ↔ λ_ab vs paper (hard targets)

| Quantity | Paper (GROUND_TRUTH §6/§11) | This work |
|---|---|---|
| B_rms plateau, S1 (40 mT, low T) | ≈ 1.8–1.9 mT (Fig. 1); digitised 1.93 | **1.96 mT** (real two-Gaussian fit, run 3366) |
| λ_ab, Sample 1 (T_c = 16 K) | **195(2) nm** | ≈ 193 nm (from B_rms = 1.958 via Eq. 3) |
| B_rms plateau, S2 (40 mT, low T) | ≈ 1.0 mT (§6); digitised 1.31 | digitised only (no S2 T-scan in corpus) |
| λ_ab, Sample 2 (T_c ≈ 12 K) | **244(2) nm** | annotated on headline (digitised curve) |
| Eq. (3) check | 195 nm → 1.912 mT; 244 nm → 1.221 mT | reproduced numerically (√ prefactor) |

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
   ghosts). The headline keeps only the well-defined low-T points; the full
   B_rms(T) shape is carried by the digitised §11 curve. χ²ᵣ ≈ 6–10 across the
   scan (raw error scaling on rebinned PSI data) — the fits converge and the
   parameters are physical, but this is not a χ²ᵣ ≈ 1 regime.
4. **Sample 2 has no 40 mT T-scan in the corpus** (GROUND_TRUTH §3 note). The
   supplied LFA_2 runs are 1.5 K vs 20 K field-dependence pairs (7–200 G). So
   the headline's Sample-2 curve is the digitised Fig. 1 (§11); a real S2 base-T
   fit (run 3663, 200 G) gives B_rms ≈ 1.56 mT — off the 40 mT ≈ 1.31 mT value
   (different field / incipient field-induced magnetism), so it is deliberately
   **not** overlaid to avoid a misleading apples-to-oranges point.
5. **No reference-program output exists** (§7) — no WiMDA `.fit`, no teaching
   docx. Grading is against the paper's printed λ_ab/n (hard) and the digitised
   Fig. 1 B_rms(T) (shape/magnitude), per §11.

## Top pick for docs

**`corpus_lifeas_brms_t`** — the headline two-sample B_rms(T), with the real
Asymmetry Sample-1 fits landing on the paper's Fig. 1 plateau (1.96 vs 1.93 mT)
and the λ_ab = 195/244 nm conversions annotated via Eq. (3). Best companion is
`corpus_lifeas_tf_fit` (the converged two-Gaussian fit that produces the base-T
plateau point). `corpus_lifeas_pair_select` is the most *distinctive* image this
example offers — the detector-pairing insight no other corpus example shows so
starkly.
