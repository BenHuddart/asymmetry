# NOTES — A molecular antiferromagnet (Magnetism)

Module: `molecular_afm.py` · Example: `Magnetism/A molecular antiferromagnet`
Data: ISIS/MUSR NeXus-v1 **HDF4**, `MUSR00017094.nxs`–`MUSR00017104.nxs`
(11 runs, April 2008). GROUND_TRUTH:
`Magnetism/A molecular antiferromagnet/GROUND_TRUTH.md`.

Sample Ni(9S3)₂[Ni(bdt)₂]₂ (two inequivalent Ni sites: Ni²⁺ S = 1, Ni⁺ S = ½;
coexisting ferri-/antiferromagnetic chains). Data resolve through
`_DATA = "Magnetism/A molecular antiferromagnet/Data/MUSR000%d.nxs"`. HDF4 loads
directly — no conversion, no `Data_hdf5/` sibling (GT §2). File metadata:
sample "Training Course Magnetic Organic Stuff", user "Aidy",
`experiment_number` = −1000, on-file `magnetic_field_state` = "TF" for every run
(instrument default; the ZF runs carry applied field 0 G — GT §3a note). On-file
grouping is 2 × 32 detectors (forward 1 / backward 2), t0 bin 34, first-good bin
40, dt ≈ 16 ns, 1960 points to ~31 µs.

## Scenarios registered

| name | render | intended docs use |
|---|---|---|
| `corpus_molafm_alpha` | Inline α calibration (Grouping window) on the 20 G **TF** run 17104 (10 K, paramagnetic): Estimate balances F/B, **α = 1.2516(7)**, before/after curves symmetric about zero. | Data-prep step — the guide's "which data set do you use for α?" (GT §4/§5 Q1). The answer is the TF-above-T_N run. |
| `corpus_molafm_zf_fit` | Converged `Oscillatory×Exponential+Constant` fit on the 1.2 K ZF run 17094, **ν = 1.556 MHz**, zoomed 0–5 µs (~8 cycles), Y-framed. | Money shot: a **low-frequency** molecular-magnet precession (period ~0.65 µs) — visibly different character from EuO's 30 MHz dense band. |
| `corpus_molafm_nu_t` | ν(T) from real per-run ZF fits (1.2 → 6 K) + fitted OrderParameter power law → **T_N ≈ 6.3 K**. | Headline order-parameter deliverable: monotonic ν(T), T_N in the measured 6–7 K bracket. `requires_fit`. |
| `corpus_molafm_zf_overlay` | 6 K (17099, ordered) vs 7 K (17100, paramagnetic) ZF spectra overlaid, block-averaged ×40. | Fixes T_N ≈ 6–7 K directly: the 6 K wiggle vs the smooth 7 K decay — "oscillation gone by 7 K". |

All four capture cleanly under `flock … capture_corpus` in one process; all PNGs
re-read as images. Sizes 57–337 KB (≤ 600 KB). `requires_fit = True` on the two
that run real iminuit fits (`…_zf_fit`, `…_nu_t`). The α and overlay scenarios
are fit-free (algebraic α estimate; standalone matplotlib overlay).

## Run selection & workflow (GROUND_TRUTH § refs)

- **α run = 17104** — 20 G TF at 10 K, paramagnetic and above T_N, so the full
  TF precession amplitude calibrates the F/B balance (GT §4/§5 Q1). Diamagnetic
  (TF) estimator returns α = 1.2516(7). Follows the `basics_alpha` dialog route.
- **ZF order-parameter branch = 17094–17099 (1.2 → 6 K)** — the runs that still
  oscillate (GT §3a). 17100–17103 (7–10 K) are paramagnetic ("gone") and
  excluded from ν(T).
- **Fit model** = `Oscillatory × Exponential + Constant` (GT §4/§5 Q4: damped
  cosine + relaxing tail). The uncalibrated α = 1 MUSR asymmetry sits on a large
  additive baseline (~15 %) absorbed by the Constant — the same route as
  EuO/Ni. Frequencies match GT §3a on the raw α = 1 data, so (like the EuO/Ni
  scenarios) α is not applied before the ZF fits; the Constant handles the
  baseline. Warm-started ν *downward* through ascending T, amplitude/damping
  allowed to grow, ν pinned to ≲2.5 MHz and λ capped at 1.5 µs⁻¹ so the
  near-T_N run cannot escape into a degenerate high-damping "decaying baseline"
  minimum (the EuO/YMnAl/Ni warm-start lesson — a cold seed collapses A_1 → 0).
- **Trend law** = OrderParameter `ν₀·[1−(T/T_N)^α]^β` with **α fixed = 1** (the
  guide leaves the form open, GT §4). Fit ν₀, T_N, β on the 6 points. T_N (the
  ν → 0 crossing) is the graded deliverable; β is *not* quoted anywhere (GT §6,
  §9) so it is not reported as a result.

## Fitted values vs GROUND_TRUTH §3a (measured) table

Warm-start chain, fit window t = 0.11–8 µs:

| Run | T (K) | ν fit (MHz) | ν GT §3a (MHz) | Δ |
|-----|-------|-------------|----------------|-----|
| 17094 | 1.2 | 1.556 | 1.55 | +0.01 |
| 17095 | 2.0 | 1.520 | 1.52 |  0.00 |
| 17096 | 3.0 | 1.426 | 1.42 | +0.01 |
| 17097 | 4.0 | 1.280 | 1.28 |  0.00 |
| 17098 | 5.0 | 1.084 | 1.07 | +0.01 |
| 17099 | 6.0 | 0.75  | 0.66 | +0.09 |

17094–17098 reproduce §3a within ±0.02 MHz. **17099 (6 K) is the hard near-T_N
run**: the coherent oscillation is low-amplitude and heavily damped, and the fit
lands at ~0.75 MHz rather than the GT damped-cosine value of 0.66 (a broad,
degenerate minimum — with A_1 unpinned it collapses to zero instead). Still
clearly non-zero and below the 5 K point, so ν(T) stays monotonic. GT §3a itself
quotes ±≈0.02 and flags this run as "near T_N — last oscillating run".

## T_N / order parameter vs GT

- OrderParameter fit (α = 1): **ν₀ ≈ 1.67 MHz, T_N ≈ 6.3 K, β ≈ 0.26**.
- **T_N ≈ 6.3 K sits inside the measured GT §6 bracket of 6–7 K** — the ν → 0
  crossing falls between the last oscillating run (17099, 6 K) and the first
  paramagnetic run (17100, 7 K). Consistent with the direct 6 K/7 K overlay.
- β ≈ 0.26 is *not graded* (no published/quoted exponent, GT §6/§9); reported
  here only for completeness. With 6 points, three of them clustered high
  (1.2–3 K where ν is nearly flat) and only two on the steep descent, β is
  loosely constrained — the frame-honest reading is "T_N ≈ 6.3 K, few points".
- The **audit correction is respected**: the 7 K run is treated as fully
  paramagnetic. The superseded "0.35 MHz @ 7 K" FFT artefact is NOT reproduced;
  17100 is used only as the paramagnetic reference in the overlay.

## Feature-demonstration opportunities

- **α from a dedicated TF run** cleanly demonstrated (17104) — a distinct
  data-prep story from the Basics silver-TF α run.
- **Low-frequency oscillation** (~1.55 MHz) is the pedagogical contrast to EuO
  (30 MHz): the zoom window is µs-scale, not sub-µs, and the wiggle is legible
  in the raw time domain — good for the "how many frequencies / time vs Fourier"
  guide question (GT §5 Q3). A Fourier-domain scenario was *not* added: the
  0.66–1.55 MHz lines sit very close to the low-frequency baseline skirt on this
  fine, low-statistics MUSR base and the FFT panel does not separate them
  cleanly enough to ship — the time-domain fit is the stronger evidence.
- **T_N by direct spectral comparison** (6 K vs 7 K overlay) is a distinctive,
  assumption-light way to present an ordering temperature — complements the
  power-law fit rather than duplicating it.

## Problems / caveats hit

- **6 K (17099) fit instability** — documented above; the single genuinely hard
  run near T_N. Handled by pinning A_1 ≥ 0.3 and capping λ; the value (0.75) is
  in the ballpark of GT 0.66 but not exact. This is the honest edge of
  resolvability, not a bug.
- **ZF fit χ²ᵣ ≈ 7.8** on 17094 — high, but expected: the fine 16 ns base with
  low per-run statistics gives small per-bin errors that the smooth damped
  cosine cannot chase. The fit *visibly* tracks the oscillation (see PNG); the
  frequency is what matters and it nails §3a.
- **Overlay baselines differ** — 17099 and 17100 have different overall α = 1
  asymmetry levels and relaxation, so the two traces do not share a baseline.
  The visual point (coherent wiggle vs smooth monotonic decay) is unaffected and
  clear; the bottom-right annotation lightly overlaps the 7 K tail but stays
  legible.
- No LF data in the installed MUSR set (the guide's LF sweeps belonged to the
  mis-transcribed EMU run list, GT §2/§5) — so no LF-decoupling scenario, unlike
  the Ni example.

## Top pick

`corpus_molafm_zf_fit` — the converged 1.55 MHz low-frequency precession is the
single most distinctive render: it reproduces the GT §3a base-T order parameter
exactly and shows the molecular-magnet's slow-oscillation character that sets it
apart from every other magnet in the corpus. `corpus_molafm_nu_t` is the natural
headline companion (T_N ≈ 6.3 K).
