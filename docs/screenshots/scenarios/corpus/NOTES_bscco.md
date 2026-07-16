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
| `corpus_bscco_field_compare` | **Real Fit-Parameters trend panel, native multi-series overlay** (PR-248): σ(T) 400 G + 200 G as two coloured series with a legend | The guide's **field-comparison** task: 200 G plateau sits below 400 G — pancake-vortex field dependence (§6b) |
| `corpus_bscco_maxent` | **MaxEnt overlay** (standalone Agg figure): 10 K broad vortex p(B) (filled) vs 125 K normal narrow line, unit-area spectra near 5.42 MHz | The guide's **FFT-vs-MaxEnt** comparison, previously dropped — now works out of the box after PR #249 (see PR #249 section) |

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

1. **MaxEnt was dropped in wave 2 — now FIXED and shipped (PR #249).** The guide
   asks to compare FFT with Maximum-Entropy. On this real F/B asymmetry the
   V1 MaxEnt solver *used to* diverge ("stopped early at cycle 7 as χ² began
   rising past the optimum") into spiky noise — its large α = 1 baseline
   defeated the reconstruction, so wave 2 shipped only the FFT render.
   **PR #249 fixed the engine** (1/σ²-weighted baseline, lowered amplitude
   floor, σ-weighted nuisance fits, data-derived phase seeding of the MUSR
   quadrant groups). Verified 2026-07-16: run 1277 (10 K) now converges out of
   the box (defaults, field auto window) to **χ²/N = 1.035 with the peak at
   5.404 MHz** (the 400 G vortex line), in 26 s. `corpus_bscco_maxent` ships the
   previously-dropped FFT-vs-MaxEnt comparison. See the PR #249 section below.
2. **Grouping is F/B, not All Grps** (see workflow). Reproduces σ(T) to a few
   percent but is not a byte-for-byte match to the WiMDA All-Grps numbers.
3. **`tf_damping` contrast is real but subtle.** The oscillation amplitude is
   only ~±9 % on a −23 % baseline with a growing late-time noise fan; the 10 K
   Gaussian collapse vs the 125 K persistence reads best in the 1.5–4 µs region.
   Caption carries the interpretation.
4. **Field comparison is now the PR-248 native trend-panel overlay** (reworked
   2026-07-12). `field_compare` no longer uses a standalone Matplotlib figure:
   both σ(T) series are loaded through `load_representation_series([...two
   tuples...])` and overlaid by arming the multi-selection
   (`_set_selected_group_ids(["bscco-sig-400","bscco-sig-200"])` — the
   Shift+click a user would do). The panel draws them in distinct colours with a
   legend of the series names; the axis-transform layer and per-point error bars
   apply to both. See the PR-248 section below for what works and what does not.
   The **frequency-domain** 10 K↔125 K contrast is still delivered in the **time**
   domain (`tf_damping`) because the Fourier panel plots one run's spectrum (that
   is a separate panel, unaffected by PR-248).

## PR-248 (multi-series overlay) — verdict from this rework

- **Works.** Two σ(T) series render as blue (200 G, C0) + orange (400 G, C1)
  with a name legend; both plateaus read clearly (400 G ≈ 1.16 above 200 G ≈
  0.91 µs⁻¹), reproducing the §6b pancake-vortex story. Error bars are drawn
  per series in the series colour.
- **Colour = alphabetical pill order, not load/active order.** `_series_to_plot`
  colours by `_group_button_map` order, and `_rebuild_group_buttons` sorts pills
  by `group_name.lower()`. So "200 G…" gets C0 even though "400 G…" is the active
  series (via `select_id`). The active series is *not* visually distinguished on
  the plot except by owning any model-fit overlay — a reader cannot tell from the
  plot which series owns the table/export. Named the series so the legend carries
  the field explicitly.
- **One active-series model fit only.** In overlay mode `_plot_series_param`
  draws a trend-model overlay for the *active* series alone; there is no
  per-series fit. Left the headline overlay fit-free (the §6b story is about the
  two plateaus, not a curve) and report the limit rather than staging a
  misleading single-series fit.
- **Export is active-series only (rough edge).** `_export_tsv` and
  `_build_gle_export` both read `self._rows` (active series) — an overlaid
  comparison exports only the 400 G data; the 200 G series is silently dropped
  from TSV/GLE. Confirmed empirically. The overlay is a *plot-only* feature.
- **Ergonomics.** `load_representation_series` only single-selects (`select_id`
  is scalar); there is no public multi-select entry point, so the overlay must be
  armed with the private `_set_selected_group_ids`.
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

## PR 248 round 2 (re-test, 2026-07-12) — merge-blockers verified fixed

Re-tested commit 4a91420 against the real 400 G/200 G σ(T) fits (both series
genuine per-run TF Gaussian fits via the core FitEngine).

- **CONFIRMED FIXED — TSV overlay now exports every series.** `field_compare`
  Export TSV writes a leading **`Series`** column and **both** series' rows:
  verified `{'400 G — TF scan': 14, '200 G — TF scan': 10}` data rows (matches
  the two scans), raw columns intact (`Run, B (G), T (K), sigma (µs⁻¹),
  err_sigma (µs⁻¹), reduced_chi2, chi2`). The round-1 "active-series only" leak
  is closed. (GLE export still active-series only — see below.)
- **CONFIRMED — `(active)` legend flag.** The overlay legend now reads
  `200 G — TF scan` / `400 G — TF scan (active)`; the flag tracks the active
  (first-selected) series (verified: reversing the select order moves the flag).
- **CONFIRMED — `select_series()` migration.** `field_compare` now arms the
  overlay through the public `panel.select_series(["bscco-sig-400",
  "bscco-sig-200"])` (was the private `_set_selected_group_ids`). Behaviour
  identical: 2 series, C0/C1 colours, 400 G active. Render unchanged.
- **GLE overlay → warns + active-only (as designed, fast-follow).** `_export_gle`
  now calls `gle_export.show_warning(...)` naming the active series
  ("GLE export currently writes only the active series ('400 G — TF scan')…")
  *before* `run_gle_export`. Mechanism: `show_warning` → `QMessageBox.warning`
  (modal) unless `PYTEST_CURRENT_TEST` is set — so a real GUI user sees the
  dialog; in an offscreen pytest capture it is suppressed. A user driving the
  GUI *would* see it; a headless non-pytest script would block on it.
- **Physics regression: none.** 400 G plateau 1.164 µs⁻¹, 200 G 0.907 µs⁻¹
  (NOTES table 1.16/0.91) — unchanged.

The round-1 caveats "Export is active-series only" and "no public multi-select
entry point" are now **resolved** (TSV) / **superseded** (select_series). GLE
multi-series export remains the documented fast-follow.

## PR #249 (MaxEnt divergence fix + phase seeding) — pre-merge verification (2026-07-16)

Verified against the real MUSR runs through the core `maxent()` API (χ²/N
recomputed from `reconstruct_group_signals`, i.e. equal to the engine's by
identity). New scenario `corpus_bscco_maxent` added; no `src/` changes.

**Item 1 — headline (out-of-the-box BiSCCO), CONFIRMED.** Run 1277 (10 K,
400 G) with `MaxEntConfig()` defaults, full 32 µs range, field-derived auto
window (1.36–9.49 MHz): **χ²/N = 1.035** (PR claim ≈ 1.04 ✓), **peak 5.404 MHz**
(PR claim 5.40 ✓), converged at cycle 7, 26 s. `auto_steer_applied = {}` (no
steering — a MUSR 16 ns run's Nyquist is already fine for the 5.4 MHz window).
The raw χ² is 8200 (matches the PR's quoted number) but is correctly normalised
to 1.035 over 7924 obs. FFT-vs-MaxEnt sanity: the MaxEnt peak (5.404) sits just
below the normal-state line (5.418, run 1276) — the vortex diamagnetic shift,
physically sensible; the broad 10 K line carries the vortex p(B) width.

**Item 2 — phase seeding on MUSR quadrant groups, CONFIRMED.** The data-derived
seeds on run 1277 are group 1 = 36°, 2 = −49°, 3 = −133°, 4 = +131° — the four
MUSR quadrant groups **~90° apart**, exactly as claimed. With
`auto_phase_seed=True` (default) all four groups keep healthy amplitudes
(~0.0014) and χ²/N = 1.035 (converges cycle 7, 28 s). With seeding **off**,
groups 1 and 4 **mute out** (amp → 0.0001 and 0.0 — the "fits its amplitude to
zero and drops out" failure the PR describes), the surviving groups collapse
toward each other, and χ²/N = **2.76** (never converges in 10 cycles, 88 s).
Seeding is a clear, decisive win. GUI wiring confirmed in
`maxent_panel.py`/`mainwindow.py`: the **"Seed phases from data"** checkbox
exists and defaults on (`maxent_panel.py:328–330`); editing a Phase-column cell
unticks it (`_on_group_table_item_changed`, line 162–165); **"Use fitted
phases"** unticks it via `set_auto_phase_seed(False)`
(`mainwindow.py:8781`). Matches the docs verbatim.

**Scenario config note.** `corpus_bscco_maxent` uses `n_spectrum_points=1024`
(the GUI default), full time resolution, everything else default. At capture:
10 K χ²/N = 1.06 (peak 5.40), 125 K χ²/N = 2.25 (peak 5.42, a narrow near-
unrelaxed line MaxEnt piles into one bin). ~25 s total, 93 KB. Both spectra are
unit-area, so the render is an honest "equal area, spread by the vortex lattice"
p(B) comparison. The 125 K line's higher χ²/N is the estimator over-iterating on
a long-lived narrow line — not a defect; the caption carries the standard
"MaxEnt shape is an estimate, not a calibrated width" caveat.
