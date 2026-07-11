# NOTES — Photo-µSR in silicon (Semiconductors)

Module: `silicon_photo.py`. Example: `Semiconductors/Photo-muSR in silicon`.
Corpus: HiFi (ISIS) HDF4 NeXus `HIFI00103277`–`103299` (23 two-period runs,
RB1520457). The corpus's unique **multi-period (light-ON / light-OFF)**
showcase. Spec: the example's `GROUND_TRUTH.md` (paper-graded against
Yokoyama *et al.*, PRL 119, 226601 (2017) = arXiv:1702.06846, Fig. 1, 291 K,
LF 10 mT).

## Scenarios registered

| name | render | requires_fit | intended docs use |
|------|--------|:---:|-------------------|
| `corpus_si_period_mapping` | Period → Red/Green mapping dialog on real run 103277: period 1 → Red (laser ON), period 2 → Green (laser OFF), each 14 008 good frames | no | THE period-mode data-handling step — the two-period `.nxs` resolved into the two light spectra (GT §1/§2) |
| `corpus_si_on_off_overlay` | Light-ON (Red) vs light-OFF (Green) asymmetry of run 103277 overlaid, 0–6 µs | no | the light-induced extra relaxation, visible by eye; motivates λ as a carrier-density yardstick (GT §1) |
| `corpus_si_lambda_fit` | Single-exponential fit on the light-ON period of the highest-Δn run 103277, first 1 µs, A₀ fixed at the light-OFF value → λ = 1.270 µs⁻¹ | yes | the core analysis step: the guide's light-ON λ extraction (GT §4 steps 3–4) |
| `corpus_si_lambda_vs_dn` | Calibration Trend 1: λ vs Δn (log–log) across the 10 injection runs 103277–103286 with the fitted power law λ = a·Δnⁿ | yes | the λ↔Δn calibration; exponent n = α (GT §4 step 5, §10a Fig 1d) |
| `corpus_si_tau_decay` | **Headline** Trend 2: Δn vs ΔT carrier decay across the 12 delay-scan runs 103287–103298 with the single-exponential fit → τ₀ | yes | the recombination lifetime, the example's primary deliverable (GT §4 step 8, §10a Fig 1e) |

## Run selection & workflow (GROUND_TRUTH §3/§4)

- **Period convention (GT §1):** period 1 = **Red = light-ON**, period 2 =
  **Green = light-OFF**. Confirmed empirically — fitting each period of run
  103277 gives λ(Red) ≈ 1.3 µs⁻¹ (strong depolarisation) vs λ(Green) ≈ 0.09
  µs⁻¹ (near-flat), so Red is unambiguously the light-ON spectrum.
- **λ workflow (GT §4):** fit the Green period with a free single exponential
  → amplitude A₀ ≈ 15.5 %; refit the Red period over the **first 1 µs** with
  **A₀ fixed** → λ. Applied per run by `_light_on_lambda`.
- **Calibration set** (ΔT = 0, runs 103277–103286): Δn is the injected density
  from the guide run table (exact). λ vs Δn → power law.
- **Delay scan** (runs 103287–103298, ΔT = 0.1–70 µs): fit λ the same way,
  invert the calibration to get Δn, fit Δn vs ΔT with a single exponential.
- **α (detector balance):** GT §4 step 1 obtains it from TF run 103299 (20 G).
  Not exercised here — the loader's default α = 1 reduction is adequate for the
  relaxation-rate ratios, and 103299 is a separate calibration surface already
  covered by the LLZ/EuO α-calibration scenarios. Noted as an opportunity below.

## Fitted values vs ground-truth targets

| Quantity | This module | Target (GT §10) | Notes |
|----------|-------------|-----------------|-------|
| λ(light-OFF), 103277 | 0.085 µs⁻¹ | 0.068(2) µs⁻¹ (paper Fig 1c) | free-A single exp; same order, slightly high (no baseline term) |
| λ(light-ON), 103277 (Δn=8.9e13) | **1.270 µs⁻¹** | ~1.29 (digitised Fig 1d) | A₀ fixed, first 1 µs — matches |
| λ(light-ON), 103282 (Δn=4.7e13) | 0.904 µs⁻¹ | 0.94(2) µs⁻¹ (exact, text) | matches within read-off error |
| Calibration exponent α | **0.652** | **0.68(4)** (paper Fig 1d) | within ~1σ |
| Calibration prefactor β | 1.304 µs⁻¹ | 1.46(4) µs⁻¹ | low; the highest-Δn points sit below the paper's curve (GT §10a notes 103277 does too) |
| **Carrier lifetime τ₀** | **10.75 µs** | **11.1(9) µs** (paper Fig 1e) | within 1σ — the headline reproduces |
| Δn(0) intercept | 9.77×10¹³ cm⁻³ | 9.4(4)×10¹³ cm⁻³ | consistent |

Full calibration λ vs digitised Fig 1d targets (µs⁻¹):

| Run | Δn (cm⁻³) | λ (module) | λ (digitised) |
|-----|-----------|-----------|---------------|
| 103277 | 8.9×10¹³ | 1.270 | ~1.29 |
| 103278 | 7.9×10¹³ | 1.154 | ~1.28 |
| 103279 | 7.1×10¹³ | 1.146 | ~1.19 |
| 103280 | 6.3×10¹³ | 1.061 | ~1.15 |
| 103281 | 5.6×10¹³ | 1.038 | ~1.06 |
| 103282 | 4.7×10¹³ | 0.904 | 0.94 (exact) |
| 103283 | 3.7×10¹³ | 0.713 | ~0.78 |
| 103284 | 2.7×10¹³ | 0.577 | ~0.65 |
| 103285 | 1.8×10¹³ | 0.427 | ~0.51 |
| 103286 | 9.3×10¹² | 0.286 | ~0.35 |

The module's λ sit systematically ~5–15 % below the digitised points at low Δn
(a pure single exponential with no baseline underestimates the small-λ tail),
which is why the fitted α (0.65) and β (1.30) come out a touch low. The
headline τ₀ is insensitive to this — it depends on the *shape* of Δn(ΔT), not
the calibration normalisation — and lands at 10.75 µs vs the paper's 11.1(9).

## Period-mode UX notes (honest)

- **The loader path works cleanly.** A two-period `.nxs` loads as one combined
  `MuonDataset` carrying `grouping['period_reduced']` (two curves);
  `select_period(combined, "red"/"green")` returns per-period datasets. The
  `period=` kwarg on `asymmetry.core.io.load` is the scriptable equivalent.
  `period_count` / `period_labels` correctly report 2 / `["red","green"]`.
- **`PeriodMappingDialog` on real data** shows the WiMDA period→Red/Green/Ignore
  matrix and defaults (period 1 → Red, period 2 → Green) exactly right. This is
  the strongest period-mode render in the corpus.
- **Loader gap — per-period good frames.** `select_period` leaves
  `grouping['good_frames'] = 1.0`, so the mapping dialog would otherwise show a
  meaningless "1". The real per-period good frames (~14 008) live in the NeXus
  `Beamlog_Good_Frames_Total` log; `_inject_real_good_frames` reads that and
  splits it evenly for the screenshot. **Worth fixing in the loader** —
  populate per-period `good_frames` from the Beamlog so any period-mode UI shows
  the true exposure without a scenario workaround. (Does not affect the physics:
  the default reduction is α = 1 and good-frame counts are display-only here.)
- **Two datasets, one run number.** Both periods of a run share
  `run.run_number` (103277); the run-number-keyed data browser would collapse
  them. The on/off-overlay and λ-fit scenarios assign derived run numbers
  (`next_derived_run_number`) + friendly labels ("103277 laser ON/OFF") so both
  survive. A period-aware browser key would remove this friction.

## Feature-demonstration opportunities (spotted, not all captured)

- **RF / Green−Red difference:** `asymmetry.core.io.periods.build_rf_difference_scan`
  forms the (Green − Red) integral-asymmetry field scan for RF-µSR — the same
  two-period machinery. Not applicable to this LF example (no swept field) but
  the period infrastructure is shared; a good cross-reference from the docs page.
- **α from TF 103299 (20 G):** the guide's step-1 detector-balance calibration
  on the one TF run — could add a `corpus_si_alpha` mirroring the LLZ
  calibration scenario. Skipped to keep this module focused on the period-mode
  story; the α-calibration UI is already documented by LLZ/EuO.
- **Waterfall of the delay scan** (light-ON spectra 103287→103298) would show
  the relaxation visibly weakening as carriers recombine — a distinctive
  companion to the Δn(ΔT) trend, if a second data-domain render is wanted.

## Top pick for docs

`corpus_si_period_mapping` for the **period-mode data-handling** page (it is the
one render no other corpus example can produce), paired with `corpus_si_tau_decay`
as the **headline result** (τ₀ ≈ 11 µs, paper-graded). `corpus_si_on_off_overlay`
is the best single "what is photo-µSR" teaser.

## Problems hit

- **Fit-range spinboxes don't commit on `setValue`.** The single-tab fit range
  is owned by the plot panel; `_fit_range_max_spin.setValue()` updates the
  display but never slices the fit dataset (the range only commits on
  `editingFinished`). The λ-fit fit initially ran over the full window and gave
  λ ≈ 0.87 instead of 1.27. Fixed by calling
  `window._plot_panel.set_fit_range(0.0, 1.0)` (the canonical owner), which
  propagates the sliced dataset to the fit tab. Worth a scenario-author note.
- **Custom trend x-axes** (Δn, ΔT) are not run-level coordinates. Solved by
  carrying them as extra `values` entries (`Dn`, `dT`) so they appear as
  `param:` x-axis options, selecting the axis via `_x_combo.findData`, and
  registering derived label metadata (Δn, ΔT with units) so every label path
  renders them properly. Clean once understood; documented in the module.
- **χ²/ν ≈ 2.7 ("poor")** on the light-ON single-exp fit — expected: the real
  spectrum has more structure than one exponential over 0–1 µs, but the fitted
  λ still matches the paper. Honest, not a bug.
