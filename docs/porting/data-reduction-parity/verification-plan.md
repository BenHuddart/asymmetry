# Data-reduction parity — verification plan

Two layers per phase: (1) **synthetic-truth tests** that always run in CI —
Poisson data generated from known parameters, estimator must recover them;
(2) **WiMDA-transcribed oracles + corpus tests** (skipped when the corpus is
absent) that pin parity and real-data behaviour. The harness ladder applies:
`structural` → `lint` → targeted `test` → full `validate` (green required at
the end of every phase).

## Phase 1 — alpha estimation

1. **Objective transcription**: implement WiMDA's diamagnetic objective
   Σ(Aᵢ/σᵢ)² and General relative-scatter objective in test code, walk the
   grid exactly as `Group.pas:1775` does (0.1/0.01/0.001 steps from α = 1)
   on fixed synthetic counts; assert the production continuous optimiser
   agrees within ±0.001 (the final grid step) on identical input.
2. **Synthetic truth**: known α with oscillating (TF), Gaussian-relaxing,
   and flat P(t); both estimators recover α within reported σ_α; coverage
   test of σ_α itself (repeat draws; empirical scatter ≈ reported σ within
   ~30% for ≥ 200 repeats).
3. **Bias contrast**: on relaxing synthetic data, ΣF/ΣB must show its
   predicted bias while General does not — documents the method choice.
4. **Corpus**: General α scatter across the HIFI 118222–118240 LF series
   smaller than across-method disagreement; diamagnetic α on a TF nickel run
   stable under bunching-factor changes (D3 reproducibility claim).
5. **GUI**: grouping-dialog method selector round-trips; estimate applies to
   all selected datasets (deadtime-Estimate convention); uncertainty shown.

## Phase 2 — backgrounds

1. **Tail-fit transcription**: `BGfit`/`estBG` transcribed (bin-integrated
   model, √N weights, σ = 10¹⁰ for ≤ 4 counts, late-half window, fixed
   starts); on moderate-count synthetic data the Poisson-MLE production fit
   agrees within mutual uncertainties.
2. **Low-count superiority**: at late-time counts ≪ 10/bin, MLE recovers the
   true flat rate where the transcribed weighted fit is biased (regression
   for D4).
3. **Bin-integration check**: fitted p₁ invariant (≪ σ) under output bin
   width changes — validates the (e^{a/2}−e^{−a/2})/a factor.
4. **Gating**: pulsed ISIS dataset now offers tail-fit but not range-average
   (no pre-t0 bins); PSI data offers range-average (and tail-fit); fixed
   manual values available everywhere.
5. **Run subtraction**: self-subtraction → zero counts, √(2N)-scaled errors;
   frame-ratio scale verified against hand arithmetic; deadtime-corrected
   consistently (D6); provenance (reference run number) persisted and
   reloaded from the project.
6. **Corpus**: tail-fit p₂ on an ISIS run is small and reported with
   uncertainty; "consistent with zero" flag fires where expected; a PSI run
   gives p₂ compatible with its pre-t0 range-average estimate.

## Phase 3 — binning, t0, exclusion, periods

1. **Edges**: variable edges follow bin0·(bin10/bin0)^(t/10 μs) exactly;
   WiMDA-formula edges agree within 0.2% (D8); constant-error edges follow
   bin0·e^{λ_μ t}; degenerate inputs (bin10 = bin0, windows shorter than
   bin0) fall back sanely.
2. **Statistics**: constant-error mode → per-bin σ flat within ~2× on real
   long-window data; weighted aggregation preserves the weighted mean and
   total information (Σ1/σ² conserved within numerical noise).
3. **Provenance**: raw histograms identical before/after any binning-mode
   change; only the reduced representation differs; Fourier/MaxEnt paths
   refuse non-fixed modes with a clear message.
4. **t0 search**: synthetic prompt peak (continuous) found exactly; synthetic
   ISIS-like pulse → edge-midpoint within ±1 bin of truth; corpus good files
   recover loader t0 (±1 continuous, ±2 pulsed); search never auto-applies —
   it fills the override controls for confirmation.
5. **Exclusion**: equivalence with manual group-membership removal on sums,
   α, asymmetry; range-text parser fuzz (orders, overlaps, reversed ranges);
   persistence round-trip through project save/load; schematic click parity
   with list editing.
6. **Period mapping**: {1→red, 2→green} reproduces the current 2-period path
   bit-for-bit; arbitrary subsets equal manual histogram addition;
   good-frame bookkeeping equals summed per-period frames; photo-μSR silicon
   runs reproduce the period-selection study's validated asymmetries when
   mapped trivially; dwell periods forced to Ignore.

## Regression gate

After **each phase** (not just the last): full `validate` plus the pinned
corpus results — CdS 5.12 K χ²ᵣ = 1.35, EuO β, repolarisation curve
(test-data.md) — unchanged. These exercise grouping, background, rebin and
periods kernels end-to-end; any drift is a stop-the-line failure since these
phases refactor shared reduction code.

## Documentation verification

`python tools/harness.py docs` clean per phase; every new function/mode has
a user-guide page or section with a "when to use this" register
(docs/user_guide/data_reduction/, fit_functions style); references in APS
style in lists.
