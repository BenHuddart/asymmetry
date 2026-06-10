# Count-domain fit modes — verification plan

Oracle strategy (fixed with Ben): **transcribe + synthetic + cross-check.**
WiMDA is a source-only oracle — no live WiMDA run, so "matches WiMDA's printed
value" is *not* a target. Each parity claim is verified by at least one of:

1. **Transcription** — the model under test reproduces the WiMDA Pascal formula
   at sampled points to round-off.
2. **Synthetic ground truth** — `core/simulate` injects a known value; the fit
   recovers it within its fitted uncertainty.
3. **Cross-check** — independent Asymmetry machinery (the α estimators, the
   deadtime calibrator, the existing `fgAll` path) agrees within explained
   tolerance.

Every phase ends `python tools/harness.py validate` green (~2 min). GUI tests
run under `QT_QPA_PLATFORM=offscreen`.

## Phase 1 — fit-mode core

| Claim | Method | Pass condition |
|---|---|---|
| Raw-count model = e^(−t·λ_μ) × existing lifetime-corrected model | transcription | identical arrays to machine precision across a t-grid |
| `fgFB` count model matches WiMDA | transcription | forward/backward = N₀·α^(±1/2)·(1 ± A) + bg·e^(t·λ_μ) to round-off |
| α recovered on synthetic TF data | synthetic | |α_fit − α₀| < 3σ_fit for α₀ ∈ {0.8, 1.0, 1.3}; N₀_F/N₀_B = α₀ |
| α–amplitude correlation reported and non-trivial | synthetic | the F+B `FitResult` exposes a covariance entry for (α, amplitude); |ρ| > 0 reported |
| α from F+B vs grouping estimators on CdS TF | cross-check | α_fit within a stated band of `estimate_alpha` diamagnetic/general/ratio; differences explained (the estimators flatten an asymmetry window, the fit weights the whole count trace — they need not coincide exactly, but should agree to a few %) |
| Single-histogram recovers envelope on EuO PSI | synthetic + real | synthetic: N₀, bg recovered within 3σ; real EuO: converges, residuals structureless, χ²ᵣ ≈ 1 |
| Single-histogram = one-group `build_grouped_count_model` | reuse test | the single-histogram model equals the grouped model evaluated on one group with amplitude ±1 |
| Poisson cost beats Gaussian at low counts | synthetic | on a low-count run, the Poisson fit's recovered parameters are closer to truth (smaller bias) than Gaussian √N; both agree at high counts |
| Existing `fgAll` multi-group fit unchanged | regression | the existing grouped/series fit tests pass byte-for-byte |

## Phase 2 — window & nuisance flexibility

| Claim | Method | Pass condition |
|---|---|---|
| Interior exclude drops the right bins | unit | excluded bins absent from the fit vector; endpoints handled inclusively per WiMDA |
| Masked-artefact fit == clean fit | synthetic | inject a spike in [t_ex0, t_ex1], exclude it; recovered parameters within tolerance of the clean-data fit; without the exclude, the fit is visibly pulled |
| Fittable t₀ recovered | synthetic | a run generated with a known offset recovers t₀ within 3σ; with t₀ fixed 0 the fit equals the pre-Phase-2 result (no-op when disabled) |
| Baseline-drift envelope | synthetic | a run with an injected e^(−(λ_b·t)^β_b) baseline recovers λ_b, β_b; with the term off, results are unchanged (no-op default) |

## Phase 3 — count loss & double pulse

| Claim | Method | Pass condition |
|---|---|---|
| Deadtime loss factor matches WiMDA | transcription | c·(1 − DT0·qq) (Simple), polynomial, power-law forms reproduce the Pascal at sampled rates |
| DT0 recovered on synthetic | synthetic | a run generated with a known deadtime recovers DT0 within 3σ |
| Promote-to-grouping is correct & reversible-in-display | unit | `promote_deadtime_to_grouping` writes the fitted DT0 into the grouping deadtime; additive mode accumulates; a before/after pair is returned; re-reducing the run with the promoted value changes counts in the expected direction |
| Fitted DT0 vs grouping calibrator | cross-check | DT0 from the fit agrees with `calibrate_deadtime_from_histograms` on a high-rate run within a stated band |
| Double-pulse model matches WiMDA | transcription | the two-pulse weighted sum reproduces `ArrayMusrFunc:170–237` at sampled points |
| dpsep round-trip | synthetic | `simulate` double-pulse data at known dpsep, fit recovers dpsep (when fittable) and the single-pulse limit (dpsep→0) matches the single-pulse model |

## Cross-cutting

- **No-op safety**: every new optional feature (exclude, t₀, baseline drift,
  deadtime, double pulse) is off by default and, when off, leaves the Phase-1
  fit numerically identical. Each has a test asserting the off-state no-op.
- **Persistence round-trip**: count-domain fit configuration (target mode,
  exclude window, enabled nuisances, cost choice) saves to and restores from a
  `.asymp` project unchanged.
- **Core/GUI split**: all fit logic is exercised by core-only tests with no Qt
  import; GUI tests cover only the selector wiring and the promote dialog.

## References

- *Muon spectroscopy* (muon-spectroscopy textbook).
- WiMDA `AsymFitFunction.pas`, `Analyse.pas` (transcription oracles only).
