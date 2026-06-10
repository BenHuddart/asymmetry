# Count-domain fit modes — study pass

Date: 2026-06-10. Branch: `feat/count-domain-fit-modes`.
Umbrella: [`wimda-parity-gap`](../wimda-parity-gap/README.md) · Wave A · project 3 · Size L (3 phases).

## Purpose

Asymmetry can fit the reduced asymmetry and (since the multi-group
time-domain port) grouped lifetime-corrected counts with one shared physics
model. It still cannot fit raw count histograms the way WiMDA's calibration
and special-mode workflows do:

- **α as a free fit parameter** — recover the detector balance α from a
  transverse-field calibration run by fitting forward and backward counts
  simultaneously, instead of estimating it on a grid.
- **Single-histogram fits** — fit one detector group's counts to
  N₀·e^(−t/τ)·(1 + A·P(t)) + bg (the musrfit *fittype 0* analogue), for
  calibration and diagnostic work, especially on continuous-source data.
- **Interior exclude range** — drop a window of bins (a laser/RF artefact, a
  spike) from the fit without splitting the run.
- **Fittable t₀ and a baseline-drift term** — let the fit absorb a small
  time-zero error and a slow non-relaxing-baseline drift.
- **Count-loss modelling in the fit** — fit deadtime parameters, then promote
  the fitted values into the grouping correction.
- **Double-pulse fitting** — the two-pulse ISIS source convolution.

The seam already exists: `core/fitting/grouped_time_domain.py`'s
`build_grouped_count_model` (N₀·(1 + A·P) + bg·e^(t/τ) on lifetime-corrected
counts) is WiMDA's `fgAll`. This project extends that seam rather than building
a parallel one.

## Scope answers fixed with Ben (2026-06-10, pre-study)

| Question | Answer |
|---|---|
| Verification oracle | **Transcribe + synthetic + cross-check.** WiMDA stays a source-only oracle: assert against (i) count models transcribed from the Pascal, (ii) synthetic ground truth from `core/simulate` at known α/dpsep, (iii) Asymmetry's own grouping-dialog α estimators. No live WiMDA run; "vs WiMDA value" is not a target. |
| α-free F+B grouping | **Two groups: one Forward, one Backward,** one shared α. Multi-detector banks are summed into F/B upstream at the grouping stage (where Asymmetry already sums detectors). Matches WiMDA's two-block `fgFB`. |
| Muon lifetime | **Fixed** at the physical value (WiMDA + existing grouped path). Optional free τ (musrfit-style) is a recorded follow-on, not implemented now. |

## Method

WiMDA source read directly at `/Users/bhuddart/Source/WiMDA/src` (ignoring
`__history/`, `__recovery/`):

- `FitTyps.pas:30` — the `Fitgrp` enum `(fgFBAsym, fgSelected, fgFB, fgAll,
  fgForward, fgBackward)` and the `Tfitpar` parameter record (α, N0, BG, the
  DT0..C4 deadtime block, the per-group `GrpAmpl/GrpPhas/GrpN0/GrpBG` arrays).
  Parameter-block reference points: `P_base=5`, `GROUP_base=47`, `BG_base=199`,
  `DT_base=202`.
- `AsymFitFunction.pas:139–317` (`ArrayMusrFunc`) — the thread-safe count model
  for every mode, the double-pulse branch (`:170–237`), and the count-loss
  branch (`:280–314`). `musrfunc` in `Analyse.pas:1348` is the scalar twin.
- `Analyse.pas` — `SecondRangeClick:6878` (interior exclude window),
  `DPsepEditChange:6913` (dpsep → pulse weights exp(∓dpsep/2τ)),
  `pname[BG_base..]:7034` (Bsln λ / Bsln β / t₀ offset), the baseline-drift
  envelope `exp(−(λ_b·t)^β_b)` at `:1291`, and `SendToGroupClick:6114`
  (promote fitted deadtime to the grouping correction).

Asymmetry side read at the worktree HEAD (`752880f`): `grouped_time_domain.py`,
`engine.py` (the iminuit `LeastSquares` driver), `parameters.py`, the
`fit_panel.py` `GlobalFitTab` grouped path, `representation/time.py`
(`TimeGroups`), `core/simulate.py`, `transform/deadtime.py`,
`transform/asymmetry.py` (`estimate_alpha`, the `diamagnetic/general/ratio`
methods from PR #34).

## Headline findings

1. **The model already exists; the statistics do not.** WiMDA's `fgAll`,
   `fgForward/Backward/Selected`, and `fgFB` are all the same count equation
   N₀·e^(−t/τ)·(1 + sign·A·P(t)) + bg with different parameter ties. Asymmetry's
   `build_grouped_count_model` is the lifetime-corrected form of exactly this.
   The genuinely missing piece is **count-space statistics** — a Poisson cost
   for low-count bins — which the asymmetry/Gaussian engine cannot express.

2. **Two of the three Phase-1 modes are pure reuse.** Single-histogram is the
   one-group degenerate case of `build_grouped_count_model`; α-free F+B is a
   two-domain fit. Only the α tie needs new model code — and only ~12 lines —
   because the iminuit engine evaluates neither `Parameter.expr` nor
   non-equality ties (see the reuse audit in
   [implementation-options.md](implementation-options.md)).

3. **Raw counts unify everything.** N_raw(t) = e^(−t/τ)·[the existing
   lifetime-corrected model]. Fitting raw counts with a selectable Poisson/√N
   cost reuses the existing model builder unchanged and is the statistically
   correct treatment at low counts. The existing `fgAll` path stays on the
   Gaussian lifetime-corrected route; the new modes default to Poisson on raw
   counts. The divergence is documented in
   [comparison.md](comparison.md).

4. **α from the fit is just N0_F/N0_B** mathematically, but reporting it with
   an uncertainty *and its correlation with the amplitude* (strong in TF runs)
   needs α to be an explicit fitted parameter, not two independent normals —
   hence the √α-coupled F+B model.

5. **Deadtime-in-fit is the metadata-heavy corner.** The WiMDA loss factor
   needs frame counts and resolution from run metadata; the promote-to-grouping
   action maps onto Asymmetry's existing per-histogram deadtime field.

## Documents

- [comparison.md](comparison.md) — WiMDA fit-mode-by-mode count models vs the
  Asymmetry seam; every divergence stated with both behaviours.
- [implementation-options.md](implementation-options.md) — the reuse audit, the
  chosen architecture, and the full three-phase implementation plan
  (file-by-file touch list, test plan, recorded follow-ons).
- [test-data.md](test-data.md) — verification corpus mapped to each mode.
- [verification-plan.md](verification-plan.md) — how each parity claim is
  verified under the transcribe-+-synthetic-+-cross-check oracle.

## Out of scope (recorded with rationale)

- **Negative-muon mode** — separate deferred portfolio brief
  (`negative-muon-analysis`); the WiMDA `Neg_*` parameter block and `Polfunc`
  are out.
- **RRF coupling** — owned by the `rrf` project.
- **KEK spill deadtime** — no data source to verify against.
- **WiMDA's fitted-baseline "Set BG" workflow** — deferred. Our baseline-drift
  term (Phase 2) is the only overlap; it is an *optional* nuisance and does not
  pre-empt a future Set-BG port. Noted in
  [comparison.md](comparison.md).
- **Free muon lifetime** — recorded follow-on (see scope answers above).

## Primary scientific source

*Muon spectroscopy* (the muon-spectroscopy textbook) for the count equation
N(t) = N₀·e^(−t/τ)·(1 + A·P(t)) + B, the detector-balance parameter α, the
Poisson nature of detector counts, and the double-pulse structure of a pulsed
source. Cited by name in each document's reference list.
