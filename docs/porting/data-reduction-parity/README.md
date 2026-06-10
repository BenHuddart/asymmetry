# Data-reduction parity — porting study

Status: **study** (implementation pending go-ahead).
Date: 2026-06-10. Branch: `feat/data-reduction-parity`.
Umbrella: [`wimda-parity-gap`](../wimda-parity-gap/README.md), Wave A, project 1
(brief: [`projects/data-reduction-parity.md`](../wimda-parity-gap/projects/data-reduction-parity.md)).

## Feature

The reduction layer between raw histograms and the asymmetry curve is where the
largest cluster of genuine WiMDA gaps sits, and every downstream analysis
inherits its quality: a weak alpha estimate or a missing background mode
degrades every fit. This study covers six capabilities, phased into three
implementation passes:

| Phase | Capability | WiMDA source |
|---|---|---|
| 1 | Diamagnetic alpha estimator (minimise Σ(Aᵢ/σᵢ)² over a TF run) | `Group.pas:1775 EstimateButtonClick` |
| 1 | "General" alpha estimator (lifetime-corrected count balance; works on relaxing LF/ZF data) | `Group.pas:1775` (method=general branch) |
| 2 | Tail-fit background for pulsed sources (exponential + flat fitted to the late-time spectrum) | `Group.pas:116 BGfit`, `estBG` |
| 2 | Background-run subtraction (frame-ratio-scaled reference run) | `Group.pas Regroup` FileBG path, `BGform.pas` |
| 3 | Variable and constant-error binning modes | `Group.pas:1411–1418` |
| 3 | Automatic t0 search (prompt-peak / pulse-edge scan) | `Group.pas:2225 SearchT0ButtonClick` |
| 3 | Per-detector exclude list | `Group2.pas ExcludeDetectors`, `nexusunit.pas:651` |
| 3 | Multi-period subset → red/green mapping | `PeriodMappingUnit.pas`, `Group.pas MapPeriods` |

Phase 3's period mapping realises the remainder of the `period-arithmetic`
candidate (the red/green *selection* slice shipped with the
[`period-selection`](../period-selection/README.md) study; this adds arbitrary
subset summation).

**Out of scope** (recorded with rationale in
[implementation-options.md](implementation-options.md)): extended deadtime
models and count-loss fitting (→ `count-domain-fit-modes`); fitted-baseline
Set BG (deferred); per-directory `default.mgp`/`default.exclude` auto-load
(disabled in WiMDA's own current source; later nicety at most); ARGUS/KEK
hardware fixers (dropped).

## Reference programs and sources

- **WiMDA** — primary parity reference; all cited routines read directly from
  the Pascal at `/Users/bhuddart/Source/WiMDA/src` (transcriptions in
  [comparison.md](comparison.md)).
- **Mantid / musrfit** — cross-checks on workflow conventions (verification
  oracle only for any GPL code; never vendored).
- **Muon spectroscopy textbook** (Blundell, De Renzi, Lancaster & Pratt,
  *Muon Spectroscopy: An Introduction*) — primary scientific source for the
  asymmetry/alpha formalism, background physics, t0 conventions, and binning
  statistics.

The target is **parity of functionality, not implementation**: modern numerics
(continuous optimisation, Poisson-appropriate likelihoods, propagated
uncertainties) replace WiMDA's grid searches and ad-hoc weights wherever the
two differ; every divergence is recorded in
[comparison.md](comparison.md#divergences-from-wimda) with both behaviours.

## Study files

- [comparison.md](comparison.md) — WiMDA algorithm transcriptions, current
  Asymmetry state, Mantid/musrfit practice, textbook physics, and the
  divergence record.
- [implementation-options.md](implementation-options.md) — design options,
  decisions (settled with Ben 2026-06-10), and the full phased implementation
  plan.
- [test-data.md](test-data.md) — corpus runs per phase, including the
  corrected identification of the LF self-consistency series (HIFI, not EMU).
- [verification-plan.md](verification-plan.md) — transcribed-oracle strategy,
  per-phase verification targets, and the corpus regression gate.

## Outcome

To be completed during the implementation pass (one milestone commit per
phase, each leaving `python tools/harness.py validate` green and the branch
main-mergeable).
