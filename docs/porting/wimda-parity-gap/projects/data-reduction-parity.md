# Project brief: data-reduction-parity

Umbrella: `wimda-parity-gap` · Wave A · Size L (3 phases, one long session each)

## Motivation

The largest cluster of genuine WiMDA gaps sits in the reduction layer between
raw histograms and the asymmetry curve. These affect *every* downstream
analysis: a weak alpha estimate or missing background mode degrades every fit.

## Scope

**Phase 1 — alpha estimation.** Port WiMDA's two estimators
(`Group.pas:1775 EstimateButtonClick`): the **diamagnetic** method (grid
search minimising Σ(Aᵢ/σᵢ)² over a TF run) and the **General** method
(lifetime-corrected count balance `F/√α + B·√α` scaled by `e^{tλ_μ}`,
minimising relative scatter — works on relaxing LF/ZF data, which the current
Mantid-style ΣF/ΣB ratio (`core/transform/asymmetry.py: estimate_alpha`)
cannot handle). Modern implementation: continuous optimisation
(`scipy.optimize.minimize_scalar`) instead of WiMDA's grid; report α with an
uncertainty. Keep ΣF/ΣB as a third, labelled method.

**Phase 2 — backgrounds.** (a) **Tail-fit background** for pulsed sources
(`Group.pas estBG/BGfit`: exp-decay + flat fitted to the late-time spectrum;
pulsed data has no pre-t0 region so range-average can't work); (b)
**background-run subtraction** (`Group.pas Regroup FileBG`, `BGform.pas`):
subtract a designated background run's histograms scaled by the frame ratio —
sample-holder/silver background and laser-off references.

**Phase 3 — binning, t0, detector exclusion, periods.**
(a) **Variable binning** (width grows from `bin0` per decade) and
**constant-error binning** (width ∝ e^{λ_μ t} so output bins have ~equal
statistics) alongside the existing fixed factor (`Group.pas:1411–1418`;
`core/transform/rebin.py`); (b) **automatic t0 search** (prompt-peak scan,
`Group.pas:2225`) surfaced in the grouping dialog for files with missing or
wrong t0; (c) **per-detector exclude list** (`Group2.pas ExcludeDetectors`)
— dead/hot detector handling at the grouping level, persisted in the project;
(d) **multi-period subset → red/green mapping** (`PeriodMappingUnit.pas`):
sum arbitrary subsets of up to N periods into the red and green sets (the
unimplemented remainder of the `period-arithmetic` candidate; extends
`core/io/periods.py`).

**Out**: extended deadtime models and count-loss fitting (→
`count-domain-fit-modes`); `default.mgp` per-directory auto-defaults (record
as a possible later nicety); fitted-baseline Set BG (deferred, revisit with
count-domain modes); ARGUS/KEK hardware fixers (dropped).

## Current Asymmetry state

`estimate_alpha` (ΣF/ΣB); `background.py` fixed/range modes (gated to
continuous-source data); `rebin.py` fixed factor; t0/good-bin overrides in
the grouping dialog; period select + G±R in `periods.py`.

## GUI/UX sketch

Everything lands in the existing grouping dialog
(`gui/windows/grouping_dialog.py`): alpha "Estimate" button grows a method
choice with a one-line explanation per method; background section grows
"Tail fit" and "Background run…" modes; binning mode dropdown
(Fixed/Variable/Constant-error) near the bunch factor; "Find t0" button;
detector-exclusion via the existing detector schematic widget
(click-to-exclude) plus a list editor. Period mapping: small matrix dialog
(period × {Red, Green, Ignore}) reachable from the period selector.

## Physics-correctness notes

Alpha methods get uncertainties (WiMDA reports a bare number). Tail-fit
background must use Poisson-appropriate weighting at low counts. Variable
binning is display/fit-input transformation — keep raw histograms intact
(provenance invariant).

## Conflicts & dependencies

Primary surfaces: `core/transform/{asymmetry,background,rebin,grouping}.py`,
`core/io/periods.py`, `grouping_dialog.py` — no Wave A overlap. Shared-file
discipline: `core/project/schema.py` (exclusion list + binning mode
persistence), `transform/__init__.py`.

## Verification sketch

Diamagnetic α vs WiMDA arithmetic on a TF corpus run; General α
self-consistency on the EMU LF series; tail-fit background on pulsed ISIS
runs vs WiMDA values; binning: constant-error mode yields ~flat σ per bin;
period mapping vs photo-μSR silicon runs 103277–103298. Corpus regressions
(CdS, EuO) must not move.
