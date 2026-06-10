# Project brief: count-domain-fit-modes

Umbrella: `wimda-parity-gap` · Wave A · Size L (3 phases)

## Motivation

WiMDA can fit raw count histograms directly, not only the reduced asymmetry:
α as a free fit parameter, single-histogram N0/background fits, deadtime
parameters inside the fit, double-pulse convolution. These are calibration
and special-mode workflows the asymmetry-only pipeline cannot express.
Asymmetry already has the foundation — the grouped count model
`N0·(1+A·P(t)) + bg·e^{t/τ}` in `core/fitting/grouped_time_domain.py` — so
this project extends an existing seam rather than building one.

## WiMDA reference

`Analyse.pas` fit-target modes (`FitTyps.pas:30` enum): `fgFB` (simultaneous
F+B count fit, α = p[1] free), `fgForward`/`fgBackward`/`fgSelected`
(single-histogram, N0 + exponential background), plus: interior exclude time
range (`SecondRange`, `Analyse.pas:6878–6911`), fittable t0 offset and
stretched-exp baseline drift (`pname[BG_base..]`), count-loss modelling in
the fit with Simple/Polynomial/Power-law deadtime models and push-back to
grouping (`CountLossModelling`, `ArrayMusrFunc:281–314`,
`SendToGroupClick:6114`), double-pulse fitting (`dpsep`,
`ArrayMusrFunc:170–237`: two time-shifted evaluations weighted by
`exp(±dpsep/2τ)`).

## Scope & phasing

**Phase 1 — fit-mode core.** α-free simultaneous F+B count fit (the proper
way to get α from a calibration run, replacing grid estimates) and
single-histogram N0/BG fits. Both are small extensions of
`build_grouped_count_model`'s nuisance structure.

**Phase 2 — window & nuisance flexibility.** Interior exclude range (mask
support in the residual builder — also useful for laser/RF artefacts);
fittable t0 offset; optional baseline-drift term.

**Phase 3 — count-loss & double pulse.** Deadtime parameters (DT0/DT1, then
polynomial/power-law extensions) as optional fit parameters with a
"promote to correction" action mirroring WiMDA's Send-to-Group; double-pulse
convolution for ISIS double-pulse mode.

**Out**: negative-muon mode (deferred brief); RRF coupling (→ `rrf`);
KEK spill deadtime (no data source).

## Current Asymmetry state

Count-domain fitting exists only as the multi-group fit
(`grouped_time_domain.py` + `MultiGroupFitWindow`): per-group N0/bg/amp/phase
nuisances over a shared normalized-polarization model. No α-free mode, no
single-histogram mode, single contiguous fit window, t0 fixed by loader.

## GUI/UX sketch

Extend the Multi-Group Fit window with a fit-target selector
(All groups / F+B with free α / Single group) rather than burying modes in
the main fit panel — count-domain fitting stays one coherent surface.
Exclude-range as a second (optional) draggable span on the plot, reusing
`draggable_handles.py`. "Promote fitted α / deadtime to grouping" as an
explicit button with a confirmation showing before/after values.

## Physics-correctness notes

Count fits must use Poisson-aware weighting at low counts (WiMDA uses
Gaussian σ throughout; document the deviation). The fitted-α path should
report correlation between α and amplitude parameters (they are strongly
correlated in TF runs — MINOS from `fit-workflow-diagnostics` will matter
here; note the synergy, not a dependency).

## Conflicts & dependencies

Primary surfaces: `grouped_time_domain.py` (+ possibly a new
`fitting/count_domain.py`), `multi_group_fit_window.py`, light `fit_panel.py`
plumbing. Wave A-disjoint. `fit-workflow-diagnostics` (Wave B) owns
`fit_panel.py`/`engine.py` — keep Phase plumbing minimal there and land
before Wave B starts.

## Verification sketch

α from fgFB-style fit vs grouping-dialog estimators on the same TF
calibration run (and vs WiMDA's value); single-histogram fit on PSI
continuous data recovers N0·e^{−t/τ} envelope; synthetic double-pulse data
(from `simulate-mode` if available, else handwritten) round-trips dpsep;
exclude-range: fit with a masked artefact matches fit of clean synthetic.
