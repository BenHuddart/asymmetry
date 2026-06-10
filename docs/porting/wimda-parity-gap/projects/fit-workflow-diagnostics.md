# Project brief: fit-workflow-diagnostics

Umbrella: `wimda-parity-gap` · Wave B (after `count-domain-fit-modes`) ·
Size M · absorbs the `minos-error-analysis` candidate (top-scored, 20)

## Motivation

A basket of fit-quality and fit-workflow items that share one surface
(`engine.py` + `fit_panel.py`): the long-standing MINOS candidate, WiMDA's
χ² quality band, sequential warm-start for temperature scans, and a mid-fit
abort. Individually small; together one coherent session.

## WiMDA reference

χ²/dof statistical target band + good/poor/**overdone** classification from
the inverse incomplete gamma (`FitOpt.Rgoodfit`, `Analyse.Chi2Update`,
`Model.pas:1447–1516`) — "overdone" flags over-parameterised fits, which no
current Asymmetry surface does; sequential warm-start "itPrevious"
(`BatchFit.pas`: carry run N's fitted values into run N+1 — essential for
scans crossing transitions where fixed seeds fail); abortable fits
(`FitStatusForm` + `StopFitting` polled per iteration); persistent on-disk
fit logs (`.fit`/`.mfit`/`.bfit`, `Fitting.pas`). MINOS itself is a musrfit
strength (WiMDA lacks it) — included here because it's the same engine seam
and the highest-scored open candidate.

## Scope

- **MINOS**: expose `Minuit.minos()` per-parameter on demand (it's slow —
  opt-in button/flag, not default), surface +err/−err in the parameter
  table and result summaries; explicit HESSE invocation where it improves
  covariance fidelity.
- **χ² quality band**: target χ²ᵣ interval from `scipy.stats.chi2`
  (confidence level configurable, default matching WiMDA's R=0.95) +
  good/poor/overdone verdict, shown with every fit result (single, batch,
  grouped, model fits — implement once in `result_summary.py`, consumed
  everywhere; coordinate with `model-function-parity` Phase 2 which wants
  the same helper).
- **Sequential warm-start**: a "seed from previous member" option in batch
  series fits (`fit_grouped_series` / Global tab batch path), ordered by the
  series order key; document interaction with existing seed averaging.
- **Mid-fit abort**: cancellation hook for the fit workers (iminuit supports
  interruption via callbacks); Stop button in the fit panel + wizard.
- **Persistent fit log** (small): optional append-to-file of the existing
  result summaries with provenance.

**Optional phase (depends on `run-arithmetic`)**: in-batch run co-adding
(WiMDA Smooth/Bin) and re-fit-of-co-added-selection / rebin-table-by-refit
(`FitTableUnit.pas:718`, `Rebinning.pas`) — record in study; implement only
if the dependency has landed and session budget allows.

## Current Asymmetry state

migrad/simplex only; χ²ᵣ reported bare; single-fit seeds batch averages but
no N→N+1 chaining; QThread workers with no cancel; in-app log panel only.

## GUI/UX sketch

MINOS as a per-fit "Asymmetric errors" action with a progress indicator;
quality verdict as a coloured chip next to χ²ᵣ with a tooltip explaining the
band (teach, don't just judge); warm-start as a seeding-mode dropdown in the
batch tab; Stop button replacing the disabled Fit button while running.

## Conflicts & dependencies

Primary surfaces: `engine.py`, `result_summary.py`, `fit_panel.py`.
Sequenced after `count-domain-fit-modes` (light `fit_panel.py` contact).
χ²-band helper shared with `model-function-parity`. Optional phase depends
on `run-arithmetic`.

## Verification sketch

MINOS vs HESSE on a deliberately asymmetric likelihood (low-stat KT fit) —
intervals differ in the documented direction; χ² band against tabulated
chi2 quantiles; warm-start on the EuO T-scan: chained seeds converge through
Tc where fixed seeds fail or wander; abort leaves state consistent (no
partial result recorded).
