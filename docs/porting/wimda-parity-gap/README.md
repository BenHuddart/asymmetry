# WiMDA functionality-parity gap study (umbrella)

Date: 2026-06-10. Branch: `study/wimda-parity-gap`.

## Purpose

Identify **all** WiMDA functionality still missing from Asymmetry and divide
it into a portfolio of projects that can be developed in parallel. The target
is **parity of functionality, not parity of implementation**: anything a user
can analyse in WiMDA should be analysable in Asymmetry, implemented with
modern numerics and clear UI/UX, favouring physically-correct approaches over
strict ports where the two are in tension.

This is an *umbrella* study. Each project below still gets its own full
study-first pass (`docs/porting/<slug>/`, five-doc template) when it starts;
the briefs in `projects/` are pre-studies that fix scope, size, phasing and
integration direction.

## Status (2026-06-11): Wave A COMPLETE

All six Wave A projects shipped, with follow-on PRs, plus two emergent
projects (exact asymmetry-error propagation #35; radical correlation
spectrum #43) and the two stray handoffs (#44) — PRs #32–#44; every
functionality item in the Wave A scope is on main. See
[wave-a-closeout.md](wave-a-closeout.md) for the audit (status table, stray
handoffs, Wave B brief revisions, collision watchlist) and
[decision-record.md](decision-record.md) for the consolidated record of
exclusions, deviations from WiMDA, and the reference-program bug ledger.
Wave B briefs require the revisions listed in the closeout before launch.
The collision-reconciliation study is done (2026-06-11): every watchlist
flag has an agreed verdict in
[reconciliation-study.md](reconciliation-study.md) and a scheduled phase in
[reconciliation-plan.md](reconciliation-plan.md); reconciliation Phase 2
must merge before `fit-workflow-diagnostics` starts.

## Documents

- [comparison.md](comparison.md) — the consolidated WiMDA → Asymmetry gap
  inventory (the canonical record; supersedes the WiMDA column of
  `docs/porting/comparison-matrix.md`).
- [decision-record.md](decision-record.md) — consolidated exclusions,
  deviations, and reference-program bug ledger (compiled at Wave A closeout).
- [wave-a-closeout.md](wave-a-closeout.md) — Wave A audit: status, stray
  handoffs, Wave B impact, collision watchlist.
- [reconciliation-study.md](reconciliation-study.md) — collision watchlist
  investigated: evidence and per-flag verdicts (decisions with Ben).
- [reconciliation-plan.md](reconciliation-plan.md) — five-phase,
  parallelisable implementation plan for the reconciliation verdicts.
- [implementation-options.md](implementation-options.md) — how the gaps were
  partitioned into projects; parallelisation (worktree) analysis and waves.
- [test-data.md](test-data.md) — verification corpus mapped to projects.
- [verification-plan.md](verification-plan.md) — how study claims and the
  eventual ports get verified.
- `projects/<slug>.md` — one-page brief per project.

## Method

Five parallel research passes (2026-06-10): four sweeping the WiMDA Delphi
source by functional slice (core data chain; fitting; frequency domain;
visualisation/periphery — every `.pas` unit in `wimda.dpr`, ignoring
`__history/`/`__recovery/`), one re-scanning Asymmetry's current capabilities
(the existing comparison matrix predates the MaxEnt, link-groups, muonium,
fit-function-parity, HAL-grouping, time-integral and period-selection
merges). WiMDA's own `FEATURE_MAP.json` was treated as untrusted after the
MaxEnt study showed it misroutes; the sweep read the units directly.

## Scoping decisions (2026-06-10, with Ben)

| Decision | Outcome |
|---|---|
| Previously-excluded items | Re-examined. **Burg all-poles MEM: now IN** (diagnostic framing) → `frequency-domain-finishers`. **Eigen.pas: out permanently** (mislabel — eigensolvers, not a spectral estimator; superseded by `np.linalg.eigh`). **Kramers–Kronig: out** (optical spectroscopy, not μSR). HDF4 `.nxs` stays out. |
| Peripheral tools | Everything inventoried; niche tools triaged port/adapt/drop in [comparison.md](comparison.md). |
| Negative-muon analysis | Deferred brief (`projects/negative-muon-analysis.md`), adapt-not-port; not in the parallel batch. |
| Live current-run monitoring | Optional late phase of `workflow-visualisation`, contingent on beamline access for testing. |
| Execution model | Parallel Claude sessions in separate worktrees; projects partitioned to minimise file overlap (see waves below). Long sessions preferred; only genuinely large projects are phased. |

## Headline findings

1. **Fit functions are done.** The `wimda-fit-function-parity` study's
   coverage was verified complete against WiMDA source — every remaining gap
   is machinery/workflow, not physics.
2. The largest genuine gaps cluster in **data reduction** (alpha estimation
   methods, tail-fit/run-subtraction backgrounds, variable/constant-error
   binning, t0 search, detector exclusion), **count-domain fit modes**
   (α-as-fit-parameter, single-histogram, double-pulse), and **MaxEnt
   completion** (time-domain reconstruction overlay, ISIS pulse shape, ZF/LF
   mode, moments).
3. Asymmetry's current curve-averaging co-add is **statistically wrong** for
   low-count runs; histogram-level run arithmetic is a correctness fix, not
   just a feature.
4. Several "gaps" dissolved on inspection: Multifit ⊂ Global tab; ALC
   workflow already shipped and exceeds WiMDA; Eigen.pas was never a spectral
   estimator; WiMDA's own MUD loader is a non-functional stub.

## Project portfolio

| # | Slug | Title | Size¹ | Phased? | Primary surfaces |
|---|---|---|---|---|---|
| 1 | `data-reduction-parity` | Alpha estimation, backgrounds, binning modes, t0 search, detector exclusion, period subset mapping | L | yes (3) | `core/transform/*`, `core/io/periods.py`, `gui/windows/grouping_dialog.py` |
| 2 | `run-arithmetic` | Histogram-level co-add / co-subtract with correct statistics | M | no | `core/data/`, `gui/panels/data_browser.py` |
| 3 | `count-domain-fit-modes` | α-free FB fit, single-histogram fits, exclude range, fittable t0/baseline, count-loss in fit, double pulse | L | yes (3) | `core/fitting/grouped_time_domain.py` (+new), `gui/windows/multi_group_fit_window.py` |
| 4 | `fit-workflow-diagnostics` | MINOS errors, χ² target band + overdone flag, sequential warm-start, mid-fit abort, fit log | M | no | `core/fitting/engine.py`, `gui/panels/fit_panel.py` |
| 5 | `frequency-domain-finishers` | Field-axis units, exclusion ranges UI, pulse-rolloff compensation, diamag removal, Burg pole scan, radical correlation (opt.) | M–L | yes (2) | `core/fourier/`, `gui/panels/fourier_panel.py` |
| 6 | `maxent-completion` | Time-domain reconstruction overlay, ISIS pulse shape, ZF/LF+SpecBG, deadtime/phase tables + exchange | L | yes (3) | `core/maxent/engine.py`, `gui/panels/maxent_panel.py` |
| 7 | `spectral-moments` | B_pk/B_ave/B_rms/skewness of field spectra + run averaging + trend export | S–M | no | new `core/fourier/moments.py`, small GUI |
| 8 | `simulate-mode` | Model → Poisson histograms → loadable NeXus; degrade-statistics | M | no | new `core/simulate.py`, new dialog |
| 9 | `model-function-parity` | Parameter-trending model functions + machinery (x2 variable, error modes, arbitrary x) | M | yes (2) | `core/fitting/parameter_models.py`, `gui/panels/model_fit_dialog.py` |
| 10 | `rrf` | Rotating-reference-frame display + fitting | S–M | no | new `core/transform/rrf.py`, `gui/panels/plot_panel.py` |
| 11 | `workflow-visualisation` | Run stepping, ASCII export+batch, raw/log views, snapped cursor readouts, events column; live-run (opt.) | M | yes (2) | `gui/panels/plot_panel.py`, `data_browser.py`, `mainwindow.py` |
| 12 | `python-user-functions` | Plugin API replacing both WiMDA DLL mechanisms | M | no | registries in `core/fitting/`, new plugin module |
| — | `negative-muon-analysis` | μ⁻ capture-lifetime analysis (adapt-not-port) | L | yes | **deferred** |

¹ S ≈ part of a session, M ≈ one long session, L ≈ 2–3 long sessions (one
per phase).

## Parallelisation waves

Chosen so concurrent worktrees touch disjoint modules (full analysis in
[implementation-options.md](implementation-options.md)):

- **Wave A** (up to 6 parallel): 1, 3, 5, 6, 8, 9
- **Wave B** (after A, up to 5 parallel): 2, 4, 7, 10, 12
  (7 after 6 — shares `maxent_panel.py`; 12 after 9 — shares
  `parameter_models.py` registry; 4 after 3 — shares fit-panel surface)
- **Wave C**: 11 (touches `plot_panel.py` + `data_browser.py`, which Waves
  A/B projects 10 and 2 also edit) (+ `negative-muon-analysis` if promoted)

Known shared-touch files even across "disjoint" projects: `mainwindow.py`
(menu/toolbar hooks — keep diffs minimal and additive),
`core/project/schema.py` (any project persisting new state bumps/extends the
schema — coordinate via small, append-only changes), `docs/porting/index.json`
(append-only). These are flagged in each brief's conflict section.

## Relationship to the existing roadmap

This study subsumes the WiMDA-sourced entries of `docs/porting/ROADMAP.md`:
`simulate-mode` and `moments-analysis` are promoted into the portfolio;
`rrf-transform`, `python-user-functions`, `phase-auto-calibration` (WiMDA
slice), `minos-error-analysis` and the period-arithmetic remainder are
absorbed into projects 10, 12, 6, 4 and 1 respectively. Non-WiMDA roadmap
items (`msr-import`, `lem-depth-profiling`, `structural-transitions`,
`bpp-relaxation`, MUD loading) are unaffected.
