# Wave A closeout audit

Date: 2026-06-11, on main at PR #43's merge; §2 updated after PR #44. Companion to the durable
[decision-record.md](decision-record.md). Three audit passes: shipped-vs-brief
verification, Wave B brief impact, and a shallow collision scan (deep
reconciliation deliberately deferred to a dedicated session).

## 1. Status — Wave A complete

| Project | PRs | Outcome |
|---|---|---|
| model-function-parity | #32, #38, #39 | Shipped + follow-ons (arbitrary-X, error modes, results recursion, x-uncertainty, `⊕`) |
| simulate-mode | #33, #37 | Shipped + follow-ons (templates, gallery, pull diagnostic, multi-group) |
| data-reduction-parity | #34, #36 | Shipped + follow-ons (exclusion chokepoint, reference-run resolution, period-mapped reload, mask, quality batch) |
| asymmetry-error-propagation | #35 | **Emergent project** (out of simulate verification): exact Poisson σ_A replaces Mantid (1+A²) |
| maxent-completion | #40 | Shipped (overlay, pulse kernel, ZF/LF + SpecBG, deadtime/phase calibration) |
| count-domain-fit-modes | #41 | Shipped + six follow-ons (FB double-pulse, dpsep scan, DT models, free-τ, overlay/promote, name guard) |
| frequency-domain-finishers | #42 | Shipped (conditioning, S/N, real+imag, Burg, diamag removal) |
| radical-correlation-spectrum | #43 | **Emergent project** (promoted from #42 optional scope): Corr/AvCorr → hyperfine axis |
| wave-a-strays | #44 | The two stray handoffs from §2, shipped after this audit caught them |

Wave A delivers the umbrella study's "analysis feature-complete" milestone:
every analysis-capability gap in the inventory is closed; remaining waves are
workflow machinery (B) and workflow-feel (C).

## 2. Stray handoffs — CLOSED (PR #44, 2026-06-11)

Two items had been deferred to another project that didn't deliver them;
both were caught by this audit and shipped together in PR #44:

1. **Two-period and count-mode simulation** (simulate-mode →
   count-domain-fit-modes; PR #41 shipped only the double-pulse slice) —
   delivered in `core/simulate.py`: period-aware synthesis verified through
   `select_period`/G−R combination, count-mode runs verified against the
   PR #41 single-histogram and α-free fit modes.
2. **Tail-fit background in the Fourier input path** (data-reduction-parity
   → frequency-domain-finishers; absent from PR #42) — delivered as a
   tail-fit mode of `core/fourier/grouped.py`'s background options, reusing
   the single core estimator (no new background path; see F3 note in §5).

With these, every functionality item in the Wave A scope is on main.

## 3. Wave B/C brief impact

Nothing was wholesale delivered out from under Wave B — MINOS, warm-start,
moments, RRF, the plugin API, co-add/co-subtract and all of Wave C remain
genuinely open. Revisions needed before launch (ranked):

1. **fit-workflow-diagnostics** — the χ² band helper already exists
   (`core/fitting/fit_quality.py`, PR #32) but is consumed only by the
   model-fit dialog: rewrite that scope item as "wire `fit_quality` through
   `result_summary.py` and the fit panels". Disambiguate "warm-start" —
   `global_fit_wizard.warm_start_source` now names a different feature
   (single-fit→global seeding, not N→N+1 chaining). Inherits from #41: unify
   `fgAll` onto the Poisson cost factory; MINOS on α. Drop stale sequencing.
2. **python-user-functions** — registration-API surface must cover fields
   added in Wave A (`category`/`domain`/`fixed_params` on components;
   `scopes`/`fwhm_factor` on parameter models; the `⊕` grammar). Seams
   unchanged (both registries still plain dicts).
3. **spectral-moments** — GUI sketch predates the post-#40 panel
   (reconstruction toggle, layout controls) and the #42/#43 "derived display
   mode" pattern (Burg, correlation), which is now the natural integration
   shape; units = pure reuse of `core/fourier/units.py`; trend target is
   richer (#38 arbitrary-X).
4. **rrf** — units helper now exists (reuse); plot integration targets the
   new time-view-mode seam (`set_time_view_modes`).
5. **run-arithmetic** — intact; update `_coadd_datasets` references (moved,
   grew compatibility checks + rebuild path) and point co-subtract at the
   existing `subtract_scaled_counts` / reference-run resolution chokepoint.
6. **workflow-visualisation** (Wave C) — intact; #36/#41 covered none of its
   scope; note "Overlay" now names multi-run overlay (rename the F,B overlay
   item); all three touchpoint files grew substantially.

## 4. Collision watchlist — RESOLVED (reconciliation study, 2026-06-11)

Every flag below now has an investigated verdict agreed with Ben and a
scheduled phase: see [reconciliation-study.md](reconciliation-study.md)
(evidence + decisions, including one new flag NEW-R1 and three partial
refutations) and [reconciliation-plan.md](reconciliation-plan.md) (five
phases). The table is kept as the original point-in-time record.

The recurring pattern is not duplicated implementations but **N-way concept
proliferation without reconciliation/promote paths**, plus a few literal
helper duplicates. Chokepoint reuse was otherwise clean (deadtime promotion,
units, pulse model).

| # | Collision | Mechanisms | Severity |
|---|---|---|---|
| F1 | Quadrature two ways | `⊕` grammar operator vs `PowerLawQuadBG` component (acknowledged in-code; parameter names differ between routes) | duplicate-UX |
| F2 | Reference-field resolution duplicated | `plot_panel._frequency_reference_for_dataset` vs `core/fourier/spectrum._reference_field_gauss` (units themselves correctly unified on FieldUnit) | duplicate-code |
| F3 | Three frequency-domain background stories | pre-FFT per-group counts (`grouped.py`) vs post-FFT σ-clip baseline (`conditioning.py`) vs MaxEnt SpecBG — stackable with no UI hint; silent double-subtraction possible | concept + UX |
| F4 | Three diamagnetic-line paths | fit-and-subtract (`diamag.py`) vs diamag exclusion band vs frequency-domain fitting; the two checkboxes co-enable on one panel | duplicate-UX |
| F5 | Four t0 surfaces, no promote path | t0 search / grouping overrides / Fourier `t0_offset_us` / count-fit `t0` nuisance — fitted t0 cannot be promoted to grouping (deadtime can) | missing reconcile |
| F6 | Deadtime write paths | count-fit promote (`promote_deadtime_to_grouping`) vs MaxEnt apply (mainwindow handler) — verify same chokepoint/semantics | duplicate-code risk |
| F7 | α determination, no promote path | three grouping-dialog estimators vs α-free count fit — the statistically best method is the only one that can't persist its α | missing reconcile |
| F8 | "Exclusion" overload | five meanings, three semantics (drop / σ-inflate / band), two parameterisations ((t1,t2) vs centre±half-width) | terminology/UX |
| F9 | Forward collision: co-subtract | run-arithmetic (Wave B) must build on `subtract_scaled_counts` / reference-run chokepoint, not beside it | future overlap |
| F10 | Three trending/accumulation containers | Global-summary accumulator (#39) + results-recursion (#38) vs `GlobalParameterFitWindow` vs `FitSeries` — separately serialized | duplicate-UX |
| F11 | Repolarisation doc split | `MuRepolarisation` (parameter_trending docs) and ALC/integral workflow (alc_mode docs) never cross-reference | docs-only |
| F12 | Two archetype tables | `core/simulate_presets.py` mirrors `docs/screenshots/data/archetypes.py` constants by hand | duplicate-data |
| F13 | Spectral-estimator triad un-cross-referenced | Burg docs vs frequency-domain fitting vs MaxEnt — no "which one when" | docs-only |
| N1 | `_rebin_group_counts` duplicated verbatim | `grouped_time_domain.py:611` ≡ `fourier/grouped.py:25`; neither uses `transform/rebin.py` | duplicate-code |
| N2 | Small helper duplicates | `_group_names`, `_optional_float` in `fourier/spectrum.py` vs `maxent/engine.py` | duplicate-code |
| N3 | Two flat-background models stackable | count-fit `background` nuisance vs grouping background correction — α/N0 bias if both active | concept + guard |
| N4 | Three component registries with overlapping names | `MODELS` / `COMPONENTS` / `PARAMETER_MODEL_COMPONENTS` | by-design; naming hygiene |
| N5 | Three fit-assessment surfaces | `fit_quality` / `result_summary` / `pull_diagnostic` — no cross-reference | docs-only |
| N6 | FFT manual phase outside the phase-exchange loop | FFT `phase_degrees`/auto-phase vs MaxEnt fitted-phase exchange | by-design; note |

Not collisions (checked): free-τ (single `MUON_LIFETIME_US` source);
FieldUnit vs plot panel units (migrated cleanly, modulo F2); simulate
multi-group vs data-group co-add (different operations).

## 5. Suggested next-session agenda (collision reconciliation) — DONE

This agenda was executed on 2026-06-11; it is superseded by
[reconciliation-plan.md](reconciliation-plan.md). The shape held up
(mechanical wins first, then the promote pair, then backgrounds, then
UX/docs), with corrections recorded in the study: the F3 stacking fear and
N3 bias are interpretive traps, not mechanical ones (the four background
options — including the PR #44 tail-fit — are mutually exclusive and applied
once); F4 has two real paths, not three; and two of F10's "three" containers
were already one (`FitSeries`).
