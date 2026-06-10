# Count-domain fit modes — implementation options & plan

This document carries the reuse audit, the chosen architecture, and the full
three-phase implementation plan. It is written to be startable cold: a later
session can implement from here plus the committed source without re-deriving.

## Decisions (fixed with Ben, 2026-06-10)

Pre-study: oracle = transcribe + synthetic + cross-check; F+B = two groups
(one F, one B); muon lifetime fixed (free-τ a follow-on). Post-study:

| Decision | Choice |
|---|---|
| Count weighting | **Poisson (Cash) default, Gaussian √N selectable.** Raw-count model. `fgAll` stays Gaussian (unification a follow-on). |
| GUI home | **Fit-target selector in the Multi-Group Fit window** (All groups / F+B free α / Single group). Minimal `fit_panel.py` contact. |
| Results storage | **Multi-Group window results** as source of truth; a single-group fit also **mirrors into the `TimeGroups` stored-fit slot** for the Individual-Groups plot. |
| Promote-to-grouping | **Explicit button + before/after confirmation,** additive-vs-replace choice, re-reduce on confirm. |
| Exclude-range UI | Optional **second draggable span** reusing `draggable_handles.py` (brief recommendation; not re-asked). |
| Double-pulse source | **Instrument metadata default → user override → optionally fittable** (brief-determined; not re-asked). |

## Reuse audit (mandatory — maps each mode onto the existing seam)

The existing count-domain machinery is `build_grouped_count_model` (returns
N₀·(1 + amplitude·P) + bg·e^(t·λ_μ) on lifetime-corrected counts),
`fit_grouped_time_domain` / `fit_grouped_series` (drive `FitEngine.global_fit`),
and `build_grouped_time_domain_groups` (the grouping → lifetime-corrected
count-domain pipeline). `MultiGroupFitWindow` wraps `GlobalFitTab(member_kind=
"groups")` in `fit_panel.py` = WiMDA `fgAll`.

| New mode | Maps onto | Verdict |
|---|---|---|
| **All groups** (`fgAll`) | `build_grouped_count_model` + `fit_grouped_time_domain` | **Reuse unchanged.** The selector's default branch. |
| **Single histogram** (`fgForward/Backward/Selected`) | The one-group degenerate case: `build_grouped_count_model` evaluated on a single dataset with amplitude fixed ±1, phase fixed 0 | **Reuse the model builder**, new thin driver entry (the ≥2-group guard in `fit_grouped_time_domain` does not apply to a one-group single fit). No new model code. |
| **α-free F+B** (`fgFB`) | A two-domain fit with a √α tie coupling one shared N₀ across F and B | **One new ~12-line model wrapper** (`build_fb_count_model`). Justified below. |

### Why the α tie needs new model code (the `Parameter.expr` finding)

The obvious "no new code" route is to express N₀_F = N₀·√α through the
parameter machinery. **This is not possible with the current engine.**
`Parameter.expr` exists as a field and `is_constrained` honours it (the
parameter drops out of `free_parameters`), but **neither `FitEngine.fit` nor
`FitEngine.global_fit` ever evaluates `expr`** — only equality link-groups
(`link_followers`) are applied, and an equality link cannot express a √α
scaling. So the √α coupling must live either in the engine (a general expr
evaluator — out of scope; the next wave owns `engine.py`) or in a model
wrapper. A wrapper is the smaller, lower-risk change.

`build_fb_count_model` (new, in `count_domain.py`) takes **global** params
`alpha`, `N0`, the physics params; a **local** param `background`; and a
**fixed local** `sign` (+1 forward, −1 backward):

```
fb_count_model(t, *, alpha, N0, background, sign, **physics):
    P = physics_fn(t, **physics)              # dimensionless polarization
    return N0 * alpha**(sign/2) * (1 + sign*P) + background*exp(t/τ_μ)   # lifetime-corrected
# raw form: multiply the whole thing by exp(-t/τ_μ)  →  N0*alpha**(sign/2)*(1+sign*P)*exp(-t/τ_μ) + background
```

This fits the **existing** global/local engine contract exactly (`alpha`, `N0`,
physics global; `background` local; `sign` fixed local), so the F+B fit reuses
all of `global_fit`'s param-packing and result extraction. α and its
correlation with the amplitude come straight from Minuit. **No engine change.**

### Raw counts without model duplication

The new modes fit **raw** counts (for an exact Poisson cost). Rather than build
a second model, note:

> N_raw(t) = e^(−t·λ_μ) × [ N₀·(1 + s·P) + bg·e^(t·λ_μ) ] = e^(−t·λ_μ)·N₀·(1 + s·P) + bg

i.e. the raw model is the **existing lifetime-corrected builder times
e^(−t·λ_μ)**. `count_domain.py` wraps `build_grouped_count_model` (and the new
`build_fb_count_model`) with the e^(−t·λ_μ) factor. No model-building
duplication; the raw-vs-corrected choice is a one-line factor.

Raw grouped counts come from `build_grouped_time_domain_groups` extended with a
`lifetime_corrected: bool = True` flag (when `False`: skip the e^(t·λ_μ) scaling,
return √N Poisson errors). This reuses the entire grouping / deadtime / bunching
/ t₀ pipeline — no duplication.

### What is genuinely new (and why it isn't duplication)

- `build_fb_count_model` — the √α tie the existing builder can't express.
- `count_domain.py` driver — selects Poisson (Cash) vs Gaussian √N and assembles
  the iminuit cost over the **shared** model. The cost/residual differ by
  statistics, not by re-implementing the model. Results reuse `FitResult` /
  `GroupedTimeDomainFitResult`. The brief's no-duplication rule targets
  model-building / residual / results-storage; none of those are duplicated.
- Keeping `engine.py` untouched is deliberate: the next-wave
  `fit-workflow-diagnostics` project owns it, and a per-domain Poisson cost is a
  larger engine change than a focused, owned driver. A follow-on records the
  eventual unification (give `FitEngine` a cost-factory and route `fgAll`
  through Poisson too).

## Architecture summary

```
core/fitting/
  grouped_time_domain.py   (existing)  + lifetime_corrected flag; + build_fb_count_model
  count_domain.py          (NEW)       raw-count driver: Poisson|Gaussian cost,
                                       single-histogram + F+B entry points,
                                       optional exclude/t0/baseline/deadtime/double-pulse,
                                       returns FitResult / GroupedTimeDomainFitResult
  engine.py                (UNCHANGED)
core/transform/
  deadtime.py              (existing)  + promote_deadtime_to_grouping helper
core/simulate.py           (existing)  + double-pulse synthesis option
core/project/schema.py     (existing)  + small additive count-fit config block
gui/windows/multi_group_fit_window.py  + fit-target selector, mode routing,
                                          promote button, results mirror
gui/panels/fit_panel.py    minimal     a mode arg threaded into the grouped fit call
gui/.../draggable_handles  reuse       second optional exclude span
docs/user_guide/fit_functions/         NEW pedagogical pages per feature
```

## Phase 1 — fit-mode core (α-free F+B + single histogram)

Ordered steps:

1. **Raw-count groups.** Add `lifetime_corrected: bool = True` to
   `build_grouped_time_domain_groups` and `build_grouped_time_domain_datasets`.
   When `False`: return raw counts, √(max(N,1)) errors, no e^(t·λ_μ) scaling.
   Refactor the per-group count-building loop into a small reusable helper so a
   single group can be built without the ≥2 guard.
2. **F+B model.** Add `build_fb_count_model(physics_fn)` to
   `grouped_time_domain.py` (sits beside `build_grouped_count_model`).
3. **Driver.** New `core/fitting/count_domain.py`:
   - `_poisson_cost` / `_gaussian_cost` builders over (t, n, model).
   - `fit_single_histogram(dataset, group_id, physics_fn, params, *, side,
     cost="poisson", t_min, t_max, exclude=None, ...)` → `FitResult`. Wraps the
     shared model with e^(−t·λ_μ); drives iminuit directly.
   - `fit_fb_alpha(dataset, forward_group, backward_group, physics_fn, params,
     *, cost="poisson", ...)` → `GroupedTimeDomainFitResult` with α in
     `shared_parameters` and the (α, amplitude) covariance populated.
   - Both honour fixed muon lifetime; both accept the Phase-2/3 optional knobs as
     defaulted-off kwargs so later phases only flip flags.
4. **GUI selector.** Add the fit-target selector to `MultiGroupFitWindow`
   (All groups / F+B free α / Single group). 'All groups' → existing path. The
   other two → new core entry points. Thread a `mode` arg through the one
   `GlobalFitTab` grouped-fit call (minimal `fit_panel.py` contact). For F+B,
   expose a Forward/Backward group picker; for single, a group picker + side.
5. **Cost toggle.** A Poisson/Gaussian control on the count-fit surface
   (default Poisson).
6. **Results.** Route results into the window's results model; for single-group,
   mirror into the `TimeGroups` stored-fit slot.
7. **Tests** (`tests/test_count_domain_fits.py`): transcription oracle for the
   raw/FB models; synthetic α recovery (α₀ ∈ {0.8,1.0,1.3}); α–amplitude
   covariance present; single-histogram synthetic + EuO real; single = one-group
   builder equality; Poisson-beats-Gaussian-at-low-counts; `fgAll` regression
   untouched. GUI smoke for the selector.
8. **Docs**: `docs/user_guide/fit_functions/` pages for the F+B α calibration
   fit and the single-histogram fit (when-to-use registers).

Phase-1 end: validate green, main-mergeable, milestone commit.

## Phase 2 — window & nuisance flexibility

1. **Exclude range (core).** `exclude: tuple[float,float] | None` on the group
   builders and the `count_domain` entry points; drop interior bins (endpoints
   inclusive per WiMDA). Works for single, F+B, and (additively) the grouped
   path.
2. **Exclude range (GUI).** Optional second draggable span via
   `draggable_handles.py`; commit to the fit config.
3. **Fittable t₀.** Optional global `t0` param in the count models: evaluate at
   t + t₀. Off (fixed 0) by default → numerically identical to Phase 1.
4. **Baseline drift.** Optional stretched-exp envelope A_eff = A·e^(−(λ_b·t)^β_b)
   on the polarization (divergence from WiMDA's offset-only application is
   documented in comparison.md). Off by default.
5. **Tests**: masked-artefact == clean fit; exclude bin-dropping unit; t₀
   recovery + off-state no-op; baseline-drift recovery + off-state no-op.
6. **Docs**: exclude-range / t₀ / baseline-drift usage notes.

Phase-2 end: validate green, milestone commit.

## Phase 3 — count loss & double pulse

1. **Deadtime in fit (core).** Optional DT0 (Simple), DT1 (Linear), C2/C3/C4
   (Polynomial), power-law forms as a post-multiply loss factor on the count
   model, reading frame normalization from run metadata with a documented
   fallback. Off by default.
2. **Promote to grouping.** `promote_deadtime_to_grouping(grouping, fitted, *,
   additive)` in `deadtime.py`: write fitted DT0 into the grouping deadtime
   field (additive or replace), return before/after. Polynomial promotion =
   follow-on.
3. **Promote GUI.** Explicit button + before/after confirmation dialog;
   re-reduce the run on confirm. Same pattern for promoting fitted α into the
   grouping balance.
4. **Double pulse (simulate).** Add a double-pulse option to `core/simulate`
   (two time-shifted copies weighted exp(∓dpsep/2τ)).
5. **Double pulse (fit).** Optional double-pulse evaluation in the count models:
   physics at t ± dpsep/2, weighted, second pulse gated at t > dpsep/2. dpsep
   from instrument metadata → user override → optionally fittable.
6. **Tests**: deadtime loss-factor transcription; DT0 recovery; promote
   correctness + additive; DT0 vs `calibrate_deadtime_from_histograms`;
   double-pulse transcription; dpsep round-trip + single-pulse limit.
7. **Docs**: deadtime-in-fit + promote workflow; double-pulse fitting page.

Phase-3 end: validate green, milestone commit; study updated with final notes.

## Persistence (schema.py — small, additive, end-of-block)

Add a `count_fit` config block: target mode, group selection (F/B/single side),
cost choice, exclude window, enabled nuisances (t₀, baseline, deadtime model,
double-pulse dpsep + fittable flag). Round-trips with the project. Keep the
change append-only to avoid collisions with the other open PRs touching
`schema.py`.

## File-by-file touch list

- `core/fitting/grouped_time_domain.py` — `lifetime_corrected` flag; per-group
  build helper; `build_fb_count_model`.
- `core/fitting/count_domain.py` — **new** driver (Poisson/Gaussian; single +
  F+B; optional knobs).
- `core/transform/deadtime.py` — `promote_deadtime_to_grouping`.
- `core/simulate.py` — double-pulse synthesis option.
- `core/project/schema.py` — additive `count_fit` block.
- `core/representation/time.py` — single-group stored-fit slot mirror (if a slot
  field must be added; minimal).
- `gui/windows/multi_group_fit_window.py` — selector, routing, cost toggle,
  promote button, results mirror.
- `gui/panels/fit_panel.py` — **minimal**: a `mode` arg threaded into the
  grouped-fit call; group/side pickers for the non-`fgAll` modes.
- `gui/.../grouping_dialog.py` (or wherever the deadtime field lives) — receive
  promoted values; re-reduce.
- `tests/test_count_domain_fits.py` (+ Phase 2/3 test files or sections).
- `docs/user_guide/fit_functions/*` and its toctree (append-only).
- `docs/porting/index.json` — this study's entry (added at study commit).

## Recorded follow-ons

1. **Unify `fgAll` onto the raw-count Poisson driver** (give `FitEngine` a
   cost-factory; route grouped fits through Poisson). Deferred to avoid
   colliding with the next-wave `engine.py` owner.
2. **Free muon lifetime** (musrfit-style) as an optional count-fit parameter.
3. **Polynomial/power-law deadtime promotion** to the grouping (Phase 3 promotes
   DT0 only; the C2/C3/C4 polynomial deadtime fits but is not promoted).
4. **DT1 (linear) and power-law deadtime in the fit** — Phase 3 ships Simple
   (DT0) + polynomial (C2–C4); the linear-in-event-fraction (DT1) and power-law
   forms need the ISIS frame-fraction/event-fraction metadata block and are
   deferred.
5. **Robust dpsep refinement** — the double-pulse separation is fixed by
   default because the non-smooth pulse-onset gate defeats migrad; a coarse
   1-D scan → migrad refinement would make dpsep fittable. (Finding recorded in
   comparison.md.)
6. **FB double-pulse** — Phase 3 ships double-pulse for the single-histogram
   mode (the round-trip target); folding it into the F+B α model is deferred.
7. **Plot overlay for count fits** — count-fit results are reported
   numerically; overlaying the raw-count fit curve on the (lifetime-corrected)
   Individual-Groups plot is a UI-polish follow-on.
8. **MINOS errors on α** — `fit-workflow-diagnostics` (Wave B) adds MINOS; α is
   the prime beneficiary given its amplitude correlation. Synergy, not a
   dependency.
9. **WiMDA fitted-baseline "Set BG" workflow** — out of scope; the Phase-2
   baseline-drift term is independent and does not pre-empt it.

## Implementation outcome (2026-06-10)

All three phases shipped, each `validate`-green (Phase 1: 2126; Phase 2: 2135;
Phase 3: 2145 passed). The architecture held: `engine.py` untouched; the new
`count_domain.py` driver reuses `build_grouped_count_model` (single-histogram)
and the one new `build_fb_count_model` (√α tie), with the raw model = the
existing builder × e^(−t/τ). The GUI surfaced through `MultiGroupFitWindow`'s
fit-target selector with minimal `fit_panel.py` plumbing (a routing branch in
`GlobalFitTab._run_grouped_time_domain_fit` + free-amplitude seeding, since the
count modes fit the model amplitude that the normalised fgAll path pins to 1).

## References

- *Muon spectroscopy* (muon-spectroscopy textbook).
- WiMDA `FitTyps.pas`, `AsymFitFunction.pas`, `Analyse.pas` (F. L. Pratt,
  ISIS/RAL) — transcription/behaviour oracle only.
- musrfit `PRunSingleHisto` (A. Suter, B. M. Wojek) — raw-count single-histogram
  reference and cost selection.
