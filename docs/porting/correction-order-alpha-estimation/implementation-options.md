# Implementation options

Two decisions: (A) how to make the `alpha` estimators consume corrected counts,
and (B) how to unify the grouping-setup previews. The approved direction is the
combination **A2 + B2**, with **A1** available as an interim first PR.

## A. Estimator input (the physics fix)

### A1 — Interim: route the existing dialog through the corrected pipeline

Minimum acceptable fix, mergeable on its own.

- Hand `AlphaCalibrationDialog` the deadtime and background policy (currently it
  receives only groups/forward/backward/excluded).
- Its `_run_alpha_estimate` worker and `_binned_curve`/`_grouped_counts` preview
  build F/B via the shared corrected builder (see A2) instead of
  `group_forward_backward` on raw counts.
- Its before/after preview then reduces the *corrected* counts, so a centred
  "after" curve is honest.
- Show a provenance line naming the deadtime/background settings in force.
- Fix `_estimate_run_alpha` (`profiles.py:1169`) on the same seam.

Pros: small, closes the physics bug immediately, low UI churn.
Cons: keeps the three modal dialogs and the two-preview architecture; the
staleness failure mode (estimate, then change corrections) remains until B lands.

### A2 — Shared corrected-F/B builder (the durable seam)

Extract the correction stages of `reduce_grouped_asymmetry` (deadtime → t0 →
group → background) up to **but not including** `binned_fb_asymmetry` into a
reusable core function returning corrected `(forward, backward, forward_error,
backward_error, common_t0, bin_width)`. Both the reduction and every `alpha`
estimate call it, so they agree by construction.

- `estimate_alpha` / `estimate_alpha_detailed` keep their current signatures
  (they already take F/B arrays) — only the *callers* change to pass corrected
  arrays and the propagated per-bin errors.
- Preserve error semantics: the `diamagnetic` weights must use raw-count Poisson
  variance plus pedestal-estimate variance (carry `forward_error`/
  `backward_error` through, do not recompute from subtracted counts).
- Reference-run background needs the `ReferenceResolver`; the builder accepts it
  (as `reduce_grouped_asymmetry` already does) and the two estimate entry points
  supply or explicitly degrade it.

This is the recommended core change and is a prerequisite for A1 too (A1 is A2
wired only into the existing dialog).

## B. Preview / UI

### B1 — Keep separate dialogs, fix their previews

Pair with A1: each modal keeps its own preview but renders through the corrected
pipeline. Cheapest, but two previews of "the same" reduction can still drift and
the modal-per-correction workflow persists.

### B2 — Unified Corrections panel (approved target)

One non-modal panel on the grouping window (reuse `PanelSection`), replacing the
deadtime, background, and alpha modals. **One pipeline, one preview truth**:
every preview renders through the shared corrected reduction; the alpha view is
that same pipeline with `alpha` toggled `1 ↔ alpha_hat`.

Concept:

- **Pipeline strip** across the top — `Deadtime → Group → Background → α` — each
  a collapsible section with its settings and a one-line summary chip when
  collapsed ("τ from file, per-detector", "flat, tail-fit 8–32 µs",
  "α = 1.037 diamagnetic"). Makes the *order* itself visible (half the physics
  lesson).
- **One live preview** below, always the full corrected reduction. In the α
  section's calibrate mode it overlays ghosted `A(t; α=1)` against solid
  `A(t; α_hat)` with a zero reference line.
- **Numeric acceptance readout** — `⟨A⟩` over the good-bin window ± uncertainty
  ("residual baseline: 0.0004 ± 0.0011 — centred") — the honest replacement for
  eyeballing. It is now actually centred because the estimator eats corrected
  counts.
- **Diagnostic per-stage toggles** — preview-only checkboxes, clearly badged
  "diagnostic view — reduction always applies all stages" — preserve the old
  dialogs' "see each correction's incremental effect" value without letting a
  partial view masquerade as the reduction. Default all-on.
- **Workflow intact** — the α section keeps "pick wTF calibration run → method →
  window → Estimate"; Estimate routes through the shared pipeline using the
  *current* panel deadtime/background settings.
- **Staleness badge** — store with `alpha_hat` a digest of the correction
  settings it was estimated under; when the user later changes deadtime/
  background, show "α estimated under different corrections; re-estimate"
  (reuse the FFT-staleness banner pattern). Closes the calibration-drift failure
  mode permanently, not just at estimate time.

Performance — a non-problem with staged caching (already this codebase's idiom):

- Cache deadtime-corrected grouped counts keyed on `(raw digest, τ table,
  grouping, t0)`.
- Background subtraction is a cheap array op on top.
- The α slider/estimate touches only `O(n_bins)` asymmetry formation.
- Full-from-raw recompute happens only when an upstream stage changes, debounced
  through the single-flight worker per the responsiveness guidelines. The old
  "preview raw counts because full reduction is expensive" rationale does not
  survive staged caching.

## Recommended sequencing

1. **PR 1 (physics fix, A2 + A1):** shared corrected-F/B builder; route the
   existing calibration dialog and `_estimate_run_alpha` through it; the dialog
   preview reduces corrected counts; add a provenance line. Fully closes the
   correctness bug and is independently shippable.
2. **PR 2 (UI, B2):** unified Corrections panel, staged caching, staleness badge;
   retire the three modals.

Splitting keeps the correctness fix reviewable and lets the UI change land
behind it. If a single PR is preferred, land A2 + B2 together and drop A1.

## PR 2 build plan (unified Corrections panel — chosen 2026-07-16)

Ben chose the full unified panel (B2). It is a **multi-commit, likely
multi-session** build; the branch carries a partial panel between increments.
Each step below is an individually-green commit.

**Retirement inventory (verified 2026-07-16).** `DeadtimeDialog`,
`BackgroundDialog` and `AlphaCalibrationDialog` are launched **only** from
`grouping/dialog.py` (`:2272`, `:2424`, `:3488`) — no external callers, so
retiring them touches only the grouping dialog. Dedicated tests to migrate:
`tests/gui/test_deadtime_dialog.py`, `test_background_dialog.py`,
`test_alpha_calibration_dialog.py`. Screenshot scenario to retarget:
`docs/screenshots/scenarios/alpha_calibration_dialog.py` (the only one).

**Foundations to reuse.** `PanelSection(collapsible=True, expanded=…, hint=,
settings_key=)` with `.set_title_suffix` (the collapsed summary chip),
`.body_layout`, `toggled` (`gui/widgets/panel_section.py`); `ActionFooter`;
`make_warning_banner(text, severity="warn")` (`styles/widgets.py:554`) for the
staleness badge; `GroupingPreviewPane` (off-thread `TaskRunner`, 300 ms debounce,
generation counter, **no caching today**); `fourier_grouping_digest(run)`
(`core/fourier/spectrum.py:307`) covers the deadtime+background+groups keys and
excludes cosmetics/alpha — the reuse candidate for the staleness digest.

**Sequence (each a green commit):**

1. **Staleness badge** (smallest, highest correctness value; lands first). Stamp
   a digest of the deadtime+background settings a calibrated α was measured under
   into the α provenance (`alpha_correction_digest`, same provenance move as PR
   1). Show a `make_warning_banner` in the grouping dialog when the current
   corrections differ from that digest and the α is `calibrated`
   ("α estimated under different corrections — re-estimate"); clear on
   re-calibrate. Closes the calibration-drift failure mode without any layout
   change.
2. **Layout scaffold.** The right pane has no `QScrollArea` and a fixed 560 px
   height; four sections + preview won't fit. Add a scroll area and an empty
   Corrections `PanelSection` container, no behaviour change, tests green.
3. **Preview unification + centring readout — DONE** (commit e34f1a5). The pane
   worker now runs on the corrected-counts seam (`corrected_grouped_counts` once,
   then `binned_fb_asymmetry` per α); the calibrate overlay forms the α=1 and
   draft-α curves from the *same* corrected counts (one reduction, two curves) and
   reports the inverse-variance-weighted residual baseline ⟨A⟩ in the status. The
   grouping dialog requests the overlay when α is calibrated. **Cross-request
   staged caching was deferred** with rationale: the pane is already off-thread +
   debounced and the single-pass overlay avoids the double reduction, so a
   cross-thread corrected-counts cache is a riskier threading change without a
   demonstrated need — revisit only if a large-detector run shows α-tick lag.
   Known cosmetic: a good window extending into the fully-decayed tail lets the
   α=1 ghost blow the Y-scale (real windows trim it; same exposure as the
   pre-existing single-curve preview) — clamp Y to the α̂ curve if it bites.
4. **Inline the sections** (one modal per commit) — IN PROGRESS.
   - **Background — DONE.** `BackgroundSectionWidget` (`background_section.py`)
     hosts the mode combo + status + reference picker inline in the Corrections
     panel; `BackgroundDialog` and its off-thread group-summation preview
     machinery retired. The dialog keeps `_background_mode`/`_background_run_payload`
     as the source of truth (the section writes them via a `changed` signal), so
     `_current_grouping_payload` is unchanged. Tests migrated to the section +
     new `test_background_section.py`. Minor deferred: the inline status drops the
     live tail-fit rate readout (the unified preview shows the subtraction). This
     confirmed the lift is a clean re-hosting — the pattern for the next two.
   - **Deadtime — DONE.** `DeadtimeSectionWidget` (`deadtime_section.py`) hosts
     the mode radios + per-detector table (Fill-all / Cal) + estimate combo +
     max-correction summary inline; `DeadtimeDialog` retired. Same source-of-truth
     pattern (`_deadtime_*` state, section folds edits via `changed`). Deleted
     deadtime_dialog.py + test; new test_deadtime_section.py.
   - **α (single) — DONE.** `AlphaSectionWidget` (`alpha_section.py`) hosts the
     TF-highlighted calibration-run picker + method combo + Estimate inline; on
     success it emits a calibrated `AlphaPolicy` the dialog applies via
     `_apply_calibrated_policy` (α spin + provenance + staleness digest), driving
     the shared preview overlay. The off-thread estimate worker moved to
     `alpha_section.py`. The section pulls its group pair / correction context
     fresh at Estimate time via a provider callback. Corrections panel now hosts
     deadtime + background + single-α inline.
   - **α (vector / per-projection) — DONE.** The standalone
     `AlphaCalibrationDialog` is **fully retired** (`alpha_calibration_dialog.py`
     + `test_alpha_calibration_dialog.py` deleted). The per-projection vector
     table (`dialog.py`) gained a shared **Calibration run** + **Method** row and
     drives its per-axis and "Estimate All α" estimates inline through the shared
     `run_alpha_estimate` / `build_alpha_request` worker on the dialog's own
     `_vector_alpha_tasks` `TaskRunner`. "Estimate All" is **serialised** one axis
     at a time (a queue + single result token) so each result routes to its own
     spin — never fired in parallel. Every result still flows through
     `_apply_calibrated_policy(slot, spin, policy)` unchanged (P_z→single sync,
     provenance, staleness digest). Run-combo population and the request builder
     were lifted to module-level helpers in `alpha_section.py`
     (`populate_calibration_run_combo`, `build_alpha_request`,
     `grouping_for_reduction`, `resolve_reference`, `good_window`) so the single-α
     section and the vector path share one implementation. The
     `alpha_calibration_dialog` screenshot scenario is retargeted to the grouping
     window's inline α section; `detector_grouping.rst` and
     `calibration_grouping_emu.rst` (both figures + captions) were updated to the
     inline flow with verbatim UI strings, and `ARCHITECTURE.md` drops the retired
     module. (While there, the stale "opens the deadtime/background dialog" prose
     left by the earlier deadtime/background inline commits was corrected too.)
5. **Diagnostic per-stage toggles** (preview-only). **Trap:** they must write to
   the preview request only, never the persisted grouping payload — otherwise
   they silently change real reductions (the exact bug class PR 1 fixed).

## Docs / provenance obligations

- Record the WiMDA fidelity divergence in this study's `README.md` decision log
  when the implementation lands (per the porting workflow's two-pass rule).
- Update the Sphinx page that documents the grouping/alpha calibration workflow
  (find via `reference/index.rst` or `grep -rl alpha docs/`), quoting new UI
  strings verbatim, and refresh the screenshot scenario if the UI changes
  visibly.
