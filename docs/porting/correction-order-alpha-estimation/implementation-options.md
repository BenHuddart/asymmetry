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
3. **Preview unification + staged caching.** Drive the shared `GroupingPreviewPane`
   from the α section with an α=1↔α̂ overlay; add staged caching (deadtime-corrected
   grouped counts keyed on `(histogram-content digest, τ table, grouping, t0)`, the
   background op cheap on top) so the corrected preview stays snappy on large
   detector counts — this is where PR 1's deferred perf cost comes due. Add the
   numeric centring readout.
4. **Modal-body lift spike + inline the sections** (one modal per commit;
   Background first — synchronous, 412 lines — to validate a modal body lifts
   into an embeddable `QWidget` before replicating). Retire each modal, migrate
   its tests, update the scenario. Order: Background → Deadtime → α.
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
