# Collision reconciliation — phased implementation plan

Date: 2026-06-11. Companion to
[reconciliation-study.md](reconciliation-study.md), which records the
evidence and Ben's per-flag decisions. Five phases, each a self-contained
session-sized package. Phase boundaries follow the conflict-surface analysis
the umbrella study used: primary surfaces are disjoint so Phases 2–4 can run
as parallel worktree sessions.

```
Phase 1 (solo, quick)
   └─► Phase 2 ∥ Phase 3 ∥ Phase 4 (parallel worktrees)
            └─────┴─► Phase 5 (after 2 and 3 merge)
```

Phase 1 lands first because it touches small spots across many modules that
later phases rebase on. Phase 5 follows 2/3 because all three touch
`mainwindow.py`. Shared-touch caveat (as in the umbrella study):
`mainwindow.py` and `core/project/schema.py` appear in more than one phase —
keep those diffs minimal and additive; `docs/user_guide/index.rst` is
append-only.

Programme-wide verification gates (every phase):

- `python tools/harness.py validate` green.
- Corpus regression for any phase touching reduction or fit inputs: CdS and
  EuO reductions bit-for-bit against main (the data-reduction-parity
  precedent).
- Behaviour-pinning before refactor: where a write path or duplicate is
  unified, first add a test pinning the current observable behaviour, then
  refactor against it.

## Phase 1 — mechanical unifications (N1, N2, F2, F12)

Scope: pure de-duplication; zero behaviour change by construction.

| Item | Change |
|---|---|
| N1 | `rebin_counts()` (count-preserving: sum counts, mean times) added to `core/transform/rebin.py`; both copies of `_rebin_group_counts` replaced by imports |
| N2 | `_optional_float` → `core/utils` coercion helper; `_group_names` → beside the grouping helpers; fourier/maxent import them |
| F2 | Core `_reference_field_gauss` publicised (e.g. `reference_field_gauss(run, dataset)`); `plot_panel._frequency_reference_for_dataset` delegates and converts units |
| F12 | Shared archetype constants get one public authority in `core/simulate_presets.py`; `docs/screenshots/data/archetypes.py` imports the shared subset; doc-only constants and the legacy 2.197 µs lifetime pin stay local |

Touch list: `core/transform/rebin.py`, `core/fitting/grouped_time_domain.py`,
`core/fourier/grouped.py`, `core/fourier/spectrum.py`,
`core/maxent/engine.py`, `core/utils/`, `gui/panels/plot_panel.py`,
`core/simulate_presets.py`, `docs/screenshots/data/archetypes.py`.

Verification: equality tests for the new shared helpers against both removed
copies' behaviour (including the F2 lookup order: dataset metadata before
run metadata); screenshot suite byte-stable (guards F12); corpus gate.

Done means: no duplicate definitions remain (grep-clean), validate green,
screenshots unchanged.

## Phase 2 — calibration promote family + count-fit surface (F5, F7, N3, F6, F8-part, NEW-R1)

Scope: the four promote paths sharing the deadtime pattern, the deadtime
chokepoint unification, and the count-fit-side F8/NEW-R1 items (kept here so
no other phase touches the fit-window files).

| Item | Change |
|---|---|
| F6 | `promote_deadtime_to_grouping` accepts per-detector value lists; `_on_maxent_apply_deadtime` routed through it (label `"maxent_fit"` kept; gains before/after + re-reduce message). Pin current MaxEnt-apply key writes in a test first |
| F7 | `promote_alpha_to_grouping` (writes `alpha`, `alpha_error`, `alpha_method="count_fit"`, reference-run provenance; suggest-only, before/after); GUI button beside the deadtime promote |
| F5 | `promote_t0_to_grouping` (µs offset → nearest `t0_bin` via bin width; sub-bin residual disclosed; per-group fitted value applied run-level with explicit note; provenance keys) |
| N3 | `promote_background_to_grouping` (fitted count background → `background_mode="fixed"` per-group values + provenance); count-fit informational note when grouping `background_correction` is active |
| F8-part | Count-fit exclude control relabelled "Skip window (µs)" |
| NEW-R1 | Count-fit exclude window persisted to the project schema + round-trip test |

Touch list: `core/transform/deadtime.py`, `core/transform/asymmetry.py` (or
a shared `core/transform/promote.py` if the four functions want one home —
implementer's call, study has no preference), `core/transform/t0.py`,
`core/transform/background.py`, `core/project/schema.py` (append-only),
`gui/panels/fit_panel.py`, `gui/windows/multi_group_fit_window.py`,
`gui/mainwindow.py` (promote handlers + MaxEnt deadtime handler).

Verification: per-promote unit tests (suggest-only: grouping untouched until
apply; before/after correctness; provenance keys; t0 rounding residual);
F6 behaviour-pinning test precedes the reroute; corpus gate (promotes are
suggest-only so reduction output must be unchanged until applied);
GUI smoke for the new buttons.

Done means: all four calibrations (deadtime, α, t0, background) reach the
grouping through suggest-only promote actions with before/after and
provenance; one deadtime write chokepoint; count-fit exclude window
round-trips.

Wave B interaction: **must merge before fit-workflow-diagnostics starts**
(shares `fit_panel.py`). MINOS-on-α and the fgAll-Poisson unification stay
in that project — not scheduled here.

## Phase 3 — frequency-panel UX (F4, F3-hint, F8-part)

Scope: the Fourier and MaxEnt panel surfaces only.

| Item | Change |
|---|---|
| F4 | Two independent diamag checkboxes replaced by one three-way control (*Leave / Fit & subtract / Exclude band*); <5 G fit-failure fallback surfaced in status text; legacy projects with both keys set load as "Fit & subtract" |
| F3-hint | Read-only status line on the Fourier panel showing the inherited grouping background state (mode or "off") |
| F8-part | MaxEnt exclude controls relabelled "De-weight window (µs)" |

Touch list: `gui/panels/fourier_panel.py`, `gui/panels/maxent_panel.py`,
`core/fourier/spectrum.py` (config plumbing only, if the three-way control
wants a single enum — schema keys `remove_diamag`/`diamag_exclusion` remain
readable either way).

Verification: state round-trip tests for the three-way control including
legacy both-set projects; spectrum outputs unchanged for each single-path
configuration (pin before refactor); GUI smoke.

Done means: diamag paths mutually exclusive in the UI with the fallback
disclosed; background inheritance visible; the two time-window exclusions no
longer share a label.

## Phase 4 — documentation package (F1, F3-ladder, F8-glossary, F11, F13, N5, N6, N4-note, F9-note, F10-doc)

Scope: docs only; no src/ changes. Written in the user guide's
when-to-use register.

| Item | Deliverable |
|---|---|
| F3 | Background-ladder page: which stage removes what (grouping modes → zero-frequency feature; σ-clip → spectral floor; SpecBG → ZF/LF central peak; count-fit nuisance on raw counts), when to enable each |
| F8 | Five-row exclusion glossary, cross-linked from each panel's docs |
| F4 | Diamag when-to-use (subtract preferred for correlation/A_μ; band as robust fallback) |
| F1 | Two-way cross-references `PowerLawQuadBG` ↔ `PowerLaw ⊕ Constant` (component docs, picker tooltip, parameter-name mapping `BG` ↔ `c_2`) |
| F11 | `alc_mode` → `parameter_trending` reverse link + complementarity paragraph |
| F13 | Spectral-estimator triad when-to-use (FFT / MaxEnt / Burg) cross-linking the three pages |
| N5 | "Assessing a fit" hub: χ² band (`fit_quality`) → `result_summary` verdicts → pull diagnostic |
| N6 | By-design note on the two phase stores + cross-refs; trigger recorded for a future pull action |
| F10 | "Trending data model" section: everything trendable is a `FitSeries`; the window is a transient view |
| N4 | Two-line registry-naming note in ARCHITECTURE.md + annotation in the python-user-functions brief |
| F9 | Annotation in the run-arithmetic brief (co-subtract builds on `subtract_scaled_counts` / reference-run chokepoint) |
| Promote docs | `alpha_calibration.rst` gains count-fit α as the fourth route; t0/background promote paragraphs beside it |

Touch list: `docs/user_guide/**`, `docs/ARCHITECTURE.md`,
`docs/porting/wimda-parity-gap/projects/python-user-functions.md` and
`projects/run-arithmetic.md`.

Verification: `python tools/harness.py docs`; link check via the docs build.

Done means: every DOCUMENT verdict in the study has its deliverable merged;
the two Wave B briefs carry their annotations.

Note: the promote-path user docs reference Phase 2's features — if run in
parallel, land the Phase 4 PR after Phase 2 merges or mark those paragraphs
against the Phase 2 PR.

## Phase 5 — trending decorations into FitSeries (F10-unify)

Scope: the selective F10 unification (the full stateless-window refactor
stays deferred with its trigger recorded in the study).

| Item | Change |
|---|---|
| F10 | `GlobalParameterFitWindow` decorations (local model fits, plot annotations) move from `global_parameter_fit_window_state` into `FitSeries.extra` keyed by batch id; restored on window show; legacy key still read on load |

Touch list: `gui/windows/global_parameter_fit_window.py`,
`gui/mainwindow.py` (trend sections), `core/representation/series.py`,
`core/project/schema.py` (append-only).

Verification: round-trip test (decorations survive save/reload attached to
their batch); re-run-fit test (decorations follow the replaced series or are
cleanly dropped — implementer documents which); legacy-state migration test.

Done means: window decorations can no longer orphan; one serialized home for
trend-attached state.

Sequencing: after Phases 2 and 3 merge (shares `mainwindow.py`); can run
parallel to nothing else.

## Wave B interactions (summary)

| Pending project | Interaction |
|---|---|
| fit-workflow-diagnostics | Phase 2 must merge first (shared `fit_panel.py`). MINOS-on-α + fgAll-Poisson belong there, not here |
| run-arithmetic | F9 constraint travels via the brief annotation (Phase 4); no reconciliation phase touches `core/data` |
| python-user-functions | N4 naming formalisation lands with its registration API; brief annotated in Phase 4 |
| spectral-moments | Consumes the trend target; Phase 4's F10/N5 docs and Phase 5's decoration home land first, which is the order the wave plan already implies |

## Effort summary

| Phase | Size | Parallel slot |
|---|---|---|
| 1 — mechanical UNIFYs | S (part-session) | solo, first |
| 2 — promote family | L (one long session) | ∥ 3, 4 |
| 3 — frequency-panel UX | M | ∥ 2, 4 |
| 4 — docs package | M | ∥ 2, 3 (land after 2) |
| 5 — trending decorations | M | after 2 + 3 |
