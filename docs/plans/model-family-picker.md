# Model family picker for both fit wizards

Status: planned 2026-09-30 on `feat/model-family-picker`. The PR is built in
phased subagent steps, with a lead review gate after each. Mockup (Design
canvas; the Scope board and the Model families board):
<https://claude.ai/artifact/3rz871CWotdJ2nmJcVHwCM>. This is the first piece
of the wizard UX work that started with
[BenHuddart/asymmetry#341](https://github.com/BenHuddart/asymmetry/pull/341).
The global wizard's stepper and Compare workspace follow in their own PR,
then the single wizard's comparison view.

## Problem

Ben's report: choosing which model families a wizard screens is hard to
discover, and irrelevant functions slow the wizard down a lot. Verified on
2026-09-30 by reading the code and rendering both wizards offscreen.

1. **One control mixes two questions.** `WizardScopeSelector`
   (`gui/widgets/wizard_scope_selector.py`) offers a single preset combo
   (`WizardScopePreset`, `core/fitting/wizard_scope.py:111`). Each preset
   fixes *both* a field geometry and a set of physics classes
   (`PRESET_QUERIES`, `:236`), so you cannot ask for "LF, dynamics and
   muonium", and the geometry the runs record is overridden rather than used.
2. **Unrecorded geometry means everything.** With no `field_direction` or
   `field_state` (PSI ROOT files, synthetic series), Auto screens every
   geometry and class: 37 components, of which 9 are slow. That is the slow
   path, and it is the usual one. The note under the combo says why ("field
   geometry not recorded — screening all component families"), but offers no
   way to fix it.
3. **Components appear by registry name.** The tree lists `StretchedExponential`,
   `RischKehr`, `FmuF_General`, …, with only a cheap/moderate/expensive
   column. `ComponentDefinition` (`core/fitting/composite.py:159`) has a
   `description`, `latex_equation`, `field_geometries`, `physics_classes` and
   `cost`, but the selector shows none of them. Exclusion reasons appear only
   as tooltips.
4. **The single wizard hides it.** On the Welcome page the selector sits inside
   the collapsed "Guide the analysis (optional)" section
   (`fit_wizard_window.py:307`).
5. **The resolver contract is an untyped dict**
   (`wizard_scope_selector.py` module docstring), which the Lean Code Rules ask
   us to replace with a typed value at the boundary.

## Settled design (decision log)

Decisions taken with Ben on 2026-09-30, from the mockup and four questions.

- **D1: labels live in the registry.** `ComponentDefinition` gains `label` (a
  short readable name, e.g. "Dynamic Gaussian KT") and `use_when` (one line,
  e.g. "Fluctuating Gaussian fields (strong collision)"). All 37 built-in time
  components are seeded from the mockup drafts (`LABELS` in the session
  scratchpad `family_data.py`, reproduced in the Phase 1 brief). Ben reviews
  the wording in the PR diff. A user function's `label` is its name and its
  `use_when` its description. Picker, fit panel and docs can all reuse these.
- **D2: physics chips combine.** A scope becomes a set of physics classes (an
  empty set means all), plus include/exclude overrides and D3's slow flag. It
  no longer carries a geometry: geometry always comes from the runs (D4).
  `GENERIC_RELAXATION` and `BACKGROUND` are always in scope, as today. Saved
  scopes are migrated in the project schema. `WizardScopePreset` and
  `PRESET_QUERIES` are deleted from runtime code; the migration holds the old
  preset → classes table.
- **D3: "slow" is the registry's `expensive` tier for now.** Pills carry a
  **slow** tag, and the footer names the slow models that are included. A
  **Leave out slow models** switch sets `skip_slow` on the scope, which
  resolves to the existing `ScopeQuery.max_cost = MODERATE` cap. A follow-up
  replaces the tier with fit times the wizard measures. A per-call benchmark
  (2026-09-30) showed the tier is a poor guide: dynamic Gaussian KT costs about
  0.1–0.3 ms per call on a 20k-point grid, while F–μ–F + third F costs about
  17 ms.
- **D4: the direction answer is saved on runs that record none.** The picker
  asks "Field direction: Zero field / Longitudinal / Transverse / Not
  recorded". The answer writes `metadata["field_direction"]` (using the
  `field_direction_from_text` vocabulary: "Zero field", "Longitudinal",
  "Transverse") **only** on runs whose file records no direction, and marks
  it `metadata["field_direction_source"] = "user"`. It persists through the
  existing per-run `metadata_overrides` in the project. It never overrides a
  recorded direction. On a series whose runs all record a direction, the row
  shows it read-only ("Recorded: Longitudinal"). Side effect: an LF answer
  lets the wizard pin B_L from the recorded field (`fit_wizard.py`,
  `_pinned_longitudinal_field`), which removes the noise-seed-dependent Δ
  role on the docs' Ag LF series
  ([BenHuddart/asymmetry#342](https://github.com/BenHuddart/asymmetry/pull/342)).
  Choosing **Not recorded** again removes only the user-set value.
- **D5: one widget, both wizards.** A new `ModelFamilyPicker`
  (`gui/widgets/model_family_picker.py`) replaces `WizardScopeSelector`,
  which is deleted along with its tests. From top to bottom:
  - the direction row, with a search box on the right;
  - "Looking for (optional)" physics chips;
  - family cards (Relaxation, Kubo–Toyabe, Oscillation, Muonium, Nuclear
    dipolar, Background, plus "Your functions" when user functions exist).
    Each card has a tri-state family box, an "n of m" count, a one-line blurb
    and one toggle pill per applicable component. Components that don't apply
    are listed under the card as "Not LF models: …" rather than hidden;
  - a details panel for the last-clicked pill, showing label, `use_when`,
    description, geometries, cost, and why it is in or out;
  - a footer line: "Will screen N of M models · K slow: …", with the Leave
    out slow models switch.

  The widget holds no physics logic: it renders a typed `ScopeView` (D6) and
  emits the edited scope plus any direction answer.
- **D6: a typed scope view from core.** `describe_scope(datasets, scope) ->
  ScopeView` (`core/fitting/wizard_scope.py`) replaces the dict resolver. It
  returns frozen dataclasses:
  - the runs' recorded geometry summary (per-geometry counts, how many runs
    record none, and whether the direction row is editable);
  - families in display order, each listing components with `name`, `label`,
    `use_when`, `description`, `geometries`, `slow`, `included` and `reason`;
  - the notes (fluorine hint, TF muonium regime);
  - the `(included, slow_included)` counts.

  `estimate_screening_cost`'s display-only weighted fit count is removed. The
  footer shows model counts, which are honest.
- **D7: placement.**
  - **Global wizard:** the picker replaces the Scope section of the Setup page
    (`global_fit_wizard_window.py` `_build_setup_page`). The Setup page layout
    is otherwise unchanged; the stepper is the next PR.
  - **Single wizard:** the picker is shown expanded on the Welcome page, above
    **Analyze**, and leaves the collapsed guidance section (which keeps the peak
    seeding). After a run it still moves into the result trail's first step, as
    today.
  - **Both:** editing the scope after a result marks it stale, as today.
- **D8: the CLI keeps `--scope <name>` as physics shortcuts.** The names
  (`auto`, `zf-static-magnetism`, …, `all`) map to physics-class sets through a
  table in `core/workflow/screen.py`. Geometry now always comes from the run and
  the existing `--geometry` override, so `--scope lf-dynamics` on a ZF run now
  screens ZF dynamics instead of forcing LF. The CLI help and `docs/cli` note
  this. SKILL.md is not edited.

## Code map (verify line numbers before editing)

- `core/fitting/composite.py:159`: `ComponentDefinition`, which gains `label`
  and `use_when`. The built-ins are the literal `COMPONENTS` dict
  (`composite.py:536`; 37 `domain == "time"` entries among 42, plus
  `parameter_models.py`). User functions build their definitions in
  `core/fitting/user_functions.py`.
- `core/fitting/wizard_scope.py`:
  - `WizardScopePreset` (`:111`), `PRESET_QUERIES` (`:236`), `WizardScope`
    (`:291`), `infer_auto_query` (`:364`), `resolve_scope` (`:476`),
    `resolve_scope_for_dataset(s)` (`:588`, `:598`),
    `estimate_screening_cost` (`:661`).
  - `_PRESET_NOTES` goes; the geometry-derived note stays.
- `core/fitting/fit_wizard.py`: scope use at `:2502`/`:2594`; canonical
  signature string with `scope.to_payload()` at `:3685`.
- `core/fitting/global_fit_wizard.py`: `scope` parameters (`:1220` onward).
- `core/workflow/screen.py:48-235`: `SCOPE_PRESETS` and `screen_run`.
  `cli/commands/wizard.py:43-98`: the `--scope` flag.
- `core/project/schema.py`: `CURRENT_SCHEMA_VERSION = 22` becomes 23. A new
  `_migrate_v22_to_v23` rewrites every persisted scope payload
  (`{"version": 1, "preset", "include", "exclude"}` →
  `{"version": 2, "physics": [...], "include", "exclude", "skip_slow": false}`)
  inside the fit-panel wizard caches: the single-run cache and the global
  cache signatures' `"scope"`, and the single-run recommendation's canonical
  signature string. The implementer locates each site from the
  `wizard_cache.py` serialisers and `_restore_wizard_cache_store` in
  `gui/panels/fit/global_tab.py:2206` and `single_tab.py`.
  - Preset → physics table:
    - `auto`, `all`: empty (all)
    - `zf-static-magnetism`: {magnetism}
    - `tf-knight-precession`: {magnetism}
    - `tf-superconductor`: {superconductivity, magnetism}
    - `lf-dynamics`: {dynamics, magnetism}
    - `fluoride-fmuf`: {molecular}
    - `muonium-radical`: {muonium}
- `gui/mainwindow.py:16545` (save) and `:17129` (load): `metadata_overrides`
  gains `field_direction` and `field_direction_source`, only when the source
  is `"user"`.
- `gui/widgets/wizard_scope_selector.py` and
  `tests/gui/test_wizard_scope_selector.py`: deleted.
- `gui/windows/global_fit_wizard_window.py`:
  - `_build_setup_page` Scope section, `_resolve_scope`, `_on_scope_changed`,
    `set_cached_recommendation` scope restore (`:1331`),
    `_analysis_signature` (`:1379`).
  - This file is also touched by #341; merge `main` in once #341 lands (never
    rebase a pushed branch).
- `gui/windows/fit_wizard_window.py`: welcome page (`:281-327`), `_scope_panel`
  (`:394`), the trail re-parenting (`:922`, `:1526`), scope restore (`:774`),
  signature (`:798`).
- Docs:
  - `docs/reference/fit_wizard.rst` and `docs/reference/global_fit_wizard.rst`
    (their Scope sections, quoting UI strings verbatim).
  - Screenshot scenarios `global_fit_wizard_setup`, plus a new
    `fit_wizard_welcome` showing the picker.
  - CHANGELOG `[Unreleased]`.

## Phases

Each phase is one subagent brief. It carries the AGENTS.md Lean Code Rules
verbatim, runs focused tests plus `--tier fast`, and ends with a lead review
(diff read, offscreen render for GUI phases) before the next phase starts.
Phases run sequentially in the main checkout.

1. **Core.**
   - D1 registry fields (all 37 labels and `use_when` lines), D2 `WizardScope`
     v2 (with `from_payload` parsing v2 only), D3 `skip_slow` → `max_cost`.
   - D6 `describe_scope` and `ScopeView`, D8 CLI table, and the schema v23
     migration with its round-trip test on a v22 fixture.
   - Delete `WizardScopePreset`, `PRESET_QUERIES`, `_PRESET_NOTES` and
     `estimate_screening_cost`.
   - Core tests under `tests/core/` and `tests/project/`.
2. **Run direction (D4).**
   - A core helper `set_user_field_direction(datasets, geometry | None)` that
     fills only runs recording none and clears only user-set values.
   - `metadata_overrides` save and load for the two keys.
   - Tests: the helper, a project save/reopen round trip, and the wizard
     pinning B_L on an LF-answered series.
3. **Widget (D5).** `ModelFamilyPicker` rendering a `ScopeView`: search,
   family boxes, pills, details panel, footer, slow switch. It has to fit the
   1180×740 wizard window (cards flow in a `FlowLayout`, and the details panel
   can collapse below 1000 px wide). Delete `WizardScopeSelector` and its
   tests; add `tests/gui/test_model_family_picker.py`.
4. **Wiring (D7).** Both windows use the picker. The direction answer reaches
   the datasets and marks the project dirty. Stale marking and cache
   signatures are carried over. Update the existing wizard window tests.
5. **Docs.** Both reference pages, the screenshot scenarios, CHANGELOG, and a
   `docs/cli` note for D8.
6. **Validate and PR.** `harness validate`, `gui-smoke`, `docs`, and the
   slow-marked wizard GUI files run focused.

## Follow-ups (not this PR)

- Measured fit times replace the cost tier behind the slow tag (D3).
- Global wizard stepper and Compare workspace (mockup boards 1–3).
- Single wizard comparison view.
- The dynamic Gaussian KT seed in a field (flagged by #342).
