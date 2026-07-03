# Shared-Foundations Audit — Execution Plan

**Branch:** `audit/shared-foundations` (worktree at `.claude/worktrees/audit-shared-foundations`, branched from `origin/main` @ `3d2359a`)
**Deliverable:** a single PR onto `main`.
**Orchestrator:** Opus session running in this worktree. Implementation by subagents — **Sonnet by default**, escalating to **Opus** where marked. Milestone reviews by Opus (Fable for the final gate if warranted).

## Goals (agreed with Ben, 2026-07-03)

1. **Consistency:** similar functionality and GUI elements across representations
   (time, frequency/FFT, MaxEnt, integral/ALC, trend/fit-series) and workflows
   (single fit, global fit, wizards, batch) must share a common base, so the app
   behaves uniformly for users and fixes land once for maintainers.
2. **Navigability:** the repository must be clearly laid out and indexed so
   future agent-driven development is efficient.

## Decisions of record

| Decision | Choice |
|---|---|
| Behavior policy | **Converge to best variant.** Each consolidated component adopts the most-featured/correct behavior everywhere; every deliberate user-visible change is logged in `docs/audit/shared-foundations/BEHAVIOR-CHANGES.md` and listed in the PR description. |
| God-file splits | **Split `fit_panel.py` and unify the two wizard windows.** `plot_panel.py` and `mainwindow.py` are explicitly out of scope for decomposition (record as follow-ups), though they DO migrate onto the new shared widgets. |
| Indexing scope | Test-suite reorg into mirrored subdirectories; internal roadmap docstrings + doc fixes; harness enforcement rules. Examples/porting-index polish is out of scope. |
| PR strategy | Single PR, commits per phase, no intermediate pushes. Ben has pre-authorized opening the one PR after the final review gate; request his live-GUI check alongside it. |
| Open PRs | None exist on the remote (verified 2026-07-03); no conflict management needed. |
| `feat/fit-wizard-scope` | 11 unmerged local commits (Scope tabs in both wizards, physics tags, tiered screening, FFT peak editing) still need development work. **Audit proceeds first; that branch is re-ported onto the restructured wizard layer after this PR merges.** Phase 3's base design must not preclude it (see Phase 3). |

## Ground rules (apply to every phase)

- Work only in this worktree. Use `.venv/bin/python` here (already provisioned;
  `tools/harness.py` re-execs into it automatically). Never use the hub checkout's venv.
- Engineering invariants in `AGENTS.md` are binding. In particular: core stays
  GUI-free; never run long work on the GUI thread; never connect worker signals
  to bare lambdas/partials that touch widgets (no receiver QObject ⇒ slot runs on
  the worker thread); hold strong references to live threads.
- Every phase ends with `python tools/harness.py validate` green (standard tier,
  ~1m40s) and a commit. GUI-touching phases also run `python tools/harness.py gui-smoke`.
- Subagents get **narrow, file-scoped tasks** with the relevant plan section
  pasted into their prompt. They must not expand scope; anything discovered
  out-of-scope goes into `docs/audit/shared-foundations/FOLLOW-UPS.md`.
- Line numbers below were sampled on 2026-07-03 and will drift — treat them as
  landmarks, re-locate with grep before editing.
- When consolidation changes behavior at a call site, add/adjust a test beside
  the behavior AND append an entry to `BEHAVIOR-CHANGES.md` (what changed, where,
  which variant won, why).

---

## Phase 0 — Baseline & characterization tests (Sonnet)

**Objective:** pin current behavior before anything moves.

1. Run `python tools/harness.py validate` and `gui-smoke`; record the pass counts
   in `docs/audit/shared-foundations/BASELINE.md`.
2. Write characterization tests (marked `gui` where they need Qt) for the
   behaviors about to be consolidated, where coverage doesn't already exist:
   - `_FloatLimitField` variants: the `plot_panel.py` (~154–200) version vs the
     `fit_panel.py` (~619–733) version — commit-on-Return, clamping, precision,
     empty-input handling. Test the *observable contract*, not the class names,
     so the tests survive consolidation.
   - TSV export output of `fit_parameters_panel.py` (~5156–5945) and
     `global_parameter_fit_window.py` — golden-string assertions on headers,
     row formatting, value formatting.
   - Fit-range commit round-trip in SingleFitTab and GlobalFitTab (spinbox edit
     → engine fit range → display).
   - Wizard result caching: same-signature re-open serves cache; changed
     signature invalidates (both wizard windows).
3. Commit: `audit(phase0): baseline + characterization tests`.

**Gate:** validate green. No review needed.

---## Phase 1 — Shared widget & utility foundations (Sonnet, sequential subtasks)

**Objective:** one implementation each for the small duplicated building blocks;
migrate every call site. Run the subtasks **sequentially** — they touch
overlapping files (`plot_panel.py`, `fit_panel.py`, `alc_panel.py`).

**1a. Unified axis-limit field + axis controls.**
New `src/asymmetry/gui/widgets/axis_limits.py`:
- `FloatLimitField` — converged behavior: the fit_panel variant wins (clamping
  via `_clamp`, Return/Enter commit, `_commit` forcing) with parametrized width
  and decimals. Delete both old classes; migrate `plot_panel.py` (~458–623
  `_create_limit_controls`), `fit_panel.py`, `alc_panel.py` (imports the
  plot_panel one today, ~332–340).
- `AxisLimitControls` — the X/Y min/max row assembled once (log-scale and
  decimation slots optional), replacing the hand-rolled toolbars in
  `plot_panel.py` and `alc_panel.py`.

**1b. Matplotlib canvas factory.**
New `src/asymmetry/gui/widgets/mpl_canvas.py` with a
`create_canvas(tight_layout=True, toolbar=False)` helper (figure + canvas +
optional toolbar, centralized import handling). Migrate the four call sites:
`plot_panel.py` (~258–290), `fit_parameters_panel.py` (~665, lazy),
`global_parameter_fit_window.py` (~130–141), `alc_panel.py` (~25).
Preserve the lazy-creation pattern in `fit_parameters_panel`.

**1c. Export utilities.**
New `src/asymmetry/gui/utils/export.py`: shared TSV writer, GLE invocation
(subprocess wrapper), export-path dialog/caching helper. Migrate
`fit_parameters_panel.py` (`_export_tsv`/`_export_gle`, ~5156–5945),
`global_parameter_fit_window.py`, `plot_panel.py` payload exporters
(~5755–6500), `alc_panel.py` `export_current_plot` (~1552). Phase 0's golden
tests must keep passing; format changes only if converging a divergence
(log it in BEHAVIOR-CHANGES.md).

**1d. Progress/cancel + small shared helpers.**
- A `FitRunControls` helper (Stop button + status label wired to a TaskRunner
  cancel path) used by SingleFitTab, GlobalFitTab, and MaxEntPanel (~362–375).
- Move `_format_param_label` (duplicated in `fit_panel.py` and
  `fit_parameters_panel.py` ~137) into `gui/utils/formatting.py`.

**Commits:** one per subtask, `audit(phase1x): ...`.

**Gate — Review A (Opus):** review the Phase 1 diff for: behavior regressions at
migrated call sites, incomplete migrations (grep for leftover duplicate
classes/functions — there must be zero), API awkwardness in the new shared
modules that Phase 2/3 will build on. Fix findings before Phase 2.

---

## Phase 2 — `fit_panel.py` decomposition + `FitTabBase` (**Opus**)

**Objective:** split the 8,934-line `src/asymmetry/gui/panels/fit_panel.py` into
a package and extract the shared tab machinery. This is the highest-complexity
phase — Opus implements, with Sonnet permitted only for mechanical
move-and-fix-imports sweeps under Opus direction.

Target layout:

```
src/asymmetry/gui/panels/fit/
├── __init__.py          # re-exports FitPanel, SingleFitTab, GlobalFitTab, FitParameterTable
├── panel.py             # FitPanel container/dispatcher (~the current shared ~3.3k lines, trimmed)
├── tab_base.py          # FitTabBase + shared delegates + FitParameterTable
├── single_tab.py        # SingleFitTab (currently ~1987–3159)
├── global_tab.py        # GlobalFitTab (currently ~3160–7600)
└── seeding.py           # _seed_group_background_and_n0 etc. (~178–440)
```

Keep `src/asymmetry/gui/panels/fit_panel.py` as a thin deprecation shim
re-exporting from the package (tests and any project code import from it), and
add a follow-up note to retire it.

`FitTabBase` owns what the report found duplicated (~1,500 lines):
- model formula box (`_make_formula_box`, ~2055 vs ~3291),
- Edit Model dialog handler,
- fit-range field pair + `_on_fit_range_committed` routing (~2127 vs ~3341),
- parameter-table construction (the inline ~80-line setups at ~1900–1980 and
  ~3400–3500 become one `_build_parameter_table()` factory),
- Stop/result controls via Phase 1's `FitRunControls` (~2163 vs ~3545),
- wizard-window caching plumbing (`_fit_wizard_window`,
  `_cached_wizard_recommendation`, `_cached_wizard_signature`,
  `_cached_wizard_log_text`; ~2026 vs ~3281),
- shared fit-precondition validation (bounds, dataset compatibility) — today
  duplicated informally in both tabs.

Subclasses keep only what genuinely differs: fit execution
(`FitEngine.run` vs `fit_grouped_time_domain`/`fit_grouped_series`),
global/local/fixed parameter classification, grouped progress callbacks,
tab-specific signals.

**Method:** extract-and-verify in small steps — (1) mechanical file split with
zero logic change, validate; (2) base-class extraction one cluster at a time,
validate after each; (3) delete the duplicated originals. Never a big-bang
rewrite.

**Gate — Review B part 1 (Opus, fresh context):** adversarial review of the
decomposition: signal connection parity (every `connect` in the old file
accounted for), state initialization order, no widget construction moved off
the GUI thread, shim completeness. Run `gui-smoke` + full `validate`.

---

## Phase 3 — Wizard-window unification (Sonnet implements from an Opus design note)

**Objective:** one base for `fit_wizard_window.py` (937 lines) and
`global_fit_wizard_window.py` (1,553 lines).

1. **(Opus, small task)** Write `docs/audit/shared-foundations/wizard-base-design.md`:
   the `WizardWindowBase` contract — template methods (`_create_worker()`,
   `_populate_results()`), progress UI, result-cache signature strategy, and
   the thread-lifecycle decision. Strong preference: replace both manual
   `QThread` lifecycles (~96–99 and ~80–120) with `TaskRunner` from
   `gui/tasks.py`, matching how the rest of the app was migrated in PR #68.
   Cite the AGENTS.md worker-signal invariant explicitly.
   The design must anticipate the pending `feat/fit-wizard-scope` re-port:
   adding extra tabs (a Scope tab before analysis) and extra worker inputs must
   be possible in subclasses without touching the base. Skim that branch's
   diff (`git log main..feat/fit-wizard-scope` in the hub) before finalizing
   the contract.
2. **(Sonnet)** Implement `src/asymmetry/gui/windows/wizard_base.py` per the
   note; refactor both windows onto it; delete the duplicated lifecycle,
   progress, caching, and error-dialog code.
3. Phase 0 wizard-caching characterization tests must pass unchanged.

**Gate — Review B part 2 (Opus):** threading-focused review (cancellation
mid-analysis, window close during a run, cache invalidation), plus
`gui-smoke` + `validate`.

---

## Phase 4 — Test-suite reorganization (Sonnet, mechanical)

**Objective:** mirror `src/` structure under `tests/`.

1. `git mv` the ~217 flat `tests/test_*.py` into `tests/core/`, `tests/gui/`,
   `tests/io/`, `tests/project/`, `tests/tools/` (classify by what the file
   imports/marks; anything genuinely cross-cutting → `tests/integration/`).
   Keep root `tests/conftest.py` as-is (session QApplication + Qt-cleanup
   fixtures are global).
2. Update anything that references test paths: `tools/harness.py` tier/subset
   selection and sharding, `pyproject.toml` pytest config, CI workflow files,
   `docs/HARNESS.md`. Note: CI shards by marker/hash, not directory, so this
   should be config-light — verify rather than assume.
3. Document the convention (new `tests/README.md`: naming, placement, markers).
4. Verify: `validate` collects the same test count as Phase 0's BASELINE.md
   (± tests added by this audit). A dropped-collection regression here would be
   silent — the count check is mandatory.

---

## Phase 5 — Harness enforcement rules (Sonnet)

**Objective:** make the consolidation self-defending. Extend
`tools/harness.py structural` with checks + matching tests in
`tests/tools/test_harness.py`:

1. **No duplicate foundations:** outside `gui/widgets/`, forbid new definitions
   of limit-field classes and direct `FigureCanvasQTAgg(` construction outside
   `widgets/mpl_canvas.py` (small explicit allowlist if genuinely needed).
2. **No bespoke QThread lifecycles in gui/:** forbid `QThread(` construction
   outside `gui/tasks.py` (allowlist any justified survivors found in Phase 3).
3. **Test placement:** every `tests/**/test_*.py` must live in one of the
   sanctioned subdirectories.
4. Each rule fails with a message pointing at the shared module to use.

---

## Phase 6 — Docs, roadmaps & index refresh (Sonnet)

1. Internal navigation docstrings (~150–300 words at module/class top: structure
   map, key entry points, signal flow) for the remaining large files:
   `mainwindow.py` (13,032 lines), `plot_panel.py` (7,293),
   `fit_parameters_panel.py` (6,287), `data_browser.py` (4,366), and the new
   `panels/fit/` package modules.
2. Update `docs/ARCHITECTURE.md` to the post-refactor layout (fit package,
   widgets/utils additions, wizard base); fix the `dialogs/` vs `windows/`
   mismatch (§ around line 95); document `core/transform/` contents.
3. Mark `core/negmu/` and `core/maxent/` status in `docs/QUALITY.md`.
4. Update `AGENTS.md`: point "Repository Shape" at the new shared foundations
   ("new axis/canvas/export/progress UI must use `gui/widgets/`+`gui/utils/`"),
   test-placement convention, and add the harness rules to the invariants list.
5. Write `docs/audit/shared-foundations/FOLLOW-UPS.md` (seeded throughout):
   at minimum plot_panel/mainwindow decomposition, fit_panel shim retirement,
   and the `feat/fit-wizard-scope` re-port onto the new wizard base.

---

## Phase 7 — Final gate & PR assembly

1. `python tools/harness.py test --tier full`, `gui-smoke`, `docs` (build-only),
   `lint`, `structural` — all green.
2. **Final review (Fable if judged warranted by residual risk, else Opus, fresh
   context):** whole-PR review against the two goals; verify BEHAVIOR-CHANGES.md
   is complete by sampling the diff for unlogged user-visible changes.
3. Assemble the PR description: goals, phase summary, behavior-change list,
   follow-ups, validation evidence. Push the branch and open the single PR
   (pre-authorized). Ask Ben for a live-GUI pass (axis fields, exports, both
   wizards, both fit tabs, MaxEnt cancel) before merge.

---

## Model & review summary

| Phase | Implementer | Review gate |
|---|---|---|
| 0 Baseline + characterization | Sonnet | — |
| 1 Shared widgets/utilities | Sonnet (sequential subtasks) | **A: Opus** |
| 2 fit_panel split + FitTabBase | **Opus** (Sonnet for mechanical sweeps) | **B1: Opus, fresh context** |
| 3 Wizard unification | Sonnet, from Opus design note | **B2: Opus** |
| 4 Test reorg | Sonnet | count-parity check |
| 5 Harness rules | Sonnet | — |
| 6 Docs/roadmaps | Sonnet | — |
| 7 Final | — | **Fable/Opus, fresh context** |

## Source reports

The full duplication and layout assessments that produced this plan are in
`docs/audit/shared-foundations/reports/` (`duplication-map.md`,
`layout-assessment.md`). Consult them for file/line specifics before each phase.
