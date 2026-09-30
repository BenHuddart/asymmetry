# Global Fit Wizard: a stepper and a Compare workspace

Status: implemented 2026-09-30 on `feat/global-wizard-stepper`, PR to follow.
Built in phased subagent steps, with a lead review gate after each. Mockup (Design
canvas, boards 1–3): <https://claude.ai/artifact/3rz871CWotdJ2nmJcVHwCM>.
This is the second piece of the wizard UX work, after
[model-family-picker.md](model-family-picker.md) (#343) and the navigation
fixes in #341.

## Problem

Ben's report (2026-09-30): the Global Fit Wizard is hard to navigate. It is
not clear what each stage does, and comparing models is neither intuitive
nor interactive. #341 fixed getting stuck on the Result page. Verified by
reading the window and rendering it offscreen, the rest remains:

1. **Three pages and five real stages.** The window is Setup / Running /
   Result (`global_fit_wizard_window.py:131`). The work is really: scope →
   screen → pick a shortlist → optimise roles → (optimise phases) → apply.
   The Result page stacks all of the later stages: the answer card, the
   Transitions card, the screening shortlist, and a trail of four expandable
   tables. Nothing says what comes next. For example, after screening, Apply
   is greyed out with no hint that the next step is to tick rows and press
   **Optimize selected**.
2. **The comparison is table-shaped.**
   - Scores are shown as absolute AIC/AICc/BIC to three decimals, not as
     differences.
   - Most optimised rows are one template that differs only in its
     Global/Local split, and the split columns are truncated.
   - There are no residuals, the screening and optimised tables are not
     linked, and you cannot overlay two candidates.
   - `GlobalFitCompareDialog` shows the right idea (A solid / B dashed, a Δ
     statistics block, a >2σ parameter flag), but it is tied to
     `GlobalFitStudy` cross-group trend fits and never reachable from the
     wizard.

## Settled design (decision log)

Decisions taken with Ben on 2026-09-30 over the mockup and four questions.
D5–D11 are lead decisions made within those answers. They are recorded here
so review can overturn them.

- **D1: a stepper replaces the pages.** A `WizardStepper` (new,
  `gui/widgets/wizard_stepper.py`) runs across the top:
  **Scope → Screen → Compare → Phases → Apply**.
  - Each step shows a number or state mark, its title, and a one-line summary.
  - Its state is one of pending, current, done, stale, running or skipped.
  - Done and stale steps are clickable. A pending step becomes clickable once
    its inputs exist.
  - Each step has its own view in a `QStackedWidget`. The Setup, Running and
    Result pages, #341's **← Back to setup** and **View results →**, and the
    trail-hosted detail panels are deleted.
- **D2: progress runs inside the running step.**
  - One `RunProgress` block (the running header, the `DecisionTrail`
    placeholders, the collapsed **Live log**, and Cancel) is re-parented into
    the view of whichever step is running:
    - screening runs in Screen;
    - optimising the shortlist runs in Compare;
    - optimising phases runs in Phases.
  - While a run is going, the stepper marks that step running. You can visit
    other steps. Scope edits keep today's stale and orphan rules
    (`_mark_analysis_stale`).
- **D3: the Screen step is a family leaderboard.** One row per
  `sorted_prescreen_assessments()` entry. Each row has:
  - a shortlist checkbox;
  - the title;
  - a Δ(metric) bar from the best row;
  - one χ²ᵣ cell per run, from `fit_results_by_run[run].reduced_chi_squared`,
    coloured good/fair/poor (≤ 1.5 / ≤ 5 / > 5) with the number always
    printed;
  - an optimisation-status badge (Not optimised / Running / Optimised /
    Failed, from `optimization_status_for_key`).

  Clicking a row previews its per-run fits in an overlay. Prescreen rows carry
  dense curves. The footer CTA is **Optimise N families →**. The raw screening
  table (AIC/AICc/BIC/params) and the portfolio table move into a collapsed
  **Details** section.
- **D4: the Compare step is the comparison workspace.**
  - The leaderboard holds the optimised rows grouped by template. Each role
    split is a sub-row carrying:
    - Global/Local chips;
    - a Δ(metric) bar;
    - an evidence weight: an Akaike weight on the ranking metric, w ∝
      exp(−Δ/2) over the optimised rows;
    - a gate badge;
    - flags (D8).
  - Clicking a sub-row makes it **A**, and **Pin as B** pins another.
  - On the right:
    - an overlay with A solid and B dashed, in per-run colours;
    - one normalised-residual strip per run, for A and B;
    - an A-vs-B parameter table: globals as value(err); locals as per-run
      values; a >2σ flag on shared globals;
    - a trend plot of the local parameter picked in the table, A and B
      overlaid.
  - The raw optimised table and the role diagnostics (Global/Local score, Δ,
    TV, roughness, rationale) move into a collapsed **Details** section.
  - This replaces `WizardSeriesCard` in the wizard, the screening shortlist,
    the optimised-fits table and the alternative chips.
- **D5: Compare leads to Apply.** Compare's footer is **Continue with A →**,
  which opens the Apply step. Apply reviews what will be handed over:
  - the model;
  - the Global/Local/Fixed roles;
  - the starting values;
  - the series warnings.

  It then offers **Apply to the global fit tab**, which is today's
  `apply_assessment_requested` path. Choosing a model and committing it stay
  two distinct acts, and the role rationale has a home. When a partitioned
  answer is chosen in Phases, Apply shows the phases and **Apply phases**
  instead (today's `apply_phases_requested`).
- **D6: Phases is its own step, active only when a break is found.**
  - Before screening it reads "Not screened yet". With no `partition_path`,
    or with the path's selected row at 0 breaks, it is skipped ("No transition
    found").
  - Otherwise it holds today's `TransitionsCard` (the penalty path,
    per-phase optimise, phase summaries) and an overlay coloured by phase
    (`phase_color`).
  - Picking a phase shows that phase's assessment in the Compare panel's
    plots, read-only.
- **D7: shortlist pre-ticking.** Screen pre-ticks the rows within Δ ≤ 10 of
  the best, on the ranking metric (Burnham & Anderson's "essentially no
  support" threshold), capped at the best three to bound the coupled-fit
  cost. The user can tick any row.
- **D8: flags come from core.** `core/fitting/model_comparison.py` (new,
  Qt-free) owns every judgement the panel shows:
  - evidence weights: `information_weights(scores)`, and `aic_weights` in
    `experiment_design.py` is rebuilt on it rather than duplicated;
  - the shortlist rule (D7);
  - per-run normalised residuals, `(y − f(t))/σ` on the dataset grid, with
    the dense curve interpolated. Residuals are not persisted
    (`_serialize_fit_result` drops them), so they are always recomputed;
  - parameter flags:
    - **not finite**, or **at lower/upper bound** (a plain fact, since a rate
      at 0 is often physical: lead review, 2026-09-30);
    - **poorly determined**: error > |value|;
    - **differs by kσ**: a shared global, A vs B, with k ≥ 2;
  - gate reasons summarised from `run_diagnostics`.

  It exposes a neutral `CandidateSummary` (key, title, global/local/fixed
  names, metric, Δ, weight, gate, flags, per-run curves and residuals,
  parameter rows) built from a `GlobalCandidateAssessment`. The single-run
  wizard adapts its `CandidateAssessment` to the same summary in the
  follow-up PR, so the panel is written for N runs from the start (N = 1 for
  the single wizard).
- **D9: one panel widget for both wizards.** `ModelComparePanel`
  (`gui/widgets/model_compare_panel.py`) renders a list of `CandidateSummary`
  plus the datasets. It emits `a_changed(key)`, `b_changed(key | None)` and
  `continue_requested(key)`. It uses `create_canvas`, tokens, `FlowLayout`
  and `format_param_label`.
- **D10: stepper summaries, from the recommendation and the picker.**
  - Scope: "*Direction* · N models".
  - Screen: "*Leader* leads", or "Screening…" while running.
  - Compare: "N role splits optimised", or "Next: optimise the shortlist".
  - Phases: "k transitions · 21 ± 3 K", "No transition found", or "Not
    screened yet".
  - Apply: "Pick a model first", "Ready: *title*", or "Applied: *title*".
- **D11: the ranking metric stays on Scope.** It re-ranks in place, as
  today, and the Screen and Compare headers name it ("ΔAICc from best").

## Code map (verify before editing)

- **Window.** `gui/windows/global_fit_wizard_window.py`:
  - pages at `:131`, stack at `:446`, `_show_running` at `:938`,
    `_update_action_enablement` at `:960`, `_show_result_page` at `:1537`;
  - modes and progress at `:146-197`, `:1226`, `:1245`; worker at
    `:215-341`, `:1114`; merge at `:1157`, `:140`;
  - `_populate_from_recommendation` at `:1489`, series-card adapters at
    `:1758-1907`, transitions adapters at `:1577-1700`;
  - tables at `:665`, `:745`, `:2067`, `:2127`; roles at `:2251`; apply at
    `:2287`; portfolio at `:2045`;
  - #341 navigation at `:554`, `:634`.
- **Core.** `core/fitting/global_fit_wizard.py`:
  - `GlobalCandidateAssessment` at `:463`, `GlobalFitWizardRecommendation`
    at `:548`;
  - `sorted_prescreen_assessments` at `:653`, `sorted_optimized_assessments`
    at `:666`, `optimization_status_for_key` at `:676`;
  - prescreen curves at `:2573`, dense-curve contract at `:466`;
  - rerank at `:3629`, comparable keys at `:3722`, serialisation at `:4089`.
- **Reuse:**
  - `gui/widgets/transitions_card.py`;
  - `gui/widgets/decision_trail.py`;
  - `gui/widgets/mpl_canvas.py:56`;
  - `gui/widgets/flow_layout.py:43`;
  - `gui/widgets/panel_section.py:52`;
  - `gui/utils/phase_colors.py`;
  - `gui/windows/global_fit_window_helpers.py:128` (`format_value_with_error`);
  - `core/fitting/experiment_design.py:953` (`aic_weights`);
  - the stats-table and >2σ patterns in `global_fit_compare_dialog.py:361-466`.
- **Apply paths.** `gui/panels/fit/global_tab.py`:
  - `_apply_fit_wizard_assessment` at `:4621` indexes the curve dicts
    directly;
  - the phases path runs `:4726` → `mainwindow._on_apply_wizard_phases` →
    `:4741`.
  - These are unchanged.
- **Tests pinning the old layout:**
  - `tests/gui/test_global_fit_wizard_window.py` (about 15 layout tests)
  - `tests/gui/test_wizard_transitions_card.py` (17)
  - `tests/gui/test_global_wizard_overview_populated.py`
  - `tests/gui/test_wizard_series_card.py`: the widget stays, because the
    single wizard does not use it. If nothing else uses it once Phase 3
    lands, delete it and its tests.
- **Docs:**
  - `docs/reference/global_fit_wizard.rst` §§ "The guided journey",
    "Running", "Result", "Transitions";
  - scenarios `global_fit_wizard_{setup,running,result,transitions}.py`.

## Phases

Each phase is one subagent brief. It carries the AGENTS.md Lean Code Rules
verbatim, runs focused tests plus `--tier fast`, and ends with a lead review
(diff read, offscreen renders for GUI phases). Phases run sequentially in the
main checkout.

1. **Core comparison.** `core/fitting/model_comparison.py`:
   - `information_weights` (with `aic_weights` rebuilt on it);
   - the shortlist rule;
   - residuals;
   - parameter flags;
   - `CandidateSummary` built from a `GlobalCandidateAssessment` and its
     datasets.

   Unit tests in `tests/core/test_model_comparison.py`.
2. **Widgets.**
   - `WizardStepper`;
   - the `RunProgress` block (trail, log and cancel, extracted from the
     Running page);
   - `ModelComparePanel`.

   Widget tests and offscreen renders at 1180×740 and 1600×900.
3. **Window restructure.** Five step views, the stepper, progress inside
   steps, the Screen leaderboard, Compare, Phases, Apply. It deletes the old
   pages, #341's buttons, the tables and the trail panels. The window tests
   are rewritten.
4. **Docs.**
   - the reference page;
   - scenarios: `global_fit_wizard_running` becomes screening in progress;
     `result` becomes Compare with A and B; `transitions` becomes the Phases
     step; add `global_fit_wizard_screen`;
   - CHANGELOG.
5. **Validate and PR.**

## Follow-ups (not this PR)

- The single-run wizard adopts `ModelComparePanel` through a
  `CandidateSummary` adapter.
- Measured fit times behind the slow tag (from #343).
- The dynamic Gaussian KT seed in a field.
