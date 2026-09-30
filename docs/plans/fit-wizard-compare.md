# Fit Wizard: a Compare section under the answer card

Status: implemented 2026-09-30 on `feat/fit-wizard-compare`, PR to follow.
The follow-up that [global-wizard-stepper.md](global-wizard-stepper.md) D8/D9
promised: the single-run Fit Wizard adopts the `ModelComparePanel` the Global
Fit Wizard's Compare step introduced in #344, with N = 1 run.

## Problem

The Fit Wizard's result page offers two ways to look past the recommendation,
and neither compares anything:

1. **Alternative chips.** The answer card shows up to three chips
   ("*title* · +1.3"). A chip swaps the card's overlay and what **Apply this
   fit** applies. There is no second curve to set against the first, no
   residuals, no parameter values, and no evidence weight: only a raw Δ.
2. **The compare table.** The trail's "candidates" step expands to a table
   of absolute AIC/AICc/BIC, gate and χ²ᵣ for every candidate. It is the
   whole pool, but it is a table of numbers with no picture.

The Global Fit Wizard's Compare step already answers this: a leaderboard with
Δ bars and evidence weights, pick A and pin B, an overlay with A solid and B
dashed over normalised-residual strips, and an A-vs-B parameter table with
flags. Its panel and its core were written for N runs so the single wizard
could adopt them.

## Settled design (decision log)

Ben decided on 2026-09-30 to keep the answer card as the verdict and replace
its chips with the shared panel. D2–D5 are lead decisions within that answer,
recorded here so review can overturn them.

- **D1: the card stays and its alternative chips go.**
  - The card keeps the headline, the confidence line and chip, the overlay
    with its residuals toggle, and **Apply this fit**.
  - A **Compare candidates** `PanelSection` (collapsible, expanded) sits
    between the card and the decision trail. It holds a `ModelComparePanel`.
  - Picking A in the panel *is* the card's selection: the card redraws A,
    and **Apply this fit** applies A.
  - One fact, one owner. The panel owns A. The card holds no selection of its
    own: it is built with a callable that returns the selected key (the
    panel's `a_key`), and redraws when the panel emits `a_changed`. There is
    no `set_selected_key` to keep in step, and no window-side `_selected_key`.
  - The panel's footer button reads **Apply A to the fit panel** here and
    applies directly: the single wizard has no Apply step. The panel takes
    its footer text as a constructor argument (default **Continue with A →**
    for the global wizard). The card's **Apply this fit** and the panel's
    button reach one window slot, which applies the given key.
- **D2: the single-run adapter.**
  - `summarise_single_candidates(recommendation, dataset, metric)` in
    `core/fitting/model_comparison.py` returns one `CandidateSummary` per
    candidate row, ranked by `metric`, with one `RunFit` each, and Δ and
    weights taken across the whole list.
  - Every candidate is listed: the null baselines and disqualified rows too,
    marked "(baseline)" and "(disqualified)" in the title as in the Details
    table.
  - Parameters are neither Global nor Local. A new `ParameterRole.FITTED`
    ("Fitted") covers a free parameter of a single-run fit; a constrained
    parameter is `FIXED`.
  - `CandidateSummary`'s `global_names` / `local_names` / `fixed_names`
    fields become one `names(role)` method over its parameter rows, so the
    names and the rows can never disagree.
  - The panel handles N = 1: no role chips, a bare symbol in the table (no
    "· fitted" suffix), a single value per side, no "at *run*" in flag lines,
    no run named in the gate summary, "Candidates" as the board caption, and
    the trend plot hidden (there is no series to trend along).
  - `summarise_candidates`' internals are shared, not mirrored: one scoring
    helper gives Δ and weights for both adapters, and the parameter-row and
    gate-summary helpers take a role and per-run reasons instead of a global
    assessment.
- **D3: curves on demand.**
  - A single-run build carries dense curves only for the rows it exposes
    (`recommended_key`, `comparable_keys`); every other row is bare
    (`CandidateAssessment`'s dense-curve contract).
  - So a `RunFit` may not have its curves yet. `RunFit.curves` is a
    `RunCurves | None` (the fit curve and its normalised residuals, built
    together), and `CandidateSummary.curves_built` says whether every run has
    them. `None` is the honest "not built yet", not a failure.
  - `SeriesFitCanvas.set_curves` draws a pending A or B without its line or
    residual strip and emits `curves_required(key)`; the panel forwards it.
  - The window routes the card's and the panel's `curves_required` into the
    one existing materialise path (`assessment_with_curves` on the window's
    `TaskRunner`). When a row's curves land, it re-summarises and hands the
    panel the same ranking through `refresh_curves`, which redraws A and B
    without rebuilding the board.
  - Curves are never built on the GUI thread.
- **D4: A and B defaults.** A is the recommended key, else the best row. B
  starts empty. A re-rank keeps A.
- **D5: the Details table stays.** The trail's "candidates" step keeps the
  raw compare table and its residual-check text. Selecting a row there makes
  it A. Only the chips are removed.

## Code map (verify before editing)

- **Core.** `core/fitting/model_comparison.py`:
  - `RunFit` at `:192`, `CandidateSummary` at `:204`;
  - `summarise_candidates` at `:229`, `_candidate_summary` at `:252`,
    `_parameter_row` at `:310`, `_gate_summary` at `:333`.
- **Single-run core.** `core/fitting/fit_wizard.py`:
  - `CandidateAssessment` at `:285` (dense-curve contract in its docstring);
  - `FitWizardRecommendation.sorted_assessments`;
  - `assessment_with_curves` at `:4476`.
- **Widgets.**
  - `gui/widgets/model_compare_panel.py`: `split_text` `:106`, `flag_lines`
    `:113`, `CompareRow` `:190`, `ModelComparePanel` `:336`, `_show_pair`
    `:558`, `_fill_table` `:597`, `_draw_trend` `:648`;
  - `gui/widgets/series_fit_canvas.py`: `set_curves` `:127`, `_redraw`
    `:146`, `_draw_strip` `:267`;
  - `gui/widgets/wizard_answer_card.py`: the alternatives strip at
    `:194-203` and `:336-452`, `set_selected_key` at `:327`.
- **Window.** `gui/windows/fit_wizard_window.py`:
  - `_build_result_page` at `:363`, `_populate_result_state` at `:846`;
  - compare table at `:1397-1505`, `_on_metric_changed` at `:1457`;
  - curves on demand at `:1511-1584`; apply at `:1615`.
- **Tests.** `tests/core/test_model_comparison.py`,
  `tests/gui/test_model_compare_panel.py`,
  `tests/gui/test_series_fit_canvas.py`,
  `tests/gui/test_wizard_answer_card.py`,
  `tests/gui/test_fit_wizard_window.py`.
- **Docs.** `docs/reference/fit_wizard.rst` § result; scenario
  `docs/screenshots/scenarios/fit_wizard_result.py`.

## Phases

One subagent, committing in slices:

1. **Plan.** This file and the `docs/PLANS.md` entry.
2. **Core.** `RunCurves`, `ParameterRole.FITTED`, `names(role)`,
   `curves_built`, the shared scoring/row/gate helpers and
   `summarise_single_candidates`, with tests.
3. **Widgets.** The panel's N = 1 handling, its footer text,
   `curves_required` and `refresh_curves`; the canvas's pending draw; the
   card without chips and with its selection read through a callable.
4. **Window.** The Compare candidates section, one selection, one apply
   slot, curves on demand for the panel. Window tests rewritten where they
   pinned the chips.
5. **Docs.** The reference page, the result scenario, CHANGELOG.
