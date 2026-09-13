# Fit tab: model row, parameter rail, results card

Status: in implementation from 2026-09-13 on `feat/fit-tab-refresh`, one PR
at the end; built phase by phase by subagents with a lead review gate after
every phase. Follows the Parameters panel revamp (`docs/plans/parameters-panel-cards.md`, PRs #316 / #318) and
reuses its widgets and idioms. Mockups (Claude Design canvas):
https://claude.ai/code/artifact/fcecdd37-e558-4b59-9199-61a7105e5b8f — page
"Recommended" is the proposal; page "Directions" holds the alternatives
explored for the two open questions.

## Problem

`SingleFitTab` and `GlobalFitTab` (`src/asymmetry/gui/panels/fit/`) predate
the BENCH chip / card grammar and were never brought up to it.

- **The inspector dock is ~300 px wide on a 13-inch display** (the
  Parameters panel now floors the deck at its own x-rail width; nothing on
  the Fit tab asks for more). The single-fit `FitParameterTable` has eight
  fixed-width columns — Name 92 · Value 88 · Fix 30 · Min 52 · Max 52 ·
  Batch 50 · Link 40 · Tie 40 ≈ 444 px — so it always shows a horizontal
  scrollbar and the Min/Max/Batch/Link/Tie columns are off-screen at rest.
  `_size_param_table_to_content` even reserves the scrollbar's height.
- **`More…` is a mystery-meat menu** (`single_tab.py:191-215`) holding three
  unrelated actions: `Drop background` (a model edit), `Send to Batch` and
  `Add to Series...` (hand-offs to another surface). None of the three is
  documented in `docs/` and nothing on the tab hints they exist. The Batch
  tab has no such menu; the Multi-Group panel's Single tab shows `Send to
  Batch` as a plain button instead, so the same action lives in two places
  under two treatments.
- **The Model section's button stack** (`Edit Function...` / `Fit Wizard...`
  / `More…`, one per line) spends three rows on three buttons because a
  grid once forced the dock width.
- **Results are prose in a box.** The single tab's `#resultBox` renders one
  green line plus a `χ²/ν = … · npar = … · ndof = … · good` sentence; the
  batch tab's is a `QTextEdit`. The Parameters panel now says the same thing
  with a verdict chip (`χ²ᵣ 0.979`) whose click opens `FitResultsWindow`.
  `Pull diagnostic…` and the carry-forward badge are two further loose
  elements orbiting the result.
- Small inconsistencies: `Fit Wizard...` vs `Global Wizard...`; the batch
  tab's `?` button that opens a `QMessageBox`; `Seeding:` combo and
  `Per-run seeds…` on separate rows; the Batch tab already uses one
  `Bounds` column (`"min, max"` text) where the Single tab uses two.

## Design (proposed)

Grammar borrowed from the Parameters panel: **chips are state** (checkable
`QPushButton` via `style_group_state_button`), **cards carry a header strip
with a tag and a verdict chip**, **rows wrap** (`FlowLayout`) instead of
widening the panel, **the full-width grid pops out** (`↗`) instead of
scrolling in place, and **segmented buttons** (`build_segmented_button_qss`)
are the light action style beside a `build_primary_button_qss` primary.

**Single tab, top to bottom, nothing scrolling at rest at 320 px.**

1. **Model** — `A(t):` + the `FormulaBox`, unchanged. Beneath it one
   wrapping row: `Edit…` · `Wizard…` (segmented). Today's `Drop background`
   action is retired: the user removes the term in the function editor
   (Ben, 2026-09-13). The explanation on its tooltip (a free background
   splits the fitted amplitude on a light-OFF A₀ run) moves to the docs.
2. **Fit range** — unchanged row (`FloatLimitField` pair, `≤ t ≤`, unit).
3. **Parameters rail** — the `PanelSection` title on the left; on the right
   three column-group chips **`Bounds` · `Links` · `Batch`** and a **`↗`**
   pop-out button. Chips toggle column groups of the same `FitParameterTable`:
   `Bounds` → the **Min** and **Max** columns (kept separate — Ben,
   2026-09-13 — at 6 characters each, which holds `-inf`, `1e6` and the
   `−∞`/`∞` glyphs); `Links` → `Link` + `Tie`; `Batch` → the read-only role
   column. Default: only `Bounds` on, giving Name 90 · Value 88 · Fix 30 ·
   Min 44 · Max 44 = 296 px. Persisted in `QSettings` under
   `fit/single/columns`. The `↗`
   opens the full eight-column grid (the live widget re-parented into a
   non-modal dialog, exactly the Parameters panel's `Fitted parameters`
   pop-out) with `Copy TSV` / `Close`.
   **Hidden state stays visible**: a linked parameter paints a small `⇄1`
   badge and a tied parameter shows `= 2·f` in accent colour inside the
   Value cell (extend `_ValueUncertaintyDelegate`, which already overlays
   ±σ), so nothing a hidden column would say is lost.
4. **Run row** — `Fit` (primary, swaps to `Stop`) · `Reset` · `Preview` ·
   stretch · **χ²ᵣ verdict chip** (same `verdict_chip_qss` + colours as the
   Parameters cards; hidden until a fit; click opens `FitResultsWindow` for
   this run). `Asymmetric errors` stays as its own checkbox row.
5. **Results card** (replaces `#resultBox`, the `Pull diagnostic…` button
   and the `#carryForwardBadge`): a `RangeCard`/`ParameterCard`-style
   surface. Header strip: tag `Fit ✓` / `Fit ⚠` / `No fit yet` ·
   `converged` · stretch · muted `ndof 595 · npar 5` (or the `seeds from
   3001` tag, which replaces the carry-forward badge and its `✕`). Body:
   one `KeyValueGrid`-style line `χ²/ν 0.9794 · verdict good [· λ at
   bound]`. Footer: the **hand-offs** as segmented buttons in a wrapping
   row — `Diagnostic…` (today's Pull diagnostic), `Add to series…`,
   `Send to Batch →`. Before any fit only `Send to Batch →` is enabled, with
   the same disabled-reason tooltips as today.

**Batch tab** — the same grammar so the two tabs read as one surface.

1. **Model** — formula; row `Edit…` · `Wizard…` · stretch · a `seeded from
   3001` tag when the function arrived via Send to Batch.
2. **Fit range** — unchanged.
3. **Parameters rail** — title, `Bounds` chip (off by default), `↗`. The
   `?` button becomes the section's hint (`PanelSection.set_hint`, the
   existing help text) — no more `QMessageBox`. Table at rest:
   Parameter · Seed · Type (the existing `Global/Local/Fixed/File` combo);
   with `Bounds` on, today's single `"min, max"` text column becomes the
   same Min · Max pair as the Single tab.
4. **Seeding row** — `Seeding` · combo · stretch · `Per-run seeds…`
   (segmented), one wrapping row.
5. **Run row** — `Run batch fit` (primary) · stretch · summary verdict chip
   `3 ✓ 1 ⚠` (clean count · flagged-or-failed count, full sentence on the
   tooltip) after a run.
6. **Results card** — tag `Batch ✓` / `Batch ⚠` · `3 of 4 converged` · χ²ᵣ
   range; body: **one verdict chip per run** (`3001 ✓ 0.98`, `3003 ⚠ 1.9`)
   in a `FlowLayout`, each opening that run's `FitResultsWindow`; footer
   hand-offs `Use as seeds` (today's `Use suggested per-run seeds` from the
   `#seedingSignpost`) · `Trends →` (bring the Parameters panel to the front
   **and** select this batch's series pill so its cards appear at once).

**Removed**: the `More…` menu and its `QToolButton`, the `Drop background`
action and `_update_drop_background_enabled` / `_on_drop_background`, the
vertical button stack, `#resultBox` on both tabs, `#carryForwardBadge`, `#seedingSignpost`
as a separate frame, the batch `?` button, the reserved horizontal-scrollbar
height in `_size_param_table_to_content`.

**Naming** (settled): `Edit…` / `Wizard…` on both tabs; the docs quote
`Edit Function...`, `Fit Wizard...`, `Global Wizard...` today and change in
the same PR.

## Options explored (page "Directions" on the canvas)

More… replacement:

| | Idea | Trade-off |
|---|---|---|
| **A (chosen)** | The hand-offs go where their meaning lives: on the results card beside `Diagnostic…`. `Drop background` is retired. | Needs the results card; a hand-off that needs no fit (`Send to Batch →`) sits on a card that may say "No fit yet". |
| B | One wrapping chip row under the formula: `Edit…` `Wizard…` `→ Batch` `→ Series…`. | Everything visible, but model editing and hand-offs mix, and the row wraps at 320 px. |
| C | Collapsible `Model options` `PanelSection` (the Multi-Group panel idiom) holding the two hand-off buttons. | Better labelled than More…, still one click away, costs a header row. |

Parameter table:

| | Idea | Trade-off |
|---|---|---|
| **A (chosen)** | Column-group chips on the section header (`Bounds` = Min · Max, `Links`, `Batch`) + `↗` pop-out. | Chips share the header row and wrap under it if the dock is narrower still; state to persist. Same chip widget as the Parameters panel's y chips, at the footer size. |
| B | One compact card per parameter (name · value · fix; expand for bounds / link / tie), `ParameterCard` idiom. | Values no longer line up in a column; five parameters cost roughly twice the height. |
| C | Hide columns automatically as the dock narrows; pop-out for the rest. | No new control, but what you see depends on window width and cannot be chosen. |

## Decisions to make

None open — see "Decisions recorded".

## Architecture notes

- `FitParameterTable` (`tab_base.py:1362`) already has `set_batch_column_visible`;
  the rail generalises it to `set_column_group_visible(group, on)` over the
  existing eight columns (`COL_MIN`/`COL_MAX` at 7 characters). The
  in-dock table and the `↗` pop-out are the same widget — the pop-out shows
  every column regardless of the chips. The grouped-single
  `FitParameterTable` (`global_tab.py:558-565`) hides Batch/Link/Tie today
  and simply has no `Links`/`Batch` chips.
- The χ²ᵣ chip and both results cards are pure views fed the frozen
  `FitResults` snapshot that `FitResultsWindow` already consumes; the batch
  card gets one snapshot per member from the existing per-run results.
- Hand-off signals (`send_model_to_batch_requested`, `add_to_series_requested`)
  keep their names and receivers; only the emitting widgets move.
- Every new row is a `FlowLayout` (`gui/widgets/flow_layout.py`) or a
  single fixed row narrower than the fit-range row, so the dock minimum
  width is set by the fit-range row.

## Coding rules for every phase

Identical to `parameters-panel-cards.md`: lean, no compatibility aliases, no
defensive guards on our own attributes, delete don't deprecate, rewrite the
tests that pin the old layout (`tests/gui/test_fit_panel_density.py` pins
the More… contract; `test_fit_workflow_guards.py`, `test_fit_slot_orchestration.py`,
`test_fit_panel_tabs.py`, `test_send_to_batch_seeds_from_single_fit.py`,
`test_fit_parameter_table.py`, `test_fit_panel_affine_ties.py`,
`test_grouped_single_fix_table.py` touch the moved actions and columns),
GUI rules from `docs/GUI_GUIDELINES.md`, validate through `tools/harness.py`.

## Phases

Built phase by phase by subagents in the main checkout on branch
`feat/fit-tab-refresh`, one commit per phase, a lead review after every
phase, one PR at the end. Each phase's brief is this section plus "Coding
rules for every phase"; an implementer reads the whole file first.

Shared vocabulary used below:

- **rail chip** — a checkable `QPushButton` styled by
  `style_group_state_button(chip, "active"|"unselected", palette="blue")`
  with `footer_font()`; identical widget to the y chips in
  `panels/fit_parameters_panel.py::_style_chip`, one size down.
- **segmented button** — a `QPushButton` with `build_segmented_button_qss()`.
- **verdict chip** — `QPushButton`/`QLabel` with `objectName ==
  VERDICT_CHIP_OBJECT_NAME`, `verdict_chip_qss(colours, widget=…)`,
  `mono_font(SIZE_NUMERIC)`; colours from `FIT_VERDICT_CHIP_COLOURS` /
  `NEUTRAL_CHIP_COLOURS` (`styles/widgets.py`).
- **tag** — `make_context_chip(text)`.
- **card surface** — the `ParameterCard`/`RangeCard` resting chrome:
  `QFrame` with an objectName, `1px solid BORDER`, `SURFACE`, radius 4,
  margins `(8, 6, 8, 6)`, spacing 4 (`gui/widgets/parameter_card.py:388`).

### Phase 1 — `FitParameterTable` column groups, narrow Min/Max, value badges (Opus)

Table only (`src/asymmetry/gui/panels/fit/tab_base.py::FitParameterTable`,
line ~1362); no tab layout changes.

- `set_column_group_visible(group: str, visible: bool)` and
  `column_group_visible(group) -> bool` over three groups: `"bounds"` →
  `COL_MIN`, `COL_MAX`; `"links"` → `COL_LINK`, `COL_TIE`; `"batch"` →
  `COL_BATCH`. Unknown group is a `ValueError`. This replaces
  `set_batch_column_visible` — delete it and update its callers
  (`global_tab.py:558-565` hides Batch/Link/Tie for the grouped-single
  surface: that becomes two group calls; tests in
  `tests/gui/test_fit_parameter_table.py`, `test_grouped_single_fix_table.py`,
  `test_cross_group_fit_dialog.py`, `test_count_fit_panel_gui.py` — grep
  `set_batch_column_visible` under `tests/`).
- Min and Max column widths: 6 characters (`char_width(6)`) instead of 7,
  in the `_column_chars` table at ~1397. Name 13 · Value 12 · Fix 4 · Min 6 ·
  Max 6 is the resting set: assert in a test that the summed width of the
  visible columns with only `bounds` on is `< char_width(42)` and that the
  full eight columns are still all sized (no zero-width column).
- `_size_param_table_to_content` (~1317) stops reserving the horizontal
  scrollbar's height; keep `ScrollBarAsNeeded` so a wide pop-out still works.
- Value-cell badges in `_ValueUncertaintyDelegate` (~1264): when the row's
  Link combo is on group *N* paint `⇄N`, when the row has an affine tie paint
  `ƒ`, both in `tokens.ACCENT` at the right edge of the Value cell after the
  ±σ overlay, so the state survives the `links` group being hidden. Expose
  the text through a pure method `FitParameterTable.value_badge(row) -> str`
  (`""` when neither) that the delegate calls and the tests assert; tooltip
  of the Value cell appends `_format_tie_formula(...)` for a tied row.
- Tests: extend `tests/gui/test_fit_parameter_table.py` — group show/hide,
  the width budget, `value_badge` for linked / tied / plain rows, and that
  hiding `links` keeps the combo/tie state (re-showing restores the widgets
  unchanged).

Gate: `python tools/harness.py test -- tests/gui/test_fit_parameter_table.py
tests/gui/test_grouped_single_fix_table.py tests/gui/test_fit_panel_affine_ties.py`
green, `structural` + `lint` green. Commit `feat(fit): column groups and
value badges on FitParameterTable`.

### Phase 2 — `FitResultsCard` widget; `FitResultsWindow` for run fits (Opus)

New `src/asymmetry/gui/widgets/fit_results_card.py`, pure view, built and
tested in isolation (`tests/gui/test_fit_results_card.py`). Nothing in the
fit tabs changes in this phase.

- `FitResultsCard(actions: Sequence[tuple[str, str]], parent=None)` —
  `actions` are `(label, tooltip)` pairs rendered as segmented buttons in a
  `FlowLayout` at the bottom of the card, in order. Signals:
  `action_triggered(str)` (the label) and `member_requested(int)` (a run
  chip was clicked).
- Anatomy on the card surface (objectName `fitResultsCardSurface`):
  header strip = tag · `ElidedLabel` headline (muted) · stretch · muted
  mono meta label **or** a second tag (`set_meta_tag`); body = word-wrapped
  rich-text detail label in `status_font()`; a `FlowLayout` strip of member
  verdict chips (hidden when empty); the action row.
- API: `set_message(text: str, *, tag: str = "No fit yet")` — neutral tag,
  `text` in the detail label, members cleared, meta cleared (used for every
  placeholder, progress and error string the tabs show today);
  `set_summary(summary: FitCardSummary)`; `set_meta_tag(text: str | None)`;
  `set_action_enabled(label: str, enabled: bool, tooltip: str | None = None)`
  (unknown label → `KeyError`).
- `@dataclass(frozen=True) FitCardSummary`: `tag: str` (`Fit ✓`, `Fit ⚠`,
  `Batch ✓`, `Batch ⚠`), `tone: Literal["ok", "warn", "error", "neutral"]`
  (tag colours: ok → `FIT_VERDICT_CHIP_COLOURS["good"]`, warn → the amber
  `(WARN_BANNER_BG, WARN, WARN_BANNER_TEXT)`, error → `["poor"]`, neutral →
  `NEUTRAL_CHIP_COLOURS`), `headline: str`, `meta: str`, `detail_html: str`,
  `members: tuple[MemberChip, ...] = ()`. `@dataclass(frozen=True)
  MemberChip`: `run: int`, `text: str` (`3001 ✓ 0.98`), `colours`,
  `tooltip: str`.
- `FitResultsWindow` (`gui/windows/fit_results_window.py`) generalised for a
  run's asymmetry fit: `FitResults` gains `title: str` (the window title;
  the Parameters panel passes `f"Fit results — {format_param_label(name)}"`
  where it builds the snapshot, `fit_parameters_panel.py:~4018`), and the
  constructor takes `editable: bool` — `Edit model fit…` exists only when
  `True`. Update `tests/gui/test_fit_results_window.py` and the panel's
  call site.
- In `tab_base.py` add the adapter the tabs will use in phases 3–4:
  `fit_results_snapshot(result, *, title: str, model: str, fit_range: str,
  runs: str) -> FitResults` building one `FitRangeResults` from a core
  `FitResult` (free parameters first, then fixed; `format_param_label`
  symbol/unit split as `fit_parameters_panel` does; chip text
  `f"χ²ᵣ {result.reduced_chi_squared:.3g}"`; verdict + colours from the same
  `_fit_summary(result)["quality"]` that `_fit_success_html` uses). Unit test
  it in `tests/gui/test_fit_results_window.py` with a fabricated `FitResult`.

Gate: new tests + `test_fit_results_window.py` + `test_fit_parameters_panel.py`
green, `structural` + `lint` green. Commit `feat(gui): FitResultsCard and
run-fit FitResultsWindow`.

### Phase 3 — Single tab restructure (Opus)

`src/asymmetry/gui/panels/fit/single_tab.py`, `tab_base.py` (shared
builders), `panel.py`, `gui/widgets/panel_section.py`.

- **Model row.** `FitTabBase._build_model_row(*buttons) -> QWidget`: a
  `FlowLayout` row holding `Edit…` (segmented; built by
  `_build_formula_box`, label changes from `Edit Function...`) followed by
  the given buttons. Single passes `Wizard…` (was `Fit Wizard...`, same
  tooltip/enable rule). Delete `_more_btn`, `_more_menu`,
  `_drop_background_action`, `_on_drop_background`,
  `_update_drop_background_enabled`, `_send_to_batch_action`,
  `_add_to_series_action`, `_update_add_to_series_enabled` and the vertical
  `model_button_layout`. The signals `send_model_to_batch_requested` and
  `add_to_series_requested` stay and are now emitted from the results card
  (`action_triggered`).
- **Parameters rail.** `PanelSection` gains `add_header_widget(widget)`:
  placed after the title with stretch 1, before the suffix label. The rail
  widget is a `FlowLayout` of three rail chips `Bounds` · `Links` · `Batch`
  and a `↗` `QToolButton` (tooltip `Show every column in a window`). Each
  chip drives `table.set_column_group_visible`; defaults `bounds` on, the
  others off; persisted as `QSettings` key `fit/single/columns` (JSON dict
  group → bool) read on construction, written on toggle. The `↗` pops the
  **live** table out: re-parent it into a non-modal `QDialog` titled
  `Fit parameters — <run label>` with every group visible while popped out,
  `Copy TSV` (tab-separated header + rows) and `Close`; a muted placeholder
  label `Shown in the pop-out window` sits in the tab meanwhile; closing the
  dialog (or `reject()`) returns the table and the chips' groups. Sizing
  follows `FitParametersPanel._size_table_dialog_to_content`
  (`fit_parameters_panel.py:~6488`, `resize_to_available`).
- **Run row.** One `QHBoxLayout`: `Fit`/`Stop` · `Reset` · `Preview` ·
  stretch · **χ²ᵣ verdict chip** (`QPushButton`, hidden until a recorded
  fit; text `χ²ᵣ 0.979`; colours from the fit summary verdict; tooltip =
  verdict + free parameters as `sym = value(err) unit`; click opens
  `FitResultsWindow(fit_results_snapshot(result, title=f"Fit results —
  {run}", …), editable=False)`, one window kept per tab and refreshed on the
  next fit). `Asymmetric errors` keeps its own row beneath. The
  `QGridLayout` footer goes.
- **Results card.** `self._results_card = FitResultsCard(actions=(
  ("Diagnostic…", <today's Pull diagnostic tooltip>),
  ("Add to series…", ""), ("Send to Batch →", <today's Send to Batch
  tooltip>)))` under a `make_section_header("Results")`. Every
  `_result_label.setText(...)` becomes `set_message(...)` (errors keep their
  text; `Fitting...` → `set_message("Fitting…", tag="Fitting")`; cancelled →
  `tag="Cancelled"`). A converged fit: `set_summary(FitCardSummary(tag="Fit
  ✓", tone="ok", headline="converged", meta=f"ndof {ndof} · npar {npar}",
  detail_html=<χ²/ν · verdict chip html from fit_quality_chip_html · Δ‖p‖ ·
  warnings from _fit_warnings_html>))`; a flagged-but-usable result
  (`_fit_result_is_usable`) uses `tag="Fit ⚠"`, `tone="warn"`; wizard
  outcomes (lines ~786, ~837) go through the same two calls. `Diagnostic…`
  enabled by `_can_run_pull_diagnostic`; `Add to series…` enabled/tooltip
  by the logic that lives in `_update_add_to_series_enabled` today;
  `Send to Batch →` always enabled. Delete `_results_group`, `_result_label`,
  `_pull_diagnostic_btn`, the `RESULT_BOX_*` imports in this file (the
  constants stay in `styles/widgets.py` — other windows use them).
- **Carry-forward badge** → `show_carry_forward_badge(text)` becomes
  `set_meta_tag(short)` with the full text as the tag's tooltip and
  `clear_carry_forward_badge` → `set_meta_tag(None)`; the tag reads
  `seeds from <run>` — derive the run from the text `panel.py:536` passes
  (change that call to pass the run number explicitly rather than parsing).
  Delete `_carry_forward_badge*`.
- Layout: the tab must not need a horizontal scrollbar at
  `char_width(40)` width: add a test that builds `SingleFitTab`, resizes it
  to that width, shows it, and asserts
  `table.horizontalScrollBar().maximum() == 0` with the default groups and
  `minimumSizeHint().width() <= char_width(40)`.
- Rewrite the tests that pin the old layout (grep `_more_btn`,
  `_result_label`, `_pull_diagnostic_btn`, `_carry_forward`,
  `Drop background`, `_drop_background`, `More…` under `tests/gui/`):
  `test_fit_panel_density.py`, `test_fit_workflow_guards.py` (Drop-background
  tests are deleted, not ported), `test_fit_slot_orchestration.py`,
  `test_fit_panel_tabs.py`, `test_send_to_batch_seeds_from_single_fit.py`,
  `test_fit_panel_minos_gui.py`, `test_fit_panel_rrf_gui.py`,
  `test_single_fit_carry_forward_badge_gui.py`,
  `test_single_fit_flagged_result.py`, `test_surface_fit_warnings_gui.py`,
  `test_pull_diagnostic_window.py`, `test_user_functions_gui.py`,
  `test_frequency_domain_fitting.py`, `test_mainwindow_additional.py`,
  `test_project_schema.py` — only where they touch the Single tab; Batch
  references wait for Phase 4. Docs screenshot scenarios that reach into
  these private attributes (grep the same names under
  `docs/screenshots/scenarios/`) are updated so they still run.

Gate: every file above green under `python tools/harness.py test -- …`,
`structural` + `lint` green. Commit `feat(fit): model row, parameter rail,
results card on the Single tab`.

### Phase 4 — Batch tab and the Multi-Group tabs (Opus)

`src/asymmetry/gui/panels/fit/global_tab.py`, `panel.py`,
`gui/windows/multi_group_fit_window.py`, `gui/mainwindow.py`.

- **Model row** via `_build_model_row`: `Edit…` · `Wizard…` (was `Global
  Wizard...`). The grouped-single `Send to Batch` button leaves the model
  row: it becomes that surface's card action (below). A `seeded from <run>`
  meta tag appears when the function arrived through Send to Batch
  (`FitPanel.send_single_model_to_batch`, `panel.py:~1194`).
- **Parameters rail** on the `Parameter Classification` section: rail with a
  `Bounds` chip (default **off**) and `↗`; settings key `fit/batch/columns`.
  The `?` button and its `QMessageBox` go; the section gets
  `set_hint("Global: one value shared by every run · Local: fitted per run ·
  Fixed: held at the seed · File: taken from run metadata")` and the full
  help text moves to the docs (Phase 6 quotes it). The single `Bounds`
  text column (`"min, max"`) splits into `Min` and `Max` columns (6 chars,
  hidden unless the chip is on) in the classification table, the
  `Fit-Function Parameters` table and the `Per-Group Parameters` table;
  update every reader/writer of the old column (grep `"Bounds"`,
  `_parse_bounds`, column index 3 helpers in `global_tab.py`).
- **Seeding row**: `Seeding` · combo · stretch · `Per-run seeds…`
  (segmented; `Edit per-group initial values…` where that variant applies),
  wrapping.
- **Run row**: `Run batch fit` (was `Run Batch Fit`; grouped: `Run grouped
  fit`) / `Stop` · `Preview` (grouped only) · stretch · a summary verdict
  chip (`QLabel` variant) `3 ✓ 1 ⚠` — ok colours when every member
  converged, warn otherwise; hidden until a run completes. `Asymmetric
  errors` keeps its row.
- **Results card** under `make_section_header("Results")`:
  batch (`member_kind="runs"` and grouped batch): actions `("Use as seeds",
  <today's signpost button tooltip>)`, `("Trends →", "Show this batch's
  trends in the Parameters panel")`; grouped single: `("Send to Batch →",
  <today's tooltip>)`. Every `_result_text.setText/setHtml` → `set_message`
  (progress strings get `tag="Fitting"`); the converged paths (~3278, ~3464,
  ~4122) → `set_summary(FitCardSummary(tag="Batch ✓"|"Batch ⚠", tone,
  headline=f"{n_ok} of {n} converged", meta=<χ²ᵣ min–max>, detail_html=<the
  stats + seeding-mode line the text shows today>, members=<one MemberChip
  per run: f"{run} ✓ {chi2:.3g}" or "⚠", colours from that member's
  verdict, tooltip = the per-run advisory flags>))`. `member_requested(run)`
  opens `FitResultsWindow(fit_results_snapshot(...), editable=False)` for
  that run (one window per run, refreshed on the next batch).
  The `#seedingSignpost` frame goes: its advisory sentence is appended to
  `detail_html` and `Use as seeds` is enabled exactly when the signpost
  would have shown (`test_fit_panel_seeding_signpost_gui.py` is rewritten
  to the card); `Open per-run seeds…` is dropped (the seeding row has it).
- **`Trends →`**: `GlobalFitTab.trends_requested = Signal()`; `FitPanel`
  and `MultiGroupFitWindow` re-emit it; `MainWindow` remembers the
  `batch_id` it records in `_on_global_fit_completed`
  (`mainwindow.py:~12779`, and the grouped equivalent) as the last batch
  per surface, and on the signal shows the Parameters panel
  (`self._parameters_stack.setCurrentWidget(self._fit_parameters_panel)`,
  `self._dock_fit_parameters.raise_()`) and calls
  `self._fit_parameters_panel.select_series([batch_id])`. The action is
  enabled only after a batch has been recorded.
- Rewrite the remaining tests (grep `_result_text`, `_seeding_signpost`,
  `Run Batch Fit`, `Global Wizard`, `"Bounds"` under `tests/gui/`):
  `test_fit_panel_tabs.py`, `test_grouped_batch_seeds_gui.py`,
  `test_fit_panel_seeding_signpost_gui.py`, `test_count_fit_panel_gui.py`,
  `test_cross_group_fit_dialog.py`, `test_grouped_single_fix_table.py`,
  `test_mainwindow_additional.py`, and any other hit; add
  `tests/gui/test_fit_trends_handoff.py` (a fabricated completed batch →
  `trends_requested` raises the Parameters dock and selects the series).
  Docs screenshot scenarios touching these names are updated to run.

Gate: those files green, `structural` + `lint` green, and the lead drives
both tabs once in `gui-smoke`. Commit `feat(fit): rail, results card and
Trends hand-off on the Batch and Multi-Group tabs`.

### Phase 5 — sweep the remaining tests (Sonnet)

Grep `tests/` and `docs/screenshots/scenarios/` for every name removed in
Phases 1–4 (`set_batch_column_visible`, `_more_btn`, `_result_label`,
`_result_text`, `_pull_diagnostic_btn`, `_carry_forward`,
`_seeding_signpost`, `Edit Function...`, `Fit Wizard...`, `Global
Wizard...`, `Run Batch Fit`, `Drop background`, `More…`) and rewrite what is
left to the new widgets — no skips, no behaviour changes. Then run `python
tools/harness.py test --tier fast`, the GUI files touched, and finally
`python tools/harness.py validate`; report any failure you cannot explain
rather than papering over it. Commit `test(fit): migrate the remaining fit
tab tests`.

Gate: `validate` green (the lead re-runs it).

### Phase 6 — docs, screenshots, changelog (Sonnet)

- `docs/reference/gui_usage.rst` "Fitting panel" (line ~832), "Single
  dataset fitting", "Fitting workflow", "Carrying a model forward between
  runs", "Batch fitting", "Fitting a group directly": describe the model
  row, the Parameters rail chips (`Bounds`, `Links`, `Batch`, `↗`), the
  χ²ᵣ chip, the Results card and its hand-offs (`Diagnostic…`, `Add to
  series…`, `Send to Batch →`, `Use as seeds`, `Trends →`), the `seeds
  from <run>` tag, quoting every UI string verbatim from the widget code;
  add the parameter-role explanation that left the `?` box and a sentence
  on removing a background term in the function editor (the light-OFF A₀
  rationale from the old tooltip).
- `docs/reference/fitting.rst` (Fix ~132, Tie ~349, Batch ~408): the
  column groups and the pop-out.
- Scenarios (`docs/screenshots/scenarios/`): `fit_wizard_gkt`,
  `quickstart_first_fit`, `fit_asymmetric_errors`, `batch_tab_group_binding`,
  `global_fit_lfkt`, `lf_kt_global_results`, `grouped_fit_ybco_knight`,
  `muon_fluorine_pbf2`, `alc_field_scan`, and
  `corpus/…llz…` (its notes widen the dock to 560 px only to show the
  Bounds column — drop that). Follow the determinism and size-budget rules in
  `docs/README.md` § "Maintaining the documentation"; run
  `python docs/screenshots/capture.py --check-registry` (or the equivalent
  flag the script offers) and `python tools/harness.py docs`.
- `CHANGELOG.md` `[Unreleased]`: **Changed** (model row, rail, results
  card, chip, hand-offs, renamed buttons) and **Removed** (`More…`, `Drop
  background`, the `?` help box, the seeding signpost frame).

Gate: `python tools/harness.py docs` green; the lead reads the pages.
Commit `docs(fit): describe the refreshed Fit tab`.

### Phase 7 — lead: validate, PR

`validate`, `structural`, `lint`, `gui-smoke`; open the PR with the summary,
the mockup link and the decisions list.

## Decisions recorded

- 2026-09-13: Min and Max stay separate columns (Ben); no merged Bounds cell
  on either tab. The rail chip that toggles them is still called `Bounds`.
- 2026-09-13: `Drop background` retired rather than turned into a chip —
  the term is removed in the function editor.
- 2026-09-13: `Send to Batch →` lives on the results card, beside
  `Diagnostic…` and `Add to series…`, enabled before any fit.
- 2026-09-13: model-row buttons are `Edit…` / `Wizard…` on both tabs.
- 2026-09-13: `Trends →` switches to the Parameters panel and selects the
  batch's series pill.
- 2026-09-13: the two Multi-Group `GlobalFitTab` instances get the results
  card and rail in the same PR.
- 2026-09-13 (lead fix-up, from Ben's screenshot review): the Value / Seed /
  per-group value columns stretch to the leftover width instead of leaving
  an empty band; the `Batch members` list sizes itself to its rows (capped
  at 8, then scrolls); both run rows wrap (`FlowLayout`) and the batch
  outcome chip reads `3 ✓ 1 ⚠` with the sentence on its tooltip, because
  the fixed-width chip pushed the grouped dock to 387 px; Min/Max are 7
  characters (6 elided `-inf` in the mono font).
- 2026-09-13 (second lead fix-up, from Ben's review of the same screenshots):
  the parameter-role help is a ⓘ button at the head of the Batch rail opening
  a popover (`gui/widgets/info_popover.py::InfoPopover`, extracted from
  `PhaseInfoPopover`), not a hint line — a permanent two-line hint costs more
  of a ~300 px dock than an explanation read once is worth, and riding the
  rail's `FlowLayout` keeps the ⓘ out of the dock's minimum width; the
  `Batch members` list caps at 3 rows; and the value columns are `Interactive`
  with a fill rule (`ElasticTable`) rather than `Stretch`, because a stretched
  column made every drag to its right pull width *out* of it and could not be
  dragged itself. Dragging now pushes the columns to the right along, and only
  a viewport change takes width back.
