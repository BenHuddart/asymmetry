# Parameters panel: parameter cards, chip rail, pop-out table

Status: implemented 2026-09-13 on `feat/parameters-panel-cards`, one PR
awaiting review; built phase by phase by subagents with a lead review gate
after every phase. Mockups:
https://claude.ai/code/artifact/66458740-3fc1-423b-9b68-4c5e11f4f5b4 (page
"Recommended" is what shipped).

## Problem

`FitParametersPanel` (`src/asymmetry/gui/panels/fit_parameters_panel.py`) was
built early and never brought up to the rest of the app. On a 13" display the
controls form (series pills, Show table, X axis, a 3-row Y selector table, the
Global-param hint, two collapsible sections, Plot mode, Model components) lives
in a scroll area above a vertical splitter, so half of it is hidden under a
scrollbar; the plot below gets what is left, and "Subplots" mode packs the
parameters into a two-column grid of tiny axes. Per-parameter controls (Model
Fit, log) live in a table row rather than with the plot they act on, and the
fitted table is a dead copy in a dialog.

## Design (settled)

**Layout, top to bottom, nothing scrolling at rest.**

1. **Series strip** — the red series pills, unchanged behaviour (click = view,
   shift-click = overlay series, right-click menu). Each pill carries a *short*
   name — the user's rename if any, else the series' `member_range`, with the
   model (then the browser-group suffix) appended only where two pills would
   otherwise read alike — elided at `_CHIP_MAX_CHARS` with the full name on the
   tooltip. The host computes the short names (only it sees the whole set) and
   passes them as `load_representation_series(short_names_by_id=…)`. The strip
   is a `FlowLayout`, so it wraps instead of widening the panel.
2. **Rail row 1 (x)** — `x` label · x-axis picker (the existing combo) · **ƒ**
   transform button (menu: presets + Custom…, label shows the active lens,
   e.g. `1/x`) · `log` checkbox · `Fold:` combo (Angle x only, as today) ·
   stretch · **Table** button (shows the pop-out) · **⋯** menu.
3. **Rail row 2 (y)** — `y` label · one **checkable chip per trendable
   parameter** (checked = its card exists) · **+** menu (`Derived parameter…`,
   `Knight shift window…`) · stretch · **Subplots │ Overlay** segmented
   toggle. The row wraps (flow layout) when chips overflow.
4. **Plot area** — in *Subplots* mode a vertical **card stack**, one
   `ParameterCard` per checked chip; in *Overlay* mode a single canvas
   (today's "Single Axes" drawing, twin y-axis for two parameters, untouched).
5. **Footer**, two rows — row 1 names the active series in full (`2 series`,
   names on the tooltip, with an overlay selected), elided, hidden when no
   series is active; row 2 is trend provenance on the left
   (`4/4 members in trend`) and the Global-held-constant note on the right
   (short form, full text as tooltip, never shrunk below its size hint).

**ParameterCard** (`gui/widgets/parameter_card.py`, pure view):

- Header: disclosure arrow · colour swatch · name (`format_param_label`,
  elided, full on tooltip) · **Fit** button · `log` checkbox · **ƒ** button
  (per-parameter y transform, label shows the active lens) · stretch · focus
  button (⤢ / ⤡) · drag grip.
- Body: one Matplotlib canvas from `create_canvas`. The fit's read-out lives
  in the header instead: a **χ²ᵣ chip** (`χ²ᵣ 0.89`) right of the Fit button,
  shown only while the parameter has an active fit with a successful range and
  coloured by the fit-quality verdict; its tooltip carries the verdict band and
  every fitted parameter (`T꜀ = 35.8(5) K`), and clicking it opens that
  parameter's `FitResultsWindow`.
- Collapsed: header only, with a QPainter sparkline (no Matplotlib) in place
  of the right-hand controls; `Fit`/`log`/`ƒ` hidden; a `derived` tag stays.
- Focused: the card takes all the stack's height, every other card collapses
  beneath it (their previous expanded state is restored on exit). Focus is
  purely a size gesture — labels are placed from the plot's right-click menu,
  which every card and the Overlay canvas carry.
- Drag the grip to reorder cards (stack-internal drag/drop).
- Derived (composite) cards: header right-click menu `Edit derived…`,
  `Remove`. Knight-shift K traces: `Remove` only (as today's Remove button).

**Card Fit button labels** (replace today's `Model Fit` family everywhere):

| state | text |
|---|---|
| no active fit | `Fit…` |
| active fit with a successful curve | `Fit ✓` |
| fit computed under a different transform (curve hidden) | `Fit ⚠` |
| ≥ 2 series pills selected | `Global fit ×N…` |

Tooltips carry today's longer explanations.

**⋯ menu**: `Export TSV…`, `Export GLE (PDF)…`, `Export GLE (EPS)…`,
checkable `Show components`, `Knight shift window…` (always; the rail `+`
menu also lists it only when `_update_knight_window_button`'s predicate says
the series looks like a Knight-shift case).

**Per-parameter y transforms.** The y lens becomes `dict[param, AxisTransform]`
(x stays global). Rationale (physics advisor, 2026-09-13): x is one shared
coordinate; y is a different physical quantity per card, so one y lens across
dimensionally unlike parameters (σ² beside raw β, ln ν beside linear λ, 1/λ
beside a raw amplitude) was only coherent while the panel was used one
parameter at a time. The model-fit transform signature is already keyed per
parameter, so this also fixes changing λ's lens stranding β's stored fit.

**Pop-out table.** The panel's live `_table` (Trend checkboxes, χ²ᵣ flags)
moves into a non-modal dialog created once and shown by the rail's Table
button; the Global-parameter values sit above it, `Copy TSV` and `Export…`
below. No hover-linking to the cards in this PR (deferred: needs blitting).

**Removed**: the Y selector table, the controls scroll area and splitter, the
`Axis transforms` and `Derived parameters` sections and their QSettings keys,
`Plot mode` combo, the labels bar and export bar under the plot, the hidden
global log-y checkbox, the Global-param hint label (folded into the footer),
`Show table` button, `New/Edit composite`, `Remove` and `Knight shift window…`
buttons (folded into menus), the copy-table dialog.

## Persisted state (`FitParametersPanel.get_state` / `restore_state`)

- `plot_mode`: `"Subplots"` | `"Overlay"`. Read `"Single Axes"` as
  `"Overlay"`.
- `selected_y_params`: unchanged key; now means "chips checked".
- `card_order: list[str]`, `collapsed_params: list[str]` — new; focus is not
  persisted.
- `y_transforms: {name: AxisTransform.to_dict()}` (identity entries never
  stored). A legacy `y_transform` key is applied to the parameters in that
  state's `selected_y_params` (unselected parameters never showed the lens).
  `axis_transform_custom_memory` keeps `"x"` and gains per-parameter keys
  `"y:<name>"`.
- Dropped: the two section-expanded QSettings keys. No project schema bump —
  the keys are additive and read-side migrated.

## Architecture notes

- **One figure per card, one for Overlay.** `self._figure`/`self._canvas`
  stay as the Overlay surface. `_draw_plot` dispatches: Overlay → the existing
  single-axes bodies of `_draw_single_series` / `_draw_multi_series`;
  Subplots → for every expanded card, `_draw_param_axes(ax, y_name, …)`
  (extracted from the current Subplots branches of both methods, one series or
  several). `_axes_tag_map` (`id(ax) → tag`) already spans figures. Annotation
  and member-hit handlers are connected to every card canvas; they redraw via
  `event.canvas.draw_idle()`. One `LoadingOverlay` covers the whole plot area.
- **Chips are the selection.** `_selected_y_parameters()` reads checked chips
  in card order; `_y_controls[name]` keeps `(fit_button, log)` pointing at the
  card's widgets so the fit-label/log-guard code keeps its shape.
- **Redraw granularity.** A card-local change (log, ƒ, collapse) redraws that
  card only; x-axis, series and mode changes redraw everything. The existing
  120 ms debounce and the off-thread trend-curve cache stay; the cache is
  keyed per parameter already.
- **Flow layout.** `gui/widgets/flow_layout.py` — the canonical Qt
  `FlowLayout` (`addItem`, `sizeHint`, `heightForWidth`, `setGeometry`), used
  by rail row 2. Reusable; no second copy.
- **Sparkline.** `parameter_card.sparkline_pixmap(xs, ys, color, size)` —
  QPainter polyline + dots, colours from tokens / the series colour.
- **GLE export** keeps its stacked-subplots output; it reads the mode from the
  toggle (`"Subplots"`), not a combo.

## Coding rules for every phase

- Lean: no compatibility aliases for removed widgets, no `hasattr`/`getattr`
  guards on our own attributes, no "in progress" flags or latches, no
  try/except around our own code; make bad states impossible by construction.
  If a guard seems necessary, stop and ask the lead in the report.
- Comments say *why* only when it is not obvious; never restate the code.
- No short single-use helpers; extract only what is used in several places.
- Delete, don't deprecate. Rewrite tests that pin the old layout; never skip.
- GUI rules from `docs/GUI_GUIDELINES.md`: colours from `tokens`, fonts from
  the builders, sizes from `metrics` (no literal pixel geometry ≥ 24), shared
  widgets extended not forked, every canvas from `create_canvas`.
- Validate with `python tools/harness.py test -- <focused files>` while
  iterating, `python tools/harness.py test --tier fast` after non-GUI changes,
  `python tools/harness.py structural` and `lint` before reporting; the lead
  runs `validate` once at the end.
- Commit at the end of the phase on this branch with a conventional message;
  never push.

## Phases

### Phase 1 — per-parameter y transforms (Opus)

Panel only, no layout change yet; the existing `Y:` transform combo drives the
transform of the *currently selected* parameters (the migration rule).

- `_y_transform` → `_y_transforms: dict[str, AxisTransform]`,
  `_y_transform_for(name)` (identity default; identity never stored),
  `_set_axis_transform("x", t)` / `_set_y_transform(name, t)`.
- Substitute at every call site: `_series_y_arrays`, `_transformed_y_axis_label`,
  `_transformed_y_export_header`, `_transform_dropped_count`,
  `_apply_transform_log_guard` (per parameter), `_overlay_suppressed_for_transform`,
  `_transform_signature(name)` (x kind/expr + that parameter's y kind/expr),
  `_model_fit_transform_sig`, the twin-axis and GLE paths, `_update_transform_suffix`.
- TSV/GLE: raw columns verbatim; one `# Y transform [<name>]: …` comment per
  transformed parameter instead of the single line.
- State: `y_transforms` key + legacy migration; `axis_transform_custom_memory`
  per parameter.
- Docs: `docs/reference/parameter_trending.rst` "Axis transforms" section
  says the lens is per parameter.
- Tests: `tests/gui/test_trend_axis_transforms.py` rewritten to the new API,
  plus: two parameters with different lenses plot/export independently; the
  legacy key migrates onto selected parameters only; changing one parameter's
  lens does not suppress another's fit overlay.

Gate: focused test files + `--tier fast` green; lead reviews the diff.

### Phase 2 — `ParameterCard`, card stack, flow layout (Opus)

New `gui/widgets/parameter_card.py` (`ParameterCard`, `ParameterCardStack`,
`sparkline_pixmap`) and `gui/widgets/flow_layout.py`, built and tested in
isolation (`tests/gui/test_parameter_card.py`, `tests/gui/test_flow_layout.py`).
The card is a pure view: it owns its canvas/figure, emits
`fit_requested(name)`, `log_toggled(name, bool)`, `transform_menu_requested(name, QPoint)`,
`focus_toggled(name)`, `expanded_changed(name, bool)`, `context_menu_requested(name, QPoint)`;
the stack owns order, focus, drag reorder and emits `order_changed(list[str])`.
`set_fit_label(text, tooltip)`, `set_result(text, tooltip, colours)`,
`set_transform_label(text)`,
`set_sparkline(xs, ys, color)`, `set_swatch(color)`.
Card chrome follows `RangeCard` (`gui/widgets/range_card.py`) for surface,
border and active-state styling; Fit button uses the segmented QSS builders.

Gate: new tests green, `structural` green, lead reviews API against Phase 3's
needs before Phase 3 starts.

### Phase 3 — panel restructure (Opus)

Rewrite the constructor to the settled layout; wire chips, cards, Overlay
surface, ⋯ / + menus, focus, footer, the pop-out table; split the draw path;
per-canvas event handlers; state keys and migrations; GLE mode naming; card
Fit labels and summary line. Remove everything listed under "Removed".
Update `tests/gui/test_fit_parameters_panel.py` and add a small helper module
`tests/gui/_trend_panel.py` (`select_params(panel, names)`, `card(panel, name)`,
`axes_for(panel, name)`) that Phase 4 will use. Docs screenshot scenarios
(`docs/screenshots/scenarios/parameter_trending_*.py`) must still run; adjust
the private calls they make.

Gate: `test_fit_parameters_panel.py` green, `structural` + `lint` green, lead
reviews the diff and drives the panel once in `gui-smoke`.

### Phase 4 — migrate the remaining tests (Sonnet)

Every other test file that pinned the old widgets (`_y_selector_table`,
`_plot_mode_combo`, `_log_y_check`, `_figure`/`_canvas` in Subplots mode,
`_derived_section`, `_show_components_check`, `_show_table_dialog`, …) is
rewritten to the helpers from Phase 3. No skips, no behaviour changes to the
panel; if a test cannot be expressed against the new layout, the report says
why. Then the whole GUI subset for the affected files, plus `--tier fast`.

Gate: `python tools/harness.py validate` green (lead runs it).

### Phase 5 — docs, screenshots, changelog (Sonnet)

`docs/reference/parameter_trending.rst` (panel anatomy, UI strings verbatim
from the widget code), any other page naming the old controls
(`grep -rl "Show table\|Plot mode\|Y parameters\|Single Axes" docs/`),
screenshot scenarios updated so the captured images show the card layout
(scenario determinism rules in `docs/README.md`), `CHANGELOG.md`
`[Unreleased]` entry under **Changed** (user-facing description, the new
names, the per-parameter transform, the state migration).

Gate: `python tools/harness.py docs` green, lead reads the page.

### Phase 6 — lead: final validate, PR

`validate`, `structural`, `lint`, `gui-smoke`; open the PR with the summary,
the mockup link and the review checklist.

## Decisions recorded

- 2026-09-13: mode toggle labelled **Subplots │ Overlay** (not "Cards").
- 2026-09-13: collapsed strips show only name, derived tag and sparkline.
- 2026-09-13: y transform per parameter, x global (advisor rationale above).
- 2026-09-13: table stays a pop-out; hover-linking deferred.
- 2026-09-13: tests rewritten to the new widgets, no aliases.
- 2026-09-13 (follow-up): `ParameterCardStack.sizeHint` returns
  `minimumSizeHint`. Each expanded card prefers its figure height, and the
  dock's scroll area summed those preferences, so two cards already asked for
  more than a 13-inch viewport and the panel scrolled while the cards sat at
  their preferred height. The stack now asks for its floor and the expanding
  cards divide whatever height the dock gives.
- 2026-09-13 (follow-up): `MainWindow._inspector_default_width` floors at the
  Parameters panel's own `minimumSizeHint().width()` plus the style's scrollbar
  extent. The 0.20 window-width fraction lands under that on a 13-inch window,
  so the deck opened with a horizontal scrollbar over the x rail.
- 2026-09-13 (follow-up): the `Fitted parameters` pop-out sizes itself to the
  table's header length and row count (via the shared `resize_to_available`,
  capped to 90 % of the work area) on every show, and only ever grows, so a
  window the user widened is not fought.
- 2026-09-13 (follow-up): the card's one-line fit summary is gone. It cost a
  text row under every figure to say what a chip can: the header now carries a
  `χ²ᵣ 0.89` chip (colours from the same `assess_fit_quality` verdict the Model
  Fit dialog uses, neutral where χ²ᵣ carries no goodness information), with the
  fitted parameters in its tooltip, and clicking it opens a per-parameter
  `gui/windows/fit_results_window.py::FitResultsWindow` — a non-modal, reusable
  pure view fed a plain `FitResults` snapshot by the panel. Values there use the
  new `format_value_uncertainty` (`35.8(5)`).
- 2026-09-13 (follow-up): short series pills, full name in the footer. A pill
  carrying the default `<model> · <run range> · <group>` label is ~330 px wide,
  so a second series pushed the panel past a 13-inch dock — the strip was the
  last row whose width followed the data. Pills now show the run range (a
  rename always wins; collisions escalate to the model, then the group suffix)
  and wrap in a `FlowLayout`; the footer's new first row names the active
  series in full, elided, so nothing is lost. `ElidedLabel.set_hover_text`
  exists for the two footer labels whose tooltip is not simply the squeezed-out
  text (the provenance explanation, and the names behind `2 series`).
- 2026-09-13 (follow-up): labels move to one right-click menu on every plot —
  `Add label here…`, `Edit label…` / `Remove label`, `Clear labels`, merged
  with the trend-point membership toggle. The cards' and Overlay's tools rows
  and the armed-button gesture are gone; in Subplots mode a label no longer
  needs a focused card.
