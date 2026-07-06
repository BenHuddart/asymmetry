# GUI Guidelines

How to build GUI in Asymmetry so it stays consistent, scale-safe, and free of
the drift the panel refresh removed. Rules first; the harness enforces the
load-bearing ones mechanically (`python tools/harness.py structural`).

The one-line rule: **colours from tokens, fonts from builders, sizes from
metrics, and extend a shared widget rather than fork it.**

GUI code lives in `src/asymmetry/gui/` and wraps `asymmetry.core`; it never
reimplements analysis. Keep long work off the GUI thread (see Threading).

## Design tokens and palette

`gui/styles/tokens.py` is the **only** source of colour. Every chrome colour is
a named semantic token (`ACCENT`, `TEXT_MUTED`, `SURFACE`, `WARN`, `ERROR`,
`OK`, `WHITE`, `WARN_BANNER_BG`, …). Reference the token; never write a hex
literal in `gui/` outside `styles/`.

- Add a token when a colour carries **meaning** (a new banner state, a new
  semantic surface) and reuse the closest existing token when the meaning
  already exists. A one-off colour used in a single place is still a token —
  give it a semantic name and a short comment.
- The only sanctioned hex literals are **specialist, non-chrome** colours:
  matplotlib scientific-plot colours, QPainter icon fills, and deliberate accent
  cycles (e.g. the fraction-group palette). These live in an allowlist in
  `tools/harness.py` — see *What the harness enforces*.

## Typography

The type scale lives in `gui/styles/typography.py` (`SIZE_BODY`, `SIZE_HEADER`,
`SIZE_NUMERIC`, `SIZE_STATUS`, `SIZE_FOOTER`) with letter-spacing constants.
Build fonts through the **builders** — `header_font()`, `section_label_font()`,
`status_font()`, `footer_font()`, and `mono_font()` from `gui/styles/fonts.py` —
never with a literal `QFont(..., point_size)`. The builders bake in the scaled
point size, weight, letter-spacing, and family so a widget's text matches the
rest of the app and tracks the UI zoom. Uppercase section headers come from
`make_section_header()` (also used inside `PanelSection`).

## Sizing

Derive pixel sizes from `gui/styles/metrics.py`, which measures the **live**
application font at call time so sizes track the zoom:

- `field_width_for(chars, widget)` — min-width for a line edit / spinbox holding
  N characters (frame padding included). Use `dialog_width(chars)` for dialogs.
- `row_height(font=None)` — table row height (line height + padding).
- `char_width(n, font=None)` — raw N-character width.

**Literal-pixel geometry is banned** for `setFixedWidth`/`setMinimumWidth`/
`setFixedHeight`/`setMinimumHeight`/`setFixedSize` with an int literal **>= 24**:
a frozen pixel size drifts once the font-driven zoom scales everything else.
Small paddings and hairlines (< 24 px) are fine. A genuine design *floor* on a
canvas/table/scroll-area that metrics cannot derive may be allowlisted in
`tools/harness.py` with a reason — but prefer a metrics helper, and prefer
letting layouts size themselves (e.g. `QComboBox.setMinimumContentsLength(N)` is
font-relative and needs no pixel width).

## The UI zoom

UI scale is **font-driven**, not per-widget tracked. `UIManager` performs a
scale change in exactly three publish steps (see `gui/ui_manager.py`):

1. Publish the scale to the font builders (`set_ui_font_scale`) so any widget
   *built after* the change is born at the active scale.
2. Set the `QApplication` font to `base × scale` so every widget inheriting the
   application font re-scales live (the baseline is cached on a QApplication
   dynamic property so repeated `MainWindow` construction never compounds).
3. Regenerate **one** scale-derived QSS block carrying the scaled *chrome* font
   sizes, selected by `objectName` (`#benchSectionHeader`,
   `QHeaderView::section`, `QGroupBox::title`, `QDockWidget::title`).

**What a new widget must do to be scale-safe: usually nothing.** If it inherits
the application font and sizes through the builders/metrics, it re-scales for
free. Two cases need a hook:

- **Fonts QSS cannot reach** — per-item `QTableWidgetItem.setFont` cells or an
  explicit mono panel font. Re-derive the font from the builders inside the
  *owning* widget on the `UIManager.ui_scale_changed` signal (connect in the
  owner, then call the handler once to apply the initial scale). There is no
  central per-widget tracking layer — the owner owns its re-derivation.
- **Chrome fonts set explicitly** — give the widget an `objectName` and add a
  size-only `font-size` rule to the QSS block so the builders' weight/family
  survive.

## Panel anatomy

Build a panel from the shared primitives, not ad-hoc layouts.

**`PanelSection`** is the one titled-section primitive (static or collapsible):

- **Everyday** controls go in static sections (`PanelSection(title)`) that are
  always visible.
- **Advanced** controls go in a collapsible section
  (`PanelSection(title, collapsible=True, expanded=False)`) with **persisted**
  state via `settings_key`. Convention: `"<panel>/sections/<slug>"` (e.g.
  `"alc/sections/calibration"`). A panel-local `_collapsible_group(...)` factory
  that wraps `PanelSection` with this key convention is the sanctioned pattern
  (see `panels/alc_panel.py`).
- Use `set_hint()` for a one-line muted description and `set_title_suffix(html)`
  for a right-aligned collapsed-section summary chip ("3 exclusions").

**`ActionFooter`** is the pinned footer for a panel's primary actions, status,
and progress: `add_primary(verb)` / `add_secondary(...)`, `set_hint(...)`,
`set_status(html)`, `show_progress(...)` (determinate or busy) / `hide_progress`.
The primary action is a **verb first** ("Run fit", not "Fit run"). For fit-tab
run controls reuse `FitRunControls` (the shared Stop/Cancel + progress widget).

**`KeyValueGrid`** renders read-only results (`set_rows([(label, value), …])`).

**Chips and result boxes** come from `gui/styles/widgets.py`:
`fit_quality_chip_html(...)`, `info_html`, `warning_html`, `error_html`,
`success_html`, `make_warning_banner`, `make_context_chip`,
`make_confidence_chip`.

Drive **conditional visibility** from core predicates (a core function returns
whether a control applies to the current dataset/mode). Never re-derive a mode
list or capability check in the GUI — that duplicates core logic and drifts.

## Shared widgets inventory

Extend the shared widget; never fork it. (A duplicate implementation fails the
structural harness.)

- **`FloatLimitField` / `AxisLimitControls`** (`widgets/axis_limits.py`) —
  validated numeric limit fields and axis min/max controls.
- **`create_canvas`** (`widgets/mpl_canvas.py`) — the only sanctioned
  `FigureCanvasQTAgg` factory.
- **`FitRunControls`** (`widgets/fit_run_controls.py`) — shared Stop/Cancel
  button + optional progress bar for fit tabs.
- **`PanelSection`** (`widgets/panel_section.py`) — the one titled/collapsible
  section primitive.
- **`ActionFooter`** (`widgets/action_footer.py`) — pinned footer: buttons,
  hint, status chip, progress.
- **`KeyValueGrid`** (`widgets/key_value_grid.py`) — read-only label/value grid.
- **Chip / result-box builders** (`styles/widgets.py`) — `fit_quality_chip_html`,
  `info/warning/error/success_html`, `make_warning_banner`, `make_*_chip`.
- **`LoadingOverlay`** (`widgets/loading_overlay.py`) — translucent busy overlay
  over one widget while work is in flight.
- **`DockHeader`** (`widgets/dock_header.py`) — BENCH dock title bar
  (`setTitleBarWidget`).
- **Wizard cards / decision trail** — `WizardAnswerCard`, `DecisionTrail`,
  `WizardScopeSelector`, `WizardSeriesCard` (`widgets/`).
- **`WizardWindowBase`** (`windows/wizard_base.py`) — subclass this for a new
  guided-wizard window; it owns the `TaskRunner`, progress UI, staleness,
  cancel/closeEvent, and styled chrome. Do not hand-roll a wizard skeleton.

Sizing/formatting helpers: `metrics` (above), `compile_gle`
(`utils/export.py`), `format_param_label` (`utils/formatting.py`).

## Threading

Never run long work (file I/O, fits, transforms, reconstructions) on the GUI
thread. Use `TaskRunner` in `gui/tasks.py` (or the fit-panel worker pattern)
with a cooperative cancel path, marshal results back as plain objects via queued
signals, hold strong references to live threads, and shut them down in
`closeEvent`. Never construct a bespoke `QThread` in `gui/`, and never connect a
worker signal to a bare lambda/partial that touches widgets — route through a
GUI-thread `QObject` method. See the threading invariants in
[AGENTS.md](AGENTS.md).

## What the harness enforces

`python tools/harness.py structural` fails fast on:

- **No second `*LimitField`** class outside `widgets/axis_limits.py`.
- **No direct `FigureCanvasQTAgg(`** construction outside `widgets/mpl_canvas.py`.
- **No bespoke `QThread(`** in `gui/` outside `tasks.py`.
- **No bespoke `*CollapsibleSection`** class, and no import of the removed
  `collapsible_section` module — use `PanelSection`.
- **No raw hex-colour literal** in `gui/` outside `styles/` (allowlisted
  specialist palettes excepted) — use a `tokens` value.
- **No literal-pixel geometry** (`setFixed/Minimum Width/Height/Size` with an
  int literal >= 24) outside the design-floor allowlist — use `metrics`.
- **Test files** live under a sanctioned `tests/<subpackage>/`.

When a review comment exposes a new rule, encode it here and in
`tools/harness.py` so the next agent discovers it.
