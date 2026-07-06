# Asymmetry Agent Map

This file is the short entry point for coding agents. Keep detailed guidance in
repo-local docs and use this file as the map.

## Start Here

- [README.md](README.md): user-facing overview, install commands, and common workflows.
- [CONTRIBUTING.md](CONTRIBUTING.md): contributor setup and review expectations.
- [docs/INDEX.md](docs/INDEX.md): documentation table of contents.
- [benhuddart.github.io/asymmetry](https://benhuddart.github.io/asymmetry/): published documentation site.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): system design, domain boundaries, and feature specs.
- [docs/GUI_GUIDELINES.md](docs/GUI_GUIDELINES.md): how to build GUI here — tokens, fonts, metrics, PanelSection/ActionFooter, the UI zoom, and the harness rules.
- [docs/HARNESS.md](docs/HARNESS.md): agent harness workflow and validation ladder.
- [docs/porting/README.md](docs/porting/README.md): mandatory study-first workflow for feature ports.
- [docs/QUALITY.md](docs/QUALITY.md): quality map and current risk areas.
- [docs/PLANS.md](docs/PLANS.md): active and deferred execution plans.

## Repository Shape

- `src/asymmetry/core/`: pure-Python analysis engine. This layer must not import
  Qt, matplotlib, or `asymmetry.gui`.
- `src/asymmetry/gui/`: PySide6 desktop application and plotting surfaces. GUI
  code wraps the core API instead of reimplementing analysis behavior.
  - `gui/panels/fit/`: the fit-setup-and-results package (`FitPanel`,
    `FitTabBase`, `SingleFitTab`, `GlobalFitTab`, `FitParameterTable`, plus
    seeding helpers) — see `docs/ARCHITECTURE.md` for the module map.
  - New axis/limit fields, Matplotlib canvases, GLE export code, parameter
    label formatting, or fit-tab run controls must reuse the shared
    `gui/widgets/` and `gui/utils/` foundations (`FloatLimitField` /
    `AxisLimitControls`, `create_canvas`, `compile_gle`,
    `format_param_label`, `FitRunControls`) rather than re-rolling a new
    implementation. New guided-wizard windows should subclass
    `gui/windows/wizard_base.py::WizardWindowBase` instead of hand-rolling a
    `TaskRunner` + progress-UI skeleton.
- `src/asymmetry/core/io/`: loader registry and file-format adapters.
- `src/asymmetry/core/fitting/`: fitting engines, built-in models, and wizard
  recommendation logic.
- `src/asymmetry/core/project/`: serialized `.asymp` project schema.
- `tests/`: pytest coverage, organized by layer under `tests/{core,gui,io,
  project,tools,integration}/` (a new test file belongs beside its layer's
  existing tests, mirroring the feature's module path) — see `tests/README.md`.
- `tools/harness.py`: executable validation harness for agents and CI.

## Engineering Invariants

- Keep the core scriptable and GUI-free. Any new analysis behavior belongs in
  `asymmetry.core` first, with GUI panels calling into it.
- Parse and validate data at file, project, or user-input boundaries. Do not
  build behavior on guessed shapes when a schema or typed object can represent
  the contract.
- Preserve experiment provenance. Loaders and transforms should keep run
  metadata, detector grouping, t0, good-bin ranges, deadtime, and background
  assumptions explicit.
- Feature ports from reference programs must start with a study pass in
  `docs/porting/<feature-slug>/` before implementation begins.
- Prefer small, focused changes with tests beside the behavior they protect.
- When a repeated review comment exposes a rule, encode it in
  `tools/harness.py`, tests, or docs so future agents can discover it.
- Never run long work (file I/O, fits, transforms, reconstructions) on the
  GUI thread. Use the worker machinery in `src/asymmetry/gui/tasks.py` (or
  the fit-panel worker pattern) with a cooperative cancel path, and marshal
  results back as plain objects via queued signals. Never connect a worker
  signal to a bare lambda/partial that touches widgets — with no receiver
  QObject the slot runs on the worker thread; route through a GUI-thread
  QObject method instead. Hold strong references to live threads, and shut
  them down in `closeEvent`.
- When adding datasets to the browser in a loop, wrap the loop in
  `DataBrowserPanel.batch_updates()` — per-add table rebuilds are O(n²).
- A user-facing change is not complete until the Sphinx docs describe it: in
  the same PR, update the owning page (find it via `reference/index.rst` or
  `grep -rl <feature> docs/`), quote UI strings verbatim from the widget
  code, and update/add the screenshot scenario when the UI changed visibly
  (CI regenerates every screenshot on merge to main, so a stale scenario is
  a stale published image). Full rules — scenario determinism, size budget,
  the science/citation standard, the docs verification ladder — live in
  `docs/README.md` § "Maintaining the documentation"; the writing voice in
  `docs/STYLE.md`.
- `tools/harness.py structural` enforces the shared-foundations rules
  mechanically, so a duplicate/bespoke implementation fails fast rather than
  drifting: no second `*LimitField` class outside
  `gui/widgets/axis_limits.py`, no direct `FigureCanvasQTAgg(` construction
  outside `gui/widgets/mpl_canvas.py`, no bespoke `QThread(` construction in
  `gui/` outside `gui/tasks.py`'s `TaskRunner`, no `test_*.py` file outside
  a sanctioned `tests/<subpackage>/` (see `tests/README.md`), and no docs
  screenshot drift (every image referenced from the .rst sources maps to a
  registered scenario and vice versa, within the per-image size budget).

## Validation Ladder

Use the smallest check that answers the question, then climb when the blast
radius grows.

In this workspace, use the project virtual environment. The harness will
re-execute itself with `.venv/bin/python` when that interpreter exists.

```bash
python tools/harness.py structural
python tools/harness.py lint
python tools/harness.py test -- tests/core/test_transforms.py   # focused, exact targets
python tools/harness.py test --tier fast                   # inner loop, non-GUI (~25s)
python tools/harness.py validate                           # standard tier, pre-push gate (~2 min)
```

`lint` and `lint-all` both run the full Ruff baseline across `src`, `tests`, and
`tools`.

### Using the test suite efficiently

The standard tier is ~4,000 tests and takes ~2 minutes locally; treat it as a
once-per-task gate, never an iteration loop.

1. **While iterating**, run exactly the tests beside the behavior you are
   changing: `python tools/harness.py test -- tests/<layer>/test_x.py` (or a
   `::node-id`). Test files mirror feature names and live under `tests/core/`,
   `tests/gui/`, `tests/io/`, `tests/project/`, `tests/tools/`, or
   `tests/integration/` (see `tests/README.md`); find coverage with
   `grep -rl "<symbol>" tests/`.
2. **After a core/non-GUI change**: `python tools/harness.py test --tier fast`
   (~25s, non-GUI).
3. **After a GUI change**: additionally run the affected GUI test files
   focused. Do not run the whole GUI subset locally while iterating — it is
   most of the suite's wall-clock, and CI shards it across three runners.
4. **Once, before committing / handing back**: `python tools/harness.py
   validate`. Do not re-run it for doc- or comment-only follow-up edits, and
   do not run `--tier full` locally unless you changed a `slow`/`integration`
   test.
5. Always go through `tools/harness.py` rather than bare pytest — it re-execs
   into `.venv` and sets `QT_QPA_PLATFORM=offscreen`. Output is quiet by
   default; pass `-v` only when you need per-test lines.

See [docs/HARNESS.md](docs/HARNESS.md#test-tiers) for the full tier breakdown.

For GUI startup or packaging work:

```bash
python tools/harness.py gui-smoke
```

For documentation-only changes:

```bash
python tools/harness.py docs
```
