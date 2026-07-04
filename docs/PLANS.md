# Execution Plans

This file tracks repo-local plans that are useful for agents. Keep short tasks
inline here; create a dedicated plan file only when a project spans multiple
subsystems or days.

## Active

### Fit Function Builder Redesign (Option C)

Status: in progress (branch `feat/fit-builder-redesign`, single PR)

Replace the calculator-style expression dialogs with a two-panel builder:
searchable component library (left) + structured model-row list (right), with
fraction groups as visual containers and an "Edit as text" escape hatch.
Both `FitFunctionBuilderDialog` and `ParameterModelBuilderDialog` build on one
shared base in `gui/widgets/function_builder/`.

Core rework included: fraction groups expose n−1 free parameters named
`f_<Component>` (last term = remainder `max(0, 1−Σ)` computed in
`CompositeModel`); legacy `fraction_i` parameter values are migrated on
project load. Deep component search (name/alias/category/param/description,
ranked) lives GUI-free in `core/fitting/component_search.py`.

Acceptance criteria:

- Both builder dialogs use the shared library + row-list base; the old
  free-text-primary editor and text-selection Fractions gesture are gone.
- Fraction groups survive structural edits; the displayed model is never
  ambiguous (no hidden parentheses).
- Pre-rework `.asymp` projects load, display, and refit with migrated
  fraction values.
- `harness validate` passes; docs updated with GUI walkthrough screenshots.

### Harness Adoption

Status: in progress

- Maintain `AGENTS.md` as the short navigation map.
- Keep `docs/HARNESS.md`, `docs/QUALITY.md`, and this file indexed from
  `docs/INDEX.md`.
- Run `python tools/harness.py structural` for fast boundary validation.
- Run `python tools/harness.py validate` before larger handoffs when local
  dependencies are available.

Acceptance criteria:

- Agents can discover architecture, validation commands, and current quality
  risks from repo-local files.
- CI runs the same harness command used locally for structural, lint, and test
  validation.
- New repeated conventions are promoted into checks or docs.

## Deferred

### GUI Journey Harness

Build a repeatable smoke workflow for the most important GUI paths:

- launch the app in offscreen mode
- load a small synthetic or fixture-backed dataset
- open grouping, fitting, Fourier, and project-save paths
- capture failures with enough context for an agent to reproduce locally

### Documentation Freshness Checks

Consider a structural check that verifies `docs/INDEX.md` references all
top-level developer docs and that public API docs build without warnings.

### Ruff Baseline Cleanup

Status: complete

`python tools/harness.py lint` now checks `src`, `tests`, and `tools`, and
`lint-all` remains as an alias for that full baseline.

### Loader Fixture Catalog

Create a small fixture index documenting which file-format assumptions are
covered by tests and which rely on synthetic data.
