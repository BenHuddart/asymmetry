# Execution Plans

This file tracks repo-local plans that are useful for agents. Keep short tasks
inline here; create a dedicated plan file only when a project spans multiple
subsystems or days.

## Active

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
