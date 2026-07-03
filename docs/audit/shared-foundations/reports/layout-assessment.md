# Repository Layout & Indexing Assessment (exploration report, 2026-07-03, main @ 3d2359a)

Produced by a read-only exploration pass; feeds PLAN.md.

## Overall: A− — structurally agent-ready; friction is in test discovery and large-file internals.

## Top-level docs

README.md, AGENTS.md, CLAUDE.md, CONTRIBUTING.md, docs/INDEX.md,
docs/ARCHITECTURE.md (577 lines), docs/HARNESS.md, docs/QUALITY.md,
docs/PLANS.md — all present, current, correctly cross-linked. No broken links
or staleness found.

## Module inventory

`src/asymmetry/core/` (173 files): `data/` (3), `io/` (11+), `fitting/` (35+),
`fourier/` (9), `transform/` (14), `representation/` (10), `project/`,
`utils/` (3, focused — no grab-bags), `maxent/` (8), `negmu/` (10+), plus
`instrument.py` (66K) and `simulate.py` (69K) at core level.

`src/asymmetry/gui/`: `panels/` (19 files), `windows/` (13), `widgets/` (14+),
`styles/`, `utils/`, `mainwindow.py` (13,032 lines), `app.py`, `tasks.py`,
`ui_manager.py`, `resources/`.

## Gaps found

| Issue | Severity |
|---|---|
| 217 `test_*.py` files flat in `tests/`, no naming/placement convention; tiering by marker only | **Medium** |
| No internal roadmap for >6k-line files (mainwindow, fit_panel, plot_panel, fit_parameters_panel) | **Medium** |
| ARCHITECTURE.md (~line 95) mentions `dialogs/` but classes live in `windows/` | Low |
| `core/negmu/` and `core/maxent/` not status-tracked in QUALITY.md or PLANS.md | Low |
| `core/transform/` referenced but not enumerated in ARCHITECTURE | Low |
| porting `index.json` (54 entries, statuses study/implemented/candidate) checks file presence but not content sections | Low |
| `examples/` referenced by README but not indexed | Low |
| TODO/FIXME grep returned 0 hits (clean, or check invalid) | info |

## Tests

- Root `tests/conftest.py` (10.8K): session-scoped QApplication, autouse
  `_cleanup_qt_widgets` (flushes DeferredDelete — prevents O(n²) MainWindow
  cost), unsaved-changes suppression.
- Sharding: `--shard K/N`, hash-stable by node id; CI = 1 non-GUI + 3 GUI
  shards. Per-test timeout 120s. Coverage ~71%.

## Harness (tools/harness.py)

`structural` (core/GUI boundary, knowledge files, porting artifacts), `lint`
(full Ruff baseline), `test --tier {fast,standard,full}` (+ `--subset`,
`--shard`), `gui-smoke`, `docs`, `validate` (structural+lint+standard). Auto
re-exec with `.venv/bin/python`. All documented in docs/HARNESS.md.

## Strengths to preserve

Clear discovery path (AGENTS.md → ARCHITECTURE → HARNESS); executable
validation ladder; formalized porting workflow with machine-readable index;
boundary enforcement in `structural`; test tiers; no grab-bag modules.
