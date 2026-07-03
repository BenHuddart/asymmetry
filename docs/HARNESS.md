# Harness Engineering for Asymmetry

This project applies the harness-engineering pattern described in OpenAI's
"Harness engineering: leveraging Codex in an agent-first world":
https://openai.com/index/harness-engineering/

The practical goal is simple: make the repository itself legible enough that an
agent can understand the product, choose the right files, run the right checks,
and improve the harness whenever it gets stuck.

## What Counts as Harness

Harness code is any repo-local artifact that improves agent reliability:

- concise maps such as `AGENTS.md` and `docs/INDEX.md`
- policy docs such as `docs/porting/README.md` that define repeatable agent workflows
- architecture docs that describe boundaries and intended dependencies
- executable checks in `tools/harness.py`
- CI jobs that run the same commands agents run locally
- tests, examples, and smoke checks that turn product intent into feedback
- quality notes and execution plans that preserve design decisions over time

## Validation Ladder

Run the smallest useful check first. Move upward when a change crosses a
boundary, touches shared behavior, or changes user-visible workflows.

| Command | Purpose |
| --- | --- |
| `python tools/harness.py structural` | Fast repository-shape checks, including the core/GUI dependency boundary. |
| `python tools/harness.py lint` | Ruff format check and lint check across `src`, `tests`, and `tools`. |
| `python tools/harness.py lint-all` | Alias for the full Ruff baseline. |
| `python tools/harness.py test -- tests/<layer>/test_name.py` | Focused pytest run while iterating on a specific behavior. |
| `python tools/harness.py test --tier fast` | Inner-loop suite: non-GUI tests only (~25s). |
| `python tools/harness.py test` | Standard tier (default): everything except `slow`/`integration` (~2 min locally). |
| `python tools/harness.py gui-smoke` | Headless GUI startup check using the app's `--smoke-test` path. |
| `python tools/harness.py docs` | Sphinx documentation build. |
| `python tools/harness.py validate` | Structural checks, lint, and the standard-tier test suite. |

### Test tiers

`test` and `validate` accept `--tier {fast,standard,full}`:

- **`fast`** — pure-Python tests only (no Qt, no file I/O, nothing `slow`).
  Over half the suite; runs in ~25s with xdist. Use this for the **inner dev
  loop**.
- **`standard`** (default) — everything except `slow` and `integration`. This is
  the pre-push gate and what CI runs. The GUI tests it adds each construct
  widgets (often a full `MainWindow`), so it takes ~2 min locally — run it once
  per task, not per iteration.
- **`full`** — every test, including `slow`/`integration`. Runs on release tags
  via the `Full test suite` workflow.

Tests inherit a type marker automatically: anything not explicitly marked `gui`
or `io` is treated as `unit` (see `tests/conftest.py`). Naming explicit targets
(`-- tests/<layer>/test_x.py` or a `::node-id`) runs exactly those, bypassing
the tier filter.

`tests/` mirrors the `src/asymmetry/` layer boundaries — `tests/core/`,
`tests/gui/`, `tests/io/`, `tests/project/`, `tests/tools/`, plus
`tests/integration/` for genuinely cross-cutting end-to-end tests, and the
already-organized `tests/negmu/`, `tests/docs/`, `tests/porting/`. See
`tests/README.md` for the placement rules. Tiering and sharding are driven by
pytest markers and node-id hashes, not by directory, so this layout is
transparent to `--tier`/`--subset`/`--shard` and to CI's path filters (which
match `tests/**` recursively).

CI shards the standard tier across four runners: the fast non-GUI tests run as a
single shard (`--subset non-gui`), and the much heavier GUI tests — which carry
the per-test `MainWindow` cost and dominate wall-clock on a 2-core runner — are
split three ways with `--subset gui --shard {1,2,3}/3`. `--shard K/N` (a
`conftest.py` option) keeps a stable 1-of-N slice partitioned by a hash of each
test's node id, so the independent shard processes cover every test exactly once
with no overlap, regardless of `pytest-randomly` ordering. The four shards
together cover exactly the standard tier.

The harness sets `QT_QPA_PLATFORM=offscreen` by default for command runs so GUI
tests and smoke checks can execute in CI and local agent sessions. The current
CI validation uses the passing baseline: structural checks, Ruff, and pytest.

When a local `.venv` exists, `tools/harness.py` re-executes itself with that
interpreter. This keeps local agent runs on the project dependency set instead
of the ambient Python environment.

## Agent Workflow

1. Read `AGENTS.md`, then open only the docs relevant to the task.
2. Identify the affected boundary: loader, transform, fitting, project schema,
   GUI panel, packaging, or documentation.
3. If the task is a feature port from WiMDA, musrfit, Mantid, or another
  reference program, complete the study pass in `docs/porting/<feature-slug>/`
  before implementation. Record entry points, data flow, dependencies, edge
  cases, test coverage, implementation differences, port seams, and comparison
  data in stable, machine-readable paths.
3. Write or update the smallest test or structural check that captures the
   desired behavior.
4. Implement the change in the layer that owns the behavior.
5. Run focused checks while iterating, then `python tools/harness.py validate`
   before handing work back when feasible.
6. If an issue repeats, add a harness rule or documentation note instead of
   relying on future memory.

## Porting Workflow

Feature ports are two-pass tasks.

### Study pass

Do not implement the feature yet. Create the study scaffold described in
`docs/porting/README.md` and compare the reference implementations first.

Required artifacts for each study:

- `docs/porting/<feature-slug>/README.md`
- `docs/porting/<feature-slug>/comparison.md`
- `docs/porting/<feature-slug>/implementation-options.md`
- `docs/porting/<feature-slug>/test-data.md`
- `docs/porting/<feature-slug>/verification-plan.md`
- `docs/porting/index.json`

Optional study-pass scaffolding:

- `tests/porting/<feature-slug>/`
- `src/porting/<feature-slug>/`

### Implementation pass

Implement only after the study exists and the approach is chosen. Use the study
docs as the source of truth, add verification tests against the reference
programs, and update the study with the final decision and comparison outcome.

## Boundaries Worth Encoding

- `asymmetry.core` remains independent of PySide6, matplotlib, and
  `asymmetry.gui`.
- GUI modules may depend on core modules, but they should not duplicate core
  analysis algorithms.
- File loaders should convert external formats into `MuonDataset`, `Run`, and
  `Histogram` objects with provenance attached.
- Project persistence should evolve through explicit schema handling in
  `asymmetry.core.project.schema`.
- Optional dependencies should be imported at the boundary that needs them and
  guarded with clear errors or skipped tests.

## When to Expand the Harness

Add a new harness check when:

- a bug would have been caught by a mechanical repository-shape rule
- an architectural boundary has become important enough to enforce
- a manual QA step can be made repeatable with a script, test, or smoke check
- a reviewer has to explain the same project convention more than once

Keep rules high signal. A harness that fails noisily for incidental style drift
will stop being trusted; a harness that encodes real project invariants will
compound.
