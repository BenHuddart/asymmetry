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
| `python tools/harness.py test -- tests/test_name.py` | Focused pytest run while iterating on a specific behavior. |
| `python tools/harness.py test` | Full pytest suite. |
| `python tools/harness.py gui-smoke` | Headless GUI startup check using the app's `--smoke-test` path. |
| `python tools/harness.py docs` | Sphinx documentation build. |
| `python tools/harness.py validate` | Structural checks, lint, and the full test suite. |

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
3. Write or update the smallest test or structural check that captures the
   desired behavior.
4. Implement the change in the layer that owns the behavior.
5. Run focused checks while iterating, then `python tools/harness.py validate`
   before handing work back when feasible.
6. If an issue repeats, add a harness rule or documentation note instead of
   relying on future memory.

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
