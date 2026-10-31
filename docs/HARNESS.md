# Harness Engineering for Asymmetry

This project applies the harness-engineering pattern described in OpenAI's
"Harness engineering: leveraging Codex in an agent-first world":
https://openai.com/index/harness-engineering/

The practical goal is simple: make the repository itself legible enough that an
agent can understand the product, choose the right files, run the right checks,
and improve the harness whenever it gets stuck.

## What Counts as Harness

Harness code is any repo-local artifact that improves agent reliability:

- `AGENTS.md`, the one binding guide for every agent (`CLAUDE.md` imports it and
  `.github/copilot-instructions.md` points to it, so no rule lives in a
  vendor-specific file; the harness caps its length and checks the import)
- architecture docs that describe boundaries and intended dependencies
- executable checks in `tools/harness.py`
- CI jobs that run the same commands agents run locally
- tests, examples, and smoke checks that turn product intent into feedback

Plans and decisions live in GitHub issues and PR bodies, not here.

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
  via the `Full test suite` workflow. It runs with reduced parallelism (`-n 2`
  and a raised per-test timeout) rather than `-n auto`: the exhaustive
  global-wizard harness test fans its own fits across a `cpu_count - 2` process
  pool, so a worker-per-core layout oversubscribes the runner and stalls
  neighbour fits past the timeout (historically crashing the release-tag run).
  All tiers also pin the numeric backends to one thread each (`*_NUM_THREADS=1`)
  so xdist's process-per-core parallelism isn't multiplied by per-process BLAS
  threads.

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

CI splits the standard tier across six runners with `--shard {1..6}/6`, each
running `-n auto --dist worksteal` on its 4 vCPUs. `--shard K/N` (a
`conftest.py` option) keeps a stable 1-of-N slice partitioned by a hash of each
test's node id, so the independent shard processes cover every test exactly once
with no overlap, regardless of `pytest-randomly` ordering, and together cover
exactly the standard tier. Each shard mixes GUI and non-GUI tests; the hash
keeps per-shard test time within about 25% of even, which is less than the
runner-to-runner variation, so the split needs no recorded durations. Every
shard pays a fixed cost of roughly 1.5 min (checkout, uv install, and test
collection, most of which is pytest rewriting asserts in ~430 test modules on
each xdist worker), so adding shards stops paying off as that fixed cost
dominates. `--subset gui`/`--subset non-gui` still select by marker for local
runs.

The harness sets `QT_QPA_PLATFORM=offscreen` by default for command runs so GUI
tests and smoke checks can execute in CI and local agent sessions. The current
CI validation uses the passing baseline: structural checks, Ruff, and pytest.

When a local `.venv` exists, `tools/harness.py` re-executes itself with that
interpreter. This keeps local agent runs on the project dependency set instead
of the ambient Python environment, which matters for fitting: the project pins
numpy 2.2.x and a working iminuit, and numpy ≥ 2.3 breaks fitting.

A per-test timeout (`--timeout=120`) makes a hung test fail rather than stall
the run.

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

## Retired planning locations

Plans and review decisions belong in GitHub issues. The structural check rejects
new files under `docs/plans/`, `docs/porting/`, `docs/studies/`, `docs/audit/`,
`docs/investigations/`, and at `docs/PLANS.md`. Existing files are frozen in
`tools/baselines/retired_docs.txt`, one strictly sorted repository-relative POSIX
file path per line. Editing an existing file is allowed while its supported
content is rescued into the owning Sphinx page, a regression test, or an invariant
comment.

When deleting a retired file, delete its baseline line in the same change. A stale
entry fails structural checks, and a file reintroduced after removal fails as a
new plan. Emptying the baseline is allowed; no placeholder study or index entry
is required. Hidden files and `__pycache__` directories are ignored.

## Test user-state isolation

The root `qapp` fixture owns QApplication construction; structural checks reject
local `qapp` fixtures and direct application construction elsewhere in tests.
The root fixtures load Qt and run window cleanup only for tests marked `gui`.
A core-only run imports no Qt through fixtures and can run without PySide6.
GUI settings (including system fallbacks), app data, and cached fit times use
per-test temporary locations; plugin discovery is isolated for every test.
The temporary roots are independent between xdist workers.

`tests/tools/test_test_isolation.py` checks a core harness subprocess with Qt
imports blocked. On Linux it also runs a GUI case with its subprocess home
pointing at a regular file, which makes real user-config paths unwritable even
for a privileged test process. Reproduce both checks with
`python tools/harness.py test -- tests/tools/test_test_isolation.py`.
