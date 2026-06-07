# CLAUDE.md

Guidance for Claude Code when working in this repository.

The shared, tool-agnostic agent map lives in `AGENTS.md` (repository shape,
engineering invariants, study-first porting workflow). Claude Code does not read
`AGENTS.md` natively, so it is imported here. Keep durable, cross-tool guidance in
`AGENTS.md`; keep only Claude-specific notes below.

@AGENTS.md

## Fixing a pull request — always work on the PR's branch

When asked to fix, update, or address CI failures / feedback on an **existing pull
request**, the change MUST land on that PR's branch — never on `main`.

Before making any commits:

1. Check out the PR's branch: `gh pr checkout <PR-number>` (or `git checkout <branch>`).
2. Confirm you are on it: `git branch --show-current` — it must show the PR's
   branch, not `main`.
3. Only then make changes; commit and push to that same branch to update the PR.

If you are on `main` (or any branch other than the PR's), stop and switch before
committing. Do not open a new branch for an existing PR unless explicitly asked.

For brand-new work (not tied to an existing PR), create a feature branch off `main`,
push it, and open a PR — don't commit directly to `main`.

## Local development

- Use the project virtualenv: `.venv/bin/python`. It pins numpy 2.2.x and a working
  iminuit; the system Python has numpy ≥ 2.3, which breaks fitting.
- CI runs `python tools/harness.py validate` (lint + structural checks + the full
  pytest suite). The harness parallelizes the suite with `-n auto --dist load`, so
  a full local `validate` completes in **~2 min**. (Historically it took 30–60 min:
  GUI tests created a `MainWindow` per test without destroying it, and because
  `deleteLater` is not dispatched without forcing `DeferredDelete`, leaked widgets
  accumulated and `MainWindow` setup degraded to O(n²). The autouse
  `_cleanup_qt_widgets` fixture in `tests/conftest.py` fixes this.)
- A per-test timeout (`--timeout=120`) is set, so a genuinely hung test fails fast
  rather than stalling the whole run.
- GUI tests need `QT_QPA_PLATFORM=offscreen` in headless environments.
