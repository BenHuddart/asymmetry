# CLAUDE.md

Guidance for Claude Code when working in this repository.

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
- CI's "Harness validation" job runs `python tools/harness.py validate` (lint +
  structural checks + the full pytest suite). The full suite is **slow — roughly
  30–60 min** because the GUI tests in `tests/test_mainwindow_additional.py` are
  serialized onto a single `xdist` worker (`--dist loadfile`). A long-running
  validate or a CI check "pending" for that long is normal, not a hang.
- GUI tests need `QT_QPA_PLATFORM=offscreen` in headless environments.
