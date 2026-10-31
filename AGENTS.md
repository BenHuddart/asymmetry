# Asymmetry agent guide

This is the one instruction file for every coding agent and contributor. Claude Code imports
it from `CLAUDE.md`; Codex, Cursor and Copilot read it directly. Its rules are binding; linked
docs give the rationale. `tools/harness.py structural` caps it at 150 lines.

These rules describe the repository we want, not always the one we have. Issues labelled
`polish` bring existing code and docs into line. Every line you add or touch meets the rule,
and an old pattern is never a reason to extend it.

## What Asymmetry is

A public, MIT-licensed Python library, CLI and PySide6 application for muon-spin spectroscopy
(μSR) analysis. Results end up in papers, so correctness and provenance come before features.
Users, instrument scientists and other μSR authors read what you commit, and your issues and PRs.

## Where to read first

| Need | Read |
|---|---|
| Layers, data model, boundaries | `docs/ARCHITECTURE.md` |
| GUI tokens, panels, threading, responsiveness | `docs/GUI_GUIDELINES.md` |
| Test tiers and placement, CI, what the harness checks | `docs/HARNESS.md`, `tests/README.md` |
| Docs upkeep, screenshots, science standard, voice | `docs/README.md`, `docs/STYLE.md` |
| Feature behaviour | `docs/reference/index.rst` and the page it names |
| Changelog and releases | `RELEASING.md` |

## Making a change

1. Find or open the issue. Anything bigger than a small fix starts from one.
2. Branch `feature/*` or `fix/*` off `main`; never commit to `main`. To update an open PR, run
   `gh pr checkout <n>`, confirm with `git branch --show-current`, and push to that branch.
3. Find the owning layer below. A bug fix starts with a test that fails.
4. Change implementation, tests, docs and `CHANGELOG.md` `[Unreleased]` together. A
   user-facing change is incomplete until the owning Sphinx page describes it, quotes UI
   strings verbatim and updates its screenshot scenario.
5. Run focused tests through the harness, never bare pytest, then `validate` once before review.
6. Open a PR with an imperative title, the body from `.github/PULL_REQUEST_TEMPLATE.md`
   (`gh pr create --body` does not apply it) and `Closes #n`.
7. After a PR with user-visible changes merges, propose a release per `RELEASING.md`: suggest
   the version and bump with a one-line reason, then wait. Never cut one unprompted.

## Issues hold the plans

- Plans, roadmaps, studies, audits, decision logs, investigations and evaluation results live in
  GitHub issues. Nothing new goes in `docs/plans/`, `docs/porting/`, `docs/studies/`,
  `docs/audit/`, `docs/investigations/` or `docs/PLANS.md`; those are being retired.
- An issue states the problem, change, tests, docs and decisions taken, enough to start from.
- What outlives the work moves into the repository as an invariant comment, a regression test
  or a paragraph in the owning doc. The reasoning behind a decision goes in the PR body.

## Layers

`asymmetry.core` is pure-Python analysis and never imports Qt, matplotlib or `asymmetry.gui`;
new behaviour goes there first and stays scriptable. `cli/` and `gui/` are thin front ends that
never reimplement analysis. Validate only at the three boundaries (file, project, user input),
producing a typed value nothing downstream re-checks.

## Facilities and instruments

An instrument scientist must be able to check what we assume about their spectrometer without
reading the program, so each instrument has one file in one shape (migrating from
`core/instrument.py` and scattered branches).

- `core/instruments/<facility>_<instrument>.py` holds one frozen `InstrumentProfile`, always in
  this order: identity (facility, beamline, the names its files use), source (pulsed or
  continuous, pulse structure), recognition (metadata, labels, filename tokens), detectors,
  grouping presets, file quirks, defaults, sources, review. It holds data, not logic.
- Every value names its source (manual, paper, file header, measured run) or is marked
  `unverified`. Never guess silently: an unrecognised instrument gets `generic_layout`.
- Code elsewhere asks the profile (`profile.source.pulsed`, `profile.quirks`). It never branches
  on an instrument or facility name. Loaders parse formats, not conventions.
- `docs/reference/instruments/<facility>_<instrument>.rst` is generated from the profile. The
  reviewer checks that page through the *Instrument review* issue form; the profile records the
  version reviewed and the reviewer, whose name is published only with their consent.
- Adding an instrument means one profile and one test taking a sample or synthetic run through
  recognition and default grouping, and nothing else.

## Other μSR programs

WiMDA, musrfit and Mantid are colleagues' work under their own terms, and Asymmetry is MIT.

- Implement from published mathematics, manuals, documented formats and observed behaviour. GPL
  source (musrfit, Mantid) and Numerical Recipes code never enter this repository in any form.
- WiMDA's source is not open source. Read it to understand a method, but never quote or
  translate it. Where existing code follows a routine closely, the docstring names the routine.
  `NOTICE` must stay true.
- Check numbers by running the other program; the test records its version, input and tolerance.
- Comparisons go in the Sphinx docs: `docs/explanation/comparison.rst`, plus a "Compared with other
  programs" section on the owning reference page. Name the version, cite the source, and state
  conventions and agreement neutrally, in terms its authors would accept. No "parity" or roadmaps.

## Science

- Physical constants live in `core/utils/constants.py`, with units in the name and a cited
  source. Never type a value anywhere else.
- Every model, correction and estimator cites its primary source in the docs and has a test
  against an independent value: an analytic limit, a published result or a cross-check.
- Provenance travels with the data: run metadata, grouping, t0, good-bin range, deadtime and
  background assumptions. Stochastic methods take a seed and record it.
- A project saved by any release keeps loading. Renames and format changes happen once, as a
  tested migration in `core/project/schema.py`; runtime code reads one name.

## Lean code

The codebase must stay small enough to read. Briefs for subagents carry this section verbatim.

- **Prevent bad states by construction.** No `hasattr`/`getattr(..., None)` on our own
  attributes, no `try/except` around our own code, no "in progress" latches. Where a bad state
  cannot be made impossible, raise an exception naming the invariant; never fall back silently.
- **One source of truth per fact.** Do not store a derived value beside its source.
- **A private helper needs two callers or a domain name.** Inline the rest; no two-line wrappers.
- **Generalise, don't mirror.** Prefer a table of cases walked by one function to parallel bodies.
- **Reuse the `gui/widgets/` and `gui/utils/` foundations;** `docs/GUI_GUIDELINES.md` lists them.
- **Never run long work on the GUI thread.** Use `TaskRunner` with a cooperative cancel path, and
  never connect a worker signal to a bare lambda that touches widgets.
- **Small modules.** A new module stays under 1,000 lines; one already over may only shrink.
- **Delete, don't deprecate.** Rewrite tests that pin old behaviour; leave no flagged-off paths.
- **Comments state the invariant.** Over three lines, one must carry a physics, units or
  coordinate argument. No history, phases, decision numbers or plan-file links.
- **Encode, don't repeat.** A rule that draws a repeated review comment becomes a check in
  `tools/harness.py structural`, as a ratchet that freezes today's count, not as a line here.

## Public repository

- Never commit credentials, absolute paths, personal notes, session hand-offs, agent transcripts
  or scorecards, one-off probes, or data whose owner has not agreed to publication.
- `docs/` is a contract, not a changelog: current behaviour only, no "previously", no dates, no
  migration narrative. Published pages never point at repository internals.
- A `CHANGELOG.md` entry names the user-visible change in one or two lines; the detail belongs in
  the docs. Describe what changed for users, not which agent or model found it.

## Test corpus

Real-data tests, agent evaluations and corpus screenshots read the WiMDA muon-school corpus,
which is never copied here. Maintainers extract the private `BenHuddart/wimda-muon-school-corpus`
release and set `ASYMMETRY_CORPUS_ROOT`, the only way code finds it (`tests/corpus.py`); tests
skip when it is unset. Never hard-code its path or commit data or results from it.

## Commands

```bash
python tools/harness.py structural                       # repository-shape checks
python tools/harness.py lint                             # ruff format --check and ruff check
python tools/harness.py test -- tests/core/test_x.py     # focused: while iterating
python tools/harness.py test --tier fast                 # non-GUI, ~25 s: after a core change
python tools/harness.py validate                         # standard tier, ~2 min: once per task
python tools/harness.py gui-smoke                        # GUI startup or packaging work
python tools/harness.py docs                             # documentation changes
```
