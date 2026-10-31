<!--
Title: an imperative sentence naming the change. The squash-merge title becomes the commit
message on main.

Agents: `gh pr create --body` does NOT apply this template; GitHub only injects it into the
web form. Read this file and follow it when writing the body.

For small fixes, Why / What changed / Gates is enough. Delete empty sections rather than
padding them. This repository is public: no private data, local paths or transcripts.
-->

Closes #

## Why

<!-- The problem, with evidence: an observed failure, measured numbers, a run that shows it. -->

## What changed

<!-- Group by area with bold lead-ins, not a file list. Say what each piece does. -->

## Reviewer notes

<!-- Things to weigh, not just read: decisions and the alternatives rejected, deviations from
     the issue, known limitations, adjacent problems deliberately not touched. -->

## Verification

<!-- What the new tests pin, not just that they exist. Name any check against real data, a
     published value or another program (with its version). -->

## Gates

- `python tools/harness.py validate`: <N> passed
- `python tools/harness.py docs`: clean (if docs changed)
- Owning docs page and screenshot scenario updated (if user-facing)
- `CHANGELOG.md` `[Unreleased]` updated (if user-visible)
- Migration in `core/project/schema.py` with a test (if the `.asymp` format changed)
