# Scoring brief for an evaluation wave

The brief handed to a scoring agent (or followed by a person) for the runs of
one wave (`run_wave.py`). It keeps scoring consistent across passes: every
2026-09-23/24 pass in `docs/plans/agent-cli-skill.md` was scored with it.

Give the scorer this file plus: the run directories, a line or two of the
physics each dataset is expected to show (its rubric's "Experiment" section is
the source), and what changed since the last pass so it can say whether the
change was used.

## What a run directory holds

- `summary.md` — the agent's final reply: **the only thing the rubric scores.**
- `rubric.md` — the rubric (Must / Should / Known traps). The rules are in
  `rubrics/README.md` beside this file; read its "How to tick" and "The number
  rule" first.
- `commands.txt` — every command the agent ran; `transcript.jsonl` — the full
  stream (large: grep it, do not read it whole).
- `project/asymmetry-work/` — the agent's work directory, including
  `cli-output.log`, everything its commands printed (the `data/` and
  `workdir/` copies are deleted after a wave to save disk).
- `cost.json` — wall time, cost, exit code, `permission_denials`.

## For each run

1. Read `summary.md` and `rubric.md`. Tick every Must and Should (yes/no, with
   a one-line justification quoting the summary).
2. **Number rule.** For each number that matters (fitted values, T_c, rates,
   alpha, run ranges), confirm a command printed it (grep `cli-output.log`,
   `commands.txt` or the transcript). A number no command printed, presented as
   a result, fails the run — that includes arithmetic on printed values
   (percentages, ratios, differences of printed columns, unit conversions,
   σ-significance). A rounded printed value, or a printed input restated in
   other units in passing, is borderline: note it, do not fail on it alone.
   Background or literature numbers quoted as such are fine.
3. **Verdict.** PASS only if every Must is ticked and the number rule holds.
   Mark a fail that rests on a judgement call as such. A run cut short by
   infrastructure (an API outage, a harness error) is **void**, not a fail.
4. For every fail, and any notable Should gap, find the **root cause** in
   `commands.txt` and the transcript: which decision went wrong, what the agent
   saw at that point (quote the command and the output line that misled it),
   and whether the fix belongs in (a) the skill text, (b) CLI or core
   behaviour, or (c) the rubric being wrong or too strict. Prefer causes that
   would generalise to other datasets over dataset-specific ones.
5. Also report the number of wizard calls, which `trend --model` laws were
   fitted, and any permission denials.

Do not modify anything. Return: a verdict table (run | verdict | failed Musts);
a root-cause paragraph per failure (≤150 words); the top three generalisable
fixes, ranked; any rubric Must that looks wrong or unclear; and, when asked, a
per-dataset reliability estimate across recent passes.

## What the loop learned about fixes

Changes to the CLI that put evidence or the next step in front of the agent —
a line the survey measured, a recipe written for it, a law named for its trend,
an `audit` of the draft — changed behaviour. Guidance added to the skill text
alone rarely did across repeats. Weigh a proposed fix accordingly.
