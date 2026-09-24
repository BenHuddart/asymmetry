# Agent CLI/skill evaluation rubrics

These rubrics score the summary an agent (Claude Code or Codex, driving the
`asymmetry` CLI) produces when handed a directory of muSR run files
from the WiMDA muon school corpus and the prompt:

> "This directory contains the data from a recent muSR experiment. Can
> you analyse these using Asymmetry and present me a summary of what
> they show?"

The corpus lives at `~/Documents/WiMDA muon school/` and is never
copied into the repository. Each rubric file documents, from its
worksheet (or paper, where no worksheet exists), what a correct
summary should and should not say — checkable from the summary text
alone, without needing to see the agent's tool calls or the raw data.

## Tiers

- **Tier A** (`ferromagnetic-nickel.md`, `fmuf-ptfe.md`,
  `spin-glass-ymnal.md`, `high-tc-cuprate.md`): must pass in full
  before a PR implementing this phase opens. Each has a clean
  worksheet, a documented calibration story (or documented absence of
  one), and a clear expected physical finding.
- **Tier B** (`copper-diffusion.md`, `spin-peierls.md`,
  `molecular-antiferromagnet.md`, `euo-psi.md`, `afm-high-tf-mdu.md`):
  run once per pass and recorded in the plan's evaluation log, but a miss
  does not block the phase. These add multi-instrument folders,
  non-worksheet (paper) sources, and non-ISIS loaders (PSI `.bin` and
  `.mdu`) to the mix. `afm-high-tf-mdu.md` is a *bounded* analysis: it
  also checks that the agent names the steps of the paper it cannot do.
- **Tier C** (none at present): the correct behaviour is to decline —
  explain that the requested workflow is out of scope for the current
  tool and stop, rather than force an answer. A decline rubric checks
  that the agent recognised *why* the case is out of scope, not just
  that it refused. `afm-high-tf-mdu.md` was the Tier C case until the
  2026-09-23 corpus audit showed the CLI can do most of it.
- **Workflow-expansion gate** (`alc-tcnq.md`, `ionic-motion-llz.md`,
  `photo-musr-silicon.md`, `cds-fourier.md`): must pass when adding or changing
  the period-selection, integral-scan, Fourier or coupled-group workflows.
  These cases were chosen specifically because an earlier CLI could not carry
  them out. They grade a representative new workflow; where batching is still
  absent, the rubric requires the agent to state that boundary.

## How to tick a rubric

1. Read the agent's final summary only — not its tool transcript,
   thinking, or intermediate output. The summary is what a real user
   would see.
2. Go through the **Must** list first. Any unchecked Must line is an
   automatic fail for that dataset. Must lines are written so a human
   can answer yes/no from the summary text alone, without needing the
   worksheet open or any tool output in hand.
3. Go through the **Should** list. Unchecked Should lines are noted in
   the evaluation log as gaps to improve the skill/prompt, but do not
   fail the pass on their own.
4. Read **Known traps** last, as a checklist of specific ways a
   plausible-sounding summary goes wrong for this dataset (wrong
   grouping, invented calibration, a number lifted from the worksheet
   or a cited paper rather than from a tool call). If the summary
   falls into one of these, mark the relevant Must/Should line as
   failed and note which trap it was in the evaluation log.
5. For Tier C datasets, use the file's own decline rubric instead of
   a Tier A/B-style Must/Should pair for physical findings — the
   correct summary here contains no analysis results at all, only a
   clear, specific explanation of scope and (for the "should" lines)
   useful survey-level context.
6. For a workflow-expansion case, verify from `commands.txt` that the named new
   command/flag was really used. The final summary remains the scored artifact;
   this extra check distinguishes a capable-sounding narrative from an actual
   exercise of the new path.

## The number rule

**A summary that quotes any numeric value — a temperature, field,
frequency, rate, exponent, run count, or run number — that is not
present in this session's actual tool output is an automatic fail**,
regardless of how many Must lines it otherwise satisfies. This
includes:

- Textbook or literature values quoted in a worksheet's background
  section (e.g. nickel's bulk Tc, EuO's cited Curie temperature) that
  the agent presents as if they were measured or fit from the data in
  this run.
- Worksheet run numbers, ranges, or table entries that were not
  actually present in, or confirmed by, the data folder the agent was
  given (several datasets in this corpus have worksheets that list
  runs beyond what is on disk, or omit runs that are on disk — see
  each file's "Known traps").
- Any fit parameter, transition temperature, or rate the agent did not
  itself obtain via a tool call in the session being scored.
- Arithmetic on printed values presented as a result: a percentage change,
  a ratio, a difference between two printed columns, a unit conversion, or a
  significance in σ that no command printed (made explicit 2026-09-24; the
  rule was already applied this way). Quoting the two printed values and
  describing the relation in words is fine.

A summary may still *discuss* such numbers in words (e.g. "the
worksheet's textbook Tc is far higher than the temperatures probed
here") without failing this rule, provided it is clear the number is
being quoted from the source material and not claimed as this
session's own result.
