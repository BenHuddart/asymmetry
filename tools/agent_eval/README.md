# Agent evaluation harness

`run_eval.py` measures whether an agent, handed a folder of μSR runs and
nothing else, produces a defensible analysis with the `asymmetry` CLI and the
packaged `asymmetry-analysis` skill. It is the loop Phase 4 of
[`docs/plans/agent-cli-skill.md`](../../docs/plans/agent-cli-skill.md) runs:
run the agent, score its summary against the dataset's rubric, and fix **the
skill text** — never the rubric, never the CLI.

## Running one evaluation

```bash
python tools/agent_eval/run_eval.py \
    --data "$HOME/Documents/WiMDA muon school/Nuclear magnetism and ionic motion/The FmuF state in PTFE/Data" \
    --rubric fmuf-ptfe \
    --out /tmp/evals/pass1-fmuf-ptfe
```

Options: `--model` (default `sonnet`), `--prompt` (default is the plan's fixed
sentence), `--max-turns` (default 80), `--claude` (path to the Claude Code
CLI).

What it does, in order:

1. Copies `--data` to `<out>/data/`. **The corpus is never written to** —
   every work directory, plot and cached spectrum lands in the copy.
2. Installs the packaged skill into `<out>/data/.claude/skills/` with
   `asymmetry skill install --agent claude --project`.
3. Runs `claude -p` in that directory with the project `.venv/bin` first on
   `PATH`, the tool set limited to `Bash(asymmetry:*)`, `Bash(ls:*)`,
   `Bash(cat:*)`, `Read`, `Glob`, `Grep`, `Write` and `Skill`, and the stream
   saved as it arrives.
4. Writes the outputs below and prints the rubric to tick.

`Skill` must stay in the tool list: without it the skill is discovered at
startup and can never be read, and the run measures nothing. The Bash
allow-list deliberately has no interpreter — an agent that can run Python can
compute a number the CLI never printed, which is exactly what the rubrics'
number rule is there to catch.

## Outputs

Everything lands under `--out`, and nothing outside it is touched:

| Path | What it is |
|---|---|
| `summary.md` | the agent's final message — **the only thing the rubric scores** |
| `commands.txt` | every Bash command the agent ran, in order |
| `transcript.jsonl` | the raw `stream-json` event stream |
| `cost.json` | wall time, turns, cost, permission denials, whether the skill was invoked |
| `workdir/` | the `.asymmetry/` work directory the agent built, plots included |
| `data/` | the copy of the dataset the agent worked in |
| `rubric.md` | the dataset's rubric, copied here to tick |

Eval outputs are **never committed**: they carry copies of the corpus. Write
them to a scratch directory outside the repository.

## Ticking the rubric

Follow [`rubrics/README.md`](rubrics/README.md). In short:

1. Read `summary.md` only. That is what a real user sees.
2. Every **Must** line is yes/no from the summary text. One unchecked Must is a
   fail for that dataset.
3. **Should** lines are recorded as gaps, and do not fail the pass.
4. Apply the number rule: grep `commands.txt` and `transcript.jsonl` for every
   number the summary quotes. A number that no command printed is an automatic
   fail, however good the rest is.
5. Note which "Known trap" a wrong summary fell into, and record the pass in
   the plan's Evaluation log.

## Cost and time

Measured on the Tier A datasets with `--model sonnet`, September 2026:

| Dataset | Runs | Wall time | Cost |
|---|---|---|---|
| The FμF state in PTFE | 30 | ~7 min | ~$3 |
| Ferromagnetic nickel | 61 | ~9 min | ~$5 |
| Spin glass YMnAl | 20 | ~5 min | ~$2 |
| A high-Tc cuprate | 48 | ~11 min | ~$6 |

A full Tier A pass is therefore roughly half an hour and a few tens of
dollars. Budget for three passes when iterating on the skill.

## Regenerating the skill's command reference

```bash
python tools/agent_eval/render_command_reference.py          # write
python tools/agent_eval/render_command_reference.py --check  # verify
```

`tests/tools/test_skill_package.py` runs the check, so a new CLI flag that
never reaches `references/commands.md` fails the suite.
