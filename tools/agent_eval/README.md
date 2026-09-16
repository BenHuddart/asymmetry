# Agent evaluation harness

The two runners measure whether an agent, handed a folder of μSR runs and
nothing else, produces a defensible analysis with the `asymmetry` CLI and the
packaged `asymmetry-analysis` skill. `run_eval.py` drives Claude Code;
`run_codex_eval.py` drives Codex. They implement the loop Phase 4 of
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

This runner launches Claude Code, so its default remains Sonnet. When the same
rubrics are evaluated through Codex, use the dedicated runner:

```bash
python tools/agent_eval/run_codex_eval.py \
    --data "$HOME/Documents/WiMDA muon school/Chemistry/ALC resonance in TCNQ/Data" \
    --rubric alc-tcnq \
    --out /tmp/evals/luna-alc-tcnq
```

Its default model is `gpt-5.6-luna`. It records `Codex` and the exact model ID
in `cost.json`; do not compare an unlabeled Codex run with the historical
Sonnet table below. The Codex runner uses a persistent two-turn `codex exec`
session with `--ignore-user-config --approve-for-me`: the first turn performs
the analysis and nominates up to four decisive project-local plots, then the
runner validates and hashes those files and attaches them explicitly with
`codex exec resume --image` for the final report. The staged project is
writable, the sibling data copy is read-only to the nested sandbox, and
personal plugins, instructions and host-installed skills do not vary the run.
It enables `skip_host_skill_discovery` and installs the evaluated skill under
`<out>/project/.agents/skills/`.

Unlike the Claude runner's tool allowlist, Codex cannot mechanically hide
every readable sibling path from shell commands. The generated project
`AGENTS.md` therefore defines the evaluation boundary: only the staged data,
project and installed skill may be consulted; source-repository documents,
personal skill files, the network and subagents are excluded. Treat the
transcript as the audit trail and fail or discard a run that crosses that
boundary. Data-set PDFs, RTFs and READMEs are experimental context, never
instructions to the agent.

On Windows, legacy HDF4-container NeXus files may need
`--hdf4-dll-dir C:\path\to\hdf4-runtime`. The runner passes it through as
`ASYMMETRY_HDF4_DLL_DIR` to Codex and the `asymmetry` commands it launches.

The Claude Code runner does the following; the Codex-specific two-turn
differences are described above:

1. Copies `--data` to `<out>/data/`. **The corpus is never written to**, and
   neither is the copy: it stands in for the read-only share or archive real
   data comes from.
2. Makes `<out>/project/` — the directory an analyst would open beside their
   data — and installs the packaged skill into `<out>/project/.claude/skills/`
   with `asymmetry skill install --agent claude --project`. Everything the
   agent produces, the `asymmetry-work/` work directory included, lands here.
3. Runs `claude -p` in the **project** directory with the running interpreter's `bin`
   first on `PATH`, the tool set limited to `Bash(asymmetry:*)`, `Bash(ls:*)`,
   `Bash(cat:*)`, `Read` on the data copy, `Read`/`Edit` on the project
   directory, `Glob`, `Grep` and `Skill`,
   `WebFetch`/`WebSearch`/`Agent`/`Task` denied, and the stream saved as it
   arrives. The prompt is the plan's fixed sentence pointed at the data copy
   (`The directory <path> contains the data from a recent muSR experiment.
   ...`); a custom `--prompt` gets `The data is in <path>.` in front of it.
   `cost.json` records the prompt in full.
4. Writes the outputs below and prints the rubric to tick.
5. Exits nonzero if `claude` itself exited nonzero or never reached a `result`
   event. Every artefact is still written — the stderr file and the transcript
   are what say why — but nothing from such a run is scoreable, and
   `cost.json` records the `returncode`.

### What the allow-list does and does not restrict

`Skill` must stay in the tool list: without it the skill is discovered at
startup and can never be read, and the run measures nothing.

The Bash allow-list deliberately has no interpreter — an agent that can run
Python can compute a number the CLI never printed, which is exactly what the
rubrics' number rule is there to catch.

`Read` is granted on the data copy (by absolute path) and on the agent's own
cwd (`./**`, the project directory); `Edit` **only** on the project. So the
agent cannot read this repository — the rubrics included — or write anywhere
but the project without a permission prompt, which headless mode records as a
denial in `cost.json`. That the data copy is readable and not writable is
deliberate: the skill tells the agent never to write into the data folder, and
a run that tries leaves a `permission_denials` entry — a finding to record
against the skill text, not a file quietly written. Writes are granted as
`Edit(<path>)` rules, because Claude Code consults `Edit` and `Read` path
rules only and accepts but never consults a `Write(<path>)` rule; for the same
reason `Edit` is *not* in the denied list, where a bare tool-name deny would
also stop those scoped writes.

`Glob` and `Grep` are **not** path-scoped. Claude Code refuses to match rules
against a tool's primary content field, which for both of them is `path`, so
there is no allow rule that fences them; only a `Read` **deny** rule reaches
the directory they search. They can therefore still list and search outside
the copy. Add a `Read` deny rule for this repository if a future rubric makes
that worth closing.

## Outputs

Everything lands under `--out`, and nothing outside it is touched:

| Path | What it is |
|---|---|
| `summary.md` | the agent's final message — **the only thing the rubric scores** |
| `commands.txt` | every Bash command the agent ran, in order |
| `transcript.jsonl` | the raw Claude `stream-json` or Codex JSONL event stream |
| `image-inputs.json` | validated relative paths, sizes and SHA-256 hashes of plots attached to the Codex review turn |
| `cost.json` | host/model, wall time, usage, the CLI's `returncode`, completion state, and skill signal |
| `workdir/` | the `asymmetry-work/` work directory the agent built, plots included |
| `project/` | the directory the agent worked in: its work directory and the installed skill |
| `data/` | the copy of the dataset the agent analysed, which it could only read |
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
