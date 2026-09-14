# Agent analysis CLI and skill (proof of concept)

Status: plan agreed with maintainer 2026-09-14; single PR on
`feat/agent-cli-skill`, built phase by phase by subagents with a lead-agent
review gate after every phase (see "Gates"). Decisions that change during
implementation are appended under "Decisions recorded".

## Goal

A user opens Claude Code (later Codex) in a directory of muon-spin
spectroscopy runs and says:

> This directory contains the data from a recent μSR experiment. Can you
> analyse these using Asymmetry and present me a summary of what they show?

The agent then carries out a preliminary analysis with no further steering:
survey the runs, calibrate alpha from a weak transverse-field run when one
exists, reduce to forward-backward asymmetry, screen models with the fit
wizard, fit the scan as a chained series, plot the trends, and write a short
summary that quotes only numbers the tools emitted.

The proof of concept answers one question: **does an agent make sensible
choices with these tools?** Polish is out of scope. Success is measured on
the WiMDA muon school datasets (see "Evaluation"), first with Claude Sonnet
driving, so that a less expensive agent is the bar.

## Non-goals (this PR)

- MCP server, GUI bridge, desktop-installer path, plugin marketplace.
- Count-domain, multi-group, Fourier and MaxEnt, negative-muon, ALC,
  rotating-frame, RF and muonium-chemistry workflows. The skill names these
  and tells the agent to stop and report rather than improvise.
- Writing a valid `.asymp` project headless (stretch phase S1 below).
- Any change to the GUI.

## Shape of the deliverable

Three layers, each usable alone; the upper ones only wrap the lower.

1. **`asymmetry.core.workflow`** (new, pure core, no Qt/matplotlib): the
   scriptable session façade. Every function takes typed inputs, returns
   JSON-serialisable results, and mirrors what the GUI does for the same
   step (reduction goes through `GroupingProfile` +
   `resolve_effective_grouping`, exactly the project-open path).
2. **`asymmetry.cli`** (existing `cli.py` becomes a package): subcommands
   over the façade, `--json` on every one, plus headless PNG plots
   (matplotlib lives here, not in core) and `skill install|check|uninstall`.
3. **The skill** `asymmetry-analysis` shipped as package data and installed
   by the CLI into `~/.claude/skills/` or `~/.agents/skills/`.

### The work directory is the session

Every command reads and writes `<folder>/.asymmetry/` (name settled below):

```
.asymmetry/
  manifest.json          # asymmetry version, settings used, run list
  survey.json            # output of `survey`
  reduced/<run>.npz      # time, asymmetry, error
  reduced/<run>.json     # metadata + reduction settings + grouping digest
  wizard/<run>.json      # serialised recommendation + narrative + recipe
  recipes/<name>.json    # a fit recipe (model + parameters + window)
  series/<name>.json     # per-run results, trend table, quality flags
  plots/*.png
```

Later commands read reduced spectra from here instead of reloading files,
so the agent has state between calls without a long-lived process. This is
the same state an MCP server would later hold in memory. Reduced spectra are
keyed on a digest of (file bytes, grouping payload, reduction settings); a
stale entry is recomputed, never trusted.

### The fit recipe

A JSON document that `wizard` emits, the agent may edit, and `fit` /
`fit-series` consume:

```json
{
  "schema": 1,
  "model": { ...CompositeModel.to_dict()... },
  "expression": "Oscillatory(exp) + Exponential + Constant",
  "parameters": { ...ParameterSet.to_dict() (value, min, max, fixed)... },
  "t_min": null, "t_max": 12.0, "rebin": 1,
  "source": {"wizard_run": 124222, "template_key": "oscillatory1_exp_relax_constant"}
}
```

This is the only contract between screening and fitting. The agent never
types a parameter table by hand; it edits a recipe.

## Command reference

All commands: `--workdir` (default `<folder>/.asymmetry`), `--json` (machine
output on stdout, human table otherwise), exit code 0 on success, 1 on a
user error with a one-line message on stderr, 2 on an internal error with a
traceback. Every JSON payload carries `"schema": 1` and `"asymmetry_version"`.

| Command | Wraps | Emits |
|---|---|---|
| `asymmetry survey <folder>` | `scan_run_files`, `load` (metadata only where the loader allows), `classify_tf_calibration_run`, `best_calibration_run_index` | per-run rows (run, file, instrument, facility, title, sample, temperature, field, field_direction, geometry, n_histograms, n_points, bin width, duration/events where recorded, has_file_deadtime), `calibration_candidates`, and `scans`: runs grouped by (geometry, field) ordered by temperature and by (geometry, temperature) ordered by field, so the agent sees the experiment's structure |
| `asymmetry alpha <folder> --run N` | `resolve_effective_grouping` with a `per_run_estimate` alpha policy (the GUI's Estimate button) | alpha, method, the run used, forward/backward groups, and a warning when the run is not a calibration candidate |
| `asymmetry reduce <folder> --runs A-B,C [--alpha X \| --alpha-from N] [--deadtime off\|from_file] [--background none\|...] [--rebin k] [--plot]` | `GroupingProfile` + `resolve_effective_grouping` + `reduce_grouped_asymmetry` | one entry per run in `reduced/`, a table of A(0), mean error, points, plus PNG per run with `--plot` |
| `asymmetry wizard <folder> --run N [--geometry ZF\|TF\|LF] [--scope <preset>] [--plot]` | `build_fit_wizard_recommendation` with a `WizardScope`; `--geometry` overrides the file's `field_direction` because ISIS files stamp `TF` on zero-field runs and some record nothing | `serialize_fit_wizard_recommendation(compact=True)`, `render_log_text`, confidence, verdict, caveat, the ranked table (key, title, AICc, chi2_red, parameter count), and a recipe written to `recipes/wizard-<run>.json`; PNG of data + recommended curve with `--plot` |
| `asymmetry fit <folder> --run N --recipe <path> [--fix name=value] [--free name] [--tmax T] [--plot]` | `FitEngine.fit` | `fit_result_summary` plus quality verdict; PNG with `--plot` |
| `asymmetry fit-series <folder> --runs A-B --recipe <path> --order temperature\|field\|run [--global p,q] [--name NAME] [--plot]` | `fit_asymmetry_series` with `seeding="auto"` (chain on ordered scans), `assess_member_quality` | `series/<name>.json`: per-run `fit_result_summary`, `reseeded_runs`, flags, and a trend table (x, each parameter, uncertainty); PNG per parameter with `--plot` |
| `asymmetry trend <folder> --series NAME [--csv PATH] [--plot]` | reads `series/<name>.json` | the trend table as JSON or CSV; one PNG per parameter |
| `asymmetry skill install\|check\|uninstall [--agent claude\|codex] [--project]` | package data | paths written; `check` verifies CLI on `PATH`, loaders import, matplotlib present, skill present and version-matched |

Stretch (phase S2): `asymmetry screen-series <folder> --runs A-B --families f1,f2`
wrapping `build_global_fit_wizard_screening_recommendation` with a scope
restricted to the named families. `--families` is **required** so the agent
must reason about the physics before spending the time; the skill says when
this is worth it and when a chained `fit-series` is enough.

## Defaults (settled with maintainer, 2026-09-14)

- **Reduction defaults match the GUI's fresh-run defaults**: alpha 1.0,
  deadtime off, no background, file t0 and good-bin window, no bunching.
  `survey` reports whether each file carries deadtime values; the skill tells
  the agent to pass `--deadtime from_file` on ISIS data that has them and to
  say so in the summary.
- **Alpha** is never guessed. If `survey` lists a calibration candidate the
  skill requires `alpha` on it and `--alpha-from`; otherwise the summary
  states that alpha was assumed 1.0.
- **Geometry** for the wizard comes from `field_direction` when the file
  records one, else from the survey's field value (0 G means ZF) and the
  agent's `--geometry`. The command echoes which source it used.
- **Series fits chain** (`seeding="auto"`), and a run whose fit failed,
  pinned a bound, or has a poor χ² verdict is flagged in the output, never
  silently included in the trend.
- **Numbers in the summary come from JSON.** The skill's summary template
  has slots; the agent fills them from command output and lists the runs it
  excluded and why.
- **Work directory name**: `.asymmetry/` inside the data folder.
- **Data stays out of the repo.** Tests build synthetic runs with
  `asymmetry.core.simulate` and write NeXus files with
  `asymmetry.core.io.nexus_writer` into `tmp_path`. The muon school folders
  are referenced only from the private eval runner's arguments.

## Module map

```
src/asymmetry/core/workflow/__init__.py     public façade re-exports
src/asymmetry/core/workflow/survey.py       survey_folder(), RunRow, ScanGroup
src/asymmetry/core/workflow/reduction.py    ReductionSettings, reduce_run(), estimate_alpha_for_run()
src/asymmetry/core/workflow/workdir.py      WorkDir: paths, digests, read/write of every artefact
src/asymmetry/core/workflow/recipe.py       FitRecipe (to/from dict, from a wizard assessment)
src/asymmetry/core/workflow/screen.py       screen_run() -> ScreenResult (serialised rec + narrative + recipe)
src/asymmetry/core/workflow/series.py       fit_series() -> SeriesOutcome (results, trend table, flags)
src/asymmetry/cli/__init__.py               main(), parser, dispatch (keeps `info`)
src/asymmetry/cli/_output.py                JSON/table rendering, exit codes
src/asymmetry/cli/commands/*.py             one module per subcommand
src/asymmetry/cli/plots.py                  matplotlib Agg renderers (only importer of matplotlib)
src/asymmetry/cli/skill.py                  install/check/uninstall
src/asymmetry/resources/skills/asymmetry-analysis/SKILL.md (+ references/)
tests/core/test_workflow_*.py               façade tests on synthetic runs
tests/core/test_cli_*.py                    CLI tests (existing test_cli.py stays)
tests/tools/test_skill_package.py           skill is packaged, frontmatter valid, install writes paths
tools/agent_eval/run_eval.py                private eval runner (paths from args)
tools/agent_eval/rubrics/<dataset>.md       expected findings per dataset
docs/reference/agent_workflow.rst           user docs for the CLI and skill
```

`pyproject.toml`: new extra `agent = ["matplotlib>=3.7,<3.11", "h5py>=3.8,<4"]`
(ROOT and HDF4 stay separate extras; the skill's `check` reports which loaders
are available), package-data entry for `resources/skills/**`, and the
`tools/harness.py` core-boundary check keeps matplotlib out of
`core/workflow`.

## Phases

Each phase is one subagent brief and ends in one commit on
`feat/agent-cli-skill`. The lead reviews the diff before running any gate.
Model suggestions are in brackets.

### Phase 1 — façade: survey, reduction, work directory, CLI skeleton [Opus]

- `core/workflow/survey.py`, `reduction.py`, `workdir.py`.
- `cli/` package with `survey`, `alpha`, `reduce` (no plots yet) and the
  existing `info`. `asymmetry.cli:main` entry point unchanged.
- Tests: synthetic folder fixture (one weak-TF calibration run, a five-run
  ZF temperature scan, one file with deadtime values) built once per session
  from `simulate` + `nexus_writer`; survey groups the scan and finds the
  calibration run; alpha on the TF run is within tolerance of the simulated
  value; reduce round-trips through the work directory and is byte-identical
  to a direct `reduce_grouped_asymmetry` call; digest invalidation on a
  changed setting.
- Gate: focused tests, `--tier fast`, then a private check by the lead on
  the nickel and PTFE folders (survey finds the TF calibration run 17293 for
  PTFE; nickel has no candidate and says so).

### Phase 2 — screening, recipes, fits, series [Opus]

- `recipe.py`, `screen.py`, `series.py`; commands `wizard`, `fit`,
  `fit-series`, `trend`.
- `wizard` writes a recipe from the recommended assessment (fitted values
  become starting values). `fit-series` accepts `--global` for parameters
  shared across runs and passes the rest as local.
- Series outcome includes the trend table with the order key (temperature
  or field read from the reduced metadata) and the quality flags from
  `fit_result_summary`.
- Tests: a synthetic scan whose relaxation rate follows a known curve; the
  wizard picks the relaxation family; the series recovers the curve within
  uncertainties; a deliberately broken run (zero counts) is flagged, not
  trended.
- Gate: focused tests, `--tier fast`, private check on nickel ZF 124218–124248
  (frequency versus temperature falls to zero near the transition) and PTFE
  17294–17322 (F-μ-F family recommended, dipolar frequency roughly flat).

### Phase 3 — plots, `agent` extra, skill packaging commands [Sonnet]

- `cli/plots.py`: data + fit per run, wizard candidate overlay, one trend
  PNG per parameter with error bars. Agg backend, fixed size, no GUI import.
- `--plot` on `reduce`, `wizard`, `fit`, `fit-series`, `trend`.
- `skill install|check|uninstall`, package data, `agent` extra.
- Tests: PNGs exist and are non-trivial; `skill install --project` into
  `tmp_path` writes `SKILL.md`; `check` reports a version mismatch when the
  manifest is edited.
- Gate: focused tests; structural check (no matplotlib under `core/`).

### Phase 4 — the skill and the evaluation loop [Opus writes, Sonnet is evaluated]

- `SKILL.md` frontmatter description tuned to fire on μSR vocabulary, the
  tool name, and the file extensions the agent will see when it lists the
  folder. Body: the workflow in order, decision rules, cost table (survey
  seconds, wizard 3–8 s per ISIS run, series seconds per run, screen-series
  minutes), the physics-reasoning prompts, the summary template, the
  stop-and-report list, and a worked example on a synthetic scan.
- `tools/agent_eval/run_eval.py`: copies a dataset folder to a temp
  directory, installs the skill, runs `claude -p` with the fixed sentence
  above and `--model sonnet`, saves the transcript, summary and the work
  directory, and prints the rubric checklist for a human to tick.
- Rubrics for the Tier A datasets below.
- Loop: run Tier A with Sonnet, read the transcripts, fix the skill text
  (never the rubric), repeat. Record each pass under "Evaluation log".
- Gate: every Tier A dataset produces a summary that passes its rubric with
  Sonnet; the triggering description fires on all should-fire prompts and
  none of the should-not-fire ones (use `/skill-creator` evals).

### Phase 5 — docs, README, changelog [Sonnet]

- `docs/reference/agent_workflow.rst`: install, commands, work directory,
  recipe format, skill install, limits. Link from `reference/index.rst`.
- README: the "Use Asymmetry from an AI coding agent" subsection under
  Installation, one line in Quick start, one bullet in Main functionality.
- `CHANGELOG.md` `[Unreleased]` entry; `docs/PLANS.md` status.
- No screenshots (no GUI change). `python tools/harness.py docs` passes.

### Stretch S1 — `.asymp` output [Opus]

`asymmetry export-project <folder> --out NAME.asymp` assembling the schema
v19 state from the work directory (datasets, grouping profile, single-fit
slots, one `FitSeries`). Study `MainWindow._on_save_project` first; the
state assembly may need to move into `core/project` to be reusable. Only
start after Phase 4 passes.

### Stretch S2 — family-limited series screening [Opus]

`screen-series` as described above, plus the skill section on when to use
it (multi-field decoupling sets such as the ionic-motion exercise, or when
single-run wizard results disagree across a scan).

## Gates

After every phase the lead:

1. Reads the diff for defensive guards (`hasattr`/`getattr(..., None)` on
   our own attributes, `try/except` around our own code, "in progress"
   flags, `if x is None: return` where the design can make `None`
   impossible). Ask, don't guard.
2. Runs the phase's focused tests, then `python tools/harness.py test --tier fast`.
3. Runs the private check on the named muon school folders.
4. Runs `python tools/harness.py structural` and `lint`.

`python tools/harness.py validate` runs once, before the PR is opened.

## Subagent brief (paste verbatim into every phase prompt)

- Work on branch `feat/agent-cli-skill` in the main checkout unless told you
  are in a worktree; in a worktree prefix every Python or harness call with
  `PYTHONPATH=<worktree>/src`.
- Use `.venv/bin/python` via `python tools/harness.py ...`; never bare pytest.
- `asymmetry.core` must not import Qt, matplotlib or `asymmetry.gui`.
- No defensive guards: make bad states impossible by construction; if unsure
  how, stop and say so in your report rather than adding a guard "for now".
- No research data, run numbers, sample names or private paths in the repo;
  tests use synthetic runs.
- Reuse existing core functions; do not reimplement reduction, alpha,
  wizard, series or summary logic.
- Commit at the end of the phase with a conventional message; do not push.
- Your report must list: files touched, tests added, commands run with
  their results, and anything you were unsure about.

## Evaluation

Datasets are the WiMDA muon school folders at
`~/Documents/WiMDA muon school/` (never copied into the repo). Each has a
worksheet whose expectations become the rubric.

**Tier A — must pass with Sonnet before the PR opens**

| Dataset | Files | Structure | Expected findings (from the worksheet) |
|---|---|---|---|
| Magnetism / Ferromagnetic nickel | 61 EMU `.nxs`, ZF 100–380 K then TF 100 G scan | no explicit calibration run; TF runs above the transition serve | oscillation below the transition, frequency falling towards zero on warming, no structure at 380 K; agent reports the transition region and the loss of oscillation at the lowest temperature |
| Nuclear magnetism / The FμF state in PTFE | 30 MUSR `.nxs`, TF 20 G calibration 17293 + ZF 17294–17322 | calibration run present | alpha from 17293; F-μ-F family recommended; dipolar coupling frequency extracted and roughly temperature independent |
| Magnetism / Spin glass YMnAl | 20 MUSR `.nxs`, TF 20 G 24563 + LF 110 G 90–290 K | calibration run present | stretched-exponential relaxation with a flat background; rate rising on cooling towards the glass transition; agent fixes the background and A(0) as the worksheet does, or explains why not |
| Superconductivity / A high-Tc cuprate | EMU TF 150 G 5–80 K and MUSR TF 400 G 10–125 K | two instruments in one folder | survey separates the two scans; Gaussian relaxation rate rising below Tc; agent reports σ(T) per field and notes the non-superconducting fraction |

**Tier B — run once, record, do not block**

Copper (TF, ZF and LF sets in one folder: survey must split them), Spin-Peierls
(TF 100 G calibration + ZF scan, line-shape change), Molecular antiferromagnet
(ZF only, multiple frequencies), EuO (PSI `.bin`, ZF, exercises the PSI loader).

**Tier C — must decline gracefully**

ALC in TCNQ (ALC is out of scope: the agent must say so), Ionic motion in
Al-LLZ (three-field decoupling sets needing simultaneous fits: the agent must
recognise it and either stop or, with S2, screen with the Kubo-Toyabe family
only), AFM transition in high TF (6 T `.mdu` from PSI HIFI: out of scope,
must say so).

**Trigger prompts** (should fire): the fixed sentence; "fit the zero-field
runs in here"; "what's in these .nxs files"; "reduce runs 17294 to 17322 and
plot the asymmetry". (Should not fire): "summarise this CSV of DFT energies";
"plot the XRD pattern in this folder".

## Evaluation log

(append one entry per Sonnet pass: date, datasets, pass/fail per rubric line,
what changed in the skill)

## Decisions recorded

- 2026-09-14: Project-file output is a stretch goal; the first aim is agents
  driving the API. Plots are in scope. Series fitting is in scope. The global
  wizard is not the default path; when used, the agent must restrict families
  by reasoning about the system. Validate in Claude Code first, Sonnet for
  evals. Commands live under `asymmetry`.
- 2026-09-14: Rubrics live in the repo under `tools/agent_eval/rubrics/`; the
  muon school corpus is not private (not yet public either) but is never
  copied into the repo. Deadtime default matches the GUI (off); the skill
  hints that ISIS data should normally be reduced with `--deadtime from_file`
  when the survey reports file deadtime values.

## Open questions for the maintainer

None outstanding (both resolved 2026-09-14, see "Decisions recorded").
