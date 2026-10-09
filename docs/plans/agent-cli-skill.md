# Agent analysis CLI and skill (proof of concept)

Status: plan agreed with maintainer 2026-09-14; single PR on
`feat/agent-cli-skill`, built phase by phase by subagents with a lead-agent
review gate after every phase (see "Gates"). Decisions that change during
implementation are appended under "Decisions recorded".

Expansion status (2026-09-15): a follow-on pass moved multi-period reduction,
integral ALC/QLCR scans, FFT spectra, and true simultaneous fitting of one run
group into scope. The original proof-of-concept non-goals and evaluation notes
below are retained as the historical baseline. Remaining gaps include MaxEnt,
count-domain and multi-group fitting, rotating-frame analysis, and automatic
batching/trending of a temperature series of simultaneous groups.

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
the WiMDA muon school datasets (see "Evaluation"). Claude Code evaluations
target Sonnet; Codex evaluations target `gpt-5.6-luna`, keeping a fast,
inexpensive agent as the bar on each host.

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

Every command reads and writes `./asymmetry-work/` — a visible directory in
the project the analysis is being done in, resolved against the current
directory and never against the data folder:

```
asymmetry-work/
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

One work directory holds one data folder: everything in it is keyed on the run
number alone, so the manifest's `folder` is binding — `survey` and `reduce`
write it, every command checks it, and a second folder needs
`--workdir asymmetry-work-<name>` of its own.

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

All commands: `--workdir` (default `./asymmetry-work`, see Decisions recorded), `--json` (machine
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

Also: collapse warning tracebacks on stderr (the wizard emits
`AsymmetryScaleWarning` stack traces from candidate seeding) into one line
per distinct warning unless `--verbose`; an agent reads stderr.

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

AFM transition in high TF (6 T `.mdu` from PSI HIFI: out of scope, must say
so). Moved to Tier B as a bounded analysis on 2026-09-23 — see the corpus
capability audit below; no decline case remains.

**Workflow-expansion gate — must exercise the added path**

ALC in TCNQ (`integral-scan`), ionic motion in Al-LLZ (`fit-global` on a
three-field group), photo-µSR in silicon (separate red/green period reductions),
and the CdS shallow-donor spectrum (`fourier`). These replace the former ALC
and ionic-motion decline cases and add explicit period/FFT coverage. A case
does not pass merely because its summary sounds plausible: the recorded
commands must contain the workflow under test.

**Trigger prompts** (should fire): the fixed sentence; "fit the zero-field
runs in here"; "what's in these .nxs files"; "reduce runs 17294 to 17322 and
plot the asymmetry". (Should not fire): "summarise this CSV of DFT energies";
"plot the XRD pattern in this folder".

## Evaluation log

One entry per Sonnet pass. Runner: `tools/agent_eval/run_eval.py`, model
`sonnet` (claude-sonnet-5), the plan's fixed prompt, `--max-turns 80`, tools
limited to `Bash(asymmetry:*|ls:*|cat:*)`, `Read`, `Glob`, `Grep`, `Write`,
`Skill`. Outputs live outside the repo (a scratch directory); only the verdicts
are recorded here.

### Pass 1 — 2026-09-14, Tier A

The skill was invoked (a `Skill` tool call) on all four; every dataset was
analysed end to end with the six-command workflow.

| Dataset | Wall | Turns | Cost | Verdict |
|---|---|---|---|---|
| fmuf-ptfe | 153 s | 22 | $0.53 | **pass** |
| ferromagnetic-nickel | 942 s | 66 | $2.49 | **pass** |
| spin-glass-ymnal | 173 s | 21 | $0.59 | **fail** (1 Must) |
| high-tc-cuprate | 423 s | 46 | $1.46 | **fail** (1 Must) |

- **fmuf-ptfe — pass.** All six Musts. Calibration from 17293 (α = 0.9566),
  deadtime `from_file`, F-μ-F family recommended and applied, dipolar coupling
  reported, temperature dependence stated, no invented numbers (checked by
  grepping every summary number against the tool results in the transcript).
  Two notes: (a) the CLI parameterises the F-μ-F dipolar coupling as the
  muon–fluorine distance `r_muF`/Å and emits no coupling *frequency* anywhere,
  so the "reports a dipolar coupling frequency" Must was ticked on the
  summary naming `r_muF` as the dipolar coupling — no possible summary could
  tick it literally; (b) the "sample-temperature column with its uncertainty"
  Should is not reachable: `survey` prints one temperature column.
- **ferromagnetic-nickel — pass.** All seven Musts, including the textbook-Tc
  trap (the summary quotes 627 K explicitly as "for reference only — not a
  result of this session"). The agent split the ZF scan into three regimes,
  screened at 352 K, and recovered the order parameter falling 6.12 → 2.80 MHz
  over 345–356 K with the oscillation gone by 360–380 K. Must 3 ("oscillation
  present in the ZF spectra at the lower temperatures scanned") was ticked on
  the plan's own reading of this dataset — below the transition, where it is
  resolvable; at a pulsed source the base-temperature internal field is too
  fast to resolve at all and A(0) simply collapses, which the summary states.
  Should unchecked: critical exponents (β, γ, w) not raised as a follow-on.
- **spin-glass-ymnal — fail.** Must "fit with a stretched-exponential (or
  comparable stretched decay) plus a flat background": the agent fit
  `Exponential + Constant`, the wizard's AICc pick, with `stretched_constant`
  ranked comparable just below it. Everything else passed — 24563 identified
  as the TF calibration, α = 1.2320 from it, the 110 G scan correctly
  reclassified from the file's `TF` stamp to LF by checking for precession in
  the reduced PNG, the rate rising on cooling towards freezing. It also
  *noticed* A₁/A_bg going degenerate at the hot end (background formally
  negative) and reported the numbers with a caveat instead of fixing one from
  a reference run (the worksheet's method — a Should).
- **high-tc-cuprate — fail.** Must "discusses how the non-superconducting
  fraction was handled": the summary never mentions it, though the fitted
  model carries the `Constant` term that represents it. Everything else
  passed — EMU and MUSR separated, 400 G and 200 G named as separate groups,
  a Gaussian σ(T) rising from 0.066 to 1.157 μs⁻¹ on cooling at 400 G, the
  diamagnetic frequency shift, and the "above Tc" reference run called out.

Skill changes after pass 1 (text only; no rubric or CLI change):

1. **"AICc ranks the candidates; the physics picks the model."** New leading
   decision rule with the four cases that matter here — glassy magnetism →
   stretched exponential, vortex lattice → Gaussian, static nuclear dipolar →
   Kubo–Toyabe, fluoride → F–μ–F — and the instruction to take a comparable
   candidate from the ranked table by hand-editing the recipe.
2. **"Say what every amplitude means."** New decision rule plus tightened
   Model/Results template slots: the `Constant` term is the non-relaxing
   fraction (for a superconductor, the non-superconducting fraction), and a
   summary must say how it was handled, not only quote a rate.
3. **Which run to screen** rewritten: pick from the `reduce` table and the
   PNGs, and for a magnet at a pulsed source screen *just below* the
   transition — a collapsing A(0) on cooling is loss of resolvable signal, not
   loss of magnetism.
4. `auto` scope expands a family only on spectral evidence: if static order is
   expected and `auto` returns a bare relaxation, re-screen with
   `--scope zf-static-magnetism`.
5. The wizard's "sample name suggests fluorine" note fires on ISIS's `F=<n>`
   title convention for every ISIS run; the skill says to ignore it unless the
   sample really is a fluoride.
6. A wide field scan at fixed temperature is an LF decoupling measurement.
7. The fix-from-a-reference-run pattern is no longer only a rescue for flagged
   runs: it is the answer whenever amplitude and background trade off at one
   end of a scan.

### Pass 2 — 2026-09-14, the two Tier A failures

| Dataset | Wall | Turns | Cost | Verdict |
|---|---|---|---|---|
| spin-glass-ymnal | 334 s | 29 | $1.26 | **fail** (1 Must, a different one) |
| high-tc-cuprate | 462 s | 45 | $1.34 | **pass** |

- **high-tc-cuprate — pass.** The non-superconducting-fraction Must is now met
  explicitly: `A_bg` is named as the non-relaxing fraction, carried in every
  results table, and summarised ("small and steady throughout all three scans
  … the great majority of stopped muons sample the superconducting volume").
  σ(T) Gaussian at 400 G rising 0.066 → 1.157 μs⁻¹, the diamagnetic frequency
  shift, both instruments and all three fields separated, run 1276 named as the
  above-Tc reference. One borderline number: the summary quotes "the nominal
  150 G Larmor value (2.033 MHz)", which is γ_μ/2π × B computed by the agent
  rather than printed by a command — allowed, because it is labelled as the
  nominal reference value and not as a measurement.
- **spin-glass-ymnal — fail, on the other half of the problem.** The model is
  now right — it hand-edited a `Stretched Exponential + Constant` recipe,
  reported β, and explained choosing it over the AICc leader — but it dropped
  the geometry check pass 1 had done and described the 110 G scan as a
  **transverse-field** scan. That is the rubric's own Known trap ("110 G is
  deliberately chosen to decouple static fields; a summary that treats this as
  a TF measurement rather than a decoupling LF misreads the experiment"), so
  Must 3 ("reports the **LF** series as fit with …") fails.

Skill changes after pass 2:

8. **New mandatory step 3a, "confirm the geometry of every non-zero-field
   scan"**, between reduce and screen: the survey's `geom` column is a
   hypothesis; 0 G is ZF whatever the file says; for a non-zero field look for
   precession at γ_μ/2π × B (13.55 kHz/G, with the worked numbers for 20, 110
   and 400 G) in the reduced PNG, zooming with `reduce --tmax 2 --plot`; no
   oscillation there means the field is longitudinal. Step 4's geometry advice
   collapses to "always pass the `--geometry` you established in step 3a".
9. The summary template's Experiment slot now requires each scan's geometry
   **and how it was confirmed**, not what the file stamp said.

### Pass 3 — 2026-09-14/15, the last failure, plus a regression check

| Dataset | Wall | Turns | Cost | Verdict |
|---|---|---|---|---|
| spin-glass-ymnal | 241 s | 18 | $0.72 | **pass** |
| fmuf-ptfe (re-run) | 219 s | 23 | $0.66 | **pass** |
| high-tc-cuprate (re-run) | 535 s | 50 | $1.62 | **pass** |

- **spin-glass-ymnal — pass.** Both halves at once: the 110 G scan is
  reclassified LF from the absence of precession at ~1.5 MHz (checked at 75 K
  and zoomed to 2 μs at 280 K), 24563 is the calibration run (α = 1.2320),
  the fit is `Stretched Exponential + Constant` with β reported, Λ rises three
  orders of magnitude on cooling to ~85 K, and — the worksheet's own method —
  `A_bg` was fixed at 3.040 % from run 24574 to break the amplitude/background
  degeneracy, cutting flagged runs from 9/18 to 3/18. Should unchecked: a
  named Tg and critical exponent as a further fit.
- **fmuf-ptfe and high-tc-cuprate re-run** against the final skill text, to
  check passes 1 and 2 were not bought at each other's expense. Both still
  pass every Must, and both improved: PTFE now confirms the ZF geometry, names
  what each amplitude means, fixes `A_bg` from the screening run, and
  concludes that the apparent `r_muF` drift above ~80 K is a fit degeneracy
  with the fluctuation rate rather than a real change in the coupling — closer
  to the worksheet's "roughly temperature independent" than pass 1 was. The
  cuprate now confirms TF geometry against the Larmor relation and states
  plainly that alpha was assumed 1.0 for the EMU scan because it has no
  above-Tc run. `ferromagnetic-nickel` was **not** re-run under the final text
  (it is the slowest and dearest dataset); its pass stands from pass 1, and the
  edits since then codify what it already did unprompted.

**Tier A result: all four datasets pass with Sonnet.**

### Pass 4 — 2026-09-15, nickel re-run on the final skill text

| Dataset | Wall | Turns | Cost | Verdict |
|---|---|---|---|---|
| ferromagnetic-nickel | 533 s | 41 | $1.42 | **pass** |

Run by the lead after the fluorine-sniff fix, so every Tier A dataset has now
passed under the final skill text. All seven Musts: the three regimes named
with their run ranges (ZF 100–380 K, 100 G 340–380 K, a 1200–4000 G field
scan at 200 K read as LF decoupling), the absence of a calibration run
stated with the 380 K TF run declared as the stand-in (α = 1.9672, the CLI's
warning acknowledged), the ZF oscillation resolved in a 325–356 K band with
the frequency falling 8.83 → 2.80 MHz, the oscillation gone from 358 K, and
the textbook Curie temperature quoted only as "not a result of this
analysis". Should lines met: the LF field scan treated as a separate
decoupling exercise; the low apparent transition flagged as something to
check against the sample. Should unchecked: critical exponents not raised.
One `Bash` call was denied by the allow-list (a filesystem-wide `find` for a
recipe file), which is the harness working as intended.

### Pass 5 — 2026-09-15, after measured Larmor precession in the survey

Maintainer testing found the survey listed no calibration candidates when
no file carried transverse-field metadata (the 2024 EMU nickel files record
none). The survey now measures precession at the Larmor frequency of each
run's recorded field with the wizard's spectrum fingerprint (commits
`a4f672b`, `01db4f1`): measured candidates ranked by SNR, `TF*` geometry when
measured, a refuted `TF` stamp reads unknown, scans key on instrument. Two
Sonnet re-runs on the unchanged rubrics:

| Dataset | Wall | Turns | Cost | Verdict |
|---|---|---|---|---|
| ferromagnetic-nickel | 719 s | 51 | $1.84 | **pass** |
| spin-glass-ymnal | 216 s | 20 | $0.83 | **pass** |

- **Nickel.** The agent's second command was `alpha --run 124251`, the
  survey's measured best candidate (376 K, SNR 428), with no workaround
  needed; the 100 G scan's geometry was read from `prec larmor` on the
  paramagnetic side and `other`/`none` below the transition, and the 200 K
  field scan from refuted stamps plus the spectra. ZF order parameter fitted
  200–356 K (10.8 → 2.95 MHz) with an Overhauser powder model chosen under
  `--scope zf-static-magnetism` after rejecting the unscoped F-μ-F pick as
  unphysical for nickel. Five `Bash` calls were denied (python3 and jq), the
  harness's number rule holding.
- **YMnAl.** Run 24563 identified from `prec larmor` (SNR 158), alpha
  1.2320 from it, the 110 G scan read as LF from `prec none` and confirmed on
  the spectra, stretched exponential with a fixed 1.3 % background, rate
  rising on cooling. All Musts.

### Pass 6 — 2026-09-15, project-level work directory

After the maintainer's request that outputs go to a visible folder in the
project directory rather than beside the data (commit `c2a4778`), the runner
gives the agent an empty project directory and the data copy read-only.

| Dataset | Wall | Turns | Cost | Verdict |
|---|---|---|---|---|
| fmuf-ptfe | 134 s | 17 | $0.74 | **pass** |

Every command was given the absolute data path; `asymmetry-work/` appeared
in the project directory and nothing was written beside the data (no
permission denials, and the allow-list grants no write there). Calibration
from the measured candidate 17293 (SNR 89), Dynamic F-μ-F + Constant, the
coupling reported as `r_muF` and read as rigid below ~45 K with motional
narrowing above, 18 flagged runs listed as not results. All Musts.

### Tier B and C — 2026-09-15, one run each, final skill text

| Dataset | Tier | Wall | Turns | Cost | Verdict |
|---|---|---|---|---|---|
| alc-tcnq | C (decline) | 91 s | 9 | $0.38 | **correct decline** |
| spin-peierls | B | 394 s | 31 | $0.95 | **pass** |

- **alc-tcnq.** Names the technique ("an avoided-level-crossing (ALC)
  resonance experiment on TCNQ"), says why the CLI cannot do it (integral
  asymmetry versus field, fit as a resonance line shape; no such command),
  produces no hyperfine couplings or resonance fields, and stops — having
  first surveyed the folder and reduced three runs to *show* the difference
  between the 100 G calibration precession and the flat 2000 G sweeps. It
  reports the real file range (19485–19612) and so avoids the worksheet's
  phantom run 19618.
- **spin-peierls.** All six Musts: 29919 identified as the TF 100 G reference
  and used for alpha (0.9685) with the reason stated, both geometries
  confirmed from the spectra, the full 26-run range reported (not the
  worksheet's subset), nothing claimed for the absent 29945–29951, a change in
  the ZF relaxation between the extremes, and no invented numbers. It also
  correctly noticed that these files carry *no* deadtime values and said it
  therefore reduced with the default.

### Codex/Luna workflow-expansion gate — 2026-09-16

Host: Codex. Model: `gpt-5.6-luna`. Runner:
`tools/agent_eval/run_codex_eval.py`. Outputs are under the external scratch
root `wimda-evals-luna-2026-09-16`; every `cost.json` records the host, exact
model and evaluation boundary. The latest result is **4/4 pass**.

| Dataset/workflow | Latest wall | Verdict | Evidence |
|---|---:|---|---|
| Al-LLZ ionic motion / `fit-global` | 306.5 s | **pass** | All 13 three-field groups were jointly fit with shared relaxation parameters and run-specific longitudinal field. |
| TCNQ ALC / `integral-scan` | 238.8 s | **pass** | All four temperature scans were integrated and fit; the agent stopped inventing hyperfine constants not emitted by the CLI. |
| Silicon photo-µSR / period selection | 515.6 s | **pass** | All 23 HIFI files were classified, red/laser-on and green/laser-off were reduced separately, run 103277 was compared in both series, and the model/window were stated without inventing a carrier lifetime. |
| CdS shallow donor / `fourier` | 199.2 s | **pass** | The final two-turn image-review run selected the 49,056,663-event 1 K run 20721, compared its full-record FFT directly with the warm reference, and cautiously identified a central diamagnetic line with visual shallow-muonium satellites. |

The ALC case failed its first run because the agent calculated hyperfine
constants in PowerShell; a narrow skill correction made the second run pass.
The photo-µSR case passed on the third run after the skill required period
closure, a stated fit window and an explicit refusal to invent worksheet-only
power/delay metadata. The first three CdS runs exposed an affordance gap:
`survey` did not emit total event counts. It now reports `total_events` in JSON
and an `events` column in the human table; pass 4 selected run 20721 correctly
from that evidence. Later runs exposed the need for a real PNG-inspection path
and clearer general guidance about full-record FFT resolution, matched cold/warm
views, and visual-only features. Pass 15 used the runner's explicit second image
turn and passed all six Musts while keeping the triplet assignment qualitative.

The first ionic-motion run completed successfully but the wrapper returned 1
while printing a Unicode rubric character on Windows after all artifacts had
been saved. The runner now replaces unencodable report characters before
printing, and its focused regression tests cover that path.

### Codex/Luna Tier A parity with the Sonnet gate — 2026-09-16

Host: Codex. Model: `gpt-5.6-luna`. Runner:
`tools/agent_eval/run_codex_eval.py`, using the same fixed prompt and unchanged
Tier A rubrics as the historical Sonnet gate. Outputs are under the external
scratch root `wimda-evals-luna-tier-a-2026-09-16`. Every scoreable run used the
two-turn image-review protocol and records the staged skill, boundary, commands,
image hashes and token usage. **All four Tier A datasets pass every Must.**

| Dataset | Passing wall | Must score | Verdict |
|---|---:|---:|---|
| fmuf-ptfe | 321.8 s | 6/6 | **pass** |
| spin-glass-ymnal | 478.4 s | 5/5 | **pass** |
| high-tc-cuprate | 359.3 s | 5/5 | **pass** |
| ferromagnetic-nickel | 364.8 s | 7/7 | **pass** |

- **PTFE.** Luna used run 17293 for alpha, fitted Dynamic F-μ-F + Constant,
  separated `r_muF` from the temperature-dependent dynamics and excluded all
  flagged values from the physical trend.
- **YMnAl.** The first clean result omitted detector-grouping provenance. A
  general skill correction made calibration summaries include alpha and the
  forward/backward grouping or instrument profile; the rerun passed with run
  24563, the 110 G LF classification and a broader-than-single-exponential
  relaxation model. One earlier launch was discarded because it attempted to
  read outside the staged skill boundary.
- **Cuprate.** Luna separated EMU 150 G, MUSR 200 G and MUSR 400 G analyses,
  reported Gaussian broadening on cooling, and explicitly interpreted the
  constant component as the non-relaxing/non-superconducting fraction.
- **Nickel.** A first scoreable run described the three regimes without giving
  every run range and left the high-temperature loss of ZF oscillation implicit.
  General summary guidance now requires per-scan run provenance and an explicit
  endpoint statement when a defining feature weakens, becomes unresolved or
  disappears. The rerun passed all seven Musts. Two additional launches were
  unscoreable infrastructure events: one could not open Codex state under the
  restricted shell, and one hit the account usage limit before image review.

### Sonnet workflow-expansion gate — 2026-09-16/17

Host: Claude Code. Model: `sonnet` (claude-sonnet-5). Runner:
`tools/agent_eval/run_eval.py`, the plan's fixed prompt, `--max-turns 80`,
run on Windows with `--hdf4-dll-dir` for the legacy HDF4-container NeXus
files. Outputs are under an external scratch root; only the verdicts are
recorded here. This is the Sonnet counterpart to the Codex/Luna gate above,
scored against the same four unchanged rubrics.

The runner had never been exercised on Windows, and three of its assumptions
were POSIX-only: the console scripts are `*.exe`, the agent stream is UTF-8
rather than the ANSI code page, and Claude Code matches permission rules
against Windows paths in POSIX form with the drive as a segment
(`Read(//c/Users/.../data/**)`). All three are fixed in the runner, which also
gained `--hdf4-dll-dir` and now records `"agent": "Claude Code"` in
`cost.json`. A Haiku smoke run confirms the staged data copy is readable, a
write into it is refused, and the HDF4 survey loads.

#### Pass 1 — 2026-09-16

| Dataset/workflow | Wall | Turns | Cost | Verdict |
|---|---:|---:|---:|---|
| TCNQ ALC / `integral-scan` | 125.9 s | 19 | $1.06 | **pass** (6/6) |
| Silicon photo-µSR / period selection | 509.2 s | 50 | $2.88 | **pass** (6/6) |
| Al-LLZ ionic motion / `fit-global` | 326.4 s | 36 | $1.94 | **fail** (2 Musts) |
| CdS shallow donor / `fourier` | 546.2 s | 64 | $2.51 | **fail** (2 Musts) |

- **alc-tcnq — pass.** All four 31-run blocks integrated with `integral-scan`
  and fitted `LorentzianLCR + Cubic`; the fitted centres, widths and reduced
  χ² match the passing Luna run to the decimal, including the two blocks whose
  centre sits a few gauss below the rubric's 3.0–3.2 kG window. Uncertainties
  came from reading the stored scan JSON, not from arithmetic: its one attempt
  to compute in `python -c` was refused by the allow-list and recorded as a
  permission denial. Gap (Should): the integration window and count-integral
  method are not named.
- **photo-musr-silicon — pass.** Both periods reduced into separate work
  directories, ON/red relaxing at 2.11 μs⁻¹ against the dark gate's
  0.146 μs⁻¹, the model and fit window stated, no carrier lifetime claimed,
  and the unexposed `P scan` coordinate called out as a metadata limitation.
  It then went further than the rubric asks and resolved the scan itself from
  its own `fit-series`: Λ falls by an order of magnitude and resets twice, a
  power ramp repeated across the block.
- **ionic-motion-llz — fail** on the model Musts. All 13 triplets were fitted
  with `fit-global` sharing the right parameters and taking `B_L` per run, but
  the model was `DynamicLorentzianKT + Constant` — the wizard's AICc pick on
  the 10 G run — rather than the Gaussian/`Keren` family the physics of a
  dense nuclear-moment garnet calls for. The reported width is therefore an
  `a_L` of 0.16 μs⁻¹ where the rubric expects `Delta` of 0.25–0.45, and
  nothing is flagged as suspect. The summary also says "14 temperatures" where
  the survey printed 13 and its own table lists 13.
- **cds-fourier — fail** on the interpretation Musts. Run 20721 was selected
  correctly from the event totals, reduced, and transformed unwindowed with a
  saved plot, and the warm reference 20729 gave a tabulated line at
  1.3916 MHz. But the cold run's empty peak table was reported as physical
  absence — "no resolvable structure", "broadband" — when the same session's
  survey reports that run precessing at the Larmor frequency with SNR 87 and
  its own time-domain fit gives a 21.6 % oscillation damped at σ = 0.554 μs⁻¹.
  The decisive PNG it read was a 0–10 MHz full-record view, which spreads the
  damped line across a few bins among noise maxima of similar height; the
  windowed plot of the same run had already been overwritten by it, since
  `fourier --plot` writes `plots/run-<n>.png` every time.

Skill changes after pass 1 (text only; no rubric and no CLI change):

1. **Which Kubo–Toyabe is a physics choice, not an AICc one.** The decision
   rule now separates the **Gaussian** KT of a dense nuclear-moment compound
   (`StaticGaussianKT`, `DynamicGaussianKT`, `Keren` in LF) from the
   **Lorentzian** KT of dilute, randomly sited moments, says that motion is
   the rate `nu` on top of a Gaussian `Delta` rather than a reason to change
   distribution, and requires the summary to name the KT used and why.
2. **Reconcile a missing line with what the session already knows.** Step 5c
   now requires an empty peak table to be checked against the `survey`
   precession column and the run's own time-domain fit — a fitted oscillation
   damped at σ is a line of width of order σ/2π, broadened past the detector
   but present — and to be re-transformed with `--fmin`/`--fmax` a few
   linewidths around the expected frequency when they disagree. It also warns
   that a second transform of a run replaces its PNG.

#### Pass 2 — 2026-09-17, the two failures on the corrected skill text

| Dataset/workflow | Wall | Turns | Cost | Verdict |
|---|---:|---:|---:|---|
| Al-LLZ ionic motion / `fit-global` | 330.9 s | 37 | $1.90 | **pass** (6/6) |
| CdS shallow donor / `fourier` | 436.3 s | 42 | $2.19 | **pass** (6/6) |

- **ionic-motion-llz — pass.** The wizard still ranked `Dynamic Lorentzian KT`
  above `Dynamic GKT` on the 10 G run (AICc 1950 against 1962) and the agent
  overrode it in as many words — the garnet's ⁷Li, ²⁷Al and La nuclei are a
  dense array, so the distribution is Gaussian — then hand-built a
  `DynamicGaussianKT + Constant` recipe and fitted all 13 triplets with
  `fit-global`, sharing `A_1`, `Delta`, `nu`, `A_bg` and taking `B_L` per run.
  The 160 K group gives `Delta` = 0.351 ± 0.002 μs⁻¹ and `nu` = 0.344 ± 0.006,
  both inside the rubric's windows; `Delta` is flat and `nu` rises 3.3× to
  404 K, and no activation energy is claimed.
- **cds-fourier — pass.** The cold run's peak table is still empty, and the
  summary now says so *and* reads the plot: three visual maxima at roughly
  1.24, 1.39 and 1.51 MHz, marked visual-only, roughly symmetric about the
  warm reference's tabulated 1.3916 MHz line and absent from that reference.
  It saved the reference spectrum under its own name rather than letting the
  second transform overwrite the first PNG. Its time-domain fit corroborates
  the reading independently: a two-oscillator model splits 1.2517/1.5269 MHz
  at 5.2 K, converging to a single line by 30 K, with the three unconverged
  runs reported as failed rather than as results.

**Sonnet workflow-expansion gate: all four datasets pass.**

### Trigger check — 2026-09-15

Six one-shot runs (`--max-turns 3`) on a copy of the PTFE folder, scored on
whether the agent issued a `Skill` tool call for `asymmetry-analysis`:

| Prompt | Fired |
|---|---|
| the plan's fixed sentence | yes |
| "fit the zero-field runs in here" | yes |
| "what's in these .nxs files" | yes |
| "reduce runs … and plot the asymmetry" | yes |
| "summarise this CSV of DFT energies" | no |
| "plot the XRD pattern in this folder" | no |

4/4 should-fire, 0/2 should-not-fire. The description was not changed.

### Corpus capability audit — 2026-09-23

Host: Claude Code, lead model Opus with five parallel audit subagents. This is
not a scored evaluation: each agent read the worksheet (or paper) and logbook
of the experiments the gates above never covered, then **ran the CLI on
representative runs** and classified every analysis step as supported,
partial, missing or out of scope, naming the smallest addition that would
close each gap and whether it is a core or a CLI/skill gap. Scratch outputs
stayed outside the repo. Every one of the 21 data folders loads in `survey`;
the gaps are all in analysis.

| Experiment | Verdict | Blocker or main gap |
|---|---|---|
| LiFeAs (PSI GPS `.bin`) | blocked | TF signal sits in the Up/Down pair but the loader pairs Back/Forw on these 5-histogram files; no CLI grouping choice; PSI background subtraction not exposed (`BACKGROUND_MODES = ("none",)`) |
| Muonium + maleic acid | blocked | the skill declines muonium kinetics although `core/fitting/mu_kinetics.py` fits k_Mu and Arrhenius; no concentration axis |
| Benzene: RF resonance | blocked | the skill declines RF although `build_rf_difference_scan` and `RFResonanceMuP` exist; green−red difference is GUI-only |
| Ca₃Co₂O₆ plateau (Redfield) | workarounds; headline blocked | λ(B) reproduces the paper's Fig. 2a; the core `Redfield` trend fit on that trend gives τ ≈ 0.91 ns and Δ ≈ 40 mT (paper 880 ps, 40.6 mT) but no command runs it |
| Critical fields in Sn | workarounds; headline blocked | Hc(T) needs a trend fit and the logged sample temperature (setpoint is 1–6 K off); LF wizard scope omits precession |
| Copper (diffusion, QLCR) | workarounds; E_a blocked | trend fit; the wizard never offers Abragam; logged vs setpoint temperature |
| Molecular antiferromagnet | workarounds; T_N blocked | trend fit; FFT peak table misses the 2.55 MHz line |
| EuO (PSI GPS) | workarounds; β blocked | trend fit; background subtraction; logged T lives in the `.bin` header |
| TRSB Re₆Zr | workarounds; gap fit blocked | ZF Δ(T) step reproduces the paper's Fig. 4; the s-wave fit needs a trend fit; `trend` crashes on a `fit-global` series |
| Basics | workarounds | `emu00044989` and `MUSR00044989` share run numbers, so every command refuses the folder; no t0/t_good offsets or custom x axis |
| Corannulene | workarounds | whole-scan ALC multi-resonance fits fail (suffixed `B0_n`/`Bwid_n` unbounded, polynomial seeds on a gauss axis); no radical repolarisation model; poor scan grouping over 383 runs |
| Benzene: high TF, ALC | workarounds | FFT resolves the 208.6/305.6 MHz radical pair; co-add and the correlation spectrum are core-only; coupling-parameterised ALC models missing |
| Benzene: repolarisation | fully analysable | `MuRepolarisation` gives A_hf directly; only `integral-scan --deadtime` missing |
| AFM transition in high TF | rubric stale | FFT and two-line tracking of the 6 T and 8 T `.mdu` scans work; only MaxEnt, wing area and DFT need declining |

Cross-cutting gaps, ranked by how many experiments they unblock:

1. **No trend-model fit in the CLI.** The last step of about eight
   experiments; the core `fit_parameter_model` already carries Redfield,
   OrderParameter, Arrhenius, SC_SWave and Linear. (CLI only.)
2. **Series axis limited to setpoint temperature, field or run.** Sn, copper
   and EuO need the logged sample temperature; maleic acid and Basics need a
   per-run value the user supplies.
3. **`fit-global` is a dead end.** `trend` raises `KeyError: 'trend'` on its
   stored fit, and the human output hides the run-local parameters.
4. **Reduction options the core has but the CLI does not expose:** detector
   pair/grouping, background subtraction by range, t0/t_good offsets, co-add,
   the green−red period difference, `integral-scan --deadtime`.
5. **The FFT peak table empties when zoomed:** `core/workflow/fourier.py`
   crops to `--fmin/--fmax` before estimating the noise, which is exactly the
   re-transform the skill prescribes (benzene, HAL `.mdu`, molecular AFM,
   LiFeAs, EuO).
6. **Recipe authoring:** no expression → recipe command, the wizard saves only
   its recommendation, a wrong parameter name gives a traceback, and the skill
   documents a `fit-series --tmin/--tmax` that does not exist.
7. **Survey scan grouping:** merges samples, geometries and time-separated
   segments; does not recurse into sub-folders (benzene); reports a false
   ≈0.095 MHz `other` line for sub-cycle signals.

Also found: the skill's rule that a `spurious_reseeded` flag disqualifies a
run discards good copper points; missing core models (anisotropic radical
repolarisation, coupling-parameterised ALC D0/D1, the analytic RF
approximation, time-domain QLCR); the ARGUS `t0_bin`/`time_zero` warning is a
false positive from a float32 bin width (8 ns effect, 38 warnings per survey).
A reported MUSR forward/backward sign flip was checked and is not real: MUSR
runs reduce to positive asymmetry; only the preset's group naming disagrees
with the file's.

Rubric corrections: `afm-high-tf-mdu` should expect a partial analysis, not a
decline (the survey reads the fields and temperatures it calls unknown);
`euo-psi` asks for "relaxation consistent with critical slowing down", which
the paper contradicts (λ stays near 2 MHz).

Follow-up: gaps 1–3 and the two rubric corrections are taken up on
`feat/trend-model-fit`: `trend --model` (the desktop trend dialog's fit, every
row with a value entering unless `--exclude`d, flagged rows named), `--order
sample_temperature_logged` and `--order <name> --x RUN=VALUE,…` on `fit-series`
and `fit-global`, and a stored trend for `fit-global`. Checked on the corpus:
the plateau λ(B) Redfield fit (D = 27.5 ± 0.4, ν = 159 ± 15 MHz over
5–36 kG), Sn ordered by logged temperature, and maleic acid λ_Mu against a
supplied concentration axis through `fit-global` and `trend --model Linear`. Reading the logged temperature from the PSI `.bin`
header is deferred: the header's per-sensor means carry no labels, and which
sensor is the sample differs between GPS and GPD.

### Reduction options — 2026-09-25, on `feat/cli-reduction-options`

Audit gap 4, taken up after the maintainer set the order of work: skill text,
rubrics and Sonnet waves wait until the CLI unlocks the whole corpus, so this
change is CLI and core only and the skill still declines RF resonance.
`reduce`, `alpha` and `integral-scan` share `--pair`, `--deadtime`,
`--background tail_fit|range[:FIRST:LAST]`, `--t0-offset`, `--t-good-offset`
and `--period …|green-red` (`cli/_reduction.py`, one `ReductionSettings`);
`survey --pair` measures precession on the named groups; `info` lists groups.
Integral scans now form their counts through `corrected_grouped_counts`, so
they take deadtime (GUI included) but still no background — the subtracted
level's error is correlated across the window and is not propagated.

Checked on the corpus (scratch outputs only):

| Experiment | Before | Now |
|---|---|---|
| LiFeAs (GPS) | blocked | `--pair Up/Down --background range`: 5.41 MHz line at 400 G; σ 0.147 → 0.69 µs⁻¹ from 18 to 1.5 K, diamagnetic shift below ~16 K; `survey --pair Up/Down` finds `larmor` on the scan |
| Benzene RF (DEVA) | blocked | `integral-scan --period green-red --deadtime from_file --tmin 0.1 --tmax 4 --model RFResonanceMuP --fix nu_RF=218`: A_μ 514.8, A_p 126.0 MHz, χ²ᵣ 1.08 |
| Benzene repolarisation | "fully analysable" | `--deadtime from_file` works, but the curve is not one isotropic-hyperfine repolarisation (χ²ᵣ ≈ 460 with the six TF100 calibration runs excluded): needs the radical repolarisation model the audit listed as missing |
| EuO (GPS) | background missing | `--background range` accepted |
| Basics t0 | missing | a 10-bin `--t0-offset` moves a 100 G TF phase by 1.415 rad (2π·f·Δt = 1.411); the folder still needs splitting by prefix |

Found along the way, not fixed here: `recipe --run` seeds `A_1` from A(0),
which is negative on a pair whose TF phase is near 90° and sits below the
amplitude's own lower bound, and never seeds `frequency` from the field
(LiFeAs fits failed until `--initial` set both); the RF couplings start from
benzene's defaults and do not converge from a start ~80 G off (seeding them
from the scan needs ν_RF inside the estimator); under a t0 shift the From-file
good window keeps its absolute last bin while a Manual window re-derives it in
aligned coordinates (3 bins apart on the simulated runs); the survey does not
itself notice that another pair carries the precession.

### ALC scans — 2026-09-25, on `feat/cli-alc-scans`

CLI and core only, like the reduction-options PR. A study subagent mapped the
passes of the benzene (liquid, solid, solution), corannulene and TCNQ ALC
folders; every change below was then checked on those folders and the survey
compared before/after on all 27 corpus folders.

- **Multi-resonance fits.** LCR components are seeded one per resonance
  (largest excursion from a robust straight baseline, bracketed by half-height
  crossings on both sides, ±3 HWHM masked before the next) and bounded inside
  the scan with a width up to a quarter of it; `Quadratic` gained its data
  seed. `integral-scan --xmin/--xmax` fits a window, an underdetermined window
  is refused, and a failed fit is printed with its parameters, scan kept.
  Results: solid 19490/21467 G; liquid o-p 28945/29539 G; TCNQ 350 K 3099 G;
  corannulene 420 K 7076 and 14824 G in windows. Whole-scan corannulene fits
  still fail: its background is a rise plus a step at 19–21 kG, not a cubic.
- **Loader.** HiFi red/green good frames now come from `frames_period_daq`
  (red read 0 → deadtime blew its asymmetry from 0.41 to 0.55); Z-sweep runs on
  a persistent main field read `Field_Main` + offset (benzene solution sat at
  ±350 G; now ~20.9 kG).
- **Green − red** is the difference of each period's count integral, so the RG
  differential ALC is clean (dips at 28.94/29.54 kG) and the RF fit needs no
  window (A_μ 514.8, A_p 124.4 MHz over the full good range).
- **Survey.** Field scans keyed by note and period count; a Larmor-precessing
  minority (calibrations) left out; a field re-measured after another
  temperature starts a repeat; grid cross-sections counted, not listed.
  Across the corpus this split the benzene and corannulene folders correctly,
  kept LiFeAs's alternating 2 K/20 K field scans whole, and dropped only spurious
  scans elsewhere (a photo-μSR power scan at −100 G read as a field scan).

Not done here: a model for the red/green differential line shape (the
worksheet's "Deriv Lorentzians"); HiFi files stamped `TF` on longitudinal ALC
runs still report geometry TF where no precession is measurable; a
single-period `integral-scan --period red` lists encoded period run numbers
(`29809001`); background in integrals; coupling seeds for `RFResonanceMuP`;
the radical repolarisation model; corannulene's step baseline.

### Open corpus items — 2026-09-25, on `feat/cli-open-items`

CLI and core only, like the two PRs before it: every item #334, #335 and the
corpus audit left open except the pass-17a evaluation items. Plan and decisions
D1–D6: `docs/plans/cli-open-items.md`. Each phase ran as a subagent; the lead
read every diff before running it, then the phase's tests and corpus check
(scratch outputs only).

| # | Item | Corpus result |
|---|---|---|
| 1 | Period run numbers | benzene RG `integral-scan --period red` and `alpha --period red` name 29809, not 29809001 (a run-ordered x axis too) |
| 2 | `recipe --run` seeding | LiFeAs 3366 (`--pair Up/Down --background range`): A_1 = 14.6 > 0, phase π, frequency 5.4216 MHz; the 400 G `fit-series` converges on all eight runs without `--initial` |
| 3 | Survey of sub-folders | benzene `data/` names its three run-holding sub-folders and exits 1; `ALC resonance/` names liquid, solid, solution |
| 4 | Two instruments in one folder (D1) | Basics refused without `--instrument`, naming EMU and MUSR (nine shared run numbers); `--instrument EMU --runs 18850-18863` reduces 13 runs; alpha per file (emu 44989–44997 1.09–1.11, MUSR 44989 1.23); a 10-bin `--t0-offset` moves the 100 G phase by 1.415 rad (2π·f·Δt = 1.411) |
| 5 | Background in integrals | LiFeAs `--background range`: errors grow as they should (0.00046 → 0.00128); magnitudes need not, since a TF pair's background carries its own asymmetry. No-background numbers are bit-identical to before |
| 6 | RF coupling seeds | benzene DEVA, `--fix nu_RF=218`: seeds A_μ 515.2, A_p 133.7; fit 514.775 ± 0.034, 124.4 ± 1.4 MHz, χ²ᵣ 1.41 over the full good range (1.17 over 0.1–4 µs); from `--initial A_mu=470 --initial A_p=90`, which failed in #334, the same minimum |
| 7 | Differential ALC line shape | benzene liquid RG with `LorentzianLCRPair ×2 + Constant --fix dB_n=44.4`: B0 28938.6 and 29536.8 G, HWHM 12.8 and 10.4 G, χ²ᵣ 1.49 (the plan expected HWHM ≈ 14, χ²ᵣ ≈ 1.3); measured red − green offset −44.38 G over 43 runs; `dB` free is degenerate, as expected |
| 8 | HiFi geometry from coils (D4) | every HiFi ALC scan now LF from `coils` (liquid 258, solid 122, solution 360, corannulene 359 runs); TF20 calibrations stay TF measured |
| 9 | PSI logged temperature (D3); implausible NeXus T | EuO 2935 → 52.76 K, 2923 → none (133.5 ± 34 K against 5 K), LiFeAs 3366 → 1.739 K; EuO 49/51 and LiFeAs 34/34 runs carry a reading. Basics: MUSR ~100 K is genuine (the header's 290 K setpoint is stale); emu ~644 K (a furnace-channel echo) and EMU 943.48 K (a railed sensor) are **not** fixed — see below |
| 10 | Co-add; correlation spectrum (D2) | benzene 3678–3682 co-added: FFT lines 40.7, 208.6 and 305.6 MHz; correlation peak 514.1 MHz (SNR ≈ 600) |
| 11 | Batch of groups; law through trend fits (D6) | maleic acid, nine groups (per-block `--alpha-from` across the 78280/78281 step, `--strategy least_squares`): k_Mu per group 2.05–2.94 (278–338 K); Arrhenius over all nine pins Ea at 0 (LAW NOT ESTABLISHED), ≤ 330 K gives Ea = 57 ± 26 meV (5.5 ± 2.5 kJ/mol, χ²ᵣ 1.14); the worksheet asks only for a comparison with Ng et al. (1981). Al-LLZ: 13 LF triplets as a batch, ν rising 0.33 → 1.10 MHz above ~260 K; `Arrhenius + Constant` Ea 256 ± 16 meV |
| 12 | `MuRepolarisation` seeds (D5) | benzene repolarisation, two terms from seeds alone: A_hf 943 and 5429 MHz, χ²ᵣ 25.9 (one term: 460) |

Lead corrections during review: one `source_run_of` in place of two private
copies; the run-ordered integral-scan axis decoded too; `dataset_nyquist_mhz`
without its silent fallbacks; the red/green Hall step recorded in probe units
and converted by the slope of `Field_Main` against `Field_Hall_Z` over the
scan's runs — the plan's per-run ratio of means carries the probe's ~1 kG zero
offset and read 42.98 G; LCR and repolarisation group seeding walked through
one table (`_GROUP_SEEDERS`); corpus run numbers removed from tests; the
phase's `SKILL.md` edit reverted (skill text waits for the skill pass).

Survey before/after on all 28 corpus folders (a worktree of `main` against the
branch): geometry changes only in HiFi folders — the ALC folders above, plus
the plateau system, photo-µSR silicon and Sn, all LF from their coils; logged
temperatures appear on the PSI folders (EuO, LiFeAs, benzene high TF) and shift
by 0.1–2.3 K on several NeXus folders now that 0 K dropout samples are left out
of the mean (benzene RF 290.7 → 293.0 K); the two benzene parent folders name
their sub-folders. No scan was split or merged. The benzene liquid RG scan now
shows geometry `-`: three of its runs log no coils and keep their TF stamp.

Not done, and why:

- **Railed and echoed NeXus sample temperatures** (Basics emu ~644 K, EMU
  943.48 K). No rule separates them from real readings: a steady real log is
  also one value over ~100 samples, the controller echo matches exactly in
  only 3 of 11 runs, and a factor-of-two gate hides the genuine MUSR 100 K. The
  survey's temperature-departure note flags them.
- **Reference-run background in integrals.** The integral transform has no
  reference loader, so such a run (and one whose tail fit fails) is excluded
  with the reason — in the GUI's integral scans too — rather than integrated
  unsubtracted.
- `.mon` sidecars never set `sample_temperature_logged` (no corpus folder has
  one); the PSI header sensor stands aside when a sidecar loads.
- `combine_runs` drops the members' instrument metadata; the CLI co-add
  overlays the first member's, the GUI's co-add does not.
- The correlation spectrum uses the entry's forward/backward pair, not every
  group; the peak table splits some lines into twins (pre-existing).
- RF dips closer than ~3 widths give no coupling seeds; the GUI's RF fit is
  still single-start.
- Survey JSON still names alpha steps, temperature departures and the best
  calibration run by run number alone (the text output labels instruments).
- For the skill pass: `SKILL.md` still says batches of global fits and laws
  through trend fits have no command; the maleic batch needed `--strategy
  least_squares` (the default `joint` failed two of nine groups).

### Sonnet trend-fit pass — 2026-09-23, on `feat/trend-model-fit`

Host: Claude Code. Model: `sonnet`. Runner: `tools/agent_eval/run_eval.py`,
the fixed prompt, `--max-turns 80`, macOS. The first run of the skill text
that teaches `trend --model` and the new scan axes, on four audit cases.
Three rubrics (`plateau-redfield`, `sn-critical-field`, `maleic-mu-kinetics`)
were written from the worksheets and handout before any run.

| Dataset | Wall | Turns | Cost | Verdict |
|---|---:|---:|---:|---|
| EuO (PSI GPS) | 1016 s | — | $3.99 | **pass** (5/5) |
| Ca₃Co₂O₆ plateau | 555 s | 46 | $2.48 | **fail** (Redfield Must) |
| Critical fields in Sn | 705 s | — | $2.56 | **fail** (sample-precession Must) |
| Maleic acid | 1188 s | — | $3.61 | **fail** (3 Musts) |

- **euo-psi — pass.** The first real use of the new path:
  `trend --model OrderParameter --param frequency --exclude <6 flagged runs>`
  over 1.5–69.3 K gave Tc = 69.17 ± 0.05 K, β = 0.443 ± 0.004,
  α = 1.54 ± 0.02, with the excluded runs named. ZF and TF blocks separated;
  paramagnetic Gaussian-KT Δ flat; the TF relaxation rising near Tc is
  reported from its own fits. Gaps: it said no trend law suits a rate
  diverging at Tc (`CriticalDivergence` exists; Step 6a's list omits it); β is
  not labelled as measured against the setpoint; the wizard's fluorine hint
  fired on the title `TF60G`.
- **plateau-redfield — fail.** Alpha from 9023, the cooldown in 9024–9030
  found from `T log/K`, λ(B) falling through the sweep — but a stretched
  exponential and a descriptive trend, never `trend --model Redfield`. The
  agent sees only the data and logbook, not the handout; nothing in the skill
  says that an LF decoupling scan of a fluctuating magnet is the case
  Redfield's law is for.
- **sn-critical-field — fail.** Excellent thermometry (the 91501–91515
  excursion to ≈8.2 K found and ordered by logged temperature, the 40 G
  transition bracketed at logged 2.8–3.2 K) but analysed as LF decoupling
  with a Gaussian envelope: the sample's precession at γ_μH_c (≈1.9 MHz) was
  never looked for. The survey says `prec none` and the LF wizard scope has
  no precession template; the skill has no type-I intermediate-state physics.
- **maleic-mu-kinetics — fail.** Found the alpha step at 78281 and the
  per-sample structure, but did not recognise the 2 G runs as muonium
  precession (γ_Mu/2π ≈ 1.39 MHz/G, a line near 2.8 MHz); fitted Kubo–Toyabe
  to "water protons", reported a 17 MHz "doublet" the wizard matched, and
  wrote the titles' relative concentrations as molar ("0.25 M"). No rate
  constant. The skill has no weak-TF muonium physics, and the wizard's
  low-field muonium matcher needs a resolved doublet (audit gap).

Skill changes proposed after pass 1 (made in pass 2 below): name `CriticalDivergence`
in Step 6a; a "which law for which scan" table in section 5 (LF decoupling of
a dynamic magnet → Redfield; activated hopping → Arrhenius; order parameter →
OrderParameter; σ(T) → SC_*; rate vs concentration → Linear); type-I
intermediate state (precession at H_c independent of the applied field,
look with `fourier` even in LF); weak-TF muonium (triplet line at
1.394 MHz/G, the diamagnetic line barely a cycle, relative concentrations
stay relative).

#### Pass 2 — 2026-09-23, after `recipe`, `wizard --include/--exclude` and a system-first skill

Between the passes: `asymmetry recipe` (a recipe from an expression, printing
every parameter name), `wizard --include/--exclude` (the engine's existing
scope overrides), the fluorine sniff no longer firing on `TF60G`/`ZF`/`LF100`,
and skill text — Step 3b (decide the system class before screening; a table
from class to scope and trend law), `CriticalDivergence`, LF Redfield, the
type-I intermediate state and weak-TF muonium. Examples use placeholder run
numbers so the skill does not carry this corpus's answers.

| Dataset | Wall | Cost | Wizard calls (pass 1 → 2) | Verdict |
|---|---:|---:|---:|---|
| Ca₃Co₂O₆ plateau | 902 s | $1.55 | 3 → 2 | **pass** (5/5) |
| EuO (PSI GPS) | 1192 s | $2.49 | 16 → 3 | **fail** (internal-field Must) |
| Critical fields in Sn | 1219 s | $3.31 | 1 → 4 | **fail** (sample-precession Must) |
| Maleic acid | 1178 s | $2.42 | 5 → 3 | **fail** (3 Musts) |

- **plateau-redfield — pass.** Decided "LF decoupling", fitted single
  exponential λ(B), then `trend --model Redfield --param Lambda --fix m=2
  --xmax 25000 --exclude 9049`: D = 31.78 ± 0.53 MHz, ν = 159 ± 11 MHz, range
  and exclusion stated. Gaps (Should): fitted 1–25 kG rather than the plateau;
  no comment on the plateau edges.
- **euo-psi — fail (regression).** Three wizard calls with physics-chosen
  scopes, as intended — but it screened only the 200 K paramagnetic run and
  chained that Gaussian-KT model through the whole ZF scan, so the ordered-state
  precession pass 1 fitted (30 MHz at 1.5 K) was never looked for, and the
  summary calls the ordered-state field "too fast to resolve". The skill does
  not say that a scan crossing a transition needs a model on each side,
  screened on each side; pass 1 did that unprompted.
- **sn-critical-field — fail.** Classified as a type-I superconductor and
  screened with `--geometry LF --scope lf-dynamics --include Oscillatory` plus
  `fourier`, as the skill now says — on run 91488 (20 G), where the spectral
  search finds no line and the wizard drops the included oscillatory
  candidates for "no support in the spectrum". `--include` widens the scope;
  it does not force a candidate past the wizard's spectral gate. On 91516
  (40 G) the same options rank the oscillatory model first. The sample line is
  ~0.3 % against ~20 % background, so the run chosen decides the outcome.
- **maleic-mu-kinetics — fail.** Now tests the weak-TF muonium hypothesis
  explicitly (`fourier` with `--tmax` crops) — but took alpha = 1.401 from the
  survey's best candidate 78281 for the whole folder, while 78251–78280 need
  ≈1.03–1.07. The deoxygenated water (A(0) = 0.07 %) and the whole "neat"
  series were therefore mis-reduced, and the collapse was read as chemistry. It
  looked for Mu in untreated water (O₂ relaxes it) and the mis-reduced neat
  runs, never in correctly reduced deoxygenated water. Wrote "0.5 M" again.

Outcome and what it says: the system-first step cut screening (EuO 16 → 3
wizard calls) and produced the first Redfield fit; single runs are noisy
(EuO passed then failed on the same dataset). Remaining gaps, in order:
(1) calibration — measure alpha on every candidate and reduce each block with
its own when they differ (the survey could print alpha per candidate and flag
a step: CLI); (2) a scan through a transition needs a model and a screen on
each side; (3) `--include` should force its components' candidates past the
spectral gate (core/CLI); (4) weak-TF muonium: look first in the
lowest-scavenger, deoxygenated sample.

#### Pass 3 — 2026-09-23/24, two repeats per dataset

Between passes: `survey` gives each calibration candidate its own alpha and
flags an `ALPHA STEP`; skill text for alpha blocks, a screen and a series on
each side of a transition, seeding a rejected physics component with
`recipe --initial frequency=…`, and where to look for weak-TF muonium.
`--include` was deliberately *not* forced past the wizard's spectral gate: on
Sn run 91488 the included oscillation was fitted and collapsed to the 1/T
resolution floor, a genuinely bad fit. Runs scored by a scoring subagent
against the unchanged rubrics, with the number rule checked in the
transcripts.

| Dataset | 3a | 3b |
|---|---|---|
| plateau-redfield | pass | pass |
| maleic-mu-kinetics | pass | pass |
| euo-psi | pass | fail (internal-field Musts) |
| sn-critical-field | fail | fail |

- **maleic** now passes twice: `ALPHA STEP` respected, hand-written Mu recipe,
  `trend --model Linear` on λ_Mu against supplied concentrations.
- **euo 3b** screened inside the transition cluster; the wizard's recipe for
  the 10 K run held `frequency_1 = 29.89 MHz` but its text only said "3
  line(s) detected", and a `fourier` over 0–20 MHz printed "No peaks", so the
  summary claimed no ordered-state oscillation. It also used "critical
  slowing" after both `CriticalDivergence` fits failed.
- **sn 3a** trusted the survey's setpoint-grouped 23-run scan and never
  reached the type-I row; **sn 3b** handled the logged temperature but its
  `fourier --fmin 0.3 --fmax 6` printed "No peaks" over a band whose strongest
  maximum was the 2.16 MHz normal-domain line, and it fitted Redfield to Λ(B).

Changes after pass 3 (all general, none dataset-specific): `fourier` detects
on the whole spectrum then restricts to the band, and lists the strongest
sub-threshold maxima as candidates; `wizard` prints its detected lines and
the recommended fit's values; `survey` prints a `TEMPERATURE:` line for runs
whose logged temperature departs from the setpoint (flags real departures in
Sn, the plateau cooldown and the EMU cuprate runs, none in nickel or YMnAl);
`wizard`/`fit-series` take `--tmin`/`--tmax`; skill text that a failed trend
law does not supply the physics and that derived σ, % and ratios are numbers.

#### Pass 4 — 2026-09-24, two repeats

| Dataset | 4a | 4b |
|---|---|---|
| plateau-redfield | pass | fail (Redfield on one rate of a two-exponential model; a hand-computed %) |
| euo-psi | pass | fail (number rule only: a hand-computed "~9 %") |
| sn-critical-field | fail | fail |
| maleic-mu-kinetics | fail | fail |

3/8, down from 5/8 — run-to-run variance is large. The new outputs were read
where they were decisive: `TEMPERATURE:` caught the Sn 8 K block and the
plateau cooldown in every run, `ALPHA STEP` was respected in both maleic runs,
EuO quoted the wizard's printed 30 MHz line. Fourier candidates were ignored.
Root causes: Sn — the wizard printed the 1.913 MHz line but the recommendation
was a relaxation and no seeded fit followed; the survey's merged 23-run 40 G
"scan" was taken as the scan. Maleic 4a took untreated water for the blank and
abandoned Mu for the folder after one failed fit; 4b printed k_Mu from
`trend --model Linear` (χ²ᵣ 23) and then withheld it. Plateau 4b fitted
Redfield to `Lambda_1` of a two-rate model because AICc preferred it.

A deeper cause surfaced while fixing these: the survey's `other` verdict named
a ~0.1 MHz "line" on nearly every weak-field and decoupling run — relaxation
leakage completing under a cycle — while the fingerprint's damped-line scan
held the real line (2.806 MHz, SNR 68 on the Mu blank; 1.91 MHz on Sn). Fixed
in the survey, with a unit test; calibration candidates on the corpus are
unchanged and the false copper/YMnAl `other` lines are gone. Also: seeded
`recipe` commands printed by `wizard`/`fourier`, √χ²ᵣ-scaled errors and a
multi-component warning in `trend --model`, and skill text (a single-component
`--param`; report a poor-χ²ᵣ law with its caveat; one negative run is not a
negative folder; no invented units).

#### Pass 5 — 2026-09-24, two repeats

| Dataset | 5a | 5b |
|---|---|---|
| plateau-redfield | fail (no Redfield: read the LF scan as decoupling a static field) | pass |
| sn-critical-field | pass | pass |
| euo-psi | fail (number rule: "~14 %", count ratios) | fail (number rule: MHz → G by hand) |
| maleic-mu-kinetics | fail (number rule: ratios, "17 K hotter") | fail (number rule: setpoint offsets; "0.5 %" unit) |

Physics Musts held in 6/8 — Sn passed twice for the first time, both runs
seeding their recipes from the survey's new `other@<MHz>` lines and fitting
H_c(T) with `OrderParameter` (α=2, β=1). Four of the five failures were hand
arithmetic in the prose; the skill's rule, buried in section 6, had not moved
that behaviour in two passes. Changes: Step 7 becomes an explicit number audit
naming the usual offenders; the survey's `TEMPERATURE:` line prints the
offsets agents were computing; `trend --model` leads with the √χ²ᵣ-scaled
error; the multi-component warning (a false positive on the Mu/diamagnetic
pair) becomes a note that distinguishes species from split rates; the wizard
stops listing sub-cycle leakage lines (still present in its payload); the LF
cue in the class table no longer depends on `prec none`. Rubric wording
fixed where a Must was unverifiable from a summary (EuO loader path) or too
narrow (plateau 9031–9034), and the README now states that arithmetic on
printed values counts under the number rule, as pass 4 already applied it.

#### Pass 6 — 2026-09-24, two repeats

| Dataset | 6a | 6b |
|---|---|---|
| plateau-redfield | pass | pass |
| sn-critical-field | fail (number rule "~3 %"; no H_c(T) — recipe seeded the whole asymmetry into the line and it collapsed) | fail (discarded the logged column after one min–max offset range hid the 6 K block; MHz → G, σ by hand) |
| euo-psi | fail (paramagnetic rates from runs still precessing; "critical slowing" after CriticalDivergence failed) | pass |
| maleic-mu-kinetics | pass | fail (no concentration axis; source of concentrations unstated) |

Plateau now passes 6 of its last 8 runs. The Step 7 audit as text still let
arithmetic through, so it became a tool: every command logs its printed output
to `<workdir>/cli-output.log`, and `asymmetry audit draft.md` lists the numbers
no command printed (a multiple or σ must match verbatim). Also: the survey's
`TEMPERATURE:` line lists consecutive blocks with offsets (relative threshold
1 %, which catches the 5–8 K maleic offsets at 300 K and flags nothing
spurious across the corpus); `trend --model` prints `LAW NOT ESTABLISHED`
when the fit failed, hit a bound or left an error as large as its value; skill
text splits a transition at the last unflagged ordered run and sets a weak
line's amplitude and background explicitly.

#### Pass 7 — 2026-09-24, two repeats (first with `asymmetry audit`)

| Dataset | 7a | 7b |
|---|---|---|
| plateau-redfield | fail (no Redfield: trended without `--model`) | fail (Redfield fitted in summary.md, dropped from the reply) |
| sn-critical-field | pass | pass |
| euo-psi | fail (no ZF internal field) | fail (no ZF internal field; no paramagnetic rate) |
| maleic-mu-kinetics | fail (untreated water as the blank) | fail (same; an audit-flagged number restored in the reply) |

Every run ran `audit` and ended with a clean `summary.md`, but the scored
reply was a fresh recap. Sn is now robust (both runs: `OrderParameter`
α=2, β=1 against logged temperature, 8 K block excluded). No EuO run found the
ordered-state line: the survey never searched zero-field runs. Both maleic
runs avoided the deoxygenated blank after the `TEMPERATURE:` line asserted it
was "not at its setpoint" — its block logs +17 to +25 K, including a neat
aqueous sample at 375 K, which points to the sensor. Changes: the survey
searches ZF runs for spontaneous precession (EuO 30.2 → 5.5 MHz then `none` at
T_c; nickel; the molecular AFM; nothing in paramagnetic, KT or F–μ–F runs);
the `TEMPERATURE:` line asks the agent to decide which to trust per block;
`LAW NOT ESTABLISHED` also for missing or zero errors; `trend` without
`--model` points at Step 6a; skill: the reply is `summary.md` as audited, the
Mu blank is the run noted deoxygenated.

#### Pass 8 — 2026-09-24, two repeats

| Dataset | 8a | 8b |
|---|---|---|
| plateau-redfield | fail (no Redfield) | fail (full analysis in summary.md; one-line reply) |
| sn-critical-field | pass | fail (dismissed the survey's falling line after failed time-domain fits) |
| euo-psi | pass | fail (number rule: derived percentages) |
| maleic-mu-kinetics | fail (k_Mu fitted in summary.md; one-line reply) | fail (declined k_Mu over block temperature offsets) |

The ZF search worked: both EuO runs got the order parameter right (Tc
69.76 ± 0.12 K in 8b). Three of six failures were delivery — agents read
their audited `summary.md` with a tool and replied with one line, taking the
tool call for the reply. Changes: the skill says the user sees only the final
message, which must be the summary text; `audit`'s clean pass says the same;
`audit` lists whole-number percentages and "factor of" ratios not printed
verbatim; `trend` without `--model` names the law for its axis and parameters
(Redfield on a field-ordered rate, with a warning on a split rate;
OrderParameter on a frequency; Linear on a supplied axis); skill: a caveat is
not a reason to withhold a fitted result.

#### Pass 9 — 2026-09-24, two repeats

| Dataset | 9a | 9b |
|---|---|---|
| plateau-redfield | pass | fail (shared the rate over a 17-field scan with `fit-global`) |
| sn-critical-field | fail (took the setpoint axis against the logged evidence) | pass |
| euo-psi | pass | pass (though it called the survey's cold-run lines "aliases") |
| maleic-mu-kinetics | fail (Arrhenius per sample; no concentration fit) | fail (reported, then withdrew, the converged slope) |

Every dataset now passes some of the time (≈50 % each since pass 5, maleic
lower). The reply-delivery fix held: no one-line replies. Changes: a
`survey_line_mhz` column beside fitted frequencies in `fit-series` trends; a
note when a series is ordered by the setpoint while the logged temperature
departs; a note when `fit-global` shares a rate over a many-field scan; a
converged poor-χ²ᵣ law says it is the result; skill text separating a
decoupling triplet from a field scan and making the logged temperature the
default axis; maleic rubric M3/M4 wording (matched setpoints; a caveated slope
counts, a withdrawn one does not).

#### Pass 10 — 2026-09-24, two repeats

| Dataset | 10a | 10b |
|---|---|---|
| plateau-redfield | pass | fail (screened the ZF end; chained fits with A_1 = 143 % made a spurious λ(B) "peak") |
| sn-critical-field | pass | pass |
| euo-psi | pass | pass |
| maleic-mu-kinetics | fail (number rule: "3.5-fold") | fail (withdrew the converged k_Mu slope a third time) |

5/8, the best so far. Over the last four runs: EuO 4/4, Sn 3/4, plateau 2/4,
maleic 0/4. Changes: `fit`/`fit-series` flag `amplitude_exceeds_data` when the
fitted amplitudes sum past three times the record's early-time |A| (robust to
precession); `audit` lists "-fold" and "N times"; `trend --model Linear` on a
supplied axis states that the slope is the rate constant to report; skill: a
sensor-trust decision holds everywhere those runs are used, and a scan is
screened mid-range, not at its zero-field end.

#### Pass 11 and a generalisation wave — 2026-09-24

| Dataset | 11a | 11b |
|---|---|---|
| plateau-redfield | fail (stated a fit range it did not fit) | pass |
| sn-critical-field | pass | pass |
| euo-psi | pass | pass |
| maleic-mu-kinetics | fail (number rule: hand-computed "~7–14 % errors") | fail (withdrew the converged slope, citing temperatures from a sensor it had judged faulty) |

Over the last six runs: EuO 6/6, Sn 5/6 (reliable); plateau ≈ 50 %; maleic
0/6, each failure now a single sentence.

Generalisation wave (one run each, same skill text): **Tier A 4/4 pass**
(nickel, PTFE, YMnAl, cuprate — no regression), hold-outs **molecular AFM
pass**, **copper fail** (TF line shape never compared; a low-T upturn in the
ZF hop rate narrated away and an Arrhenius law fitted across it). Additions
that helped: ZF `other@` lines (nickel, molecular AFM), the audit (nickel,
copper), `amplitude_exceeds_data` (nickel). That hurt: `LAW NOT ESTABLISHED`
on an undetermined nuisance prefactor (lost YMnAl's T_g = 84.4 ± 1.6 K), and
"it is the result" on a cuprate fit with scaled errors larger than its values
(judged on unscaled errors). Fixed: the law verdict uses scaled errors of the
physical parameters only; the report gives the fitted span, units and a
turning-point note; `amplitude_exceeds_data` at 1.5×; skill rows for spin
glass (freezing at the A(0) collapse), TF line shape for hopping, and
quadrupolar level crossings.

**Environment note.** The eval copies (~4 GB in the scratchpad) pushed the
system disk to 93 % and iCloud offloaded ~2,200 corpus files ("dataless",
reading as empty). The scratch copies were deleted and the runner now refuses
a copy shorter than its source; some target files stayed offloaded.

#### Pass 12 — 2026-09-24, two repeats

| Dataset | 12a | 12b |
|---|---|---|
| plateau-redfield | pass | pass |
| sn-critical-field | fail (number rule: a hand-summed A(0) the audit matched to a logged time bin) | fail (no falling frequency: the wizard's line hint replaced the model and the fit fell to a 0.025 MHz branch) |
| euo-psi | pass | pass |
| maleic-mu-kinetics | pass | pass |

6/8, the best pass; maleic passed both runs for the first time (k_Mu reported
with a caveat in both). Changes: the wizard's line hint adds the line to the
recommended model with a small amplitude; `frequency_unresolved` flags a free
frequency under two cycles in the informative window; `audit` ignores bulk
arrays in the log and its success message asks for the text with no preface
(all eight replies had opened "Clean audit"); skill: a line the survey tracks
along a scan is a measurement to quote.

#### Pass 13 — 2026-09-24, two repeats

| Dataset | 13a | 13b |
|---|---|---|
| plateau-redfield | pass | pass |
| sn-critical-field | pass (Must 3 met by quoting the survey's line run by run) | fail, judgement (called the logged 8 K block a sensor fault) |
| euo-psi | pass | fail, judgement ("critical slowing down" after three CriticalDivergence fits printed LAW NOT ESTABLISHED) |
| maleic-mu-kinetics | fail, borderline (the converged slope downgraded to "qualitative") | pass |

5/8 strictly, 8/8 leniently. Passes 12–13 combined: plateau 4/4, EuO 3/4,
maleic 3/4, Sn 1/4. No reply opened with the audit any more. The failures are
guidance the tools already print being overridden, so the next changes turn
two of them into mechanism: the wizard writes recipe `line-<run>` (the
recommendation plus a detected line it does not fit, amplitude started small;
on Sn 91516 it fits the line at 1.902 ± 0.012 MHz with amplitude
0.36 ± 0.07 %), and `audit` lists a law's vocabulary when every logged fit of
that law printed `LAW NOT ESTABLISHED`.

#### Pass 14 — 2026-09-24, two repeats

| Dataset | 14a | 14b |
|---|---|---|
| plateau-redfield | pass | fail (left the two-rate series for a flagged stretched one, then fitted no law) |
| sn-critical-field | pass (Tc 3.511 ± 0.038 K against logged T) | fail (called the 8 K block a sensor fault) |
| euo-psi | pass | fail, judgement ("consistent with critical slowing" from fits it flagged unresolved) |
| maleic-mu-kinetics | fail, judgement (a hand-computed "1–2σ" agreement) | pass |

Passes 12–14: plateau 5/6, EuO 4/6, maleic 4/6, Sn 2/6. A general trap found:
three runs used `reduce --tmax 2` to zoom a plot, which cut the stored
reduction every later command reads. Changes: `reduce --plot-tmax` zooms the
PNG only; `wizard`/`fit`/`fit-series` note runs reduced to a window;
`fit-series` ends with the `trend` step; the two-rate hint names
`Exponential + Constant`; `audit` treats "N combined/standard errors" as
derived; skill text on the stored window, on a missing expected signal as
evidence for the logged temperature, and that flagged fits carry no physics.

#### Pass 15 — 2026-09-24, two repeats (three runs voided)

| Dataset | 15a | 15b |
|---|---|---|
| plateau-redfield | void (API outage mid-run, on track) | pass |
| sn-critical-field | void (API outage; had fitted `line-91516` and was trending on logged T) | fail (a 120 s wizard went to the background and the agent ended its turn) |
| euo-psi | void (API outage) | fail, judgement (no paramagnetic λ(T): the ordered recipe was carried above T_c) |
| maleic-mu-kinetics | pass | pass |

An `ENOTFOUND` API outage cut three sessions at the same moment; those runs
are infrastructure, not skill, results. `--plot-tmax` was used for every zoom
and no run cut its stored reduction. Changes: skill text on long commands
(shell timeout of minutes, no `| tail`, never end a turn with a command in the
background) and on fitting the paramagnetic side of a transition with a
relaxation-only recipe to report λ(T); maleic Known traps (78251's title/notes
conflict; Arrhenius on λ_Mu is not the reaction Ea).

#### Where the night's loop ended — 2026-09-24 10:00

Passes 12–15 on the current tooling (voids excluded): plateau 6/7, maleic
6/8, EuO 4/7, Sn 2/7; the generalisation wave passed Tier A 4/4 and one of two
hold-outs. EuO and maleic failures are now mostly a single sentence of
interpretation; Sn remains the hard case — its signal is ~0.3 % against a
~20 % background, and it now has a working path (survey `other@` lines,
`line-<run>` recipe, `OrderParameter` against logged T) that about half the
runs follow. What moved the numbers most were CLI changes that put evidence
or the next step in front of the agent (ZF spontaneous-line search, survey
`other@<MHz>`, `audit`, the scan-specific `trend` hints, `line-<run>`), not
additional skill text; skill text alone rarely changed behaviour across passes.

#### Generalisation wave 2 — 2026-09-24, on the pass-15 tooling

Host: Claude Code, `sonnet`, `run_wave.py --set tier-a` and `--set hold-out`,
scored by one scoring subagent per wave with `scoring_brief.md`.

| Dataset | Verdict |
|---|---|
| ferromagnetic-nickel | pass |
| fmuf-ptfe | pass |
| spin-glass-ymnal | pass |
| high-tc-cuprate | pass (a judgement concern: a low-T σ turnover at 200 G read as physics while the chain may follow another line) |
| molecular-antiferromagnet | pass (paramagnetic ν(T) fitted with a relaxation-only series, as the pass-15 skill text asks) |
| copper-diffusion | fail (TF line shape never compared, as in pass 11; unprinted differences "within about 2 G") |

Tier A holds at 4/4 with no regression from passes 11–15; the molecular AFM
passes again; copper fails on the same Must as in pass 11. Copper's root
causes are general: `fit-series` carried one envelope through each TF scan,
so no output could show a change of shape, and `trend` then called the
applied-field precession frequency "an order parameter"; the survey put an
unlabelled longitudinal point of the 40 K field scan into each TF
temperature scan; `audit` matched the bare "2" of "within about 2 G" and the
"~2" of "a factor of ~2". The EMU upturn was handled (the turning-point note
led to a split fit) but the ARGUS one was not. In Tier A all three nickel
`OrderParameter` fits printed `FAILED` with Tc ≈ 357.7 K and β ≈ 0.38: over
320–356 K the shape exponent α trades off against y0, and Minuit exhausts its
calls; α fixed at 1 converges (Tc 357.74 K, β 0.387). YMnAl withheld a
determined T_g because ν was not.

A first pass-16 wave was voided: every session ended "Request timed out" at
the same moment (as in pass 15).

#### Pass 16 — 2026-09-24, reliability repeats on the pass-15 tooling

| Dataset | 16a | 16b |
|---|---|---|
| plateau-redfield | pass (free-m Redfield gave m = −1.55, printed as "determined") | fail, harness (turn ended with a wizard in the background) |
| sn-critical-field | pass (Tc 3.488 ± 0.013 K against logged T; the 6 K block kept as real) | fail, harness (same) |
| euo-psi | fail, judgement (no ZF paramagnetic λ: the ordered recipe ran over all 38 ZF runs and the 16 lineless warm runs were excluded, not refitted) | pass (a relaxation-only ZF series above T_c) |
| maleic-mu-kinetics | pass (k_Mu slope with a caveat; but Arrhenius on λ_Mu read as a reaction Ea — the Should-level trap) | pass (declined Arrhenius without k_Mu at more temperatures) |

In 16a two runs hit the shell's 120 s default on `wizard`/`fit-series` and
recovered by polling. In 16b polling was refused (`sleep` blocked, command
substitution denied), both agents left the wizard in the background and ended
their turn, and the headless session ended with it — the pass-15b Sn failure
again. An interactive session is woken when the command finishes, so these
are harness results; they are scored as fails to stay comparable with 15b.
One agent read "a shell timeout" as the `timeout` command, which macOS lacks.
The EuO 16a failure is the pass-15b one: the skill text on fitting the
paramagnetic side was in context and not acted on, while the output showed
only flags and the `OrderParameter` hint.

Passes 12–16 on the pass-15 tooling (voids excluded): plateau 7/9, maleic
8/10, EuO 5/9, Sn 3/9; counting the three background-job endings as void,
plateau 7/8 and Sn 3/8.

Changes after pass 16 (from these two waves and generalisation wave 2, all
general, none dataset-specific):

- `fit-series` weighs the other relaxation envelope on every converged run of
  a single-envelope recipe (Gaussian ↔ Exponential, same parameter count) and
  reports an `envelope` column; `fit-series` and `trend` note a change of shape
  along the scan (copper's ARGUS TF scan: Gaussian at 56 K, exponential from
  103 K).
- `fit-series` names a block of lineless, badly described runs at one end of
  a precession scan as the other side of a transition and prints the
  relaxation-only `recipe`/`fit-series` commands for exactly those runs (EuO's
  16 paramagnetic ZF runs; nickel's paramagnetic ZF runs; Sn above T_c).
- `trend` no longer calls a frequency held within 10 % "an order parameter";
  a law not established names its next step (`--fix alpha=1` for
  `OrderParameter`, `--fix m=2` for `Redfield`, and a non-positive free `m` is
  itself not established; otherwise hold the undetermined parameters and
  report the determined ones).
- `survey` names a line-free run of a mostly-TF temperature scan that also
  sits in a field scan with no transverse line (only copper's 20898 and 76942
  across all 16 rubric folders).
- `audit` always lists numbers after difference phrases ("within about 2 G",
  "differ by") and hedged ratios ("a factor of ~2").
- Harness: `run_eval.py` sets ten-minute shell timeouts; the skill names the
  shell tool's timeout parameter.

#### Pass 17a — 2026-09-24, first run on the post-pass-16 tooling

| Dataset | 17a |
|---|---|
| plateau-redfield | pass (Redfield `--fix m=2` over 3–38 kG: D 27.50 ± 0.29, ν 174 ± 7 MHz) |
| sn-critical-field | fail, judgement (called the logged 8 K block a sensor fault — the 13b/14b failure) |
| euo-psi | fail, number rule (run durations subtracted by hand from `info` start/stop stamps; all Musts met, paramagnetic side fitted) |
| maleic-mu-kinetics | pass (k_Mu at 278 and 298 K; Arrhenius declined for want of temperatures) |
| copper-diffusion | fail (TF scans filed as "calibration only" and never fitted, so the envelope check never ran on them) |
| molecular-antiferromagnet | pass (`OrderParameter` failed free, converged with `--fix alpha=1` as the new Next step said) |

Records, passes 12–17a (voids excluded): plateau 8/10, maleic 9/11, EuO
5/10, Sn 3/10; molecular AFM 3/3 and copper 0/3 across the generalisation
runs. No turn ended with a command in the background, though agents still
set their own 60–300 s timeouts.

What the new outputs did: the audit's difference phrases fired in every run
and were acted on; `--fix alpha=1` rescued the molecular AFM's T_c; the
survey named copper's stray LF runs. What misfired, to fix next (all CLI):

- **The lineless-end note fired falsely on Sn** (91526–91529), because a
  free Gaussian width absorbed the weak line; with σ held those runs carry
  H_c lines. It should first propose holding the envelope width from the start
  run, and skip runs sitting at the applied field's Larmor frequency.
- **The envelope note assumes a temperature axis.** On copper's LF field
  scan it called a decoupled static Kubo–Toyabe "motional narrowing"; on a
  field axis a rate extremum should point at a level crossing and
  `integral-scan`, not narrowing.
- **`trend` accepts a law against the wrong axis**: Redfield fitted against
  temperature (EuO) printed `Next: --fix m=2`. A free Redfield m of 52 was
  printed as determined — a runaway exponent should not be.
- **Calibration runs that are also a physics scan are never fitted**
  (copper's TF scans): the survey could say so, and `audit` could list survey
  scans with no series.
- **Sn sensor-fault call (third time):** the survey could cross-check a
  `TEMPERATURE:` block against the line evidence — a block that loses the
  line another block shows at the same setpoint supports the logged value.
- **`audit` matches bare numbers anywhere in the log**, so hand-computed
  durations ("27 minutes") passed; `info` could print durations, and `audit`
  could match a number with its unit.
- An Arrhenius fit across a flat low-T plateau (copper ARGUS, χ²ᵣ 92) still
  gets no plateau note.

Paused here at the maintainer's request.

#### Where evaluation stands — handoff, 2026-09-24

Host Claude Code, model `sonnet`, one run per case per wave, scored against the
rubrics with `scoring_brief.md`. Counts exclude voids (API outages).

| Case | Set | Record | Window | Typical failure now |
|---|---|---|---|---|
| plateau-redfield | trend-fit | 8/10 | passes 12–17a | harness: a wizard left in the background (16b); otherwise passes |
| maleic-mu-kinetics | trend-fit | 9/11 | passes 12–17a | a sentence of interpretation (a withdrawn or hand-derived number) |
| euo-psi | trend-fit | 5/10 | passes 12–17a | the paramagnetic side not fitted on its own (16a), number rule (17a) |
| sn-critical-field | trend-fit | 3/10 | passes 12–17a | calls the logged 8 K block a sensor fault (13b, 14b, 17a); harness (16b) |
| ferromagnetic-nickel | Tier A | 2/2 | pass 11, wave 2 | — (all `OrderParameter` fits failed in wave 2; now `--fix alpha=1`) |
| fmuf-ptfe | Tier A | 2/2 | pass 11, wave 2 | — |
| spin-glass-ymnal | Tier A | 2/2 | pass 11, wave 2 | — (withheld a determined T_g once) |
| high-tc-cuprate | Tier A | 2/2 | pass 11, wave 2 | — (a low-T σ turnover over-read once) |
| molecular-antiferromagnet | hold-out | 3/3 | pass 11, wave 2, 17a | — |
| copper-diffusion | hold-out | 0/3 | pass 11, wave 2, 17a | TF line shape never reported; in 17a the TF scans were never fitted |

Pass 17a is the only wave on the current tooling (commits after pass 16, plus
the three corrections made before shipping: the envelope note names motional
narrowing only on a temperature axis, the lineless-end note asks for a
held-width refit first, and a law fitted against the wrong axis is noted
with no `--fix` step offered). None of those corrections has been through a wave yet.

Where the failures sit. **CLI gaps** (listed under pass 17a): calibration
runs that are also a physics scan are never fitted, and no output says so; the
survey's `TEMPERATURE:` line does not cross-check a block against the line
evidence; `audit` matches bare numbers, so hand-computed durations pass;
`info` prints no run duration; no plateau note for an activated law fitted
across a flat low-T region; a runaway free Redfield `m` (52) is printed as
determined. **Agent judgement** (follow-up, possibly model-limited): the Sn
sensor-fault call against the printed `prec none` evidence; interpreting a
caveated result as grounds to withhold it; not acting on skill text that the
output does not repeat (the paramagnetic-side rule, long-command timeouts —
agents still set 60–300 s themselves). **Harness**: whole-wave API outages
(three waves voided so far); `run_eval.py` now sets ten-minute shell timeouts.

To pick up: rerun `run_wave.py --set trend-fit --set hold-out` twice on the
current tooling before changing anything, so the corrections above have a
baseline; then the CLI gaps in the order listed.

### Sonnet 5.5 — 2026-09-29, on `feat/skill-sonnet-5-5`

Host Claude Code 2.1.284, model `claude-sonnet-5-5` (the CLI must be recent
enough to know it; 2.1.258 refused the id and its `sonnet` alias was still
`claude-sonnet-5`). Runs cost $0.4–2.3 and 1–10 min, against $2–5 and
15–20 min for `claude-sonnet-5`. From the second wave on, waves ran from a
detached worktree snapshot of the branch (`PYTHONPATH=<snapshot>/src`) so code
could change while a wave ran. Scoring was delegated per wave with
`scoring_brief.md`; verdicts only are recorded here.

**The pass-17a failures, first.**

| Case | Sonnet 5 record | 5.5 baseline | After the fixes |
|---|---|---|---|
| euo-psi | 5/10 | 1/2 | 2/2 |
| sn-critical-field | 3/10 | 2/2 (no H_c law) | 2/2, OrderParameter T_c 3.4848 ± 0.0263 K |
| copper-diffusion | 0/3 | 0/2 | 0/2, then 2/2 |
| maleic-mu-kinetics | 9/11 | — | 1/2, then 2/2 |
| Tier A + plateau + molecular AFM | — | — | 6/6 |
| workflow gate (TCNQ, Al-LLZ, silicon, CdS) + spin-Peierls | — | 5/5 | — |

What moved them: a #336 regression fixed (series fits restarted every
unpinned `frequency` at the applied field's Larmor value, losing the Sn
critical-field and muonium lines); `audit` listing survey scans no fit covered
(copper's TF scans "used only for alpha" were then fitted and the envelope
change reported); the eval runner giving the agent `Edit`; skill text for
#334–#336 features, neat = 1 / blank = 0 concentrations, one command per shell
call.

**The rest of the corpus.** Rubrics written this session from the worksheets
and papers, each checked against what the CLI prints: `lifeas-psi`,
`trsb-re6zr`, `basics`, `corannulene-alc`, `benzene-high-tf`, `benzene-rf`,
`benzene-repolarisation`, `benzene-alc` (sets `corpus-2026`, plus `workflow`
and `tier-b` in `run_wave.py`).

Waves 1–6 iterated on the failures; a final wave then ran all 20 other cases
once on one snapshot (`final-a`), and the last cases again on the finished
tooling.

| Case | Record on 5.5 | Final state |
|---|---|---|
| 20 cases: Tier A (4), trend-fit (4), hold-outs (2), workflow gate (4), spin-Peierls, high-TF AFM, TRSB, LiFeAs, benzene RF, benzene repolarisation | each passed on every run of the final waves | 20/20 in `final-a`; TCNQ, copper, cuprate and nickel 4/4 again after the quieter-notes change |
| trsb-re6zr | fail, fail, then 2/2 | pass |
| afm-high-tf-mdu | 3 fails, then 3 of the last 4 | pass |
| basics | fail ×2, then 5 of the last 6 | pass (judgement calls on the bad-sensor Must) |
| benzene-high-tf | fail ×3, then 5 of the last 6 | pass |
| benzene-alc | fail ×2, then 6 of the last 8 (2/2 on the latest tooling) | pass |
| corannulene-alc | 2 of 10 | the one case that does not pass reliably: see below |

Corannulene's physics lands in every recent run (windowed fits of both
temperatures, motional averaging, no model for couplings); what fails is the
comparison between the two scans' lines — once left without a direction, once
stated backwards (fixed on the last pair by a skill line: quote both centres
and widths) — and Must 8, the reason no whole-scan fit is given: the
background, a rise with a step near 20 kG, which no polynomial follows. On the
last pair the CLI's dip search still named the cold scan's rise–plateau–rise
background (19.6–29.8 kG) as another dip, and that note displaced the
background sentence; the background sentence now prints on every poor fit. A
test that separates a dip on a steep slope (benzene solid's 21.47 kG line,
lower on one side than its own bottom) from a rise–plateau–rise step was tried
and reverted: it rejected the real line. Open: a background model for a step
(the GUI's `FermiStep` did not fit it either), or a dip test on residuals from
a smoothed background.

CLI changes that answered them (the output lines agents act on):
`survey` names the detector pair that precesses (`PAIR:`), each scan's
samples, repeat sets to co-add (`REPEATS:`), two-period scans, and prints a
large folder's findings before its run table; runs of different bin widths
never share a scan. `trend` brackets a width or rate step by where it leaves
each end's level, names a phase linear in field as a t0 offset and an
amplitude falling with frequency as the frequency response, names a held
line's shift, and offers a two-line recipe for a held high-field line.
`fourier` names its window and notes peaks within two resolution elements.
`fit-series` starts a Larmor-following line at each run's field (keyed on the
field the recipe was seeded at). `integral-scan` prints errors, at-bound marks
and a failed fit's reasons; `fit` stores its result so `audit` counts it;
`--alpha-from` a single-period run serves a period scan. Later waves added:
`integral-scan` naming the window of a dip a fit left out (checked by fitting a
line there: centre and flanks inside, depth five errors below the background),
a note when a windowed line runs off its range, the background as the cause of
a poor long-range fit, no-model and `--json` notes; `fourier` naming lines
outside its band and A_μ = ν₁ + ν₂; `trend` readings printed beside a law and
leaving out unreliable rows; a mistyped data folder named as missing.

Skill text alone again rarely changed behaviour — the high-TF two-line
paragraph was read and ignored until `fourier` and `trend` printed the pair
and the recipe. Open maintainer question: `spurious_reseeded` flags every
reseeded run even when the reseed rescued it (GUI rule since #167); one
copper run discarded a real 1 K ν upturn because of it.

### Haiku 5.5 — 2026-10-07, on main 0.26.0

Host Claude Code 2.1.293 (2.1.291 ran `claude-haiku-5-5` but warned
`unrecognized_model`), model `claude-haiku-5-5`, all 24 cases once from a
snapshot of main. $5.13 for the wave ($0.03–0.55 and 1–8 min a case), against
about $30 for a Sonnet 5.5 wave. Scored per wave with `scoring_brief.md`.

| Set | Haiku 5.5 | Passed |
|---|---|---|
| Tier A | 0/4 | — (nickel and spin glass fail on one derived number each) |
| trend-fit | 2/4 | plateau, maleic |
| hold-out | 2/2 | copper, molecular AFM |
| workflow | 0/4 | — (TCNQ, LLZ on one derived number; silicon on an unstated window) |
| tier-b | 1/2 | spin-Peierls |
| corpus-2026 | 3/8 | LiFeAs, corannulene, benzene RF |

8/24 strict; 13–16/24 with the judgement-call fails relaxed. Haiku skips
Step 7 (no `summary.md`, no `audit`) in five runs, files the next command a
CLI note prints under "Not done", rarely opens a plot, and writes small
arithmetic into prose. It follows printed CLI output far more than skill text:
several passes reached the fit `audit` listed.

Most number-rule fails passed a clean `audit`: "by about N" and "agree … to
about N" were not difference phrases, decimal percentages ("0.5 %", "1.2–1.3 %
spread", "±0.6 %") were exempt as asymmetries, and "ΔAICc about 11" matched a
`ionic-11` name. The audit now lists numbers after a change verb's "by" and an
agreement's "to", numbers named as a comparison ("4 points better", "a … %
spread"), relative `±` percentages, and Δ-quantities not printed verbatim; on
the wave's 24 summaries it newly flags every such number, and one ambiguous
"falls by 3500 G". Open, model-agnostic: `alpha --run N` and `reduce
--alpha-from N` print different alphas for one run (five cases discussed the
gap); fit commands do not print the fitted window and `fit-global` takes no
`--tmin/--tmax`; SKILL.md names `StaticGaussianKT` (the CLI's is
`StaticGKT_ZF`).

#### Wave 2 — the audit fix (`fix/audit-derived-numbers`)

12/24: Tier A 3/4, trend-fit 1/4, hold-out 2/2, workflow 4/4, tier-b 1/2,
corpus-2026 1/8. Nickel, spin glass and Sn flipped to pass on numbers the new
rules listed; every run that skipped the audit (fmuf, plateau, basics, AFM)
failed. LiFeAs, corannulene and maleic regressed on analysis, two of them by
taking the audit's "or say in the summary which runs cannot be fitted"
escape. Scorers found the audit's remaining holes ("a margin of 3.6", "a shift
of about 0.03", "±0.2 MHz", "a factor of six", "4,200") and false positives
(table error columns, "×" against a printed "x", "by 5000 G" meaning *at*,
a run number before "discrepancy").

#### Pass 3 — on `feat/haiku-cli-loop` (overnight loop)

Result commands end with the audit step; `audit` no longer takes a sentence in
place of a fit; context-derived numbers verify only as verbatim tokens of three
or more significant digits; noun differences, bare-`±` spreads, ratio words,
separators and powers of ten; `trend` prints a held line's shift with its
error; `reduce`, `survey` and `alpha` say which counts alpha was measured on
(the "mismatch" was `reduce` re-measuring on deadtime-corrected counts);
`audit` names a hyperfine coupling quoted without A_μ = ν₁ + ν₂ when `fourier`
printed it. Misfire check: on the 48 summaries of waves 1–2 the new matcher
adds only real derived numbers and drops the false positives above.

#### Waves 3–4 and passes 4–6 — overnight, 2026-10-08

Wave 3 (pass 3): the audit ran in 23 of 24 runs (it ran in 20 in wave 2, fewer
in wave 1), and lifeas, plateau and maleic recovered by fitting the runs the
audit listed. Wave 4 (pass 4): **16/24** — Tier A, trend-fit and workflow each
4/4 (fmuf and EuO pass for the first time; EuO fitted its paramagnetic side),
hold-out 2/2 (copper, molecular AFM), tier-b 0/2, corpus-2026 2/8 (LiFeAs,
benzene RF).

Pass 4 added `audit`'s A_μ = ν₁ + ν₂ relation check, the `integral-scan`
RESONANCE verdict for a clean windowed line, the whole lineless block in the
`fit-series` note, gap-law verdict/rival/normal-state prompts in `trend`, and
`NOTES SCANS:` in `survey` (scans stepped only in run notes). Pass 5: gap-law
Tc seeded from the data (a 20 K default could pin σ_bg at zero); ratio words,
drifts and resolution-element counts in the matcher; "do not reply yet" while
scans are unfitted; survey scan lines list their runs (benzene-alc fitted 16 of
76 runs twice from "(run A -> B)"). Pass 6: `integral-scan` names TF stamps a
kilogauss integral contradicts (repolarisation, TCNQ ALC); `fourier` offers a
two-line test for a tesla-field line and names MaxEnt as unavailable; `wizard`
names a component at twice that line a harmonic; a poor whole-scan resonance
fit prints window commands; `COMPARE` sets one line resolved in two scans side
by side with which is broader; `audit` prints the commands for each unfitted
scan. Rejected after replaying it on the stored series of waves 1–3: a
"recipe overreaches" note (it fired only on cases that already passed).

#### Wave 5 and passes 7–8 — 2026-10-08

Wave 5 (passes 5–6), 20 of 24 scored: Tier A 4/4 and trend-fit 4/4 again;
benzene high-TF (first pass: a plain and a correlation transform of the
co-add, A_μ stated as the sum), benzene repolarisation (first pass: the TF
stamp argued from the integral) and corannulene (COMPARE used) recovered;
spin-Peierls, AFM, TRSB, basics and benzene ALC still fail. The agents read
the right advice and skip it — the two-line test, `trend` on a series, an
announced dip.

Pass 7: `wizard` calls a component at twice a tesla-field line its harmonic;
`audit` prints the commands for each unfitted scan; `trend` names a change of
relaxation shape between differently shaped terms (replayed on 323 stored
series: spin-Peierls only); `audit` names a correlation spectrum with no plain
transform. Pass 8: `audit` holds the reply while a series has not been read
with `trend`, a tesla-field two-line test has no two-line fit, an announced
dip has no fit, or a survey-lineless run has only precession fits flagged as
not describing it (with relaxation-only commands, so the gate cannot loop).

#### Waves 6–7, passes 9–11 and a Sonnet 5.5 regression wave — 2026-10-08

Wave 6 (pass 7): 17/24; wave 7 (pass 8): **20/24** — Tier A 4/4, trend-fit
3/4, hold-out and tier-b 3/4, workflow 4/4, corpus-2026 6/8; no audit hold made
an agent loop (at most 5 audit rounds). Remaining fails: AFM (a wizard recipe
whose second line was the harmonic silenced the two-line hold), TRSB (`trend`
offered no gap law for the TF width), basics (no audit, one exercise) and Sn
(read a warm block on the setpoint axis).

Pass 9: `integral-scan` names repeated points (a return pass) and stops
announcing dips another stored scan fitted; COMPARE says errors cannot tell two
lines apart instead of "the same"; `trend`'s held-line note says nothing for or
against order. Pass 10: `trend` offers `SC_SWave`/`SC_DWave` on a TF width when
the line shifts diamagnetically while the width grows ("if the sample is a
superconductor"; replay: cuprate, LiFeAs, TRSB, one CdS series); the two-line
hold counts only two frequencies within 2 %; only `fit-series` series need
`trend`; `reduce`/`integral-scan` note file deadtimes left off. Pass 11: every
hold ends "do not reply yet", short scans are advisory, a failed two-line fit
is no test, and `fourier`'s barely-separated pair prints its two-line recipe.

**Sonnet 5.5 regression** on 6a63aad4 (pass 9): **21/24** — fails AFM (a
barely-separated pair ignored), basics ("about 20 G" by hand) and corannulene
(the rise–plateau–rise background named as a dip by the pre-existing dip
search), each a case that was already unreliable for Sonnet; no fail traced to
an overnight line. The holds cost Sonnet 1–3 turns a case (13 `trend` calls on
`fit-global` children in LLZ — fixed in pass 10).

#### Waves 8–9 and passes 12–13 — 2026-10-08

Wave 8 (pass 9): 19/24; wave 9 (pass 11): 17/24. Across waves 4–9 Haiku
scores 16–20/24: Tier A, trend-fit and workflow pass almost every wave, and
the misses rotate among basics, AFM, corannulene, TRSB, Sn and benzene ALC.
In wave 9 AFM passed every physics Must for the first time (both pairs fitted,
the harmonic rejected, MaxEnt named) and failed only on a hand-computed line
separation.

Pass 12: `fit` notes a fitted pair within 2 % is still two lines (AFM had
fitted 813.601/813.542 MHz and reported one); `integral-scan`'s fit table
gives fields and couplings their units (benzene RF quoted A_μ in gauss); window
commands stay inside the scan. Pass 13: the audit restates its hold when it
also lists numbers (basics and benzene ALC replied over unfitted scans because
the hold printed only on the numbers-clean branch); the survey's TEMPERATURE
note separates a block the apparatus could reach — a different measurement,
never quoted at its setpoints (Sn, twice) — from an impossible reading;
`fit`'s pair note prints the separation with its error.

#### Wave 10 and pass 14 — 2026-10-08

Wave 10 (pass 12): 18/24 — Tier A and trend-fit 8/8, tier-b and workflow 7/8,
corpus-2026 3/8. AFM ran the two-line fit (813.601/813.542 MHz, lower χ²ᵣ)
and still called a χ²ᵣ drop of 0.0055 "small" on ~389k points; benzene RF
obeyed `integral-scan`'s "no radical ALC or hyperfine model is available",
which is false for an RF scan. Pass 14: a green-red scan names
`RFResonanceMuP`, and `fit`'s pair note says to compare χ², not χ²ᵣ, with the
degrees of freedom.

#### Wave 11 and where the overnight loop ended — 2026-10-08 05:40

Wave 11 (pass 13): 17/24 — Tier A and trend-fit 7/8 (maleic tried relaxation
fits on Mu-precessing 2 G runs), tier-b and workflow 6/8 (an AFM agent declined
the whole 8-group HIFI folder as "multi-group", fixed in the skill's decline
row; TCNQ quoted a "0.04 MHz apart" difference of printed values), corpus-2026
4/8. Haiku 5.5 across the night: 8, 12, 14, 16, 19, 17, 20, 19, 17, 18, 17 of
24; Sonnet 5.5 on pass 9: 21/24. Tier A, trend-fit and workflow pass almost
every wave from wave 4 on; the misses rotate among basics (stops early),
corannulene (the hot 14.8 kG dip), TRSB (LF decoupling from an all-flagged
series), benzene ALC (an agent overriding RESONANCE), AFM and Sn.

Next targets, in order of how often they cost a wave:
- an all-flagged LF series should print the relaxation-only comparison with
  ZF (TRSB);
- the frequency-response note should fire on `A_n`/`frequency_n` pairs
  (basics' muonium amplitudes);
- `frequency_unresolved` should print the `--fix frequency_N=` command, and a
  mixed-sample scan the `--order concentration --x` template (maleic);
- the matcher should catch small-number differences ("0.04 MHz apart");
- the final reply should be held to the audited file.

#### Wave 12 — the held background — 2026-10-08

Wave 12 (the first held-background rule): 18/24 — Tier A and trend-fit 8/8
(maleic passed: a hand-written Mu + diamagnetic recipe and an
`--order concentration` series), tier-b and workflow 6/8, corpus-2026 4/8.
Benzene ALC and basics passed; the failures were LiFeAs (trusted the
coarse `survey_line_mhz` column over a 6σ fitted shift), TRSB (LF decoupling
again), corannulene, benzene high-TF ("hyperfine coupling" for A_μ without
"sum", so the A_mu check never keyed), AFM (vortex-lattice reading only) and
copper (an unprinted "difference is about 17 meV").

The hold fired on one run each in copper, TCNQ and spin-Peierls, and missed
the runaways it was written for: nickel's paramagnetic series sat at
A_1 ≈ 74, A_bg ≈ −41 (below the cancelling ratio), and most other runaways
(spin glass, plateau, Sn, photo-Si, copper ARGUS) had stopped mid-run with
`failed`. The rule was reworked as recorded under "Open questions".

#### Wave 13 — the reworked hold — 2026-10-08

Wave 13 (the reworked held-background rule): 19/24 — Tier A and trend-fit
7/8, tier-b and workflow 7/8, corpus-2026 5/8. LiFeAs, benzene high-TF (the
audit's A_μ "sum" check fired), copper and TCNQ passed; the failures were
spin glass (screened a featureless paramagnetic run and fitted the whole scan
with `Exponential + Constant`), AFM (a difference of two printed columns that
the audit passed), TRSB (LF decoupling still not read as static fields),
corannulene (the background and motional-narrowing notes not carried into
the reply) and benzene ALC (a model-less `integral-scan` counted as fitted).

The hold fired in seven cases (nickel in four series, copper, plateau, spin
glass, Sn, EuO, TCNQ) and every summary mentioned it, though nickel's only
in part. Held amplitudes sat at the data's early-time asymmetry: nickel's
paramagnetic A_1 at 33.1–33.3 % beside a reduced A(0) of 33.3 %, plateau's at
21.3–21.6 %, copper's ARGUS LF scan at 23.5–26.3 % (wave 12: up to 288 %). No
verdict changed for the worse. Series runs that were held still carried the
free fit's `failed` flag, because the member quality was assessed before the
hold; held runs are now reassessed. Runaways in oscillatory and two-rate
models (nickel's field and TF series, one plateau run) stayed unheld, and the
CLI does not say why.

After wave 13 the audit's one relation check (A_μ = ν₁ + ν₂) became a table
of printed requests aimed at the summary — a held `A_bg`, the deadtime used,
why a long-range resonance fit is not a result, a return pass, the missing
radical ALC model, `trend` steps, shifts, frequency response, shape changes
and extrema — each checked by lenient keywords; a missing one holds the reply
and quotes the printed line back. Replayed over the 120 replies of waves 9–13
it holds 13, among them corannulene's missing background reason (wave 12),
copper's unstated hold (wave 12) and benzene ALC (wave 13), and accepts the
passing replies' own wording ("one cubic cannot follow both dips"). A
negated statement ("I do not claim a diamagnetic shift") still satisfies it.

#### Wave 14 — the say-so check — 2026-10-08

Wave 14: 18/24 — Tier A and trend-fit 5/8, tier-b and workflow 8/8 (AFM
passed for the first time since wave 1), corpus-2026 5/8. The failures were
spin glass (the featureless screen again), maleic (declined the concentration
series because the logged temperatures differ), the cuprate ("up to about
12 K", an unprinted difference the matcher found inside other numbers), TRSB
(a Λ pinned at zero in LF read as a defect, not as static ZF fields),
basics (stopped after six commands to ask "Shall I continue?", no audit) and
corannulene. The say-so check held twice (nickel: the runs A_bg was held on;
plateau: the held runs and a trend span); both agents added the statement
within two or three audits, with no loops. Corannulene again wrote only that
the full-range fit "fails", which the check had accepted ("Cubic … fails");
it now needs the reason — a background, baseline or polynomial that cannot
follow or describe the data, or one that rises or steps — and holds
corannulene's replies in waves 12–14 while passing benzene ALC's "one
polynomial could not follow the whole range".

#### Wave 15 — composition, quiet screens and decoupling — 2026-10-08

Wave 15: 18/24 (19 counting benzene RF, which dropped one clause earlier
waves did not enforce) — Tier A and trend-fit 7/8, tier-b and workflow 7/8,
corpus-2026 4/8. Maleic passed for the first time since wave 12: it took the
survey's `composition:` block, gave pure water 0 and the neat solution 1,
dropped the untreated water, and quoted the slope with its error, though it
still hedged it against the logged temperatures. The cuprate passed again.
Spin glass screened the cold end after the wizard's new note, but the
recommended Risch–Kehr series flagged every run and only the recommended
recipe is written, so it fell back to `Exponential + Constant` (the stretched
exponential ranked second, 4.6 AICc behind). TRSB took the decoupling note
on its 10 K field scan and wrote the static reading, but fitted the 100 G
scan with the wizard's high-confidence Kubo–Toyabe recipe (the constant 42
AICc behind), got flagged widths, and called the scan unfittable, so the
note never fired there. Corannulene gave the background reason (Must 8) but
fitted only the hot 7 kG line. Basics ran the audit this time but converted
a frequency to "about 21 G" by hand, which the matcher let through. AFM
reported both lines of its two-line fit yet headlined "a single line". All
three say-so holds were satisfied at the next audit and each statement was
right for the data.

### Haiku 5.5, round 2 — 2026-10-08/09, on `feat/haiku-cli-loop-2`

A second overnight loop on the same 24 cases, from main after #361 (wave 15:
18/24). Same rules: printed CLI output before skill prose, every note, hold and
matcher change replayed over the stored outputs of waves 1–15 (and the Sonnet
5.5 regression wave) before it ships, waves run from a snapshot worktree of the
pushed commit, scored by three Sonnet scorers.

#### Pass 1

- The decoupling note reads the fitted curves, not a rate: each side's loss of
  asymmetry by 8 µs, against zero-field runs within the field series' own
  temperature span. Wave 15's TRSB fitted the 100 G scan with the wizard's
  Kubo–Toyabe, whose width the field leaves unconstrained rather than small,
  so the rate-based note never fired there. Replay: it now fires on every TRSB
  wave with a field scan, and on copper only where a zero-field run shares the
  field scan's 40 K (where the nuclear fields are indeed static).
- For a relaxation recommendation the wizard also writes `wizard-<run>-alt`,
  the best other relaxation model within 10 AICc with at most one more
  parameter, and `fit-series` names it when the recommended recipe's series
  flags half its runs `failed` or `amplitude_exceeds_data` (spin glass: the
  Risch–Kehr series flagged 10 of 10; the stretched exponential ranked 4.6
  AICc behind). Without the relaxation-only and parameter limits the replay
  fired on most oscillatory series of passing cases (nickel, cuprate).
- `audit` reads a conversion phrase ("would be", "corresponds to", "equivalent
  to") as making the next number derived: replay flags only basics' "about
  21 G" (wave 15) and "about 20 G" (Sonnet 5.5).
- A dip counts as held only by a fit of the same scan whose line, centre ±
  width, lies inside the dip's window. Corannulene's poor whole-scan fit
  (χ²ᵣ 40, a 3.6 kG-wide line) and the cold scan's 15.3 kG line had silenced
  the hot scan's 14.8 kG dip in both `integral-scan` and `audit`; replay: the
  hold now fires in exactly the waves that left that line unfitted (8, 10, 11,
  15) and nowhere else.
- `fit`'s pair note: a pinned rate on one line does not cancel the χ² gain;
  `fourier`: one peak does not exclude a doublet below the resolution (AFM,
  wave 15).

#### Pass 2

The audit's false positives on printed values, mined from the audit outputs in
the transcripts of waves 9–15: a hedged integer that rounds away trailing zeros
("about 20740 G" for 20740.6), a field in kG for one printed in G, an error in
parentheses ("2.91 (±0.12)"), "1σ errors", "> 3σ" and "2× LorentzianLCR". Of
the lines those audits flagged, 167 flags go and none is added; the final
replies of waves 13–15 are unchanged. A first version also widened hedged
decimals ("about 11.6" for a sum of two printed amplitudes); the replay caught
it and the rounding is limited to integers.

#### Pass 3

Wave 16's number slips, all passed by the audit: nickel's "3×" (the log held
"13x", and a multiple was matched as a substring), EuO's "differs … by up to
about 0.06 MHz" (two stacked hedges hid the "by"), copper's "the shift is
about 2.7 %". A multiple now verifies only as a whole printed token, hedges
stack, a difference noun followed by a hedged "is"/"was" makes the number
derived (an unhedged "the printed shift is −0.050" quotes a printed value and
stays quiet), and "62 %" matches a printed "62%". Replay over waves 9–16: the
four wave-16 slips, copper's wave-12 "difference is about 17 meV", the
cuprate's wave-14 "up to about 12 K", and two more setpoint-offset
differences; nothing else.

#### Wave 16 and pass 4

Wave 16 (pass 1): **19/24** strict, 22/24 counting three one-number slips —
Tier A and trend-fit 6/8, tier-b and workflow 7/8, corpus-2026 6/8. Spin
glass passed (it screened a run whose recommendation was the stretched
exponential), AFM passed for the second time (the two-line fit ran after the
new `fourier` line and the reply names two lines), corannulene passed for the
first time since wave 7 (the dip announcement for the hot 14.8 kG line now
survives the poor whole-scan fit, and the agent fitted it). The decoupling
note fired on TRSB and the agent wrote the static reading. Fails: nickel
("3×"), EuO ("by up to about 0.06 MHz"), copper ("the shift is about 2.7 %"),
all fixed in pass 3; TRSB (the zero-field scan fitted with the wizard's
high-confidence Bessel, a 0.1 MHz "precession" with no spectral line, so no
Kubo–Toyabe width to trend); basics (a temperature reading across calibration
runs, a Must the scorer called strict). Six of the 24 replies were retyped
after the last audit; none of the retyped text carried a new unprinted number.

Pass 4: a recommendation in the `Oscillatory` category the spectrum shows no
line for gets a NOTE (a precession model fitting a relaxation's shape) and the
best relaxation model as `wizard-<run>-alt`, however far behind. Replay over
every stored screening (waves 1–16): it fires on 16 of TRSB's 47 zero-field
screenings and twice on copper's (both zero-field Kubo–Toyabe physics), and
nowhere else; a muonium relaxation template is not counted as precession. The
decoupling note also asks for a static Kubo–Toyabe refit when the zero-field
series has no static-width term (TRSB only, 7 stored series).

#### Pass 5

The audit matches a written number to any printed value within one unit of
its last digit, not half: Haiku truncates as often as it rounds ("−0.0353" for
−0.03537). Replay over the audit outputs of waves 12–15: 171 more intermediate
flags go; the final replies change only where a printed value was truncated
(event counts, a field) and on one textbook constant quoted as background.
When the audit lists numbers and holds nothing else, it now ends "Then run
audit again: reply only once it prints 'No unprinted numbers found'" — about
one run a wave replied after an audit that still listed numbers.

#### Wave 17 and pass 6

Wave 17 (passes 2–3) — Tier A and trend-fit 7/8 (fmuf screened with
`zf-static-magnetism`, which leaves out the F-μ-F family; it had passed every
wave from 4 to 16, and recognising the sample by name would put a corpus
sample name in the code), tier-b and workflow 8/8 (copper's slip gone, AFM a
second straight pass). Rounding false positives fell (about 9 in the second
group against many in wave 16), and the new catches fired ("1.5σ", "about
2 errors", "1.0 AICc behind") and were fixed by the agents.

Pass 6, for the two habits that remain: stopping early and retyping. Every
result command now ends, before the audit step, with "Still unfitted:" and the
surveyed measurements no fit holds (the audit's own list). Replayed on the
final state of waves 16–17 it names something only in the three runs that
stopped short (maleic and TRSB in wave 16, basics in wave 17, which stopped
after six commands without an audit). A clean audit prints the passed draft
between marker lines for the reply to copy whole: six of eight replies in one
wave-17 group were retyped, one with an unaudited garbled clause.

#### Wave 17 corpus group and pass 7

The wave-17 corpus group scored 3/8 (wave 17 total **18/24**): TRSB passed
(SC laws fitted, decoupling note quoted), but LiFeAs (the shift reported
without its direction), basics (stopped after six commands, no audit),
corannulene (no statement that the cold scan lacks the 7 kG line), benzene RF
and benzene ALC (a converged broad solid-state line dismissed) failed.

Pass 7:
- The RF note in `integral-scan` was gated on a logged field step between the
  periods — the mark of a differential ALC scan — so it printed for benzene
  ALC's o-p scans and never for the RF scan in any of 17 waves. It now prints
  for a green−red scan with no step; a scan with a step gets the "no radical
  ALC model" note, which now covers `LorentzianLCRPair` fits.
- `trend`'s shift line says which way the line sits at low temperature and
  asks for the direction; the audit's say-so check now needs a direction word
  or a signed value beside "shift". Replay over the 107 stored replies whose
  logs printed the line: 21 held, nine of them LiFeAs (wave 17's fail among
  them) and six TRSB, where the diamagnetic shift is the physics.
- `audit` pools the fits of every work directory that surveyed the same data
  folder: benzene ALC's agent surveyed one folder into two work directories
  and fitted in the second, and four audits listed 17 "never fitted" scans.

#### Wave 18, passes 8–9

Wave 18 (passes 4–5): Tier A and trend-fit **8/8** (fmuf back on the
`fluoride-fmuf` scope), tier-b and workflow **8/8**; corpus group below. The
"Then run audit again" line fired in eight runs and was obeyed every time;
every run of the first two groups ended on a clean audit. The lineless
precession note did not fire (no Oscillatory recommendation without a line in
these groups).

Pass 8: `integral-scan` names a resolved line that another analysed scan
covers with no fitted line within five widths of it, with the command to fit
that scan on the same window (corannulene's cold scan lacks the hot 7 kG line,
a Must no reply in wave 17 met). Replay: corannulene in 11 waves, three benzene
ALC scan pairs.

Pass 9: the `RESONANCE` verdict takes windowed lines up to χ²ᵣ 8, not 4.
Wave 17's benzene ALC agent dismissed a converged solid-state line at χ²ᵣ 5.0
that had no verdict; every stored windowed line between 4 and 8 with both
flanks and a five-error depth is a real resonance (benzene solid, o-p and
CHMu, corannulene 7 and 14.9 kG, TCNQ), and none is a step.

#### Wave 18 corpus group and pass 10

The wave-18 corpus group scored 3/8 again (wave 18 total **19/24**): LiFeAs,
benzene high-TF and repolarisation passed. TRSB took the new lineless-
precession note (it fitted the zero-field series with the relaxation
alternative and said the Bessel was no evidence of a field) but hedged its
headline away from a TRSB candidate; corannulene fitted a 175-run block mixing
the low-field decoupling runs with the ALC scan; basics copied a step note onto
a V-shaped steering curve and a saturating range curve; benzene RF gave no
resonance fields; benzene ALC again dismissed the solid's broad line (wave 18
ran before the χ²ᵣ 8 verdict).

Pass 10: along a supplied axis `trend` names an amplitude that is lowest or
highest inside the scan, beyond both ends by five combined errors, as a
minimum or maximum (not a step), and the audit holds a reply that does not
say so; every step note says which way the parameter goes and where it levels
off. Replay over all stored series: without the supplied-axis limit the
extremum reading fired on single odd runs in 13 cases; with it, on basics'
steering scan in all 10 waves that fitted it, and nowhere else.

#### Pass 11

- `survey` cuts a field sweep whose widening steps (a decoupling curve, 1,
  1.8, 3 … G) give way to even ones (a level-crossing scan) into two scans,
  at the first field of the even run — corannulene's 50 K block of 175 runs
  from 0 to 30 kG mixed both, and the agent fitted it whole. Replayed on the
  field scans of every stored survey, it cuts only that block, at 5000 G.
- After an `RFResonanceMuP` fit, `integral-scan` prints the two resonance
  fields the couplings imply (benzene RF gave none in wave 18), or says the
  couplings are not a result when the fields fall outside the scan (wave 17's
  unseeded fit returned A_μ ≈ 24700 MHz and no warning).

#### Pass 12

From wave 19's first group (6/8: fmuf screened without the F-μ-F family
again; the cuprate quoted "consistent … to about two combined standard
errors"): a number word before "errors", "sigma" or "standard deviations" is a
derived significance unless a command printed that phrase ("more than three
errors" is the CLI's own); a sentence that negates a law ("an Arrhenius law
was not established") no longer trips the law-vocabulary check. Rejected after
replay: "below"/"above" as comparison words — they flag every "flat at 0.17
µs⁻¹ above 6.4 K" in the stored replies.

#### Wave 19 and pass 13

Wave 19 (passes 6–9): **19/24** — Tier A and trend-fit 6/8 (fmuf without the
F-μ-F family again; the cuprate's "two combined standard errors", fixed in
pass 12), tier-b and workflow 8/8, corpus-2026 5/8. Benzene RF passed with the
RF note, benzene ALC with the solid's lines reported under the χ²ᵣ 8
`RESONANCE` verdict, TRSB again. The echoed draft took: in two groups 13 of 15
audited replies were byte-identical to the audited file (one added a question
block after it, one retyped). "Still unfitted" fired in most runs and was
acted on in all but basics, which stopped after the quartz exercise and wrote
no summary.md "since you asked for the summary here". LiFeAs failed on a
frequency `trend` called held (within 10 %) without a shift line: its first
and last runs are within five errors though the cold and warm sides differ.
A two-block shift test was tried and rejected after replay: it adds shift
lines to the passing TF series of five cases and drops 36 that the end-point
test prints for copper and the cuprate.

Pass 13: the closing audit step says it applies however the summary was
asked for, and even to stop early; the audit's list of short unfitted scans
leaves out runs that measured alpha (TCNQ's agent fitted its 100 G
calibration runs to clear it).

#### Wave 20 and pass 14

Wave 20 (passes 10–12): **23/24** — Tier A and trend-fit 8/8, tier-b and
workflow 8/8, corpus-2026 7/8, the best wave of either round. Basics was the
only fail: it used the new minimum note on the steering scan and the range
curve's levelling point, but fitted the low-field quartz muonium with one damped
cosine (19 runs flagged), called it unusable, and never reached a clean audit
(a scan crossing two samples stayed listed although the draft said why).
Corannulene passed with the survey's split of its 50 K block, benzene RF with
the printed resonance fields, LiFeAs, TRSB, benzene ALC again. In 23 of 24
runs the reply was the audited file or differed only in wording.

Pass 14: `survey` names runs whose line sits near muonium's weak-field triplet
frequency for their field on a `MUONIUM:` line (replay: basics' 16 quartz runs
and maleic's four muonium-in-water runs, nowhere else); a scan crossing samples
is listed but no longer holds the reply or the "Still unfitted" line, as its
own text ("or say in the summary why not") always promised (maleic gave up on
its audit over one in wave 19); a single Lorentzian on a stepped green−red scan
is pointed at `LorentzianLCRPair` with `dB` held at the step.

#### Pass 15

A whole-number percentage right after an asymmetry's name ("A_1 is about
33 %", "the amplitude stays near 21 %") is that asymmetry in its unit, not a
ratio, and verifies against the printed value (the scorers counted about 20–30
such false flags a group). Replay over the audit outputs of waves 12–15: 80
false flags go, three are reworded, the final replies of waves 16–20 are
unchanged; a first version matched every word starting with "a" ("about 4 %
below") and the replay caught it.

Wave 21's first group (pass 13): 7/8, fmuf the miss for the third time in
five waves. A generic cue — a zero-field screen whose scope left out the
F-μ-F family and found no line — was rejected after replay: it would print in
13 of the 24 cases. Recognising the sample by its material name would put a
corpus sample name in the CLI; recorded as an open question.

#### Wave 21 and pass 16

Wave 21 (pass 13): **20/24** — Tier A and trend-fit 7/8 (fmuf), tier-b and
workflow 7/8 (TCNQ: "move the centres by up to about 100 G", a difference the
audit passed because 100 is printed elsewhere and counts three significant
digits), corpus-2026 6/8 (basics; corannulene reported a 7355 G line the CLI
said ran off its window). The closing step got every agent to write and audit
summary.md, basics included; basics then replied over a "do not reply yet".

Pass 16:
- `audit` pools fits by the data folder each work directory's manifest is
  bound to, not by survey: basics split its exercises into work directories
  without surveys of their own, and the audit listed 79 fitted runs as
  unfitted (19 after; replay changes nothing else but two older runs with the
  same layout).
- A field scan most of whose runs precess gets `wizard`/`fit-series` commands
  from the audit, not `integral-scan` (basics' silver scan, LiFeAs's TF field
  scans and maleic's muonium field scan in the replay).
- Rejected after replay: counting an integer's trailing zeros out of its
  significant digits ("about 100 G") — it flagged nine labels such as "100 G
  TF screen" and "by 170–190 K" in passing replies, and missed the TCNQ
  wording anyway.

#### Pass 17 — one skill row

fmuf failed in waves 19, 21 and 22, each time screened with
`zf-static-magnetism`; in wave 22 the agent knew the sample was rich in
fluorine and argued that F–μ–F "is not the claimed situation for a polymer".
That is a physics misreading no printed line can correct without naming the
sample, so the skill's scope row for fluorine now says fluorine counts by
formula or by material — fluorides, fluorinated polymers and organics — and
that the muon binds between two F nuclei in all of them, with
`--geometry ZF --scope fluoride-fmuf` and `r_muF` to report. No other corpus
case holds fluorine.

### Things this loop found that are not skill problems

Recorded here rather than fixed, because Phase 4 changes skill text only:

- **The wizard's fluorine hint fired on every ISIS run** — fixed at the
  Phase 4 gate. `wizard_scope._FLUORINE_TOKEN` matched the `F` of ISIS's
  `F=<gauss>` title convention, so `nickel_T=100_F=0` "suggested fluorine"
  and promoted the F-μ-F family in the GUI wizard as well as the CLI. The
  token now excludes an `F` followed by `=`; the skill's workaround paragraph
  was removed.
- **`AsymmetryScaleWarning` is not collapsed for warnings raised in the
  wizard's worker processes.** `cli._collapse_repeated_warnings` replaces
  `warnings.showwarning` in the parent only, and macOS spawns workers, so a
  `wizard` run can still print several identical multi-line warning blocks on
  stderr. The skill tells the agent they are benign noise.
- **No command emits an F-μ-F dipolar coupling *frequency*.** The family
  parameterises the coupling as `r_muF` in ångström. The `fmuf-ptfe` rubric's
  "reports a dipolar coupling frequency" Must therefore cannot be met
  literally by any summary; it was ticked on the summary naming `r_muF` as the
  dipolar coupling. Either the CLI should emit ω_D as a derived quantity or
  the rubric should say "coupling".
- **`survey` prints one temperature column**, so the `fmuf-ptfe` Should about
  the sample-temperature column and its uncertainty is unreachable. (Since
  2026-09-23 it also prints the logged sample temperature, `T log/K`, where
  the file records one; the uncertainty is still not shown.)
- **`alpha` takes no `--workdir`**, unlike every other command. Harmless, but
  the skill has to say so.

## Decisions recorded

- 2026-09-15: Agent evaluation targets are host-specific: Claude Code uses
  Sonnet and Codex uses `gpt-5.6-luna`. Every recorded pass names both the host
  and model; the existing Sonnet evaluation log remains historical evidence.
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
- 2026-09-14 (Phase 2 gate): a recipe built from a wizard assessment takes
  start values and fixed flags from the wizard's fit but its **bounds from
  the model's static defaults** (`seed_parameters(model, SeedContext())`).
  Every bound the wizard sets is a per-run search window (the peak window,
  0.5–2× a seeded Delta/A_hf/r_muF, Nyquist and duration caps) and carrying
  them clamped the nickel series at the 300 K window edge. This diverges
  from the GUI's Apply button, which copies the wizard's bounds into the
  single-fit table; the recipe is meant to travel across a scan, the table
  is not. Series fits also re-seed run-bound values (applied field, spectral
  peak) from each run's own record unless a person pinned them, using the
  existing `Seed.run_bound` marker; the recipe records pins in `pinned`.
- 2026-09-14 (Phase 2 gate): `fit-series --global` **pins** a parameter at
  its recipe value in every run; `fit_asymmetry_series` is block-separable
  and cannot fit a shared parameter. A scripted simultaneous fit is not in
  this PR.
- 2026-09-14 (Phase 2 gate): `fit-series --start RUN` chains outward from a
  chosen run (descending and ascending branches, each its own chain), so the
  recommended flow is `wizard --run N` on the run with the clearest structure
  then `fit-series --start N`. Chaining from the cold end with a recipe fitted
  at 300 K lost the 100–250 K nickel runs; outward chaining recovers them.
- 2026-09-14: `screen_templates` (explicit template lists) was left out of
  the façade: without a `PeakAnalysis` seed context multiplet templates seed
  every line at one frequency, which is not a fit of that template.
- 2026-09-15: the work directory is a **visible `asymmetry-work/` in the
  current (project) directory**, not a hidden `.asymmetry/` inside the data
  folder: raw data routinely sits on a share or in a read-only archive, which
  is neither writable nor a place an analyst wants caches and plots to appear,
  and a hidden directory is one the person driving the analysis cannot find
  when they need a plot or want to delete a stale session. The consequence —
  the work directory no longer being uniquely determined by the data folder —
  is paid for by making the manifest's `folder` binding: one work directory
  serves one data folder, checked on every command, so two folders with
  overlapping run numbers can never collide in one cache.

## Open questions for the maintainer

None outstanding (both resolved 2026-09-14, see "Decisions recorded").

The amplitude/constant degeneracy raised on PR #361 was resolved 2026-10-08:
Ben chose to hold `A_bg` at zero for slow relaxations. `fit` and `fit-series`
refit with `A_bg = 0` when a fit is `amplitude_exceeds_data`, an amplitude
and a free `A_bg` have opposite signs and a rate or static width decays by
less than half over the fitted window, and keep the held fit when it converges
inside the data at a χ² cost under 25 χ²ᵣ; a NOTE says so. The first version
also required a converged free fit and a cancelling ratio of 2; wave 12 showed
it held nothing in nickel's paramagnetic series (A_1 ≈ 74, A_bg ≈ −41, many
runs `failed` mid-runaway). Replayed over the stored wave 9–12 series, the
χ² cost separates cleanly: genuine flat valleys (nickel, Sn, spin glass,
plateau, photo-Si, slow copper) cost under ~15 and come back at physical
amplitudes; wrong models (maleic's muonium fitted as relaxation, the
molecular antiferromagnet) cost thousands and keep their free fits.
