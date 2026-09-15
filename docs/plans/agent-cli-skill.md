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
`--workdir asymmetry-work/<name>` of its own.

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
  the sample-temperature column and its uncertainty is unreachable.
- **`alpha` takes no `--workdir`**, unlike every other command. Harmless, but
  the skill has to say so.

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
