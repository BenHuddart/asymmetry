# `asymmetry` command reference

Generated from the CLI's own parser by
`tools/agent_eval/render_command_reference.py`; do not edit by hand.

Every command takes `--json` (machine-readable payload on stdout) and writes
its state into the work directory `<folder>/.asymmetry`, so the next command
picks it up. Exit codes: 0 success, 1 user error (one line on stderr),
2 internal error (a traceback).

## `asymmetry`

```
usage: asymmetry [-h] [--version] [--verbose]
                 {survey,alpha,reduce,wizard,fit,fit-series,trend,skill,info} ...

Asymmetry — μSR data analysis

positional arguments:
  {survey,alpha,reduce,wizard,fit,fit-series,trend,skill,info}
    survey              List the runs in a folder with their metadata, scans and
                        calibration runs
    alpha               Estimate the forward/backward balance alpha from one run
    reduce              Reduce runs to forward/backward asymmetry and cache them in
                        the work directory
    wizard              Screen a reduced run against the fit wizard's candidate models
    fit                 Fit one reduced run with a recipe
    fit-series          Fit a recipe across a scan of reduced runs, chained along the
                        scan order
    trend               Print (or export) the parameter trend of a stored series
    skill               Install, check or remove the asymmetry-analysis agent skill
    info                Show metadata for a data file

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --verbose             Show every warning as Python's own traceback-style block,
                        instead of collapsing repeats of the same warning to one line
                        on stderr
```

## `asymmetry survey`

```
usage: asymmetry survey [-h] [--json] [--workdir WORKDIR] folder

positional arguments:
  folder             Directory holding the run files

options:
  -h, --help         show this help message and exit
  --json             Emit the machine-readable payload
  --workdir WORKDIR  Work directory to write survey.json into (default:
                     <folder>/.asymmetry)
```

## `asymmetry alpha`

```
usage: asymmetry alpha [-h] --run RUN [--json] folder

positional arguments:
  folder      Directory holding the run files

options:
  -h, --help  show this help message and exit
  --run RUN   Run number to estimate alpha on
  --json      Emit the machine-readable payload
```

## `asymmetry reduce`

```
usage: asymmetry reduce [-h] --runs RUNS [--alpha ALPHA] [--alpha-from ALPHA_FROM]
                        [--deadtime {off,from_file}] [--rebin REBIN] [--tmin TMIN]
                        [--tmax TMAX] [--plot] [--json] [--workdir WORKDIR]
                        folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --runs RUNS           Run numbers, e.g. '17294-17296,17300'
  --alpha ALPHA         Fixed alpha to reduce with
  --alpha-from ALPHA_FROM
                        Estimate alpha on this run (a weak-TF calibration run) and use
                        it
  --deadtime {off,from_file}
                        Deadtime correction (default: off, matching the GUI's fresh-
                        run default)
  --rebin REBIN         Merge this many bins (default: 1)
  --tmin TMIN           Discard points below this time/µs
  --tmax TMAX           Discard points above this time/µs
  --plot                Write plots/reduced-<run>.png for each run
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to write into (default: <folder>/.asymmetry)
```

## `asymmetry wizard`

```
usage: asymmetry wizard [-h] --run RUN [--geometry {ZF,TF,LF}] [--scope PRESET]
                        [--plot] [--json] [--workdir WORKDIR]
                        folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --run RUN             Run number to screen
  --geometry {ZF,TF,LF}
                        Applied-field geometry, overriding the survey's and the file's
                        (ISIS stamps TF on zero-field runs and some files record
                        nothing)
  --scope PRESET        Candidate-family scope preset (default: auto, from the run's
                        geometry)
  --plot                Write plots/wizard-<run>.png of data + recommendation
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to read and write (default:
                        <folder>/.asymmetry)
```

## `asymmetry fit`

```
usage: asymmetry fit [-h] --run RUN --recipe RECIPE [--fix NAME=VALUE] [--free NAME]
                     [--tmin TMIN] [--tmax TMAX] [--plot] [--json] [--workdir WORKDIR]
                     folder

positional arguments:
  folder             Directory holding the run files

options:
  -h, --help         show this help message and exit
  --run RUN          Run number to fit
  --recipe RECIPE    Recipe file, or the name of one in the work directory's recipes/
  --fix NAME=VALUE   Hold a parameter at a value (repeatable)
  --free NAME        Release a parameter the recipe holds (repeatable)
  --tmin TMIN        Fit only above this time/µs
  --tmax TMAX        Fit only below this time/µs
  --plot             Write plots/fit-<run>.png
  --json             Emit the machine-readable payload
  --workdir WORKDIR  Work directory to read (default: <folder>/.asymmetry)
```

## `asymmetry fit-series`

```
usage: asymmetry fit-series [-h] --runs RUNS --recipe RECIPE [--fix NAME=VALUE]
                            --order {temperature,field,run} [--global P,Q]
                            [--start RUN] [--name NAME] [--plot] [--json]
                            [--workdir WORKDIR]
                            folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --runs RUNS           Run numbers, e.g. '17294-17322'
  --recipe RECIPE       Recipe file, or the name of one in the work directory's
                        recipes/
  --fix NAME=VALUE      Hold a parameter at a value (repeatable)
  --order {temperature,field,run}
                        Scan quantity the series is ordered and trended along
  --global P,Q          Parameters held identical across every run. The batch is
                        block-separable, so these are pinned at their recipe value and
                        not fitted; everything else is free per run
  --start RUN           Chain outward from this run in both directions, instead of
                        from the first run in scan order. Screen the run with the
                        clearest structure ('asymmetry wizard --run N'), then start
                        the series there ('--start N'): every fit then warm-starts
                        from a neighbour nearer the run the recipe describes
  --name NAME           Name to store the series under (default: series-<recipe stem>)
  --plot                Write plots/<name>/<run>.png per run and
                        plots/<name>-trend-<param>.png per free parameter
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to read and write (default:
                        <folder>/.asymmetry)
```

## `asymmetry trend`

```
usage: asymmetry trend [-h] --series SERIES [--csv CSV] [--plot] [--json]
                       [--workdir WORKDIR]
                       folder

positional arguments:
  folder             Directory holding the run files

options:
  -h, --help         show this help message and exit
  --series SERIES    Name the series was stored under
  --csv CSV          Also write the table to this CSV file
  --plot             Write plots/<series>-trend-<param>.png for every free parameter
  --json             Emit the machine-readable payload
  --workdir WORKDIR  Work directory to read (default: <folder>/.asymmetry)
```

## `asymmetry skill`

```
usage: asymmetry skill [-h] {install,check,uninstall} ...

positional arguments:
  {install,check,uninstall}
    install             Install the skill for an agent
    check               Report whether the CLI, loaders and skill are ready for agent
                        use
    uninstall           Remove an installed skill

options:
  -h, --help            show this help message and exit
```

## `asymmetry skill install`

```
usage: asymmetry skill install [-h] --agent {claude,codex} [--project] [--into INTO]
                               [--force] [--json]

options:
  -h, --help            show this help message and exit
  --agent {claude,codex}
  --project             Install under ./.claude or ./.agents instead of the home
                        directory
  --into INTO           Install under this directory instead of the agent's usual
                        location
  --force               Overwrite a target directory even if it was not written by a
                        previous install
  --json                Emit the machine-readable payload
```

## `asymmetry skill check`

```
usage: asymmetry skill check [-h] [--agent {claude,codex}] [--json]

options:
  -h, --help            show this help message and exit
  --agent {claude,codex}
                        Check only this agent's skill install (default: every agent)
  --json                Emit the machine-readable payload
```

## `asymmetry skill uninstall`

```
usage: asymmetry skill uninstall [-h] --agent {claude,codex} [--project] [--into INTO]
                                 [--json]

options:
  -h, --help            show this help message and exit
  --agent {claude,codex}
  --project             Remove ./.claude or ./.agents instead of the home directory's
                        copy
  --into INTO           The directory it was installed under, if not the agent's usual
                        location
  --json                Emit the machine-readable payload
```

## `asymmetry info`

```
usage: asymmetry info [-h] file

positional arguments:
  file        Path to a μSR data file

options:
  -h, --help  show this help message and exit
```
