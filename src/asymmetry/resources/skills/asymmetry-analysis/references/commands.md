# `asymmetry` command reference

Generated from the CLI's own parser by
`tools/agent_eval/render_command_reference.py`; do not edit by hand.

Every command takes `--json` (machine-readable payload on stdout).

`survey`, `reduce`, `wizard` and `fit-series` persist state in the work
directory `./asymmetry-work` — in the directory you run the command from, not
in the data folder — so the next command picks it up. `fit` and `trend` read
that state and add only what `--plot` (and `trend --csv`) asks for. `alpha`
and `info` are stateless — they load, print and write nothing — and `skill`
writes into the agent's own skill directory instead.

One work directory holds one data folder's session. For a second folder in the
same project, pass `--workdir asymmetry-work-<short-name>` and keep using it
for that folder's commands.

Exit codes: 0 success, 1 user error (one line on stderr), 2 internal error
(a traceback).

## `asymmetry`

```
usage: asymmetry [-h] [--version] [--verbose]
                 {survey,alpha,reduce,integral-scan,wizard,recipe,fit,fit-global,fit-series,trend,fourier,audit,skill,info}
                 ...

Asymmetry — μSR data analysis

positional arguments:
  {survey,alpha,reduce,integral-scan,wizard,recipe,fit,fit-global,fit-series,trend,fourier,audit,skill,info}
    survey              List the runs in a folder with their metadata, scans and
                        calibration runs
    alpha               Estimate the forward/backward balance alpha from one run
    reduce              Reduce runs to forward/backward asymmetry and cache them in
                        the work directory
    integral-scan       Build an integral-asymmetry scan and optionally fit a field-
                        scan model
    wizard              Screen a reduced run against the fit wizard's candidate models
    recipe              Write a fit recipe for a model expression, bypassing the
                        wizard
    fit                 Fit one reduced run with a recipe
    fit-global          Fit multiple reduced runs simultaneously with shared
                        parameters
    fit-series          Fit a recipe across a scan of reduced runs, chained along the
                        scan order
    trend               Print, export or fit the parameter trend of a stored series
    fourier             Transform a reduced run and report resolved frequency peaks
    audit               List the numbers in a draft summary that no command's output
                        printed
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
  --workdir WORKDIR  Work directory to write survey.json into (default: ./asymmetry-
                     work)
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
                        [--tmax TMAX] [--plot-tmax PLOT_TMAX] [--period RED|GREEN|N]
                        [--plot] [--json] [--workdir WORKDIR]
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
  --tmin TMIN           Discard points below this time/µs from the stored reduction
                        every later fit uses
  --tmax TMAX           Discard points above this time/µs from the stored reduction
                        every later fit uses; to zoom the plot only, use --plot-tmax
  --plot-tmax PLOT_TMAX
                        Draw the reduced PNG only up to this time/µs; the stored
                        reduction keeps it all
  --period RED|GREEN|N  Select one period from a multi-period file. The common two-
                        period labels are red (period 1) and green (period 2)
  --plot                Write plots/reduced-<run>.png for each run
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to write into (default: ./asymmetry-work)
```

## `asymmetry integral-scan`

```
usage: asymmetry integral-scan [-h] --runs RUNS [--name NAME] [--alpha ALPHA]
                               [--alpha-from ALPHA_FROM] [--period RED|GREEN|N]
                               [--tmin TMIN] [--tmax TMAX]
                               [--method {integral,differential}]
                               [--order {field,temperature,run}] [--model MODEL]
                               [--initial NAME=VALUE] [--fix NAME=VALUE]
                               [--baseline MODEL] [--baseline-regions LO:HI,...]
                               [--plot] [--json] [--workdir WORKDIR]
                               folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --runs RUNS           Run numbers in the scan
  --name NAME           Stored scan name
  --alpha ALPHA         Fixed detector balance
  --alpha-from ALPHA_FROM
                        Estimate alpha on this run
  --period RED|GREEN|N  Select one acquisition period
  --tmin TMIN           Integration-window start / µs
  --tmax TMAX           Integration-window end / µs
  --method {integral,differential}
  --order {field,temperature,run}
  --model MODEL         Optional field-scan expression, e.g. 'LorentzianLCR + Cubic'
  --initial NAME=VALUE  Fit start (repeatable)
  --fix NAME=VALUE      Fixed fit value (repeatable)
  --baseline MODEL      Fit and subtract this baseline model first
  --baseline-regions LO:HI,...
                        Non-resonant x ranges used by --baseline
  --plot                Write plots/<name>.png
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to write into (default: ./asymmetry-work)
```

## `asymmetry wizard`

```
usage: asymmetry wizard [-h] --run RUN [--geometry {ZF,TF,LF}] [--scope PRESET]
                        [--include C,D] [--exclude C,D] [--tmin TMIN] [--tmax TMAX]
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
  --include C,D         Time-domain components to add to the scope's families, e.g.
                        'Oscillatory' for a line in an LF run
  --exclude C,D         Components to drop from the scope, e.g.
                        'VortexLattice,VortexLatticePowder'
  --tmin TMIN           Screen only above this time / µs
  --tmax TMAX           Screen only below this time / µs (the recipe keeps the window)
  --plot                Write plots/wizard-<run>.png of data + recommendation
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to read and write (default: ./asymmetry-work)
```

## `asymmetry recipe`

```
usage: asymmetry recipe [-h] --expression EXPRESSION --name NAME [--run RUN]
                        [--initial NAME=VALUE] [--fix NAME=VALUE] [--tmin TMIN]
                        [--tmax TMAX] [--json] [--workdir WORKDIR]
                        folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --expression EXPRESSION
                        Time-domain model, e.g. 'Oscillatory * Exponential + Constant'
  --name NAME           Name to store the recipe under
  --run RUN             Seed amplitudes, background and applied field from this
                        reduced run
  --initial NAME=VALUE  Starting value (repeatable)
  --fix NAME=VALUE      Hold a parameter at a value (repeatable)
  --tmin TMIN           Fit window start / µs
  --tmax TMAX           Fit window end / µs
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to write into (default: ./asymmetry-work)
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
  --workdir WORKDIR  Work directory to read (default: ./asymmetry-work)
```

## `asymmetry fit-global`

```
usage: asymmetry fit-global [-h] --runs RUNS --recipe RECIPE [--fix NAME=VALUE]
                            [--free NAME] --shared P,Q [--field-param NAME]
                            [--strategy {joint,profiled,least_squares}]
                            [--order QUANTITY] [--x RUN=VALUE,...] [--name NAME]
                            [--plot] [--json] [--workdir WORKDIR]
                            folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --runs RUNS           Runs in one simultaneous-fit group
  --recipe RECIPE       Recipe file, or the name of one in the work directory's
                        recipes/
  --fix NAME=VALUE      Hold a parameter at a value (repeatable)
  --free NAME           Release a parameter the recipe holds (repeatable)
  --shared P,Q          Parameters fitted once across all runs
  --field-param NAME    Set this parameter from each run's field and hold it
                        (repeatable)
  --strategy {joint,profiled,least_squares}
  --order QUANTITY      Quantity the runs are ordered and trended along: temperature
                        (the setpoint), sample_temperature_logged, field or run, read
                        from the files; or any other name, whose value for every run
                        you give with --x
  --x RUN=VALUE,...     Per-run values of a quantity the files do not record, e.g. '--
                        order concentration --x 101=0,102=0.25,103=0.5'
  --name NAME           Stored fit name
  --plot                Write one fitted plot per run
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to read and write (default: ./asymmetry-work)
```

## `asymmetry fit-series`

```
usage: asymmetry fit-series [-h] --runs RUNS --recipe RECIPE [--fix NAME=VALUE]
                            --order QUANTITY [--x RUN=VALUE,...] [--tmin TMIN]
                            [--tmax TMAX] [--global P,Q] [--start RUN] [--name NAME]
                            [--plot] [--json] [--workdir WORKDIR]
                            folder

positional arguments:
  folder             Directory holding the run files

options:
  -h, --help         show this help message and exit
  --runs RUNS        Run numbers, e.g. '17294-17322'
  --recipe RECIPE    Recipe file, or the name of one in the work directory's recipes/
  --fix NAME=VALUE   Hold a parameter at a value (repeatable)
  --order QUANTITY   Quantity the runs are ordered and trended along: temperature (the
                     setpoint), sample_temperature_logged, field or run, read from the
                     files; or any other name, whose value for every run you give with
                     --x
  --x RUN=VALUE,...  Per-run values of a quantity the files do not record, e.g. '--
                     order concentration --x 101=0,102=0.25,103=0.5'
  --tmin TMIN        Fit only above this time / µs
  --tmax TMAX        Fit only below this time / µs
  --global P,Q       Parameters held identical across every run. The batch is block-
                     separable, so these are pinned at their recipe value and not
                     fitted; everything else is free per run
  --start RUN        Chain outward from this run in both directions, instead of from
                     the first run in scan order. Screen the run with the clearest
                     structure ('asymmetry wizard --run N'), then start the series
                     there ('--start N'): every fit then warm-starts from a neighbour
                     nearer the run the recipe describes
  --name NAME        Name to store the series under (default: series-<recipe stem>)
  --plot             Write plots/<name>/<run>.png per run and
                     plots/<name>-trend-<param>.png per free parameter
  --json             Emit the machine-readable payload
  --workdir WORKDIR  Work directory to read and write (default: ./asymmetry-work)
```

## `asymmetry trend`

```
usage: asymmetry trend [-h] --series SERIES [--csv CSV] [--plot] [--model EXPR]
                       [--param PARAM] [--xmin XMIN] [--xmax XMAX]
                       [--initial NAME=VALUE] [--fix NAME=VALUE] [--exclude RUNS]
                       [--json] [--workdir WORKDIR]
                       folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --series SERIES       Name the series was stored under
  --csv CSV             Also write the table to this CSV file
  --plot                Write plots/<series>-trend-<param>.png for every free
                        parameter (with --model, only for --param, with the fitted
                        curve)
  --model EXPR          Fit this parameter-vs-x expression to --param, e.g.
                        'OrderParameter', 'Redfield'
  --param PARAM         The trend column --model is fitted to
  --xmin XMIN           Fit range start, in x units
  --xmax XMAX           Fit range end, in x units
  --initial NAME=VALUE  Model start value (repeatable)
  --fix NAME=VALUE      Hold a model parameter at this value (repeatable)
  --exclude RUNS        Leave these runs out of the fit (every other run with a value
                        enters)
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to read and write (default: ./asymmetry-work)
```

## `asymmetry fourier`

```
usage: asymmetry fourier [-h] --run RUN [--name NAME]
                         [--window {none,hann,cosine,gaussian,lorentzian}]
                         [--padding PADDING] [--tmin TMIN] [--tmax TMAX]
                         [--phase PHASE] [--filter-tau FILTER_TAU] [--fmin FMIN]
                         [--fmax FMAX] [--peaks PEAKS] [--plot] [--json]
                         [--workdir WORKDIR]
                         folder

positional arguments:
  folder                Directory holding the run files

options:
  -h, --help            show this help message and exit
  --run RUN             Reduced run number
  --name NAME           Stored spectrum name (default: run-<N>)
  --window {none,hann,cosine,gaussian,lorentzian}
  --padding PADDING     Zero-padding factor (default: 4)
  --tmin TMIN           Transform-window start / µs
  --tmax TMAX           Transform-window end / µs
  --phase PHASE         Phase rotation / degrees
  --filter-tau FILTER_TAU
                        Time constant for the lorentzian/gaussian window / µs
  --fmin FMIN           Lowest reported frequency / MHz
  --fmax FMAX           Highest reported frequency / MHz
  --peaks PEAKS         Maximum peaks to report
  --plot                Write plots/<name>.png
  --json                Emit the machine-readable payload
  --workdir WORKDIR     Work directory to read and write (default: ./asymmetry-work)
```

## `asymmetry audit`

```
usage: asymmetry audit [-h] [--workdir WORKDIR] [--json] draft

positional arguments:
  draft              The draft summary (a text or Markdown file)

options:
  -h, --help         show this help message and exit
  --workdir WORKDIR  Work directory whose cli-output.log to check against (repeatable;
                     default: every ./asymmetry-work* directory here)
  --json             Emit the machine-readable payload
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
                               [--link] [--force] [--json]

options:
  -h, --help            show this help message and exit
  --agent {claude,codex}
  --project             Install under ./.claude or ./.agents instead of the home
                        directory
  --into INTO           Install under this directory instead of the agent's usual
                        location
  --link                Symlink the packaged skill instead of copying it, so edits in
                        a checkout reach the agent without reinstalling (development)
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
usage: asymmetry info [-h] [--json] file

positional arguments:
  file        Path to a μSR data file

options:
  -h, --help  show this help message and exit
  --json      Emit the machine-readable payload
```
