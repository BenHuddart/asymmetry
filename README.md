<img align="right" src="https://raw.githubusercontent.com/BenHuddart/asymmetry/main/docs/_static/logo-on-light.svg" alt="Asymmetry: a red A and damped wave over a blue time axis" width="18%">

# Asymmetry

A Python library, command-line tool, and desktop application for muon-spin spectroscopy (μSR).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/BenHuddart/asymmetry/blob/main/LICENSE) [![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](#project-status-and-validation) [![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://benhuddart.github.io/asymmetry/)

**[Download the desktop application](https://github.com/BenHuddart/asymmetry/releases/latest)** ·
[Documentation](https://benhuddart.github.io/asymmetry/) ·
[Issue tracker](https://github.com/BenHuddart/asymmetry/issues) ·
[Changelog](https://github.com/BenHuddart/asymmetry/blob/main/CHANGELOG.md)

## Overview

Asymmetry provides tools for reducing, analysing, and visualising μSR data through Python,
command-line, and graphical interfaces. It supports detector grouping and calibration, time-
and frequency-domain fitting, Fourier and maximum-entropy analysis, parameter trends, and
saving analysis sessions for reuse.

- **Data loading:** read supported ISIS NeXus, PSI BIN/MDU, and ROOT acquisitions, including
  multi-period runs. See the [loading guide](https://benhuddart.github.io/asymmetry/reference/loading_data.html)
  for supported layouts and required readers.
- **Reduction and calibration:** select detector groups, estimate detector balance, apply
  timing, deadtime, and background settings, and rebin or combine compatible data.
- **Fitting:** fit time-domain signals, raw detector counts, multiple detector groups, or
  simultaneous run series using built-in models and composite expressions.
- **Spectra:** compute Fourier spectra with windowing and phase controls, or reconstruct
  spectra with a maximum-entropy solver.
- **Scans and trends:** analyse integral field/RF scans, compare fitted parameters across
  runs, and fit parameter trends with physical models.
- **Simulation:** generate synthetic detector counts for teaching and checking an analysis
  with known inputs and recorded random seeds.
- **Projects and export:** save analysis sessions with their settings and provenance, reopen
  them, and export plot data, figures, and GLE scripts.
- **Scripting and agents:** use the Python API or the `asymmetry` CLI, with JSON output,
  headless plots, and a packaged skill for Claude Code and Codex.

## Project status and validation

> **Asymmetry is beta software under active development.** Its code and documentation have
> been written by AI coding agents (Claude Code and Codex). Throughout development, numerical
> behaviour has been checked through automated tests, independent mathematical references,
> and comparisons with established μSR software. See the
> [software comparisons](https://benhuddart.github.io/asymmetry/explanation/comparison.html)
> for tested cases, agreement, and known limitations.
>
> Check scientific claims and model definitions against primary literature and other trusted
> sources, and validate results for your analysis before relying on them for publication.
> APIs may change between releases.

Projects saved by released versions remain loadable. The
[numerical accuracy guide](https://benhuddart.github.io/asymmetry/explanation/numerical_accuracy.html)
explains evaluation targets and their scientific scope.

## Installation

### Desktop application

[Download the latest release](https://github.com/BenHuddart/asymmetry/releases/latest) for
**Windows x64** or **macOS on Apple Silicon**. The installer includes Python and the GUI
dependencies. Open Asymmetry from the Windows Start menu or the macOS Applications folder.
Linux and Intel Mac users can use a Python installation.

### Python installation

Asymmetry requires Python 3.11 or later; clean installations are checked on Python 3.11,
3.12, and 3.13. Use a virtual environment as described in the
[installation guide](https://benhuddart.github.io/asymmetry/getting_started/installation.html).
For a version published on PyPI, install the full end-user Python feature set and launch:

```bash
python -m pip install "asymmetry[all]"
asymmetry-gui
```

If the version you need is not available on PyPI, use the
[source installation](https://benhuddart.github.io/asymmetry/getting_started/installation.html#source-installation).
The core library and CLI can also be installed on their own:

```bash
python -m pip install asymmetry
```

Extras select additional Python dependencies and can be combined:

```bash
python -m pip install "asymmetry[gui,hdf5,root]"
```

| Extra | Provides |
|---|---|
| `all` | GUI, file readers, headless plots, and GLE support; excludes development tools and CUDA |
| `gui` | Desktop interface and Matplotlib |
| `agent` | Headless plots and HDF5 loading for CLI/agent workflows |
| `hdf5` | HDF5 NeXus reading and writing |
| `hdf4` | Legacy HDF4 NeXus reading |
| `root` | Supported μSR ROOT file reading |
| `gle` | GLE script generation |
| `gpu` | Optional CUDA acceleration for MaxEnt |
| `dev` | Contributor tools for testing, lint, and documentation |

HDF4 can require native libraries, and compiling GLE scripts requires the external GLE
program. GPU use requires compatible NVIDIA hardware and drivers. See the
[installation guide](https://benhuddart.github.io/asymmetry/getting_started/installation.html#external-programs-and-runtimes)
for platform requirements. The desktop installers bundle their HDF4 runtime.

## Quick start

For a complete GUI analysis with no measurement files, follow the
[synthetic first-analysis walkthrough](https://benhuddart.github.io/asymmetry/getting_started/quickstart.html).
It generates detector counts, fits a relaxation, and saves a reopenable project and report.

### Fit synthetic data in Python

This example needs only the core installation. Time is in μs, asymmetry is in percent,
and `sigma` is in percentage points. The seeded Gaussian observations are a
demonstration, not acquired detector counts.

```python
import numpy as np
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.models import exponential_relaxation

time_us = np.linspace(0.0, 10.0, 201)
sigma = np.full_like(time_us, 0.3)
signal = 24.0 * np.exp(-0.3 * time_us) + 2.0
rng = np.random.default_rng(23)
asymmetry = signal + rng.normal(0.0, sigma)

parameters = ParameterSet([
    Parameter("A0", 20.0, min=0.0),
    Parameter("Lambda", 0.2, min=0.0),
    Parameter("baseline", 1.0),
])
result = FitEngine().fit_arrays(
    time_us, asymmetry, sigma, exponential_relaxation, parameters
)
print(result.parameters["Lambda"].value)
```

The fitted rate should be close to the generating value, 0.3 μs⁻¹. Continue with the
[API cookbook](https://benhuddart.github.io/asymmetry/reference/cookbook.html) for acquisition
loading, calibration, reduction, and fitting recipes.

### Use the CLI or an AI coding agent

For a published release, install the headless workflow with pipx:

```bash
pipx install "asymmetry[agent]"
asymmetry --help
asymmetry skill install --agent claude  # or --agent codex
asymmetry skill check
```

Restart the agent after installing or upgrading its skill. After upgrading Asymmetry, rerun
`asymmetry skill install` and `asymmetry skill check`. Add `hdf4` or `root` to the extras when
your files require them. The desktop installer does not put this CLI on `PATH`.
See the [CLI and agent guide](https://benhuddart.github.io/asymmetry/reference/agent_workflow.html)
for source installation, skill management, and analysis commands.

## Documentation and support

- [Installation](https://benhuddart.github.io/asymmetry/getting_started/installation.html)
  and [first analysis](https://benhuddart.github.io/asymmetry/getting_started/quickstart.html)
- [Worked analyses](https://benhuddart.github.io/asymmetry/workflows/index.html)
  and [feature reference](https://benhuddart.github.io/asymmetry/reference/index.html)
- [API cookbook](https://benhuddart.github.io/asymmetry/reference/cookbook.html)
  and [API reference](https://benhuddart.github.io/asymmetry/api/index.html)
- [Software comparisons](https://benhuddart.github.io/asymmetry/explanation/comparison.html)
  and [numerical accuracy](https://benhuddart.github.io/asymmetry/explanation/numerical_accuracy.html)
- [Report an issue](https://github.com/BenHuddart/asymmetry/issues)
  or [contribute](https://github.com/BenHuddart/asymmetry/blob/main/CONTRIBUTING.md)

Asymmetry is maintained by Ben Huddart. Models and file formats are implemented from
published mathematics, documented formats, and observed behaviour. We acknowledge the work
behind WiMDA, musrfit, and Mantid; see
[NOTICE](https://github.com/BenHuddart/asymmetry/blob/main/NOTICE) for attribution and the
[MIT licence](https://github.com/BenHuddart/asymmetry/blob/main/LICENSE) for Asymmetry's terms.
