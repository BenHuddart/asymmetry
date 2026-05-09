# Asymmetry

A Python toolkit for muon-spin spectroscopy (μSR) data reduction, fitting, and visualization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Asymmetry combines a pure-Python analysis library with a PySide6 desktop application for
day-to-day μSR analysis: loading raw or reduced data, applying detector grouping and asymmetry
transforms, fitting time-domain signals, exploring parameter trends, and saving full GUI sessions
for later reuse.

## Main functionality

- **Data loading**: load ISIS muon NeXus `.nxs` / `.nexus`, PSI BIN/MDU, and ROOT files through
  a common API, including multi-period NeXus runs.
- **Core data workflows**: apply grouping, estimate alpha, compute asymmetry, rebin data, and
  co-add compatible datasets with propagated uncertainties.
- **Time-domain fitting**: fit single datasets or simultaneous multi-dataset series using built-in
  μSR models and calculator-style composite expressions assembled with arithmetic operators and
  parentheses.
- **Longitudinal-field KT support**: includes `LongitudinalFieldKT` / `LFKuboToyabe` models with
  field-aware defaults (`B_L` can be initialized from run metadata where available).
- **Global-fit parameter typing**: per-parameter role selection for `Global`, `Local`, `Fixed`,
  and `File` (file-backed, per-dataset value) workflows.
- **Parameter-model fitting**: fit field-, temperature-, or run-dependent parameter trends,
  including superconducting penetration-depth workflows.
- **Derived composite parameters**: define expression-based parameters in the Fit Parameters panel
  with safe parsing and first-order uncertainty propagation (including covariance support when
  available).
- **Fourier analysis**: compute FFT spectra with selectable windows and zero padding.
- **Logbook and metadata handling**: inspect run metadata, build searchable run logbooks, and use
  metadata columns inside the GUI browser.
- **Interactive GUI**: browse loaded runs, inspect plots, adjust grouping, run fits, trend fitted
  parameters, adjust UI scale, and export plots.
- **Project persistence**: save and reopen `.asymp` project files containing datasets, browser
  state, plot state, fit state, and Fourier settings.
- **Extensible I/O**: register custom loaders at runtime for additional file formats.
- **Optional publication export**: export trend and plot data for GLE-based figure generation.

## Installation

Asymmetry requires Python 3.10 or later. For local or development use, installing from the Git
repository is the recommended path.

### Clone the repository

```bash
git clone https://github.com/BenHuddart/asymmetry.git
cd asymmetry
```

### Recommended user install (full functionality, no dev tools)

```bash
python -m pip install -c constraints.txt ".[gui,hdf5,root,gle]"
```

This installs the full end-user feature set (GUI + optional file/export support) without
development dependencies.

### Other install options from the checked-out repository

```bash
# Core library only
python -m pip install -c constraints.txt .

# With GUI support
python -m pip install -c constraints.txt ".[gui]"

# With GUI and optional file/export support
python -m pip install -c constraints.txt ".[gui,hdf5,root,gle]"

# Everything, including development dependencies
python -m pip install -c constraints.txt ".[all]"

# Editable install for contributors
python -m pip install -c constraints.txt -e ".[all]"
```

### Install directly with pip from GitHub

```bash
# Core library
python -m pip install "git+https://github.com/BenHuddart/asymmetry.git"

# Full end-user feature set (no development extras)
python -m pip install "asymmetry[gui,hdf5,root,gle] @ git+https://github.com/BenHuddart/asymmetry.git"
```

Using [constraints.txt](constraints.txt) is recommended for local installs so the scientific Python
stack stays within the versions tested by the project.

### Optional dependencies

| Component | Requirement |
|-----------|-------------|
| Core | NumPy, SciPy, iminuit |
| GUI | PySide6, Matplotlib |
| NeXus / HDF5 | h5py |
| ROOT import | uproot |
| GLE export | `gleplot` (current git version with foldered exports) plus a local GLE installation |
| Development | pytest, pytest-cov, ruff, Sphinx |

## Quick start

### Load a dataset in Python

```python
from asymmetry.core.io import load

dataset_or_periods = load("data/EMU00012345.nxs")
dataset = dataset_or_periods[0] if isinstance(dataset_or_periods, list) else dataset_or_periods

print(dataset.summary())
```

The same `load(...)` entry point works for supported NeXus, PSI, and ROOT files.

### Launch the GUI

```bash
asymmetry-gui
```

Within the GUI, use **Edit Function...** (single/global fit) and **Edit Model...** (parameter
trending) to build expression-based models with grouped terms and live validation.

## Documentation

The Sphinx documentation source lives in [docs](docs). The intended GitHub Pages site for the
project is:

- [https://benhuddart.github.io/asymmetry/](https://benhuddart.github.io/asymmetry/)

To build the documentation locally:

```bash
python -m pip install -c constraints.txt -r docs/requirements.txt
make -C docs html
```

Then open `docs/_build/html/index.html` in a browser.

The documentation covers installation, data loading, detector grouping, vector-polarization
workflows, fitting, parameter trending, superconducting analysis, project files, and the API
reference.

## Project structure

```text
src/asymmetry/
├── core/           # Analysis engine, data model, loaders, transforms, fitting, Fourier tools
├── gui/            # PySide6 application, panels, dialogs, and windows
├── cli.py          # Command-line entry point
└── __main__.py     # python -m asymmetry
```

Runnable examples live in [examples](examples), including data loading, transforms, composite
models, parameter trending, project files, and custom loader registration.

## Testing

```bash
python tools/harness.py test
python -m pytest --cov=src/asymmetry --cov-report=term
```

For agent and CI-aligned validation, prefer the harness ladder in the next section.

## Agent harness

Asymmetry keeps agent-facing project knowledge in the repository. Start with
[AGENTS.md](AGENTS.md), then use the harness commands for repeatable validation:

```bash
python tools/harness.py structural
python tools/harness.py lint
python tools/harness.py test
python tools/harness.py validate
```

See [docs/HARNESS.md](docs/HARNESS.md) for the validation ladder and agent workflow.
The lint step checks `src`, `tests`, and `tools`. Locally, the harness prefers
the project `.venv` automatically when it exists.

## Executable releases

Executable GUI releases are built in GitHub Actions and attached to GitHub Releases.

- Workflow: [.github/workflows/release.yml](.github/workflows/release.yml)
- Trigger: push a git tag matching `v*` (for example `v0.1.0`)
- Runners:
  - `macos-15-intel` for Intel macOS DMG
  - `macos-14` for Apple Silicon macOS DMG
  - `windows-latest` for NSIS installer
- Packaging:
  - PyInstaller `onedir` build for fast startup
  - macOS DMG output per architecture (dmgbuild settings + generated drag-to-Applications background)
  - Windows NSIS installer with desktop/start-menu shortcuts and uninstaller

Release artifacts are built on shared runners and uploaded to the GitHub Release page; executables are not committed to the repository.

### Trigger a release build

```bash
git tag v0.1.0
git push origin v0.1.0
```

Pre-release tags are also supported (for example `v0.1.0-rc.1`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT License. See [LICENSE](LICENSE).
