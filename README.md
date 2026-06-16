# Asymmetry

A Python toolkit for muon-spin spectroscopy (μSR) data reduction, fitting, and visualization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://benhuddart.github.io/asymmetry/)

⬇️ **[Download the latest release](https://github.com/BenHuddart/asymmetry/releases/latest)** · 📖 [Documentation](https://benhuddart.github.io/asymmetry/) · [Changelog](CHANGELOG.md) · [Issue tracker](https://github.com/BenHuddart/asymmetry/issues) · [Contributing](CONTRIBUTING.md)

## Overview

Asymmetry combines a pure-Python analysis library with a PySide6 desktop application for
day-to-day μSR analysis: loading raw or reduced data, applying detector grouping and asymmetry
transforms, fitting time-domain signals, exploring parameter trends, and saving full GUI sessions
for later reuse.

## Project status

> **Asymmetry is alpha software under active development.** The WiMDA parity programme is
> feature-complete as of v0.4.0, so the core analysis engine and GUI now cover the full reference
> μSR workflow, but the inherited functionality still requires extensive real-world testing.
> Behaviour, APIs, and the `.asymp` project format may still change between releases, and some
> rough edges are expected. Validate results against an established tool before relying on them
> for published work, and please report issues you hit.

## Main functionality

- **Data loading**: load ISIS muon NeXus `.nxs` / `.nexus` (modern HDF5 and legacy HDF4
  containers), PSI BIN/MDU, and ROOT files through a common API, including multi-period
  NeXus runs.
- **Core data workflows**: apply grouping, estimate alpha, compute asymmetry, rebin data, and
  co-add compatible datasets with propagated uncertainties.
- **Time-domain fitting**: fit single datasets or simultaneous multi-dataset series using built-in
  μSR models and calculator-style composite expressions assembled with arithmetic operators and
  parentheses.
- **Count-domain fitting**: fit raw forward/backward detector counts directly (single-histogram or
  F+B), with deadtime, background, and double-pulse terms.
- **Grouped time-domain fitting**: fit multiple detector groups jointly for Knight-shift, vortex-state,
  and geometry-sensitive observables. A **Single** tab fits one run's groups; a **Batch** tab fits a
  multi-run series with the same model, collecting results for parameter trending.
- **Longitudinal-field KT support**: includes `LongitudinalFieldKT` / `LFKuboToyabe` models with
  field-aware defaults (`B_L` can be initialized from run metadata where available).
- **Batch and global fit parameter typing**: per-parameter role selection — `Global` (shared across
  runs), `Local` (per-run), `Fixed` — with the relationship (batch vs. global) derived automatically
  from the role table. Results are recorded as a trendable `FitSeries`.
- **Parameter-model fitting**: fit field-, temperature-, or run-dependent parameter trends,
  including superconducting penetration-depth workflows. The Fit Parameters panel is
  representation-aware: switching between F-B Asymmetry, Detector Groups, and FFT views
  automatically shows the series for the active representation. Series buttons use red
  accents for visual identity; member datasets are highlighted red in the browser while
  the panel is visible. Right-click a series button to **rename** it, **select its
  members** in the browser (true selection), or **delete** the series from the project.
- **Derived composite parameters**: define expression-based parameters in the Fit Parameters panel
  with safe parsing and first-order uncertainty propagation (including covariance support when
  available).
- **Frequency-domain analysis**: compute grouped FFT spectra with explicit
  apodisation, selectable phase modes, per-run phase tables, and manual or
  estimated phase correction; or reconstruct spectra with a maximum-entropy
  solver, including pulsed-source response and deadtime handling.
- **Simulation**: generate synthetic single- or multi-period count runs from a model, for testing,
  teaching, and fit validation.
- **Logbook and metadata handling**: inspect run metadata, build searchable run logbooks, and use
  metadata columns inside the GUI browser.
- **Interactive GUI**: browse loaded runs, inspect plots, adjust grouping, run fits, trend fitted
  parameters, adjust UI scale, and export plots.
- **Project persistence**: save and reopen `.asymp` project files containing datasets, browser
  state, plot state, fit state, Fourier settings, and per-run Fourier phase-table state.
- **Extensible I/O**: register custom loaders at runtime for additional file formats.
- **Optional publication export**: export trend and plot data for GLE-based figure generation.

## Installation

### Download the desktop application (recommended for most users)

The easiest way to run Asymmetry is the prebuilt desktop installer — no Python setup required.

- **[Download the latest release →](https://github.com/BenHuddart/asymmetry/releases/latest)**

Installers are published on the [Releases page](https://github.com/BenHuddart/asymmetry/releases)
for each tagged version:

| Platform | Artifact |
|----------|----------|
| Windows | Inno Setup installer (`.exe`) with desktop/start-menu shortcuts |
| macOS (Apple Silicon) | `.dmg`, drag-to-Applications |
| macOS (Intel) | `.dmg`, drag-to-Applications |

### Install from source (Python users and contributors)

For scripting against the library, or for development, install from the Git repository.
Asymmetry requires Python 3.10 or later.

#### Clone the repository

```bash
git clone https://github.com/BenHuddart/asymmetry.git
cd asymmetry
```

#### Recommended Python install (full functionality, no dev tools)

```bash
python -m pip install -c constraints.txt ".[gui,hdf5,hdf4,root,gle]"
```

This installs the full end-user feature set (GUI + optional file/export support) without
development dependencies.

#### Other install options from the checked-out repository

```bash
# Core library only
python -m pip install -c constraints.txt .

# With GUI support
python -m pip install -c constraints.txt ".[gui]"

# With GUI and optional file/export support
python -m pip install -c constraints.txt ".[gui,hdf5,hdf4,root,gle]"

# Everything, including development dependencies
python -m pip install -c constraints.txt ".[all]"

# Editable install for contributors
python -m pip install -c constraints.txt -e ".[all]"
```

#### Install directly with pip from GitHub

```bash
# Core library
python -m pip install "git+https://github.com/BenHuddart/asymmetry.git"

# Full end-user feature set (no development extras)
python -m pip install "asymmetry[gui,hdf5,hdf4,root,gle] @ git+https://github.com/BenHuddart/asymmetry.git"
```

Using [constraints.txt](constraints.txt) is recommended for local installs so the scientific Python
stack stays within the versions tested by the project.

#### Optional dependencies

| Component | Requirement |
|-----------|-------------|
| Core | NumPy, SciPy, iminuit |
| GUI | PySide6, Matplotlib |
| NeXus / HDF5 | h5py |
| HDF4 (legacy ISIS `.nxs`) | pyhdf — see note below |
| ROOT import | uproot |
| GLE export | `gleplot` (current git version with foldered exports) plus a local GLE installation |
| Development | pytest, pytest-cov, ruff, Sphinx |

##### Reading legacy HDF4 `.nxs` files

Pre-~2015 ISIS muon runs (the format WiMDA reads natively) are NeXus v1 stored
in an **HDF4** container. Reading them needs the optional `hdf4` extra:

```bash
python -m pip install "asymmetry[hdf4]"
```

On **Linux and macOS** the `pyhdf` wheels bundle the HDF4 C library, so that is
all you need. On **Windows**, `pyhdf` additionally requires the HDF4 runtime
(`hdf.dll` / `mfhdf.dll`): install the conda-forge `hdf4` package, or run
`python packaging/windows/fetch_hdf4_dlls.py <dir>` and point the
`ASYMMETRY_HDF4_DLL_DIR` environment variable at `<dir>`. The **pre-built
Windows and macOS binaries bundle the HDF4 runtime**, so no extra setup is
needed there. See [docs/user_guide/loading_data](docs/user_guide/loading_data.rst)
for details.

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
  - `windows-2025` for the Inno Setup installer
- Packaging:
  - PyInstaller `onedir` build for fast startup
  - macOS DMG output per architecture (dmgbuild settings + generated drag-to-Applications background)
  - Windows Inno Setup installer with desktop/start-menu shortcuts and uninstaller

Release artifacts are built on shared runners and uploaded to the GitHub Release page; executables are not committed to the repository.

### Trigger a release build

```bash
git tag v0.1.0
git push origin v0.1.0
```

Pre-release tags are also supported (for example `v0.1.0-rc.1`).

## Acknowledgements and prior art

Asymmetry stands on the shoulders of the established μSR analysis programs and reimplements
analysis methods studied from them. Models, conventions, and algorithms were learned from these
projects and the μSR literature, then **independently reimplemented** in Python — no source code
was copied or translated. We gratefully acknowledge:

- **WiMDA** — μSR data-analysis program by **Francis L. Pratt** (freely-available freeware; its
  source is not published under an open licence). F. L. Pratt, *Physica B* **289–290**, 710 (2000).
- **musrfit** — A. Suter and B. M. Wojek, *Physics Procedia* **30**, 69 (2012). Licensed under the
  GNU GPL.
- **Mantid** — *Nuclear Instruments and Methods A* **764**, 156 (2014). Licensed under the GNU GPLv3.

Where a model or file format follows one of these tools, the source files cite it directly. See
[NOTICE](NOTICE) for the full attribution statement, and `docs/porting/` for the study notes that
underpin each reimplementation. Physical constants and model formulae are cited to the μSR
literature (notably Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An Introduction*).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Contact

Asymmetry is developed and maintained by **Ben Huddart** (University of Oxford,
Department of Physics), <benjamin.huddart@physics.ox.ac.uk>.

For bug reports and feature requests, please use the
[issue tracker](https://github.com/BenHuddart/asymmetry/issues).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT License. See [LICENSE](LICENSE).
