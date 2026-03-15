# Asymmetry

A Python library for muon-spin spectroscopy (μSR) data analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-288%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-71%25-yellowgreen.svg)](htmlcov/)

## Overview

Asymmetry provides a modular core engine for μSR data reduction, fitting, and
Fourier analysis, together with an interactive graphical front-end inspired by
[WiMDA](https://shadow.nd.rl.ac.uk/wimda/) for visualization and experiment
management.

### Key features

- **Data I/O** — Load WiMDA (`.wim`) files with automatic asymmetry calculation
  and metadata extraction (temperature, field, title, comments).
- **Data browser** — Sortable run table with Excel-style column filters and
  multi-selection for co-adding or global fitting. Right-click for context
  menu actions: co-add selected runs, separate combined datasets, or delete.
- **Interactive plotting** — Matplotlib-based plot panel with adjustable axis
  limits, bunch factor for noise reduction, and error-bar display.
- **Single-dataset fitting** — Build composite `A(t)` models from
  μSR components (exponential, Gaussian, oscillatory, stretched exponential,
  static Gaussian Kubo-Toyabe, constant background) with ``+``, ``-``, ``*``,
  ``/`` operators, then fit with
  [iminuit](https://scikit-hep.org/iminuit/) using parameter bounds and fixing.
- **Default explicit background** — New fits start with
  ``A(t) = Exponential + A_bg`` so constant background is always visible and
  editable from the start.
- **Global (simultaneous) fitting** — Fit multiple datasets at once with shared
  and per-dataset parameters. Suited for temperature or field series analysis.
- **Fitted-parameter trends** — After a global fit, inspect how varying
  parameters depend on field, temperature, or run number. Supports logarithmic
  axes and export to CSV.
- **GLE export** — Generate publication-quality vector figures of parameter
  trends via [gleplot](https://github.com/bhuddart/gleplot) and the
  [GLE Graphics Layout Engine](http://glx.sourceforge.io/). Data files include
  headers and global parameters as comments. Output formats: PDF, EPS.
- **Fourier analysis** — FFT with configurable apodization windows (Hann,
  Hamming, Blackman, Bartlett) and zero-padding.
- **Co-adding** — Average selected datasets with correct error propagation.
- **Command-line interface** — `asymmetry info` for quick environment checks.
- **Responsive parameter-model fitting UI** — Parameter-model and cross-group
  parameter fits run in the background with clear in-progress feedback.

## Installation

```bash
# Core library only
pip install -c constraints.txt -e .

# With GUI
pip install -c constraints.txt -e ".[gui]"

# Everything (GUI + optional formats + dev tools)
pip install -c constraints.txt -e ".[all]"
```

To avoid environment drift (for example incompatible NumPy/Numba combinations),
use the shared [constraints.txt](constraints.txt) file for local installs.

### Dependencies

| Component | Requirements |
|-----------|-------------|
| Core | Python ≥ 3.10, NumPy, iminuit ≥ 2.0 |
| GUI | PySide6 ≥ 6.5, Matplotlib ≥ 3.7 |
| GLE export | gleplot, GLE (compiled from source or binary) |
| Dev | pytest, ruff, Sphinx |

## Quick start

### Scripting

```python
from asymmetry.core.io import load

# Load a .wim file
dataset = load("data/EMU00012345.wim")

print(f"Run:  {dataset.run_number}")
print(f"T =   {dataset.metadata['temperature']} K")
print(f"B =   {dataset.metadata['field']} G")

# Plot the asymmetry
import matplotlib.pyplot as plt
plt.errorbar(dataset.time, dataset.asymmetry, yerr=dataset.error, fmt=".")
plt.xlabel("Time (μs)")
plt.ylabel("Asymmetry")
plt.show()
```

### GUI

```bash
# Launch the graphical interface
asymmetry-gui
```

Or from Python:

```python
from asymmetry.gui.app import main
main()
```

## Documentation

Full documentation is available in the `docs/` directory and can be built with Sphinx:

```bash
cd docs
pip install -r requirements.txt
make html
```

Then open `docs/_build/html/index.html` in your browser.

The documentation includes:

- **Installation guide**: Detailed setup instructions
- **User guide**: Tutorials for data loading, processing, GUI usage, fitting
  (single and global), Fourier analysis, and GLE export
- **API reference**: Complete API documentation auto-generated from docstrings
- **Architecture**: Design principles and project structure (see `docs/ARCHITECTURE.md`)

See `docs/README.md` for more information about the documentation system.

## Project structure

```
src/asymmetry/
├── core/           # Pure-Python analysis engine (no GUI dependencies)
│   ├── data/       # Data model: MuonDataset, Run, Histogram, Logbook
│   ├── io/         # File I/O loaders (plugin-based; WiMDA supported)
│   ├── transform/  # Asymmetry calculation, grouping, rebinning
│   ├── fitting/    # Fitting engine with built-in μSR models & global fits
│   ├── fourier/    # FFT, apodization windows
│   └── utils/      # Physical constants (γ_μ, τ_μ)
├── gui/            # PySide6 graphical front-end
│  Testing

Asymmetry has a comprehensive test suite with 97 tests and 71% coverage:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest tests/ --cov=src/asymmetry --cov-report=term

# Run specific test file
python -m pytest tests/test_fitting_engine.py -v
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## License

MIT License - see [LICENSE](LICENSE) for details. cli.py          # Command-line interface
```

## License

MIT
