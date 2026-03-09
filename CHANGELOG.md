# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-09

### Added
- Initial release of Asymmetry μSR data analysis library
- Core data structures: `MuonDataset`, `Run`, `Histogram`
- WiMDA `.wim` file loader with metadata extraction
- Logbook/run-table management for multiple datasets
- Data transformations: asymmetry calculation, grouping, rebinning
- Fitting engine with iminuit backend
- Built-in μSR fit models: exponential, Gaussian, stretched exponential, oscillatory, static GKT
- Global (simultaneous) fitting with shared and local parameters
- Fourier analysis: FFT with apodization windows (Hann, Hamming, Blackman, Bartlett)
- PySide6-based GUI with data browser, plot panel, fit panel, and fit parameters panel
- GLE export integration via gleplot for publication-quality parameter trend plots
- Interactive plot panel with Matplotlib backend
- Data browser with sortable columns and Excel-style filters
- Fit parameters panel with matplotlib and GLE plotting options
- Command-line interface: `asymmetry` and `asymmetry-gui`
- Comprehensive test suite with 97 tests and 71% coverage
- Sphinx documentation with user guide and API reference
- Support for Python 3.10, 3.11, 3.12, and 3.13

### Technical Details
- Pure-Python core library with no GUI dependencies
- Plugin-based architecture for data loaders and fit models
- Separation of concerns: core analysis engine vs. GUI
- Full scriptability for batch processing and Jupyter workflows
