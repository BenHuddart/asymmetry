# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Parameter-model field component `GaussianLCR` in Eq. (4) notation from PRL 135, 046704 (2025): `lambda_LCR(B) = f * G(B; B0; Bwid)`.

### Changed
- Parameter-model docs now explicitly state that Redfield exponent `m` is dimensionless.
- In the model-fit GUI for field-series parameter fits, component selection now avoids redundant constants: Lambda-like y-parameters show `Lambda_bg` and hide `Constant`, while non-Lambda y-parameters show `Constant` and hide `Lambda_bg`.
- Model parameter fits and cross-group parameter fits now run asynchronously in the GUI, with explicit in-progress status text and temporary control locking to avoid blocking the UI.
- Asymmetry uncertainty calculation now follows Mantid `AsymmetryCalc` behavior, including default uncertainty for zero-denominator bins.
- Main plot now suppresses bins with non-positive grouped denominator (`F + alpha*B <= 0`) to avoid displaying undefined asymmetry points.
- Main plot bunch-factor control has been removed; bunching is managed in the Grouping workflow.
- Grouping deadtime correction now uses Mantid-style good-frame normalization: `Ncorr = N / (1 - N*tau/(dt*good_frames))`.
- Grouping dialog bunching-factor input range has been expanded to support large values.
- Main-plot limit controls now use independent `Auto X` / `Auto Y`; manual X/Y limits apply on Enter and no longer require an Apply button.
- `Auto Y` now computes limits from points inside the current X range and prioritizes reliable foreground points.
- Run Info now provides include-in-browser checkboxes and per-row log plotting in both the primary table and Advanced subwindow.
- Project persistence now round-trips grouping settings and restored plot limits more reliably, including list-returning/multi-period loader paths.

### Documentation
- Updated GUI user guide to document current axis-control behavior, Run Info include/log workflows, Advanced subwindow usage, and persistence details for dynamic columns and grouping settings.

### Tests
- Added regression and end-to-end persistence tests for:
	- plot-limit restore with dataset replot,
	- grouping override save/restore,
	- multi-period restore dataset selection,
	- project round-trip restoring grouping and axis limits,
	- Run Info/Data Browser synthetic column integration.

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
