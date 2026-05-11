# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2026-05-11

### Changed
- Packaged GUI builds now load the splash logo more reliably from bundled resources and include the spinbox SVG arrow assets required by the Qt stylesheet.
- GLE export compilation now runs from the export bundle directory so generated `.dat` and `.fit` sidecars resolve correctly in packaged builds, and preview handling is more robust after export failures.
- Co-add compatibility now ignores per-run frame-count metadata when grouping settings otherwise match, and combined-run log temperatures now display as event-weighted averages.
- Global-fit parameter role selections are preserved after fit completion, and long fit-function formulas wrap more cleanly in the fit UI.

## [0.2.1] - 2026-04-17

### Changed
- Combined datasets now mirror the grouping state of their hidden source datasets, and grouping edits on a combined row update the hidden sources before rebuilding the combined result.

### Documentation
- Documented the stricter co-add grouping rules, mixed-source restriction, and the way grouping edits on combined datasets propagate back to their source runs.

### Tests
- Added regression coverage for grouping-compatible co-adds, blocked mismatched/mixed-family co-add attempts, and grouping edits applied through combined datasets.

## [0.2.0] - 2026-04-03

### Added
- Parameter-model field component `GaussianLCR` in Eq. (4) notation from PRL 135, 046704 (2025): `lambda_LCR(B) = f * G(B; B0; Bwid)`.

### Changed
- Parameter-model docs now explicitly state that Redfield exponent `m` is dimensionless.
- In the model-fit GUI for field-series parameter fits, component selection now avoids redundant constants: Lambda-like y-parameters show `Lambda_bg` and hide `Constant`, while non-Lambda y-parameters show `Constant` and hide `Lambda_bg`.
- Model parameter fits and cross-group parameter fits now run asynchronously in the GUI, with explicit in-progress status text and temporary control locking to avoid blocking the UI.
- Composite fit models now share a single amplitude parameter across each multiplicative/divisive chain instead of assigning a separate amplitude to every factor.
- Asymmetry uncertainty calculation now follows Mantid `AsymmetryCalc` behavior, including default uncertainty for zero-denominator bins.
- Main plot now suppresses bins with non-positive grouped denominator (`F + alpha*B <= 0`) to avoid displaying undefined asymmetry points.
- Main plot bunch-factor control has been removed; bunching is managed in the Grouping workflow.
- Grouping deadtime correction now uses Mantid-style good-frame normalization: `Ncorr = N / (1 - N*tau/(dt*good_frames))`.
- Grouping dialog bunching-factor input range has been expanded to support large values.
- Main-plot limit controls now use independent `Auto X` / `Auto Y`; manual X/Y limits apply on Enter and no longer require an Apply button.
- `Auto Y` now computes limits from points inside the current X range and prioritizes reliable foreground points.
- Run Info now provides include-in-browser checkboxes and per-row log plotting in both the primary table and Advanced subwindow.
- Advanced Run Info metadata filtering now includes an inline search field, and the summary table exposes sample orientation for promotion into the Data Browser.
- Data Browser extra metadata columns now use friendly labels for known Run Info fields.
- Project persistence now round-trips grouping settings and restored plot limits more reliably, including list-returning/multi-period loader paths.
- Main-plot GLE export has moved from File/toolbar menu actions to in-panel controls: **Export Plot(s) to GLE** with a PDF/EPS format selector.
- Main-plot GLE export now supports data-only or fitted exports for plotted datasets, label-based sidecar filenames, and exports data as error bars plus fits as line curves when present.
- Main-plot `.dat` sidecars now include run/grouping metadata headers and are rewritten after GLE save so metadata survives helper-generated file overwrites.
- Main-plot ``.fit`` sidecar headers now include fit-function descriptions, fit statistics, and fitted parameter values/uncertainties when available.
- Grouping launch now preselects the highlighted runs, and newly loaded runs inherit the most recent in-browser grouping payload from the highest run number when possible.
- Two-period red/green grouping mode now computes `G-R` and `G+R` in asymmetry space (`A_G - A_R`, `A_G + A_R`) with uncertainty propagation by quadrature.
- Multi-run overlays in RG mode now use contrasting colors for additional runs so selected traces remain visually distinguishable.

### Documentation
- Updated the composite-model guide to document shared amplitudes across multiplicative/divisive chains and the resulting formula/parameter-table behavior.
- Updated the GUI user guide to document alpha display, Run Info search/orientation workflows, friendly Data Browser metadata headers, grouping preselection/template inheritance, and persistence details for dynamic columns and grouping settings.
- Documented the main-plot export workflow for plotted datasets, including data-only exports, label-based ``.dat``/``.fit`` naming, metadata-rich ``.dat`` headers, annotation export, and ``.fit`` header metadata.
- Documented two-period RG mode behavior in the GUI guide, including mode definitions, asymmetry-space `G±R` formulas, uncertainty propagation, and plotting color behavior.

### Tests
- Added regression and end-to-end persistence tests for:
	- plot-limit restore with dataset replot,
	- grouping override save/restore,
	- multi-period restore dataset selection,
	- project round-trip restoring grouping and axis limits,
	- Run Info/Data Browser synthetic column integration.
- Added regression tests covering two-period `G-R` asymmetry-space subtraction and multi-run RG overlay color distinction.

## [0.1.0] - 2026-03-09

### Added
- Initial release of Asymmetry μSR data analysis library
- Core data structures: `MuonDataset`, `Run`, `Histogram`
- ISIS muon NeXus file loader with metadata extraction
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
