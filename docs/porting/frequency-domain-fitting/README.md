# Frequency-Domain Fitting Study

Status: implementation pass selected

This study covers fitting displayed Fourier spectra in Asymmetry.  The V1
decision is to fit the displayed, real-valued frequency-domain spectrum with
simple peak and background models, then reuse the existing global-fit and
parameter-trending workflows.

## Current Decision

- Fit target: displayed spectrum, not the complex FFT.
- Canonical x axis: absolute frequency in MHz.
- V1 models: Gaussian peak, Lorentzian peak, constant background, linear
  background.
- Derived trend quantities: centre and FWHM remain stored as ``nu0`` and
  ``fwhm`` in MHz, with field equivalents exposed as ``B0`` and ``Bwid``.
- Persistence: frequency-fit state is separate from time-domain fit state.

## Reference Programs

- WiMDA: frequency spectra are primarily inspected visually; fitting is done
  through separate fit-table/model workflows after quantities are extracted.
- musrfit: frequency-domain workflows are driven by FFT and phase correction;
  model fitting remains centred on user-authored functions.
- Mantid: frequency-domain/ALC workflows include peak fitting and parameter
  extraction, especially Gaussian/Lorentzian peak positions and widths.

## Asymmetry Baseline

- Fourier spectra are cached as plottable ``MuonDataset`` objects.
- The existing ``FitEngine`` can fit any x/y/error arrays.
- Time-domain fit UI, global fitting, project persistence, and fitted-parameter
  trends already exist and should be reused.

## Study Files

- ``comparison.md``: reference-program comparison and Asymmetry seams.
- ``implementation-options.md``: evaluated implementation approaches.
- ``test-data.md``: synthetic validation data strategy.
- ``verification-plan.md``: checks required for the implementation pass.
