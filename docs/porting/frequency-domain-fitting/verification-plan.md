# Frequency-Domain Fitting Verification Plan

## Core

- Fit Gaussian and Lorentzian synthetic spectra through ``FitEngine``.
- Verify parameter metadata for ``height``, ``nu0``, ``fwhm``, ``bg``, and
  ``slope``.
- Verify ``nu0``/``fwhm`` to ``B0``/``Bwid`` conversion.

## GUI

- Frequency view enables the Fit panel only after a cached spectrum exists.
- Single frequency fit overlays on the frequency plot.
- Global frequency fit uses selected runs with cached spectra.
- Missing cached spectra do not trigger recomputation in V1.
- Results reach the Parameters panel as the ``Frequency Domain`` group.

## Persistence

- Schema v4 projects migrate to v5 with an empty ``frequency_fit_state``.
- Schema v5 saves/restores cached Fourier spectra, frequency-fit settings,
  frequency plot overlays, and trend-panel output.

## Documentation

- User guide includes the FFT-to-fit workflow.
- API docs expose spectral helpers and document the shared fitting engine.
- Project-file docs describe the new frequency-fit namespace.
