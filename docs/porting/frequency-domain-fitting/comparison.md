# Frequency-Domain Fitting Comparison

## WiMDA

WiMDA's Fourier workflow focuses on transforming grouped detector-count
signals, phase correction, apodisation, MaxEnt, background subtraction, and
spectral inspection.  It does not provide a modern simultaneous global peak-fit
workflow directly on FFT spectra.  Its fit-table/model tools are the closer
analogue for fitting extracted quantities versus field or temperature.

## musrfit

musrfit has extensive frequency-domain preprocessing and phase-correction
machinery, including ``phaseOptReal`` and linear optimizer phase terms.  The
core fitting model remains user-function based rather than a distinct
point-and-click FFT peak fitter.  For Asymmetry V1, musrfit mainly informs the
phase/preprocessing boundary; complex-spectrum fitting is deferred.

## Mantid

Mantid's Muon ALC and frequency-domain workflows are the closest reference for
peak fitting.  Gaussian and Lorentzian peaks plus simple backgrounds produce
peak centres and widths that can be trended against field, temperature, or run.
This maps cleanly onto Asymmetry's existing parameter-trending panel.

## Asymmetry Seams

- Fourier spectra are already represented as ``MuonDataset(time=freq_mhz,
  asymmetry=display_values, error=...)`` with ``plot_domain=frequency``.
- ``FitEngine.fit`` and ``FitEngine.global_fit`` are x-axis agnostic.
- ``CompositeModel`` can host frequency components if the parameter metadata is
  added.
- ``FitParametersPanel`` already accepts fit results and datasets by run number.
- Project state needs a separate namespace so frequency fits do not overwrite
  time-domain fit state.

## Implementation Difference

Asymmetry intentionally fits the displayed real-valued spectrum in V1.  This is
less general than fitting the complex FFT, but it matches what users inspect in
the GUI and keeps the workflow consistent with existing single/global/trending
tools.
