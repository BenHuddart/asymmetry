# Frequency-Domain Fitting Implementation Options

## Option A: Reuse FitEngine And CompositeModel

Add frequency-domain components to the composite registry, switch the Fit panel
between time and frequency domains, and feed cached Fourier spectra into the
existing single/global fit paths.

Decision: selected for V1.

Pros:

- Minimal numerical risk.
- Reuses global fitting and fitted-parameter trending.
- Keeps the GUI workflow familiar.
- Keeps frequency fits scriptable through the core fitting API.

Cons:

- The component registry must carry both time-domain and frequency-domain
  functions until a fuller registry filter is added.
- Complex-spectrum fitting is deferred.

## Option B: Separate Spectral Fit Engine

Create a new fitting subsystem dedicated to Fourier spectra.

Pros:

- Clean conceptual separation.
- Easier to add complex-valued costs later.

Cons:

- Duplicates global fitting, parameter tables, persistence, and trend export.
- Higher risk of inconsistent UI behavior.

## Option C: Complex FFT Fit From Start

Fit real and imaginary FFT channels together using a custom residual.

Pros:

- More complete for phase-sensitive workflows.

Cons:

- Requires new UI concepts, persistence shape, error model, and documentation.
- Not necessary for the V1 peak-centre/FWHM workflow.

## V1 Interface

- Default frequency model: ``GaussianPeak + ConstantBackground``.
- Alternative peak model: ``LorentzianPeak + ConstantBackground``.
- Optional background: ``LinearBackground``.
- Fit x values are absolute MHz even when the frequency plot displays field or
  relative frequency.
- Trend table receives ``nu0``, ``fwhm``, and derived ``B0``/``Bwid`` values.
