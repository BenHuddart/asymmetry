# Fourier Transform Test Data

## Synthetic Cases For The First Slice

1. Pure cosine dataset:

   - signal: `A * cos(2 * pi * f * t)`
   - expectation: dominant frequency appears in the real spectrum with phase 0

2. Phase-shifted cosine dataset:

   - signal: `A * cos(2 * pi * f * t + phi)`
   - expectation: applying `-phi` as a manual FFT phase rotates power back into
     the real component at the dominant frequency

3. Windowed and padded dataset:

   - signal: same as case 2, but with hann window and padding
   - expectation: dominant-frequency bin stays stable and phase rotation still
     increases the dominant real-component amplitude compared with the unrotated
     transform

4. Lifetime-corrected decaying oscillation:

   - signal: `exp(-t/tau_mu) * (1 + A * cos(2 * pi * f * t))`
   - expectation: musrfit-style lifetime correction flattens the early/late
     oscillation envelope before the FFT stage

5. Detector quadrature reconstruction:

   - signals: detector residuals proportional to `cos`, `sin`, `-cos`, `-sin`
   - expectation: a Mantid-style phase table reconstructs non-zero real and
     imaginary quadratures even when the naive grouped detector sum is zero

## Reference Comparison Targets

- WiMDA grouped FFT with manual phase correction on a simple oscillatory run
- musrfit `musrFT --fourier-option real --phase <val>` on the same or similar
  oscillatory input
- Mantid frequency-domain analysis with manual shift or a simple phase-table
  entry where available

These reference comparisons are follow-on validation targets. The first code
slice will use synthetic datasets inside pytest because they are deterministic
and cheap to run in the harness.

The synthetic comparison contracts now live in
`tests/test_fourier_reference_methods.py`.

## Data Contracts To Preserve

- frequency axis remains in MHz because input time is in microseconds
- existing `fft_asymmetry` return shape remains backward compatible
- new complex-spectrum surface must preserve windowing, padding, and time-range
  cropping behavior