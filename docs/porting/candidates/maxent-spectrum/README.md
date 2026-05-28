# MaxEnt frequency reconstruction

**Status:** candidate (Asymmetry has a stub at
`src/asymmetry/core/fourier/maxent.py`; the algorithm is unimplemented).

## What

Production MaxEnt (maximum-entropy) frequency-spectrum reconstruction.
The classic μSR application is super-resolved FFT lineshape recovery —
sharper than apodised FFT, robust to short time windows, with
self-consistent phase optimisation. Two reference implementations bound
the design space: WiMDA's Burg-method pole-scan and Mantid's iterative
entropy maximisation with phase refinement.

## Why

- Asymmetry's frequency-domain panel is currently FFT-only with manual
  apodisation. For users analysing very short pulsed runs or trying
  to resolve closely-spaced precession peaks (e.g. Cu(pyz)₂(ClO₄)₂
  three-frequency AFM), MaxEnt is materially better.
- Both WiMDA and Mantid ship working MaxEnt — Asymmetry is the only
  one of the four with a stub.
- The Fourier user-guide page currently has to qualify "we recommend
  apodised FFT" because MaxEnt isn't an option.

## Prior art

- **WiMDA `MaxEnt.pas`** — Burg autocorrelation
  estimation (`memcof`), final-prediction-error (FPE) pole-count
  optimisation, time-domain reconstruction via inverse cosine
  transform. Mathematically self-contained; no external entropy
  iteration. Faster but less self-consistent.

- **Mantid `MuonMaxent`** — iterative entropy-maximisation
  algorithm with per-detector phase refinement. Slower but
  produces phase-consistent group-resolved spectra. Includes
  uncertainty estimates on the reconstructed spectrum.

- **musrfit:** limited MaxEnt support; not a flagship feature.

## Why this is roadmap-tractable, with caveats

- The stub at `core/fourier/maxent.py` already defines the API
  signature (`maxent(dataset, n_freq=512, f_max=None) → (freqs, spectrum)`).
- The infrastructure to call MaxEnt from the GUI exists
  (Fourier panel + frequency plot panel).
- Reference implementations are available in C++/Pascal — porting
  to numpy is mechanical for the Burg path, more involved for the
  Mantid iterative path.

Caveats:

- MaxEnt is the algorithmically heaviest item in this roadmap. Burg
  is ~200 lines of numpy; Mantid's iterative path is ~600.
- Validation requires a regression oracle — generate canonical
  synthetic test cases and check against Mantid output.

## Out of scope for this candidate

- Group-resolved MaxEnt with phase optimisation (Mantid-style). Start
  with single-channel reconstruction; extend to grouped later.
- Uncertainty estimation on the reconstructed spectrum.
- Real-time MaxEnt during interactive frequency-domain scrolling.
