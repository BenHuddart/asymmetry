# Frequency-Domain Fitting Test Data

Use synthetic spectra for deterministic tests:

- Gaussian peak plus constant background.
- Lorentzian peak plus linear background.
- Three-run Gaussian series with known centre shifts and FWHM changes.
- Cached Fourier-spectrum project payloads built from small arrays rather than
  raw-file fixtures.

Required tolerances:

- Peak centres recover within one frequency-bin width for noisy spectra and
  tighter than that for exact synthetic curves.
- FWHM recovers within 10 percent for noisy spectra.
- Derived ``B0`` and ``Bwid`` match MHz-to-G conversion using the muon
  gyromagnetic ratio.

GUI tests may construct ``MuonDataset`` frequency spectra directly with
``metadata["plot_domain"] = "frequency"`` and inject them into
``_frequency_spectra_by_run`` to avoid recomputing FFTs in every test.
