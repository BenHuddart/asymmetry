# Rotating Reference Frame (RRF) transform

**Status:** candidate.

## What

A demodulation transform that rotates the TF μSR signal into a
reference frame co-rotating with the applied field at γ_μ B_app.
Output is the slowly-varying envelope around B_app, with the rapid
Larmor oscillation subtracted off. The transform makes very-high-TF
and vortex-lattice signals tractable in the time domain — without it,
typical TF = 200 mT data oscillates at 27 MHz and the GUI's 8 μs
default window contains ~200 cycles.

## Why

- Asymmetry's current TF visualisation strategy at high field is to
  switch the central plot to the Frequency domain and compute the
  FFT. That works for inspection but breaks down for *fitting*: the
  fit panel is time-domain, and ~10⁴ oscillation samples per dataset
  is wasteful when only the envelope carries information.
- RRF is the standard workaround used at every facility for
  vortex-lattice studies (Sonier RMP 72, 769, 2000).
- Asymmetry's vortex-lattice screenshot scenario
  (`docs/screenshots/data/archetypes.py::make_ybco_vortex_lattice`)
  already synthesises a signal at 27 MHz that would benefit from
  RRF demodulation for visual inspection.

## Prior art

- **Mantid `RRFMuon`** — standalone algorithm:
  `Framework/Muon/src/RRFMuon.cpp`. Takes input workspace + reference
  frequency, outputs the in-phase and quadrature envelopes.

- **musrfit `PRunAsymmetryRRF`** — RRF embedded as an alternative
  asymmetry run type. Selected via the RUN block; output is the
  RRF-demodulated asymmetry rather than the bare TF signal.

- **WiMDA:** ❌

## Why this is roadmap-tractable

- The algorithm is short (~30 lines of numpy): multiply A(t) by
  `exp(-i ω_ref t)`, low-pass filter to remove the 2·ω_ref aliasing
  band, return real (in-phase) and imaginary (quadrature) components.
- API addition to `core/transform/`: e.g.
  `rrf(dataset, reference_freq_mhz, *, lowpass_us=None) → tuple[MuonDataset, MuonDataset]`
  for in-phase and quadrature.
- GUI exposure: domain button or "Apply RRF" toggle in the plot panel
  toolbar; reference frequency derived from dataset field automatically.

## Out of scope for this candidate

- Per-detector / per-group RRF (Mantid supports this; defer to
  follow-up).
- Automatic reference-frequency selection (assume field metadata
  provides it).
