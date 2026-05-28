# Muonium-radical hyperfine analysis

**Status:** candidate. Surfaced by the practical-workflow
catalogue (workflow #7) during the documentation pass.

## What

Specialised models and workflow support for analysing muoniated
radicals and muonium hyperfine sub-levels in chemistry / semiconductor
samples. Includes:

- TF muonium pair functions (`HighTFMuonium`, `LowTFMuonium`,
  `TFMuonium`).
- ZF muonium decay (`ZFMuonium`).
- Muonium-decoupling-curve parametric model for field-scan
  experiments.
- Diagnostic "is this a hyperfine pair or two precessing sites?"
  helper using the FFT spacing.

## Why

- Asymmetry's existing F-μ-F components handle two-spin nuclear
  hyperfine but not the electron-coupled muonium case.
- Mantid ships all four `*Muonium*` fit functions; musrfit
  reconstructs them via composite Bessel functions. Asymmetry
  cannot fit muonium signals today.
- The Amato-Morenzoni textbook Ch 7 places muonium spectroscopy in
  semiconductors as one of the central μSR applications.

## Prior art

- **Mantid**: `HighTFMuonium`, `LowTFMuonium`, `TFMuonium`,
  `ZFMuonium`, `MuoniumDecouplingCurve` in
  `Framework/CurveFitting/src/Functions/`.
- **musrfit**: composed via `PTheory` building blocks; not exposed
  as dedicated functions.
- **WiMDA**: ❌.

## Roadmap position

This is an umbrella candidate that depends on
[`theory-library-expansion`](../theory-library-expansion/)
landing first. Recommend ranking after the theory-library
expansion ships, with a target tier of **Next** to **Now**
depending on perceived demand from chemistry / semiconductor
users.
