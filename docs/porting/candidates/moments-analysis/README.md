# Moments analysis

**Status:** candidate — **superseded by the implementation-grade study
[`spectral-moments`](../../spectral-moments/README.md)** (2026-06-12). This
candidate is kept for provenance; new work happens under the `spectral-moments`
slug.

## What

Compute the first few statistical moments (m₀, m₁, m₂) of a
frequency-domain spectrum and derive linewidth-style summaries (α and
β widths, RMS, peak position via parabolic extrapolation). Tool for
quickly characterising lineshapes without fitting a parametric model.

## Why

- Lightweight characterisation that complements full Gaussian /
  Lorentzian fits — particularly useful when the lineshape is not
  well-described by a textbook form.
- WiMDA ships this and users have come to expect it on inspection
  panels.

## Prior art

- **WiMDA `Moments.pas`** — form-driven moments computation with
  cutoff and x-range controls; supports per-run averaging.
- **musrfit, Mantid, Asymmetry:** ❌.
