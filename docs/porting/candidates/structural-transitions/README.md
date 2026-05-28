# Structural / non-magnetic phase transitions

**Status:** candidate. Surfaced by practical-workflow #14.

## What

Tooling and documentation for detecting and characterising
non-magnetic phase transitions in μSR data — typically a kink or
discontinuity in fitted parameters (λ, ν, A\ :sub:`0`) at the
transition temperature, sometimes with a phase-coexistence
fraction split.

## Why

- Many physically interesting transitions (structural, charge-
  density-wave, orbital-order) leave subtle μSR signatures that
  are easy to miss without dedicated trend-analysis tools.
- Mantid, musrfit, and WiMDA all leave this to manual visual
  inspection of the parameter-trending table.

## Roadmap position

This candidate is mostly a **trend-analysis toolkit** rather than
a new fit function. Useful but lower impact than the algorithmic
candidates. Defer to Later tier.

## Out of scope

- Quantitative kink-detection algorithms — leave to follow-up.
- Volume-fraction analysis (two-component composites are already
  supported via `composite_models`).
