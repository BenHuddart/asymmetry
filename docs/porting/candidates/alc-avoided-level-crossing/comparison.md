# ALC: comparison

Only Mantid has a working ALC interface. The implementation comparison
is therefore "what Mantid does" vs "what Asymmetry could do".

## Mantid ALC workflow

1. **Data loading tab**: select a run series spanning a field scan.
2. **Baseline modelling tab**: fit a polynomial (or spline) to the
   integrated asymmetry vs field; user marks "exclude" regions where
   resonances sit.
3. **Peak fitting tab**: subtract the baseline; fit Lorentzian or
   Gaussian peaks to the residual; output resonance positions, FWHMs,
   amplitudes, integrated areas.
4. **Results**: TableWorkspace export, plot to ADS.

## Proposed Asymmetry adaptation

Build as a new top-level window analogous to the existing
`GlobalParameterFitWindow`. Reuse:

- `DataBrowserPanel` for the run-series selection (drag-select a
  filter on the field column).
- The existing parametric-model fit machinery for the baseline
  polynomial (already supports polynomial models).
- A new lightweight peak-fitting widget (Lorentzian / Gaussian)
  that consumes the baseline residual.

## What needs to land before ALC

- Robust per-run integrated-asymmetry extraction across a field
  series (the parameter-trending panel does most of this).
- A configurable plot widget with "exclude region" gestures.
