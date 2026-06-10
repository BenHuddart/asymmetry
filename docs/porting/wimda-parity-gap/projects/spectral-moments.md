# Project brief: spectral-moments

Umbrella: `wimda-parity-gap` · Wave B (after `maxent-completion`) ·
Size S–M · promotes the `moments-analysis` candidate

## Motivation

Moments are the main *quantitative* consumer of the MaxEnt field spectrum:
B_rms → penetration depth, skewness/β → vortex-lattice structure. Asymmetry
now has the spectrum but no way to reduce it to these numbers. WiMDA-only
feature; small, pure-core; high value for superconductivity users.

## WiMDA reference

`Moments.pas:152–423`: over a user x-range with an amplitude cutoff (% of
peak): B_pk (5-point parabolic peak extrapolation), B_ave, ⟨B_ave−B_pk⟩,
B_rms about mean and about peak, skewness α = ∛|m₃|/√m₂, lineshape asymmetry
β = (B_ave−B_pk)/B_rms,pk; run-to-run averaging accumulators with errors;
one-click export rows (run, field, T, moments) into the fit table for trend
fitting. Listed as a follow-on slice in `docs/porting/maxent/comparison.md`.

## Scope

- New Qt-free `core/fourier/moments.py`: `spectrum_moments(freq_or_field,
  amplitude, *, x_range, cutoff_fraction)` → dataclass of the WiMDA moment
  set; works on any spectrum (MaxEnt or FFT), in field or frequency units.
- Uncertainties: propagate from spectrum errors where available; else
  bootstrap over noise realisations (study decides — WiMDA's run-averaging
  errors are scatter-based, which the trend layer already handles).
- GUI: a compact readout in/next to the MaxEnt panel when a spectrum is
  active (range + cutoff controls reuse draggable handles); "Send to trend"
  action that records moments per run into the results/trending machinery so
  B_rms(T) etc. fit like any parameter series (the modern equivalent of
  WiMDA's fit-table export).
- Persist moment-extraction settings in the representation recipe.

**Out**: nothing significant — this is deliberately a small, complete
project.

## Current Asymmetry state

Absent; `core/fourier/` and `core/maxent/` provide the spectra and error
estimates to consume.

## Physics-correctness notes

Moments are cutoff- and range-sensitive — show the integration window on the
plot so the choice is visible, and record it in provenance. The parabolic
peak extrapolation is fine; document that B_pk on a noisy spectrum is the
fragile member of the set.

## Conflicts & dependencies

Primary surfaces: new core module + a contained `maxent_panel.py` /plot hook
— sequenced after `maxent-completion` (same panel file). Trend integration
uses existing series machinery read-mostly.

## Verification sketch

Closed-form synthetic distributions (Gaussian: skewness 0, known B_rms;
skewed two-Gaussian mixture: all moments analytic) through the full
spectrum→moments path; WiMDA `Moments.pas` arithmetic as behavioural oracle
on one shared spectrum; vortex-lattice-like asymmetric lineshape sanity check
(β sign convention matches WiMDA/literature).
