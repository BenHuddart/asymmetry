# Frequency-domain finishers — test data

Data strategy (Ben, 2026-06-11): **synthetic-first where needed.** Pulse
compensation and Burg *must* be synthetic (they need known, controlled
frequency content). For field-axis, exclusions and diamag-removal, use a real
corpus run where one is readily identifiable, else synthetic with documented
parameters. Tests stay hermetic: synthetic fixtures are generated in-test via
`core/simulate`, real-corpus checks are opt-in/smoke-gated on data presence.

## Synthetic generator

`core/simulate.py` provides `build_builtin_template("ideal_pulsed_fb")` +
`simulate_run(template, group_signals=..., seed=...)`. A group signal is a
callable `t(µs) → fractional asymmetry`. Known-frequency fixtures:

```python
sig = lambda t: 0.15 * np.cos(2*np.pi * f_mhz * t + phi) * np.exp(-t/T2)
run = simulate_run(build_builtin_template("ideal_pulsed_fb"),
                   group_signals={fwd: sig, bwd: lambda t: -sig(t)}, seed=1)
```

Pulse broadening for the compensation fixture: convolve the asymmetry with the
ISIS arrival kernel implied by `core/maxent/pulse.py` (parabolic half-width `w`,
pion pole) — equivalently, build the time signal whose per-frequency amplitude
is multiplied by `R(ν)` from `pulse_amplitude_phase`, so that dividing by `R(ν)`
recovers the flat-amplitude input. This keeps the forward and inverse models on
the *same* pulse lineshape, which is the property under test.

## Fixtures by feature

| Feature | Fixture | Known truth |
|---|---|---|
| Field axis (γ_μ·B) | synthetic TF run, `f = γ_μ·B`, B set in metadata | peak at B (Gauss) within one bin |
| FFT ≡ MaxEnt in field units | same run, FFT and MaxEnt | peak field agrees within one bin |
| Exclusions | synthetic spectrum with a line at f_excl | bins inside (centre±width) are zero; line elsewhere survives |
| PSI RF-harmonics preset | synthetic spectrum with spikes at 50.63×{1..5}+DC | spikes removed after preset; signal line untouched |
| Diamag exclusion slot | synthetic run, B in metadata | slot centre = γ_μ·B; that line excluded |
| Pulse compensation | pulse-broadened synthetic, flat input amplitude vs f | amplitude flat (within tol) after ÷R(ν); uncompensated rolloff documented |
| Compensation guard | broadened synthetic with content past the first node of G | gain capped/cut; no overflow/NaN |
| Baseline (σ-clip) | synthetic spectrum: flat offset + sparse peaks + noise | offset removed to <1 noise-σ; peaks preserved; converges ≤ cap |
| Baseline ≡ WiMDA at 1 iteration | same fixture, iterations=1 | matches single-pass 2σ-clip mean |
| S/N at peak | synthetic line + known noise | peak S/N ≈ analytic; mean error finite, non-zero |
| Real+Imag view | phased synthetic line | real channel peaks, imag near-zero at correct phase |
| Burg doublet | two close lines, short window | **characterisation test** — see verification-plan |
| Burg FPE | synthetic with M known lines | FPE-optimal pole count tracks line count; boundary warning fires |
| Diamag fit-and-subtract | synthetic damped cosine at known f/field | fitted field ≈ truth; residual spectrum flat at that line |

## Real-corpus opportunistic checks (smoke-gated)

The WiMDA Muon School corpus (`~/Documents`, HDF5 `.nxs` + PSI `.bin/.mdu`;
loadable subset only — see memory `hdf4_loader_gap`) supplies optional parity
smoke tests, skipped when data is absent:

- a known-field TF run → field-axis peak at γ_μ·B (sanity vs metadata);
- a PSI run carrying RF harmonics → before/after the harmonics preset;
- a TF run with a clear diamagnetic line → fit-and-subtract reports the
  applied field.

These mirror the `maxent_corpus_smoke` pattern: `pytest.importorskip`-style
guards on file presence, never required for CI green.

## Project round-trip

The FFT recipe gains keys (exclusions, baseline mode, pulse-comp params); the
existing round-trip tests in `tests/test_project_schema.py`
(`test_fourier_recipe_applied_even_without_freq_fit`) must continue to pass and
a new case must confirm the new keys survive save→load and recompute an
identical spectrum.
