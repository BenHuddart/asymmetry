# Time-integral asymmetry / field-scan observable — porting study

Status: **study** (no implementation yet).

## Feature

The *integral-counting* method reduces a whole muon time spectrum to a **single
number per run** — the asymmetry integrated over a time window — and then plots
that number against a run variable (almost always magnetic field) across a
**series of runs**. The resulting curve is the basic observable for:

- **ALC** — Avoided Level Crossing resonance spectroscopy (LF field scan;
  resonant dips in integral asymmetry / polarisation where energy levels cross).
- **Repolarisation** — LF decoupling curves (integral polarisation recovers as
  the longitudinal field is raised).
- **QLCR** — Quadrupolar Level Crossing Resonance (an ALC variant for
  quadrupolar nuclei).

This is fundamentally a **two-level** feature:

1. **Per-run reduction**: counts (or asymmetry) over `[t_min, t_max]` → one
   `(value, error)` scalar. No fit required.
2. **Series assembly**: collect the scalars across many runs and order them by an
   independent variable (field / temperature / run) to form the scan curve.

It was flagged as a gap by the WiMDA Muon School corpus testing: Asymmetry can
fit time-differential spectra and trend *fitted* parameters across a series, but
has no way to produce the integral observable directly, which **unblocks
ALC / repolarisation / QLCR** workflows.

## Reference programs

- **WiMDA** — the **ALC mode**. Count-based integral `A = (F − B)/(F + B)`
  summed over the good-bin window per run, plotted vs field; includes a
  differential `dA/dB` transform. No alpha balance in the ALC path.
- **Mantid** — `PlotAsymmetryByLogValue` (the most complete reference) feeding
  the three-step **Muon ALC interface** (load → baseline → peak fit). Two
  reduction types — *Integral* (integrate counts, then form asymmetry, alpha
  applied) and *Differential* (form asymmetry per bin, then integrate) — over a
  single `[TimeMin, TimeMax]`, with red/green period combination and a sample-log
  x-axis (`LogValue` + `Function`).
- **musrfit** — **no native integral observable** (important negative result).
  ALC is only an instrument tag; data is fitted as ordinary time-differential
  asymmetry (fittype 2). `mupp` plots *fitted* parameters vs an independent
  variable — the closest analog, but it never integrates counts.

See [comparison.md](comparison.md) for the cross-program comparison and
[implementation-options.md](implementation-options.md) for the recommended port.

## Recommendation (summary)

Port **Mantid's `PlotAsymmetryByLogValue` model as the behavioural contract**
(alpha-aware, *Integral* + *Differential* types, single time window, red/green
support, sample-log x-axis), with **WiMDA's count-integral as the default
*Integral* formula** and its `dA/dB` differentiation as an optional series
transform. **Reject** musrfit's parameter-only approach for the core observable
(it does not integrate). Realise it as:

- a Qt-free core transform `integrate_asymmetry(...)` (per-run scalar), reusing
  the existing `compute_asymmetry` kernel and the good-bin / period machinery;
- a **field-scan series** that assembles the scalars across runs, reusing the
  existing `FitSeries` ordering (`order_key ∈ {field, temperature, run}`) and the
  trend-plotting surface (`fit_parameters_panel.py`);
- ALC baseline-subtraction + peak-fitting deferred to a follow-up (it depends on
  this observable existing first; see the `alc-avoided-level-crossing` candidate).

See [verification-plan.md](verification-plan.md) and [test-data.md](test-data.md)
for how the port will be validated against the reference programs and corpus.
