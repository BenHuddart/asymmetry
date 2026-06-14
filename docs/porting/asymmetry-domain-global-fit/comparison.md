# Comparison: asymmetry-domain LSQ vs count-domain Poisson global fit

| Aspect | Asymmetry-domain global fit (new `fit_global`) | Count-domain global fit (`fit_grouped_series` / `fit_grouped_time_domain`) |
| --- | --- | --- |
| Signal fitted | `MuonDataset.asymmetry` (already α-corrected, background-subtracted) with per-point `MuonDataset.error` | Lifetime-corrected grouped detector counts built from histograms |
| Statistic | Gaussian weighted least squares, Σ_d Σ_i ((A_i − μ_i)/σ_i)² | Cash/Poisson (`POISSON_COST`) on raw counts (default), or √N Gaussian |
| Model | A normalized asymmetry/polarization model `f(t, **params)` | A normalized polarization wrapped by the per-group `N0·(1+amp·P)+bg` count model |
| Per-dataset nuisances | Whatever the model exposes (e.g. amplitude `A`, baseline) — free choice | Reserved nuisance block (`N0`, `background`, `amplitude`, `relative_phase`) |
| Input object | Plain asymmetry `MuonDataset`s (no histograms needed) | `GroupedTimeDomainGroup`s built from a run with detector histograms + grouping |
| Low-count fidelity | Biased when counts are low (√N-Gaussian weight under-weights low bins) | Statistically faithful (Cash removes the low-count bias) |
| Discoverability | High — one call on the traces the user already has | Low — requires histograms, grouping, the nuisance contract, count builders |

## When to use which

- **Use `fit_global` (asymmetry-domain LSQ)** for the standard convenience
  workflow: you already have asymmetry traces (`.time`, `.asymmetry`, `.error`)
  and want to share a physics parameter (a rate, a frequency, a field width)
  across several of them in one least-squares call — e.g. a global Keren fit
  sharing Δ and ν across LF fields, then extracting an activation energy from the
  shared rate's temperature dependence. This is what users reach for first and it
  matches the per-point asymmetry errors the rest of the asymmetry-domain UI uses.

- **Prefer the count-domain Poisson path** when counts are low enough that the
  Gaussian √N weighting biases the fit (late-time bins, weak signals, short
  counting). There the Cash statistic on raw Poisson counts is the
  statistically-correct objective, at the cost of needing the detector
  histograms, grouping, and the nuisance-block contract.

Both share **one** minimiser seam: `FitEngine.global_fit` concatenates the
datasets and drives iminuit. The only difference is which array goes into
`.asymmetry` (asymmetry vs counts) and which `CostFactory` is used (Gaussian vs
Cash). `fit_global` is therefore not a second implementation — it is the
asymmetry-domain, Gaussian-cost façade of the same engine call the count-domain
family already uses.

## Caveat recorded for review

The asymmetry-domain LSQ result is only as good as the per-point `σ_A` carried on
the dataset. Asymmetry errors are propagated from the count errors at asymmetry
build time (see the `asymmetry-error-propagation` study); for low counts that
propagation is the same Gaussian approximation Cash is designed to avoid. The
docs state this trade-off plainly so users do not read "convenience" as
"statistically equivalent".
