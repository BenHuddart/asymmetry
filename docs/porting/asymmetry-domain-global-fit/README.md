# Asymmetry-domain global (shared-parameter) fit

**Status:** study → implemented (Option A)
**References studied:** in-repo `asymmetry.core.fitting` (no external reference
program — this is an *ergonomics/discoverability* port of an existing internal
capability, not a port from WiMDA/musrfit/Mantid).

## The gap

A clean-room API-testing pass against the WiMDA Muon School corpus found that
Asymmetry can share fit parameters across runs **only in the count domain**:

- `fit_grouped_series(relationship="global", …)` and
  `fit_grouped_time_domain(…)` share physics parameters across runs/groups, but
  they operate on *lifetime-corrected grouped counts* (Cash/Poisson cost) built
  via `build_grouped_time_domain_groups` / `build_grouped_time_domain_datasets`
  into `GroupedTimeDomainGroup` objects.

There was **no documented, discoverable way to share parameters across several
`.asymmetry` traces in a single least-squares call**. The user-expected
asymmetry-domain workflow — e.g. a global Keren fit sharing Δ and ν across
several LF fields at each temperature (the Al-LLZ ionic-motion corpus, target
E_a = 0.19(1) eV) — could not drive a global fit without dropping into the
count-domain machinery, which is high-friction and undiscoverable from the docs.

## The key finding (entry points / data flow)

`FitEngine.global_fit` (`src/asymmetry/core/fitting/engine.py:499`) **already is**
an asymmetry-domain simultaneous least-squares fitter:

- it concatenates `ds.time` / `ds.asymmetry` / `ds.error` across datasets,
- builds an iminuit `LeastSquares` cost by default (`cost_factory=None`),
- shares the named `global_params` across all datasets while giving each dataset
  its own copy of every `local_params` entry (internally `f"{name}_{run}"`),
- returns per-dataset `FitResult`s plus the shared-global `ParameterSet`.

The count-domain `fit_grouped_*` family is a *wrapper* around this method: it
builds temporary `MuonDataset`s whose `.asymmetry` field holds (raw or
lifetime-corrected) counts and passes a `POISSON_COST` / Cash factory. In other
words **the asymmetry-domain global fit already exists inside the engine** — what
was missing is a first-class, discoverable entry point in the `fit_*` family that:

1. takes asymmetry datasets directly (no count machinery, no `GroupedTimeDomainGroup`);
2. does not require the caller's datasets to carry **unique `run_number`s** —
   `MuonDataset.run_number` falls back to `metadata["run_number"]` defaulting to
   `0`, so several runless asymmetry datasets silently collide on key `0` and
   `global_fit` raises "duplicate dataset run numbers". This is the single
   sharpest discoverability trap.
3. returns a friendly result bundle (shared globals + uncertainties, per-dataset
   locals + uncertainties, and a **combined** reduced χ²), instead of a bare
   `tuple[dict, ParameterSet]`.

## Decision

Add a thin, GUI-free convenience function **`fit_global`** (+ `GlobalFitResult`)
in `src/asymmetry/core/fitting/asymmetry_global.py` that **wraps**
`FitEngine.global_fit`. It does **not** reimplement the minimiser, the
concatenation, or the cost — it reuses the engine. See
[`implementation-options.md`](implementation-options.md) for the chosen shape and
[`comparison.md`](comparison.md) for the asymmetry-vs-count-domain trade-off.

This complements rather than duplicates `fit_grouped_series`: the new function is
the asymmetry-domain least-squares sibling of the count-domain Poisson/Cash one.

## Relationship to the global-fit wizard

`build_global_fit_wizard_recommendation(…)` and friends *recommend* a global/local
parameter partition (which physics to share across a series); they do not execute
a fit. Their recommended `global_params` / `local_params` lists feed straight into
`fit_global`. The wizard is left untouched — recommend vs. execute stays separated.
