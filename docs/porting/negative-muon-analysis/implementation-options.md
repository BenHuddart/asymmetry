# Implementation options & chosen approach

The full, prescriptive, phased plan is in **[`plan.md`](plan.md)**. This file
records the design space and why each choice was made, including the reuse-audit
verdicts and the two "almost-but-not-quite fits" trade-offs.

## Chosen approach (summary)

A new **`core/negmu/`** package (Qt-free, scriptable, unregistered):

| Module                     | Responsibility                                              | Phase |
|----------------------------|-------------------------------------------------------------|-------|
| `core/negmu/lifetimes.py`  | Element lifetime table (literature-anchored) + accessors    | 1 |
| `core/negmu/model.py`      | Multi-exponential raw-count model builder                   | 1 |
| `core/negmu/fit.py`        | Single-group fit; `(time,counts)` and `(dataset,group)` entries | 1 |
| `core/negmu/fit.py` (+)    | α-coupled forward/backward simultaneous fit                 | 2 |
| `core/negmu/ratio.py`      | Capture-ratio report (derived quantities)                   | 2 |
| `core/negmu/background.py` | Set-as-BG component subtraction                             | 3 |
| `core/negmu/polarisation.py` | Optional μ⁻SR polarisation multiplier (None/LorGau/Diamag) | 4 |
| `core/simulate.py` (+)     | `simulate_capture_run` (reuses `_sample_and_build_run`)     | 1 |
| docs + API autodoc + toctree | experimental user-guide page                              | 5 (threaded) |

Fitting reuses the shared engine (`drive_minuit`, `FitResult`,
`Parameter`/`ParameterSet`); only the Cash/Gaussian count cost is local.
Lifetimes are **fixed at the table value by default**, any τ freeable.

## Options considered

### A. Where does the multi-exponential model live?

1. **Reuse `count_domain` / `FitEngine` directly.** Rejected — proven
   non-fit (`comparison.md` §3): `count_domain` is single-envelope; `FitEngine`
   is Gaussian-on-asymmetry. Neither expresses Σ_i N_i e^{−t/τ_i} on Poisson
   counts.
2. **Composite of `ExponentialRelaxation` components fitted via `FitEngine`.**
   The functional form matches (`Λ_i = 1/τ_i`, fix to pin), but the statistics
   are wrong (Gaussian √N on counts that span orders of magnitude; the Cash
   improvement of `count_domain` is exactly why this matters). Also forces the
   counts into a percent-asymmetry `MuonDataset` and would surface the
   components in the GUI if registered. Rejected for the fit; the *form* insight
   is acknowledged.
3. **New `core/negmu/` package with a dedicated model + small count fitter
   reusing the shared engine.** **Chosen** (Ben's decision). Cleanly bounded,
   correct statistics, no GUI exposure, disjoint from Wave B.

### B. Forward/backward handling (Ben: "also add α-coupled F+B fit")

- **Per-group single-histogram fits** (Phase 1) — fit one detector group's raw
  counts; the ratio report combines two `FitResult`s. Always available.
- **α-coupled F+B simultaneous fit** (Phase 2) — shared per-element amplitudes
  `N_i` and shared `τ_i`, with the detector balance split `N_F = √α·Σ…`,
  `N_B = (1/√α)·Σ…` (mirroring `build_fb_count_model`), separate backgrounds.
  *Both* delivered. Divergence from WiMDA: WiMDA fits **independent** per-side
  amplitudes (`NF`,`NB`); the coupled fit **shares** `N_i` (capture populations
  are isotropic for the lifetime method), so ratios are identical per side by
  construction — the independent per-group fits remain available when a genuine
  F/B asymmetry is wanted. Documented in the plan.

### C. Lifetime default — fixed vs free (Ben: fixed)

τ fixed at the table value by default; `spec.free_tau` frees a chosen subset.
Matches elemental identification and avoids the ill-conditioning of several
near-degenerate free lifetimes at realistic counts.

## Reuse audit / adapt-vs-new trade-offs (the two near-misses)

Per the binding reuse requirement, the full per-work-package audit (existing
functions + import paths + one-line justification per new module) is in
[`plan.md`](plan.md) §"Reuse audit". Two cases where existing machinery
**almost** fits and the verdict required judgement:

1. **The count cost (Cash/Gaussian).** `count_domain._poisson_cash` /
   `_gaussian_chi2` are exactly the costs needed — but they are **private** and
   `count_domain` is **off-limits to modify** (scope). Re-exporting them would
   be a modification; importing a private name is fragile. **Verdict:**
   replicate the ~6-line Cash statistic and the √N Gaussian in
   `core/negmu/fit.py`, citing the source. Everything else (Minuit construction,
   limits, `drive_minuit`, `FitResult` packing) is reused. This is the minimum
   honest duplication; if the implementer judges a shared `core/fitting/
   count_cost.py` extraction is cleaner, **STOP and ask Ben** (it would touch
   `count_domain`, which is out of scope here).
2. **Synthetic data.** `core/simulate` bakes in a single `exp(−t/τ_μ)` envelope
   (it is an *asymmetry-imprinting* generator) and **cannot** natively produce a
   multi-τ capture histogram. Writing an inline Poisson draw in the tests would
   be a "bespoke generator" the reuse rule forbids. **Verdict** (Ben): add a new
   public `simulate_capture_run` to `core/simulate.py` that builds the multi-exp
   expectation directly and **reuses** the existing seeded-Poisson sampler +
   Run/provenance assembly (`_sample_and_build_run`). Additive change to a shared
   file; produces a real `Run` so the `(dataset, group)` fit path is exercised
   end-to-end. See [`test-data.md`](test-data.md).

## Non-goals

No GUI of any kind; no picker registration; no real-data validation claims; no
new fitter engine, parameter machinery, or results framework; μ-XRF / muonic
X-ray elemental analysis out of scope (lifetime method only). Full list in
[`plan.md`](plan.md) §"Non-goals".
