# Test data

No external corpus is used in CI (the WiMDA Muon School / Al-LLZ data that
motivated the gap is not vendored). The behaviour is validated on **synthetic**
asymmetry datasets with injected, known ground truth, so the test is
deterministic and dependency-free.

## Synthetic generator (in `tests/test_asymmetry_global_fit.py`)

A shared physics parameter and a per-dataset local parameter, exponential model:

```
A_d(t) = amp_d · exp(−λ · t)
```

- **Global:** `lambda` (shared relaxation rate), injected as one true value.
- **Local:** `amp` (per-dataset amplitude), a different true value per dataset.
- Several datasets (≥ 2) on a common time grid, each with Gaussian noise at a
  fixed per-point `σ` and a deterministic RNG seed.

This mirrors the motivating Keren case (a rate shared across fields, amplitude
free per field) while staying analytic and fast.

## Assertions

1. **Recovery** — the global fit recovers the injected `lambda` within tolerance
   and each dataset's `amp` within tolerance.
2. **Tighter constraint** — σ(`lambda`) from the global fit is smaller than
   σ(`lambda`) from independent single-dataset fits (pooling data constrains the
   shared parameter better). This is the core scientific justification.
3. **Single dataset** — `fit_global` with one dataset reproduces the ordinary
   `FitEngine.fit` reduced χ² and parameter values.
4. **Edge cases** — mismatched parameter names raise a clear error;
   global/local overlap raises; fixed and bounded parameters are respected;
   non-finite/zero errors are rejected at the boundary.
