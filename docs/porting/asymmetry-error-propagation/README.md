# Asymmetry error propagation — exact Poisson vs Mantid independent-propagation

## What this study is about

`compute_asymmetry` in
[`src/asymmetry/core/transform/asymmetry.py`](../../../src/asymmetry/core/transform/asymmetry.py)
computes the statistical error on the forward–backward asymmetry

```
A(t) = (F − αB) / (F + αB)
```

as

```
σ_A = sqrt( (F + α²B) · (1 + (N/D)²) ) / |D|      with N = F − αB, D = F + αB.
```

This is the **Mantid `AsymmetryCalc` error model**: it propagates the numerator
`N` and denominator `D` as if they were **independent**. They are not — both are
built from the same two Poisson counts `F`, `B`, so `cov(N, D) = F − α²B ≠ 0`.
Dropping that covariance **over-estimates** `σ_A`.

The exact Poisson propagation (var `F` = `F`, var `B` = `B`, keeping the
correlation) is

```
var(A) = 4 α² F B (F + B) / (F + αB)⁴          →   (1 − A²)/(F + B)   at α = 1.
```

The shipped formula gives `(1 + A²)/(F + B)` at α = 1. The ratio of the two is

```
σ_shipped² / σ_exact²  =  (1 + A²)/(1 − A²)     (at α = 1)
```

— exact at `A = 0`, ≈ 9 % in variance (≈ 4.5 % in σ) at `A = 0.21`, and growing
without bound as `|A| → 1`.

## How this surfaced

The simulate-mode verification suite
([`docs/porting/simulate-mode/`](../simulate-mode/), verification §2). Refitting
synthetic runs against a **known truth** centres reduced χ² on
`E[(1 − A²)/(1 + A²)] < 1` instead of 1, and parameter pull distributions come
out too narrow. See
[`tests/test_nexus_writer.py::TestRefitRecovery`](../../../tests/test_nexus_writer.py),
which currently documents and centres its acceptance band on that biased
expectation. The simulate-mode "as-implemented" notes flagged this as a
follow-on investigation; this study is that investigation.

## Why it matters beyond synthetic data

The error array returned by `compute_asymmetry` becomes the per-point σ in the
iminuit `LeastSquares` cost
([`src/asymmetry/core/fitting/engine.py`](../../../src/asymmetry/core/fitting/engine.py)),
so it sets the χ² weights **and** the fitted-parameter covariance for **every
fit in the program** — not just simulate-mode. Over-estimated σ_A means:

- reduced χ² biased low (looks like a "better" fit than the data supports);
- fitted parameter uncertainties inflated, most for parameters tied to
  high-asymmetry regions (amplitudes, baselines).

Measured impact (controlled known-truth fit, A₀ = 0.22 exponential, 300 seeds):
amplitude σ inflated **+3.2 %**, decay-rate σ **+1.4 %**, ⟨χ²ᵣ⟩ 0.987 vs 0.999,
amplitude pull SD 0.974 (too narrow) vs 1.005. Full numbers in
[`test-data.md`](test-data.md).

## Entry points and data flow

`compute_asymmetry(forward, backward, alpha) → (asymmetry, error)`

Production callers whose `error` reaches a fit:

| Caller | File:line | Role |
| --- | --- | --- |
| NeXus loader | [`core/io/nexus.py:503`](../../../src/asymmetry/core/io/nexus.py) | dataset error → fit σ |
| PSI loader | [`core/io/psi.py:882`](../../../src/asymmetry/core/io/psi.py) | dataset error → fit σ |
| ROOT loader | [`core/io/root.py:494`](../../../src/asymmetry/core/io/root.py) | dataset error → fit σ |
| Rebin | [`core/transform/rebin.py:190`](../../../src/asymmetry/core/transform/rebin.py) | rebinned error → fit σ |
| Time-domain F-B | [`core/representation/time.py:78`](../../../src/asymmetry/core/representation/time.py) | representation → fit σ |
| Field-scan integral | [`core/transform/integral.py:144,151`](../../../src/asymmetry/core/transform/integral.py) | ALC/QLCR observable error |
| Simulate reduction | [`core/simulate.py:669`](../../../src/asymmetry/core/simulate.py) | synthetic dataset error |
| GUI grouping | [`gui/mainwindow.py:3432`](../../../src/asymmetry/gui/mainwindow.py) | manual grouping → fit σ |

Sink: `LeastSquares(time, asymmetry, error, model)` in
[`core/fitting/engine.py:165`](../../../src/asymmetry/core/fitting/engine.py)
(single fit) and the concatenated-error global fit (lines 406/464).

There is **already an exact implementation in the codebase**: the simulate-mode
builder uses `var = (1 − A²)/N` / `2α√(FB(F+B))/(F+αB)²` internally
([`core/simulate.py:193`](../../../src/asymmetry/core/simulate.py) and the
diamagnetic objective). So the two halves of the program currently disagree on
the asymmetry error model.

## Recommendation (summary)

**Switch `compute_asymmetry` to exact Poisson propagation.** It is the
physically correct, textbook result and matches WiMDA and musrfit; Mantid is the
outlier. See [`implementation-options.md`](implementation-options.md) for the
chosen option and the migration note (which tests pin the old values and must
change in the same commit). `compute_asymmetry_with_count_errors` already uses
the correct correlated form and needs no change.
