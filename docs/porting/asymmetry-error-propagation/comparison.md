# Comparison — asymmetry error formula across reference programs

## The derivation (verified)

`A = N/D`, `N = F − αB`, `D = F + αB`. Partial derivatives:

```
∂A/∂F = (D − N)/D² =  2αB/D²
∂A/∂B = −α(D + N)/D² = −2αF/D²
```

With Poisson variances `var(F) = F`, `var(B) = B` (and `cov(F,B) = 0`):

```
var(A) = (2αB/D²)² · F + (2αF/D²)² · B
       = 4α² F B (F + B) / D⁴          ← EXACT
```

At α = 1, `D = F + B` and `1 − A² = 4FB/(F+B)²`, so

```
var(A) = 4FB/(F+B)³ = (1 − A²)/(F + B).
```

### Where the shipped formula comes from

Treat `N` and `D` as **independent** with `var(N) = var(D) = F + α²B ≡ Q` and use
relative-variance addition for a ratio:

```
var(A)/A² = var(N)/N² + var(D)/D²
⇒ var(A) = Q·(A²/N² + A²/D²) = Q·(1/D² + N²/D⁴)
         = (F + α²B)·(1 + (N/D)²)/D²        ← SHIPPED
```

At α = 1 this is `(1 + A²)/(F + B)`. The discrepancy is exactly the dropped
covariance `cov(N, D) = var(F) − α²var(B) = F − α²B`. The independent-propagation
result is **larger** than exact whenever `|A| > 0`:

```
σ_shipped² / σ_exact²  =  (F + α²B)(D² + N²) / (4α² F B (F + B))
                       =  (1 + A²)/(1 − A²)      at α = 1.
```

### Numerical confirmation (Monte Carlo, 200 k Poisson draws/case)

| F | B | α | A | MC var | exact | shipped | shipped/MC | exact/MC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10000 | 10000 | 1.0 | 0.000 | 5.00e-5 | 5.00e-5 | 5.00e-5 | 1.000 | 1.000 |
| 6000 | 4000 | 1.0 | 0.200 | 9.61e-5 | 9.60e-5 | 1.04e-4 | 1.083 | 0.999 |
| 12000 | 8000 | 1.0 | 0.200 | 4.79e-5 | 4.80e-5 | 5.20e-5 | 1.086 | 1.002 |
| 5000 | 5000 | 1.5 | −0.200 | 9.28e-5 | 9.22e-5 | 1.08e-4 | 1.165 | 0.993 |
| 20000 | 10000 | 1.0 | 0.333 | 2.98e-5 | 2.96e-5 | 3.70e-5 | 1.243 | 0.994 |
| 3000 | 2700 | 0.9 | 0.105 | 1.72e-4 | 1.72e-4 | 1.78e-4 | 1.037 | 1.003 |

`exact/MC ≈ 1.000` to MC noise in every case; `shipped/MC` runs 1.04–1.24,
matching `(1+A²)/(1−A²)` (e.g. A = 0.333 → 1.25). **Derivation confirmed.**

## What the reference programs actually compute (oracle)

GPL sources used as a verification oracle only — described in math, never copied.

| Program | Asymmetry error | Form | Source |
| --- | --- | --- | --- |
| **Mantid `AsymmetryCalc`** | `√(F + α²B)·√(1 + A²) / (F + αB)` | **independent-propagation `1 + A²`** (over-estimate) | [docs.mantidproject.org/.../AsymmetryCalc-v1](https://docs.mantidproject.org/nightly/algorithms/AsymmetryCalc-v1.html) |
| **WiMDA** (Pratt, ISIS) | count-statistics, `∝ 1/√(counts)` | `(1 − A²)/(F+B)` family (exact) | WiMDA manual §5.2 "constant-error binning"; Pratt, Physica B 289–290, 710 (2000) |
| **musrfit** (PSI) | single-histogram `√N` log-likelihood; `√((1−A²)/N)` when forming A | exact `1 − A²` | [lmu.web.psi.ch/musrfit](https://lmu.web.psi.ch/musrfit/) |
| **Textbook** (Blundell et al., OUP 2022) | `σ_A = √((1 − A²)/N)`, `N = F+B` | exact `1 − A²` | standard binomial/Poisson result |

### Key oracle finding

**Mantid `AsymmetryCalc` is the odd one out.** Its published algorithm
documentation states the error verbatim as `√(F + α²B)·√(1 + A²)/(F + αB)` — the
`1 + A²` independent-propagation form, which is exactly what Asymmetry shipped
(the code comment "Match Mantid AsymmetryCalc error model" is accurate). WiMDA,
musrfit, and the standard μSR literature all use the `1 − A²` correlation-aware
form. The discriminator is the sign in `(1 ± A²)`: Mantid `+`, everyone else `−`;
they coincide only at `A = 0`.

So the question is not "which program is right to copy" but "do we want
bug-for-bug Mantid compatibility, or the physically correct error?" Asymmetry's
stated invariant is correct science in the core, and its own simulate-mode
builder already uses the exact form — the codebase is internally inconsistent
today.

## Caveats

- Confidence on Mantid's formula: **high** (verbatim in algorithm docs).
- Confidence on WiMDA's exact algebraic form: **medium** (not written in closed
  form in public docs; inferred from the count-statistics binning model). High
  confidence it is *not* the `1 + A²` form.
- `α ≠ 1`: the over-estimate ratio is no longer simply `(1+A²)/(1−A²)` but the
  full `(F+α²B)(D²+N²)/(4α²FB(F+B))`; the bias persists and grows with α
  imbalance (see the α = 1.5 MC row, ratio 1.165).
- The `denominator == 0 → σ = 1` sentinel is shared by both forms; keep it.
