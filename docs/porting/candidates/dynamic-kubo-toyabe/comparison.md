# Dynamic Kubo–Toyabe: implementation comparison

| Aspect | Mantid | musrfit | WiMDA | Asymmetry |
|---|---|---|---|---|
| Function name | `DynamicKuboToyabe` | `dynKTLF`, `dynKTZF` (in `PTheory`) | dynamic KT via `musr-function-registry` DLL | ❌ |
| Algorithm | Strong-collision time-step convolution | Memory-kernel quadrature | DLL-provided (Pascal interface) | n/a |
| Public symbols | `Functions::DynamicKuboToyabe::function1D` | `PTheory::DynamicKTLF`, `PTheory::DynamicKTZF` | exported by `musrfunctions.dll` | n/a |
| Parameters | A, Δ (Gaussian width), ν (fluctuation rate), B_L | A0, Δ, ν, B_LF, φ | A, Δ, ν, B_L | — |
| Validated limits | Static (ν=0) reduces to `StaticKuboToyabe` | Same | Same | — |
| Test data | `Framework/CurveFitting/test/Functions/DynamicKuboToyabeTest.h` | `src/tests/spirit/` golden curves | regression-tested via UI | — |

## Numerical strategies

**Time-step convolution (Mantid).** For each output time `t_n`, integrate
`P_static(t_n) · exp(-ν · t_n) + ∫₀^{t_n} ν · exp(-ν · (t_n - t')) ·
P_static(t') dt'`. The integral becomes a discrete sum over previous
output times. O(N²) per evaluation; trivially parallelisable.

**Memory-kernel quadrature (musrfit).** Cast as
`P_dyn(t) = ℒ⁻¹ [P̃_static(s + ν) / (1 - ν · P̃_static(s + ν))]`
where ℒ is the Laplace transform. Avoids the O(N²) cost but
requires careful handling of the Laplace inversion.

**Recommendation for Asymmetry:** time-step convolution first — easier
to validate and aligns with the existing per-bin evaluation style of
`models.py`. Quadrature can land as an optimisation later if profiling
shows the convolution is a bottleneck.

## Edge cases the study should document

- `ν · t_max ≫ 1`: ensure the exponential-decay limit is reached
  cleanly without catastrophic cancellation.
- `B_L · γ_μ ≫ Δ`: decoupling limit; output should approach `exp(-λ·t)`
  with `λ → 2Δ²/ν`.
- `ν = 0` exactly: must call straight through to static KT (avoid
  divide-by-zero in the integral).
