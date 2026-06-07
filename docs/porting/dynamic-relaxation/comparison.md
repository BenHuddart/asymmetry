# Dynamic relaxation: reference-program comparison

How the four functions are implemented across the reference programs, and the
parameter conventions Asymmetry must reconcile.

## Function availability

| Function | Mantid | musrfit | WiMDA | Asymmetry (now) |
|---|---|---|---|---|
| Static Gaussian KT (ZF) | ✓ | `statGssKT` | ✓ | ✓ `StaticGKT_ZF` |
| Static Gaussian KT (LF) | ✓ | (in dynGssKT, ν=0) | ✓ | ✓ `LFKuboToyabe` |
| Dynamic Gaussian KT (ZF+LF) | `DynamicKuboToyabe` | `dynGssKT` | ✓ | ✗ |
| Static/Dynamic Lorentzian KT | partial | `statLrKT`/`dynLrKT` | ✓ | ✗ |
| Keren | ✗ (named) | `combiLGKT`/user | ✓ | ✗ |
| Abragam | ✗ (named) | user function | ✓ | ✗ |

## Numerical strategy for the dynamic (strong-collision) functions

All three programs solve the same Volterra integral equation of the second kind
that links the dynamic polarisation G_d(t) to the static G_s(t) for a
strong-collision (Markovian) process at rate ν:

    G_d(t) = g(t) + ν ∫₀ᵗ g(t−τ) G_d(τ) dτ,   with  g(t) = e^{−νt} G_s(t).

Laplace domain (used to derive limits, not for evaluation):

    Ĝ_d(s) = ĝ(s) / (1 − ν ĝ(s)).

- **Mantid** (`DynamicKuboToyabe.cpp`): fixed-step **time-step convolution** on a
  uniform grid (default step `eps`), recursive trapezoidal accumulation. Static
  G_s is the Gaussian ZF/LF KT. Validates against the slow/fast analytic limits.
- **musrfit** (`PTheory::CalcDynKTLF`): memory-kernel quadrature — convolves the
  static response with the exponential fluctuation kernel; finer adaptive grid,
  faster but more code.
- **WiMDA**: equivalent strong-collision integration in the Pascal user-function
  DLL.

**Takeaway for Asymmetry:** the **time-step convolution (Mantid-style)** is the
easiest to validate and matches the precision/perf trade-off already used by
`LFKuboToyabe` (cached `scipy.integrate.quad`). It is the recommended seam (see
implementation-options.md).

## Static building blocks needed

| Static G_s(t) | Closed form? | Source |
|---|---|---|
| Gaussian ZF | yes: `1/3 + 2/3(1−Δ²t²)e^{−Δ²t²/2}` | already in `models.py` |
| Gaussian LF | yes (Hayano integral term) | already in `models.py` |
| Lorentzian ZF | yes: `1/3 + 2/3(1−at)e^{−at}` | Uemura 1985 — new, trivial |
| Lorentzian LF | semi-analytic (spherical Bessel j₀,j₁ + one integral) | Uemura 1985 — new |

The Gaussian dynamic functions reuse the **existing** static engine; only the
Lorentzian statics are new code.

## Analytic functions (no convolution)

- **Abragam** (single component):
  `G(t) = exp[ −(σ²/ν²)(e^{−νt} − 1 + νt) ]`.
  Limits: ν→0 → `exp(−σ²t²/2)` (Gaussian); ν→∞ → `exp(−(σ²/ν) t)` (exponential,
  motional-narrowed rate σ²/ν).
- **Keren** (LF dynamic Gaussian), `P(t) = exp[−Γ(t)]` with
  Γ(t) = (2Δ²/(ω₀²+ν²)²) × [ (ω₀²+ν²) ν t
         + (ω₀²−ν²)(1 − e^{−νt} cos ω₀t) − 2 ν ω₀ e^{−νt} sin ω₀t ],
  ω₀ = γ_µ B_L. At ω₀=0: Γ = (2Δ²/ν²)(e^{−νt}−1+νt) = 2× the Abragam exponent
  (two transverse ZF components). These two are pure NumPy — no integration.

## Parameter-convention reconciliation

| Quantity | This study | Notes |
|---|---|---|
| Gaussian width | `Delta` (µs⁻¹) | as in existing KT models |
| Lorentzian width (rate) | `a` (µs⁻¹) | HWHM of the Lorentzian field dist. (reuse `a` ParamInfo, but add a dedicated `a_L` to carry µs⁻¹ unit + description) |
| fluctuation/hop rate | `nu` (MHz) | existing ParamInfo; ν⁻¹ = correlation time |
| longitudinal field | `B_L` (G) | existing; ω₀ = γ_µ B_L |
| Abragam width | `sigma` (µs⁻¹) | existing; single-component static width |
| amplitude | `A` / `A0` (%) | existing |

musrfit and WiMDA both quote ν as a rate in MHz (≡ µs⁻¹); Asymmetry adopts the
same, consistent with the corpus guides.
