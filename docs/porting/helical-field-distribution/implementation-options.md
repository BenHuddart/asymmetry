# Implementation options

## 1. Replace the closed form, or add components — CHOSEN: add

The shifted-Overhauser closed form is exact for a fixed-direction modulated
field, so it is a model in its own right, not only an approximation. Adding
`HelicalPowder`/`HelicalCrystal` keeps saved projects bit-identical and lets the
user choose by geometry. Replacing in place was rejected.

## 2. Evaluating the transform

Benchmarked on 2 000–10 000-point grids up to ω_max t ≈ 7 500.

- **Direct quadrature in θ, u = B² uniform** (the helix phase). The cosine part
  is entire in u and needs N ≈ Δω t nodes, but the sine part (needed for a phase)
  has a near-kink at B_min and converges only as N ≳ 1/r. 8–90 ms per curve.
  Rejected as the main path.
- **Chebyshev–Bessel series** of h with J_n from an upward recurrence. Cost
  ∝ terms × points, independent of t; the recurrence is unstable for n > Δω t.
  **Chosen** for Δω t above twice the series length.
- **Gauss–Chebyshev quadrature of D·cos over the arcsine variable**, weights h at
  the nodes. N ≈ max(Δω t/2, structure of h); smooth for every r > 0. **Chosen**
  for early times (where it is also the cheaper path) and for small r, where the
  series is long.
- Endpoint (steepest-descent) contour integrals: non-oscillatory Laplace
  transforms at large t, but they diverge individually as t → 0 and lose
  accuracy when ω_min t is small. Rejected.
- Tabulating G(ω_max t, r): oscillatory in both arguments. Rejected.
- Numba/C: not a dependency; the NumPy form is fast enough.

## 3. Small r — CHOSEN: closed form below r = 1e-6

Series length and quadrature nodes grow as 1/√r. For r < 1e-6 the r = 0
transform (J₀, and J₀/H₀ with a phase) differs from the exact line by
≈ 0.4 r² ω_max t ln(1/(r ω_max t)) < 1e-8 out to ω_max t ≈ 2e4, which is below
what the quadrature resolves there. Bounding `ratio` away from 0 was rejected
(it would make r = 0 unreachable, where the line is exact and cheap).

## 4. Tolerances — CHOSEN: 1e-12, calibrated counts

Series terms `(1 + 0.3 r) ln(1/ε)/ln ρ + 4` and structure nodes
`max(0.34, 0.34 + 0.045 log₁₀(1e5 r)) ln(1/ε)/ln ρ + 12`, calibrated against
converged references for 1e-5 ≤ r ≤ 0.95 (`test-data.md`). Discrete changes of
the counts move the line by < 1e-12, invisible to finite-difference gradients.

## 5. Phase — CHOSEN: parameter kept, fixed at 0 by default

Every field B on the ellipse pairs with −B, so the zero-field phase is exactly
zero. A free phase mostly absorbs a time-zero offset (phase ∝ frequency) or the
closed form's lineshape error. Kept so instrumental offsets can still be
modelled.

## 6. Ratio above 1 — CHOSEN: relabel the axes

`ratio > 1` is mapped to `frequency·ratio`, `1/ratio` with the crystal's axes
swapped: the same ellipse. Mathematically exact, so no bound is imposed.

## 7. Single-crystal parametrisation — CHOSEN: angles in degrees

`theta_h` (polarization to helix-plane normal) and `phi_h` (in-plane, from the
B_max axis), matching the registry's degree convention for angles. Only a², c²
enter.

## Deferred

- Fit Wizard templates for `HelicalPowder` (after real-data trials).
- Detector axis not along P̂₀ (weights `(P̂₀·b)(n̂·b)`), e.g. a spin-rotated
  geometry analysed with transverse detector pairs.
- Distributions of B_max (disorder) beyond a multiplicative relaxation envelope.
