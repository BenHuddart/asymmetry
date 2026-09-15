# Comparison of the distributions and their transforms

## Geometries

| Field at the muon | |B| distribution | Transform | Component |
|---|---|---|---|
| Fixed direction, B = B₁ cos(q·r) (spin-density wave) | (2/π)/sqrt(B₁² − B²), 0 ≤ B ≤ B₁ | J₀(γ_μB₁t) | `OverhauserPowder`, `Bessel` |
| Fixed direction, B = B_av + ΔB cos(q·r), ΔB < B_av | 1/(π sqrt((B − B_min)(B_max − B))) | J₀(γ_μΔB t) cos(γ_μB_av t + φ) | `OverhauserPowderCutoff`/`Centre` |
| Vector on a centred ellipse (helix, cycloid) | (2/π) B/sqrt((B² − B_min²)(B_max² − B²)) | no closed form | `HelicalPowder`, `HelicalCrystal` |

The helical density is the shifted Overhauser density times
`h(B) = 2B/sqrt((B + B_min)(B + B_max))`. At r = 0 the helical density is the
spin-density-wave one (a collinear field reversing sign), so `HelicalPowder`
reduces to `OverhauserPowder`; the fixed-direction form does not. At r = 1 all
three are a single cosine.

In u = B² the helical density is exactly an arcsine density on
[B_min², B_max²] (uniform in the helix phase), which gives the independent
reference used in the tests.

## Exact identities used

- Chebyshev–Bessel: with x = (B − B_av)/ΔB and h(x) = Σ c_n T_n(x),
  `∫ D(B) cos(γBt + φ) dB = Σ c_n J_n(γΔB t) cos(γB_av t + φ + nπ/2)`, since
  `(1/π)∫₀^π cos(nψ) e^{iz cos ψ} dψ = iⁿ J_n(z)`. c₀ = 1 (normalisation) is the
  shifted-Overhauser line.
- Singularities of h at B = −B_min give geometric decay with Bernstein parameter
  ln ρ = arccosh((1 + 3r)/(1 − r)); the crystal weight's 1/B² pole gives
  ln ρ = 2 artanh(√r).
- r = 0 with a phase: the sine transform of (2/π)/sqrt(B₁² − B²) is the Struve
  function H₀, so the line is cos φ J₀ − sin φ H₀.

## Single crystal

For polarization P̂₀ with components a, c along the B_max, B_min axes,
`b_z² = c² + (a² − c²) λ(B)`, `λ = (1 − B_min²/B²)/(1 − r²)` (cross term averages
out over the helix phase); `W₀ = ⟨b_z²⟩ = (a² + c² r)/(1 + r)`. Orientation
averages a² = c² = ⅓ recover W₀ = ⅓ and a constant precessing weight ⅔. With P̂₀
along the B_max axis the precessing weight vanishes at B_max: the upper edge
drops out of the oscillation.

## Approximation error of the closed form (precessing line, max over t)

| r | 0 | 0.1 | 0.2 | 0.46 | 0.8 | 0.99 |
|---|---|---|---|---|---|---|
| max error | 0.315 | 0.236 | 0.166 | 0.099 | 0.032 | 0.001 |

A fit of the closed form with a free phase to an exact, zero-phase line returns
+2° to +9° (larger for smaller r), an amplitude 0.3–6 % high and f_av slightly
low; the exact form returns φ = 0 to 1e-3°.

## Evaluation cost (Apple M2, NumPy/SciPy, one line, phase 0)

Milliseconds per evaluation, powder / crystal. The closed form costs 0.05–0.2 ms
on the same grids.

| Grid | r = 1e-4 | r = 0.01 | r = 0.1 | r = 0.46 | r = 0.9 |
|---|---|---|---|---|---|
| 6 400 points to 8 µs, f = 28 MHz | 11 / 16 | 2.0 / 2.8 | 0.81 / 1.0 | 0.58 / 0.67 | 0.74 / 0.79 |
| 10 000 points to 10 µs, f = 120 MHz | 24 / 39 | 2.1 / 2.8 | 0.96 / 1.2 | 0.71 / 0.79 | 0.70 / 0.73 |
| 2 000 points to 16 µs, f = 5 MHz | 3.4 / 4.7 | 1.2 / 1.3 | 0.42 / 0.56 | 0.29 / 0.36 | 0.32 / 0.33 |
| 492 points, 2–50 ns, f = 150 MHz | 0.75 / 1.0 | 0.16 / 0.20 | 0.09 / 0.10 | 0.07 / 0.08 | 0.05 / 0.06 |

Accuracy against dense helix-phase averages: ≤ 1e-14 for 3e-3 ≤ r ≤ 1, and
≤ 3e-12 for 1e-5 ≤ r ≤ 1e-3 (sinh-clustered samples near B_min).
