# Muonium components — test data

No corpus dependency. Physics is checked against the WiMDA arithmetic; the fit
round-trip uses a signal generated from the component itself (genuine muonium).

## WiMDA-arithmetic checks (closed form)

- g-factors equal the WiMDA literals (`gm = 0.01355342`, `ge = 2.8024` MHz/G).
- `MuoniumTF` at `field`, `A_hf`: in-band transitions straddle `ν_d = γ_µ·field`
  symmetrically, separation = `A_hf`; out-of-band pair weight `(1−δ) ≈ 0`.
- Positive-frequency convention: `tf_muonium(t=0; …, φ) = cos(φ)` (all lines +φ).
- `MuoniumZF`: `f1=A_hf−D_mu`, `f2=A_hf+D_mu/2`, `f3=3D_mu/2`; with `f_cut=0` the
  weights are `1,2,2` normalised by 6; with `f_cut>0` the Lorentzian rolls them off.

## Self-consistency fit fixture (genuine muonium)

Generated from `MuoniumTF * Exponential + Constant` with well-separated
satellites so the fit is well-conditioned:

- `A = 20`, `field = 100` G, `A_hf = 2.0` MHz (satellites at `ν_d ± 1` MHz),
  `phase = 0.3`, `Lambda = 0.2` µs⁻¹, `A_bg = 0.5`,
- t-grid 0–12 µs, Gaussian noise σ = 0.2.

Seeded near truth (e.g. `A_hf = 2.1`), the fit recovers `A_hf ≈ 2.0` at χ²ᵣ ≈ 1.
(Verified: χ²ᵣ = 1.01, `A_hf = 2.0001 ± 0.0002`.)

## Why not CdS

Shallow-donor CdS (tiny `A_hf`, TF 100 G) is fit with three independent
oscillating lines + link groups, which reach χ²ᵣ ≈ 1.35; the constrained muonium
parameterisation tops out at χ²ᵣ ≈ 22 there (see comparison.md). So the muonium
components are verified on genuine-muonium synthetic data, not the CdS corpus.
