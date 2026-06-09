# Link groups — test data

## Synthetic fixture (in-repo, no corpus dependency)

The WiMDA corpus files are **not** in the repository, so the linking tests use a
synthetic **damped-cosine triplet + flat background**, mimicking the CdS
three-line signal:

```
A(t) = Σ_{k∈{−,0,+}} a_k · cos(2π f_k t + φ) · exp(−λ t) + bg
f₋ = f₀ − δ,   f₀,   f₊ = f₀ + δ
```

Reference values used by the tests:

- `f₀ = 1.389` MHz (central Larmor line, ≈ γ_µ·100 G)
- `δ   = 0.121` MHz  → splitting `2δ = 0.242` MHz
- shared `λ = 0.30` µs⁻¹, shared `φ = 0`
- satellite amplitudes equal; central amplitude larger
- flat background `bg`
- t-grid 0–12 µs, realistic per-point error; values chosen so a sensibly-seeded
  fit converges to χ²ᵣ ≈ 1.

The triplet is built directly from the shipped `Oscillatory`/`Exponential`/
`Constant` components (`CompositeModel.from_expression(...)`), so the test
exercises the real model path, not a bespoke function.

## What the synthetic tests assert

1. **Equality**: with the three relaxation rates (and the two satellite
   amplitudes, and the three phases) placed in link groups, after the fit every
   follower equals its group main exactly, and the follower's reported
   uncertainty equals the main's.
2. **Free-set reduction**: followers are excluded from the free parameters; the
   fit's free-parameter count drops by the number of followers; reduced-χ² uses
   the reduced count.
3. **Recovery**: the three free frequencies come back symmetric about `f₀` and
   the splitting `f₊ − f₋ ≈ 0.242` MHz.
4. **Round-trip**: link groups survive an `.asymp` save/load.

## CdS real-data acceptance (corpus, not committed)

Run 20720-class 5.2 K CdS file from the Muon School corpus
(`~/Documents/WiMDA muon school/Semiconductors/Shallow donor state in cadmium
sulphide/`). Used only for the manual/engine acceptance check in
verification-plan.md; never committed and never imported by the test suite.
