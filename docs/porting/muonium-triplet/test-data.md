# Muonium triplet — test data

## Synthetic fixture (in-repo, no corpus dependency)

Same approach as the link-groups tests: a **damped symmetric triplet + flat
background**, built from the component itself, so the suite never depends on the
WiMDA corpus.

```
A(t) = e^(−λ t) · [ A_c·cos(2π f₀ t + φ)
                   + A_s·cos(2π (f₀ − Δ/2) t + φ)
                   + A_s·cos(2π (f₀ + Δ/2) t + φ) ]   + bg
```

Reference values:

- `f₀ = 1.389` MHz, `Δ (hyperfine) = 0.242` MHz → satellites at 1.268 / 1.510 MHz
- `λ = 0.30` µs⁻¹, `φ = 0`
- `A_c = 10`, `A_s = 6`, `bg = 0.5`
- t-grid 0–12 µs, per-point error ~0.15; converges to χ²ᵣ ≈ 1 from sensible seeds.

## What the synthetic tests assert

1. **Shape**: the component function emits exactly three lines at `f₀` and
   `f₀ ± Δ/2` (FFT peak positions / direct frequency check), symmetric about `f₀`.
2. **Single-parameter symmetry**: varying `hyperfine` moves both satellites
   together and keeps them symmetric; `f_centre` shifts all three rigidly.
3. **Round-trip recovery**: fitting the synthetic data recovers
   `f_centre ≈ 1.389`, `hyperfine ≈ 0.242`, shared `λ`/`φ`, with χ²ᵣ ≈ 1, using
   **fewer free parameters** than three independent lines (6 + background = 7 vs
   13), and the hyperfine constant comes straight out as the fitted `hyperfine`.
4. **Composite integration**: `CompositeModel.from_expression("MuoniumTriplet +
   Constant")` builds with the expected `param_names`; the component appears in
   the builder category map; `.asymp` save/load round-trips the model + params.
5. **Picklability**: the component function is a module-level callable (batch/
   global fits pickle it).

## CdS real-data acceptance (corpus, not committed)

Run EMU00020721 (≈5.12 K, TF 100 G), Data_hdf5 copy, from the Muon School
corpus. Used only for the manual/engine acceptance in verification-plan.md; never
committed, never imported by the suite. The link-groups study already verified
this run yields central f≈1.389 MHz, 2δ≈0.242 MHz, χ²ᵣ≈1.35 with three free-
frequency lines; the triplet must reproduce it with the constrained form.
