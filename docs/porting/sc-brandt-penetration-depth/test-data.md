# Test data

CI has **no corpus access**, so all automated tests use synthetic σ(B₀)
generated from the model with known (λ, B_c2) plus noise, and assert recovery.

## Synthetic round-trip (primary test)

1. Pick truth: λ = 195 nm, B_c2 = 25 T (b spans ~0–0.024 over 100 G–6000 G —
   the LiFeAs Sample-1 range) and a higher-b case (B_c2 = 0.5 T) to exercise
   the field-dependence shape away from the plateau.
2. Field grid in Gauss matching the corpus sweep
   (100, 200, 800, 1600, 3200, 4000, 6000 G) plus a dense high-b grid.
3. `σ = brandt_field_width_sigma(B0, λ, B_c2, powder=…)`; add small Gaussian
   noise (seeded `numpy.random.default_rng`).
4. Fit with `fit_parameter_model` + the registered component; assert
   λ within a few % of truth and B_c2 recovered when the grid reaches
   high enough b to constrain it.

## Analytic anchors (no fitting)

- `g(0) = 1`: `σ(B₀→0) == lambda_nm_to_sigma_us(λ)` (ties to existing helper).
- `g(1) = 0`: `σ(B₀ = B_c2) == sigma_bg` (clamped, no NaN); `b > 1` ⇒ σ = σ_bg.
- Bracket: at b→0 the unnormalised bracket = 2.21.
- Powder vs single crystal: `σ_powder = σ_single / √3` at equal (λ, b).
- Quadrature background: `σ(σ_bg>0) == hypot(σ_VL, σ_bg)`.
- Monotonic decrease of σ in B₀ for `0 < b < 1` at fixed λ.

## Edge / validation cases

- λ ≤ 0 and B_c2 ≤ 0 return finite (guarded) values, never NaN/inf in the
  residual path.
- Scalar and array `B0` inputs both work; output dtype float64.

## Corpus validation (manual, NOT in CI) — reported in the PR

LiFeAs PSI `.bin` runs under
`~/Documents/WiMDA muon school/Superconductivity/LiFeAs/data/`:

- Sample 1 (LFA, T_c 16 K) field sweep at 2 K: runs 3375/3377/3379/3381/
  3383/3385/3387 (800/1600/3200/4000/200/6000/100 G) → fit powder model →
  expect λ_ab ≈ **195 nm** (target 195(2) nm).
- Sample 2 (LFA_2, T_c ≈ 12 K) 1.5 K low-field set 3663/3665/3667/3693/3695/
  3697 → expect λ_ab ≈ **244 nm** (target 244(2) nm), noting Sample 2 needs
  the field-induced magnetic σ_M ∝ B₀^½ term the paper adds (out of scope
  here; report the caveat).

Sanity B_rms plateau (powder, b→0): 195 nm → 1.91 mT, 244 nm → 1.22 mT vs
Fig. 1 ≈ 1.9 / 1.0–1.3 mT.
