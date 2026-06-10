# Test data and measured impact

All numbers from the project venv (`.venv/bin/python`, numpy 2.2.x, iminuit).
Scripts are throwaway reproductions; the key results are tabulated here so the
study is self-contained.

## 1. Monte Carlo validation of the derivation

200 000 Poisson draws per `(F, B, α)` case; compare sample variance of
`A = (F−αB)/(F+αB)` against exact `4α²FB(F+B)/(F+αB)⁴` and shipped
`(F+α²B)(1+(N/D)²)/D²`. See table in [`comparison.md`](comparison.md).

Result: `exact/MC ≈ 1.000` within MC noise everywhere; `shipped/MC` = 1.04–1.24,
tracking `(1+A²)/(1−A²)`. Derivation confirmed.

## 2. Fitted-uncertainty impact (controlled known-truth fit)

Model `A(t) = A₀·exp(−λt)`, A₀ = 0.22, λ = 0.6 µs⁻¹, 400 bins over 0.05–8 µs,
total counts decaying with the muon lifetime (N₀ = 3×10⁴ at t=0). Poisson F, B
per bin; fit with iminuit `LeastSquares` using each error model; 300 seeds.

| Model | ⟨χ²ᵣ⟩ | ⟨σ_A₀⟩ | ⟨σ_λ⟩ | pull SD A₀ | pull SD λ |
| --- | --- | --- | --- | --- | --- |
| shipped (1+A²) | 0.9874 | 1.61e-3 | 0.0082 | 0.974 | 1.025 |
| exact (1−A²) | 0.9988 | 1.56e-3 | 0.0081 | 1.005 | 1.040 |

- σ_A₀ shipped/exact = **1.032** (amplitude uncertainty inflated ~3.2 %).
- σ_λ shipped/exact = **1.014** (decay-rate uncertainty inflated ~1.4 %).
- χ²ᵣ shipped/exact = **0.989** (shipped biased low; exact centres on 1).
- Amplitude pull narrows to 0.974 under the shipped (over-estimated) σ and
  corrects to 1.005 under exact — i.e. the shipped errors are genuinely too big.

The amplitude (high-A parameter) takes the larger hit; the rate, constrained by
the late-time low-A tail, less so. Magnitude scales with the asymmetry amplitude
of the dataset.

## 3. Corpus σ ratio (WiMDA Muon School corpus)

At α = 1 the per-point ratio `σ_shipped/σ_exact = √((1+A²)/(1−A²))` depends only
on A, so it is dataset-independent given the asymmetry. Across loaded runs
(EuO .bin, LiFeAs GPS, benzene): median ratio 1.00 for low-amplitude / near-zero
runs, rising to **1.23–1.31** median on high-asymmetry chemistry runs
(benzene, |A| median 0.45–0.51), with per-bin ratios up to ~26 in early/low-stat
bins where A → 1 (those bins carry negligible fit weight). So the corpus spans
the full range from "negligible" to "≈30 % in σ on the dominant bins."

Caveat: a few loaded datasets report `asymmetry ≈ 0` across most bins (loader/
grouping fills), so their median ratio is 1.0; this measures the analytic A-
dependence, not a per-dataset fit re-run. The decision-grade fit impact is §2.

## 4. Reference: exact form already in-tree

- `compute_asymmetry_with_count_errors`
  ([`asymmetry.py:96`](../../../src/asymmetry/core/transform/asymmetry.py)) —
  `2|α|√((B·e_F)² + (F·e_B)²)/D²`, which with `e_F=√F, e_B=√B` is exactly the
  recommended form.
- Simulate builder
  ([`simulate.py:193`](../../../src/asymmetry/core/simulate.py)) —
  `√((1−A²)/N)`; diamagnetic objective `2α√(FB(F+B))/(F+αB)²`.

These are the oracles the new `compute_asymmetry` should reproduce.
