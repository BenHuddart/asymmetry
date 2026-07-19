# Beta Correction — Test Data

## No β ≠ 1 reference data exists

Neither the local musrfit checkout's examples (`$MUSRFIT_SRC/doc/examples/`)
nor the WiMDA Muon School corpus contains a run analysed with a detector
β ≠ 1. The msr files that *name* a `beta` FITPARAMETER (e.g.
`test-asy-MUD.msr` param 4 `beta 0.892`) use it as the `generExpo` stretch
exponent in the THEORY block — no `beta` line appears in any RUN block.
Verification therefore rests on algebraic identities and synthetic counts
(see `verification-plan.md`), not on golden files from a reference program.

## Synthetic construction (used by the core tests)

Two-detector model with known ground truth, the same construction the α
estimation tests use (`tests/core/test_alpha_estimation.py`):

    F(t) = N₀,f · exp(−t/τ_μ) · (1 + A₀,f · P(t))
    B(t) = N₀,b · exp(−t/τ_μ) · (1 − A₀,b · P(t))

with `P(t) = cos(ω t)` (wTF), `α_true = N₀,f/N₀,b`, `β_true = A₀,b/A₀,f`.
Reducing with `alpha=α_true, beta=β_true` must recover `A(t) ≈ A₀,f·P(t)`
(up to Poisson noise when noise is added; exactly, when counts are the noise-
free model).

Useful parameter points:

- `β = 1` — regression identity vs the pre-port formula (must be exact).
- `β = 0.8, 1.25` — asymmetric pair covering both directions.
- Degenerate guards: `β ≤ 0`, non-finite β → lenient fallback to 1.0;
  `F = 0`/`B = 0` one-sided bins keep the σ = 1 sentinel; `βF + αB = 0` bins
  return the 0/1 sentinels.

## Existing real data (β = 1 regression only)

Any corpus run already covered by the reduction tests doubles as a β = 1
regression: reducing with an explicit `beta=1.0` (and with no `beta` key in
the grouping) must be bit-identical to the pre-port output. The
`tests/core/test_correction_order_alpha.py` fixtures are the natural host.

## musrfit cross-check (formula-level)

The musrfit-convention equivalence is pinned numerically without musrfit
binaries: for random positive (f, b) arrays and random (α, β),

    ours(F=f, B=b, alpha=α, beta=β)
      == musrfit_form(f, b, alpha_m=1/α, beta_m=β)

where `musrfit_form = (α_m f − b)/(α_m β_m f + b)` is transcribed directly
from `PRunAsymmetry.cpp:1412` into the test.
