# Verification plan

To run once Option A (exact propagation) is implemented.

## Unit-level (formula correctness)

1. **Exact-form equality.** For random `F, B > 0` and several α, assert
   `compute_asymmetry(F,B,α).error == 2|α|√(FB(F+B))/(F+αB)²` to float tolerance.
2. **α = 1 identity.** Assert `error² == (1 − A²)/(F+B)` for α = 1.
3. **Agreement with the count-error path.** Assert
   `compute_asymmetry(F,B,α).error ≈ compute_asymmetry_with_count_errors(F,B,√F,√B,α).error`.
   This ties the two public functions to one model permanently.
4. **One-sided counts.** `F=1, B=0`: assert the chosen floor behaviour (decide:
   floor σ, or fall through the zero-denominator sentinel). Replaces the old
   `√2` pin in `test_one_sided_counts_have_nonzero_error`.
5. **Zero denominator.** `F=B=0` still returns σ = 1 (sentinel unchanged).

## Statistical (the property that motivated the change)

6. **Refit χ²ᵣ → 1.** `TestRefitRecovery::test_round_trip_refit_recovers_parameters`
   with `expected_chi2r = 1.0` (drop the `(1−A²)/(1+A²)` correction). Band
   `3√(2/dof)` should still pass.
7. **Pulls ~ N(0,1).** `TestRefitRecovery::test_pull_distribution_over_seeds`
   should pass with the existing bands and the amplitude pull SD moving from
   ~0.97 toward 1.0. Confirm mean within `4/√n` and var within `4√(2/(n−1))`.
8. **Monte-Carlo coverage (optional, not in CI).** Reproduce §2 of
   [`test-data.md`](test-data.md): ⟨χ²ᵣ⟩ → ~1.00, pull SD → ~1.0.

## Oracle cross-check

9. Spot-check 2–3 `(F,B,α)` triples against the exact analytic value and the
   Monte-Carlo variance (the §1 table). Do **not** assert against Mantid's
   `AsymmetryCalc` output — that is the value we are deliberately leaving behind;
   if a Mantid-parity check is ever wanted, gate it behind Option C.

## Regression sweep

10. `python tools/harness.py validate` green after updating every test listed in
    the migration note of [`implementation-options.md`](implementation-options.md).
11. Grep `Asymmetry-testing` worktree and any saved `.asymp` goldens for pinned
    σ_A; regenerate as needed.

## Acceptance

- All formula unit tests pass against the exact form.
- χ²ᵣ band re-centred on 1 and passing; pulls consistent with N(0,1).
- Full `validate` green.
- Docstring no longer claims Mantid-compatibility for the error model; the
  divergence is recorded as intentional.
