# Implementation options

## Option A — switch to exact Poisson propagation (RECOMMENDED)

Replace the error branch of `compute_asymmetry` with the correlation-aware form:

```
σ_A = 2·|α|·√( F·B·(F + B) ) / (F + αB)²        (denominator ≠ 0)
σ_A = 1                                          (denominator == 0, unchanged)
```

Equivalent to `√(4α²FB(F+B))/D²` = `√((1−A²)/(F+B))` at α = 1. This is the same
form already used by `compute_asymmetry_with_count_errors` (with √F, √B count
errors substituted) and by the simulate-mode builder — so it unifies the two
halves of the codebase on one correct model.

- **Pros**: physically correct; matches WiMDA, musrfit, textbook; fixes the
  low-χ²ᵣ bias and the too-narrow pulls; makes the core self-consistent.
- **Cons**: every fitted error changes slightly (≤ a few %); diverges from
  Mantid `AsymmetryCalc` (a deliberate, documented divergence — Mantid is wrong
  here); the code comment and a handful of pinned-value tests must change.
- **Edge cases**: `F = 0` or `B = 0` (one-sided counts) → σ = 0 under the exact
  form, because A = ±1 is then a boundary value with no first-order spread. The
  shipped form gave √2 there. This is correct (a deterministic boundary has zero
  linearised variance) but a literal **0** is a poor fit weight — it would make
  that bin infinitely weighted. Mitigation: floor the per-bin σ at a small
  positive value when `F·B = 0` (e.g. reuse the existing zero-denominator
  sentinel path, or set σ from the larger of the two counts). Decide and test
  explicitly; this is the one real behavioural wrinkle.

## Option B — keep the Mantid form, document the bias

Leave `compute_asymmetry` unchanged; rely on the docstring/study to record that
σ_A is an over-estimate.

- **Pros**: zero churn; bit-for-bit Mantid parity preserved.
- **Cons**: ships a known-incorrect error into every fit; leaves the core
  inconsistent with its own simulate builder; the simulate verification suite
  has to keep centring on a biased expectation. Rejected.

## Option C — make it selectable (mantid | exact), default exact

Add an `error_model` argument / project setting.

- **Pros**: explicit Mantid-parity escape hatch for cross-validation.
- **Cons**: a public surface and a second code path to test for a formula that
  is simply wrong; no user has asked for the Mantid value as a feature. Deferred
  — can be added later if cross-program parity testing needs it. Not needed for
  correctness.

## Decision

**Option A.** Switch to exact propagation with an explicit one-sided-counts
floor. Record the Mantid divergence in the docstring (replace "Match Mantid
AsymmetryCalc error model" with a note that Mantid over-estimates via
independent N/D propagation and we use exact Poisson on purpose).

## Migration note — every fitted error changes; tests that pin the old values

The switch changes `σ_A` for all `|A| > 0`, so it must land as one commit that
updates the formula **and** these tests together. Tests identified on `main`:

### Pins a specific numeric σ_A — must be recomputed

- [`tests/test_transforms.py::test_one_sided_counts_have_nonzero_error`](../../../tests/test_transforms.py)
  — asserts `err == √2` for `F=1, B=0`. Under exact propagation this is the
  one-sided-counts edge case (→ 0 before flooring). Update to the chosen floor
  behaviour and rename to reflect it.
- [`tests/test_transforms.py::test_supplied_count_errors_use_musrfit_style_propagation`](../../../tests/test_transforms.py)
  — exercises `compute_asymmetry_with_count_errors`, which is **already exact**;
  no change expected, but re-run to confirm it still passes (it is the template
  the new `compute_asymmetry` should match).
- [`tests/test_time_integral_asymmetry.py::test_integral_shares_compute_asymmetry_error_model`](../../../tests/test_time_integral_asymmetry.py)
  — pins `error ≈ shipped(F_int,B_int,α=1.3)`. Recompute against the exact form.
- [`tests/test_representation_model.py::test_fb_asymmetry_matches_core_pipeline`](../../../tests/test_representation_model.py)
  — element-wise `assert_allclose(ds.error, expected_err)` where `expected_err`
  is computed by the shipped formula inline. Update the inline expectation to the
  exact formula (it recomputes, so just mirror the new core math).
- [`tests/test_mainwindow_additional.py`](../../../tests/test_mainwindow_additional.py)
  (~line 2314) — combines two `compute_asymmetry` errors in a vector subtraction;
  re-derive the expected combined error.

### Centred on the biased expectation — must be re-centred on 1

- [`tests/test_nexus_writer.py::TestRefitRecovery::test_round_trip_refit_recovers_parameters`](../../../tests/test_nexus_writer.py)
  — `expected_chi2r = mean((1−A²)/(1+A²))`. After the switch, χ²ᵣ centres on
  **1**, so set `expected_chi2r = 1.0` and drop the `(1−A²)/(1+A²)` correction
  and its explanatory comment.
- [`tests/test_nexus_writer.py::TestRefitRecovery::test_pull_distribution_over_seeds`](../../../tests/test_nexus_writer.py)
  — pulls should already be ~N(0,1); under-narrowed today. The σ_var band should
  still hold and, ideally, tighten. Re-run; widen the band only if MC variance
  genuinely demands it.

### Survives unchanged (sanity-checked, no action)

- `test_zero_denominator_uses_default_error` — sentinel = 1 in both forms.
- All asymmetry-value-only tests (`test_equal_counts`, `test_known_asymmetry`,
  `test_alpha_scaling`, the integral asymmetry-value tests) — error not asserted.

### Out-of-tree downstream

Any pinned σ_A in the testing worktree (`Asymmetry-testing`) or saved `.asymp`
golden projects with stored errors. Grep before merge; regenerate goldens.

## As implemented (2026-06-10)

Option A landed in
[`compute_asymmetry`](../../../src/asymmetry/core/transform/asymmetry.py):
`σ_A = 2|α|√(FB(F+B))/(F+αB)²`, with the one-sided floor realised by computing
the exact error only on `informative = safe & (F·B > 0)` bins and leaving every
other bin at the `1.0` no-information sentinel (shared with the zero-denominator
case). The docstring/comment no longer claim Mantid compatibility for the error
model and point here for the divergence.

Deviations from the plan, all minor:

- **One-sided floor = the existing sentinel**, not a new computed floor. `F·B=0`
  bins reuse the `err = np.ones_like(f)` default rather than a separate clamp,
  so there is a single sentinel value (1.0) and no extra branch. Confirmed it
  reads as "essentially unweighted" against real σ_A ≈ 0.01–0.1.
- **Most "recompute" tests needed no edit.** `test_integral_shares_compute_asymmetry_error_model`,
  `test_fb_asymmetry_matches_core_pipeline`, and the `test_mainwindow_additional`
  vector-subtraction test all derive their expected error by *calling*
  `compute_asymmetry`, so they stayed green automatically. Only two genuine
  pins changed: the `√2` literal in `test_transforms` (renamed
  `test_one_sided_counts_use_default_error`, now expects `1.0`) and the χ²ᵣ
  centring in `TestRefitRecovery` (now `abs(χ²ᵣ − 1.0) < 3√(2/dof)`).
- **Added verification unit tests** (verification-plan items 1–3) in
  `test_transforms`: exact α=1 identity, general-α exact form, and agreement
  with `compute_asymmetry_with_count_errors` at Poisson `√N` count errors.

Verification outcome: `python tools/harness.py validate` green — 2000 passed,
1 xfailed, structural + lint ok. The refit χ²ᵣ band re-centred on 1 passes and
the pull-distribution test passes with the corrected (no longer over-narrow)
errors.
