# Verification plan

## Ladder

1. `python tools/harness.py structural` — study layout + `index.json` entry.
2. `python tools/harness.py lint` — Ruff baseline (src/tests/tools).
3. `python tools/harness.py test -- tests/test_sc_vl_lineshape.py` — the new
   anchor + shape + round-trip suite.
4. `python tools/harness.py test -- tests/test_fit_function_docs.py` — the
   component is documented in its category page + `component_docs.py`.
5. `python tools/harness.py validate` — full suite before PR.

## Acceptance criteria

- Second moment of `p(B)` ties to `brandt_field_width_sigma[_powder]`
  (`rel=1e-3`); powder = single-crystal / `sqrt(3)`; rate ∝ `λ^-2`.
- Lineshape skew `> 1` (positively skewed FLL line).
- `R(0)=1`; degenerate `B0≥B_c2` / `λ≤0` / `B_c2≤0` → `R≡1`.
- Synthetic round-trip recovers `λ_ab` within `abs=10 nm`.
- `VortexLattice` / `VortexLatticePowder` registered, params
  `[A, field, phase, lambda_ab, Bc2]`, and compose through `CompositeModel`.
- No Qt / matplotlib / `asymmetry.gui` imports in the new core code.
- Docs: `oscillation.rst` gains a `VortexLattice / VortexLatticePowder` section;
  applicability + references in `component_docs.py`.

## Manual corpus validation (reported in PR, not CI)

Fit the LiFeAs Up/Down (groups 3/4) data with
`VortexLatticePowder * Gaussian + Oscillatory + Constant` and report the fitted
`λ_ab` against 195(2) nm, **with the honest caveat** that the single-run powder
fit is data-degeneracy-limited; the robust recovery uses a normal-state-constrained
nuclear rate or the field-dependence. Recorded in the `wimda-eval` cookbook.

## Risks / open questions (resolved)

- Width vs the existing Brandt convention → calibrate to
  `brandt_field_width_sigma` (option B1) so the toolkit has one `λ` convention.
- Skew lost if modelled as a real envelope → self-contained complex oscillation
  component (option C1).
- Performance in the minimiser → cache the field-distribution build (option D).
- Powder orientation average → `3^{1/4} λ_ab` second-moment approximation,
  consistent with `SC_Brandt_VortexLattice_Powder` (documented in `comparison.md`).
