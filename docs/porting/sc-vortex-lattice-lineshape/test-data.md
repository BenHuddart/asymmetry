# Test data

## Synthetic (CI — primary proof)

The lineshape is its own generator, so the round-trip is self-contained and
needs no corpus binaries:

1. **Analytic anchors.** For several `(λ, B0, B_c2)`, the second moment of the
   sampled `p(B)` (converted to a rate) must equal
   `brandt_field_width_sigma[_powder]` to `rel=1e-3` (calibration is exact by
   construction). The powder rate must be the single-crystal rate divided by
   `sqrt(3)` (`3^{1/4}` length factor). The rate must scale as `λ^-2`.
2. **Shape.** Skew of `p(B)` is positive and `> 1` (sharp low cutoff, high tail).
3. **Relaxation.** `R(0) = 1`; `|R|` decays; `B0 ≥ B_c2` / `λ ≤ 0` → `R ≡ 1`.
4. **Round-trip.** Build `VortexLatticePowder * Gaussian(nuclear) + Oscillatory
   (background) + Constant + noise` with known `λ_ab = 195 nm`, fit with
   `scipy`, recover `λ` within `abs=10 nm`. Seeded RNG for determinism.

These live in `tests/test_sc_vl_lineshape.py` and run in CI (~7 s).

## Corpus (manual — reported in the PR / cookbook, not CI)

LiFeAs PSI `.bin` GPS data,
`wimda-corpus/Superconductivity/LiFeAs/` (Pratt 2009). No real binaries are
committed (corpus policy). Reference targets in `LiFeAs/GROUND_TRUTH.md`:

- Sample 1 (LFA, T_c 16 K): `λ_ab = 195(2) nm`; B_rms(40 mT) plateau ≈ 1.8–1.93 mT.
- 40 mT T-scan runs 3366–3373 (Up/Down detector pair, groups 3/4); field sweep
  at 2 K / 20 K, runs 3375–3387.

Manual validation is documented in the `wimda-eval` cookbook
(`docs/testing/reports/superconductivity/lifeas/`). Honest outcome: the lineshape
recovers the correct second moment and is window-stable on synthetic data; on the
LiFeAs *powder* run the absolute `λ_ab` remains data-degeneracy-limited (fast VL
⊗ slow nuclear ⊗ persistent Ag background overlap in a single run), so the
headline 195 nm needs the normal-state-constrained or field-dependence procedure,
not a single-run lineshape fit.
