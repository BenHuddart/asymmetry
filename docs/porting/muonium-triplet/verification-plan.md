# Muonium triplet — verification plan

## Automated (in-repo, CI)

Synthetic damped triplet (see test-data.md), in `tests/test_muonium_triplet.py`
(component/engine) and an addition to `tests/test_composite_model.py`
(registry/builder integration):

1. Three lines at `f₀`, `f₀ ± Δ/2`, symmetric about `f₀` (direct + FFT-peak check).
2. `hyperfine` moves both satellites symmetrically; `f_centre` shifts all three.
3. Fit recovers `f_centre`, `hyperfine`, shared `λ`/`φ` at χ²ᵣ ≈ 1 with fewer
   free params than three independent lines; hyperfine read directly off the fit.
4. `CompositeModel.from_expression("MuoniumTriplet + Constant")` builds with the
   expected `param_names`; component appears in the builder category map.
5. `.asymp` save/load round-trips the model and parameters.
6. Component function is module-level / picklable.

Run: `python tools/harness.py validate` from the worktree venv.

## Manual acceptance (CdS real data, corpus only)

Fit EMU00020721 (≈5.12 K, TF 100 G) with `MuoniumTriplet + Constant`, window
0.1–10 µs. Pass criteria:

- converged, χ²ᵣ ≈ 1.3
- `f_centre ≈ 1.389` MHz
- `hyperfine ≈ 0.242` MHz (the hyperfine constant, read directly)
- satellites symmetric by construction; **strictly fewer free parameters** than
  the three-independent-line link-group fit, with no worse χ²ᵣ

Plus a smoke check of the CdS deliverable: a batch fit over the T-series trends
`A_sat` (Mu⁰ satellite amplitude) vs T for the Arrhenius/ionisation-energy plot.

## Outcome

_Filled in during the implementation pass._
