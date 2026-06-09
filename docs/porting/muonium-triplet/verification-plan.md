# Muonium components — verification plan

These components are scoped to **genuine muonium** and verified by
self-consistency and against the WiMDA arithmetic — **not** against the CdS
corpus (shallow-donor CdS is fit with three independent lines + link groups; see
comparison.md).

## Automated (in-repo, CI) — `tests/test_muonium.py`

1. g-factors match the WiMDA literals (`gm`, `ge`) to ~5e-7.
2. `MuoniumTF` in-band transitions straddle `ν_d = γ_µ·B`, symmetric about it,
   separation = `A_hf`; the out-of-band pair carries weight `(1−δ) ≈ 0`.
3. Positive-frequency convention: every line shares `+φ` (the normalised sum is
   `cos(φ)` at `t = 0`).
4. `MuoniumLowTF` has two transitions (one in band, one far out). `MuoniumZF`
   line frequencies `f1=A_hf−D_mu, f2=A_hf+D_mu/2, f3=3D_mu/2` and the `f_cut`
   Lorentzian amplitude weights.
5. `A_hf` moves the satellites symmetrically about the Larmor line.
6. **Self-consistency round-trip**: generate a noisy signal from
   `MuoniumTF*Exponential + Constant` (genuine muonium, well-separated
   satellites) and recover `A_hf` (and `field`, `Lambda`) at χ²ᵣ ≈ 1.
7. Registry/builder integration: the three components register with the expected
   `param_names`, sit under a `Muonium` category, and appear in the builder
   category map; `OscillatoryField*Exponential + MuoniumTF*Exponential +
   Constant` builds with `A_hf` exposed.
8. A model containing a muonium component round-trips through
   `CompositeModel.to_dict`/`from_dict`.
9. The component functions are module-level / picklable.

Run: `python tools/harness.py validate` from the worktree venv.

## Outcome

Implemented: `src/asymmetry/core/fitting/muonium.py` (g-factors from existing
constants; `tf_muonium` / `low_tf_muonium` / `zf_muonium`, positive-frequency
convention) + three `ComponentDefinition`s (`MuoniumTF` / `MuoniumLowTF` /
`MuoniumZF`, category `Muonium`), `A_hf` / `D_mu` / `f_cut` parameter metadata,
and applicability docs. `tests/test_muonium.py` (12 tests) green.

Self-consistency verified engine-side: a signal built from `MuoniumTF` at
`field = 100 G`, `A_hf = 2.0 MHz` is recovered at χ²ᵣ = 1.01,
`A_hf = 2.0001 ± 0.0002 MHz`.

Recorded non-goal: these components do **not** meet a CdS χ²≈1.3 bar (χ²ᵣ ≈ 22
vs 1.35 for independent lines), by design — CdS is served by link groups, per
WiMDA's own guidance.
