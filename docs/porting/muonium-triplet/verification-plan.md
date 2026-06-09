# Muonium components — verification plan

## Automated (in-repo, CI)

Synthetic fixtures (see test-data.md), in `tests/test_muonium.py`
(component/engine) and additions to `tests/test_composite_model.py`
(registry/builder integration):

1. `MuoniumTF` frequencies/weights match the WiMDA arithmetic: in-band lines
   straddle `ν_d = γ_µ·B` with separation `A_hf`; out-of-band pair weight
   `(1−δ) ≈ 0`. `MuoniumLowTF` two lines (with `−w` sign). `MuoniumZF`
   `f1,f2,f3` + `a1,a2,a3`/`f_cut` weights.
2. `A_hf` moves the satellites symmetrically; `field` shifts them with the
   diamagnetic line.
3. Round-trip: `OscillatoryField*Exponential + MuoniumTF*Exponential + Constant`
   recovers `A_hf ≈ 0.242` MHz and `λ` at χ²ᵣ ≈ 1; hyperfine read directly.
4. `CompositeModel.from_expression` builds each component with the expected
   `param_names`; components appear in the builder category map.
5. `.asymp` save/load round-trips a model containing a muonium component.
6. Component functions are module-level / picklable.

Run: `python tools/harness.py validate` from the worktree venv.

## Manual acceptance (CdS real data, corpus only)

Fit EMU00020721 (≈5.12 K, TF 100 G) with
`OscillatoryField*Exponential + MuoniumTF*Exponential + Constant`, `field` fixed
at 100 G, window 0.1–10 µs. Pass criteria:

- converged, χ²ᵣ ≈ 1.3
- central diamagnetic line at `≈ 1.355` MHz (γ_µ·100 G)
- `A_hf ≈ 0.242` MHz (the hyperfine constant, read directly)
- satellites symmetric by construction; no worse χ²ᵣ than the three-independent-
  line link-group fit, with `A_hf` as a single physical parameter

Plus a smoke check of the CdS deliverable: a batch fit over the T-series trends
the muonium-component amplitude (Mu⁰ fraction) vs T for the Arrhenius/ionisation-
energy plot.

## Outcome

_Filled in during the implementation pass._
