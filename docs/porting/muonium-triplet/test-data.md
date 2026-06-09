# Muonium components — test data

## Synthetic fixtures (in-repo, no corpus dependency)

Built from the ported component functions themselves, so the suite never depends
on the WiMDA corpus.

### TF muonium (the CdS-relevant case)

`MuoniumTF` at `field = 100` G, `A_hf = 0.242` MHz, `A = 1`, `phase = 0`. The
reference behaviour (validated against the WiMDA arithmetic):

- diamagnetic line `ν_d = g_µ·100 ≈ 1.3553` MHz (modelled separately by
  `OscillatoryField`),
- in-band satellites at `≈ 1.234` and `≈ 1.476` MHz, symmetric about `ν_d`,
  separation `≈ 0.242` MHz = `A_hf`,
- the two extra transitions (~280 MHz) carry weight `(1−δ) ≈ 0`.

Full synthetic CdS-like signal for the round-trip test:

```
A(t) = e^(−λ t)·A_d·cos(2π·γ_µ·B·t + φ)            # central (OscillatoryField)
     + e^(−λ t)·MuoniumTF(t; A_s, B, A_hf, φ)        # satellites
     + bg
```

with `B = 100` G, `A_hf = 0.242` MHz, `λ = 0.30` µs⁻¹, `φ = 0`, sensible
amplitudes, flat `bg`, t-grid 0–12 µs, per-point error ~0.15.

### ZF muonium

`MuoniumZF` at `A_hf, D, f_cut` chosen so `f1=A_hf−D`, `f2=A_hf+D/2`, `f3=1.5D`
are distinct and in band; assert the three frequencies and the `f_cut` Lorentzian
amplitude weighting match the WiMDA formula.

## What the synthetic tests assert

1. **Frequencies match WiMDA arithmetic**: `MuoniumTF` in-band lines straddle
   `ν_d` with separation `A_hf`; the out-of-band pair carries weight `(1−δ)≈0`.
   `MuoniumLowTF` gives its two lines (with the `−w` sign). `MuoniumZF` gives
   `f1,f2,f3` with the `a1,a2,a3`/`f_cut` weights.
2. **Single-parameter splitting**: varying `A_hf` moves the satellites
   symmetrically about `ν_d`; varying `field` shifts them with the diamagnetic
   line.
3. **Round-trip recovery**: fitting the synthetic CdS-like signal with
   `OscillatoryField*Exponential + MuoniumTF*Exponential + Constant` recovers
   `A_hf ≈ 0.242` MHz and `λ`, χ²ᵣ ≈ 1, with `A_hf` read directly.
4. **Composite/registry integration**: each component builds via
   `CompositeModel.from_expression(...)` with the expected `param_names`, appears
   in the builder category map, and renders its equation/applicability.
5. **`.asymp` round-trip** of a model containing a muonium component.
6. **Picklability**: the component functions are module-level callables.

## CdS real-data acceptance (corpus, not committed)

Run EMU00020721 (≈5.12 K, TF 100 G), Data_hdf5 copy. Used only for the
manual/engine acceptance in verification-plan.md; never committed, never imported
by the suite.
