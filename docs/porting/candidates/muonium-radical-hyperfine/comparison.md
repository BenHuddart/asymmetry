# Muonium-radical hyperfine: comparison

| Aspect | Mantid | musrfit | WiMDA | Asymmetry |
|---|---|---|---|---|
| TF muonium fit functions | ★ four specialised | ◐ via composites | ❌ | ❌ |
| ZF muonium model | ✅ | ◐ | ❌ | ❌ |
| Hyperfine decoupling curve | ✅ | ◐ | ❌ | ❌ |
| FFT diagnostic for pair detection | ❌ | ❌ | ❌ | ❌ (proposed) |
| Reference | `*Muonium*.cpp` | `PTheory` | n/a | n/a |

## Models to port

| Model | Mantid path | Asymmetry placement |
|---|---|---|
| `HighTFMuonium` | `Framework/CurveFitting/src/Functions/HighTFMuonium.cpp` | new component in `core/fitting/composite.py` |
| `LowTFMuonium` | `Framework/CurveFitting/src/Functions/LowTFMuonium.cpp` | new component |
| `TFMuonium` | `Framework/CurveFitting/src/Functions/TFMuonium.cpp` | new component |
| `ZFMuonium` | `Framework/CurveFitting/src/Functions/ZFMuonium.cpp` | new component |
| `MuoniumDecouplingCurve` | `Framework/CurveFitting/src/Functions/MuoniumDecouplingCurve.cpp` | new parametric model in `core/fitting/parameter_models.py` |

Each model is ~50-150 lines of numpy. Parameters are standard
hyperfine constants (A_μ for the muon hyperfine coupling; the
Breit-Rabi energy levels follow analytically).

## Edge cases

- High-field vs low-field regime: the qualitative shape of the
  muonium spectrum changes when the muon-electron hyperfine
  coupling becomes comparable to the external Zeeman energy.
  Asymmetry's existing `_validate_field_strength` style helpers
  can route to the right model.
- Anisotropic muonium (in non-cubic semiconductors): a follow-up
  candidate; defer to Later tier.
