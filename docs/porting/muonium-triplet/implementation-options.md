# Muonium components — implementation options & chosen design

## Option 1 — three independent lines + link groups (already shipped, NOT this)

`Oscillatory*Exponential ×3 + Constant` with link groups sharing
amplitude/phase/relaxation; frequencies free, symmetry recovered not enforced.
The flexible, model-free baseline. The muonium components are the
physics-faithful complement, not a replacement.

## Option 2 — phenomenological `(centre, splitting)` triplet (rejected)

A single self-contained `MuoniumTriplet(A_centre, A_sat, f_centre, hyperfine,
Lambda, phase)`. Rejected in favour of full WiMDA parity: the faithful
`TFMuonium` already reduces to the symmetric pair (see comparison.md), so the
phenomenological shortcut buys nothing and is less faithful.

## Option 3 — full WiMDA parity (CHOSEN)

Port WiMDA's three muonium oscillation functions as **undamped, baseline-free**
components, composed with `* Exponential` / `+ Constant` like `Oscillatory`.

### Functions (top-level, picklable)

In a new module `src/asymmetry/core/fitting/muonium.py`. g-factors from
`asymmetry.core.utils.constants`:
`G_MU = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA` (MHz/G),
`G_E = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / (2*pi)` (MHz/G).

Three private functions evaluate the forms in [comparison.md](comparison.md):

- `_muonium_tf(t, A, field, A_hf, phase)`: the four-line TF signal, amplitude
  `A`, with `phase` added to each cosine.
- `_muonium_low_tf(t, A, field, A_hf, phase)`: the ν₁₂ and ν₂₃ lines only, at the
  same weights and with negative frequencies, following WiMDA `LowTFMuonium`.
- `_muonium_zf(t, A, A_hf, D, f_cut, phase)`: three cosines at `A_hf − D`,
  `A_hf + D/2` and `3D/2` with weights 1, 2, 2 over 6, each weight multiplied by
  the low-pass factor `1/(1 + (f/f_cut)²)` when `f_cut > 0`, following WiMDA
  `ZFmuonium`.

Notes: frequency arithmetic, `(1±delta)` weights, the `-w` sign in LowTF, and the
ZF `/6` and `f_cut` Lorentzian are ported verbatim. `A` is the standard leading
amplitude; `phase` is radians (WiMDA degrees adapted). Guard `A_hf > 0` /
`field`-domain validity by returning a finite penalty (as `_general_fmuf_component`
does) so the minimiser never sees NaN at trial points.

### Registry & metadata

- Add three `ComponentDefinition` entries to `COMPONENTS` in `composite.py`
  (`category="Muonium"`, `domain="time"`), each with `param_names`,
  `param_defaults`, `param_info`, `formula_template`, `latex_equation`.
- `parameters.py` `PARAM_INFO_REGISTRY`: reuse `A`, `field`, `phase`; **add**
  `A_hf` (hyperfine, MHz, `default_min=0.0`, label Aµ), `D` (anisotropy, MHz),
  `f_cut` (cutoff, MHz, `default_min=0.0`). Names are distinct from `A`/`A_bg` so
  they are not folded into chain-amplitude logic.
- `component_docs.py` `FIT_COMPONENT_APPLICABILITY`: a blurb per component.

### Defaults / seeding

`MuoniumTF`/`MuoniumLowTF`: `A=10, field=100, A_hf=0.24, phase=0` (`field`
auto-seeds from run metadata like other field params; fix it when B is known).
`MuoniumZF`: `A=10, A_hf=1.0, D=0.5, f_cut=0.0, phase=0`. Seed `A_hf` away from 0
(at `A_hf→0`, `x→∞` and the split collapses — same seed-trap class as
OrderParameter `Tc`).

### No GUI code changes

The builder dropdown groups by `category` automatically and the Component-Info
dialog renders `latex_equation` + `param_info`; the components are also reachable
by typed expression (`OscillatoryField*Exponential + MuoniumTF*Exponential +
Constant`). `.asymp` round-trips via the existing model/param dict path.

### CdS model

`OscillatoryField*Exponential + MuoniumTF*Exponential + Constant`: central
diamagnetic line + muonium satellites (each damped) + background. `field` fixed
at the run field (100 G); `A_hf` is the fitted hyperfine constant; the
muonium-component amplitude is the Mu⁰ fraction for the Arrhenius trend.

### Decisions taken

- Port all three WiMDA functions (full parity), not just TF.
- Central diamagnetic line kept as a separate `OscillatoryField` (faithful).
- Undamped components + composition for damping (consistent with `Oscillatory`);
  do **not** bake in relaxation.
- `A_hf` (not WiMDA's `A`) for the hyperfine, to avoid colliding with the
  amplitude convention; `phase` in radians.
