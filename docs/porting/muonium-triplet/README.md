# Muonium triplet model component

Study-first port of a **symmetry-enforcing muonium triplet** time-domain fit
component: a central diamagnetic Mu⁺ Larmor line plus two Mu⁰ satellites
**symmetrically placed about it**, driven by a **single splitting parameter**
(the hyperfine constant). Follow-on from the link-groups port (PR #27): link
groups let three independent lines *share* amplitude/phase/relaxation but leave
the frequencies free, so the satellite symmetry is only *recovered* from the
data. This component *enforces* the symmetry and exposes the hyperfine coupling
directly as one fitted parameter.

Motivating exercise: the WiMDA Muon School **CdS shallow-donor** problem (TF
100 G). The CdS docx: *"satellite lines appear … symmetrically placed about the
central Mu⁺ line … This frequency splitting is equal to the hyperfine constant."*

## WiMDA reference

`src/Extrafunctions/muoniumfunctions.dpr`:

- **TFMuonium** (transverse field, 4 transitions) and **LowTFMuonium** (2
  transitions): parameters `(Field B [G], A [MHz], Phase [deg])`; derive the
  transition frequencies from the coupled (I=½,S=1) level structure via
  `x = (g_e+g_µ)·B/A`, `E_i = A/4·(…)`, `w_ij = E_i − E_j`.
- **ZFmuonium** (zero-field axial): `f1=A−D, f2=A+D/2, f3=3D/2` — confirmed **not**
  the symmetric `f, f±δ` pattern, so it does not model the TF CdS triplet.
- `src/Plot.pas` `w12(A,field)` / `w34(A,field)` compute the two Mu⁰ satellite
  frequencies straddling the diamagnetic line `ν_d = γ_µ·B`, drawn as reference
  lines in the spectrum viewer.

Key point: WiMDA does **not** ship a "central + 2 symmetric satellites"
parameterisation. The TF muonium functions emit the QM transition set; a user
fitting the CdS triplet either uses those or builds three independent
oscillations. Our component is a phenomenological, user-facing
`(centre, splitting)` parameterisation of the *observed* TF structure — see
[comparison.md](comparison.md).

## Chosen design — full WiMDA parity

Port WiMDA's muonium oscillation functions **faithfully**, mapped onto Asymmetry
conventions, rather than inventing a phenomenological `(centre, splitting)`
triplet. Three new baseline-free, **undamped** components (damping/background by
composition, exactly like `Oscillatory`):

| Component | WiMDA fn | Params (Asymmetry conventions) | Lines |
| --- | --- | --- | --- |
| `MuoniumTF` | `TFMuonium` | `A`, `field` (B, G), `A_hf` (hyperfine, MHz), `phase` (rad) | 4 transitions |
| `MuoniumLowTF` | `LowTFMuonium` | `A`, `field`, `A_hf`, `phase` | 2 transitions |
| `MuoniumZF` | `ZFmuonium` | `A`, `A_hf`, `D`, `f_cut` (MHz), `phase` | 3 (zero-field axial) |

**Why this is faithful *and* gives the symmetric triplet for free.** Numerically
validating `TFMuonium` at the CdS regime (B = 100 G, A_hf = 0.242 MHz):

- the two in-band transitions `|w12| = 1.234`, `|w34| = 1.476 MHz` straddle the
  diamagnetic line `ν_d = γ_µ·B = 1.355 MHz` **symmetrically**, split = 0.2420
  MHz ≈ the hyperfine coupling;
- the other two transitions sit at ~280 MHz with weight `(1−δ) → 0` (since
  `δ = x/√(1+x²) → 1` for the shallow-donor large-`x` limit), so they
  **auto-suppress**.

So porting `TFMuonium` verbatim *is* the symmetry-enforcing triplet, with the
hyperfine coupling `A_hf` as the single physical splitting parameter — no
phenomenological shortcut needed.

**Conventions kept consistent with Asymmetry:**

- leading amplitude `A` (engages the standard chain-amplitude sharing, so
  `MuoniumTF * Exponential` collapses to one amplitude and damps — exactly like
  `Oscillatory`); the hyperfine is `A_hf` (distinct from `A`/`A_bg`, so it is
  *not* treated as a scaling param);
- the **central diamagnetic Mu⁺ line stays a separate `OscillatoryField`** (it is
  a different species; WiMDA's muonium functions also exclude it);
- **phase in radians** (Asymmetry) rather than WiMDA's degrees;
- g-factors from existing constants: `g_µ = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T ·
  GAUSS_TO_TESLA`, `g_e = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / 2π`;
- module-level, picklable component functions (batch/global parallelism).

CdS model: `OscillatoryField*Exponential + MuoniumTF*Exponential + Constant` —
central diamagnetic line + the muonium satellites, each damped, plus background.
`A_hf` reads off the hyperfine constant directly; the muonium-component amplitude
trends as the Mu⁰ fraction for the Arrhenius/ionisation-energy plot.

Complements (does not replace) link groups: three independent
`Oscillatory*Exponential` lines + link groups remain the flexible, model-free
option; the muonium components are the physics-faithful, hyperfine-parameterised
option.

See [comparison.md](comparison.md) for the WiMDA physics and the validation,
[implementation-options.md](implementation-options.md) for the exact seams,
[test-data.md](test-data.md) for the synthetic fixture, and
[verification-plan.md](verification-plan.md) for the CdS acceptance bar.

**Status: STUDY (design agreed: full WiMDA parity) — implementation in progress.**
