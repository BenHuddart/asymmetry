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

## Chosen design (for sign-off)

A self-contained, **additively-used** damped triplet component
**`MuoniumTriplet`** (used as `MuoniumTriplet + Constant`):

```
A(t) = e^(−λ t) · [ A_c·cos(2π f₀ t + φ)
                   + A_s·cos(2π (f₀ − Δ/2) t + φ)
                   + A_s·cos(2π (f₀ + Δ/2) t + φ) ]
```

Parameters (6): `A_centre`, `A_sat`, `f_centre` (MHz), `hyperfine` Δ = full
satellite splitting = the hyperfine constant (MHz, satellites at f₀ ± Δ/2),
`Lambda` (shared relaxation, µs⁻¹), `phase` (shared, rad).

Why self-contained and additive: `_is_scaling_parameter` treats only the exact
names `A`/`A_bg` as collapsible chain amplitudes, so distinct names
(`A_centre`/`A_sat`) sidestep the multiplicative amplitude-sharing machinery;
built-in damping and phase avoid a `* Exponential` chain that would double the
amplitude. The satellite amplitude `A_sat` is a first-class fitted parameter, so
the CdS **Arrhenius trend of the Mu⁰ satellite amplitude vs T** drops out of the
batch/trend workflow directly.

Complements (does not replace) link groups: three independent
`Oscillatory*Exponential` lines + link groups remain the flexible option (free
frequencies, per-line phases); `MuoniumTriplet` is the constrained,
symmetry-enforcing option with the hyperfine coupling as one parameter.

See [implementation-options.md](implementation-options.md) for the exact seams,
[test-data.md](test-data.md) for the synthetic fixture, and
[verification-plan.md](verification-plan.md) for the CdS acceptance bar.

**Status: STUDY — awaiting design sign-off before implementation.**
