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
| `MuoniumZF` | `ZFmuonium` | `A`, `A_hf`, `D_mu`, `f_cut` (MHz), `phase` | 3 (zero-field axial) |

**Scope (decided after verification): a general muonium-spectroscopy feature,
not the CdS tool.** Verification showed that faithful `TFMuonium` does **not**
meet a CdS χ²≈1.3 bar — see comparison.md — and WiMDA's own CdS docx routes that
shallow-donor case to *three independent oscillating lines* (= Asymmetry's link
groups, PR #27), which is where CdS stays. These components target *genuine*
muonium, where all transitions and their `(1±δ)` weights matter; they are
verified by self-consistency (generate from the component, recover the
parameters) and against the WiMDA arithmetic, not against the CdS corpus.

**Positive-frequency (same-phase) convention.** WiMDA's `TFMuonium` uses the
signed `w12` (negative), placing the lower satellite at phase `−φ`, while its own
`LowTFMuonium` negates `w12` to make it positive — an internal inconsistency.
Physically the precession lines share one phase, so the port uses `|w|` (positive
frequencies, `+φ` for all lines). This is a deliberate, documented deviation from
`TFMuonium`'s literal signed form.

**Behaviour in the shallow-donor limit.** Numerically validating `TFMuonium` at
the CdS regime (B = 100 G, A_hf = 0.242 MHz):

- the two in-band transitions `|w12| = 1.234`, `|w34| = 1.476 MHz` straddle the
  diamagnetic line `ν_d = γ_µ·B = 1.355 MHz` symmetrically, split = 0.2420
  MHz ≈ the hyperfine coupling;
- the other two transitions sit at ~280 MHz with weight `(1−δ) → 0` (since
  `δ = x/√(1+x²) → 1`), so they auto-suppress.

The line *positions* are therefore correct, but the constrained `(field, A_hf)`
parameterisation is over-determined relative to three independent frequencies,
and (with the literal signed phase) mis-phases the lower satellite — which is why
the shallow-donor CdS fit is better served by independent lines + link groups
(see comparison.md). These components shine for genuine muonium where the four
transitions and their weights are physically present.

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

Usage: compose like any oscillation, e.g.
`MuoniumTF*Exponential + Constant` (or add an `OscillatoryField` term for a
co-existing diamagnetic line). `A_hf` reads off the hyperfine coupling directly.

Complements (does not replace) link groups: three independent
`Oscillatory*Exponential` lines + link groups remain the flexible, model-free
option (and the right one for shallow-donor CdS); the muonium components are the
physics-faithful, hyperfine-parameterised option for genuine muonium.

See [comparison.md](comparison.md) for the WiMDA physics and the verification
finding, [implementation-options.md](implementation-options.md) for the exact
seams, [test-data.md](test-data.md) for the synthetic fixtures, and
[verification-plan.md](verification-plan.md) for the verification done.

**Status: IMPLEMENTED — faithful WiMDA muonium components (positive-frequency
convention), shipped as a general muonium feature; CdS remains served by link
groups.**
