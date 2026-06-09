# Muonium triplet — WiMDA physics vs. the chosen parameterisation

## WiMDA TF muonium (verified in source)

`src/Extrafunctions/muoniumfunctions.dpr`, constants `gm = 0.01355342`,
`ge = 2.8024` (dimensionless g-factors).

**TFMuonium** (lines ~58–78), params `(B [G], A [MHz], Phase [deg])`:

```
x      = (ge + gm)·B / A
d      = (ge − gm)/(ge + gm) ≈ 0.99037
delta  = x / sqrt(1 + x²)
E1=A/4·(1+2d·x)   E2=A/4·(−1+2√(1+x²))
E3=A/4·(1−2d·x)   E4=A/4·(−1−2√(1+x²))
w12=E1−E2  w14=E1−E4  w23=E2−E3  w34=E3−E4
signal = ¼·[ (1+delta)·cos(2π w12 t) + (1−delta)·cos(2π w14 t)
            +(1+delta)·cos(2π w34 t) + (1−delta)·cos(2π w23 t) ]   (phase in deg/360)
```

**LowTFMuonium** keeps only `w12`,`w23`. **ZFmuonium** is zero-field axial
(`f1=A−D, f2=A+D/2, f3=3D/2`) and does **not** produce symmetric satellites.

`src/Plot.pas` exposes the satellite pair directly:

```
w12(A,B) = A/2·(1 + d·x − √(1+x²))
w34(A,B) = A/2·(1 − d·x + √(1+x²))      x = (ge+gm)·B/A
```

drawn as reference lines straddling the diamagnetic line `ν_d = γ_µ·B = gm·B`.

**Conventions:** frequency MHz, time µs, the `2π` factor explicit; WiMDA phase in
degrees (`(deg)/360·2π`). The free-muonium hyperfine `A ≈ 4463 MHz` is *not*
hardcoded — it is the fitted `A`.

## The CdS regime — faithful TFMuonium auto-reduces to the symmetric pair

CdS is a **shallow-donor** muonium: the hyperfine coupling is tiny
(`A_µ ≪ 4463 MHz`; the link-groups fit found the splitting `≈ 0.242 MHz`). The
key realisation is that the **faithful** WiMDA `TFMuonium` already produces
exactly the observed structure in this regime — no phenomenological
re-parameterisation is needed.

Validating `TFMuonium` numerically at B = 100 G, A_hf = 0.242 MHz
(`x = (ge+gm)·B/A_hf ≈ 1163`, `δ = x/√(1+x²) ≈ 1`):

```
ν_d = g_µ·B            = 1.3553 MHz   (central diamagnetic, separate component)
w12 = −1.2344 MHz  weight (1+δ)=2.000   in band
w34 = +1.4764 MHz  weight (1+δ)=2.000   in band   → |w12|,|w34| straddle ν_d
w14 = +280.36 MHz  weight (1−δ)=0.000   suppressed
w23 = +280.12 MHz  weight (1−δ)=0.000   suppressed
|w34| − |w12| = 0.2420 MHz  ≈ A_hf  (the hyperfine constant)
```

So the two in-band satellites straddle `ν_d` symmetrically with separation equal
to the hyperfine coupling, and the two extra transitions vanish via the `(1−δ)`
weight. Porting `TFMuonium` verbatim therefore *is* the symmetry-enforcing
triplet, with `A_hf` the single splitting parameter — while remaining fully
faithful to WiMDA (including the high-field/strong-coupling case where all four
lines matter).

## Mapping WiMDA → Asymmetry (full parity)

| Aspect | WiMDA | Asymmetry port |
| --- | --- | --- |
| Functions | TFMuonium / LowTFMuonium / ZFmuonium | `MuoniumTF` / `MuoniumLowTF` / `MuoniumZF` |
| Hyperfine input `A` | param `A` (MHz) | `A_hf` (renamed; `A` is the amplitude) |
| Field `B` | param (Gauss) | `field` (Gauss), via existing γ_µ constants |
| Amplitude | external scale | leading `A` (standard chain-amplitude param) |
| Damping | external | by composition (`* Exponential`) |
| Phase | degrees `(deg)/360·2π` | `phase` in radians |
| Central diamagnetic line | separate oscillation | separate `OscillatoryField` |
| g-factors | `gm=0.01355342`, `ge=2.8024` | from `MUON_…·GAUSS_TO_TESLA`, `ELECTRON_…/2π` |
| Frequency weights `(1±δ)`, ZF `f_cut` Lorentzian | as in source | ported verbatim |

Deliberate, documented divergences: amplitude named `A` and applied as the
standard leading scale; damping/phase via composition rather than baked in; phase
in radians; ZF anisotropy named `D_mu` (the bare `D` is already a Redfield
parameter). The frequency arithmetic and amplitude weights are ported exactly.
The central diamagnetic Mu⁺ line is a separate `OscillatoryField`, matching WiMDA.

## Two findings from verification

**1. Frequency sign / phase (positive-frequency correction).** `TFMuonium` uses
the *signed* `w12` (negative), so `cos(2π w12 t + φ)` puts the lower satellite at
`−φ`; yet `LowTFMuonium` *negates* `w12` to make it positive — WiMDA is
internally inconsistent. Physically the precession lines share one phase, so the
port uses `|w|` (positive, `+φ` for all lines). With the literal signed form a
nonzero φ flips the lower satellite and badly mis-fits same-phase data.

**2. Why CdS is *not* fit with these components.** Even with the positive-phase
correction, fitting the real CdS 5.12 K run with
`OscillatoryField*Exponential + MuoniumTF*Exponential + Constant` tops out at
χ²ᵣ ≈ 22, whereas three independent damped lines reach χ²ᵣ ≈ 1.35. The cause:
the `(field, A_hf)` parameterisation ties the central line and both satellites to
two numbers (symmetric about `γ_µ·B`), which is over-determined when the data's
lines are not *exactly* symmetric — and the χ² landscape is razor-sharp (the
three-line model itself jumps to χ²ᵣ ≈ 22 if its parameters are merely rounded to
four decimals). This matches WiMDA's own CdS docx, which says to fit *three
independent oscillating functions* (= link groups), not `TFMuonium`. These
components are therefore scoped to **genuine muonium** and verified by
self-consistency + the WiMDA arithmetic, not against the CdS corpus.
