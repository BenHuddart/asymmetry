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

## The CdS regime, and why a (centre, splitting) parameterisation

CdS is a **shallow-donor** muonium: the hyperfine coupling is tiny
(`A_µ ≪ 4463 MHz`; the engine link-groups fit found `2δ ≈ 0.242 MHz`). At TF
100 G the diamagnetic line sits at `ν_d ≈ gm·100 ≈ 1.355 MHz` and the two Mu⁰
satellites straddle it symmetrically at `ν_d ± (A_µ/2)` to leading order. The
docx states the splitting *equals* the hyperfine constant.

The full WiMDA TFMuonium 4-level expression is overkill and field-coupled
(`x = (ge+gm)·B/A`); for the shallow-donor small-A regime the observable is
simply **a central line + two symmetric satellites**. So rather than port the
4-transition QM, the component is parameterised by what the experiment measures:

- `f_centre` — the central (≈ diamagnetic) frequency,
- `hyperfine` Δ — the **full** satellite splitting `f₊ − f₋`, i.e. the hyperfine
  constant; satellites at `f_centre ± Δ/2`.

This makes the fitted parameter the physical observable, and enforces the
symmetry the docx describes (which free-frequency link-group fits only recover).

## Mapping to / divergence from WiMDA

| Aspect | WiMDA TFMuonium | This component |
| --- | --- | --- |
| Inputs | Field B, hyperfine A | centre f₀, splitting Δ (observed) |
| Lines | 4 transitions (or 2) | central + 2 satellites (3) |
| Symmetry | from the level structure | enforced: f₀, f₀±Δ/2 |
| Damping/phase | external | built-in (shared λ, shared φ) |
| Amplitudes | (1±delta) weights | free `A_centre`, `A_sat` |
| Hyperfine read-out | post-fit from w_ij | `hyperfine` Δ directly |

Divergences are deliberate and documented: (a) we model the *observed* 3-line TF
structure, not the 4-level QM, because the shallow-donor small-A limit makes the
extra TFMuonium lines negligible and the symmetric pair the physical picture;
(b) damping/phase are built in so the component is usable as a single additive
term; (c) amplitudes are free so `A_sat` can be trended (Arrhenius vs T) — the
actual CdS deliverable. A faithful 4-level `TFMuonium` (field-parameterised) is
recorded as a possible future addition for high-field/strong-coupling muonium,
but is not what the CdS exercise needs.
