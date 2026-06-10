# Comparison: WiMDA Model-layer functions & machinery vs Asymmetry

Date: 2026-06-10. WiMDA source: `/Users/bhuddart/Source/WiMDA/src` (read
directly; `__history/`/`__recovery/` ignored). Asymmetry at `main` 19f242b.

Every formula below was transcribed from the Pascal, not lifted from the
umbrella brief. Where the brief and the source disagree in detail, the source
wins and the delta is noted.

## 1. Function-by-function transcription (`fitfunctions.pas`)

WiMDA constants (line 17–19): `gmu2 = 0.01355342` MHz/G, `gel = 2π·2.8024`
rad·MHz/G — declared but **unused** by the built-ins (B₀ in `muonrep` is a
direct fit parameter).

### 1.1 Polynomial fit up to fifth order (`func0`, 100–108)

```
y = p1 + p2·x + p3·x² + p4·x³ + p5·x⁴ + p6·x⁵
```

Plain monomial basis, all six coefficients always present (fix unused ones at
0). Asymmetry: ABSENT (only `Constant` and `Linear`). Port as `Polynomial`
with parameters c0…c5, defaults c0=0, c1=1, rest 0, scope `common`.
`Constant`/`Linear` remain as the simple cases (no deprecation).

### 1.2 Power law (`powerlaw`, 87–90)

```
y = p1·x^p2 + p3        (Delphi power(); negative x with non-integer n faults)
```

Asymmetry `PowerLaw`: `y = a·|x|^n + c` with `|x|` floored at 1e-12
(`parameter_models.py:60`). **Divergence (kept)**: WiMDA evaluates `x^n`
directly and can fault/NaN for x<0, non-integer n; Asymmetry mirrors via
|x| and never faults. Identical on x > 0, which is the physical domain
(field, temperature).

### 1.3 Power law, BG in quadrature (`powerlawBGquad`, 93–97)

```
y = sqrt( (p1·x^p2)² + p3² )
```

Asymmetry: ABSENT — composite algebra has sums and products but no √ of a
sum, so it is not expressible. Port as dedicated `PowerLawQuadBG`
(a, n, BG; scope `common`), evaluating `hypot(a·|x|^n, BG)` (same |x| guard
as `PowerLaw`; `hypot` for overflow safety). Used for e.g. λ(T) where a
T-independent background rate adds in quadrature to a power-law term.

**Deferred design note — generic quadrature combinator.** The general want is
`y = sqrt(f² + g²)` for arbitrary registry components f, g (an `⊕` operator
in the composite grammar alongside `+` and `*`). That is a composite-syntax
extension touching the expression parser, GLE export and the builder dialog —
out of proportion for one function. Recorded for a future composite-algebra
pass; `PowerLawQuadBG` covers the only WiMDA instance.

### 1.4 2 Lorentzians + cubic BG (`func3`, 111–126)

```
L1 = p1·p3² / (p3² + (x−p2)²)        (skipped entirely if p1 = 0 exactly)
L2 = p4·p6² / (p6² + (x−p5)²)        (skipped entirely if p4 = 0 exactly)
BG = p7 + p8·(x−p2) + p9·(x−p2)² + p10·(x−p2)³     ← centred on Pos 1!
y  = L1 + L2 + BG
```

Each Lorentzian is identical to Asymmetry's `LorentzianLCR`
(`f / (1 + ((B−B0)/Bwid)²)` — algebraically equal with f=Ampl, B0=Pos,
Bwid=Wid). Asymmetry recipe: `LorentzianLCR + LorentzianLCR + Polynomial`.

**Divergences (documented, not replicated):**

- *BG centring*: WiMDA's cubic is in powers of (x − Pos 1); Asymmetry's
  `Polynomial` is in powers of absolute x. The model spaces are identical
  (any cubic in (x−a) is a cubic in x) but **fitted BG coefficients do not
  transfer 1:1** between programs, and WiMDA's BG shifts whenever Pos 1
  moves during the fit. Asymmetry's absolute-x form decouples background
  from peak position — preferred.
- *Exact-zero amplitude test*: WiMDA returns L=0 only when the amplitude is
  exactly 0.0 (a float equality used to "switch off" a peak). Asymmetry
  expresses one-peak fits by just not adding the second component; no
  magic value.

### 1.5 Thermal activation, 2 components (`func2`, 129–142)

```
c = 1.60e-19 / 1.38e-23  (= e/k_B with 3-significant-figure constants)
y = p1·exp(−p2·c/x) + p3·exp(−p4·c/x)     for x > 0;  y = 0 for x ≤ 0
```

(`ex()` is a guarded exp that returns 0 for |arg| > 750.)

Asymmetry: EXPRESSIBLE as `Arrhenius + Arrhenius`
(`y = a·exp(−Eₐ/k_B T)`, Eₐ in **meV**, CODATA
k_B = 8.617333262e-2 meV/K, scope `temperature`). Docs recipe, not code.

**Divergences (documented):**

- *Units*: WiMDA Eₐ in eV; Asymmetry in meV. Conversion: Eₐ[meV] =
  1000 × Eₐ[eV].
- *Constant accuracy*: WiMDA's e/k = 11594.20 K/eV vs CODATA
  11604.52 K/eV — WiMDA activation energies are systematically
  **0.089 % low** (the umbrella brief said ≈0.3 %; the verified figure is
  0.089 %). Asymmetry keeps CODATA.
- *Low-T guard*: WiMDA returns 0 for x ≤ 0; Asymmetry floors |T| at 1e-9 K
  (`parameter_models.py:79`) — both give ~0 in any physical fit window.

### 1.6 Internal field vs T for ordered magnet (`func5`, 145–159)

```
x > Tc:        y = p5                              (clamps to background)
Tc > 0, x≤Tc:  y = p1·|1 − |x/Tc|^|p3||^p4 + p5
Tc ≤ 0:        y = p1 + p5
```

Asymmetry: EXPRESSIBLE as `OrderParameter + Constant`
(`y0·[1 − (T/Tc)^α]^β`, clamped to exactly 0 at/above Tc; α, β, Tc clipped
to their physical non-negative domain). Docs recipe, not code.

**Divergence (documented, not replicated):** WiMDA mirrors negative
parameter excursions with `abs()` (|x/Tc|, |p3|, |q2|); Asymmetry clips
α, β ≥ 0 and Tc > 0 instead. On the physical domain (0 ≤ T ≤ Tc, positive
exponents) the two are *identical* — note for the record that WiMDA's
`abs(q2)` can never differ from `q2` below Tc since |x/Tc|^|α| ≤ 1 there;
the mirroring only changes behaviour for unphysical negative-parameter
trials during minimisation, where Asymmetry's clipping collapses the
degenerate sign instead of silently reporting it.

Both clamp above Tc; both add the constant background everywhere
(WiMDA via p5, Asymmetry via `+ Constant`), so the recipe is exact.

### 1.7 Divergence of relaxation rate (`WidthDiv`, 162–174)

```
x ≠ Tc:  y = p3 + p4 / |x − Tc|^p2
x = Tc:  y = p3                       (exact-equality guard)
```

Asymmetry `CriticalDivergence`: `y = a·|T−Tc|^(−ν) + c` with |T−Tc| floored
at 1e-9 (`parameter_models.py:84–89`). PRESENT — naming map for docs:
Tc↔Tc, alpha↔ν, Min rate↔c, scaling↔a.

**Divergence (documented):** at the singular point WiMDA returns MinRate
(suppresses the divergent term); Asymmetry returns the (huge) floored value
`a·(1e-9)^(−ν) + c`. Any real fit excludes the critical point (that is what
union multi-range is for); no change needed.

### 1.8 Repolarisation of isotropic Mu (`muonrep`, 78–84)

```
p2 > 0:  y = p1·(0.5 + (x/p2)²) / (1 + (x/p2)²) + p3      (p2 = B0, x = B)
p2 ≤ 0:  y = p1 + p3
```

**Physics verification.** This is exactly the textbook time-averaged
longitudinal polarization of isotropic muonium: in LF only the 2↔4
transition mixes states, P_z(t) = (1−a₂₄) + a₂₄·cos ω₂₄t with
a₂₄ = 1/[2(1+x²)], and the observed (time-averaged) repolarisation curve is

  1 − a₂₄ = (½ + x²)/(1 + x²),    x = B/B₀,   B₀ = ω₀/(|γₑ| + γ_μ)

(*Muon Spectroscopy: An Introduction*, §4 LF treatment of isotropic Mu;
vacuum Mu A = 4463.302 MHz ⇒ B₀ ≈ 0.1585 T = 1585 G). WiMDA's formula is
therefore the physically-correct form and ports unchanged; only the
parameterisation improves.

**Port (`MuRepolarisation`, scope `field`):** parameters
(a_Mu, A_hf [MHz], a_Dia) with

  B₀[G] = A_hf / (γₑ/2π + γ_μ/2π)[MHz/G]

derived from `core/utils/constants.py`
(`ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G`/2π = 2.80249514 MHz/G;
`MUON_GYROMAGNETIC_RATIO_MHZ_PER_T`·1e-4 = 0.013553882 MHz/G; sum
2.81604902 MHz/G). Fitting A_hf directly is what the analysis is *for*
(repolarisation is the standard method of estimating the hyperfine constant
when precession is unresolvable); WiMDA users had to convert B₀ by hand.
y(0) = a_Mu/2 + a_Dia (the ½ lost to the unobserved fast oscillation),
y(∞) = a_Mu + a_Dia.

**Divergences (documented):**

- *Parameterisation*: WiMDA fits B₀ in G; Asymmetry fits A_hf in MHz with
  B₀ derived. Conversion: A_hf[MHz] = 2.81604902 × B₀[G].
- *Degenerate branch*: WiMDA's p2 ≤ 0 branch returns the fully-repolarised
  value a_Mu + a_Dia; Asymmetry instead bounds A_hf > 0 (hard minimum), so
  the branch is unreachable.
- Pairs with the time-integral asymmetry observable (PR #23) which produces
  the integral-vs-B series this component fits.

## 2. Machinery transcription (`Model.pas`, `Modelxrange.pas`, `Numlib.pas`)

### 2.1 Error modes (`getmodeldata` 560–654, `setydefault` 491–546, Model.dfm radio group)

Five radio buttons: `UseColumn`, `UsePercent`, `UseAbsolute`, `UseNone`,
`UseEstimate`. Per-point σ assignment when building (x, y, σ):

| Mode | σᵢ | Notes (verified) |
|---|---|---|
| Column | error-column value | `DerivedErrors := false`; the only mode using the table's propagated errors |
| Percent | (pct/100)·yᵢ | pct from the shared text box; **proportional to the y value**, so yᵢ = 0 ⇒ σᵢ = 0 and the point is silently skipped by the χ² accumulation (`Chi2Update` requires yerr > 0) |
| Absolute | constant `errabs` | from the shared text box; parse failure ⇒ errabs := 1 |
| None | 1 | the `else` branch of the radio cascade (unit weights) |
| Estimate | constant `errabs` (same as Absolute) | after each successful fit: `errabs := errabs·sqrt(ch2dof)` and the text box is updated (`FitButtonClick` 685–690) — the user refits by hand until χ²ᵣ → 1 |

**Estimate-mode equivalence (the reason we do NOT port the iteration).**
With constant σ = e, χ²(e) = χ²₁/e² where χ²₁ is the unweighted (σ=1) value;
the minimiser's location is independent of e (uniform weight scaling).
WiMDA's update e ← e·√(χ²(e)/ν) = √(χ²₁/ν) converges in **one step** from
any starting e to the fixed point e\* = √(χ²₁/ν), at which χ²ᵣ = 1 and the
Gauss–Newton parameter errors equal the σ=1 errors × √(χ²₁/ν). The modern
equivalent is therefore: fit unweighted, then rescale parameter errors by
√(χ²/dof) — the standard "estimate errors from scatter" of unweighted
least squares. Asymmetry implements exactly that (no iteration, no mutating
input errors). One subtlety: WiMDA's displayed χ²ᵣ after the rescale is 1 by
construction and carries no goodness information — Asymmetry's UI must make
the same caveat visible (χ²ᵣ is meaningless in this mode; the χ² quality
verdict is suppressed).

**Asymmetry today:** `fit_parameter_model(..., yerr)` takes the propagated
errors from the fit series (Column ≡ current behaviour) or `yerr=None` → unit
weights (None mode, core-only, not reachable from the GUI). Additionally a
stabilisation floor clamps σᵢ to ≥ 50 % of the median positive σ
(`_stabilize_parameter_model_errors`, `parameter_models.py:1266`). WiMDA has
no floor. **Decision recorded in implementation-options.md:** the floor
remains Column-mode-only; user-specified Percent/Absolute σ and unit weights
are used verbatim (floor would silently override an explicit user choice).

### 2.2 Union multi-range (`Modelxrange.pas`; `getxranges` 548–558; `getmodeldata` 580–602)

A memo holds up to 20 lines "from to" (free-format, two columns). When
`UseMultirange` is checked, a point enters the fit if it lies in **any**
window (OR-combination); one model is fitted across the union. Otherwise the
single [Xfrom, Xto] applies. The canonical use is fitting
`CriticalDivergence` to a λ(T) series while excluding the critical region
around Tc.

**Asymmetry today:** `ModelFitRange(x_min, x_max, model, …)` — multiple
*ranges* exist but each carries its own independent model (a different
feature: piecewise modelling, which WiMDA lacks — keep it). The gap is
windows *within* one range. Port: `ModelFitRange` gains
`windows: list[tuple[float, float]] | None`; mask = OR over windows
(falling back to x_min/x_max when absent).

### 2.3 χ² quality verdict (`Chi2UpdateClick` 1447–1516; `Numlib.pas chilow/chihigh` 120–138; `FitOpt.pas` 59–87; `Fitucode.pas:1093`)

Verified semantics, with R = `Rgoodfit` (default 0.95, user-clamped to
[0.5, 0.999]) and ν = n − n_free:

```
P = CDF_χ²(χ²; ν) = Gammp(ν/2, χ²/2)
P < (1−R)/2   →  "overdone"  (purple)     e.g. P < 0.025 at R = 0.95
P > (1+R)/2   →  "poor"      (brown)      e.g. P > 0.975
else          →  "good"      (green)
Target band:  [chilow, chihigh] where CDF(ν·chilow·…) — i.e.
              χ²ᵣ ∈ [χ²inv(α/2, ν)/ν, χ²inv(1−α/2, ν)/ν],  α = 1−R
ν ≤ 0         →  "No target range (dof=0)"
fit failed    →  "Failed to converge" (red)
```

(`chilow`/`chihigh` invert the CDF by Brent's method; the upper bracket is
[ν, 5ν], which can fail for tiny ν — scipy's `chi2.ppf` has no such limit.)

**Asymmetry today:** χ² and χ²ᵣ numbers only (`ParameterModelFitResult`,
displayed in `ModelFitDialog._select_range`). Port: a **shared core helper**
(new `core/fitting/fit_quality.py`) returning verdict + band from
`scipy.stats.chi2.ppf`; `fit-workflow-diagnostics` (Wave B, time-domain fit
panel) reuses it. Display in `ModelFitDialog` only for now, with a tooltip
that teaches: what the band means at this ν, why too-low χ²ᵣ ("overdone")
usually means overestimated errors or overfitting, and that the verdict
assumes Column-mode (real) errors.

### 2.4 x2 second analytic variable — OUT OF SCOPE (requirements record)

Verified plumbing: `X2Use` (column) / `X2useVal` (fixed value) / else 0
(`getmodeldata` 621–626); a distinct-value list `x2list` is built for
plotting one model curve per x2 value; the model signature is
`f(x, x2, p)`; **none of the eight built-ins reads x2** — it reaches user
DLLs only. The workflow it serves (e.g. λ(T) surfaces measured at several
fields, fitted simultaneously with shared parameters) is covered in
Asymmetry by cross-group global/local fitting (`global_fit_parameter_model`
+ the cross-group dialog), which is strictly more general (per-parameter
global/local/fixed roles rather than a single extra scalar).

If a genuinely two-variable *component* is ever wanted, the requirements on
`ParameterCompositeModel` would be:

1. Component evaluation signature gains an optional second array
   `x2: NDArray | None` (broadcast against x).
2. Per-point x2 sourcing: a second series column (arbitrary parameter) or a
   constant; `ParameterGroupData`-style plumbing into the dialog.
3. Distinct-x2 curve families in plotting and GLE export (WiMDA's `x2list`).
4. Registry scoping extended so two-variable components only appear when an
   x2 source is configured.
5. Project schema: serialise the x2 source choice.

### 2.5 Out of scope / superseded (verified)

- **`*fit.dll` user models** (`LoadModelDLLs` 1133–1242; `UserUnit/UserUnit.pas`):
  runtime discovery of `*fit.dll`, entry points `getfnlist`/`getparams`/
  `details`/optional `getresults` (uppercase FORTRAN variants
  `GETFNLIST`/`GETPARAMSP`/`DETAILS`). → `python-user-functions` project
  (Wave B; lands after this one because it generalises the same registry).
- **ReloadFit** (`ReloadFitClick`): reparses the `.mfit` text log to restore
  parameters. Superseded by `.asymp` persistence (and by the recorded
  persistence follow-on).
- **Second-level Model Fit Table** (`ModelFitTableUnit.pas`, 198 lines):
  verified to be a bare auto-saving text editor into which `UpdatePar`
  (Model.pas 1395–1436) appends one row per model fit (χ²ᵣ + value/error
  pairs); load/save/print only, no analysis. Asymmetry follow-on: "send
  model-fit results to the results table" so model-fit outputs can
  themselves be trended — cleaner than a separate table widget.
- **GLE export of model fits** (`SetUpGLE`): Asymmetry's GLE export already
  covers trend plots; no changes in this project (standing instruction).

## 3. Divergence summary (both behaviours stated)

| # | Topic | WiMDA | Asymmetry (this project) |
|---|---|---|---|
| D1 | Power-law negative x | faults/NaN (`power`) | mirrors via \|x\| (existing convention, also in `PowerLawQuadBG`) |
| D2 | 2-Lorentzian BG | cubic centred on Pos 1; coefficients move with the peak | `Polynomial` in absolute x; coefficients independent of peaks; **not 1:1 transferable** |
| D3 | Peak switch-off | amplitude == 0.0 float test | omit the component |
| D4 | Activation energy | eV, e/k 0.089 % low | meV, CODATA k_B |
| D5 | Order parameter, unphysical trials | `abs()` mirroring | clip to physical domain (identical on physical domain; clamp above Tc in both) |
| D6 | Critical divergence at T = Tc | returns MinRate (equality guard) | floored \|T−Tc\| ≥ 1e-9 (point excluded from fits in practice) |
| D7 | Mu repolarisation parameter | B₀ in G fitted directly; p2 ≤ 0 branch | A_hf in MHz fitted, B₀ derived from CODATA-based constants; A_hf > 0 bound |
| D8 | Estimate error mode | mutate-σ manual iteration, χ²ᵣ display forced to 1 | one-pass unweighted fit + √(χ²/dof) parameter-error rescale; χ²ᵣ verdict suppressed with explanation |
| D9 | Percent mode at y = 0 | point silently dropped (σ = 0 test) | σᵢ = pct·\|yᵢ\|; zero-σ points masked out (existing finite/positive mask) — same effect, made explicit in docs |
| D10 | Error floor | none | 50 %-of-median floor retained **only** for Column mode |
| D11 | χ²ᵣ band inversion | Brent on [ν, 5ν] (can fail at tiny ν) | `scipy.stats.chi2.ppf` (exact) |
| D12 | Multi-range entry | free-text memo, max 20 | structured window list per range, no fixed cap |
