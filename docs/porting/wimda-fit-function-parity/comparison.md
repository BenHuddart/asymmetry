# Comparison: WiMDA Time-Domain Functions vs Asymmetry / Textbook

Notation reference: *Muon Spectroscopy: An Introduction* (Blundell, De Renzi,
Lancaster, Pratt; OUP 2022) = **MS-Intro**. WiMDA code references are to
`$WIMDA_SRC/src/` at the revision studied (June 2026 checkout).

## Cross-cutting conventions

| Aspect | WiMDA | Asymmetry | Decision for ports |
|---|---|---|---|
| Time | µs | µs | unchanged |
| Frequency | MHz | MHz | unchanged |
| Field | Gauss; γ_µ/2π = 0.01355342 MHz/G (`AsymFitFunction.pas:46`) | Gauss; `MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = 135.538817` (`core/utils/constants.py`) | Asymmetry constants (slightly more precise; CODATA-consistent) |
| Electron γ | 2.8024 MHz/G (`muoniumfunctions.dpr:13`) | `ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G = 17.6085963023` | Asymmetry constants |
| Phase | degrees, `+φ/360` inside cos(2π(...)) | radians | radians (Asymmetry convention) |
| Oscillation sign | mixed; `LowTFMuonium` uses −w (negative frequency) | positive-frequency convention (see `core/fitting/muonium.py` docstring) | positive frequency |
| Amplitudes | percent asymmetry, components share a "relaxing amplitude" budget with optional fractions | percent, free per-component `A`, `{frac}` groups available | unchanged |
| Safe exp | clamp |exponent| > 750 (DLLs) / > 100 (built-in) | `np.clip(exponent, -700, 0)` style guards | Asymmetry idiom |

Gaussian relaxation convention: Asymmetry's `Gaussian` is exp(−(σt)²) which
matches WiMDA `rtGau`. MS-Intro Eq. 5.12 uses exp(−Δ²t²/2) (the `rtGau2`
variant). The component info text should state the e^{−(σt)²} convention
explicitly and give the mapping σ = Δ/√2 to the textbook form, rather than
adding a second component that differs only by √2.

## A. Built-in grid gaps

### A1. Risch–Kehr relaxation (`rtRK`)

WiMDA (`AsymFitFunction.pas:621-636`):

```pascal
gt := abs(x[k] * parameters[7*i+2]);
if gt < 20 then RK := ex(gt) * erfc(sqrt(gt))
else RK := 1 / sqrt(pi*gt);          // asymptotic form
if parameters[7*i+2] < 0 then RK := 2 - RK;   // mirrored branch for Γ<0
```

Math: G(t) = e^{Γt}·erfc(√(Γt)). Spin relaxation of a muon (or muonium)
polarization coupled to a 1D-diffusing carrier/defect; the long-time tail decays
as (πΓt)^{−1/2}.

Source: R. Risch and K. W. Kehr, Phys. Rev. B **46**, 5246 (1992). **Not in
MS-Intro** (closest context: §8.4 mobile excitations) — flagged for review.

Port notes: implement with `scipy.special.erfcx` (`erfcx(x) = e^{x²}erfc(x)`, so
G(t) = erfcx(√(Γt))) — numerically stable for all Γt, no branch needed. Γ ≥ 0
enforced; WiMDA's negative-Γ mirror `2 − RK` is an unphysical fitting convenience
we propose **not** to port.

### A2. Frequency-normalised stretched exponential (`rtFstr`)

WiMDA (`AsymFitFunction.pas:608-620`): if the component oscillates at frequency f
(`otFRotation`), relaxation = exp(−(λ_n·2πf·t)^β); otherwise exp(−(λ_n·2πt)^β).
The rate parameter λ_n is dimensionless ("normalised to the oscillation
frequency"); useful when damping scales with the precession frequency (e.g.
field-proportional inhomogeneous broadening across a multi-field data set).

Asymmetry has no cross-parameter coupling inside a component, but `Parameter.expr`
constraints can tie an ordinary `StretchedExponential` rate to a frequency
parameter (`Lambda = 2*pi*c*frequency_1`). Options in
`implementation-options.md`; default recommendation is to rely on `expr` and
document the recipe rather than port a special component.

### A3. Scaled frequency rotation (`otScaledFRotation`)

WiMDA (`AsymFitFunction.pas:456-461`): cos(2π(f·c·t + φ/360)) with scale factor c
as the third parameter — used to tie several components to one frequency times a
fitted/fixed ratio. Same expressiveness exists via `expr`/link groups on
`Oscillatory.frequency`. Recommendation: do not port; document the recipe.

### A4. Bessel oscillation (withdrawn `otBessel` — proposed re-introduction)

P(t) = J₀(γ_µB₁t) for an incommensurate spin-density-wave field distribution —
MS-Intro Eq. 6.47 (and Eqs. 6.44–6.48 for the Overhauser distribution origin and
the asymptotic cos(γ_µB₁t − π/4) behaviour). Standard in musrfit (`TFieldCos`
analogue `bessel`) and Mantid (`Bessel`). WiMDA withdrew it
(`FitTyps.pas:130`), but parity in *data types that can be fitted* argues for
including it: fitting SDW materials in WiMDA today requires the withdrawn
function or external software. Parameterize as A·J₀(2πν t + φ) by frequency
(plus a field-parameterized sibling, or a single component following whichever
parameterization we pick for oscillations) using `scipy.special.j0`.

## B. Muonium DLL gaps

Shared background (MS-Intro §4.4): isotropic muonium in field B has
x = B/B₀ with B₀ = A/(γ_e+γ_µ), energies E₁..E₄ (Eq. 4.49/Breit–Rabi), and TF
transition frequencies ν₁₂, ν₂₃, ν₁₄, ν₃₄ with amplitudes a₁₂ = a₃₄ =
(1+δ)/4, a₁₄ = a₂₃ = (1−δ)/4 where δ = x/√(1+x²) (Table 4.2 / Fig. 4.12). The
already-ported `MuoniumTF` implements all four lines; the gaps below are the
high-field reductions.

### B1. High-TF muonium pair (`MuoniumPairRot`, `muoniumfunctions.dpr:26-35`)

```pascal
wminus := (ge-gm)*p1/2;                       // p1 = B (G)
Omega  := sqrt(sqr(w0)+sqr((ge+gm)*p1))/2 - w0/2;   // w0 = A (MHz)
f2 := Omega - wminus;  f1 := f2 + w0;
result := 0.5*( cos(2π(f1·t+(φ)/360)) + cos(2π(f2·t+φ/360)) );
```

Physics: at high field only the ν₁₂ and ν₃₄ lines have weight (MS-Intro
Fig. 4.12, Eq. 4.65: A = ν₁₂+ν₃₄); both carry amplitude ½ (of the muonium
fraction). f₁−f₂ = A exactly in this implementation. Parameters: B (G),
A (MHz), φ.

Equal ½/½ amplitudes are the x→∞ limit of (1±δ)/4 normalised to the pair; the
WiMDA form is an approximation valid when δ→1 lines are unobservable. The port
should state this regime in the component info (valid for x ≳ a few, i.e.
B ≫ B₀ ≈ 1585 G for vacuum Mu).

### B2. Powder-averaged anisotropic high-TF pair (`AnisMuoniumPairRot`, `muoniumfunctions.dpr:38-55`)

As B1 but with an axial anisotropic hyperfine component D: for each cosθ on a
15-point midpoint grid, d = (D/2)(3cos²θ−1), lines at f₁+d/2 and f₂−d/2, averaged
over the polycrystalline (PCR) distribution. Parameters: B (G), A (MHz), D (MHz)
— note **no phase parameter** in WiMDA (slot taken by D; global phase enters via
the shared `ph` argument). MS-Intro Eqs. 4.66–4.68 give the axial hyperfine
tensor convention: A_⊥ = A_iso − D/2, A_∥ = A_iso + D... (we will follow MS-Intro
Eq. 4.68's (A_iso, D) decomposition and document the mapping to WiMDA's (A, D)).

Port notes: keep a phase parameter (Asymmetry components are not slot-limited);
make the angular grid size an implementation detail (fixed 32–64 point
Gauss–Legendre rather than WiMDA's 15 midpoints — verify convergence in tests).

**Implementation-pass finding (review-confirmed).** WiMDA's `f2 := f2 − d/2`
acts on the *signed* (negative) `f2`, so the observed line |f2| shifts **up**
by d/2 — both lines co-shift and the pair sum tracks the orientation's
effective coupling A(θ) = A_hf + d. A literal `−d/2` applied to the positive
ν₁₂ of our positive-frequency convention flips that shift (an early version
of this port did exactly that). Furthermore the symmetric ±d/2 split is only
approximate: the exact 4-level Hamiltonian distributes the shift unevenly
(∂ν₁₂/∂A = 0.27, ∂ν₃₄/∂A = 0.73 at 3 kG) and the line *splitting* also
depends on A_⊥ at O(D·A/B). **Implemented:** exact batched diagonalization of
`H = γ_e B S_z^e − γ_µ B S_z^µ + S^e·A(θ)·S^µ` per powder orientation,
selecting the two strongest σ_x^µ transitions; D = 0 reduces exactly to the
isotropic pair. Fitted D is therefore not directly comparable with WiMDA's
`PCR Hi TF Mu`.

### B3. Muonium LF relaxation (`MuLFrel`, `muoniumfunctions.dpr:118-132`)

```pascal
w0 = 2π·4463 MHz (vacuum Mu hyperfine, hard-coded as 2*pi*4464);
x := 2*Gpl*B/w0;  // Gpl=(ge-gm)/2 — NB: code sets Gpl=Gmi=(ge-gm)/2
w12 := w0/2*(1+(Gmi/Gpl)*x-sqrt(1+sqr(x)));
lam := (1-x/sqrt(1+sqr(x)))*sqr(deltex)*tauc/(1+sqr(w12*tauc));
result := exp(-lam*t);
```

Physics: longitudinal-field T₁ relaxation of muonium undergoing electron
spin-exchange collisions; λ(B) follows the ν₁₂ transition with exchange coupling
δ_ex and correlation time τ_c. Source: R. Kadono et al., Phys. Rev. Lett.
**64**, 665 (1990) (cited in the WiMDA source comment as "Kadono PRL64,665").
**Not in MS-Intro** (general spin-exchange context: MS-Intro §12) — flagged for
review, including the suspicious `Gpl = Gmi = (ge−gm)/2` (the standard
Breit–Rabi ν₁₂ uses (γ_e+γ_µ) in x; this looks like a WiMDA bug or deliberate
approximation — the implementation pass must re-derive λ(B) from the cited paper
rather than transliterate).

Parameters: δ_ex (MHz), τ_c (µs), B (G). WiMDA hard-codes A = 4463 MHz (vacuum
muonium); we propose exposing A_hf as a fixed-by-default parameter for
consistency with the other muonium components.

**Implementation-pass resolution.** The correct citation is Kadono et al.,
PRL **64**, 665 (1990), *"Delocalization of muonium in NaCl"* (companion to
Kiefl et al., PRL **62**, 792 (1989), KCl). The recent JPSJ treatment by the
same lead author (T. U. Ito & R. Kadono, J. Phys. Soc. Jpn. **94**, 064601
(2025); arXiv:2410.23575) restates the intra-triplet relaxation as its Eq. 22,

> 1/T₁µ ≈ Δₙ²·ν/(ω₁₂²+ν²),   x₁₂ = ω₁₂/ω₀ ≃ ½[1 − √(1+x_p²) + x_p]   (Eq. 23)

and **explicitly attributes Eq. 22 to reference [27], which is exactly Kadono
et al., PRL 64, 665 (1990)** — i.e. the same paper WiMDA cites. With τ_c = 1/ν
this is identically δ_ex²·τ_c/(1+(ω₁₂τ_c)²). It builds its approximate ω₁₂ from
(γ_e−γ_µ)-type averages, so WiMDA's ω₁₂ choice is a literature convention; we
use the *exact* Breit–Rabi ω₁₂ (`_tf_levels`) instead, which is strictly more
accurate (it also recovers the high-field electron-flip splitting that Eq. 23's
saturation at ω₀/2 drops).

**Correction (2026-06-13, reviewer-approved).** The cited Eq. 22 carries **no
field-dependent amplitude prefactor** — the LF decoupling enters *solely*
through ω₁₂(B). WiMDA additionally multiplies by `(1 − δ)`,
δ = x/√(1+x²); that factor is **not** in the cited source. It is also *not* the
spin-exchange amplitude `a₂₄ = 1 − g_z(x_p)` of the same paper's Eq. 17 (which
has a different functional form — `1−g_z ∝ 1/(1+x_p²)`, value ½ not 1 at x=0 —
and rides the *ω₂₄* transition, not ω₁₂). So WiMDA's `(1 − δ)` matches neither
mechanism: it force-quenches λ→0 at high LF, whereas the genuine Kadono
intra-triplet rate quenches only as ω₁₂(B) carries the Lorentzian off-peak.
**Resolution: the `(1 − δ)` prefactor is removed.** Shipped form:

> λ(B) = δ_ex²·τ_c/(1+(2πν₁₂τ_c)²),  ν₁₂ from the exact Breit–Rabi levels,
> δ_ex in MHz (≡ µs⁻¹, no 2π — consistent with Δ/ν conventions elsewhere).

Pinned by `test_muonium_lf_relaxation_matches_kadono_eq22` at B ∈ {0, 100, 1000,
5000} G. Fitted δ_ex/τ_c values are not directly comparable with WiMDA's (which
retains the spurious prefactor and the approximate ω₁₂).

## C. Dipolar DLL gaps

### C1. Gaussian-broadened Kubo–Toyabe (`GBKTB`, `KuboToyabe.pas:193-213`)

Static LF Gaussian KT averaged over a Gaussian distribution of Δ:
quadrature over 41 points i ∈ [−20,20], Δᵢ = |Δ(1 + w·i/7)|, weight
exp(−(i/7)²) — i.e. a Gaussian in Δ of standard deviation w·Δ/√2 truncated at
±2.86σ (w is "rel width"). Used for disordered/dilute-moment systems where a
single Δ is too sharp (e.g. distribution of nuclear environments).

MS-Intro coverage: KT itself Eqs. 5.26–5.27; the Δ-broadening is a
phenomenological extension (no textbook equation) — document as "Gaussian
distribution of Δ" with the KT citations. Related published variants exist
(e.g. Noakes–Kalvius); the implementation pass should document which exact
weighting we adopt. Port via deterministic Gauss–Hermite quadrature instead of
WiMDA's ad-hoc grid; verify against direct numerical integration.

### C2. Dynamic F–µ–F (`FmuFdyn`, `dipolarfunctions.dpr:129-171`)

Strong-collision dynamicization (MS-Intro Eq. 5.30: P(t) = Pˢ(t)e^{−νt} +
ν∫₀ᵗ P(t−t′)Pˢ(t′)e^{−νt′}dt′) applied to the static linear F–µ–F polarization,
discretized on a uniform grid with step tt = 0.01/max(ν, ω_d), plus a fast-
fluctuation shortcut exp(−2ω_d²t/ν) when νt > 10 and ν > 10ω_d. Parameters:
ω_d (MHz), ν (MHz), t_max (µs, grid horizon — an implementation artifact).

Port notes: Asymmetry already has exactly this strong-collision machinery for
`DynamicGaussianKT`/`DynamicLorentzianKT` (`_dynamic_kt_grid` in
`core/fitting/models.py`) — reuse the same Volterra solver with the static
F–µ–F kernel; drop the user-visible t_max parameter (derive the grid from the
data range, as the dynamic KT port already does). Parameterize by r(µ–F) for
consistency with `FmuF_Linear`, or ω_d — decision in options doc.

**Implementation-pass refinement (review-driven).** WiMDA's fixed switch to
the bare motional-narrowing exponential at ν > 10ω_d (and ours initially at a
fixed ν = 12 µs⁻¹) leaves a discontinuity in the model — measured 2.5 % at
r = 1.17 Å up to ~30 % at short trial distances. Implemented instead: the
Volterra solver runs to the crossover ν = 12·ω_d (with a stability ceiling
from the grid cap), beyond which an **Abragam-form interpolation**
exp[−(2ω_d²/ν²)(e^{−νt} − 1 + νt)] takes over (same ν→∞ limit, correct
quadratic short-time form). Branch seam measured at 0.24 %/0.56 %/2.5 % for
r = 1.17/0.8/0.6 Å, regression-tested.

### C3. F–µ–F–F triangle (`Ftriangle`/`Fequitriangle` → `polarize.pas`, `matrices.pas`)

Full quantum solution for muon + three ¹⁹F spins (16-dimensional Hilbert space):
`f3calc` builds the dipolar Hamiltonian from the three distances (r₁, r₂, r₃ in
Å), diagonalizes, and `polarise` computes P(t) = (1/3)P_z + (2/3)P_x from the
eigenvector overlap coefficients (`polarize.pas:104-123`; energies in kHz —
note the `/1000` in the cosine arguments). The equilateral convenience wrapper
maps a single r to (2r·?…) — `Fequitriangle` calls `Polarise(t, 2*p1,
p1*sqrt(3)/2, p1/(2*sqrt(3)))` (`dipolarfunctions.dpr:178-181`); the
implementation pass must reverse-engineer `f3calc`'s exact geometry convention
from `matrices.pas` before trusting these arguments.

MS-Intro coverage: F–µ–F formalism §4.5 (Eqs. 4.72–4.81); the three-fluorine
extension follows the same dipolar Hamiltonian construction. Asymmetry's
`FmuF_General` already solves a 3-spin (8-dim) problem numerically with powder
averaging — the natural port is to extend `muon_fluorine/dipolar.py` to N=3
fluorines with explicit geometry, reusing the existing eigen-solver and caching
patterns rather than transliterating the Pascal.

Note: WiMDA's (1/3)P_z + (2/3)P_x is a two-orientation proxy for the powder
average, not a full angular average (contrast `FmuF_General`, which integrates
over orientations). The port should do a proper powder average and record the
difference in `comparison` results.

**Implementation-pass findings (f3calc decode).** `matrices.pas:48-56`
resolves to: muon at the origin; F1 = (0, r3, r1/2), F2 = (0, −r3, r1/2)
(symmetric pair, both at distance √(r3²+r1²/4)); F3 = (0, r2−r3, 0). Only the
three **µ–F** couplings are built (`dips11/21/31`) — the F–F dipolar couplings
are omitted entirely — and the constant 180.4 kHz·Å³ matches the µ–F dipolar
constant used elsewhere. The `Fequitriangle` wrapper's arguments
`(2r, r√3/2, r/(2√3))` do **not** produce an equilateral fluorine triangle in
this geometry, so the WiMDA wrapper appears internally inconsistent.
**Decision (implemented):** Asymmetry's `FmuF_Triangle` uses an explicit,
documented geometry (collinear F–µ–F at `r_muF` + third F at distance `r3`,
angle `phi3` from the axis), includes **all six** pairwise couplings, and does
a full powder average. Verified: it reproduces `FmuF_General(r, r, 180°)` to
≤ 5×10⁻⁹ as r3 → ∞. Fitted distances are deliberately *not* comparable with
WiMDA's `F-u-F-F`.

### C4. Single spin-J dipole + quadrupole (`ZFdipgen`, `dipolarfunctions.dpr:33-78`)

Muon coupled to one nucleus of spin J with dipolar frequency f_dip and
quadrupolar frequency f_quad (both MHz): closed-form eigenvalues per m-block
(λ±(m) from a 2×2 diagonalization), P(t) = (P_z + 2P_x)/3 per the
polycrystalline recipe. Source: M. Celio and P. F. Meier, Hyperfine Interact.
**17–19**, 435 (1984). **Not in MS-Intro** beyond the quadrupole Hamiltonian
(Eq. 4.87) and ALC discussion — flagged for review.

Use case: ZF precession from muon–quadrupolar-nucleus pairs (e.g. µ⁺–⁹³Nb,
µ⁺–⁶³Cu) where the F–µ–F spin-½ formalism does not apply.

**Implementation-pass finding (review-confirmed): WiMDA's `ZFdipgen` is wrong
for every J > 1/2.** Its per-block mixing angle is reconstructed from
`cos² 2α` (`csqa := 0.5*(1+sqrt(csq2a))`), discarding the sign of
`cos 2α = −q1/W_m`. Verified against exact diagonalization of
`H = ω_d(S·I − 3 S_z I_z) + ω_q I_z²` (which reproduces the closed form's
eigenvalues to machine precision): the |·| variant deviates by up to ~0.56 of
the normalised polarization for J ∈ {1, 3/2, 5/2, 9/2}; J = 1/2 is the unique
case where the sign cannot matter. Asymmetry's `dipolar_spin_j` uses the
signed mixing angle, which matches exact diagonalization to < 3×10⁻¹⁴
(regression-tested via an independent exact-diagonalization reference in
`tests/test_wimda_parity_components.py`). The amplitude–frequency pairing of
the P_x sum and the (P_z + 2P_x)/3 polycrystalline average were verified
correct as coded. **Fitted parameters are not comparable with WiMDA for
J > 1/2.**

### C5. Single spin-½ dipole family (`ZFdipole`, `ZFprotondipole`, `ZFelectrondipole`, `dipolarfunctions.dpr:81-100,184-192`)

All three evaluate the same Meier spin-½ pair polarization
(Meier, HFI **17–19**, 427 (1984); identical math to MS-Intro Eq. 4.80's
⟨P_z(t)⟩ = 1/6·[1 + cos ω_d t + 2cos(ω_d t/2)·...] form — WiMDA writes it as
(1 + e^{−λ_t t}(cos ωt + 2cos 1.5ωt + 2cos 0.5ωt))/6):

- `ZFdipole`: ω = γ_µ·B_dip with B_dip (G) fitted directly; extra transverse
  relaxation e^{−λ_t t} applied **only to the oscillating 5/6 part**.
- `ZFprotondipole`: B_dip = c·µ_p/r³ with r (Å) fitted (c = 5.05 const,
  source comment notes it "should be 2π·10·(m_µ/m_p)" — re-derive, don't copy).
- `ZFelectrondipole`: B_dip = 8290/r³ (electron moment at distance r Å).

Asymmetry's `MuF` covers the fluorine case parameterized by r. Proposal:
one generalized `DipolarPair` component — field-parameterized ω_d **or**
nucleus+distance parameterization (F, H, e⁻, or explicit γ), plus optional λ_t
on the oscillating part, matching MS-Intro Eq. 4.80 notation. Exact shape in
options doc.

## D. Pressure-cell DLL

- `BeCu ZF` = (1−f)·KTZ(Δ) + f·e^{−λt} (`pressurefunctions.dpr:22-25`): pure
  composite of existing components — no new physics. Skip or ship as a
  documented composite preset.
- `BeCu LF 110G` (`pressurefunctions.dpr:31-42`): empirical 5th-order polynomial
  λ(T) for one specific cell at one field, exp(−(λt)^2.5). Calibration data, not
  physics; proposed **skip** (flag to user).

## E. Already-covered functions (verified equivalent, no action)

| WiMDA | Asymmetry | Verified notes |
|---|---|---|
| `otFRotation` | `Oscillatory` | phase deg→rad only |
| `otBRotation` | `OscillatoryField` | γ_µ constants differ in 6th digit (0.01355342 vs 0.0135538817 MHz/G) — fit-irrelevant |
| `otKuboToyabe` ZF/LF/dyn | `StaticGKT_ZF`/`LongitudinalFieldKT`/`DynamicGaussianKT` | prior port (dynamic-relaxation study) |
| `otKeren` | `Keren` | prior port |
| `rtLor`/`rtGau`/`rtStr`/`rtAbragam` | `Exponential`/`Gaussian`/`StretchedExponential`/`Abragam` | conventions identical |
| muonium TF/LowTF/ZF | `MuoniumTF`/`MuoniumLowTF`/`MuoniumZF` | prior port (muonium-triplet study; positive-frequency convention documented there) |
| `FmuFdipole`/`FmuFdipoler` | `FmuF_Linear` | r ↔ ω_d mapping: ω_d/2π = 0.1804305903/r³ MHz (r in Å) per `dipolarfunctions.dpr:119` |

## Non-textbook sources — reviewer sign-off

All non-textbook components shipped in this study were checked against their
primary source or an independent numerical reference. The verification strategy
favours an *independent* oracle (exact diagonalization, brute-force integration,
or an established closed form) over transliterating the paper, since that also
catches errors in the paper/WiMDA themselves (it did, twice: C4 and B3).

| # | Source | Component(s) | Checked against | Verdict | Date |
|---|---|---|---|---|---|
| A1 | Risch & Kehr, PRB **46**, 5246 (1992) | `RischKehr` | Established 1D-diffusion closed form G(t)=e^{Γt}erfc(√Γt), evaluated via `erfcx`, plus the (πΓt)^{−1/2} long-time asymptote (`test_risch_kehr_matches_erfc_form_and_asymptote`) | ✅ Verified. WiMDA's negative-Γ mirror `2−RK` is an unphysical fitting convenience, deliberately **not** ported (Γ≥0 enforced). | 2026-06-13 |
| B3 | Kadono et al., PRL **64**, 665 (1990) | `MuoniumLFRelax` | Eq. 22 of Ito & Kadono, JPSJ **94**, 064601 (2025) (= arXiv:2410.23575), which attributes that BPP form to this exact PRL ([27]); pinned at B∈{0,100,1000,5000} G (`test_muonium_lf_relaxation_matches_kadono_eq22`) | ⚠️→✅ **Corrected.** WiMDA's `(1−δ)` prefactor is in neither the cited Eq. 22 (NHF/intra-triplet, our ω₁₂ mechanism) nor Eq. 17 (spin exchange, amplitude `a₂₄=1−g_z`, transition ω₂₄). Prefactor **removed**; see the correction note in §B3 above. | 2026-06-13 |
| C4 | Celio & Meier, HFI **17–19**, 435 (1984) | `DipolarSpinJ` | Exact diagonalization of H = ω_d(S·I − 3S_zI_z) + ω_q I_z² with polycrystalline (P_z+2P_x)/3 average, agreement <3×10⁻¹⁴ (`test_spin_j_matches_exact_diagonalization`) | ✅ Verified by independent exact diagonalization. **WiMDA's own `ZFdipgen` is wrong for every J>½** (drops the sign of cos 2α); our signed implementation is correct. Fitted params not comparable with WiMDA for J>½. | 2026-06-13 |
| C5 | Meier, HFI **17–19**, 427 (1984) | `DipolarPairField`, `ProtonDipole`, `ElectronDipole` | Spin-½ closed form = MS-Intro Eq. 4.80 ⟨P_z⟩=⅙[1+cos ω_dt+2cos(ω_dt/2)·…]; J=½ limit of the exact-diag reference (`test_spin_half_reduces_to_meier_pair`) | ✅ Verified against the textbook closed form. Proton/electron γ-ratio scalings checked (`test_dipole_pair_frequency_scales_with_gyromagnetic_ratio`). | 2026-06-13 |
| C5 | Brewer et al., PRB **33**, 7813 (1986) | `FmuF_Linear` (pre-existing) | Prior port; MS-Intro Eq. 4.80; r↔ω_d mapping ω_d/2π=0.1804305903/r³ MHz | ✅ Verified (pre-existing component, textbook-derivable). | 2026-06-13 |
| C1 | *no canonical citation* (cf. Noakes & Kalvius, PRB **56**, 2352 (1997)) | `GaussianBroadenedKT` | Brute-force numerical average of the LF Gaussian KT over a Gaussian Δ-distribution, plus zero-width→LFKT limit (`test_gbkt_matches_brute_force_average`, `test_gbkt_zero_width_reduces_to_lf_kt`) | ✅ Verified as a phenomenological Δ-distribution via deterministic Gauss–Hermite quadrature (not WiMDA's ad-hoc 41-point grid). Documented as phenomenological, not paper-cited. | 2026-06-13 |
| A3 | Overhauser (Bessel-modulated relaxation) | `Bessel` | j₀ Bessel-function form against direct numerical evaluation of the Overhauser integral (`test_bessel_matches_overhauser_integral`) | ✅ Verified against the integral form. | 2026-06-13 |

**Outcome:** all sources signed off. One shipped-physics correction (B3, this
session); two WiMDA-side bugs documented as divergences we deliberately do not
reproduce (A1 negative-Γ mirror, C4 sign of cos 2α).
