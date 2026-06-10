# WiMDA Time-Domain Fit Function Parity

Status: **implemented** (decisions agreed 2026-06-10; see the decision table in
`implementation-options.md`; tests in `tests/test_wimda_parity_components.py`;
user docs in `docs/user_guide/fit_functions/`, with WiMDA migration recipes in
`docs/user_guide/fit_functions/index.rst`).

Components shipped: `RischKehr`, `Bessel`, `GaussianBroadenedKT`,
`MuoniumHighTF`, `MuoniumHighTFAniso`, `MuoniumLFRelax`, `DynamicFmuF`,
`FmuF_Triangle`, `DipolarPairField`, `ProtonDipole`, `ElectronDipole`,
`DipolarSpinJ` (per-nucleus dipole family per Decision 3). The component
picker was regrouped into Relaxation / Oscillation / Kubo-Toyabe / Muonium /
Nuclear dipolar / Background submenus.

**Documentation convention (agreed 2026-06-10):** the user-facing component
info must not cite textbook equation numbers. Expressions are kept consistent
with *Muon Spectroscopy: An Introduction* (Blundell, De Renzi, Lancaster,
Pratt; OUP 2022), but each component's applicability text is followed by an
APS-style reference list citing the original literature
(`FIT_COMPONENT_REFERENCES` / `PARAMETER_MODEL_REFERENCES` in
`core/fitting/component_docs.py`, rendered by the component-info dialog).
Applicability text uses rendered symbols (Greek letters, unicode
sub/superscripts); enforced by
`tests/test_wimda_parity_components.py::test_applicability_text_cites_via_reference_lists`.

## Problem statement

Asymmetry should be able to fit every kind of time-domain muon data (FB asymmetry
or individual detector groups) that WiMDA can fit. This study inventories every
time-domain fit function in the WiMDA source, maps each onto Asymmetry's existing
component registry, and identifies the gaps to close. Model functions for
parameter trending (fit-table functions of temperature/field) are explicitly out
of scope for this pass and will be addressed separately.

Primary documentation/notation reference for the implementation pass:
S. J. Blundell, R. De Renzi, T. Lancaster, F. L. Pratt, *Muon Spectroscopy: An
Introduction* (OUP, 2022) — cited below as **MS-Intro** with equation numbers.
Functions whose physics is not covered by MS-Intro carry their original literature
citation and are flagged in `comparison.md` for reviewer sign-off.

## How WiMDA structures time-domain fitting

WiMDA composes the asymmetry model as a sum of up to six components, each of which
is the product of an **oscillation** factor and a **relaxation** factor, plus a
relaxing baseline:

```
A(t) = A_bg·exp(-(λ_bg·t)^β_bg) + Σ_i  A_i · Osc_i(t) · Rel_i(t)
```

This matches MS-Intro Eq. 15.3, A(t) = Σᵢ AᵢGᵢᵒˢᶜ(t)Gᵢʳᵉˡ(t) + A_bg·G_bgʳᵉˡ(t).
Asymmetry's `CompositeModel` (sums/products of `COMPONENTS` entries) expresses the
same algebra, so no architectural change is needed — parity is purely a matter of
the component catalogue.

The WiMDA function space has two layers:

1. **Built-in grid** — oscillation types × relaxation types hard-coded in
   `src/AsymFitFunction.pas` (`MusrFun`, lines 340–699), type constants in
   `src/FitTyps.pas:122-142`, UI help text in `src/Analyse.pas:6726-6876`,
   Kubo-Toyabe numerics in `src/KuboToyabe.pas`.
2. **User-function DLLs** — plug-in oscillation/relaxation functions loaded at
   runtime. Shipped DLL sources live in `src/Extrafunctions/`:
   `muoniumfunctions.dpr`, `dipolarfunctions.dpr` (+ `polarize.pas`,
   `matrices.pas`), `pressurefunctions.dpr`.

Note: `src/fitfunctions.pas` ("Standard fit models") is the **table-fit** library
(functions of T or B for parameter trending) — out of scope here.

## Complete WiMDA time-domain inventory and gap map

### Built-in oscillation types (`FitTyps.pas:123-130`)

| WiMDA | Form | Asymmetry coverage |
|---|---|---|
| `otNone` | 1 | n/a (component algebra) |
| `otFRotation` | cos(2π(f·t) + φ) | ✅ `Oscillatory` |
| `otBRotation` | cos(γ_μB·t + φ) | ✅ `OscillatoryField` |
| `otScaledFRotation` | cos(2π(f·c·t) + φ), scale factor c | ⚠️ **gap** — see options doc (likely covered by `expr` parameter constraints) |
| `otKuboToyabe` | static ZF (`KTZ`), static LF (`KTB`), dynamic (`KTD`) | ✅ `StaticGKT_ZF`, `LongitudinalFieldKT`, `DynamicGaussianKT` |
| `otKeren` | Keren analytic dynamic-LF approximation | ✅ `Keren` |
| withdrawn: `otBessel`, `otKTdist`, `otDelayRot` | — | Bessel proposed for (re)introduction: MS-Intro Eq. 6.47 (incommensurate SDW); standard in musrfit/Mantid |

### Built-in relaxation types (`FitTyps.pas:132-142`)

| WiMDA | Form | Asymmetry coverage |
|---|---|---|
| `rtNone` | 1 | n/a |
| `rtLor` | exp(−λt) | ✅ `Exponential` |
| `rtGau` | exp(−(σt)²) | ✅ `Gaussian` (same convention) |
| `rtGau2` | exp(−(σt)²/2) | ➖ reparameterization of `Gaussian` (σ′=σ/√2) — recommend **skip**, document equivalence |
| `rtStr` | exp(−(λt)^β) | ✅ `StretchedExponential` |
| `rtRK` | Risch–Kehr: e^{Γt}·erfc(√(Γt)) | ❌ **gap — port** |
| `rtSig2` | exp(−s₂·t²), s₂=σ² | ➖ reparameterization of `Gaussian` — recommend **skip** |
| `rtAbragam` | exp(−(σ/ν)²[e^{−νt}−1+νt]) | ✅ `Abragam` |
| `rtFstr` | stretched exp with rate normalised to component's 2π×frequency | ⚠️ **gap** — decision needed (cross-parameter coupling) |
| withdrawn: `rtHH` | — | not ported |

### Muonium DLL (`Extrafunctions/muoniumfunctions.dpr`)

| WiMDA | Physics | Asymmetry coverage |
|---|---|---|
| `High TF Muonium` (`MuoniumPairRot`) | ν₁₂/ν₃₄ high-field pair, MS-Intro §4.4/Eq. 4.65 | ❌ **gap — port** |
| `PCR Hi TF Mu` (`AnisMuoniumPairRot`) | powder average of anisotropic (axial D) high-TF pair, MS-Intro Eqs. 4.66–4.68 | ❌ **gap — port** |
| `Low TF Muonium` | intratriplet pair | ✅ `MuoniumLowTF` |
| `TF Muonium` | all four TF transitions | ✅ `MuoniumTF` |
| `ZF muonium` | axial ZF triplet | ✅ `MuoniumZF` |
| `Mu LF reln` (`MuLFrel`) | muonium spin-exchange T₁ in LF (Kadono et al., PRL **64**, 665 (1990)) | ❌ **gap — port** (non-textbook source, flagged) |

### Dipolar DLL (`Extrafunctions/dipolarfunctions.dpr`)

| WiMDA | Physics | Asymmetry coverage |
|---|---|---|
| `Gau broad KT` (`GBKTB`, `KuboToyabe.pas:193-213`) | static LF Gaussian KT averaged over a Gaussian distribution of Δ (relative width) | ❌ **gap — port** |
| `F-u-F dip ZF PCR` (`FmuFdipole`, ω_d input) | linear F–μ–F powder average, MS-Intro Eq. 4.81 | ✅ `FmuF_Linear` (r-parameterized; ω_d variant trivially related) |
| `F-u-F dip ZF PCR` (`FmuFdipoler`, F–F distance input) | as above | ✅ `FmuF_Linear` |
| `dyn F-u-F ZF PCR` (`FmuFdyn`) | strong-collision dynamicization of F–μ–F (MS-Intro Eq. 5.30 framework) | ❌ **gap — port** |
| `F-u-F-F ZF PCR` (`Ftriangle` → `polarize.pas` 16×16 eigenproblem) | muon + 3 fluorines, triangle geometry (r₁,r₂,r₃) | ❌ **gap — port** |
| `uFFF eq tri ZF PCR` (`Fequitriangle`) | equilateral special case of the above | ❌ **gap** (special case; may fold into the triangle component) |
| `Dip gen ZF PCR` (`ZFdipgen`) | muon + single spin-J nucleus, dipolar + quadrupolar (Celio & Meier, HFI **18**, 435 (1984)) | ❌ **gap — port** (non-textbook source, flagged) |
| `Dipolar ZF PCR` (`ZFdipole`) | muon + single spin-½ dipole, B_dip input, transverse relaxation λ_t on the oscillating part (Meier, HFI **18**, 427 (1984); math = MS-Intro Eq. 4.80) | ⚠️ partial — `MuF` covers fluorine at distance r without the λ_t damping or B/r/nucleus generality |
| `Proton dip ZF PCR` (`ZFprotondipole`) | as above with proton at distance r | ❌ **gap** |
| `Electron dip ZF PCR` (`ZFelectrondipole`) | as above with electron-moment scaling | ❌ **gap** |

### Pressure-cell DLL (`Extrafunctions/pressurefunctions.dpr`)

| WiMDA | Physics | Asymmetry coverage |
|---|---|---|
| `BeCu ZF` | (1−f)·KT_ZF(Δ) + f·exp(−λt) empirical RIKEN BeCu cell signal | ⚠️ expressible today as a composite (`StaticGKT_ZF + Exponential`); **decision**: dedicated convenience component or skip |
| `BeCu LF 110G` | empirical T-parameterized polynomial rate, exp(−(λ(T)·t)^2.5) at 110 G | ⚠️ instrument/cell-specific calibration — recommend **skip**, document |

### Adjacent WiMDA features confirmed out of scope

- Table-fit / model functions of T, B (`fitfunctions.pas`) — next pass.
- Negative-muon lifetime analysis, double-pulse mode, count-loss/deadtime
  corrections (`AsymFitFunction.pas` wrapper layers) — instrument corrections,
  deadtime already ported separately.
- Relaxing stretched-exponential baseline — already expressible with Asymmetry's
  component algebra (`Constant`, products).

## Proposed port list (summary)

Nine genuinely new physics components, one generalization, two decisions:

1. **RischKehr** relaxation (built-in grid gap).
2. **MuoniumHighTF** — high-field ν₁₂/ν₄₃ pair.
3. **MuoniumHighTFAnisotropic** — powder-averaged axial-anisotropy pair.
4. **MuoniumLFRelaxation** — Kadono spin-exchange T₁.
5. **GaussianBroadenedKT** — Δ-distributed static LF KT.
6. **DynamicFmuF** — strong-collision dynamicized F–μ–F.
7. **FmuF_Triangle** — muon + 3 fluorines (general triangle + equilateral mode).
8. **DipolarSpinJ** — single spin-J with quadrupole (Celio–Meier).
9. **Bessel** (J₀ oscillation, SDW) — re-introduction of a function WiMDA itself
   withdrew; textbook-standard.
10. Generalize the single-dipole family (`MuF` ∪ `ZFdipole`/proton/electron
    variants): nucleus choice + optional transverse damping.
11. Decision: `ScaledFRotation` and `Fstr` (cross-parameter conveniences) — port
    vs. cover with `expr`/link-group machinery.
12. Decision: pressure-cell components — skip vs. "Sample environment" category.

See `implementation-options.md` for grouping (GUI submenu) proposals, parameter
defaults/limits policy, and the open decisions; `comparison.md` for per-function
math, conventions, and source flags; `verification-plan.md` for the acceptance
checks; `test-data.md` for fixtures.

## Study artifacts

- [comparison.md](comparison.md) — per-function math, WiMDA code refs, textbook
  mapping, convention differences.
- [implementation-options.md](implementation-options.md) — design options,
  grouping, defaults/limits, open decisions.
- [test-data.md](test-data.md) — reference data and fixtures.
- [verification-plan.md](verification-plan.md) — acceptance criteria.
