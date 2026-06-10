# Implementation Options: WiMDA Fit Function Parity

## Architecture (settled by prior ports)

All new functions are baseline-free entries in `COMPONENTS`
(`core/fitting/composite.py`), with:

- pure-numpy/scipy evaluation functions in `asymmetry.core` (no Qt),
- `ParamInfo` entries in `core/fitting/parameters.py` (unicode/latex/GLE labels,
  units, `default_min`),
- user-facing applicability text in `core/fitting/component_docs.py`, written to
  MS-Intro notation with equation citations,
- tests beside the behavior in `tests/`.

Heavy components (dynamic F–µ–F, triangle F–µ–F–F, broadened KT) follow the
established grid + LRU-cache pattern (`_dynamic_kt_grid`,
`_general_spectral_terms_cached`). No fitting-engine or serialization changes
are needed: new components are forward-compatible additions to the `.asymp`
schema (old projects unaffected; projects using new components simply require a
newer version to open, consistent with prior component additions).

## Proposed components

Names follow existing registry style. Defaults assume the common Asymmetry
starting point A = 25 % and rates of order 0.5–1 µs⁻¹.

| # | Component | Params (defaults) [limits] | Notes |
|---|---|---|---|
| 1 | `RischKehr` | A (25), Gamma (1.0 µs⁻¹) [Γ ≥ 0] | `erfcx(√(Γt))`; drop WiMDA's Γ<0 mirror branch |
| 2 | `Bessel` | A (25), frequency (1.0 MHz) [ν ≥ 0], phase (0 rad) | A·J₀(2πνt + φ); category General |
| 3 | `MuoniumHighTF` | A (25), field (G) [B ≥ 0], A_hf (4463 MHz) [>0], phase (0) | ½(cos ω₁₂ + cos ω₃₄); positive-frequency convention |
| 4 | `MuoniumHighTFAniso` | A (25), field, A_hf (4463), D (10 MHz), phase (0) | powder-averaged axial pair; Gauss–Legendre angular grid |
| 5 | `MuoniumLFRelax` | A (25), delta_ex (MHz) [≥0], tau_c (µs) [≥0], B_LF (G) [≥0], A_hf (4463, fixed default) | re-derive λ(B) from Kadono PRL 64, 665 — do **not** transliterate the WiMDA expression |
| 6 | `GaussianBroadenedKT` | A (25), Delta (0.5 µs⁻¹) [≥0], B_LF (0 G) [≥0], width (0.2 rel.) [0 ≤ w] | Gauss–Hermite average of static LF KT over Δ distribution |
| 7 | `DynamicFmuF` | A (25), r_muF (1.17 Å) [>0], nu (1 MHz) [≥0] | strong-collision solver reused from dynamic KT; ν=0 → `FmuF_Linear` |
| 8 | `FmuF_Triangle` | A (25), r1, r2, r3 (Å) [>0] | 16-dim eigen-solve, powder average, LRU cache; equilateral = r1=r2=r3 via link groups (no separate component) |
| 9 | `DipolarSpinJ` | A (25), f_dip (MHz) [≥0], f_quad (0 MHz), J (3/2) [fixed by default; half-integer steps] | Celio–Meier closed form |
| 10 | `DipolarPair` (generalizes `MuF`) | A (25), B_dip (G) [≥0] **or** r+nucleus, lambda_T (0 µs⁻¹) [≥0] | Meier spin-½ pair, e^{−λ_T t} on oscillating part only; see Decision 3 |

Positivity policy: every rate, width, field, distance, and frequency above gets a
hard `min = 0` (or strictly positive where 1/r³ appears); phases unbounded;
amplitudes bounded below at 0 by default but overridable (negative amplitudes
are legitimate for some geometries — keep A free where the existing components
do). β-type exponents bounded (0, 2] following the stretched-exponential
precedent.

## GUI grouping (submenus)

`fit_function_builder.py` already groups the picker by
`ComponentDefinition.category`. Current categories: General, Muonium,
Muon-Fluorine, Frequency Domain. With ~10 new components, "General" becomes
unwieldy. Proposed re-categorization (pure metadata, no architecture change):

- **Relaxation** — Exponential, Gaussian, StretchedExponential, RischKehr,
  Abragam, Keren, MuoniumLFRelax(?)
- **Oscillation** — Oscillatory, OscillatoryField, Bessel
- **Kubo-Toyabe** — StaticGKT_ZF, LongitudinalFieldKT, DynamicGaussianKT,
  DynamicLorentzianKT, GaussianBroadenedKT
- **Muonium** — MuoniumTF, MuoniumLowTF, MuoniumZF, MuoniumHighTF,
  MuoniumHighTFAniso (+ MuoniumLFRelax if not under Relaxation)
- **Nuclear dipolar** (renames Muon-Fluorine) — MuF/DipolarPair, FmuF_Linear,
  FmuF_General, FmuF_Triangle, DynamicFmuF, DipolarSpinJ
- **Background** — Constant
- **Frequency Domain** — unchanged

Existing saved projects only store component names, not categories, so renaming
categories is serialization-safe.

## Open decisions (for joint sign-off before implementation)

### Decision 1 — `ScaledFRotation` and `Fstr`: port vs. document recipe

Both are cross-parameter conveniences (frequency × scale; rate × frequency).
Asymmetry's `expr` constraints and link groups already express them.

- **Option A (recommended): don't port.** Add a short "WiMDA migration" note in
  the user guide showing the `expr` recipe for each.
- **Option B: port as components** with an extra scale parameter
  (self-contained, but duplicates machinery and bloats the menu).

### Decision 2 — Pressure-cell functions

- **Option A (recommended): skip both.** `BeCu ZF` is exactly
  `StaticGKT_ZF + Exponential` (document as a preset/recipe); `BeCu LF 110G` is
  a single-instrument empirical calibration curve.
- **Option B: add a "Sample environment" category** with both, for one-to-one
  WiMDA familiarity.

### Decision 3 — Single-dipole parameterization

WiMDA offers the spin-½ dipole three ways (B_dip; proton r; electron r);
Asymmetry has `MuF` (fluorine r).

- **Option A (recommended): one `DipolarPair` component parameterized by ω_d
  (MHz) + optional λ_T**, plus keeping `MuF` as-is; document the ω_d ↔ (γ_j, r)
  conversion in the component info (a small table for ¹⁹F, ¹H, e⁻). Avoids a
  per-nucleus component explosion and matches MS-Intro Eq. 4.76 notation.
- **Option B: nucleus-choice enum parameter** (fits the WiMDA UX more closely
  but introduces a non-numeric parameter type the engine doesn't currently
  support).
- **Option C: three thin components** (`DipolarPairField`, `ProtonDipole`,
  `ElectronDipole`) mirroring WiMDA exactly.

### Decision 4 — `rtGau2`/`rtSig2` reparameterized Gaussians

- **Option A (recommended): skip;** document σ-convention mapping
  (σ_Gau2 = σ/√2, s₂ = σ²) in the `Gaussian` component info text.
- **Option B: add `GaussianKT`-style σ²/2 variant** for textbook-notation
  familiarity (MS-Intro Eq. 5.12 uses Δ²t²/2).

### Decision 5 — Bessel re-introduction

WiMDA withdrew `otBessel`; including it exceeds strict parity but closes a real
data-type gap (incommensurate SDW). Recommend **include** (it is 5 lines of
scipy and textbook-documented).

### Decision 6 — Phasing of the port

- **Option A (recommended): one branch, one PR**, components landed in 3 commits
  (relaxation/oscillation; muonium; dipolar) with tests each. Single review.
- **Option B: three stacked PRs** by group (smaller reviews, more overhead).

## Effort & risk notes

- Low risk (closed forms): RischKehr, Bessel, MuoniumHighTF, DipolarSpinJ,
  DipolarPair — direct formulas + tests.
- Medium: MuoniumHighTFAniso (angular average convergence),
  GaussianBroadenedKT (quadrature choice, cache).
- Higher: DynamicFmuF (Volterra solver reuse; performance), FmuF_Triangle
  (16-dim eigensolve; must reverse-engineer `matrices.pas` `f3calc` geometry
  conventions and validate against `FmuF_General` in the 2-spin limit),
  MuoniumLFRelax (re-derivation from the paper; WiMDA expression suspect).
- The fit wizard (`fit_wizard.py`) references a candidate-template list; new
  components are opt-in there and can be added selectively later — out of scope
  for this port except where trivially safe.
