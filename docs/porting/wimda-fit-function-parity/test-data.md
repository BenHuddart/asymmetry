# Test Data: WiMDA Fit Function Parity

## Synthetic fixtures (primary)

Each new component gets golden-value tests generated from an independent
implementation (not a copy of the production code path):

- **RischKehr** — compare against `mpmath`/high-precision
  `exp(g)·erfc(sqrt(g))` at g ∈ {1e-6, 0.1, 1, 10, 19.9, 20.1, 100, 1e4}
  (bracketing WiMDA's asymptotic switch at g = 20 to prove the erfcx
  implementation is seamless there).
- **Bessel** — `scipy.special.j0` vs the MS-Intro Eq. 6.45 integral
  (1/π)∫ dB cos(γB t)/√(B₁²−B²) evaluated by quadrature.
- **MuoniumHighTF / MuoniumHighTFAniso** — at D = 0 the anisotropic component
  must equal the isotropic pair; the pair frequencies must equal the ν₁₂, ν₃₄
  extracted from the already-verified `MuoniumTF` component at the same (B,
  A_hf); ν₁₂+ν₃₄ = A_hf (MS-Intro Eq. 4.65).
- **MuoniumLFRelax** — golden values from a symbolic re-derivation
  (sympy notebook committed under `tests/porting/wimda-fit-function-parity/`)
  of λ(B) from Kadono et al.; disagreement with WiMDA's expression is expected
  and must be quantified and recorded in `comparison.md`.
- **GaussianBroadenedKT** — width → 0 must reduce to `LongitudinalFieldKT`
  within 1e-9; finite width verified against brute-force trapezoid integration
  over the Δ distribution.
- **DynamicFmuF** — ν = 0 must reproduce `FmuF_Linear` exactly; large-ν tail
  vs the motional-narrowing exponential; mid-regime vs an independent
  Volterra integral-equation solve at loose grid tolerance.
- **FmuF_Triangle** — degenerate geometries: r₃ → ∞ must approach
  `FmuF_Linear`-like 3-spin behavior (cross-check against `FmuF_General` with
  matching geometry); equilateral case symmetry checks (invariance under
  distance permutations).
- **DipolarSpinJ** — J = ½, f_quad = 0 must equal the Meier spin-½ pair
  (`DipolarPair`/`MuF` math, MS-Intro Eq. 4.80); J = ½ with quadrupole must be
  quadrupole-independent (spin-½ has no quadrupole moment) — note WiMDA does
  not enforce this; decide whether to warn or absorb.
- **DipolarPair** — λ_T = 0 against MS-Intro Eq. 4.80 closed form; ω_d ↔ r
  conversion against the constants used by `MuF`.

## WiMDA cross-check traces (secondary)

Where practical, digitized curves evaluated by WiMDA itself (run under Wine or
on the instrument PC) for identical parameters, stored as small CSV golden
files. Required at minimum for: `GaussianBroadenedKT` (quadrature scheme
differs by design — record expected deviation), `FmuF_Triangle` (geometry
convention check vs `polarize.pas`), `MuoniumHighTFAniso` (15-point vs
Gauss–Legendre grid). Tolerances per `verification-plan.md`. If a WiMDA runtime
is unavailable, transliterate the Pascal into a throwaway Python reference under
`tests/porting/wimda-fit-function-parity/` (study scaffolding, not shipped).

## Real data (acceptance smoke tests)

From the local testing corpus (WiMDA Muon School data, `~/Documents`; see
`docs/testing/`):

- **F–µ–F**: any fluoride run in the corpus (or published PTFE-like spectrum,
  MS-Intro Fig. 4.22) — `FmuF_Linear` vs `DynamicFmuF` (small ν) sanity fit.
- **Kubo-Toyabe family**: ZF/LF runs already used by the dynamic-relaxation
  port tests — confirm `GaussianBroadenedKT(width→0)` reproduces published Δ.
- **Muonium**: quartz/CdS-type runs used in the muonium-triplet port —
  high-TF pair fits vs full `MuoniumTF` fits should agree on A_hf at high field.

No new instrument data is required; real-data checks are qualitative
(fit converges, parameters physical), with quantitative goldens carried by the
synthetic fixtures.
