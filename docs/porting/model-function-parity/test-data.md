# Test Data: Model Function Parity

All function oracles are transcribed from `fitfunctions.pas` formulas and
evaluated to full double precision (values below computed 2026-06-10 with
numpy; embed as constants in the tests, pattern of
`tests/test_wimda_parity_components.py`).

## 1. Function oracles (exact, synthetic)

### 1.1 `Polynomial` (WiMDA `func0`)

Coefficients (c0…c5) = (0.5, −1.2, 0.8, −0.05, 0.002, −0.0001):

| x | y |
|---|---|
| 0.0 | 0.5 |
| 1.0 | 0.0519 |
| 2.5 | 1.787109375 |
| 7.0 | 17.2713 |
| 10.0 | 28.5 |

Plus: exact round-trip — generate y from known coefficients on an x grid
(no noise), fit with all six free, recover coefficients to ≲1e-6; repeat
with c3…c5 fixed at 0 fitting a quadratic.

### 1.2 `PowerLawQuadBG` (WiMDA `powerlawBGquad`)

(a, n, BG) = (2.0, 1.5, 3.0), y = √((a·xⁿ)² + BG²):

| x | y |
|---|---|
| 0.5 | 3.082207001484488 |
| 1.0 | 3.605551275463989 |
| 4.0 | 16.278820596099706 |
| 9.0 | 54.08326913195984 |

Limits: y(0) = |BG|; large-x asymptote → a·xⁿ; equality with
`PowerLaw(a,n,c=0)` when BG = 0.

### 1.3 `MuRepolarisation` (WiMDA `muonrep`)

WiMDA form y = a_Mu·(½ + (B/B₀)²)/(1 + (B/B₀)²) + a_Dia. Asymmetry fits
A_hf with B₀ derived: B₀[G] = A_hf[MHz] / 2.8160490 [MHz/G] where
2.8160490 = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G/(2π)
+ MUON_GYROMAGNETIC_RATIO_MHZ_PER_T·1e-4 (compute from `constants.py` in the
test, do not hard-code).

Vacuum muonium oracle: A_hf = 4463.302765 MHz ⇒ B₀ = 1584.952 G
(0.1585 T, textbook value ≈ 0.15853 T to book rounding).

(a_Mu, a_Dia) = (15.0, 8.0), A_hf = 4463.302765 MHz:

| B [G] | y |
|---|---|
| 0 | 15.5 (= a_Mu/2 + a_Dia) |
| 100 | 15.529737440954086 |
| 1584.952 (= B₀) | 19.25 (= a_Mu·0.75 + a_Dia) |
| 5000 | 22.3151898002739 |
| 20000 | 22.9531925884069 |
| ∞ limit | 23.0 (= a_Mu + a_Dia) |

Recovery test: generate exact y(B) on a log-spaced 0–20 kG grid from known
(a_Mu, A_hf, a_Dia), fit, recover A_hf ⇒ B₀ ≈ A/(γₑ+γ_μ) to ≲1e-6 relative.

### 1.4 Composite recipes

- **`Arrhenius + Arrhenius`** (WiMDA `func2`): WiMDA constant
  e/k = 1.60e-19/1.38e-23 = 11594.2029 K/eV; CODATA 11604.5181 K/eV
  (ratio 1.00088969, i.e. WiMDA Eₐ 0.089 % low). Oracle: evaluate the WiMDA
  formula with Eₐ = {0.10, 0.25} eV, amplitudes {5, 50} at
  T = {50, 100, 200, 300} K and assert the Asymmetry composite with
  Eₐ[meV] = 1000·Eₐ[eV]·(11604.5181/11594.2029) matches to ≲1e-12
  (constant-conversion identity), while the naive 1000× conversion agrees
  only to ~0.1 % — this pins down D4 numerically.
- **`OrderParameter + Constant`** (WiMDA `func5`): on the physical domain
  the forms are identical; assert equality of the two formulas (WiMDA
  transcription vs component sum) for (B₀=29.9, Tc=69.2, α=1.23, β=0.417,
  B_bg=3.0) at T = {5, 30, 60, 69.19, 69.2, 80} K (the last two exercise
  the clamp). Real-data check in §2.
- **2 Lorentzians + cubic**: assert `LorentzianLCR(f,B0,Bwid)` equals the
  transcribed WiMDA peak term `Ampl·Wid²/(Wid²+(x−Pos)²)` for
  (3.5, 1200, 80) on a field grid (algebraic identity), and that re-centring
  the WiMDA cubic (p7..p10, centre Pos1) to absolute-x coefficients
  reproduces `Polynomial` exactly (documents D2).

### 1.5 χ² quality bands (`fit_quality` helper)

`scipy.stats.chi2.ppf` oracle at R = 0.95 (α = 0.05):

| ν | χ²ᵣ band low | χ²ᵣ band high |
|---|---|---|
| 5 | 0.16624232269733252 | 2.566500398806005 |
| 10 | 0.3246972780236841 | 2.0483177350807393 |
| 20 | 0.4795388696132433 | 1.708480345141917 |
| 50 | 0.6471472739131731 | 1.4284039037501284 |

Verdicts: (χ² = 2, ν = 10) → overdone (CDF ≈ 0.0037 < 0.025);
(χ² = 25, ν = 10) → poor (CDF ≈ 0.9945 > 0.975); (χ² = 9, ν = 10) → good.
Edge cases: ν = 0 → no verdict; failed fit → no verdict; R clamped to
[0.5, 0.999] (WiMDA `FitOpt` behaviour, kept).

### 1.6 Estimate-errors-from-scatter (fixed-point equivalence)

Synthetic linear data, seed 42: x = linspace(1, 10, 12),
y = 2 + 0.7x + N(0, 0.35). Unweighted OLS gives
p = (1.84514937, 0.71914734), χ²(σ=1)/ν = 0.1202971846 (ν = 10),
rescaled errors (0.21917779, 0.03544948), and the WiMDA fixed point
errabs\* = √(χ²₁/ν) = 0.3468388453. Assertions:

1. scatter-mode `fit_parameter_model` reproduces the OLS parameters and the
   rescaled errors (≲1e-6);
2. explicitly iterating the WiMDA scheme (refit with constant σ = errabs,
   update errabs ← errabs·√(χ²ᵣ)) converges to errabs\* after **one**
   iteration and further iterations are stationary — the documented
   equivalence, tested rather than asserted in prose.

### 1.7 Error modes (Percent / Absolute / None)

Synthetic quadratic series with known per-point errors. Assert: Percent
gives σᵢ = pct·|yᵢ| (zero-y point masked out, D9); Absolute gives uniform σ;
None gives unit weights and identical parameters to Absolute (uniform
weights ⇒ same minimiser) but different parameter errors; Column applies the
stabilisation floor, the other modes bypass it (D10).

### 1.8 Union multi-range

λ(T) = `CriticalDivergence`(a=2, Tc=69.2, ν=0.7, c=0.05) + 3 % Gaussian
noise on T = linspace(40, 100, 61). Fit with windows
[40, 64] ∪ [74, 100] (critical region excluded): recover (a, Tc, ν, c)
within uncertainties; assert the mask equals the OR of the windows and that
a point in neither window is excluded; single-window behaviour falls back
to (x_min, x_max) exactly.

## 2. Corpus data (WiMDA Muon School, `~/Documents/WiMDA muon school/`)

| Dataset | Use | Status |
|---|---|---|
| `Magnetism/Magnetic ordering in EuO` (PSI .bin, runs 2928–2943) | `OrderParameter + Constant` recipe reproduces the PR #15 GUI-verified result: Tc = 69.2(1) K, β = 0.417(7), α = 1.23(5) | required (regression vs known numbers) |
| `Chemistry/ALC resonance in TCNQ` (integral-vs-field series via PR #23 time-integral observable) | exercise `MuRepolarisation` + `LorentzianLCR` machinery on a real B-scan; ALC resonances sit on a repolarisation-like baseline — qualitative only | optional / best-effort |
| Isotropic-Mu LF repolarisation B-scan | **none found in the corpus** (TCNQ is ALC; CdS is a TF shallow-donor study; Si is photo-μSR) — synthetic oracle (§1.3) is the quantitative test, recorded here so nobody hunts again | n/a |

Union multi-range on real data: the EuO λ(T) series diverges at Tc — fitting
`CriticalDivergence` to the EuO rate trend excluding [64, 74] K is the
canonical WiMDA workflow and doubles as the corpus multi-range check
(qualitative: fit converges and verdict is good/poor, no golden numbers).

## 3. Reference programs

- WiMDA itself (Wine/VM cross-check on an identical text table): optional,
  only if convenient — the Pascal-transcribed oracles above are the
  authoritative parity record.
- Mantid/musrfit: GPL — verification oracles only, never vendored. Not
  needed for Phase 1/2 (no overlapping model-function implementations
  beyond what the synthetic oracles cover).
