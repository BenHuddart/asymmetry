# Beta Correction — Reference Comparison

## musrfit (the reference for this port)

### Where β lives

- Declared in the msr RUN block as `beta <n>` or `beta funX` — always a
  reference to a FITPARAMETER entry (or FUNCTIONS entry), never an inline
  literal (`$MUSRFIT_SRC/src/classes/PMsrHandler.cpp:3577–3603`; `alpha`
  parsing `:3549–3575`). Whether the correction is *fitted* or *fixed* is
  decided solely by the referenced parameter's step (`fStep == 0` → fixed).
- `alpha` is **mandatory** for fit type 2 (`PRunAsymmetry.cpp:129–134` errors
  without it); `beta` is optional — omitted means β ≡ 1 (`:161–162`).
- Stored per run as parameter numbers: `fAlphaParamNo` / `fBetaParamNo`
  (`$MUSRFIT_SRC/src/include/PMusr.h:1210–1211`, default −1).

### The α/β case tags

`PRunAsymmetry.cpp:152–183` classifies each run into `fAlphaBetaTag`:

| tag | condition | data asymmetry | theory transform (`CalcTheory`, `:550–601`) |
|-----|-----------|----------------|---------------------------------------------|
| 1 | α ≡ 1, β ≡ 1 | `(f−b)/(f+b)` | `f` (identity) |
| 2 | α ≠ 1, β ≡ 1 | `(αf−b)/(αf+b)` | `(f(α+1)−(α−1)) / ((α+1)−f(α−1))` |
| 3 | α ≡ 1, β ≠ 1 | `(f−b)/(βf+b)` | `f(β+1) / (2−f(β−1))` |
| 4 | α ≠ 1, β ≠ 1 | `(αf−b)/(αβf+b)` | `(f(αβ+1)−(α−1)) / ((α+1)−f(αβ−1))` |

"≡ 1" means the referenced parameter is *fixed and exactly 1.0*; a free
parameter (or one fixed at another value) always takes the correction branch.
`CalcChiSquare` (`:310`) uses the tag-4 formula uniformly with a = b = 1
substituted for the unity cases. Values are re-read every iteration —
parameter (`par[no−1]`) or function (`EvalFunc`), `:243–293`.

### Data asymmetry and error (the pipeline this port mirrors)

`PRunAsymmetry::PrepareData` (`:610+`): raw F/B → t0 alignment → background
subtraction (fixed `:894/:918/:926` or estimated `:933/:1044`) → packing
(`:1083+`) → asymmetry per packed bin (`:1404–1421`):

    asym  = (α·f − b) / (α·β·f + b)                       // :1412
    error = 2/((f+b)²) · √(b²·σf² + σb²·f²)               // :1418 — α/β-INDEPENDENT

α and β enter **only** the value, never musrfit's error. The RRF variant
(`PRunAsymmetryRRF.cpp:1321/:1327`) and the BNMR helicity-difference variant
(`PRunAsymmetryBNMR.cpp:1514/:1527`) repeat the same structure.

### Automatic estimation

None, for either α or β. (Exception: fit type 5 BNMR estimates α from
helicity-summed count ratios per the manual.) musredit's RUN-block dialog
(`$MUSRFIT_SRC/src/musredit_qt6/musredit/PGetAsymmetryRunBlockDialog.cpp:87–88,
129–163`) is a plain `QIntValidator` line edit for the parameter number.

## musrfit user manual (formulas)

From "Asymmetry Fit (fit type 2)" and "The RUN Block":

- `α = N₀,b/N₀,f`, `β = A₀,b/A₀,f`; both default 1 when not specified.
- Raw: `a(k) = [(N_f−B_f) − (N_b−B_b)] / [(N_f−B_f) + (N_b−B_b)]`
- Fit-space: `a(t) = [(αβ+1)A(t) − (α−1)] / [(α+1) − (αβ−1)A(t)]`
- Display-space (rearranged):
  `A(t) = [(α−1) + (α+1)a(t)] / [(αβ+1) + (αβ−1)a(t)]
        = [α(N_f−B_f) − (N_b−B_b)] / [αβ(N_f−B_f) + (N_b−B_b)]`

## Asymmetry (current baseline)

- `core/transform/asymmetry.py` — `A = (F − αB)/(F + αB)`, α on backward
  (`α_ours = 1/α_musrfit`); exact Poisson error
  `σ_A = 2|α|·√(F·B·(F+B))/(F+αB)²` with the num/den covariance kept
  (`:56–88`); count-error variant `:93–135`.
- Applied per output bin after counts-then-ratio binning
  (`core/transform/rebin.py::binned_fb_asymmetry`), fed by the corrected
  pipeline deadtime → grouping → background
  (`core/transform/reduce.py::corrected_grouped_counts`).
- No β anywhere in core or GUI (verified by grep — every existing `beta` is a
  stretched-exponential relaxation exponent in the fitting models).

### Convention mapping (ours ↔ musrfit)

Multiply musrfit's numerator and denominator by `α_ours = 1/α_musrfit`:

    musrfit: (α_m·f − b)/(α_m·β·f + b)  ≡  (f − α_o·b)/(β·f + α_o·b)  :ours

so **β is numerically identical in both conventions** (`β = A₀,b/A₀,f`), and
our port is `A = (F − αB)/(βF + αB)`. Pinned by an equivalence test
(`verification-plan.md`).

## WiMDA — no equivalent (do not conflate)

WiMDA's `AFbeta` ("Bsln beta", `$WIMDA_SRC/src/Analyse.pas:7035`, model use
`AsymFitFunction.pas:688`) is the **stretched-exponential exponent** of the
baseline relaxation `exp(−(λt)^β)` — a relaxation shape parameter fitted in
the model, unrelated to detector balance. WiMDA corrects detector balance with
α only. Mantid's `AsymmetryCalc`/`AlphaCalc` likewise have no
asymmetry-amplitude balance.

## Divergences (deliberate, recorded)

1. **Error propagation.** musrfit's data error is α/β-independent (raw-a
   propagation); we propagate exactly through the corrected formula
   (`var(A) = α²(1+β)²FB(F+B)/(βF+αB)⁴`). Same class of divergence already
   recorded for α in `docs/porting/asymmetry-error-propagation/`.
2. **Where the correction lives.** musrfit warps the *theory* into raw-a
   space during fitting; we bake α/β into the reduced curve the fitter sees.
   Equivalent for fixed α/β; fitting β free requires the count-domain path
   (deferred).
3. **Declaration.** musrfit's β is a fit-parameter reference; our v1 β is a
   fixed scalar on the grouping/profile (matching how our α is stored), with
   the fixed-at-1 collapse mirrored by omitting the key when β = 1.
