# Project brief: model-function-parity

Umbrella: `wimda-parity-gap` · Wave A · Size M (2 phases) · pre-identified by
Ben as the parameter-domain analogue of `wimda-fit-function-parity`

## Motivation

WiMDA's Model layer fits parametric models to trended fit parameters.
Asymmetry's `parameter_models.py` already exceeds WiMDA in breadth (31
components incl. the SC gap library), but specific WiMDA functions and
fitting-machinery options are missing. A focused deep-dive (2026-06-10)
produced the full comparison below; the project's study pass should lift it
verbatim and verify.

## WiMDA reference

`fitfunctions.pas:216–305` ("Standard fit models", 8 functions — none reads
x2; x2 reaches user DLLs only); `Model.pas` (error modes `getmodeldata`
560–654, Estimate rescale 685–690, x2 plumbing, ReloadFit 1541–1616, DLL
loading 1133–1242); `Modelxrange.pas` (union multi-range, up to 20
intervals); `ModelFitTableUnit.pas` (second-level results table);
`UserUnit/` (`*fit.dll` API incl. FORTRAN variant).

## Function gap table (from the deep-dive)

| WiMDA function | Status | Plan |
|---|---|---|
| Polynomial (≤5th order) | ABSENT | Add `Polynomial` component (c0…c5, unused coefficients fixable at 0) |
| Power law | PRESENT (`PowerLaw`, safer \|x\| guard) | none |
| Power law (BG in quadrature) | ABSENT (no sqrt in composite algebra) | Dedicated `PowerLawQuadBG`; note a generic quadrature combinator as a deferred idea |
| 2 Lorentzians + cubic BG | PARTIAL | `LorentzianLCR + LorentzianLCR + Polynomial`; do **not** port WiMDA's Pos-1-centred BG coupling or exact-zero amplitude test — document that coefficients don't transfer 1:1 |
| Thermal activation (2-component) | EXPRESSIBLE: `Arrhenius + Arrhenius` | Docs recipe; keep CODATA k_B and meV (WiMDA: eV with 3-sig-fig constants, ≈0.3% off) — document conversion |
| Internal field vs T (ordered magnet) | EXPRESSIBLE: `OrderParameter + Constant` | Docs recipe; keep clamp-above-Tc (WiMDA's abs() mirroring is unphysical — do not replicate) |
| Divergence of relaxation rate | PRESENT (`CriticalDivergence`) | none (naming map in docs) |
| Repolarisation of isotropic Mu | ABSENT as component (awkwardly `Constant − Lorentzian`) | Add `MuRepolarisation` — **WiMDA's formula is the textbook isotropic-Mu time-averaged polarization** (verified), so the port is the physically-correct form. Prefer (a_Mu, A_hf MHz, a_Dia) parameterisation deriving B₀ = A/(γ_e+γ_μ) from `core/utils/constants.py`; scope `field`; pairs with the time-integral observable (PR #23) |

## Machinery gaps

| Feature | Status | Plan |
|---|---|---|
| Error modes Percent / Absolute / None | PARTIAL (Column always; None core-only) | Small core transform of yerr + GUI selector |
| Error mode Estimate (σ ← σ·√χ²ᵣ iterated) | ABSENT | Modern equivalent: one-pass unweighted fit + post-hoc parameter-error rescale by √(χ²/dof) (the fixed point of WiMDA's manual iteration) — an "estimate errors from scatter" toggle |
| Union multi-range (one model over interval union) | DIFFERENT (current ranges fit independent models) | Support a list of (min,max) windows OR-combined per range — small mask change + GUI |
| χ² quality band (good/poor/**overdone**) | PARTIAL (χ²ᵣ only) | Shared with `fit-workflow-diagnostics` — implement the scipy.stats.chi2 helper once, in whichever lands first; coordinate |
| Arbitrary X column (param-vs-param trends) | ABSENT (x ∈ field/T/run) | Phase 2; scopes mechanism degrades gracefully to "common" |
| x2 second analytic variable | ABSENT | **Out of scope** — cross-group global/local fitting already covers the simultaneous-multi-field workflow; the deep-dive's `ParameterCompositeModel` requirements list is recorded in the study for if a genuinely two-variable component is ever wanted |
| Second-level Model Fit Table | ABSENT | Phase 2 as "send model-fit results to the results table" so trending recurses naturally — cleaner than a separate table widget |
| `*fit.dll` user models | ABSENT | → `python-user-functions` (registries are plain dicts; design there covers `details`/`getresults`) |
| ReloadFit (.mfit reparse) | superseded by `.asymp` persistence | none |

## Phasing

**Phase 1 — functions**: `Polynomial`, `MuRepolarisation`,
`PowerLawQuadBG`; composite recipes + unit-delta documentation; applicability
text + references per `component_docs.py` conventions (the
`test_fit_function_docs.py` enforcement applies).
**Phase 2 — machinery**: error-mode selector incl. scatter-estimate;
union multi-range; χ² quality indicator; arbitrary-X and results-table
recursion if session budget allows (else record as follow-on).

## Conflicts & dependencies

Primary surfaces: `parameter_models.py`, `model_fit_dialog.py`. Wave
A-disjoint. `python-user-functions` (Wave B) generalises these registries —
land first. χ²-band helper shared with `fit-workflow-diagnostics`.

## Verification sketch

Per-function numerical oracles transcribed from `fitfunctions.pas` (same
pattern as `tests/test_wimda_parity_components.py`); EuO `OrderParameter +
Constant` recipe reproduces PR #15 results; `MuRepolarisation` on a corpus
LF B-scan (Chemistry/Semiconductors PSI sets) or exact synthetic — recover
B₀ ≈ A/(γ_e+γ_μ); union multi-range on a λ(T) divergence series excluding
the critical region (the canonical WiMDA use); WiMDA cross-check via
Wine/VM on an identical table if convenient.
