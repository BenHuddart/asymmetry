# Model Function Parity (WiMDA Model layer)

Status: **study complete, implementation plan agreed** (decisions with Ben
2026-06-10; plan in [implementation-options.md](implementation-options.md));
awaiting go-ahead to implement. Parent umbrella:
`docs/porting/wimda-parity-gap/` (project brief
`projects/model-function-parity.md`, Wave A, size M, 2 phases).

## Problem statement

WiMDA's Model layer fits parametric models to *trended fit parameters* — the
second analysis level, after time-domain fitting: λ(T), ν(T), amplitude(B),
rate(B), and so on. Asymmetry's `core/fitting/parameter_models.py` already
exceeds WiMDA in breadth (31 components including the superconducting-gap
library, plus cross-group global/local fitting WiMDA has no equivalent of),
but specific WiMDA model *functions* and fitting *machinery* options are
missing. This study verifies the umbrella brief's gap tables directly against
the WiMDA Pascal source and fixes the implementation approach.

Primary scientific reference: S. J. Blundell, R. De Renzi, T. Lancaster,
F. L. Pratt, *Muon Spectroscopy: An Introduction* (OUP, 2022). Per the
established documentation convention, user-facing text never cites equations
by number; APS-style reference lists live in
`core/fitting/component_docs.py` (`PARAMETER_MODEL_REFERENCES`).

## How WiMDA structures the Model layer

The fit table (one line per run: x columns + fitted parameters with errors) is
plain text; the Model form (`Model.pas`) selects an X column, a Y column and
optionally an error column, builds (x, y, σ) triples, and minimises χ² with
the same 1972 Gauss–Newton engine (`Fitucode.pas FITE`) used for time-domain
fits. The function library is a registry built at startup
(`fitfunctions.pas init_database_std`, 8 functions) plus optional user
`*fit.dll` libraries discovered at runtime (`UserUnit/` API: `getfnlist` /
`getparams` / `details` / optional `getresults`, with uppercase FORTRAN
variants). Every function has signature `f(x, x2, p)`; the second analytic
variable `x2` is plumbed through the table machinery but **none of the eight
built-ins reads it** — it reaches user DLLs only.

Asymmetry's equivalent is the `PARAMETER_MODEL_COMPONENTS` registry
(composable, scoped per x-variable) driven by `ModelFitDialog`, fitted with
iminuit. Architecture is strictly more general except for the specific gaps
below.

## Verified gap inventory (functions)

All eight `fitfunctions.pas` functions were read and transcribed
([comparison.md](comparison.md) has the per-function math and divergences):

| WiMDA function (`fitfunctions.pas:216–305`) | Status in Asymmetry | Action |
|---|---|---|
| Polynomial fit up to fifth order | ABSENT | **Add `Polynomial`** (c₀…c₅, unused coefficients fixable at 0) |
| Power law | PRESENT (`PowerLaw`, safer \|x\| guard) | none |
| Power law (BG quad) | ABSENT (no √ in composite algebra) | **Add `PowerLawQuadBG`**; generic quadrature combinator recorded as deferred design note |
| 2 Lorentzians + cubic BG | EXPRESSIBLE | Docs recipe: `LorentzianLCR + LorentzianLCR + Polynomial`; coefficients do **not** transfer 1:1 (WiMDA centres the cubic on Pos 1) |
| Thermal activation (2 component) | EXPRESSIBLE | Docs recipe: `Arrhenius + Arrhenius`; document eV→meV conversion and WiMDA's 0.089 % constant error |
| Internal field vs T for ordered magnet | EXPRESSIBLE | Docs recipe: `OrderParameter + Constant`; keep physical clamp above T꜀ |
| Divergence of relaxation rate | PRESENT (`CriticalDivergence`) | Naming map in docs |
| Repolarisation of isotropic Mu | ABSENT as component | **Add `MuRepolarisation`** — WiMDA's formula is exactly the textbook time-averaged isotropic-Mu polarization (verified); parameterise (a_Mu, A_hf in MHz, a_Dia), derive B₀ = A/(γₑ+γ_μ) from `core/utils/constants.py`; scope `field` |

## Verified gap inventory (machinery)

| Feature (`Model.pas`) | WiMDA behaviour (verified) | Action |
|---|---|---|
| Error modes (`getmodeldata`, 560–654; radio group in `Model.dfm`) | Column / Percent (σ=pct·y per point) / Absolute (constant σ from text box) / None (σ=1) / Estimate (σ=constant, box rescaled by √χ²ᵣ after each fit — manual iteration) | **Error-mode selector** in core + dialog; Estimate replaced by its fixed point: one-pass unweighted fit + post-hoc parameter-error rescale by √(χ²/dof) |
| Union multi-range (`Modelxrange.pas`, `getxranges`/`getmodeldata`) | Up to 20 (from, to) windows from a memo, OR-combined into one mask; one model over the union | **Multi-window fit ranges**: `ModelFitRange` carries a list of windows |
| χ² quality (`Chi2UpdateClick`, 1447–1516; `Numlib chilow/chihigh`; `FitOpt Rgoodfit`) | Two-sided χ² CDF test at confidence R (default 0.95, clamp [0.5, 0.999]): CDF < (1−R)/2 → "overdone", > (1+R)/2 → "poor", else "good"; displays χ²ᵣ target band [χ²(α/2)/ν, χ²(1−α/2)/ν] | **Shared core helper** (scipy.stats.chi2) + verdict display in `ModelFitDialog`; `fit-workflow-diagnostics` (Wave B) will reuse it |
| Arbitrary X column (`columnscan`, X/Y/E column spinners) | Any fit-table column as x (or y, or σ) | **Stretch** (param-vs-param trending); else follow-on |
| x2 second analytic variable | Column or fixed value; distinct-value list for per-x2 plot curves; only user DLLs consume it | **Out of scope** — requirements recorded in comparison.md; cross-group global/local fitting covers the workflow |
| Second-level Model Fit Table (`ModelFitTableUnit.pas`) | A plain text-table editor auto-saving model-fit rows (χ²ᵣ + parameter/error pairs) | Follow-on: "send model-fit results to the results table" so trending recurses naturally |
| `*fit.dll` user models (`LoadModelDLLs`, 1133–1242) | Runtime DLL discovery, Delphi + FORTRAN entry points | → `python-user-functions` project (Wave B) |
| ReloadFit (`.mfit` reparse) | Restores parameters by reparsing the fit log | Superseded by `.asymp` persistence; none |

## Scope decisions (agreed 2026-06-10)

- **Session-only state**: model fits are not persisted in `.asymp` today; the
  new machinery stays session-only. "Persist model fits in projects" recorded
  as a follow-on.
- **GUI exposure in `ModelFitDialog` only**: core helpers are written
  generically, but the cross-group fit dialog keeps its current UI; exposure
  there is a follow-on.
- Composite recipes (activation, ordered magnet, 2-Lorentzian LCR) are
  **documented, not coded** — the component algebra already expresses them.
- GPL programs (Mantid, musrfit) serve as verification oracles only.

## Study artifacts

- [comparison.md](comparison.md) — per-function transcriptions, machinery
  semantics, every WiMDA divergence with both behaviours stated, x2
  requirements record.
- [implementation-options.md](implementation-options.md) — design choices,
  agreed decisions, and the phase-by-phase implementation plan.
- [test-data.md](test-data.md) — numerical oracles (WiMDA-transcribed) and
  corpus mapping.
- [verification-plan.md](verification-plan.md) — acceptance criteria.
