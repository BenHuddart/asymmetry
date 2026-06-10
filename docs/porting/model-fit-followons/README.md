# Model-fit follow-ons (WiMDA Model layer, second pass)

Status: **items 1–3 shipped** (2026-06-10, PR #38 merged to `main`); the four
remaining follow-ons (quadrature `⊕`, single-fit-range export, cross-group
x-uncertainty, cross-fit global accumulation) are designed and being landed in a
**second pass** on `feat/model-fit-finish` — see
[implementation-options.md §"Second pass — finishing follow-ons A–D"](implementation-options.md). Implements the Model-layer follow-ons recorded
by the merged *model-function-parity* project (PR #32, `d64820c`), namely items
1, 2, 4 and the relevant slice of item 5 of its
[`implementation-options.md` §Follow-ons](../model-function-parity/implementation-options.md).
Parent umbrella: [`docs/porting/wimda-parity-gap/`](../wimda-parity-gap/README.md)
(Wave A).

This study is the backbone for four scoped deliverables, in order:

1. **Arbitrary X column** — trend any fitted parameter against any *other*
   fitted parameter (param-vs-param), not only run-level temperature/field/run.
2. **Cross-group exposure of error modes + fit windows** — core support in
   `global_fit_parameter_model` first, *then* un-hide the controls on
   `CrossGroupFitDialog`.
3. **Send model-fit results to the results table** — route cross-group fit
   outputs back into the fit-parameters panel as a new trendable series, so
   trending recurses (supersedes WiMDA's second-level Model Fit Table).
4. **STRETCH ONLY** — generic quadrature combinator (`⊕`) in the composite
   grammar. Implemented only if 1–3 land green with budget to spare; otherwise
   its design state is recorded and it stays a follow-on.

## Why a second study

The first pass (*model-function-parity*) shipped the **core** machinery these
build on — `ErrorMode`/`apply_error_mode`, `ModelFitRange.windows`/
`windows_mask`, the shared `fit_quality.py` verdict helper, and
`fit_parameter_model(error_mode=…, windows=…)` — but deliberately confined the
GUI to the single-series `ModelFitDialog`, and never reached the arbitrary-X
stretch (it is woven through the ~4 kloc `fit_parameters_panel.py`, a
panel-level feature). The four items here are exactly the deferred Model-layer
surfaces. The core is already x-agnostic and already carries the error-mode /
window plumbing; the work is mostly **GUI wiring + one core gap**
(`global_fit_parameter_model` ignores error modes and windows).

## Decisions taken before the study (attended session, 2026-06-10)

| # | Question | Decision |
|---|---|---|
| A | Item 3 target surface & recursion model | **Same panel, new series.** Cross-group outputs become a dedicated *Model fit results* `_GroupFitData` series inside `FitParametersPanel`; recursion falls out of item 1's arbitrary-X (pick any column — incl. a model param — as the next x). |
| B | Item 3 row sources | **Cross-group local params + cross-group global params.** Single-fit `ModelFitDialog` ranges are *not* exported (recorded as a follow-on). The cross-group per-group local parameters are the natural many-row source that makes re-trending meaningful. |
| C | Item 1 x-uncertainty | **Opt-in effective-variance (Orear/York) + plot bars.** A default-OFF "Account for x uncertainty" toggle, enabled only when x is a fitted parameter, inflating each point's variance by `(df/dx)²·σ_x²` inside the existing iminuit cost (preserves bounds/seeding; reduces *exactly* to current χ² when off). Horizontal x error bars are drawn on the trend plot whenever x is a fitted parameter, independent of the toggle. Documented as a physical-correctness divergence from WiMDA. |

Rationale for C is in [comparison.md §3](comparison.md); ODR/total-least-squares
was rejected because ODRPACK has no box-constraint support and our parameter
models lean on bounds heavily.

## How WiMDA does the arbitrary-X part (verified directly, not from summary)

WiMDA's fit table (`FitTableForm.FitTable`) is plain text: one line per run,
whitespace/comma-separated columns (run-level x columns + fitted parameters and
their errors). `Model.pas` exposes four independent `TSpinEdit` column pickers —
`Xcolumn`, `Ycolumn`, `Ecolumn`, `X2column` (`Model.pas:18–40`) — each clamped to
the live column count by `checkcolumns` (`Model.pas:405–437`). The data builder
`getmodeldata` (`Model.pas:560–631`) reads each as a 1-based column index via

```pascal
function columnscan(s: ansistring; c: integer): double;   { Model.pas:318–345 }
```

which just splits the row on space/comma/tab and parses the `c`-th token as a
double (returns 0 if the column is absent). **There is no scoping, no type
distinction, and no x-error concept** — *any* column can be X, Y or the error
column E; the fit range windows (§ comparison 2.2) are tested against the chosen
X column's values. Picking a fitted-parameter column as X is therefore exactly
param-vs-param trending, and x carries no uncertainty into the fit (only the E
column feeds σ_y). This confirms the prior summary; Asymmetry's improvement over
WiMDA is the opt-in effective-variance treatment of x-uncertainty (decision C).

## Study artifacts

- [comparison.md](comparison.md) — WiMDA columnscan transcription (verified);
  current Asymmetry surfaces with exact seams (file:line); the item-3 recursion
  model and `_FitRow` schema analysis; the effective-variance derivation and
  the ODR rejection; every divergence with both behaviours.
- [implementation-options.md](implementation-options.md) — chosen options,
  ordered phase plan, file-by-file touch list, test plan, recorded follow-ons.
  *(Written after the post-study decision checkpoint.)*
- [test-data.md](test-data.md) — numerical oracles (effective-variance,
  recursion round-trip, degenerate two-identical-groups equality) and corpus
  mapping (EuO λ vs ν).
- [verification-plan.md](verification-plan.md) — acceptance criteria per phase.

## Out of scope (rationale recorded)

- **x2 second analytic variable** — still covered by cross-group global/local
  fitting (model-function-parity comparison §2.4); not revisited.
- **python-user-functions** — Wave B owns the registry generalisation; land
  after it if surfaces collide.
- **Single-fit-range export to the results table** — deferred per decision B;
  the cross-group path is the trendable source. Recorded as a follow-on.
- **GLE/UI polish passes** — only the labels/curves the four items require.
