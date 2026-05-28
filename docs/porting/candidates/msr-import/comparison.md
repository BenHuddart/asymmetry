# `.msr` import: comparison

Only musrfit has a working `.msr` parser. The implementation
comparison is therefore "what musrfit does" vs "what Asymmetry
needs to do".

## `.msr` block summary

| Block | Purpose | Asymmetry equivalent |
|---|---|---|
| RUN | Per-run paths, run type (asymmetry / single histogram / mu-minus), forward / backward / α / β | `MuonDataset` + grouping |
| THEORY | Theory function expression and parameter indices | `CompositeModel.from_expression` |
| FUNCTIONS | Inline expressions (Boost.Spirit grammar) | Asymmetry composite expressions are stronger; map directly |
| COMMANDS | Minuit2 commands (MIGRAD / MINOS / SAVE) | Fit panel actions |
| FITPARAMETER | Parameter table: number, name, value, step, min, max | `ParameterSet` |
| PLOT | View / runs / range / Fourier settings | Plot panel + Fourier panel state |
| STATISTIC | Post-fit results (chi²/ndof, per-param +err / -err) | `FitResult` + (after MINOS candidate) `errors_minos` |
| GLOBAL | Cross-run config (lifetime, units) | Asymmetry uses module constants |

## Theory-function-name mapping table

A concrete deliverable of the study pass: produce a JSON file
`docs/porting/candidates/msr-import/theory_name_map.json` mapping
musrfit `PTheory` slugs to Asymmetry COMPONENTS / MODELS slugs.
Entries fall into three categories:

1. **One-to-one mapping** (most cases): `dynKTLF` → (lands with
   dynamic-kubo-toyabe candidate)
2. **Composite expression**: musrfit `exp + cos` → Asymmetry
   `Exponential * Oscillatory + Constant`
3. **Unsupported**: musrfit `Bessel` → not in Asymmetry until the
   theory-library-expansion candidate lands

The mapping file is also useful for the reverse-direction `.msr`
export.

## Edge cases the study should document

- Multi-run `.msr` files (RUN blocks for multiple runs) → use
  Asymmetry's global-fit machinery.
- `.msr` with user-function references → fail with a clear error
  pointing at the python-user-functions candidate as the future
  solution.
- `.msr` with units in unusual conventions → document Asymmetry's
  canonical units (μs, MHz, Gauss) and warn on conversion.
- Parameters with shared values across runs → Asymmetry's
  Global / Local parameter classification handles this.
