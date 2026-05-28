# musrfit `.msr` import

**Status:** candidate.

## What

Read musrfit `.msr` files and produce an equivalent Asymmetry project
state: datasets, theory model, parameter values + bounds, fit range,
group definitions. Optionally also export Asymmetry projects to `.msr`
for the reverse direction.

## Why

- The `.msr` file is the canonical workflow artifact for the entire
  musrfit-using community. Importing it makes Asymmetry an option
  for anyone with an existing analysis without forcing them to
  reproduce setup from scratch.
- Cross-tool reproducibility: a benchmark `.msr` from PSI can be
  rendered identically in Asymmetry and compared.
- Aligns with the comparison narrative in the public-facing
  comparison chapter — "if you're coming from musrfit, you can load
  your `.msr` directly".

## Prior art

- **musrfit:** authoritative `.msr` parser at
  `src/classes/PMsrHandler.cpp`. The format is documented in the
  user manual (`doc/html/user-manual.html`).
- **WiMDA, Mantid:** ❌. Neither imports `.msr`.

## Why this is roadmap-tractable

- The `.msr` format is plain text with a small grammar (RUN, THEORY,
  FUNCTIONS, COMMANDS, PLOT, FITPARAMETER, STATISTIC, GLOBAL blocks).
- Asymmetry already has all the receiving infrastructure:
  parameter sets, composite models, project state, data loaders.
- Mapping from musrfit theory function names → Asymmetry component
  names is finite and small (~34 entries); document any mismatches
  explicitly.

## Out of scope (for now)

- `.msr` *export* — leave for a follow-up once import is solid.
- Full round-trip preservation of comments and formatting (musrfit's
  `PMsrHandler` is meticulous about this; Asymmetry doesn't need to
  match initially).
- Loading user-function plugin libraries referenced by the `.msr` —
  surface a clear error if the file references a user function.
