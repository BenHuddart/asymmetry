# Project brief: simulate-mode

Umbrella: `wimda-parity-gap` · Wave A · Size M (one long session) · promotes
the existing `simulate-mode` candidate (tier "now", score 16)

## Motivation

Synthetic data generation closes three needs at once: teaching, fit
validation (does the pipeline recover known parameters?), and test-data
manufacture for every other project in this portfolio (double-pulse,
pulse-shape, binning-mode verification all want synthetic runs). WiMDA is
the only reference program that has it.

## WiMDA reference

`Simulate.pas`: Generate synthesises Poisson-noised count histograms from
the current fit model + parameters using the loaded run as instrument
template (grouping, α, per-group t0, lifetime envelope, double-pulse
halving); Save Simulation copies the loaded `.nxs` and overwrites
counts/t0/title so the synthetic run is loadable (lines 38–165).
`DegradeStats.pas`: multiply every bin by a factor and Poisson-resample —
"what would this look like with half the beam time" (lines 33–48).

## Scope

- `core/simulate.py` (Qt-free): model + parameters + instrument template
  (grouping, t0s, α, frames, bin width) + total events → per-detector count
  histograms via the existing grouping/asymmetry kernels run forward;
  Poisson noise; returns a first-class `Run`/`MuonDataset` with provenance
  metadata marking it synthetic.
- NeXus round-trip: write the synthetic run as a loadable HDF5 `.nxs`
  (template-copy approach like WiMDA, or minimal-writer — decide in study).
- **Degrade statistics** folded in: Poisson thinning of a *loaded* run by a
  factor (the same sampling core; trivial once simulate exists).
- GUI: a Simulate dialog (new window) — pick model (reuse the fit-function
  builder), parameters, events, noise seed; result appears in the Data
  Browser like any run, clearly badged.
- Promote `docs/screenshots/data/archetypes.py` synthesis helpers into the
  new core module where they overlap (roadmap already suggested this).

**Out**: simulating sample-environment logs; event-mode simulation;
instrument templates beyond "copy a loaded run".

## Current Asymmetry state

Nothing user-facing; synthesis exists only inside tests and screenshot
tooling. Candidate study folder: `docs/porting/candidates/simulate-mode/`.

## GUI/UX sketch

File menu → "Generate synthetic run…". Dialog with model picker, parameter
table (seeded from the current fit if one exists — WiMDA's nicest property),
events spinner, optional fixed RNG seed for reproducibility, "Save as
NeXus…" secondary action. Degrade-statistics as a Data Browser context-menu
action ("Degrade statistics…", factor + seed).

## Physics-correctness notes

Sample counts as Poisson draws of the *expected counts* (N0 envelope ×
(1 + A·P(t)) + bg), not Gaussian noise added to asymmetry — errors then
propagate correctly through the real reduction chain by construction.
Fixed seeds for test reproducibility.

## Conflicts & dependencies

New module + new dialog: effectively conflict-free (one `mainwindow.py` menu
hook, one `data_browser.py` context-menu entry — keep both additive).
Downstream consumers: `count-domain-fit-modes`, `frequency-domain-finishers`,
`maxent-completion`, `rrf` test plans all benefit; none block on it.

## Verification sketch

Round-trip: simulate from a model fitted to a real corpus run → save NeXus →
reload through `NexusLoader` → refit recovers parameters within errors;
pull-distribution check over many seeds (pulls ~ N(0,1)) — a stronger test
than WiMDA ever had. Degrade: thinned run's per-bin errors scale as 1/√f.
