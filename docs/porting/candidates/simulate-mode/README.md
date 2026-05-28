# Simulate mode

**Status:** candidate.

## What

A first-class "simulate" tool that generates synthetic μSR datasets
(time-domain asymmetry + per-bin uncertainty + optional histograms
and grouping) from a chosen theory model and parameter values. The
output is a real `MuonDataset` that can be loaded into the data
browser, plotted, fitted, FFT'd — exactly as if it were a real run.

## Why

- **Teaching:** the muon spectroscopy textbook ships dozens of
  archetype figures (Fig 5.6 LF decoupling, Fig 6.6 EuO T-scan,
  Fig 6.8 Cu(pyz)₂(ClO₄)₂ three-frequency AFM). Students who want
  to reproduce these in the GUI currently have to write Python.
- **Documentation:** Asymmetry's screenshot pipeline already
  simulates data via `docs/screenshots/data/archetypes.py`. Exposing
  that capability in the GUI removes the developer-only barrier.
- **Fit validation:** before committing to a fit on real data, users
  routinely want to confirm "what would the fit return if the truth
  were X?". Simulate enables round-trip closure tests.
- **Cross-tool benchmarking:** the comparison matrix in
  `docs/porting/comparison-matrix.md` is more defensible if users can
  generate identical synthetic data in Asymmetry vs musrfit vs
  Mantid and verify the fit outputs match.

## Prior art

- **WiMDA:** `Simulate.pas` — form-driven simulation that writes
  synthetic count histograms to disk. The only reference program
  with an integrated simulate tool.
- **musrfit:** users write `.msr` files by hand and run the engine in
  generate mode. No GUI affordance.
- **Mantid:** can simulate via `CreateSampleWorkspace` and the
  general Fit framework, but no muon-specific simulate UI.

## Why this is roadmap-tractable

- Asymmetry has all the building blocks already:
  - MODELS / COMPONENTS evaluators in `core/fitting/`.
  - `MuonDataset` and (since the YBCO Knight scenario) full
    `Run` synthesis with histograms in
    `docs/screenshots/data/archetypes.py`.
  - The data browser accepts any `MuonDataset`.
- The implementation is a new GUI dialog that ties the existing
  pieces together: model picker (reuse `FitFunctionBuilderDialog`),
  parameter spin-boxes, noise controls (counts-per-bin), time
  axis (t_min, t_max, n_points), then a "Create" button that calls
  the synthesis helper and adds the resulting dataset to the
  browser.

## Out of scope for this candidate

- Multi-detector grouping in the simulate dialog. Initially just
  produce single-channel asymmetry; extend to per-group histograms
  in a follow-up.
- Built-in archetype gallery ("simulate EuO T-scan"). Mention as a
  Later-tier nice-to-have once the basic dialog is shipped.
