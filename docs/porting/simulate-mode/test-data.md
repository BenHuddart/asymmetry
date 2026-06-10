# Simulate mode: test data

## In-repo synthetic fixtures (primary; no external dependency)

Unit and round-trip tests build their own template `Run` objects in memory —
the same pattern as `tests/test_maxent.py::_synthetic_run` — so the suite is
self-contained and deterministic:

- **Minimal F/B template**: 2 detectors, 2000 bins, 0.016 μs bin width,
  t0_bin 40, good bins [45, 1990], α = 1.15, good_frames set — exercises the
  α split, per-detector t0, good-bin propagation and the NeXus writer
  without any file dependency.
- **Multi-detector template**: 8 detectors in 2 groups of 4 with staggered
  per-detector t0 bins (PSI-style) — exercises within-group event division
  and t0 alignment.
- **Degrade source**: a simulated run from the minimal template (so the
  thinning tests know the true λ per bin).

Models for generation: `Exponential + Constant` (fast, analytic),
`Oscillatory × Exponential` (TF-like, tests phase/frequency recovery), and
`StaticGaussianKT` (ZF shape, no oscillation) — all from the existing
`COMPONENTS` registry.

## WiMDA Muon School corpus (opt-in, skip-if-missing)

Corpus root: `~/Documents/WiMDA muon school/` (see `docs/testing/`). Corpus
tests follow the existing `pytest.mark.skipif(not os.path.exists(...))`
convention (`tests/test_period_selection.py:346`).

| Corpus run | Role |
|---|---|
| Magnetism / Ferromagnetic nickel (HDF5 `.nxs`) | End-to-end template: load → fit TF precession → simulate from the fitted model → save NeXus → reload → refit. The headline round-trip verification target. |
| Semiconductors / CdS shallow donor `Data_hdf5/EMU000207xx.nxs` | Second HDF5 template with different detector count/binning (EMU); template-extraction generality. |
| Nuclear magnetism / EuO (PSI `.bin`) | Template from a *non-NeXus* source — exercises the standalone writer's independence from the template file format (impossible in WiMDA). |

HDF4 `.nxs` (the Basics folder) stays out of scope per the standing corpus
decision — not loadable, therefore not a valid template or round-trip target.

## WiMDA as oracle

No numerical golden files are taken from WiMDA: its simulation is
intentionally unreproducible (unseeded Delphi global RNG), so only
*distributional* agreement is checkable. The verification plan tests the
statistical properties (envelope normalisation, α split, Poisson variance,
1/√f degrade scaling) directly against their analytic expectations instead —
stronger than any cross-program file comparison.
