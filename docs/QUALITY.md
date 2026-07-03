# Quality Map

This document is the lightweight system of record for Asymmetry's current
quality posture. It is meant to guide agents toward the right validation path,
not to replace tests or architecture docs.

## Current Grades

| Area | Grade | Notes | Primary Checks |
| --- | --- | --- | --- |
| Core data model and transforms | B+ | Broad coverage exists for grouping, asymmetry, rebinning, deadtime, and background paths. Keep boundary cases explicit. | `python tools/harness.py test -- tests/test_transforms.py tests/test_deadtime.py tests/test_background_correction.py` |
| File loaders | B | NeXus, PSI, and ROOT loaders have focused tests. Format fixtures and provenance assumptions are the main risk. | `python tools/harness.py test -- tests/test_nexus_loader.py tests/test_psi_loader.py tests/test_root_loader.py` |
| Fitting and model logic | B+ | Model, parameter, wizard, and global-search tests cover much of the numerical behavior. Watch for slow or flaky optimization changes. | `python tools/harness.py test -- tests/test_fitting_engine.py tests/test_fit_wizard.py tests/test_global_search.py` |
| GUI workflows | B | Many panels and dialogs are covered with headless Qt tests, but true interactive QA remains partly manual. Plot responsiveness now uses display-only, viewport-aware decimation in `PlotPanel`; keep validating real pan/zoom and grouped-time workflows after render-path changes. | `python tools/harness.py gui-smoke` and targeted GUI tests |
| Project persistence | B+ | `.asymp` schema paths are well covered. Schema migrations should stay explicit and tested. | `python tools/harness.py test -- tests/test_project_schema.py` |
| Negative-muon analysis (`core/negmu/`) | B | Capture-lifetime model, polarisation, ratio, background, and fit paths each have a focused test file under `tests/negmu/`; a headless GUI smoke test guards the panel wiring. Documented in `docs/api/negmu.rst`. Younger than the primary μ⁺ path — watch for edge cases in less-common background/ratio configurations. | `python tools/harness.py test -- tests/negmu/` |
| MaxEnt reconstruction (`core/maxent/`) | B | Engine, calibration, pulse/pulse-oracle, run-hint, and window tests cover the core reconstruction under `tests/core/`; deadtime-promotion and reconstruction-display paths are covered under `tests/gui/`. Ported and studied per `docs/porting/maxent` and `docs/porting/maxent-completion`. Optimization-heavy code — watch for slow or nondeterministic changes. | `python tools/harness.py test -- tests/core/test_maxent.py tests/core/test_maxent_calibration.py tests/gui/test_maxent_reconstruction_gui.py` |
| Documentation | B | Sphinx docs and standalone architecture docs exist. Keep docs indexed and run a docs build for user-facing changes. | `python tools/harness.py docs` |
| Agent harness | B+ | Root map, harness docs, structural checks, full-repo Ruff, and CI are in place. Expand rules only when they protect real project invariants. | `python tools/harness.py structural` |
| Lint baseline | B | Full-repo Ruff format and lint checks are clean across source, tests, and harness code. `E501` is ignored because Ruff format owns ordinary wrapping and preserves some long scientific/UI strings. | `python tools/harness.py lint` |

## Risk Areas

- GUI behavior can pass unit tests while still feeling awkward in real use.
  Prefer small smoke workflows and screenshots when changing visual flows.
- Plot responsiveness improvements must preserve the contract documented in
  `docs/ARCHITECTURE.md`: full-resolution arrays stay authoritative for fit
  inputs, limits, and export, while only canvas artist density is reduced.
- Loader behavior depends on external file-format conventions. Preserve
  provenance in tests when fixing a format edge case.
- Optimization code can become slow or nondeterministic. Add focused tests with
  synthetic data and bounded tolerances.
- Generated artifacts such as docs builds and coverage reports should remain
  ignored and should not be committed.
- Ruff format is the preferred way to handle broad mechanical wrapping. Avoid
  hand-wrapping long scientific prose just to satisfy cosmetic line-length
  checks.

## Quality Rules

- Every new public behavior should have either a test, an example, or a docs
  update. Risky behavior needs tests.
- Cross-layer changes need boundary validation. For example, a new core fitting
  capability that appears in the GUI should have core tests and at least one GUI
  integration test or smoke path.
- Repeated review feedback should become a harness rule, a documented invariant,
  or a reusable helper.
