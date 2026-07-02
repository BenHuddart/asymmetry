# Fit / trend / representation consolidation — test data

Data used to reproduce the audited behaviour and to anchor the regression tests.
Nothing here is a new file format; the consolidation reuses the existing loaders
and project schema.

## Live-audit corpus

The behaviour review (README) and the F1–F22 findings (comparison.md) were
produced by driving the real GUI over the **WiMDA Muon School** corpus at
`~/Documents/WiMDA muon school-indexed/` (entry points `INDEX.md` and
`GROUND_TRUTH_INDEX.md`). Each representation was exercised end-to-end: load →
single fit → batch fit → trend → trend model-fit → save/reload. The corpus is
not committed; it is the manual-verification substrate, not a unit-test input.

### Named runs referenced by the plan

- **EuO 2960 (ZF)** — the recorded FFT spectrum whose DC/apodisation spike sits
  ~3× taller than the real ~30 MHz peak. Drives:
  - Phase 5.2 peak-seeding DC-guard unit test (`tests/test_spectral_seed.py`):
    the guarded argmax must select ~30 MHz, not DC.
  - Phase 6 MaxEnt ZF-window unit test (`tests/test_maxent_window.py`): a
    synthetic ZF signal at 30 MHz must yield a reconstruction window containing
    30 MHz, while a TF run keeps its field-centred window.
- **EuO temperature scan** — the trend whose Tc moves toward ~69 K once the two
  garbage members are excluded (Phase 2.3 exclude-round-trip test).
- **F-B / grouped / frequency runs** — used for the chip-filtering and
  group-formation GUI tests (Phase 0.2, 0.3).

## Committed fixtures

- **Audit round-trip project** — a small recorded `.asymp` capturing the audit
  state (duplicate series chips + a legacy root `fit_ui_state`) is added under
  `tests/` for the migration and round-trip tests. Loading it must:
  - fold the legacy un-keyed fit state into `fit_states.{time,frequency}`
    (schema v10 → v11 migration, Phase 0.1);
  - collapse identical-signature duplicate series to a single chip
    (Phase 1.3 load-time dedupe);
  - never reproduce the frequency model in the time-domain form.
- **Synthetic pathological fits** — constructed in-test (no data file) for the
  member-quality flag unit tests (`tests/test_series_quality.py`): a failed
  member, a member with `|σ/value|` above threshold, a bound-pinned parameter,
  and a spurious-branch reseed.
- **Synthetic spectra / signals** — constructed in-test for the DC-guard and
  MaxEnt-window tests where the exact peak/DC ratio matters and a recorded file
  would be brittle.

## Schema fixtures

The project-schema tests (`tests/test_project_schema.py`) build minimal versioned
dicts inline (`{"schema_version": N, "datasets": [...]}`) and assert the
migration chain, including the new v10 → v11 fold. No external file is needed;
the smallest reproducing dict is preferred over a recorded project for migration
coverage.
