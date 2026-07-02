# Fit / trend / representation consolidation — verification plan

How each phase is proven. The gate for every phase is
`python tools/harness.py validate` (standard tier; the harness re-executes
itself with the worktree `.venv/bin/python`); `gui-smoke` runs after phases that
touch startup or panel construction. The inner loop is
`python tools/harness.py test --tier fast` plus the focused files named per
phase.

## Per-phase verification

### Phase 0 — data-integrity bugs

- **0.1 keyed fit-form round-trip (F21c).**
  - Core: `tests/test_project_schema.py::TestSchemaMigrationV10toV11` — v10 → v11
    folds legacy root/`frequency_fit_state` keys into `fit_states.{time,frequency}`,
    drops the legacy copies, and defaults empty blocks when absent.
  - Panel: `restore_domain_state` routes a frequency blob without touching the
    live time form, and refuses a mismatched `domain` tag.
  - GUI: a full save → migrate → load → restore round-trip preserves each
    domain's form independently (extends the existing
    `test_project_round_trip_restores_frequency_fit_state`), plus a legacy-fixture
    load (root `fit_ui_state` only).
- **0.2 chip filtering (F21a/b).** Build one series in each of F-B, grouped,
  frequency plus one integral scan; save/reload; assert each Parameters view
  lists exactly its own chips.
- **0.3 group formation under sort (F9).** Sort by T (K), select runs,
  `create_data_group`; assert the header and member rows are visible immediately
  and no invisible-but-selected runs remain.
- **0.4 FFT restore render (F21d).** Save a project with a computed FFT, reload,
  switch to the FFT view, run the event loop until recompute completes; assert
  the plot has data and no orphan legend entries.

### Phase 1 — series identity, replacement, naming (D4, D8)

Core unit tests for signature narrowing, label carry-over on replacement, and
dedupe-on-load (`tests/test_series_identity.py`); GUI test that re-running a batch
with changed classification updates the single chip in place.

### Phase 2 — trend quality gating (D3)

Core unit tests for flag computation on synthetic pathological fits
(`tests/test_series_quality.py`) and extension of the grouped-series tests for
χ²ᵣ/errors; GUI tests for the failure summary banner, the χ²ᵣ / include-in-trend
table columns, and an exclude round-trip persisted through save/load (excluding
the two garbage EuO members moves Tc toward 69 K on the recorded fixture).

### Phase 3 — presentation provenance (D2)

GUI test: fit run A, select unseen run B, assert the carry-forward badge text and
that recording a fit for B clears it; a reload restores badges correctly.

### Phase 4 — default classification (D5)

Unit test on classification defaults per representation; GUI test that a default
grouped batch over a T-scan yields a varying-parameter trend (the audited
"No varying fit parameters" outcome becomes impossible without explicit opt-in).

### Phase 5 — frequency-domain fitting (D6)

`tests/test_spectral_seed.py` (DC-guarded peak seeding finds the ~30 MHz EuO peak,
not DC); GUI tests for the editable frequency range round-trip and for Run Batch
Fit enabling as soon as spectra exist (no selection poke).

### Phase 6 — MaxEnt ZF window (D7)

`tests/test_maxent_window.py`: a synthetic ZF signal at 30 MHz gets a window
containing 30 MHz; a TF run keeps its field-centred window; the divergence
message names the active window and Window control.

### Phase 7 — DataGroup ↔ FitSeries linking (D1)

Project round-trip of `data_groups` (including legacy projects without the block);
recorder stamping of `source_group_id`; "Fit this group" produces a series with
the correct members regardless of browser sort or visibility.

### Phase 8 — small UX safety items

Wheel-guard applied to dock numeric fields (F20); trend-model combo suggestions
(F4); default run-number sort after multi-file load (F8). Covered by focused GUI
tests where behaviour is observable.

## Acceptance criteria (whole feature)

1. Round-trip preserves — per representation — series chips, batch/single form
   contents, and renders restored FFT spectra (F21 closed).
2. Forming a data group while sorted never hides rows; batches cannot silently
   operate on invisible selections (F9 closed).
3. Re-running a batch never yields two identically-named chips; all series follow
   `<model> · <members>[ · <group>]` (F13/F22 closed).
4. Every batch member exposes χ²ᵣ + errors + flags in tables and trend plots;
   failed members are listed; trend points are click-excludable and exclusions
   persist (F2/F3/F5/F12/F14 closed).
5. A default batch on any representation produces a varying-parameter trend
   (F10 closed).
6. Frequency: fit range editable, default seed finds the 30 MHz EuO peak (not DC),
   Run Batch Fit enables as soon as spectra exist (F15/F16/F17 closed).
7. MaxEnt converges on ZF EuO 2960 with default settings (window contains 30 MHz)
   and its divergence message names the Window control (F19 closed).
8. Unseen-run carry-forward is visibly badged (F6 closed).
9. `python tools/harness.py validate` green; `gui-smoke` green.

## Migration / behaviour-change notes to call out in release notes

- All-Local batch defaults (Phase 4) and DC-guarded frequency seeding (Phase 5.2)
  change default fit outcomes — intended, but user-visible.
- `.asymp` schema changes are additive and read legacy keys; only load-time
  identical-signature dedupe (Phase 1.3) merges data, and it logs what it merged.
