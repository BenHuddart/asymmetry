# Verification plan

## Study-pass exit criteria (this document set)

- [x] Two cases identified and shown to be one abstraction (projections).
- [x] Current implementation seams mapped (`comparison.md`).
- [x] Data-model, UI, colour, fit-target, and storage options recorded with the
      chosen option marked (`implementation-options.md`).
- [x] Naming settled: `projection` / `AsymmetryProjection`.
- [x] "all" affordance resolved (Ben, 2026-06-13): keep as a text action.

## Implementation-pass checks (per the suggested order)

### 1. Data model + migration
- New unit tests: `PresetGrouping.projections` declared for EMU vector and the
  re-expressed MuSR/HiFi TF presets; the single `_detect`-style helper derives
  the same pairs the three old string-match helpers did (regression-pin the old
  outputs first, then prove parity).
- Schema round-trip: vector grouping with per-projection alpha saves/loads
  unchanged; pre-change `.asymp` projects migrate (legacy `alpha_px/py/pz` and
  `vector_axis` → `active_projection`).
- `python tools/harness.py test -- tests/test_instrument.py tests/test_project_*`

### 2. Chip bar + frame-tint
- GUI tests: chip bar appears iff ≥2 projections; floor-of-one (last chip won't
  release); selecting a subset renders exactly those stacked subplots in fixed
  semantic order; combo removed.
- Colour orthogonality test (extends the RG-mode fixture in `test-data.md`):
  trace colour = run, frame tint = projection.
- `python tools/harness.py gui-smoke`

### 3. Projection-keyed fit storage (schema v9)
- Distinct fits on `P_x`/`P_y`/`P_z` of one run persist and restore
  independently; pre-v9 migration lands the existing fit on the default
  projection. No regression for single-projection (longitudinal) datasets —
  `None`-keyed slot behaves exactly as the old single slot.

### 4. Selectable subplots + fit-panel echo
- GUI tests: clicking a subplot retargets the fit; fit panel header reflects the
  active projection; hiding the active projection's chip moves the target;
  single-projection view needs no selection; fit curve overlays only the active
  subplot (the existing `_fit_curves_by_key` display path, now fed from persisted
  slots).

### 5. TF end-to-end
- Re-express MuSR TF as one preset with two projections; verify the *same* chip
  bar / subplot / fit-target flow works with non-Cartesian projection labels —
  the proof that vector and TF are genuinely unified.

## Full ladder
- `python tools/harness.py validate` green (lint + structural + full suite).
- `python tools/harness.py gui-smoke` green.
- `docs/user_guide/vector_polarization.rst` updated to describe projections, the
  chip bar, frame-tint, and selectable-subplot fitting; `docs` target clean.

## Non-goals (explicitly unverified this pass)
- Per-projection batch/global fits.
- Joint (shared-parameter) fitting across projections — storage is *shaped* for
  it (`results_by_projection` mirror) but not implemented or tested here.
