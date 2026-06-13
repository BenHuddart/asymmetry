# Verification plan — workflow-visualisation

The checks that gate each ADOPT/ADAPT item. Climb the validation ladder: unit tests
beside the behaviour, then `structural`/`lint`, then a full `validate`, then
`gui-smoke` for the panel wiring. Corpus comparisons are env-gated; pure-fixture
variants run in CI.

## Per-item gates

### 2. Data-only / ASCII export (ADAPT)
- **Round-trip:** export current plot as text (data only) → parse → arrays equal the
  plotted `t/asymmetry/error`; provenance header carries run number, grouping,
  forward/backward pair, α, deadtime.
- **Content switch:** data-only writes `.dat` only; data+fit writes `.dat`+`.fit`;
  fit-only writes the resampled model and no `.dat`.
- **No-clutter:** assert the single export control hosts both actions (no second
  top-level export button added).
- **Shared path:** assert the GLE export and the text export call one payload-write
  helper (no duplicated writer).
- **Optional x-range:** with the limit-to-range flag on, rows outside `[x_min,
  x_max]` are dropped; default off preserves every-point behaviour.

### 3. Events columns (ADOPT)
- **Good events:** `good_events_mev` equals `Σ counts[first_good_bin..last_good_bin]`
  over grouped detectors ÷1e6 (ISIS **and** PSI run); distinct from the all-bins
  `counts_mev`.
- **Events/frame:** equals good events ÷ `good_frames`; shows "—" when `good_frames`
  ≤0/absent (synthetic run).
- **Add-column UI:** the header "Add column…" action adds/removes the new columns and
  the chosen set persists across a panel-state round-trip.
- **Shared helper:** browser good-range sum and the export `events_grouped` come from
  one core helper (assert equality on a fixture).

### 4. B-from-log (ADOPT)
- **Preference:** on a run with a field log, enabling field-from-log changes the B
  column to the log mean (≠ header scalar) and tints the cell.
- **Override:** per-run override beats the global toggle (parallel to temperature).
- **Combined runs:** event-weighted mean across constituents (reuses the temperature
  weighting path).
- **No-log fallback:** a run with no field log returns the header scalar (no silent
  coil substitution), cell untinted.

### 6. Log-count diagnostic (ADOPT)
- **Straight line:** a synthetic pure decay `N₀e^(−t/τ_μ)` renders linear on the log
  axis; fitted log-slope ≈ −1/τ_μ within tolerance.
- **Non-positive bins:** zero/negative bins masked, not plotted as −∞; masked count
  surfaced.
- **Gating:** the log-scale checkbox is visible only on the raw-counts view and
  hidden on fb_asymmetry/groups/frequency; the flag persists (additive key).
- **Consistency:** linear and log renders of the same run show the same data (one is
  the other's log).

### 7. F,B balance overlay (ADAPT, borderline — only if confirmed)
- **Coincidence:** with the corpus run's known-good α, forward and α·backward
  envelopes coincide (overlay gap within tolerance).
- **Sensitivity:** perturbing α by ±20% opens a visible, monotonic gap.
- **Display-only:** the overlay creates no new α value and exposes no promote path
  (assert no α mutation on the grouping).

### 8. Cursor readouts (ADAPT, subset per checkpoint)
- **Snap:** hovering near a point emits the nearest cached `(t, A, err)`; the status
  bar shows `t, A±err`.
- **Parabolic peak (exact):** unit test against `parabpkextrap` formula on synthetic
  parabolas with known vertices; reject when `a≥0` or vertex outside the neighbour
  span.
- **Windowed average:** the drag-select readout equals `integrate_curve(t, A, err,
  t1, t2)` `(mean, mean_error)` over the same window.
- **S/N:** equals `|y/err|` at the snapped point; guarded at err=0.

## Cross-cutting gates
- **Full `validate`** green (lint + structural + the whole pytest suite, parallel).
- **`gui-smoke`** green — panels construct and wire (new toggles, column menu,
  export menu, cursor signal) without error.
- **No schema bump:** project open/save round-trips an existing `.asymp`; all new
  persisted state is additive and absent-tolerant (old projects load unchanged).
- **Corpus regressions untouched:** existing reduction/fit goldens unchanged (this
  session is GUI-surface + two browser columns + one core peak helper; no analysis
  numerics change).
- **Append-only shared files:** `data_browser.py`/`plot_panel.py`/`mainwindow.py`
  edits stay additive where Wave-B/③ sessions also touch them (keys, menu entries).

## Out-of-scope (no gates)
- Items 1, 5, 9 (REJECT) — nothing shipped; their deferred niceties, if ever taken
  up, verify then.
- Item 10 (design-only) — the hook contract is documented, not built; it will gain a
  fixture-driven `reload_from_source` test when a beamline makes it testable.
