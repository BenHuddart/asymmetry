# Period selection — verification plan

## Unit tests (`tests/test_period_selection.py`)

1. **Correct period extracted** — synthetic two-period dataset; `select_period`
   for red/green returns the matching arrays; default combined dataset equals
   red.
2. **Label & scalar access** — `"red"`/`"green"`, `PeriodMode.RED/GREEN`, and
   1-based integers all resolve to the right period; case-insensitive.
3. **Errors on bad input** — out-of-range integer → `ValueError`; unknown label
   → `ValueError`; `bool`/other types → `TypeError`.
4. **Provenance preserved** — each per-period dataset keeps t0, good-bin window,
   grouping, temperature and field; carries its own `period_number`,
   `run_label`, `good_frames`, `dead_time_us`.
5. **GUI/core agreement** — `select_period_histograms` returns the same
   histograms the GUI path uses (same function), and `combine_period_asymmetry`
   reproduces the G−R / G+R arithmetic the GUI previously inlined.
6. **load(period=...)** — selecting at load time equals `select_period` on the
   full load result; 3+ period `list` path indexes by period number.

## Integration / corpus validation

`validate_photomusr.py` (run against the WiMDA corpus, not in CI):

1. Load `HIFI00103277.nxs`; extract light-OFF (Green) and light-ON (Red).
2. Assert light-ON relaxes more than light-OFF.
3. Fit light-OFF single exponential → A0; fix A0; fit light-ON first ~1 µs → λ.
   Assert both fits converge and λ(on) > λ(off) — the first point of the
   λ-vs-Δn calibration the guide describes.

## Validation ladder

```
python tools/harness.py structural
python tools/harness.py lint
python tools/harness.py test -- tests/test_period_selection.py
```

## Result

Implemented and validated 2026-06-07. Core API + GUI refactor agree on the same
period histograms by construction (shared functions). Photo-µSR workflow
reproduced end-to-end through the scriptable core API.
