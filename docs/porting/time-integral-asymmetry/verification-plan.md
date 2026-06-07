# Time-integral asymmetry — verification plan

## Objective

Show that (1) the per-run integral scalar matches the reference programs and a
hand-computed value, and (2) the assembled field scan orders and renders
correctly, before any GUI work is trusted.

## Layer 1 — per-run reduction (`integrate_asymmetry`)

| # | Check | Oracle / expectation |
| --- | --- | --- |
| 1 | `method="integral"`, flat synthetic asymmetry over window | Exact analytic `(value, error)` — loader-independent unit test |
| 2 | `method="integral"`, `alpha=1.0` on a real run | Equals `(F−B)/(F+B)` over the same good-bin window (WiMDA formula) |
| 3 | `method="integral"`, `alpha≠1.0` on a real run | Equals Mantid `PlotAsymmetryByLogValue` `Type=Integral` for that run |
| 4 | `method="differential"` | Equals Mantid `Type=Differential`; **differs** from Integral when asymmetry is non-flat (assert the divergence) |
| 5 | Error model | Matches `compute_asymmetry`'s Mantid-compatible error (shared kernel — assert it is the *same* code path, not a re-derivation) |
| 6 | Window defaulting | Unset `t_min/t_max` ⇒ full good-bin window; explicit window slices via `MuonDataset.time_range()` |
| 7 | Validation | `t_min ≥ t_max`, out-of-span window, `alpha ≤ 0`, bad `method` → `ValueError`/`TypeError` |
| 8 | Red/green | Each `period_mode` reduces the correct period (cross-check vs `select_period`) |

## Layer 2 — field scan assembly

| # | Check | Expectation |
| --- | --- | --- |
| 9 | Ordering | Points ordered by `order_key` using `FitSeries.sort_members()`; `field`/`temperature`/`run` all correct |
| 10 | Missing log | A run with no field value is excluded with a recorded reason; the scan still builds (Mantid parity) |
| 11 | Curve shape — repolarisation | Real LF scan: integral asymmetry **increases monotonically** with field (decoupling recovery) |
| 12 | Curve shape — ALC/QLCR | If an ALC scan is found: a **resonant dip** at the crossing field |
| 13 | `differentiate_scan` (`dA/dx`) | Forward difference + quadrature error; gated on max x-gap (WiMDA `dA/dB` parity, %/kG when x is field) |
| 14 | Full-table parity | Asymmetry scan `(x, y, σ)` table equals the Mantid golden file within counting precision |

## Layer 3 — persistence & GUI (lightweight in this port)

| # | Check | Expectation |
| --- | --- | --- |
| 15 | Recipe round-trip | `FieldScan` representation `to_dict`/`from_dict` preserves `{t_min, t_max, method, alpha, period_mode, order_key}`; arrays recomputed on load (recipe-only, like other representations) |
| 16 | Schema migration | Adding the representation type does not break loading existing `.asymp` projects (bump + migration if required) |
| 17 | GUI smoke | Scan curve renders in the trend surface with the x-axis selector (B/T/Run); `python tools/harness.py gui-smoke` passes |

## Tooling / commands

```bash
# fast inner loop on the new transform + series tests
python tools/harness.py test -- tests/test_time_integral_asymmetry.py
# structural (the study layout this doc belongs to)
python tools/harness.py structural
# full ladder before PR
python tools/harness.py validate
```

Mantid golden files are produced **out of tree** (GPL-3 oracle, not vendored —
same rule as the MaxEnt study) and committed as plain `(x, y, σ)` text/CSV under
the test fixtures, with the generating Mantid script recorded in
[test-data.md](test-data.md).

## Definition of done (Phase 1)

- `integrate_asymmetry` matches WiMDA (α=1) and Mantid (Integral & Differential)
  on a pinned corpus run.
- A pinned LF scan reproduces the expected monotonic repolarisation curve and the
  Mantid golden table.
- Tests live in `tests/test_time_integral_asymmetry.py`; `validate` is green.
- This study's `index.json` entry flipped to reflect implementation, with
  `src_path`/`tests_path` filled in and a `notes` summary of the final decisions
  (as the period-selection and field-geometry entries do).
- ALC baseline/peak-fitting explicitly **out of scope** and recorded as the
  `alc-avoided-level-crossing` follow-up.
