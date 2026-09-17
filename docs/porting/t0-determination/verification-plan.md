# t0 determination — verification plan

Ladder per `AGENTS.md`: focused test files while iterating, `--tier fast`
after core changes, affected GUI files focused, `validate` once per PR.

## P0 — correctness (core)

| Test | Pins |
|---|---|
| `tests/core/test_grouping_profiles.py::test_fresh_payload_with_out_of_group_detector_is_from_file` | F1: payload `t0_bin` = F/B max, one non-group detector higher ⇒ `from_file` |
| `…::test_manual_inferred_only_from_effective_override_for_legacy_payloads` | R1 legacy rule |
| `tests/project/test_grouping_profile_migration.py::test_project_open_heals_zero_delta_manual_policy` | R1 heal, with a log assertion |
| `tests/core/test_t0_resolver.py` (new): `effective_detector_t0_bins` with/without override, malformed override ignored | R2 |
| `tests/core/test_fourier_grouped.py`, `tests/core/test_maxent_engine.py`, `tests/core/test_count_domain_fits.py`: a Manual policy with delta ≠ 0 changes the grouped signal/axis identically to reduction | F2, F3 |
| `tests/core/test_promote_calibrations.py::test_promote_t0_round_trip_recovers_truth` | F4: corrupt by +3 and −3 bins, fit, promote, `t0_bin` within 1 bin of truth; `first_good_bin` shifted with it |
| `tests/tools/test_structural.py` | R2 harness rule fails on a bare `apply_grouping_aligned(` outside the chokepoints |

## P1 — loaders

| Test | Pins |
|---|---|
| `tests/io/test_nexus_loader.py`: per-detector `time_zero` array ⇒ `detector_t0_bins`, `t0_bin` = group max; scalar ⇒ broadcast | F6 |
| `tests/io/test_nexus_loader.py`: `t0_bin` attr N ⇒ `Histogram.t0_bin == N − 1` regardless of the axis (exact-edge file included); `last_good_bin` attr == n_bins ⇒ `good_bin_end == n_bins − 1`; `time_zero` inconsistent with `t0_bin` ⇒ `t0_source == "conflict"` and the attribute wins | F17, T12 |
| Corpus (gated `ASYMMETRY_WIMDA_CORPUS`): `tests/porting/t0-determination/isis_t0_header_survey.py` reproduces `floor+1` on 100 % of unambiguous files and `last_good_bin == n_bins` everywhere | T12 (done 2026-09-17 on 1,245 files) |
| `tests/io/test_{psi,root,nexus}_loader.py`: all-zero/missing t0 ⇒ `t0_source == "missing"`, `t0_bin == 0` | F5 |
| `tests/io/test_nexus_loader.py`: loader dataset axis equals `(k − t0_bin)·w` | F8 |
| `tests/io/test_psi_loader.py`: good window derived over F/B detectors only | F13 |
| Reader parity (gated `ASYMMETRY_MUSRFIT_DATA`): `t0_bin` per detector equals musrfit's `GetT0Bin` for each example file | parity |

## P2 — GUI

| Test | Pins |
|---|---|
| `tests/gui/test_grouping_dialog.py`: the file/detected/Δ line is present and identical in all three modes; detection runs once per preview run (extend `test_grouping_dialog_perf.py`) | R3 |
| verdict tests: missing t0 ⇒ error; `|Δ| > tol` ⇒ warn; single-detector outlier ⇒ per-detector warn; large spread ⇒ strategy warn | R4 |
| Manual offset: editing the spin on run A stores `offset = spin − file_common(A)`; run B resolves `file_common(B) + offset`; migration of an absolute `value` | R5 |
| `tests/gui/test_plot_panel.py`: good-window mask on staggered-t0 data uses the common t0 | F11 |
| `tests/io/test_nexus_writer.py`: file-values and effective-values writes are each self-consistent | F7 |

## Corpus sweep (sets the R4 thresholds before they ship)

Script under `tools/` (not a test): for every file in the two gated corpora,
load, run `find_t0_for_run`, and record per instrument: file common t0,
consensus, `Δ = consensus − file`, per-detector `|est_i − t0_i|`, spread,
strategy, bin width. Output a table into this study
(`threshold-calibration.md`) and choose `tol_source` as the smallest integer
above the 95th percentile of `|Δ|` on files with a trustworthy header.
Record outliers with their cause (dead detector, wrong facility token,
1-based header).

Expected from the existing corpus tests: EMU pulsed within ±2 bins of the
header (`tests/io/test_t0_search.py:118-135`), GPS continuous within ±1 bin
per detector (`:137-149`).

## Docs verification

`python tools/harness.py docs` after R9; regenerate the `t0_search`
screenshot scenario once the new line exists (docs README § scenarios).
