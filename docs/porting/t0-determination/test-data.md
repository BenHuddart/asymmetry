# t0 determination — test data

No research data, run numbers, sample names or private paths belong in the
repo; corpus tests stay `skipif`-gated behind environment variables.

## Synthetic (in-repo, deterministic)

| Record | Purpose |
|---|---|
| `tests/io/test_t0_search.py` fixtures: continuous prompt peak, pulsed rising edge, ties, early spike | `find_t0` strategies and tie-breaks |
| `tests/core/test_count_domain_fits.py::_pulsed_tf_run` | promotion sign round-trip (corrupt `t0_bin` by ±k, fit free `t0`, promote, expect the truth) |
| Staggered-t0 PSI-style run (`tests/core/test_grouping_profiles.py` `[5,5,6,6]` pattern) extended with an out-of-group detector whose t0 exceeds the group max | F1 inference, R1 explicit policy, R5 offset semantics |
| Run with a missing header t0 (all zeros) | F5 `t0_source = "missing"` and the error verdict |
| NeXus v2 in-memory file with a per-detector `time_zero` array (`tests/io/test_nexus_loader.py` builders) | F6 payload keys, 1-based heuristic on a bin-centre axis (tie case) |

## Corpora (gated)

| Corpus | Env var / location | Instruments | Use |
|---|---|---|---|
| musrfit example data | `ASYMMETRY_MUSRFIT_DATA` → `$MUSRFIT_SRC/doc/examples/data` | PSI GPS/Dolly/LEM ROOT and BIN, NeXus | reader parity of `t0_bin`, per-detector tables; continuous-source `|Δ|` distribution |
| WiMDA muon-school corpus | `ASYMMETRY_WIMDA_CORPUS` (replace the hard-coded `~/Documents/WiMDA muon school/…` paths in `tests/io/test_t0_search.py:118-123`) | ISIS EMU, HiFi, MuSR NeXus v1/v2 | pulsed `|Δ|` distribution; 0/1-based check of the pulse position vs `t0_bin` |
| Mantid unit-test files (public names): `emu00102347.nxs_v2`, `MUSR00015189.nxs`, `deltat_tdc_dolly_1529.bin` | Mantid data repository | EMU v2 (TimeZero 0.16 µs, FirstGoodData 0.384), MUSR v1 (0.55 / 0.656), Dolly bin (per-histogram t0 0.1582–0.1602 µs) | loader oracles for the µs↔bin conversions |
| User's private PSI GPS MusrRoot series (15 detectors, 0.1 ns bins) | local only, never referenced from tests | GPS | manual re-check of F1 after the fix (delta must be 0 in From-file mode; profile healed) |

## Oracles transcribed from the references

- musrfit `musrt0_getMaxBin` (`$MUSRFIT_SRC/src/musrt0.cpp:145-160`): argmax,
  lowest tie — matches `find_t0(pulsed=False)`.
- musrfit packing time (`PRunSingleHisto.cpp:1245-1275`):
  `dt·((fgb − 0.5) + p/2 − t0) + k·p·dt` — must equal Asymmetry's merged-bin
  stamp for the same `fgb`, `p`, `t0`.
- WiMDA `SearchT0` (`$WIMDA_SRC/src/Group.pas:2225-2256`): argmax on group
  sums, *highest* tie — differs by construction; record, do not match.
- Mantid `LoadPSIMuonBin` common t0 = max over histograms
  (`LoadPSIMuonBin.cpp:186-195`) — same rule as `common_t0_for_groups` when
  every histogram is grouped.
