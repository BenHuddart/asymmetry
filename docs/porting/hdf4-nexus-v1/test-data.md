# Test data — HDF4 NeXus v1

The WiMDA-muon-school corpus is a ready-made oracle: it ships **every HDF4 file
alongside its converted HDF5 twin**, so direct-HDF4 reads can be graded against a
known-good HDF5 read of the same run.

Corpus root (this machine): `C:\Users\benhu\Source\wimda-corpus`. Convention per
example folder: HDF4 in `data/` (or `Data/`), converted HDF5 in `data_hdf5/`
(or `Data_hdf5/`). Directories already fully HDF5 (e.g. Ferromagnetic nickel) have
no HDF4 twin and are not relevant here.

## Golden pairs (HDF4 ↔ converted HDF5)

| Example | HDF4 (read under test) | Converted HDF5 (oracle) | Why chosen |
|---|---|---|---|
| TRSB (Re₆Zr) | `Superconductivity/TRSB/data/MUSR00038176.nxs` | `…/TRSB/data_hdf5/MUSR00038176.nxs` | The motivating case; MUSR, ZF. 100 runs available. |
| Basics concepts | `Basics/data/MUSR00044989.nxs` | `Basics/data_hdf5/MUSR00044989.nxs` | 2-group F/B reference; verified α 1.1033. |
| Basics (EMU) | `Basics/data/EMU00018850.nxs` | `Basics/data_hdf5/EMU00018850.nxs` | EMU dialect. |
| HiFi multi-period | a HiFi RF/ALC HDF4 run with `[128,nb]` counts (locate in benzene ALC or AFM sets) | its `*_hdf5` twin | Exercises period-major reshape (2×64). |
| MuT-era | an older MuT run lacking `IDF_version` (corpus README cites these exist) | its `*_hdf5` twin | Oldest dialect: `HH:MM:SS`, space timestamps, logs under `sample/`. |

Enumerate the full HDF4 inventory with `hdf4tree.is_hdf4` over every `data/`
file (the corpus README states **1,913 HDF4 files** convert cleanly) to build the
exhaustive parity sweep.

## Per-pair fields to compare (must match within tolerance)

Reduced `MuonDataset` from each side:
- `asymmetry` / `asymmetry_error` arrays (allclose, tight rtol ~1e-9 — same
  counts, same reduction kernel).
- raw grouped counts, `time` axis, `n_bins`.
- `t0` / `first_good_bin` / `last_good_bin`.
- `grouping` assignment, detector→group map.
- `dead_time` values.
- metadata: `run_number`, `temperature` (K, with the Celsius-unit rule),
  `field`, `field_state`, `instrument`, `title`, `start/stop`.
- NXlog series (`nexus_time_series`) presence + `sample_temperature_logged`.

## Independent integrity check

The corpus `nxs4to5/verify.py` already compares an HDF4 source against its v2
conversion bit-for-bit (counts) plus analysis-critical metadata. Use it as a
second, program-independent confirmation that the `data_hdf5/` oracle faithfully
represents the HDF4 source (so any Asymmetry mismatch is Asymmetry's, not the
corpus's).

## Bundled regression fixtures (for CI, no external corpus)

Pick **2–3 small HDF4 files** (one MUSR, one EMU, one multi-period/MuT) and
commit them under `tests/data/hdf4/` with their converted HDF5 twins, so the
parity test runs in CI without the full corpus. Keep them small (truncate runs if
needed via the corpus tooling). Gate the full 1,913-file sweep behind an env var
(e.g. `ASYMMETRY_WIMDA_CORPUS`) like the existing musrfit-data tests.
