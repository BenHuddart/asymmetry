# Verification plan — HDF4 NeXus v1 read

Grade the direct-HDF4 read against the corpus's converted-HDF5 twin (the oracle),
across dialects, plus dependency/edge behaviour. Implementation is accepted only
when all pass.

## 1. Round-trip parity (the core test)

For each golden pair (`test-data.md`): load the HDF4 file directly and the
converted HDF5 twin via `NexusLoader`, then assert the reduced `MuonDataset`s
match:

- `np.allclose(ds_hdf4.asymmetry, ds_hdf5.asymmetry, rtol=1e-9, atol=1e-12)` and
  likewise for `asymmetry_error`, `time`, grouped counts.
- equal `t0` / `first_good_bin` / `last_good_bin`, `grouping`, `dead_time`.
- equal scalar metadata (`run_number`, `temperature`, `field`, `field_state`,
  `instrument`, `title`); equal `sample_temperature_logged` (within rtol).
- multi-period pair → same number of period datasets, each matching.

Bundled fixtures (`tests/data/hdf4/`) run in CI; the full 1,913-file corpus sweep
runs when `ASYMMETRY_WIMDA_CORPUS` is set (parametrized over every `is_hdf4`
file ↔ its `*_hdf5` twin), asserting parity and zero load failures.

## 2. Detection / routing

- `is_hdf4()` true on HDF4 magic, false on HDF5 and on non-NeXus files.
- `NexusLoader.load()` routes an HDF4 `.nxs` through the adapter and an HDF5
  `.nxs` through `h5py` — verified by loading one of each and checking both
  succeed (regression against the current `OSError: file signature not found`).
- `_detect_layout` over the adapter returns `("v1", "run")` for a `muonTD` file.

## 3. Adapter-surface unit tests

Pin each h5py behaviour `_load_v1` uses (see implementation-options "adapter
seam" table) directly on `_Hdf4Handle/Group/Dataset`: `__contains__`,
`__getitem__` (+ `KeyError`), `keys()`, `get()` → None on miss, `.attrs.get`,
`np.asarray`, nested `get("a/b/c")`, char-SDS string decode (int8 **and** uint8).

## 4. Dialect coverage

Parametrize §1 over: MUSR, EMU, HiFi multi-period, and a MuT-era file (no
`IDF_version`, `HH:MM:SS` duration, space-separated timestamp, logs under
`sample/`). Each must reduce and match its twin — proving the dialects the corpus
reader handles survive the adapter.

## 5. Dependency-absent behaviour

- With `pyhdf` importable: HDF4 load works.
- Simulating `pyhdf` absent (monkeypatch the import to `None`): loading an HDF4
  file raises a clear `ImportError` naming `pyhdf` / `asymmetry[hdf4]` — **and
  HDF5 `.nxs` loading still works** (the HDF4 dependency must not become a hard
  requirement). Mirror the existing `_require_h5py` test.

## 6. Resource hygiene

After loading an HDF4 file, the file is not locked (Windows): the same path can
be re-opened/renamed/deleted in the test — confirms `read_tree` closed its
`V`/`SD`/`HDF` handles.

## 7. Independent oracle cross-check (optional, corpus-gated)

For a sample of pairs, run the corpus `nxs4to5/verify.py` (HDF4 source vs its
v2 twin) to confirm the oracle itself is faithful, isolating any parity failure
to Asymmetry rather than the conversion.

## 8. Regression ladder

- `python tools/harness.py structural` — study layout + `index.json` entry.
- `python tools/harness.py lint`.
- `python tools/harness.py test -- tests/test_nexus_loader.py tests/test_hdf4_*`
  (new), then `validate`.

## Acceptance

- §1 parity passes on all bundled fixtures and (when run) the full corpus sweep,
  zero load failures.
- §2–§6 green; §5 proves HDF4 stays optional.
- `decision-record.md` HDF4 row updated from "Out" to the shipped status, citing
  this study; user-guide loading docs note HDF4 support + the `asymmetry[hdf4]`
  extra.
