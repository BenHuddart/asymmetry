# Comparison — reading ISIS muon NeXus v1 (HDF4)

How the reference programs and the corpus converter read the legacy HDF4 `.nxs`
container, and what Asymmetry already provides.

## The format

- **Container:** HDF4 (magic `\x0e\x03\x13\x01`). Written by the classic NeXus
  C API over HDF4 Vgroups/SDS. `h5py` cannot open it.
- **Schema:** NeXus v1, single top-level group `/run`, `analysis = "muonTD"`
  (older MuT-era files: no `IDF_version`). Histograms at
  `/run/histogram_data_1/counts [n_spectra, n_bins]` with `t0_bin`,
  `first_good_bin`, `last_good_bin`, `resolution`; `grouping`, `alpha`,
  `time_zero`, `corrected_time`; `instrument/detector/deadtimes`;
  `sample/{name,temperature,magnetic_field,...}`; NXlog groups for T/field.
- This is the **same schema** Asymmetry's `_load_v1` already reads — the modern
  v2 layout (`/raw_data_1`, HDF5) is the *other* family it also reads.

| Reference | HDF4 read path | Notes |
|---|---|---|
| **WiMDA** | Native — its loader reads `/run` muonTD HDF4 directly | The behavioural contract; this *is* the format WiMDA users have. |
| **Mantid** | `LoadMuonNexus1` (`MuonNexusReader`) via NeXus/HDF4 | v1 semantics oracle; GPL — study only, do not vendor. |
| **musrfit** | No ISIS-muon-v1 HDF4 loader of interest | Not a useful reference here. |
| **corpus `nxs4to5/`** | `hdf4tree.py` (pyhdf Vgroup walk → `Group`/`Dataset` tree) + `v1_to_v2.py` (schema translation to v2 HDF5) | **Proven**: reads all 1,913 corpus HDF4 files; outputs parse identically to genuine ISISICP v2. Pure-Python (`pyhdf`+numpy). MIT-compatible to re-implement. |

## What Asymmetry already has (the reuse surface)

`src/asymmetry/core/io/nexus.py`:
- `NexusLoader.load()` → `h5py.File(path)` → `_detect_layout` → `_load_v1` /
  `_load_v2`.
- `_detect_layout` already returns `"v1"` for `run` + `analysis ∈ {muonTD,
  pulsedTD}` / `IDF_version == 1`.
- `_load_v1` consumes the handle through a **small, uniform accessor surface**:
  `"<name>" in node` (`__contains__`), `node["<name>"]` (`__getitem__`),
  `node.keys()`, `node.get(...)`, `node.attrs`, and `np.asarray(dataset)`, all
  funnelled through helpers `_read_optional`, `_require_group`,
  `_require_dataset`, `_safe_str/_safe_int/_safe_float`, `_extract_tree`,
  `_extract_time_series`. This is exactly the surface an adapter must satisfy.

## Implication

Because the v1 *schema* reader exists and is exercised by the corpus's converted
files today, the port reduces to **(a)** an HDF4 container reader (port
`hdf4tree.py`) and **(b)** a thin h5py-shaped adapter so `_load_v1` runs over the
HDF4 tree unchanged. No second schema parser, no fit/transform changes.

## Licensing

- Mantid `LoadMuonNexus1` is **GPL** — behavioural oracle only, never copied.
- The corpus `nxs4to5/` reader is local project tooling we author; re-implement
  it cleanly in `asymmetry.core.io` (no GPL lineage). `pyhdf` is MIT-licensed.
