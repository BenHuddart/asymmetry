# Implementation options — HDF4 NeXus v1 read

Three ways to let `NexusLoader` open an HDF4 v1 `.nxs`. All share two
prerequisites:

- **Detection.** Sniff the 4-byte HDF4 magic `\x0e\x03\x13\x01` *before*
  `h5py.File()` (h5py raises `OSError` on HDF4). Reuse `hdf4tree.is_hdf4(path)`.
- **Dependency.** `pyhdf` as an optional extra `asymmetry[hdf4]` with a graceful
  `ImportError` ("install pyhdf or asymmetry[hdf4]"), mirroring `_require_h5py`.

## Option A — in-memory HDF4 → v2, feed `_load_v2`

Port the corpus `v1_to_v2.convert` to build a v2 structure (in memory or a temp
HDF5), then run the existing `_load_v2`.

- **+** Reuses the most-validated corpus path end-to-end (verify.py proves it).
- **−** Heaviest: re-implements period-major reshaping, bin-centre→edge
  conversion, grouping/deadtime tiling, log → `selog` remap — all to then re-read
  it. Two transforms where one would do. More surface to drift from `_load_v1`.

## Option B — h5py-compatible adapter over the HDF4 tree (RECOMMENDED)

Port `hdf4tree.py` (pyhdf Vgroup walk → `Group`/`Dataset` tree), then wrap it in a
thin adapter implementing the read-only h5py surface `_load_v1` uses, and route
`load()` through it.

- **+** Reuses **both** validated readers: Asymmetry's `_load_v1` schema logic
  *and* the corpus HDF4 tree reader. Smallest new code; one transform.
- **+** `_load_v1` stays format-agnostic — future container quirks are isolated in
  the adapter.
- **−** Must faithfully reproduce the slice of h5py semantics `_load_v1` relies on
  (enumerated below) — bounded and testable.

### The adapter seam (what `_load_v1` requires)

A `_Hdf4Group` wrapping `hdf4tree.Group` and a `_Hdf4Dataset` wrapping
`hdf4tree.Dataset` must provide:

| h5py behaviour used | Adapter implementation |
|---|---|
| `"name" in node` | `name in group.children` |
| `node["name"]` | return wrapped child group/dataset; `KeyError` if absent |
| `node.keys()` | `group.children.keys()` |
| `node.get(name)` (used by `_read_optional`) | wrapped child or `None` |
| `node.attrs` (mapping, `.get`) | `Dataset.attrs` / group attrs dict |
| `np.asarray(dataset)` | `dataset.data` (already an ndarray) |
| string decode | `hdf4tree._to_str` handles int8/uint8/bytes char SDS |
| nested path `get("a/b/c")` | already implemented by `Group.get` — adapter must mirror via repeated `__getitem__` |

`_load_v1` does **not** need write, slicing of large datasets, references, or
attribute iteration beyond `.get`, so the surface is small. Confirm by grepping
every `handle`/`entry`/`node` access in `_load_v1` and its helpers
(`_extract_tree`, `_extract_time_series`, `_read_optional`, `_require_*`,
`_safe_*`, `_read_temperature_kelvin`, `_logged_sample_temperature`) and pinning
each to an adapter method in tests.

### Wiring

```
def load(self, filepath):
    if hdf4tree.is_hdf4(filepath):          # new branch
        self._require_pyhdf()
        root = hdf4tree.read_tree(filepath)  # ported into asymmetry.core.io
        handle = _Hdf4Handle(root)           # adapter, no h5py
        version, entry = self._detect_layout(handle)   # returns ("v1","run")
        result = self._load_v1(handle, entry, str(path))
    else:
        self._require_h5py()
        with h5py.File(path, "r") as handle:
            version, entry = self._detect_layout(handle)
            result = self._load_v1(...) if version=="v1" else self._load_v2(...)
```

`_detect_layout` already works unchanged over the adapter (it only uses
`"run" in handle`, `handle[...]`, `_read_optional`, `keys()`).

## Option C — standalone direct HDF4 → MuonDataset parser

A new loader that reads the HDF4 `/run` tree straight into `MuonDataset`.

- **−** Duplicates `_load_v1`'s field mapping, good-bin/period/grouping handling
  — exactly the logic that already exists and is tested. Rejected.

## Recommendation

**Option B.** Port `hdf4tree.read_tree` + `is_hdf4` into a new
`src/asymmetry/core/io/hdf4.py` (pyhdf-guarded), add the `_Hdf4Handle/Group/
Dataset` adapter there, and add the HDF4 branch to `NexusLoader.load`. Keep
`.nxs`/`.nexus` on the same `NexusLoader` (extension is shared; the magic byte
disambiguates container). Ship `pyhdf` as `asymmetry[hdf4]`.

Export to v2 HDF5 (write path) is **out of scope** for this study — but Option B
leaves it a clean follow-on (the ported tree reader + corpus `v1_to_v2` mapping
→ `nexus_writer`).

## Risks / edge cases to carry into implementation

- **MuT-era dialects:** no `IDF_version`, `HH:MM:SS` durations, space-separated
  timestamps, logs nested under `sample/` or `instrument/beam/`. The corpus
  reader already handles these (README "Handled v1 dialects"); the adapter must
  not lose them — covered by the all-corpus parity test.
- **Multi-period HDF4** (e.g. HiFi RF/ALC `[128, nb]` = 2×64): `_load_v1` already
  splits periods via `_split_period_counts`; confirm the HDF4 `counts` array
  arrives in the same `[n_spectra, n_bins]` orientation the converter assumes.
- **Char SDS sign:** HDF4 char data may be int8 or uint8; reuse `_to_str`'s
  uint8 coercion to avoid negative-byte decode errors.
- **pyhdf resource handles:** `read_tree` must close `V`/`SD`/`HDF` handles
  (the corpus version does, in `finally`) — no lingering file locks on Windows.
