# HDF4 NeXus v1 (`.nxs`) read support — study

**Slug:** `hdf4-nexus-v1` · **Status:** shipped (Option B — read support) ·
**References:** WiMDA (native reader), Mantid (`LoadMuonNexus1`), the
WiMDA-muon-school `nxs4to5/` converter (proven pyhdf reader).

## Goal

Let Asymmetry open **legacy ISIS muon NeXus v1 files stored in an HDF4
container** directly — the format WiMDA reads natively (`/run` entry, `muonTD`
definition) — so users with archival ISIS data load them without a manual
pre-conversion step.

## Motivation / why revisit a closed decision

HDF4 `.nxs` is listed as a **standing exclusion** in the parity programme
(`docs/porting/wimda-parity-gap/decision-record.md` §1: *"HDF4 `.nxs` — Out
(standing decision reaffirmed) — Coverage boundary, not a bug"*). This study
**proposes reversing that decision**, on three grounds that were not weighed
when it was made:

1. **It is WiMDA's native on-disk format.** Asymmetry ports WiMDA; reading the
   exact files users already have is a direct parity/usability win, not a
   nice-to-have. The exclusion treated it as "coverage boundary," but every
   pre-~2015 ISIS muon run a WiMDA user owns is HDF4.
2. **The reader is now cheap and pure-Python.** `pyhdf` ships binary wheels for
   Windows/macOS/Linux (no system HDF4 C library), so HDF4 read support is a
   small optional dependency — the same shape as the existing
   `asymmetry[hdf5]`/h5py extra.
3. **A proven reader already exists, in-corpus.** The WiMDA-muon-school
   `nxs4to5/` tool (`hdf4tree.py` + `v1_to_v2.py`) reads **all 1,913 HDF4 files
   in the corpus** cleanly and its converted outputs parse identically to genuine
   v2 files. It is reference code we can study and re-implement.

This was surfaced by the API session-5 reproduction run (TRSB ships its 100 runs
as HDF4 `.nxs`; Asymmetry could only read the corpus's converted `data_hdf5/`
copies). Note: cross-format `.nxs_v2` / `.RAW` support is **explicitly NOT a
goal** — only the HDF4 container of the same v1 NeXus schema.

## Key finding — the gap is the *container*, not the *schema*

Asymmetry's `NexusLoader` **already understands the v1 `/run` / `muonTD`
schema**:
- `nexus.py:_detect_layout` returns `"v1"` when `handle` has a `run` group whose
  `analysis` is `muonTD`/`pulsedTD` or `IDF_version == 1`.
- `nexus.py:_load_v1` reads `histogram_data_1/counts`, `corrected_time`,
  `grouping`, `dead_time`, `time_zero`, `first/last_good_bin`, `sample/*`,
  `instrument/*`, NXlog series, etc., and reduces to `MuonDataset`.

The **only** blocker is the file *container*: `h5py.File()` raises
`OSError: file signature not found` on an HDF4 file (HDF4 magic
`\x0e\x03\x13\x01` ≠ HDF5 magic `\x89HDF`). So we do **not** need a new schema
parser — we need a way to feed the existing `_load_v1` an HDF4-backed handle.

## Recommended approach (detail in `implementation-options.md`)

**Option B — h5py-compatible adapter over an HDF4 tree.** Port the corpus
`hdf4tree.py` Vgroup reader (pyhdf → in-memory `Group`/`Dataset` tree), wrap it in
a thin adapter exposing the small read-only h5py surface `_load_v1` uses
(`__contains__`, `__getitem__`, `keys()`, `get()`, `.attrs`, array coercion), and
in `load()` branch on the HDF4 magic to build that handle instead of
`h5py.File`. Reuses **both** Asymmetry's validated v1 schema reader **and** the
corpus's validated HDF4 tree reader; far lighter than porting the full v1→v2
conversion.

`pyhdf` becomes an optional `asymmetry[hdf4]` extra with a graceful
"install pyhdf" error, mirroring the h5py pattern.

## Ready-made verification oracle

The corpus ships every HDF4 file **and** its converted HDF5 twin
(`<example>/data/*.nxs` ↔ `<example>/data_hdf5/*.nxs`). The parity test is
direct: load the HDF4 file → load the converted HDF5 → assert identical reduced
asymmetry, counts, t0/good-bins, grouping, and metadata. See
`test-data.md` / `verification-plan.md`.

## Files

- [comparison.md](comparison.md) — how each reference reads HDF4 v1; what
  Asymmetry already has.
- [implementation-options.md](implementation-options.md) — Options A/B/C, the
  adapter seam, dependency + detection handling.
- [test-data.md](test-data.md) — corpus HDF4↔HDF5 golden pairs and dialects.
- [verification-plan.md](verification-plan.md) — parity, dialect, and
  dependency-absent tests.

## Open questions — resolved

- **Reversal confirmed** by the maintainer; the decision-record HDF4 row is
  flipped from "Out" to shipped, citing this study.
- **pyhdf wheels are not uniform across platforms.** On Linux (manylinux,
  ~771 KB) and macOS (~535 KB) the wheels bundle the HDF4 C library, so
  `pip install asymmetry[hdf4]` works out of the box (CI is Linux → green with
  no extra system package). The **Windows** wheel (~188 KB) is the
  `_hdfext` extension only and links external `hdf.dll` / `mfhdf.dll`, which it
  does **not** bundle — confirmed via `dumpbin /dependents`. Asymmetry sources
  those DLLs from the conda-forge `hdf4` package (the same library Mantid
  bundles in its Windows installer); `packaging/windows/fetch_hdf4_dlls.py`
  fetches them and `ASYMMETRY_HDF4_DLL_DIR` / the frozen-app bundle dir makes
  them discoverable (`os.add_dll_directory`). The graceful-`ImportError` path
  keeps HDF4 optional on all platforms.
- **Bundling the HDF4 runtime into the released Windows/macOS binaries** is a
  scoped follow-on (PyInstaller spec + CI workflow): macOS via
  `collect_dynamic_libs("pyhdf")`; Windows by staging the conda-forge DLLs into
  the build. This PR ships the loader, parity tests, and docs.
- **A one-shot `convert`/export path** (write v2 HDF5) remains a possible
  follow-on, reusing the ported tree reader + the corpus `v1_to_v2.py` mapping.
  This study is **read-only**.
