# MusrRoot TDirectory-based RunHeader — study

**Slug:** `musrroot-tdirectory` · **Status:** in-progress (fix landing
alongside this study) · **References:** musrfit (`TMusrRunHeader`,
`PRunDataHandler::ReadRootFile`).

## Goal

Read PSI MusrRoot ``.root`` files whose ``RunHeader`` is written with the new
**``TDirectory``-based** layout, so run title, run number, sample,
temperature, field, time resolution, and per-detector ``t0``/good-bin ranges
load correctly instead of falling back to filename or histogram-title
guesses.

## Motivation

PSI's 2026 FLAME DAQ ("MuSRrootHeader" generator) writes ``RunHeader`` as
nested ``TDirectory``s rather than the legacy ``TFolder`` streaming.
`RootLoader` already claimed to support "both layouts" (see
`docs/reference/loading_data.rst`, added ahead of a working parser), but the
actual per-leaf encoding — labelled, globally-numbered ``TObjString``
entries, not a flat name→value map — was not implemented, so new-format files
loaded with blank title, ``0.0`` temperature/field, a run number guessed from
the filename, and verbose histogram-title fallback detector labels. This was
found examining **a 2026 FLAME commissioning-era data file examined locally**
(no path, filename, run number, sample name, proposal number, proposer name,
or measured value from that file is recorded anywhere in this repository —
only the header *encoding rules*, which are format facts, not run data).

## Reference: musrfit's TDirectory commit series

musrfit (Andreas Suter, PSI) added the ``TDirectory`` layout and the reader
for it in six commits during 2025–2026, in `$MUSRFIT_SRC`:

| Commit | Date | Summary |
|---|---|---|
| `e38fa479` | 2025-04-04 | Skeleton for MusrRoot handling `TDirectoryFile` rather than `TFolder`. |
| `7e28402e` | 2025-09-28 | Start implementing `TDirectory` infrastructure. |
| `4917e5c7` | 2025-09-29 | Reader for MusrRoot written with `TDirectory` rather than `TFolder` (`TMusrRunHeader.cpp`/`.h`). |
| `ea646e01` | 2025-09-29 | MusrRoot version bumped to 2.0 (CMake soname bump only, no format change). |
| `df7b8433` | 2025-09-30 | `PRunDataHandler::ReadRootFile` handles `TDirectory` in addition to the deprecated `TFolder`. |
| `dbbaf554` | 2026-04-15 | MusrRoot XML-schema validator files moved to a DOI (`https://doi.org/10.5281/zenodo.19593555`) instead of a URL — documentation-only. |

`4917e5c7` and `df7b8433` are the load-bearing commits: the first defines the
per-leaf string encoding and its inverse parse in `TMusrRunHeader.cpp`
(`GetFirst`/`RemoveFirst`/the `Last('@')`/`Last('-')`/`First(':')` index
arithmetic — see `comparison.md`), the second wires
`PRunDataHandler::ReadRootFile` to try `TDirectory` first and fall back to
`TFolder`. `ea646e01` confirms the MusrRoot version bump that marks
`TDirectory` as canonical; `dbbaf554` is unrelated to the on-disk layout.

## Key finding — the format is label-keyed, not position-keyed

Every leaf under `RunHeader` (and its subfolders `RunInfo`,
`DetectorInfo/DetectorNNN`, `SampleEnvironmentInfo`,
`MagneticFieldEnvironmentInfo`, `BeamlineInfo`, `RunSummary`) is a
`TObjString` whose **key name and payload both** carry the same encoded
string:

```
"NNN - Label: Value -@type"
```

`NNN` is a single global counter shared across *all* subfolders (musrfit
writes it from one `fPathNameOrder`-ordered loop over every header field, see
`comparison.md`), so numbering within one subfolder is not contiguous. A
parser that keys on `NNN` or on positional order is wrong; the port keys
strictly on `Label`.

## Implemented approach

Walk every `TDirectory` leaf in `RootLoader` and key the result by label
rather than by number or position, decoding each leaf with the loader's
existing `_parse_musrroot_string` — verified (and pinned by a unit test) to
reproduce musrfit's inverse-parse semantics (`TMusrRunHeader::GetFirst` /
`RemoveFirst`, the `Last('@')` / `Last('-')` / `First(':')` index arithmetic
in `TMusrRunHeader.cpp`), including internal-hyphen preservation. See
`implementation-options.md` for the delimiter rules and the three
Asymmetry-specific divergences (`RunSummary` provenance capture, instrument
name normalisation, and the clean-form fixture fallback keyed on the TKey
name).

## Files

- [comparison.md](comparison.md) — the encoding rules and type codes, as
  implemented by musrfit and read back by Asymmetry.
- [implementation-options.md](implementation-options.md) — the delimiter
  parsing algorithm, backward-compatibility fallback, and the divergences
  from musrfit's own use of the data.
- [test-data.md](test-data.md) — synthetic fixture strategy (no real PSI data
  in the repository) and the locally-verified real-file check.
- [verification-plan.md](verification-plan.md) — what is tested and how
  correctness is graded.

## Scope boundary

Legacy `TFolder`-based MusrRoot files and pre-2011 LEM ROOT files are
unaffected by this study — they already load through the existing flat-name
`RunInfo`/`DetectorInfo` reader, which this work leaves untouched.
