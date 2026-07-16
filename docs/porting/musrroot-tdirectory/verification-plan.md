# Verification plan — MusrRoot `TDirectory` header parsing

## 1. Delimiter-parse unit coverage

Directly against the label/value/type-code extraction (see
`implementation-options.md`):

- Plain `@0`/`@1`/`@2` string/int/double entries parse to the expected label,
  value, and type.
- `@3` physical-quantity entries parse correctly across all four clause
  shapes musrfit's writer can emit: value+unit only; value+unit+description;
  value+unit+demand; value+unit+demand+description. The FLAME
  `Time Resolution: 0.09765625 ns; SiPM` form (unit + description, no
  demand) is one required case.
- A value containing an internal hyphen survives intact (the last-`" -@"`
  rule, not the first `'-'` after the colon).
- The non-encoded, "clean-form" fallback (`"Label: Value"`) is accepted.
- `@4`/`@5`/`@6` vector entries split on `;` correctly.

## 2. Header-walk / label-keying coverage

- Given a synthetic `RunHeader` tree with non-contiguous `NNN` numbering
  across subfolders (see `test-data.md`), every field resolves by label
  regardless of its numeric prefix or its position among siblings.
- `RunSummary` entries (free text, no `"NNN - Label: Value -@type"`
  structure) are not mis-parsed as encoded header fields, and their joined
  text lands verbatim in `metadata["musrroot_run_summary"]`.
- Instrument matching accepts the lowercase `flame` string equivalently to
  the existing uppercase PSI instrument names.

## 3. End-to-end load coverage (synthetic fixture)

Loading a full synthetic `TDirectory`-layout fixture through `RootLoader`
(or the public `load()` entry point) yields a `MuonDataset`/`Run` whose:

- `title`, `run_number`, `sample`, `temperature`, `field`, and
  `time_resolution` metadata match the fixture's encoded values (not the
  blank/`0.0`/filename-guessed fallbacks the loader previously fell back to).
- Detector metadata carries the fixture's short detector names (not the
  verbose histogram-title fallback) plus per-detector `t0` and good-bin
  ranges.
- `metadata["musrroot_run_summary"]` equals the fixture's `RunSummary` text.

## 4. Regression: legacy layouts unaffected

- A `TFolder`-layout MusrRoot fixture (the pre-existing flat-name format)
  still loads identically before and after this change.
- A pre-2011 LEM ROOT fixture (no `RunHeader` at all) still loads through the
  existing fallback path unaffected.

## 5. Real-file cross-check (not reproducible from the repo)

The fix was confirmed by hand against a 2026 FLAME commissioning-era file
examined locally (see `test-data.md`): title, run number, sample,
temperature, field, time resolution, and per-detector labels/`t0`/good-bin
ranges resolved to sensible values instead of the previous blank/zero/guessed
fallbacks. This check is not part of the committed test suite (no real file
is committed) and is recorded here only as corroborating evidence that the
synthetic-fixture coverage generalises to a genuine FLAME file.

## 6. Regression ladder

- `python tools/harness.py structural` — porting-study layout + `index.json`
  entry (this study).
- `python tools/harness.py lint`.
- `python tools/harness.py test -- tests/io/test_root_loader.py` (owned by
  the implementation change landing alongside this study), then
  `python tools/harness.py test --tier fast`, then `validate` once before
  handing back.

## Acceptance

- §1–§4 pass for both new-format and legacy-format fixtures; no regression on
  `TFolder` or pre-2011 LEM ROOT loading.
- §5's local corroboration is consistent with §1–§3 (documented here, not
  re-run as part of CI).
- `docs/reference/loading_data.rst` describes the new-format specifics
  (encoding, label-keying, `RunSummary` provenance, case-insensitive
  instrument matching) and `CHANGELOG.md` records the fix under
  `[Unreleased]`.
