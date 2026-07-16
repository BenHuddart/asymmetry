# Test data — MusrRoot `TDirectory` header parsing

## No real PSI data ships in this repository

This study, and the tests written against it, do not commit any real PSI
ROOT file, nor reference one by path, filename, run number, sample name,
proposal number, or proposer name. The gap that motivated this work was
found while examining **a 2026 FLAME commissioning-era data file examined
locally** — that file lives outside the repository and stays there; only the
header *encoding rules* it demonstrates (documented in `comparison.md`) are
recorded here, since those rules are format facts published by musrfit, not
data belonging to the run.

## Synthetic encoded fixtures (the primary test vehicle)

Because the encoding is fully specified by musrfit's writer (see
`comparison.md`), fixtures are built by **encoding synthetic values** with
the same `"NNN - Label: Value -@type"` scheme, using `uproot` (or a small
helper) to write a `TDirectory` tree shaped like a real `RunHeader`:

- A minimal `RunHeader/RunInfo` `TDirectory` with a handful of representative
  leaves: a `@0` string (title), an `@1` int (run number), a `@3` physical
  quantity with only a unit (temperature), a `@3` physical quantity with a
  unit and a free-text description (time resolution, mirroring FLAME's
  `Time Resolution: 0.09765625 ns; SiPM`), and a `@3` physical quantity with a
  demand/set-point clause.
- A `RunHeader/DetectorInfo/Detector001` `TDirectory` with a short detector
  name, histogram number, `t0` bin, and good-bin range, to exercise
  per-detector metadata.
- A `RunHeader/RunSummary` `TDirectory` (or plain `TObjArray`, matching
  musrfit's own `TDirectory`-path `RunSummary` handling) with free-text lines
  that are *not* in the `"NNN - Label: Value -@type"` form, to confirm the
  parser does not attempt to decode it as a header entry and instead captures
  it verbatim.
- Non-contiguous `NNN` numbering across subfolders (e.g. `RunInfo` uses
  `003`, `031`, `048`; `DetectorInfo/Detector001` uses `007`, `012`) to prove
  the parser keys on label, not on the counter.
- A value containing an internal hyphen (e.g. a description like
  `"pre-run check"` or a signed offset) to confirm the last-`" -@"` delimiter
  rule keeps it intact.
- A "clean-form" entry (`"Label: Value"`, no `NNN -` prefix or `-@type`
  suffix) to exercise the backward-compatible fallback from
  `implementation-options.md` §Decision 3.
- A lowercase `flame` instrument string, to exercise case-insensitive
  instrument matching.

These fixtures are small, fully synthetic, and safe to commit (they carry no
real experimental data) — they are the basis for the parity/unit tests
described in `verification-plan.md`. (The other agent building the fixtures
and tests for this fix owns their exact location under `tests/io/`; this
study records the *content* they need to cover, not their file paths, since
`tests/` is out of scope for this documentation pass.)

## Real-file verification: performed locally only

Beyond the synthetic fixtures, the fix was checked by hand against the 2026
FLAME commissioning-era file mentioned above — confirming that title, run
number, sample, temperature, field, time resolution, and per-detector
labels/`t0`/good-bin ranges now resolve to sensible values instead of the
blank/zero/guessed fallbacks. That check was run locally and is not
reproducible from the repository; it is recorded here only as a confirmation
that the synthetic-fixture coverage was cross-checked against one real
instance of the format, not as a source of committed test data.
