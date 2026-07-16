# Implementation options — MusrRoot `TDirectory` header parsing

## Parsing algorithm (implemented, not a design choice)

Because musrfit's own writer and reader define the encoding exactly (see
`comparison.md`), there is no real design space for the delimiter parse
itself — the port reimplements musrfit's index arithmetic directly:

```
label = s[s.find('-') + 1 : s.find(':')].strip()
value = s[s.find(':') + 1 : s.rfind(' -@')].strip()
type_code = s[s.rfind('@') + 1:]
```

applied per leaf `TObjString` while walking every `TDirectory` under
`RunHeader` (`RunInfo`, `DetectorInfo/DetectorNNN`,
`SampleEnvironmentInfo`, `MagneticFieldEnvironmentInfo`, `BeamlineInfo`).
Results are collected into a `label -> (value, type_code)` map **per
subfolder**, keyed on `label`; the leading `NNN` counter is discarded once
parsing succeeds (it is a global write-order index, not a stable per-field
key — see `comparison.md`).

The three decision points below are genuine choices, made for Asymmetry
rather than dictated by the musrfit source:

## Decision 1 — `RunSummary`: capture as provenance, not mine or discard

musrfit never reads the MusrRoot `RunSummary` block (its ring-anode-HV string
search targets the legacy LEM-ROOT run summary, a different structure); the
block is write-only from musrfit's own perspective. Two options existed:

- **Discard it** (mirror musrfit's *effective* behaviour for every field
  Asymmetry doesn't model) — simplest, but throws away free-text run
  provenance (operator notes, sample-environment remarks) that has no other
  home in the header.
- **Attach it verbatim as provenance** (chosen) — `RunSummary`'s lines are
  joined and stored on the loaded run as
  `metadata["musrroot_run_summary"]`, so nothing is lost even though
  Asymmetry does not attempt to parse structure out of it. This matches the
  existing pattern of preserving raw experiment provenance rather than
  discarding unmodelled fields (see `AGENTS.md` "Engineering Invariants").

## Decision 2 — instrument matching: case-insensitive

Existing `RunInfo`/`Instrument` matching (facility/beamline defaults) compared
instrument strings case-sensitively against known PSI instrument names.
FLAME's `TDirectory` files write the lowercase `flame`. Rather than adding
`flame` as a special case alongside the existing uppercase names, a header
instrument value that case-insensitively matches a known PSI instrument name
(`LEM`, `FLAME`, `GPS`, `DOLLY`, `GPD`, `HAL`, `LTF`) is normalised to the
canonical upper-case form in the loader's metadata, so every downstream
comparison and the displayed name agree; unknown instrument strings are kept
verbatim. (The GPS grouping trigger was already case-insensitive and needed
no change.)

## Decision 3 — backward-compatible fallback for "clean" entries

Synthetic test fixtures (see `test-data.md`) and, potentially, hand-edited or
tool-generated files may write a `TObjString` whose TKey name is just a clean
label (`"Run Number"`) with the value as the payload — a form no real DAQ
produces. The parser therefore discriminates on the **TKey name**: a name
that parses as the full `"NNN - Label: Value -@type"` encoding is decoded
(taking the value from the payload, which is canonical), while a
non-parsing name falls back verbatim to the pre-fix behaviour (raw path key
plus trailing-`-@type` stripping). The discriminator deliberately is not the
payload: clean-form payloads such as `"2026-01-01 10:00:00"` (leading digits,
hyphen, colon) false-parse as encoded entries, which would corrupt exactly
the fixtures the fallback exists to keep working. Real musrfit-written files
carry the encoding in both the key name and the payload, so the name is a
safe discriminator and real files always take the decoded path.

## Parsing: existing regex retained, verified against musrfit's semantics

musrfit's reader uses left/right index searches (label between the first `-`
and the first `:`; value up to the **last** ` -@`) so hyphens inside values
survive. Asymmetry's existing `_parse_musrroot_string` regex — a lazy body
with an *anchored optional* trailing `-@type` group — reproduces exactly that
semantics: the anchor forces the lazy group to expand to everything before
the final suffix, so a value containing `" - "` is preserved intact. This was
verified rather than assumed, and is pinned by a dedicated unit test with an
internal hyphen in the value; no new parser was written.
