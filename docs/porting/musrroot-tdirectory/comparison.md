# Comparison — MusrRoot `RunHeader` encoding, `TFolder` vs `TDirectory`

How musrfit writes and reads each `RunHeader` layout, and what Asymmetry's
`RootLoader` needs to reproduce for the new one.

## Legacy `TFolder` layout (already supported)

`RunHeader` is a `TFolder` whose immediate children are named objects
(`TObjString`/`TParameter`-style) directly holding a value — for example a
`RunInfo` folder containing a `TObjString` named `"Sample Temperature"` whose
string payload is the value. Asymmetry's existing flat-name reader walks
these folders by name and is untouched by this study.

## New `TDirectory` layout (this study)

`RunHeader` and its subfolders (`RunInfo`, `DetectorInfo/DetectorNNN`,
`SampleEnvironmentInfo`, `MagneticFieldEnvironmentInfo`, `BeamlineInfo`,
`RunSummary`) are themselves `TDirectory` objects. Every leaf is a single
`TObjString` whose **key name** (the name ROOT lists it under) and whose
**string payload** both carry the identical encoded string:

```
"NNN - Label: Value -@type"
```

musrfit writes this format in `TMusrRunHeader.cpp` (`$MUSRFIT_SRC`, added by
`4917e5c7`) — see e.g. the `Form("  %03d - %s: %s -@%d", ...)` calls building
string/int/double entries, and the `Form("  %03d - %s -@%d", ...)` call for
physical quantities. `NNN` is a zero-padded three-digit index drawn from a
**single counter shared across every subfolder** (`fPathNameOrder` is one
ordered list covering the whole `RunHeader` tree), so numbering is not
contiguous within any one subfolder — an entry numbered `007` may sit next to
one numbered `031` in the same `TDirectory`.

### Delimiter-parsing rules (from `TMusrRunHeader`'s own inverse parse)

musrfit's own reader (`4917e5c7`, functions around `TMusrRunHeader.cpp:1490`)
recovers the three fields with fixed index arithmetic rather than a general
tokenizer, and Asymmetry's port mirrors this exactly rather than inventing a
regex:

- **Label**: the substring between the string's **first** `'-'` and its
  **first** `':'`.
- **Value**: the substring between the string's **first** `':'` and its
  **last** `" -@"`.
- **Type**: the substring after the string's **last** `'@'`.

Because the value is delimited by the *last* `" -@"` rather than the first
`'-'` after the colon, hyphens that occur naturally inside a value (for
example in a free-text description, or a signed number) survive intact — only
the trailing `" -@<digit>"` type suffix is stripped.

### Type codes

| Code | Meaning | Payload shape |
|---|---|---|
| `@0` | string | raw text |
| `@1` | int | decimal integer |
| `@2` | double | decimal float (fixed precision) |
| `@3` | physical quantity | `value [+- err] unit[; SP: demand][; description]` |
| `@4` | string vector | `;`-separated list |
| `@5` | int vector | `;`-separated list |
| `@6` | double vector | `;`-separated list |

The `@3` physical-quantity format is itself compound: musrfit's writer
(`TMusrRunHeader.cpp`, the `MRH_TMUSR_RUN_PHYSICAL_QUANTITY` branch) emits up
to four optional clauses depending on which of value/error/demand/description
are set. FLAME's own commissioning data uses the plain
`value unit; description` form, e.g. `Time Resolution: 0.09765625 ns; SiPM`
— here `SiPM` is the free-text description, not a unit; the unit is `ns`.

### `RunSummary` is free text, not the encoded format

`RunSummary` holds plain lines (no `"NNN - Label: Value -@type"` structure at
all). musrfit does not read the MusrRoot `RunSummary` block at all — the
`RA-L`/`RA-R` ring-anode-HV string search in `PRunDataHandler.cpp` operates on
the *legacy LEM-ROOT* run summary (the `TObjArray` read alongside
`TLemRunHeader`), a different structure in a different file lineage. Rather
than discarding `RunSummary`, the port attaches it verbatim as run provenance
(see `implementation-options.md`).

## Histogram path (unaffected, already supported)

`4917e5c7` and `df7b8433` do not change where histograms live for
`TDirectory` files: `histos/DecayAnaModule/hDecay%03d`, vs
`histos/hDecay%03d` for `TFolder` files. `RootLoader` already branches on this
and is not part of this study's change.

## What Asymmetry already had before this fix

`RootLoader` already **detected** `TDirectory`-layout files (walking into
`RunHeader` as a `TDirectory` rather than a `TFolder`, and finding histograms
at the `DecayAnaModule` path) but did not implement the
`"NNN - Label: Value -@type"` decode — it read raw `TObjString` payloads
without stripping the numbering/type suffix, so metadata lookups keyed on
exact label strings never matched and fell through to the generic fallbacks
described in `README.md`.
