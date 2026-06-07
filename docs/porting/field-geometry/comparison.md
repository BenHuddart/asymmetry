# Field geometry — cross-program comparison

All citations are from local source checkouts under `~/Source/`. Line numbers are
as found during the 2026-06-07 study pass.

## Summary table — what each program reads to determine TF / LF / ZF

| Program | Field(s) read for geometry | Reads `magnetic_field_state`? | Classifies ZF? | Stored as | Fallback when absent | Drives fitting? |
|---|---|---|---|---|---|---|
| **Asymmetry (current)** | `instrument/detector*/orientation` | **No** | No | `field_direction` ("Longitudinal"/"Transverse") | passthrough of raw string | No |
| **Mantid** | `instrument/detector*/orientation` (1st char `t`→Transverse) | **No** | No | run log `main_field_direction` + output prop `MainFieldDirection` | default `"Longitudinal"` (V1/V2 helper); PSI-bin: undetermined | No — only default grouping on MUSR/CHRONUS |
| **musrfit** | none — stores free-text `Setup`/`Orientation` + numeric field | reads but **discards** it | No | `PRawRunData.fSetup` / `fOrientation` / `fField` (informational) | `"n/a"` / `"??"` | No — user declares geometry in `.msr` THEORY block |
| **WiMDA** | none for analysis — field *value*; ISIS `a_selected_magnet` 'L'/'T'/'A' for logbook only | **No** | value-based report only | `mrun.info.field` (string); logbook column | user grouping (`.grp`) + analysis-mode enum | No — user picks grouping & mode |

**Headline: not one of the three reference programs uses
`sample/magnetic_field_state` to classify a run's field geometry.**

---

## Mantid (`~/Source/mantid`)

**Mechanism — detector `orientation` string, first character only.**

- Muon NeXus V2 path constant
  `instrument/detector_1/orientation`:
  `Framework/DataHandling/src/LoadMuonNexusV2NexusHelper.cpp:35`.
  Classification at `:117-130`:
  ```cpp
  std::string mainFieldDirection = "Longitudinal"; // default
  NXChar orientation = m_entry.openNXChar(NeXusEntry::ORIENTATON);
  orientation.load();
  if (std::tolower(orientation[0]) == 't') { mainFieldDirection = "Transverse"; }
  ```
  catch at `:126-128` ("no data - assume main field was longitudinal").
- Muon NeXus V1 (legacy/HDF4) path `run/instrument/detector/orientation`:
  `Framework/Muon/src/LoadMuonNexus1.cpp:771-786` — note the V1 check is
  **case-sensitive** `orientation[0] == 't'` (uppercase `'T'` would misclassify).
- Applied-field magnitude is read **separately** into `sample_magn_field` and is
  **not** used for geometry: `LoadMuonNexusV2NexusHelper.cpp:45,239`.
- Result stored as run log `main_field_direction`
  (`LoadMuonNexus1.cpp:791`; V2 via `SinglePeriodLoadMuonStrategy.cpp:38,43` and
  `MultiPeriodLoadMuonStrategy.cpp:38,44`) and output property `MainFieldDirection`
  (`LoadMuonNexusV2.cpp:254-255`).
- **PSI `.bin`** (`LoadPSIMuonBin.cpp`): does **not** classify — declares
  `MainFieldDirection` as int default 0 and never sets it (`:106-107`); stores
  raw header orientation as a sample log `Orientation` (`:354,618`).
- **Consumer**: only default grouping for MUSR/CHRONUS
  (`muon_group_pair_context.py:47-50`; `MUSR_Definition.xml:25,28`). No path ties
  `main_field_direction` to fit-function selection or asymmetry maths.
- **Default discrepancy** (flagged): `LoadMuonNexus.cpp:77` declares the property
  default `"Transverse"`, whereas the V1 loader and V2 helper default
  `"Longitudinal"`. Reachability not fully traced.

## musrfit (`~/Source/musrfit`)

**Mechanism — none. Free-text metadata + a numeric field; geometry is user-declared.**

- Data model `PRawRunData` has no geometry flag — only
  `fSetup{"n/a"}`, `fOrientation{"n/a"}`, `fField{PMUSR_UNDEFINED}`:
  `src/include/PMusr.h:942,951,953`.
- Per-format reads in `src/classes/PRunDataHandler.cpp`:
  - MusrRoot: `RunInfo/Setup` (`:1840-1842`), `RunInfo/Sample Magnetic Field`
    with G/T scaling (`:1856-1864`), `RunInfo/Sample Orientation` (`:1916-1918`).
  - PSI-bin: `SetSetup(GetComment())` (`:2711`), `SetOrientation(GetOrient())`
    (`:2715`), field parsed from string (`:2722-2729`); struct fields
    `char fOrient[11]; char fSetup[11];` in
    `src/external/MuSR_software/Class_MuSR_PSI/MuSR_td_PSI_bin.h:74-75`.
  - NeXus (`src/include/PRunDataHandler.h`): IDF1 setup←`/run/notes` (`:572-574`),
    field←`/run/sample/magnetic_field` (`:550-558`), orientation hardcoded `"??"`
    (`:583`); IDF2 field←`/raw_data_1/sample/magnetic_field` (`:752-760`),
    orientation hardcoded `"n/a"` (`:786`).
- `magnetic_field_state` **is** read at raw HDF level (IDF1 HDF5,
  `src/external/nexus/PNeXus.cpp:3668`) into an internal map used only for console
  `Dump` — **never mapped into `PRawRunData`**, so it is discarded.
- Downstream `fSetup`/`fField` are used only for plot titles/headers
  (`PMusrCanvas.cpp:915,942`). Fit *type* comes from the `.msr` `fittype`
  keyword (`PMusr.h:89-95`); TF-cosine vs ZF/LF behaviour is whichever THEORY
  function the user writes. Fitting does not need a geometry tag.

## WiMDA (`~/Source/WiMDA`, Free Pascal)

**Mechanism — none for analysis; user chooses grouping and mode.**

- `magnetic_field_state` is **not referenced anywhere** (empty grep across
  `src/*.pas`).
- Applied-field **value**: `nexusunit.pas:929-930` (`field := fget('magnetic_field')`),
  V1 `nexusunit_.pas:865`; stored as a formatted string into `mrun.info.field`.
- ISIS instrument selected-magnet flag `'L'/'T'/'A'` →
  chooses which selog field to print in the logbook only:
  `nexusunit.pas:2562-2571` (`L`→`Long_Field`, `T`→`Trans_Field`,
  `A`→`Field_ZF_magnitude`). HIFI `Field_Mode` similar (`:2633-2645`).
- PSI header `orientation`/`mode`/`expmode` strings are read into the record
  (`muondata.pas:127,158,198`) but only displayed (`LogbookUnit.pas:708`); `mode`
  is never read for logic.
- Geometry is the user's choice: detector grouping from `.grp`
  (`Group.pas:291 loadgroup`) and analysis-mode enum
  `Fitgrp = (fgFBAsym, fgSelected, fgFB, fgAll, fgForward, fgBackward)`
  (`FitTyps.pas:30`). CHRONUS TF is a user checkbox `ChronusTF` (`Group.pas:110`)
  that swaps to a hardcoded UDLR map (`nexusunit.pas:2114-2126`).
- Only automatic ZF-vs-field distinction is value-based and logbook-only:
  `nexusunit.pas:666-675` (`if Nexus_field <> 0` … else ZF magnitude). Exact
  `<> 0`, no tolerance. Does not change analysis geometry.

---

## NeXus / IDF spec note (what is canonical)

The corpus files declare `definition = 'pulsedTD'` (ISIS pulsed muon
time-differential). In the ISIS muon NeXus convention, `sample/magnetic_field_state`
is the field-state string (`'TF'`/`'LF'`/`'ZF'`) describing the **applied-field
geometry for the run**, and `instrument/detector*/orientation` describes the
**detector-bank orientation** (instrument build). These are semantically distinct
exactly as the bug report states.

**Caveat — could not cite the canonical spec text from a local source.** The
NeXus `definitions` repository (NXmuon / the ISIS muon application definition) is
**not** present under `~/Source/`. The semantics above are inferred from (a) the
field names and their grouping (`sample/…` vs `instrument/detector/…`) and (b) the
observed data (TF runs with L-oriented banks; TF at 0 G). A future pass should
confirm against the published NXmuon / ISIS muon format definition before relying
on edge-case wording (e.g. whether `'ZF'` is an official enumerated value — it
does not appear in this 2050-file corpus).
