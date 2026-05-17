# Deadtime Correction Comparison

## Scope

This comparison covers the parts of deadtime correction that decide behavior:

- how the correction is enabled
- where deadtime values come from
- when the correction is applied in the reduction pipeline
- what assumptions the code makes about good frames, bin width, and detector scope
- what tests or validation surfaces currently exist

## Summary

All four codebases share the same core file-based correction formula for the
non-paralyzable case:

`N_corr = N_obs / (1 - N_obs * tau / (bin_width * good_frames))`

The main differences are not the formula itself. They are the source model and
workflow around it:

- WiMDA exposes file, manual, loaded, estimated, and per-detector calibrated
  deadtime workflows from the grouping UI.
- musrfit applies file-based deadtime through the core reduction path and reads
  mode selection from `.msr` files, but its main correction path still reports
  `estimate` as not implemented.
- Mantid externalizes deadtime into a table workspace and allows selecting the
  source from file, workspace, other file, or none.
- Asymmetry now implements the shared file-based formula together with
  WiMDA-style file, manual, estimated, and per-detector calibrated deadtime
  payloads in the grouping workflow, with calibration writing into the manual
  detector-value table.

## Program Comparison

| Program | Entry points | Data source(s) | Correction ordering | Dependencies | Edge cases | Test coverage |
| --- | --- | --- | --- | --- | --- | --- |
| WiMDA | `Group.pas` (`CCoff`, `CCman`, `CCload`, `CCest`), `WiMDA_Main.pas` auto-load, `nexusunit.pas` internal NeXus deadtimes | Manual UI entry, `dt*.dat` calibration files, estimated values, some internal NeXus deadtimes | Applied per histogram inside `ReGroup` before grouped counts and before asymmetry plotting | Grouping UI state, calibration files, instrument-specific frame normalization, detector bunching | KEK special branch, early exit for large deadtimes, warning when calibration run is later than data run, multiple correction models | No automated tests found in the repo scan |
| musrfit | `PRunBase::DeadTimeCorrection`, called by `PRunAsymmetry` and related run classes; `.msr` parsing in `PMsrHandler` | File-provided deadtime vectors plus good frames; `.msr` selects `file`, `estimate`, or `no` | Applied to raw detector histograms before higher-level asymmetry/group calculations | `PRawRunData`, good frames, time resolution, `.msr` configuration, NeXus loader path | Warns and exits when input is missing; `estimate` logs as not implemented in the main correction path | No direct deadtime unit tests found in the repo scan |
| Mantid | `MuonProcess`, `MuonPreProcess`, `ApplyMuonDetectorGrouping`, `ApplyDeadTimeCorr`, Muon GUI corrections tab | Deadtime table from data file, table workspace, other file, or none | Applied to workspaces before grouping and asymmetry calculation | `ITableWorkspace`, `goodfrm` run log, equal bin sizes, spectrum-number mapping, ADS/GUI table selection | Fails when `goodfrm` is missing or zero, when bin width is zero, or when the table row count exceeds histogram count; subset-spectrum tables are supported | `ApplyDeadTimeCorrTest.h`, `MuonProcessTest.h`, GUI docs |
| Asymmetry | `asymmetry.core.transform.deadtime`, `GroupingDialog`, `MainWindow._prepare_grouping_histograms` | File-provided `dead_time_us`, manual per-detector entry, WiMDA-style per-detector calibration, reference-run estimate | Applied to histograms before grouping and before asymmetry calculation | Histogram model, grouping metadata, GUI toggle, loader-provided metadata, explicit deadtime provenance | File mode stays the default selected source; apply warns if the current reference run lacks file values | `tests/test_deadtime.py`, `tests/test_psi_loader.py`, `tests/test_grouping_dialog.py`, `tests/test_mainwindow_additional.py` |

## Detailed Notes

### WiMDA

- `Group.pas` applies correction detector-by-detector in `ReGroup` via
  `ccorrect` and `ccorrectRG` before grouped counts are formed.
- The grouping UI exposes the workflow directly inside the deadtime group box:
  - `File`
  - `Off`
  - `Man`
  - `Auto Estimate`
  - `Auto Load`
  - per-histogram manual entry via `HEdit`
  - a shared `DT0edit` / status label pair for estimated or model values
  - `Load`, `Save`, and `Cal` buttons beside the mode controls
- `WiMDA_Main.pas` auto-searches the data directory, and then `cal/`, for
  `dt*.dat` files when load mode is active.
- `Group.pas` also contains an estimation path (`CCest`) that fits an initial
  signal model and pushes the resulting value into `hdtime` for all histograms.
- `CalibrateButtonClick` fits the same model per detector, stores one
  calibrated `hdtime[j]` value for each detector, then switches into the
  explicit deadtime-value path.
- WiMDA therefore couples source selection, estimation, and correction tightly
  into the grouping workflow.

### musrfit

- `PRawRunData::DeadTimeCorrectionReady` requires both good frames and a
  deadtime parameter vector.
- `PRunBase::DeadTimeCorrection` reads the correction mode from `.msr` global
  and per-run settings, then applies file-based correction when data is ready.
- The main run-processing path prints an info message for `estimate`, rather
  than computing estimated deadtimes there.
- musrfit also contains separate `PNeXusDeadTime` estimation utilities in its
  external NeXus support, which suggests that estimation is treated as a
  separate calibration workflow rather than a core reduction default.

### Mantid

- `MuonProcess` and related workflow algorithms treat deadtime correction as a
  pre-grouping workspace transform.
- `ApplyDeadTimeCorr` consumes a table workspace keyed by spectrum number, so
  deadtime provenance is externalized instead of embedded directly in the run
  grouping payload.
- Mantid's Muon GUI exposes four source choices: from data file, from table
  workspace, from other file, or none.
- Mantid's loader utilities note that PSI data does not always provide a
  `DeadTimeTable`, which matches Asymmetry's current PSI limitation.

### Asymmetry

- The core transform in `src/asymmetry/core/transform/deadtime.py` already
  matches the musrfit and Mantid formula, including good-frame normalization.
- `GroupingDialog` now keeps `File` selected by default, exposes manual,
  estimate, and `Cal` actions, and resolves explicit deadtime payloads before
  the main window applies grouping.
- `MainWindow` applies deadtime before grouping, preserves deadtime provenance,
  and lets selected and future datasets inherit manual, calibrated, or
  estimated payloads.
- Existing docs now describe the ported WiMDA deadtime controls rather than the
  earlier file-only subset.

## Main Differences To Carry Forward

1. The formula is already shared; the missing parity is mostly about source
   selection and provenance.
2. WiMDA is the broadest source model in scope here: file, manual, loaded,
  estimated, and calibrated deadtime are all first-class, but Asymmetry only
  needs the file/manual/estimate surface plus the calibrated manual table.
3. WiMDA exposes those modes directly in the grouping dialog, not as a separate
  corrections workspace or detached import wizard.
4. Asymmetry already has an established rule for estimate-like values:
  compute from the reference dataset only, then apply the resulting grouping
  payload to the selected targets.
5. Asymmetry already auto-applies the latest grouping payload to future loaded
  datasets by inheriting from the highest-run grouped dataset in the browser.
6. That existing inheritance rule is the correct target behavior for estimated
  deadtime too.
7. Mantid is the broadest boundary model: deadtime is a reusable table
   workspace that multiple workflows can consume.
8. musrfit is narrower than WiMDA in the main reduction path: file-based mode
   is implemented, estimate is acknowledged but not completed there.
9. Asymmetry is currently closest to the musrfit and Mantid file-based subset,
   but without Mantid's external table contract.

## Selected Port Target

Chosen for the implementation pass:

- port WiMDA's grouping-window deadtime mode surface into Asymmetry
- keep Asymmetry's current core correction formula and histogram preparation
  path
- port WiMDA's `Cal` per-detector calibration action as an explicit deadtime
  table workflow inside the manual detector-value editor
- make `estimate` obey Asymmetry's existing alpha-style semantics:
  - selected reference dataset only for the calculation step
  - apply the resolved payload to all selected datasets
  - auto-apply the same payload to future loaded datasets through existing
   grouping inheritance behavior

## Candidate Port Seams

### Seam 1: Source normalization

Normalize all deadtime sources into one Asymmetry payload with:

- source kind
- per-detector tau values
- good-frame requirement
- provenance metadata

### Seam 2: Loader and import adapters

Adapters can populate the normalized payload from:

- file metadata
- a Mantid-style table representation
- future estimated values

### Seam 3: GUI and project persistence

Persist which deadtime source is selected and whether it was available,
imported, estimated, or inherited from file metadata.