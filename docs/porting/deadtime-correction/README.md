# Deadtime Correction Study

Status: implemented against the approved WiMDA-first direction

This study records how deadtime correction is implemented in WiMDA, musrfit,
Mantid, and the current Asymmetry codebase before any additional porting work.

The main result of the study was that Asymmetry already contained the shared
file-based non-paralyzable correction formula in
`src/asymmetry/core/transform/deadtime.py`, but it did not yet expose the full
correction-source model used by the reference programs.

## Decision Update

Chosen reference behavior: WiMDA.

Chosen implementation direction for the implementation pass:

- adapt WiMDA's grouping-window deadtime workflow to Asymmetry's
  `GroupingDialog`
- keep Asymmetry's existing core correction formula in
  `src/asymmetry/core/transform/deadtime.py`
- port WiMDA's `Cal` per-detector calibration action as an explicit deadtime
  table workflow in the grouping dialog
- make deadtime `estimate` follow the same rule that alpha estimate already
  follows in Asymmetry:
  - compute from the selected reference dataset only
  - apply the resulting deadtime payload to all selected datasets
  - let future loaded datasets inherit that payload through the existing
    grouping-template workflow

This direction has now been implemented in Asymmetry's core transform,
grouping dialog, and main-window grouping apply path.

## Scope

- WiMDA histogram deadtime correction and estimation
- musrfit file-based deadtime correction and `.msr` deadtime mode selection
- Mantid deadtime-table workflows and `ApplyDeadTimeCorr`
- Current Asymmetry core, GUI, and test seams

## Study Files

- `comparison.md`: implementation comparison across all four programs
- `implementation-options.md`: candidate ways to close the remaining gaps
- `test-data.md`: proposed comparison datasets and expected outputs
- `verification-plan.md`: how a later implementation pass should be validated

## Current Asymmetry Baseline

- Core owner: `src/asymmetry/core/transform/deadtime.py`
- GUI entry points:
  - `src/asymmetry/gui/windows/grouping_dialog.py`
  - `src/asymmetry/gui/mainwindow.py`
- Existing tests:
  - `tests/test_deadtime.py`
  - `tests/test_psi_loader.py`
  - `tests/test_grouping_dialog.py`
  - `tests/test_mainwindow_additional.py`

Asymmetry now supports these source workflows:

- file-provided per-detector deadtime values plus good-frame metadata
- WiMDA-style manual deadtime entry through a detector-value table
- WiMDA-style deadtime estimation
- WiMDA-style `Cal` per-detector calibration from the selected reference run
  into that manual table

Asymmetry still does not support:

- Mantid-style table-workspace or external-file deadtime selection

## Candidate Port Seams

1. Loader and import boundary: normalize deadtime sources into a common
   per-detector payload with provenance.
2. Grouping and project metadata boundary: persist which source is active and
   why.
3. Core transform boundary: keep the existing formula path, but accept a richer
   source contract instead of only `dead_time_us` lists from file metadata.
4. Grouping-template inheritance boundary: reuse the same main-window behavior
  that already lets grouping choices apply to newly loaded datasets.

## Open Questions

- Which reference behaviors are in scope for parity: only file-based correction,
  or also manual, estimated, and external-table sources?
- The user selected WiMDA-style file, manual, estimate, and `Cal` behaviors for
  the implementation pass, while intentionally dropping WiMDA's historical
  calibration-file load path from the live UI.
- Mantid-style table-selection UX is not the chosen primary target.
- WiMDA's higher-order model-panel variants and KEK-specific branch remain
  separate scope decisions for the implementation pass.