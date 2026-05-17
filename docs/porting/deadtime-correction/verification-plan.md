# Deadtime Correction Verification Plan

## Goal

Verify that any future deadtime correction changes in Asymmetry are equivalent
to the approved subset of WiMDA, musrfit, and Mantid behavior.

## Baseline Assertions

These assertions should remain true unless the study docs are updated with a
new decision:

1. The file-based non-paralyzable formula matches musrfit and Mantid.
2. Good-frame normalization is required for file-based correction.
3. Missing file deadtime values do not trigger an estimator fallback unless the
   user explicitly selected deadtime `estimate` mode.
4. Deadtime is applied before grouping and asymmetry calculation.
5. Deadtime `estimate` uses the selected reference dataset only.
6. The resolved deadtime payload is applied to all selected target datasets.
7. Future loaded datasets inherit the latest grouping deadtime payload using
   the same project/grouping rule already used elsewhere in Asymmetry.
8. WiMDA-style `Cal` fits one deadtime value per detector from the selected
   reference dataset and persists that explicit table.

## Planned Checks

### 1. Core formula parity

Add focused unit tests around the shared formula using synthetic cases from the
comparison manifest.

Target area:

- `src/asymmetry/core/transform/deadtime.py`

### 2. Source-contract characterization

If new source modes are added, add tests that prove the normalized source
payload maps to the same corrected counts regardless of where tau values came
from.

Target areas:

- loader adapters
- grouping/project metadata
- deadtime preparation before grouping

### 3. GUI gating behavior

Verify when the deadtime toggle is enabled, disabled, or forced off.

Target areas:

- `GroupingDialog`
- main window grouping flow

### 4. Estimate propagation behavior

Verify that deadtime estimate follows the same high-level workflow as alpha
estimate today.

Target areas:

- `GroupingDialog`
- `MainWindow._open_shared_grouping_dialog`
- `MainWindow._load_files`

Checks:

- the selected reference run is the only dataset used for the estimate step
- all selected targets receive the resolved deadtime payload
- future loaded datasets inherit the deadtime payload through the existing
   grouping-template mechanism

### 5. WiMDA UI parity checks

Verify that the chosen Asymmetry grouping-window controls preserve the intended
WiMDA mode semantics even if the layout is adapted for Qt.

Target areas:

- deadtime mode selector state
- manual detector-value editor state
- per-detector `Cal` action and calibrated-table status display
- estimate status/provenance display

### 6. Reference-program comparisons

Where portable fixtures exist, compare Asymmetry outputs against:

- musrfit file-based deadtime outputs
- Mantid `ApplyDeadTimeCorr` outputs
- WiMDA calibrated-table, estimate, or manual deadtime workflows

## Acceptance Criteria For Implementation Pass

1. All approved synthetic cases pass in Asymmetry.
2. Existing deadtime tests still pass.
3. Any newly supported source mode has provenance recorded in grouping or
   project state.
4. Deadtime estimate and inheritance behavior are covered by focused GUI/main-
   window tests.
5. Comparison results are written back into this study directory.

## Harness Commands

Expected focused checks during implementation:

- `python tools/harness.py test -- tests/test_deadtime.py`
- `python tools/harness.py test -- tests/test_psi_loader.py`
- `python tools/harness.py test -- tests/test_grouping_dialog.py`
- `python tools/harness.py test -- tests/test_mainwindow_additional.py`

If new scaffolding tests are added under `tests/porting/deadtime-correction/`,
they should be runnable independently before widening to the existing suite.