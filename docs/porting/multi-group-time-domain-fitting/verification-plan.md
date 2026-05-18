# Multi-Group Time-Domain Fitting Verification Plan

## Goal

Verify that any future multi-group or multi-detector time-domain fitting added
to Asymmetry matches the approved subset of WiMDA, musrfit, and Mantid
behavior.

## Baseline Assertions

These assertions should remain true unless the study docs are updated with a
new decision:

1. One simultaneous-fit objective can operate over multiple explicit domains.
2. Parameter sharing is explicit through global or local roles, not hidden in
	UI state.
3. Grouped-count domains preserve explicit per-domain `N0`, background, and
	relative phase.
4. Count-domain lifetime handling is explicit and stable.
5. Asymmetry-domain simultaneous fitting remains supported and backward
	compatible.
6. Detector phase metadata remains a separate contract from the physical model
	parameters.
7. Group or detector provenance is preserved through fitting and result
	reporting.

## Planned Checks

### 1. Core domain-model characterization

Add focused unit tests around the new fit-domain contract.

Target areas:

- future `asymmetry.core.fitting` domain adapter module
- existing simultaneous-fit engine entry points

Checks:

- domain kind is explicit
- required metadata for counts versus asymmetry domains is validated
- domain ordering does not silently change parameter mapping

### 2. Shared and local parameter behavior

Add tests that prove the same physical parameter can be:

- shared across all included domains
- local to each domain
- fixed globally

Target area:

- `FitEngine.global_fit` or its successor adapter layer

### 3. Grouped-count equation parity

Add deterministic synthetic tests for the approved grouped-count model.

Checks:

- per-group `N0` is identifiable and correctly applied
- per-group background is identifiable and correctly applied
- per-group relative phase changes the observed count trace in the expected way
- lifetime handling matches the approved formulation

### 4. Asymmetry backward compatibility

Verify that current asymmetry-only simultaneous fitting still works unchanged.

Target areas:

- existing global-fit wizard and engine tests

Checks:

- old asymmetry workflows still fit
- parameter-role recommendations still map cleanly onto the simultaneous-fit
  engine

### 5. Reference-program comparisons

Where stable fixtures exist, compare Asymmetry outputs against:

- WiMDA `fgAll` grouped-count fits
- WiMDA LF-sequence multifits
- musrfit multi-RUN global fits
- musrfit single-histogram count fits
- Mantid TF asymmetry normalized-domain behavior

Detector phase-table behavior should be validated separately against:

- Mantid `PhaseQuadMuon`

### 6. GUI and persistence checks

When a later implementation adds user-facing controls, verify:

- selected domains are persisted correctly
- local versus global role choices are persisted correctly
- per-group or per-detector nuisance parameters keep provenance

## Acceptance Criteria For The Implementation Pass

1. Synthetic grouped-count cases pass against the approved equations.
2. Existing asymmetry global-fit tests still pass.
3. Shared versus local parameter roles are covered by focused tests.
4. At least one WiMDA or musrfit reference comparison is automated for grouped
	counts.
5. Detector-phase metadata is either validated mechanically or explicitly left
	deferred in the docs.

## Harness Commands

Expected focused checks during a later implementation pass:

- `python tools/harness.py structural`
- `python tools/harness.py test -- tests/test_global_fit_wizard.py`
- `python tools/harness.py test -- tests/test_global_fit_wizard_window.py`

If dedicated porting tests are added, they should be runnable independently
before widening to the existing suite.