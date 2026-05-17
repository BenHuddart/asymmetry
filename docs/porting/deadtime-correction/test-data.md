# Deadtime Correction Test Data

## Chosen Primary Dataset

Primary comparison dataset: synthetic analytical cases owned by Asymmetry.

Reasoning:

- The shared file-based formula is explicit and easy to verify analytically.
- The synthetic cases are stable, redistributable, and runnable in the harness.
- The reference repos do not expose one obvious small committed fixture that
  covers every source mode in a portable way.

## Primary Cases

### Case A: File deadtime with good frames

Inputs:

- counts: `[100.0, 200.0]`
- tau_us: `0.01`
- bin_width_us: `0.02`
- good_frames: `1000.0`

Expected outputs:

- corrected counts:
  - `105.26315789473684`
  - `222.22222222222223`
- applied: `true`
- method: `file`

Reference rationale:

- This is the exact formula shared by musrfit, Mantid, and current Asymmetry.

### Case B: Missing file deadtime values

Inputs:

- counts: `[100.0, 95.0, 90.0]`
- no `dead_time_us`
- `use_deadtime=true`

Expected outputs:

- counts unchanged
- applied: `false`
- no deadtime method recorded

Reference rationale:

- This captures the current Asymmetry behavior and the documented PSI path.

### Case C: Partial per-detector correction payload

Inputs:

- detector 1 tau_us: `0.0`
- detector 2 tau_us: `0.01`
- same counts and good frames for both

Expected outputs:

- detector 1 unchanged
- detector 2 corrected

Reference rationale:

- This mirrors Mantid's support for subset-spectrum correction via its deadtime
  table and tests the intended normalized-source seam for Asymmetry.

### Case D: Estimated deadtime uses the reference dataset only

Inputs:

- two loaded datasets with different early-time count envelopes
- one selected reference dataset
- deadtime mode: `estimate`

Expected outputs:

- the estimator runs once against the selected reference dataset only
- the resolved deadtime payload records `method: estimate`
- the same resolved payload is applied to every selected target dataset

Reference rationale:

- This is the same semantic rule Asymmetry already uses for alpha estimate in
  `GroupingDialog`: compute from the reference run, do not average over all
  loaded runs.

### Case E: Future loaded datasets inherit the latest deadtime grouping payload

Inputs:

- an existing grouped dataset with an estimated or manually selected deadtime
  payload
- a newly loaded dataset with no explicit deadtime override yet

Expected outputs:

- the new dataset inherits the latest grouping payload, including deadtime mode
  and provenance fields
- file-backed deadtime values may still be used when the chosen mode is file,
  but the active grouping template remains the owner of the user's deadtime
  choice

Reference rationale:

- This matches the current Asymmetry rule used for grouping inheritance when
  new datasets are loaded.

## Secondary Candidate Reference Data

These are useful for a later implementation pass, but are not yet the primary
portable comparison set:

- Mantid EMU NeXus test data used by `ApplyDeadTimeCorrTest.h` and
  `MuonProcessTest.h`
- WiMDA `dt*.dat` calibration files loaded from the data directory or `cal/`
- musrfit external NeXus example workflows that can emit file and estimated
  deadtime values

## Caveats

1. Mantid test names reference files supplied by Mantid's test environment, not
   obviously vendored in this workspace.
2. WiMDA code references calibration files, but the repo scan did not identify a
   small committed example pair that should be treated as canonical.
3. musrfit contains deadtime estimation utilities, but its main reduction path
   still treats estimate as not implemented.
4. WiMDA's estimator is a fit-based workflow, so synthetic harness cases for the
  implementation pass should split numerical formula checks from propagation
  and provenance checks.

## Planned Comparison Manifest

The machine-readable case list for the implementation pass lives in:

- `tests/porting/deadtime-correction/reference-cases.json`