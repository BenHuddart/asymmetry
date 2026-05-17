# Fourier Transform Verification Plan

## First Slice Checks

Run the smallest checks that can falsify the phase-handling hypothesis:

1. `python tools/harness.py structural`
2. `python tools/harness.py test -- tests/test_fourier.py`
3. `python tools/harness.py test -- tests/test_gui_panels_basic.py tests/test_project_schema.py`
4. `python tools/harness.py test -- tests/test_fourier_reference_methods.py`

## Behavior To Verify

### Core FFT

- manual phase rotation changes the complex spectrum as expected
- WiMDA-style `t0` offset changes the complex spectrum through the expected
  frequency-dependent phase term
- the dominant-frequency real component increases when the correcting phase is
  applied to a phase-shifted cosine
- existing `fft_asymmetry` callers still receive the same three-array contract
- WiMDA-style projection remains equivalent to the current manual phase-rotation
  seam in Asymmetry

### WiMDA Target

- the active parity target is WiMDA grouped/manual Fourier behavior
- per-group phase tables and automatic phase estimation are the next feature
  slices to validate
- excluded frequency bands zero the requested FFT bins before plotting
- the main-window frequency axis can switch between absolute and field-relative display
- averaged grouped FFTs can carry WiMDA-style estimated error bars

### Reference Differences

- musrfit lifetime correction remains a distinct advanced preprocessing step
- musrfit's linear phase family remains outside the current WiMDA-first scope
- Mantid's detector phase-table combination remains advanced functionality,
  distinct from any single-trace grouped FFT path

### GUI State

- Fourier panel defaults include the new phase controls
- `get_state()` and `restore_state()` round-trip the new phase fields

## Follow-On Verification

After grouped or detector-specific phase work lands:

1. add targeted tests for phase-table parsing and persistence
2. add targeted tests for exclusion-range parsing, relative x-axis display, and
  averaged grouped-error estimates
3. compare one synthetic grouped case against a detector-phase-corrected case
4. run `python tools/harness.py validate` when the feature crosses more of the
   GUI and project boundaries