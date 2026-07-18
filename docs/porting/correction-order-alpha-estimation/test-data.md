# Test data

Datasets and synthetic fixtures for validating that `alpha` estimation consumes
corrected counts and that the reduced wTF asymmetry centres on zero.

## Real corpus

The WiMDA Muon School corpus (`~/Documents/WiMDA muon school/`, guide in
`docs/testing/`) provides real wTF calibration runs:

- A **weak-TF calibration run** (field in `WEAK_TF_FIELD_RANGE_GAUSS`, see
  `src/asymmetry/core/data/calibration.py:46`) with a non-negligible background —
  the primary end-to-end oracle: after the fix, the estimated `alpha` must centre
  the *background-subtracted* reduced asymmetry on zero.
- A run with **file-provided per-detector deadtime** and materially different F/B
  count rates — exercises the deadtime half of the ordering.
- A run with an associated **empty-sample / reference run** for the
  reference-run background mode — exercises reference pre-correction.

Loaders in scope (`.nxs` HDF4 via `core/io/hdf4.py`, `.bin`, `.mdu`) already
cover these; no new loader work.

## Synthetic fixtures (deterministic, primary for CI)

Synthetic F/B lets the correct `alpha` be known exactly, so the test asserts a
number rather than eyeballing. Build in `tests/` (not shipped data):

1. **Background-bias fixture.** Construct `F = a_true · μ(t) + bg_F` and
   `B = μ(t) + bg_B` with a known efficiency ratio `a_true`, a muon-decay
   envelope `μ(t) = N0·exp(-t/τ)·(1 + A·cos(ω t + φ))` (wTF), and flat pedestals
   `bg_F`, `bg_B` chosen so `bg_F/bg_B ≠ a_true`.
   - Estimate on **raw** counts → biased `alpha ≠ a_true`; reduced (subtracted)
     asymmetry has a non-zero baseline. (Pins the bug.)
   - Estimate on **background-subtracted** counts → `alpha ≈ a_true`; reduced
     asymmetry centred. (Pins the fix.)
   Assert the residual baseline `⟨A⟩` over the good window crosses from
   significantly non-zero to ≈ 0 within its uncertainty.

2. **Deadtime-bias fixture.** Two detectors per group with different rates and
   known `τ`. Apply the non-paralyzable model in reverse to synthesise raw
   counts from a known corrected signal, so the corrected `alpha = a_true`.
   - Estimate on raw grouped counts → biased; on deadtime-corrected → `a_true`.
   Run across all three methods; assert the `ratio` method shows the largest raw
   bias (per `comparison.md` sensitivity table).

3. **Reference-run fixture.** Sample run + empty-sample reference with **different
   rates** and a small **t0 offset** between them. Correct order (deadtime + t0 +
   group the reference, then scale, then subtract) recovers `a_true`; a
   raw-reference subtraction does not. Pins Q2 pre-correction.

4. **`general`-method stability fixture.** A background-heavy wTF case: confirm
   the `general` estimator is unstable on un-subtracted counts (large / divergent
   objective at late t) and stable after subtraction. (Divergence D3.)

## Reference oracles

- **WiMDA** (`$WIMDA_SRC`): estimate `alpha` on a corpus wTF run with deadtime +
  background enabled in its reduction settings; record its `alpha` and compare to
  Asymmetry's post-fix estimate on the same run/settings (target: agreement
  within estimator tolerance; document any residual divergence).
- **musrfit** (`$MUSRFIT_SRC`): a `.msr` asymmetry-mode fit with background
  subtraction and fixed/fitted `alpha`; confirm Asymmetry's centred reduced
  asymmetry matches the musrfit-subtracted-histogram convention qualitatively.
- **Mantid** (`$MANTID_SRC`): `MuonPreProcess` (deadtime) → `AlphaCalc`; use as
  the deadtime-ordering oracle only.

## Expected-value manifest

For each synthetic fixture record, in the test module, the known `a_true`, the
expected raw-count (biased) `alpha`, the expected corrected `alpha`, and the
tolerance. These are the golden values the verification tests assert against.
