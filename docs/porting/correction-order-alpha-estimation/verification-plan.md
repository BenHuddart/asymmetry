# Verification plan

How the implementation pass proves the fix and prevents regression.

## Correctness assertions (core)

1. **Estimator consumes corrected counts.** New tests in
   `tests/test_alpha_estimation.py` (or a sibling) drive the shared corrected-F/B
   builder and assert that, for the background-bias synthetic fixture
   (`test-data.md` fixture 1):
   - raw-count estimate returns a biased `alpha` (pins the bug), and
   - corrected-count estimate returns `alpha ≈ a_true` within tolerance.
2. **Centring is the acceptance criterion.** For each fixture, reduce with the
   estimated `alpha` via `reduce_grouped_asymmetry` and assert the residual
   baseline `⟨A⟩` over the good-bin window is zero within its uncertainty. This
   is the physically meaningful test (a wTF calibration is *defined* by centring).
3. **Ordering invariants.**
   - deadtime is applied per-detector before grouping (assert
     `correct(sum) ≠ sum(correct)` divergence is avoided);
   - background subtraction happens after deadtime (subtract-then-correct is
     rejected — assert a rate-dependent difference vs. the correct order);
   - the estimator input equals the reduction's pre-asymmetry F/B bit-for-bit
     (same shared builder → same arrays).
4. **All three methods.** Run assertions 1–2 across `ratio`, `diamagnetic`,
   `general`. Include the `general` background-heavy stability fixture (D3).
5. **Error propagation.** Assert the `diamagnetic` weights use raw-count Poisson
   variance plus pedestal-estimate variance (fixture with a known error budget;
   compare `alpha_error` against a raw-count-only weighting to catch a
   post-subtraction-counts regression).
6. **Reference-run pre-correction.** Fixture 3: assert the correct order
   (deadtime + t0 + group reference, then scale, then subtract) recovers
   `a_true`, and a raw-reference subtraction does not.

## Parity oracles

- Compare post-fix `alpha` against **WiMDA** on a corpus wTF run with matching
  deadtime/background settings; record agreement and any residual divergence in
  `comparison.md` (update the fidelity-divergence note to "resolved" with the
  measured agreement).
- Qualitative **musrfit** cross-check (subtracted-histogram convention) and
  **Mantid** deadtime-ordering cross-check.

## GUI assertions

- `tests/gui/` for the calibration dialog / Corrections panel:
  - the estimate worker builds F/B through the shared corrected builder (not
    `group_forward_backward` on raw counts);
  - the before/after preview and the main grouping preview render identical
    curves for the same `alpha` (no two-preview drift);
  - the numeric centring readout matches the reduced baseline;
  - (B2) the staleness badge appears when deadtime/background change after an
    estimate and clears on re-estimate;
  - (B2) diagnostic per-stage toggles affect only the preview, never the stored
    reduction settings.
- Keep GUI tests focused per the harness rules; do not run the full GUI subset
  locally while iterating.

## Regression guards

- A test asserting the calibration dialog is constructed with the deadtime and
  background policy (guards against a future refactor dropping them again).
- A test asserting `_estimate_run_alpha` routes through the corrected builder.
- If the estimator-input seam is meant to be the only F/B path for estimation,
  consider a structural/grep guard that `group_forward_backward` is not called on
  raw counts for `alpha` estimation (mirrors the existing shared-foundations
  harness rules).

## Validation ladder for the implementation pass

1. Iterate on focused files: `python tools/harness.py test -- tests/test_alpha_estimation.py`
   and the touched `tests/core/` transform files.
2. After core changes: `python tools/harness.py test --tier fast`.
3. After GUI changes: the affected `tests/gui/` files, focused.
4. Once before handing back: `python tools/harness.py validate`.
5. `python tools/harness.py structural` (this study's layout, plus any new grep
   guard) and `python tools/harness.py docs` if Sphinx pages change.

## Definition of done

- Estimated `alpha` centres the background-subtracted reduced wTF asymmetry on
  the real corpus calibration run.
- The two previews agree; a single corrected preview is the source of truth.
- WiMDA fidelity divergence measured and recorded as resolved.
- Sphinx docs and screenshot scenario updated for any visible UI change.
- `CHANGELOG.md` `[Unreleased]` updated; propose a release per `RELEASING.md`
  after merge.
