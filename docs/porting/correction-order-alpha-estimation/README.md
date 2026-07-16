# Correction Order vs. Alpha Estimation Study

Status: study — approved physics direction pending an implementation pass on this
branch (`feat/correction-order-alpha-estimation`).

This study records how deadtime correction, background subtraction, detector
grouping, and detector-balance (`alpha`) estimation should be ordered in the
weak-transverse-field (wTF) calibration workflow, and how the grouping-setup UI
should present their combined effect. It exists because Asymmetry currently
estimates `alpha` on **raw grouped counts** while the reduction it feeds applies
deadtime and background corrections first — so a calibrated `alpha` that centres
the calibration dialog's preview does **not** centre the reduced wTF asymmetry.

The physics direction below was reviewed by a domain consult (µSR data reduction)
and cross-checked against the reference programs. It is recorded here as the
source of truth for the implementation pass.

## The bug this study addresses

Two independent code paths form forward/backward (F/B) spectra and they diverge:

- **Reduction path** — `reduce_grouped_asymmetry`
  (`src/asymmetry/core/transform/reduce.py:86`): deadtime → group → background
  subtraction → `binned_fb_asymmetry(alpha)`.
- **Alpha-estimate path** — both the interactive `AlphaCalibrationDialog`
  (`src/asymmetry/gui/windows/grouping/alpha_calibration_dialog.py:107`) and the
  project per-run estimate `_estimate_run_alpha`
  (`src/asymmetry/core/project/profiles.py:1183`) build F/B via
  `group_forward_backward` (`src/asymmetry/core/transform/grouping.py:500`),
  which only aligns t0 and sums raw `Histogram.counts`. **No deadtime, no
  background.**

Consequence: `alpha` is calibrated on `(signal + background)` counts and then
applied to background-subtracted, deadtime-corrected counts. Because the
background pedestal does not track the F/B efficiency ratio, the estimate
balances the *totals*, and the reduced wTF asymmetry acquires a constant offset.
The alpha dialog's own before/after preview reduces the same raw counts, so it
shows a nicely centred "after" curve — it is misleading about the very quantity
it exists to calibrate. The main grouping preview
(`src/asymmetry/gui/windows/grouping/preview_pane.py:359`) runs the full
corrected reduction, so the two previews disagree for the same `alpha`.

## Approved physics direction

The governing principle: **`alpha` is a property of the signal channel** (solid
angle × detector efficiency × geometry for F relative to B), not of the total
count stream. The invariant to encode:

> the `alpha` estimator consumes the output of the same pipeline the reducer
> uses, one step before the asymmetry is formed.

Concretely, `alpha` must be estimated on **deadtime-corrected → t0-aligned →
grouped → background-subtracted** counts, for all three estimator methods
(`ratio`, `diamagnetic`, `general`). The required correction order and the
per-method sensitivities are recorded in `comparison.md`; the reference-run
background must itself be deadtime-corrected, t0-aligned, and grouped identically
before scaling (also in `comparison.md`).

This is both a **code bug** and a **WiMDA fidelity divergence**: WiMDA — the
reference the `diamagnetic` method was ported from — estimates `alpha` from the
spectra as corrected by the current reduction settings, so the port has diverged
from its reference on exactly this point.

## Approved UI direction

Unify to a single corrected-reduction preview so "what you see is what you
reduce". Every preview renders through one shared corrected-reduction function;
the alpha view is that same pipeline with `alpha` toggled between `1` and `alpha_hat`.
Retire the three separate modal dialogs (deadtime, background, alpha) in favour
of a non-modal Corrections panel with a pipeline strip, one live preview, a
numeric centring readout, preview-only per-stage diagnostic toggles, and a
staleness badge when `alpha` was estimated under different correction settings.
The full concept and its interim fallback are in `implementation-options.md`.

## Scope

- Correct ordering of deadtime, grouping, background, and `alpha` estimation.
- Making the `alpha` estimators (`estimate_alpha`, `estimate_alpha_detailed`)
  and both estimate entry points consume corrected F/B counts.
- Reference-run background pre-correction before subtraction.
- Unifying the grouping-setup previews onto the corrected reduction, and the
  Corrections-panel UI concept.
- Recording the WiMDA fidelity divergence and pinning it with tests.

Out of scope for the implementation pass unless explicitly pulled in:

- New background estimation *modes* (this study reuses existing modes).
- New deadtime *sources* (covered by the `deadtime-correction` study).
- Changing the estimator math itself (`ratio`/`diamagnetic`/`general` formulae
  are unchanged — only their inputs change).

## Current Asymmetry baseline

Core:

- `src/asymmetry/core/transform/asymmetry.py` — `estimate_alpha` (`:174`),
  `estimate_alpha_detailed` (`:478`), `_diamagnetic_objective` (`:257`).
- `src/asymmetry/core/transform/grouping.py` — `group_forward_backward`
  (`:500`), `apply_grouping_aligned` (`:101`).
- `src/asymmetry/core/transform/reduce.py` — `reduce_grouped_asymmetry`
  (`:86`), the canonical corrected order.
- `src/asymmetry/core/transform/deadtime.py` — `apply_deadtime_correction`
  (`:14`), `prepare_histograms_with_deadtime`.
- `src/asymmetry/core/transform/background.py` —
  `apply_grouped_background_correction` (`:274`), `resolve_background_mode`
  (`:117`).
- `src/asymmetry/core/project/profiles.py` — `_estimate_run_alpha` (`:1169`),
  `_apply_alpha_policy` (`:1150`).

GUI:

- `src/asymmetry/gui/windows/grouping/dialog.py` — `GroupingDialog`,
  `_estimate_alpha` (`:3375`), deadtime/background configure buttons.
- `src/asymmetry/gui/windows/grouping/alpha_calibration_dialog.py` — the modal
  alpha calibration dialog with its own raw-count preview.
- `src/asymmetry/gui/windows/grouping/deadtime_dialog.py`,
  `background_dialog.py` — the two other modals.
- `src/asymmetry/gui/windows/grouping/preview_pane.py` — the corrected live
  preview (`:359`).

Existing tests to extend (see `test-data.md` for the mapping):

- `tests/test_alpha_estimation.py`
- `tests/core/` transform coverage for `reduce`, `background`, `deadtime`.
- `tests/gui/` grouping-dialog and calibration-dialog coverage.

## Candidate port seams

1. **Core estimator input seam.** Introduce a single corrected-F/B builder that
   `estimate_alpha`/`estimate_alpha_detailed` callers use, reusing the
   `reduce_grouped_asymmetry` correction stages up to (but not including) the
   asymmetry step. Both estimate entry points route through it.
2. **Calibration-dialog data seam.** Hand the dialog the deadtime and background
   policy (it currently receives only groups/forward/backward/excluded) and make
   its preview and worker call the shared corrected builder.
3. **Preview unification seam.** One corrected-reduction render function shared
   by the grouping preview and the alpha before/after overlay.
4. **Provenance/staleness seam.** Tag a stored `alpha_hat` with a digest of the
   correction settings it was estimated under; surface a staleness badge on
   change (reuse the FFT-staleness banner pattern).

## Open questions (resolve during the implementation pass)

- Does the implementation land as one PR (unified Corrections panel) or as the
  interim fix (route the existing dialog's preview + worker through the corrected
  pipeline) followed by a UI PR? See `implementation-options.md`.
- Error propagation: after background subtraction the estimator's per-bin `sigma`
  must still carry raw-count Poisson variance plus the pedestal-estimate
  variance. Confirm `diamagnetic` weights use the propagated errors, not
  post-subtraction counts.
- Reference-run mode: the estimator needs the `ReferenceResolver`; confirm the
  calibration dialog and per-run path can both supply it, or degrade explicitly.
