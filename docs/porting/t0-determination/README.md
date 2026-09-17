# Time-zero (t0) determination — audit and study

**Status:** study (2026-09-17). No behaviour changed in this pass.

**Question.** How should Asymmetry decide the analysis time-zero of a run —
from the file header, from an automatic search, or from the user — and how
should it show the user when those sources disagree? The user's stated
policy for this study: *default to file values, always display the value
our automatic method finds, and warn when the two diverge significantly.*

**Why now.** Opening the grouping window on a fresh PSI GPS MusrRoot run
showed the t0 mode already set to **Manual**. That turned out to be an
inference bug (the stored common t0 is the max over the forward/backward
group detectors, but the "is this manual?" check compares against the max
over *all* detectors), and pulling on it exposed that the t0 policy is only
honoured by the reduction path — grouped Fourier, MaxEnt and count-domain
fits align on the file t0 regardless of mode, and the count-fit t0
promotion moves `t0_bin` in the wrong direction. Getting t0 right is a
precondition for every analysis, so this study audits the whole lifecycle
against WiMDA, Mantid and musrfit before any fix lands.

## Files

| File | Contents |
|---|---|
| [comparison.md](comparison.md) | What t0 *means* per facility (with the facility/author documentation), then how each program reads it, searches for it, lets the user override it, aligns detectors, and warns — with file:line citations — and a divergence table. |
| [implementation-options.md](implementation-options.md) | Findings ranked by consequence, the recommended design (file default, always-visible detected value, tolerance-based divergence warning, one effective-t0 chokepoint), and the alternatives considered. |
| [test-data.md](test-data.md) | Corpora and synthetic records available to calibrate the divergence tolerance and to pin parity. |
| [verification-plan.md](verification-plan.md) | Tests to add per phase, oracles transcribed from the reference programs, and the corpus sweep that sets the warning thresholds. |

## Reference-source placeholders

`$WIMDA_SRC`, `$MANTID_SRC`, `$MUSRFIT_SRC` as defined in
[../README.md](../README.md). Line numbers were taken on 2026-09-17 from the
local checkouts.

## Headline findings (details in comparison.md § Asymmetry and implementation-options.md)

1. **Manual-mode misclassification on fresh files** —
   `_t0_policy_from_payload` compares the stored `t0_bin` (group max) with
   the max over all `detector_t0_bins`. Confirmed on a 15-detector GPS run:
   one detector outside the F/B groups carried a header t0 8 bins above the
   group max, so the fresh draft opened as Manual. Both stored profiles in
   that project already carry the mislabelled policy.
2. **Policy honoured only by reduction** — `EFFECTIVE_DETECTOR_T0_KEY` is
   read by `reduce.py` and `group_forward_backward` only; grouped Fourier,
   MaxEnt and count-domain fits call `common_t0_for_groups` /
   `apply_grouping_aligned` without the override. MaxEnt additionally builds
   its time axis from the *shifted* `grouping["t0_bin"]` while its counts are
   aligned on the file t0.
3. **Count-fit t0 promotion has the wrong sign** — the model evaluates
   `exp(-(t + t0)/τ)`, so a positive fitted `t0` means the stored bin is too
   *late*, yet `promote_t0_to_grouping` adds `round(t0/w)`. Verified
   numerically: with the stored t0 corrupted by +3 bins the fit returned
   +1.9 bins and promotion moved `t0_bin` from 103 to 105 (truth 100).
4. **No side-by-side display or divergence warning** exists, although
   `docs/reference/data_reduction/t0_search.rst` says the found value "sits
   beside the file value". Mantid and WiMDA have none either; musrfit's
   `musrt0` at least draws the file t0 as a reference line.
5. **Missing header t0 silently becomes bin 0** in every loader. musrfit
   treats a file t0 ≤ 0 as absent, falls back to the prompt-peak estimate and
   prints a loud warning; Mantid and WiMDA trust whatever is there.
6. **NeXus runs are second-class** — the loader writes `t0_bin` from
   detector 0 (not the group max) and no `detector_t0_bins`, so per-detector
   `time_zero` arrays (which ISIS v2 files can carry, and which Mantid reads)
   never reach the profile machinery.

## Decision summary (proposed — awaiting confirmation)

- Keep **From file** as the default and make the policy an explicit stored
  choice; never infer Manual from a value comparison again.
- Always compute the automatic estimate for the preview run (cached, off the
  GUI thread) and show it read-only beside the file value in every mode,
  with the per-detector spread.
- Warn when `|detected − file|` exceeds a per-source tolerance (initially 2
  bins continuous, 3 bins pulsed, to be calibrated by the corpus sweep in
  verification-plan.md), when any single detector disagrees with the file
  by more than that, when the file t0 is missing/zero, or when the file t0
  lies outside the histogram.
- Route *every* consumer of per-detector alignment through one
  `effective_detector_t0_bins` resolver, harness-enforced.
- Re-express **Manual** as an offset from the file t0 (what it already is
  at apply time), with the spinbox still showing the resolved absolute bin
  for the preview run.
- Fix the promotion sign, the NeXus loader's payload, the `nexus_writer`
  mix of file and effective values, and the plot mask's detector-0 axis.
