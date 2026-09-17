# Time-zero (t0) determination — audit and study

**Status:** implemented (2026-09-17), on branch `feat/t0-determination`, per
[`docs/plans/t0-determination.md`](../../plans/t0-determination.md) (six
phases, one PR). All decisions below (D1-D12) landed as designed; the
"Decision summary" section is superseded by the plan's decision log, which
also records anything refined during implementation.

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
| [isis-header-index-base.md](isis-header-index-base.md) | Resolves the 0/1-based question for ISIS header bins (1-based, inclusive) from 1,245 files, and quantifies the sub-bin t0 the integer index discards. |

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
6. **ISIS header bins are 1-based** (resolved from the corpus, see
   isis-header-index-base.md); the loader's axis-vote heuristic gets this
   right on most files but not on exact-edge ones, and the integer `t0_bin`
   discards up to half a bin (8 ns) of the file's exact `time_zero`.
7. **NeXus runs are second-class** — the loader writes `t0_bin` from
   detector 0 (not the group max) and no `detector_t0_bins`, so per-detector
   `time_zero` arrays (which ISIS v2 files can carry, and which Mantid reads)
   never reach the profile machinery.

## Decision summary (as implemented — see the plan's decision log for D1-D12)

- Kept **From file** as the default and made the policy an explicit stored
  choice (`T0Policy.mode`); Manual is never inferred from a value comparison
  for a payload written since (D1). A file with no header t0 at all falls
  back to the detected consensus (D7).
- The automatic estimate for the preview run is computed off the GUI thread,
  cached per run digest, and shown read-only beside the file value in every
  mode, with the per-detector spread (D11).
- Warns when `|detected − file|` exceeds a per-source tolerance — **2 bins
  continuous, 3 bins pulsed**, as calibrated by the 1,245-file ISIS survey in
  [isis-header-index-base.md](isis-header-index-base.md) rather than the
  placeholder numbers this study proposed — when any single detector
  disagrees with its own file t0 by more than that, when the per-detector
  spread exceeds four tolerances, when the file t0 is missing, or when the
  file t0 lies outside the histogram (D8, D9: a warning never blocks Apply).
- Every consumer of per-detector alignment — reduction, grouped Fourier,
  MaxEnt, count-domain fits, the deadtime window, the plot mask — is routed
  through one `effective_detector_t0_bins` resolver, harness-enforced (D10).
- **Manual** is stored as a signed offset from each run's own file t0
  (`T0Policy.offset_bins`), with the spinbox still showing the resolved
  absolute bin for the preview run (D3). A pre-v21 absolute value is
  converted to the equivalent offset, or healed to From file when the offset
  resolves to zero everywhere, on project open (D2; schema v21).
- ISIS header bins decode as 1-based deterministically (D5), and the exact
  `time_zero`/MusrRoot `Time Zero Bin` sets the time-axis stamp rather than
  the nearest integer bin (D4) — resolving R10 in favour of keeping the
  exact value; a conflicting `time_zero`/`t0_bin` pair defers to the
  attribute (D6).
- Fixed the promotion sign (D12), the NeXus loader's payload, the
  `nexus_writer` mix of file and effective values, and the plot mask's
  detector-0 axis.
