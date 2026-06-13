# Test data & oracle anchors — workflow-visualisation

Public repo: corpus paths are referenced via the `$ASYMMETRY_MUSRFIT_DATA` /
testing-corpus convention and gated behind an env var; WiMDA oracle citations use
`$WIMDA_SRC` placeholders (GPL — behaviour read, never copied). Unit tests that need
no corpus run unconditionally; corpus comparisons skip when the env var is unset.

## Corpus runs

The loadable testing corpus (HDF5 `.nxs`, PSI `.bin`/`.mdu`; HDF4 `.nxs` is out of
scope) lives under the testing-corpus root. Items here exercise:

| Item | Corpus need | Why |
|---|---|---|
| 3 Events columns | Any run with a grouping (good-bin range) and a frame count — ISIS `.nxs` (Nickel/Nuclear) and PSI `.bin`/`.mdu` (EuO/Chemistry) | Verify good-range MEv and events/frame against the run's own histograms |
| 4 B-from-log | A TF/LF run whose NeXus file carries a **magnetic-field time-series** log (not just the header scalar) | Confirm the log-derived mean differs from, and is preferred over, the header field |
| 6 Log-count | Any decay histogram with a clean exponential tail | The straight-line-on-log check; t0/background deviations visible |
| 7 F,B balance | A TF run with a known good α (the diamagnetic-α corpus already used in `test_alpha_estimation.py`) | Overlay coincidence under the correct α; visible gap under a wrong α |
| 8 Cursor readouts | A frequency-domain spectrum (FFT/MaxEnt) with a resolved peak; an asymmetry curve with a flat region | Parabolic-peak vs the spectrum's bin grid; windowed-average vs `integrate_curve` |
| 2 Export | Any plotted run (+ a fit for data+fit/fit-only) | Round-trip the text export → parse → match the plotted arrays |

**PSI naming.** PSI `.bin`/`.mdu` runs (EuO, Chemistry) carry different
filename/field conventions from ISIS; include at least one PSI run in the events-
column and B-from-log checks so the metadata accessors are not ISIS-only. (Run
stepping is REJECTED, so the PSI *filename-pattern* boundary cases from the brief's
verification sketch are moot — the browser lists PSI files the same as ISIS.)

## Oracle anchors (`$WIMDA_SRC`, behaviour-only)

For each verifiable item, the WiMDA source whose **numeric behaviour** the Python
reimplementation is checked against:

| Item | `$WIMDA_SRC` anchor | Quantity to match |
|---|---|---|
| 3 Good events / ev-per-frame | `LogbookUnit.pas:595–626` | Σ good-range counts ÷1e6; sum ÷ frame count |
| 4 B-from-log | `WiMDA_Main.pas:729–733`, `nexusunit.pas` log-average | Mean of the field log channel over valid points |
| 6 Log-count | `Plot.pas:1542–1548` | Straight-line slope −1/τ_μ on the count axis (qualitative + slope check) |
| 7 F,B balance | `Plot.pas:2701–2711` | Overlay of the two group traces (qualitative; balance under α) |
| 8 Parabolic peak | `Plot.pas:106–186` (`parabpkextrap`) | Vertex `x=−b/2a` from 3 points — **exact** formula match on synthetic parabolas |
| 8 Windowed average | `Plot.pas:1209–1226` | `mean ± √(Σy²/n − mean²)/n` — matches `integrate_curve` |
| 2 Export header/cols | `WiMDA_Main.pas:1414–1869` | Provenance fields present; content switch (data/both/fit) |

**Bug-ledger guards (do not oracle against these WiMDA behaviours).** From the
[decision record §3](../wimda-parity-gap/decision-record.md): #6 — General-α has no
interior minimum at realistic statistics (relevant if the F,B balance overlay is
used to *judge* α: the overlay is the visual that exposes exactly this WiMDA
failure, so it is a complement, not an oracle); #7 — t0 search ceiling/peak bugs
(the log-count diagnostic is the *human* check that catches t0 mis-placement).

## Golden files

- **Export round-trip:** write a small golden text export for one ISIS run (data
  only) and one with a fit (data+fit), parse it back, assert the columns equal the
  plotted arrays and the header carries run/grouping/α provenance.
- **Parabolic peak:** synthetic 3-point parabolas with known vertices (pure unit
  test, no corpus) — the exact-formula gate.
- **Windowed average:** assert the cursor readout equals `integrate_curve` over the
  same window on a fixture curve (pure unit test).
- **Events columns:** for one ISIS and one PSI run, assert good-range MEv equals a
  direct numpy sum over `[first_good_bin, last_good_bin]` ÷1e6, and events/frame
  equals that ÷ `good_frames` (env-gated corpus test; pure-fixture variant for CI).
