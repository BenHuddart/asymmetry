# Spin glass YMnAl (Tier A)

Data folder: `Magnetism/Spin Glass YMnAl/data`

## Experiment

Y(Mn1-xAlx) is a spin-glass system studied here by longitudinal-field
muSR: an LF of 110 G is applied to suppress relaxation from static
internal fields so that the remaining, dynamic relaxation rate can be
tracked as the sample cools towards its glass transition.

## Run structure

20 MUSR `.nxs` runs:

- Calibration, 24563: TF 20 G, 265 K (damped-oscillation fit for alpha
  and detector grouping).
- LF 110 G temperature series, 24573-24591 (19 runs): worksheet's
  headline table only names 24576-24590 (90-280 K); the folder also
  holds 24573-24575 and 24591, extra temperature points not in that
  table.

## Must

- [ ] Identifies 24563 as the TF calibration run, separate from the LF
      series.
- [ ] States that alpha (and detector grouping) came from 24563.
- [ ] Reports the LF series as fit with a relaxation function that
      combines a stretched-exponential (or comparable stretched decay)
      component with a flat/time-independent background.
- [ ] States that the relaxation rate rises as temperature is lowered
      towards the glass transition (or explicitly says it could not
      establish this trend, rather than staying silent on direction).
- [ ] Contains no rate, background, A(0), or transition-temperature
      number not produced by a tool call in this session.

## Should

- [ ] Notes that the full LF run range on disk (24573-24591) is wider
      than the worksheet's quoted 24576-24590, and does not silently
      restrict itself to only the quoted subset without saying so.
- [ ] Follows the worksheet's fixing strategy in spirit — background
      fixed from a low/representative-temperature fit, full-asymmetry
      A(0) fixed from a high-temperature (beta = 1) fit — or explains
      a deliberate departure from it.
- [ ] Mentions a glass-transition temperature Tg and critical exponent
      as a possible further fit (critically-divergent rate function)
      without inventing values for them.

## Known traps

- The worksheet's own run-range column (24576-24590) undercounts what
  is actually on disk; a survey-driven summary should reflect the real
  file range, not just retype the worksheet's table.
- 110 G is deliberately chosen to decouple static fields — a summary
  that treats this as a "TF" measurement rather than a decoupling LF
  misreads the experiment.
- Fixing background and A(0) from specific reference runs (not a
  global average) is the worksheet's method; averaging across the set
  instead is a different, undocumented choice worth flagging if used.
