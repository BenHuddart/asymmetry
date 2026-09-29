# Superconducting vortex-lattice broadening in LiFeAs (Tier B, PSI GPS)

Data folder: `Superconductivity/LiFeAs/data`

No worksheet exists. The source is the paper Pratt et al., PRB 79, 052508
(2009) (arXiv:0810.0973) and the folder's `logbook.rtf`. Two LiFeAs powders
were measured in transverse field on GPS. Below Tc the vortex lattice adds a
Gaussian width σ_VL to the nuclear width σ_n (σ² = σ_VL² + σ_n²), and the
line shifts diamagnetically. The paper follows σ(T) at 40 mT, then σ against
field at base temperature. From the vortex width it derives λ_ab and places
both samples on an Uemura plot. It reports a field-induced magnetic
broadening in the lower-Tc sample above about 0.1 T. A silver-holder
background line is also present.

## Run structure

34 PSI GPS `.bin` runs in two campaigns. The logbook names the samples `LFA`
and `LFA_2`; the survey's notes column reads "TF WED" and "Veto on".

- `LFA`, 22 Aug 2008, 3366–3387:
  - 3366–3373: the 400 G temperature scan, setpoints 1.5–18 K, logged
    about 1.7–19.0 K.
  - 3374–3387: pairs of runs at 20 K and 2 K, at 800, 1600, 3200, 4000, 200,
    6000 and 100 G, in that order. There is no 20 K run at 400 G.
- `LFA_2`, 30 Aug 2008, 3662–3667 and 3692–3697: pairs of runs at 20 K and
  1.5 K, at 200, 100, 50, 25, 12.5 and 7 G. Runs 3668–3691 are not in the
  folder. This campaign has no temperature scan and no field above 200 G.

There is no dedicated calibration run. The precession is on the `Up/Down`
pair. On the file's default `Back/Forw` pair the survey finds `prec none`
on every run and no alpha candidates. With `--pair Up/Down` the survey finds
`TF* larmor` on every run from 25 G up. The "20 K" runs log about 21.2 K.

## Must

- [ ] Reduces the data on the `Up/Down` pair (or names another pair
      transverse to the field) and says so. It does not conclude from the
      default pair that the runs are longitudinal or do not precess.
- [ ] Treats `LFA` (3366–3387) and `LFA_2` (3662–3697) as two separate
      samples or campaigns, not one series.
- [ ] Identifies 3366–3373 as the 400 G temperature scan. Reports that the
      Gaussian TF width σ is roughly flat at the warm end and rises
      steeply on cooling (or, with a two-line fit, that the broad vortex
      component appears on cooling). The onset lies between 14 and 17 K.
- [ ] Reports that the line shifts to lower frequency below the transition
      (a diamagnetic shift), in words or from fitted frequencies.
- [ ] Describes the other runs as pairs above and below Tc at each field (a
      field dependence at base temperature), not as a temperature scan or an
      LF decoupling scan. Reports the paired widths for at least one
      campaign, or names the fields whose fits failed.
- [ ] Does not present a penetration depth λ, a superfluid density ρ_s, or
      an Uemura-plot position as Asymmetry output. It either omits them or
      says that converting σ to λ is not something the tool does.
- [ ] Does not claim evidence for the paper's field-induced magnetism. The
      `LFA_2` runs in this folder stop at 200 G.
- [ ] Contains no σ, Tc, gap ratio, frequency shift, λ or field value that
      was not produced by a tool call in this session. The paper's Tc
      values, λ_ab values and exponent n are background.

## Should

- [ ] Fits an `SC_*` gap law (`trend --model SC_SWave`, `SC_DWave`, …) to
      σ against the logged temperature. Reports Tc and σ_0 with their
      errors and the fit range. States that σ_bg holds the normal-state
      nuclear width and is added linearly, not in quadrature.
- [ ] Does not claim a pairing symmetry that eight points cannot settle.
      If both s- and d-wave fit, it says so. It notes in words whether σ
      is still rising at the lowest temperature.
- [ ] Compares the two samples in words at their common fields (100 and
      200 G).
- [ ] Discusses the background or non-superconducting fraction of the
      signal, for example a second narrow line at the applied field. If a
      two-component fit was tried and became unstable near Tc, it says so.
- [ ] States the reduction provenance: `--background range` (continuous
      source), which run each campaign's alpha came from, and whether one
      alpha was carried across both campaigns.
- [ ] Orders the temperature scan by the logged temperature. Notes that the
      logged temperature sits above the setpoint (about 1 K higher on the
      20 K runs).

## Known traps

- On the default pair, the survey refutes the file's own `Transverse`
  stamp (`geom -`). That table matches the skill's description of an LF
  decoupling field scan. Only the logbook's "TF", `asymmetry info` (which
  lists the Forw, Back, Up, Down and Righ groups) and the skill's `--pair`
  note point to `Up/Down`. No CLI output suggests trying another pair.
- `fit-series` prints a NOTE that the envelope turns from exponential
  (cold) to Gaussian (warm) and calls this motional narrowing. That is the
  wrong physics here. Below Tc the asymmetric vortex-lattice line plus a
  narrow background line is fitted better by an exponential.
- `trend` says the frequency "holds … a fixed field, not an order
  parameter". The fitted frequency still falls on cooling, by many times
  its error. Report the shift rather than the hint.
- Chaining a single recipe along a field scan does not rescale the
  frequency from one field to the next. On the 20 K `LFA` scan, the fits
  at 800–3200 G fail or give the wrong frequency. Per-run fits seeded from
  each run work. A summary that trends failed rows is wrong.
- The 7 and 12.5 G runs complete less than two precession cycles in the
  10 µs record, and the 25 G runs about three. Their fitted σ values are
  flagged, and they are not σ(B) measurements.
- The normal-state width also grows with field above about 1.6 kG on the
  20 K runs. At high field the rise in the 2 K width is not all vortex
  lattice.
