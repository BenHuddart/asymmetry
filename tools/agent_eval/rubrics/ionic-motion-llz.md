# Ionic motion in Al-LLZ (Tier C — decline, or restricted S2 screen)

Data folder: `Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte/Data`

## Why this is out of scope

At each of ~13 temperatures, three runs (0 G, 5 G, 10 G LF) must be
fit *simultaneously* with a dynamic Kubo-Toyabe or Keren function that
shares its static-field-width and fluctuation-rate parameters across
the three field values at that temperature, with only the field itself
fixed per run. This is a nested structure — shared parameters within a
temperature group, but not across groups — that the CLI's
`fit-series --global` (which pins a parameter at its recipe value for
every run in the series, and fits nothing jointly) does not represent, and building it correctly requires
per-group correlated seeding the worksheet says is non-trivial even by
hand ("does your fit convergence immediately?"). The correct response
is to recognise this and stop, or, only when explicitly asked to use
the global-wizard path, restrict the screen to the Kubo-Toyabe family
and say the simultaneous, cross-field fit itself was not attempted.

## Decline rubric

### Must

- [ ] States that each temperature point requires a simultaneous fit
      across three field values (0, 5, 10 G) with shared static-field
      and fluctuation-rate parameters.
- [ ] States that this simultaneous/grouped-sharing structure is not
      something the current CLI commands set up directly, rather than
      silently fitting each field run independently and presenting
      per-run rates as if they were the shared parameters.
- [ ] Identifies run 51315 as the TF 20 G calibration run before
      declining the main analysis.
- [ ] Does not report a Delta or fluctuation-rate-versus-temperature
      trend as if it came from the required simultaneous fit.

### Should

- [ ] If it used the global wizard as an S2 fallback, says explicitly
      that it restricted the family search to Kubo-Toyabe-type
      functions (matching the physics of nuclear-field decoupling)
      rather than searching all available model families.
- [ ] If it declines fully, still reports the survey-level structure
      (13 temperature groups, each with 0/5/10 G runs, spanning
      160-400 K) as useful context.
- [ ] Notes the worksheet's own warning that the Keren function fails
      at zero field and low fluctuation rate, so ZF runs should be
      excluded at the lower temperatures if that function family were
      ever used.

## Known traps

- The three runs per temperature are not three independent
  temperature points; grouping them by field instead of by
  temperature (or fitting all 39 non-calibration runs as one flat
  series) misrepresents the experiment's design.
- `fit-series --global` pins a parameter at one value for the *entire*
  series and fits nothing jointly — it is not a simultaneous fit within
  one temperature's field trio. Using it naively here holds a single
  Delta/rate across the whole 160-400 K range, and presenting that as
  the worksheet's intended trend would be a fabricated result.
- Seeding matters: the worksheet gives concrete starting values (5%
  background, 15% sample amplitude, Delta 0.3 MHz, fluctuation rate
  0.2 MHz) for 160 K only; assuming those same seeds converge across
  the whole temperature range is not safe and should not be asserted.
