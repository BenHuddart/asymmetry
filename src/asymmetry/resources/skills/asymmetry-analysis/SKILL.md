---
name: asymmetry-analysis
description: Analyse μSR run folders with the Asymmetry CLI (.nxs/.bin/.mdu/.root). Use to survey, select periods, calibrate, reduce, inspect Fourier spectra, fit ALC/QLCR field scans, screen time-domain models, jointly fit run groups, or summarise ZF/TF/LF scans.
---

# Preliminary μSR analysis with the `asymmetry` CLI

You have a folder of muon-spin-spectroscopy runs and a command-line tool that
surveys them, reduces them to forward–backward asymmetry, screens fit models,
fits a scan, and trends the results. This skill is the recipe for driving it
end to end and writing a summary a spectroscopist would recognise.

Full flag reference: `references/commands.md` beside this file. Every command
also has `--help`.

## 1. What this does, and what it does not

**In scope.** Preliminary forward–backward asymmetry analysis of:

- zero-field (ZF), transverse-field (TF) and longitudinal-field (LF) runs;
- a temperature scan or a field scan at one geometry;
- named or numbered acquisition periods in a multi-period run;
- Fourier spectra and quantitative peak finding on reduced data;
- integral-asymmetry field scans, including ALC/QLCR resonance fits;
- a simultaneous group of runs with genuinely shared fit parameters;
- a parameter trend fitted with a physical law — an order parameter, an
  Arrhenius or Redfield law, a superconducting gap, a linear rate law — along
  temperature (setpoint or logged), field, or a quantity you supply per run;
- ISIS NeXus (`.nxs`) and PSI (`.bin`, `.mdu`) files, one forward group
  against one backward group.

"Preliminary" is the operative word. You produce a defensible first pass — the
right model family, a trend, and flagged runs — not a publication analysis.

**Out of scope — say so plainly and stop.** Do not improvise around these. If
one is what the folder is for, run `survey` (it is cheap and honest), report
what the survey shows, say which of these cases it is and *why the tool cannot
do it*, and stop without producing fit numbers.

| Case | How the survey shows it |
|---|---|
| Count-domain fitting | You need per-detector counts with N₀ and a relaxation term, not asymmetry. `reduce` only produces asymmetry. |
| Multi-group / orientation-resolved analysis | The run has many detector groups that must be fit together (angle-dependent Knight shift, crystal rotations). This CLI reduces exactly one forward/backward pair. |
| Maximum-entropy spectra | `fourier` provides an FFT and peak table, not maximum entropy reconstruction. Do not describe its output as MaxEnt. |
| Negative-muon (μ⁻) elemental analysis | Gamma spectra, elemental lines. Not asymmetry data. |
| Rotating-reference-frame or RF-resonance runs | Titles or notes naming RF; data modulated at a reference frequency. |
| A series of simultaneous groups | `fit-global` fits one group of runs jointly and `trend` reads that group's run-local parameters, but there is not yet one command that repeats the coupled fit for every temperature and trends the *shared* parameters. Fit and report each group separately; do not substitute independent fits. |
| A trend of fitted trend parameters | `trend --model` fits one stored series. A law fitted *across* several such fits — an Arrhenius law through rate constants each fitted at one temperature — has no command. Report each fit's parameters; do not fit the second level by hand. |
| A fragment of a published multi-field campaign | No self-contained scan in the survey's `scans` list, and fields the files do not record. Check `scans` first: two complete temperature scans at two recorded fields are analysable even with a large gap in run numbers between them. |

A folder the tool can *load* is not automatically a folder the tool can
*analyse*. Loadability is not scope.

## 2. The workflow

Run every command from the project directory you are working in, and pass the
data folder as the command's argument — an absolute path is fine, and is what
you want when the data sits on a share or in an archive. `survey`, `reduce`,
`wizard`, `recipe`, `integral-scan`, `fourier`, `fit-global` and `fit-series` write into
`./asymmetry-work/` — the work directory,
in the project, never in the data folder — so the next command picks the state
up; `fit` and `trend` read it and add only what `--plot` (and `trend --csv`)
asks for. `alpha` and `info` are stateless — they load a file, print, and
write nothing — and `skill` writes into the agent's own skill directory, not
the work directory.

**Never write anything into the data folder.** It is the experiment's record,
and it is often read-only.

One work directory holds one data folder's runs — everything in it is keyed on
the run number alone. So for a **second data folder** in the same project, pass
`--workdir asymmetry-work-<short-name>` and keep passing that same `--workdir`
for every command on that folder. Otherwise do not pass `--workdir` at all: the
default is right, and a directory that already holds another folder's session
says so rather than mixing the two.

Add `--json` when you need to parse a payload; the default human table is
usually easier to read and is what these examples show.

### Step 1 — survey the folder

```bash
asymmetry survey <folder>
```

Read it properly before doing anything else. It gives you, per run:
temperature, field, geometry, **`prec`**, detector orientation, point count,
gross **event count**, number of acquisition **periods**, whether the file
carries deadtime values, title and notes; then the
**alpha-calibration candidates**, then **`scans`** — runs grouped into
temperature scans at fixed field and field scans at fixed temperature.

**`geom` and `prec` together are the geometry evidence.** For every run with a
non-zero recorded field the survey reduces the spectrum and looks for precession
at that field's Larmor frequency (γ_μ/2π × B = 13.55 kHz/G). `prec` reports what
it found:

| `prec` | Means |
|---|---|
| `larmor` | A strong line at the Larmor frequency of the recorded field. The field **is** transverse, and `geom` reads `TF*` — the `*` says the spectrum decided it, not the file. |
| `other@<MHz>` | A line at a frequency other than the Larmor one, printed beside it. The muon sees a field that is not the applied one: an **internal** field (an ordered magnet below its transition), muonium in a weak TF (1.394 MHz/G), the critical field inside a type-I superconductor's normal domains. The frequency tells you which. |
| `none` | No line worth the name. Whatever the file stamps, **this field is not precessing the muon**. The file's claim is refuted, so `geom` reads `-`: either the field is longitudinal, or it is transverse with no resolvable line. |
| `-` | Not measurable: a Larmor frequency above the record's Nyquist frequency (a kilogauss-scale field at a pulsed source), or no field recorded. |

**Zero-field runs are searched too.** There `other@<MHz>` is **spontaneous
precession** — a static internal field, the signature of long-range magnetic
order — and `none` means no line was resolved in that record (a paramagnet, a
Kubo–Toyabe, or an order whose field is too large or too broad to resolve).
A ZF temperature scan whose `other@` frequency falls on warming and gives way
to `none` is an order parameter going to zero at the transition.

**A line the survey tracks along a scan is a measurement.** When `other@`
changes smoothly from run to run — an internal field, a critical field, a
muonium line — quote that progression in the summary as the line's trend, run
by run, even if a time-domain fit of a weak line will not converge; say that
the values are the survey's spectral lines, and use your fits where they
succeed.

`scans` groups by **instrument** and the held quantity, never by geometry, so a
physical scan stays one scan even where the measurement resolves only part of it
— and two instruments in one folder never merge. When the members disagree the
group prints a `geometry:` line tallying them
(`TF measured on 12 of 21 runs; 9 unresolved`), which is itself a finding: a
transverse-field scan through a magnetic transition resolves above it and not
below.

`scans` is the experiment's structure. Work out from it:

- How many distinct scans there are. Two scans (say a ZF temperature scan and
  a TF one) are **two separate analyses**; do not merge them.
- What each scan is *for*. A temperature scan at a fixed field follows the
  sample through a transition. A **field scan at a fixed temperature over a
  wide range is a decoupling measurement** — a longitudinal field, stepped to
  recover polarisation — not a temperature scan and not a TF measurement.
- Whether more than one instrument is present. The header line names every
  instrument found (`— EMU, MUSR`). Run numbers from different instruments are
  different campaigns even in one folder: analyse them separately and say so.
  `asymmetry survey <folder> --json` carries `instrument` per run.
- Whether the run numbers run in temperature order. Very often they do not —
  a coarse pass followed by infill points. Never assume run number tracks
  temperature; read the temperature column.
- Which runs are *not* part of any scan (detector tests, a lone reference run,
  an above-Tc run). Mention them; do not analyse them as scan members.

**When the survey prints a `TEMPERATURE:` line**, the listed runs were not at
their setpoint — a cryostat still cooling, a block of runs at the wrong
temperature, or a sensor offset. It lists them in blocks with each block's
offset (T log − T/K), and you decide which to trust, block by block, and say
why. The logged sample temperature is the default — it is the measured one;
the setpoint needs a reason (a logged value the sample could not have had).
Whichever you decide for a block holds everywhere those runs are used: a
sensor you judged faulty for one series cannot supply an argument against a
comparison in another. A block sitting at a different temperature from its neighbours is a
different measurement, not a faulty thermometer to ignore. A steady offset the
sample could not have had — a liquid logged above its boiling point, a
cryostat base temperature below what the setpoint allows — points to the
sensor, and then the setpoint is the better axis. The data can settle it: a
signal missing where the setpoint predicts one (a line that should be there
below a transition) is evidence that the logged temperature is the real one. The `scans` block still groups them by
setpoint, so take its temperature scans as provisional: order those series by
`sample_temperature_logged`, split off runs that sit far from the rest, and
quote logged temperatures in the summary.

**Check temperature provenance before interpreting a temperature scan.** Some
cryostat files keep the setpoint parked while the sample temperature changes;
then many distinct runs can appear at one nominal temperature. Several runs at
exactly the same setpoint inside an otherwise recognisable temperature scan are
by themselves enough to trigger this check — do not describe them as repeated
measurements at that temperature. The survey's `T log/K` column is the logged
sample temperature (`sample_temperature_logged` in `--json`) where the file
records one; `T/K` is the setpoint. Call the setpoint a setpoint, order the
series by the logged value (`--order sample_temperature_logged`) when the two
disagree, and do not claim that run number is a temperature proxy. PSI `.bin`
files carry no logged column yet; for those, say that the trend is against the
setpoint rather than constructing a false precision.

`asymmetry info <file>` prints one file's metadata if you need to check a
single file directly.

### Step 2 — decide alpha, and never guess it

Alpha is the forward/backward detector balance. It is measured on a **weak
transverse-field** run, where the precession is undamped and the balance shows
up cleanly.

Keep the detector calibration provenance complete. Alpha is only one part:
also state which forward/backward detector grouping or instrument profile was
used, and whether the same calibration run established or validated it. If the
loader supplied a fixed instrument grouping rather than fitting groups from the
run, say that instead; do not silently make the grouping source disappear from
the report.

```bash
asymmetry alpha <folder> --run <calibration run>
```

(`alpha` takes no `--workdir` and no `--plot`.)

The survey's candidate list has two sources, marked in the block it prints:

- `[measured]` — the spectrum was seen precessing at the Larmor frequency of its
  recorded field, with the SNR quoted. This is the strong evidence, and it finds
  candidates the file's own metadata hides: a TF scan above a magnetic
  transition is listed run by run, and so is a 100 G run in a file recording no
  field state at all.
- `[metadata]` — nothing could be measured (no recorded field, or a Larmor
  frequency above Nyquist) and the file's own transverse-field evidence is all
  there is.

Then:

- **The survey prints an `ALPHA STEP`** → alpha changed partway through the
  folder (a sample change, a moved detector, a second instrument), and **no
  single run calibrates it**. Each candidate line carries its own measured
  alpha. Split the runs into blocks at each step and reduce every block with
  `--alpha-from` a calibration run *inside that block*; never carry the
  `(best)` run's alpha across a step. A block with no candidate of its own is
  reduced with the neighbouring block's alpha only if you say so and why. An
  asymmetry that collapses or turns negative in one block is the signature of
  the wrong alpha, not of physics — check that before interpreting it.
- **The survey lists calibration candidates and no step** → use one and report
  the value. When several are listed, prefer the one the survey marks
  `(best)` — the strongest measured precession — unless it is a member of the
  scan you are about to analyse and a separate run is available. A dedicated run at 20 G is
  the calibration; twelve runs of a 100 G paramagnetic scan are candidates
  *because* the sample is paramagnetic there, and calibrating on one of them is
  legitimate — **say which run alpha came from**.
- **No candidates at all** → nothing in this folder precesses at its own
  applied field, so there is nothing to measure alpha on. Reduce with the
  default alpha 1.0 and state in the summary that alpha was assumed to be 1.0
  and was not measured. Silently defaulting is a failure; say it.

`asymmetry alpha <folder> --run N` prints its own `precession` line — "at the
Larmor frequency: yes/no" in words — so you can see for any run, candidate or
not, whether it is the kind of run alpha can be measured on.

### Step 3 — reduce the scan to asymmetry

```bash
asymmetry reduce <folder> --runs 102-107 --alpha-from 101 --deadtime from_file --plot
```

- `--runs` takes ranges and commas: `102-107`, `102-105,107`.
- `--alpha-from <run>` measures alpha on that run and uses it; `--alpha <x>`
  sets it directly. Without either, alpha is 1.0.
- `--deadtime from_file` on ISIS data **whenever the survey's `dt` column says
  `yes`** — the files carry per-detector deadtime values and the GUI's fresh
  default (off) leaves them unused. Say in the summary that you applied the
  file's deadtime values.
- `--plot` writes `asymmetry-work/plots/reduced-<run>.png`. Read a few of
  them with the Read tool — the lowest-temperature, the highest, and one in the
  middle. That is how you learn whether there is an oscillation, a Kubo–Toyabe
  dip, or featureless relaxation, before any model is chosen.
- `reduce --tmin/--tmax` **cut the stored reduction** every later command
  reads — the wizard and every fit then see only that window. To zoom a plot,
  use `reduce --plot --plot-tmax 2`, which leaves the record whole. To restrict
  one screen or fit, pass `--tmin`/`--tmax` to `wizard`, `recipe`, `fit` or
  `fit-series` instead (a wizard window is kept in the recipe it writes).
  `--rebin k` merges bins.
- `--period red`, `--period green` or `--period N` selects one acquisition
  period before calibration or reduction. For ISIS photo-μSR files the usual
  convention is red/light-ON and green/light-OFF, but confirm that against the
  experiment notes and the spectra. The selected period is recorded in the
  cache digest. Because reduced files are keyed by source run number, use
  separate work directories if you need two periods of the same run cached at
  once.

For a two-period photo-μSR folder, close the loop rather than merely proving
that both periods load:

1. Identify the weak-TF detector-calibration run separately from the science
   sequence. It is for alpha, not a science point.
2. Reduce the same representative science run as red and green in **different
   work directories**. Confirm ON/OFF from the experiment metadata or logbook
   and from which trace gains the photo-induced relaxation; do not assume the
   colour convention alone.
3. For the standard carrier-rate screen, fit the dark/OFF period first with a
   pure exponential and free amplitude. Fit the light/ON period over the early
   window with that amplitude fixed. **Do not finalise the report without
   stating the fitted time window and fit quality.**
4. If titles or experimental context split the remaining runs into an injected-
   density calibration and a laser-delay scan, name both roles in the summary.
   `--order run` can show that the rate changes, but run number is not density
   or delay: do not turn it into a carrier lifetime or calibration exponent
   unless the required x values and trend fit were actually available.

Reduce only the runs of the scan you are analysing. Re-running `reduce` is
cheap (results are cached on a digest and reused).

### Step 3a — confirm the geometry of every non-zero-field scan

Do this before screening anything, every time, and report the answer. ISIS
files stamp `field_direction` unreliably: a zero-field run in a scan that opened
with a weak-TF calibration keeps the `TF` stamp, a longitudinal decoupling field
is routinely stamped `TF` too, and EMU files from 2024 record no field state at
all.

**Start from the survey's own measurement** (step 1's `prec` column), which is
evidence, not a stamp:

- A run at **0 G is ZF**, whatever the file says.
- `geom TF*` with `prec larmor` → **transverse, measured**. Take it.
- `geom -` with `prec none` at a non-zero field → the file's stamp was
  **refuted**: the muon is not precessing in the applied field. That is either
  an **LF** measurement — the field is along the beam — or a TF run whose line is
  not resolvable (a broad field distribution, or a frequency too fast for the
  pulse). **A fixed-temperature field scan of such runs is LF decoupling**, and
  this is how you confirm it; a single such run in a temperature scan wants the
  PNG looked at before you call it. Pass `--geometry LF` once you have decided.
- `prec other@<MHz>` → the muon precesses in a field that is not the applied
  one (see the table above for what the frequency can mean). That says nothing
  about the applied field's direction, so `geom` falls back to the file; judge
  it from the scan the run belongs to and the reduced PNG.
- `prec -` → nothing was measurable. Judge it from the PNG as below.

**Confirm on the reduced PNG** — always for `other` and `-`, and as a sanity
check otherwise. A **transverse** field precesses the muon at γ_μ/2π × B —
13.55 kHz/G, so 20 G ≈ 0.27 MHz, 110 G ≈ 1.5 MHz, 400 G ≈ 5.4 MHz — a plainly
visible oscillation filling the early-time window. No oscillation at that
frequency means the field is longitudinal. (Use `reduce --plot --plot-tmax 2` to zoom
the early window if the full range is too compressed to judge.)

Then pass the right `--geometry ZF|TF|LF` to `wizard` for every run you screen,
and say in the summary what each scan's geometry is and how you established it.
Calling a decoupling LF series "TF" misreads the whole experiment even when the
fitted model is right.

### Step 3b — decide what the system is, then scope everything to it

Before screening anything, write down in one line **what the sample is and what
the experiment measures**. The evidence is already in hand: the sample and
title text and the notes in the survey, any logbook or README in the folder
(read it), the field and geometry of each scan, and the run structure. Then
take the wizard's scope and the trend law from that class, not from `auto`.
`auto` is for a sample you genuinely cannot place: it expands a family only
where its spectral search finds support, which on real data means repeated
re-screens — and it will happily offer a vortex lattice to a ferromagnet.

| The system | What gives it away | Screen with | Trend law (Step 6a) |
|---|---|---|---|
| Magnet ordering in zero field | ZF temperature scan with `prec other@<MHz>` on the cold runs (spontaneous precession) | `--geometry ZF --scope zf-static-magnetism` | `OrderParameter` on the frequency below the transition; `CriticalDivergence` on a rate diverging towards it |
| Spin glass or frozen moments | no spontaneous line; the rate rises and stretches on cooling, then A(0) collapses at the freezing temperature (the fast-relaxing fraction leaves the resolvable window) | `--scope zf-static-magnetism` (ZF) or `lf-dynamics` (LF) | `CriticalDivergence` on the rate from above the freezing; place T_g where the rate peaks or A(0) collapses, not where the wizard stops finding structure |
| Fluctuating moments decoupled by a field | a **field** scan at one temperature whose runs do not precess at their field (`prec none` where measurable, `-` above Nyquist) — longitudinal decoupling, whatever zero field showed | `--geometry LF --scope lf-dynamics`; a **single** exponential rate λ | `Redfield` on λ(B), over the field range one process dominates |
| Nuclear dipolar fields, muon or ion hopping | a dense-nucleus compound; Kubo–Toyabe dip in ZF; in TF a Gaussian envelope that turns exponential on warming (motional narrowing) | ZF/LF: `zf-static-magnetism` / `lf-dynamics`, Gaussian KT (see decision rules). TF: fit the series with one Gaussian or exponential envelope; `fit-series` weighs the other shape on every run (the `envelope` column) — report the shape against temperature | `Arrhenius` on the hop rate `nu` over the range where it rises; state any low-temperature upturn separately |
| Quadrupolar level crossing | an LF scan over a narrow field range (tens of gauss) at one low temperature, in a compound with quadrupolar nuclei (Cu, Al, Nb …); a dip in the integral asymmetry, or fits whose χ²ᵣ spikes, at particular fields | `integral-scan` over the field range (Step 5b), not a decoupling analysis | the scan's resonance fit (`GaussianLCR`/`LorentzianLCR` + background) |
| Type-II superconductor | TF scan through Tc, `prec larmor`, line broadening on cooling | `--geometry TF --scope tf-superconductor` | an `SC_*` gap model on σ(T) |
| Type-I superconductor, intermediate state | a pure elemental superconductor (Sn, Pb, In, Al …) in a field **below H_c**, often LF on a tilted foil; `prec none` at the applied field | `--geometry LF --scope lf-dynamics --include Oscillatory`, and `fourier` to find the line — muons in the normal domains precess at γ_μ·H_c whatever field is applied | `OrderParameter` with `--fix alpha=2 --fix beta=1` (H_c(0)[1−(T/T_c)²]) on the frequency, against the logged temperature |
| Fluoride | fluorine in the sample name | `--scope fluoride-fmuf` | — |
| Muonium chemistry in a weak TF | water, solutions, gases, a few gauss TF paired with ~100 G diamagnetic runs; samples differing by concentration in the titles | not the wizard — see "Weak-TF muonium" in section 5 and write the recipe with `asymmetry recipe` | `Linear` on λ_Mu against the concentration you supply with `--order concentration --x …` |
| Radicals at high field, level crossings | kilogauss TF lines at hundreds of MHz; ALC field scans | `--scope muonium-radical`; `fourier`; `integral-scan` | the scan's own resonance fit |

Say in the summary which class you decided and on what evidence. If the
evidence contradicts it once you look at the spectra, change your mind and say
that too.

### Step 4 — screen one run with the wizard

```bash
asymmetry wizard <folder> --run 102 --geometry ZF --plot
```

The wizard fits a scoped set of candidate models to one reduced run and ranks
them. Screen **the run whose spectrum shows the effect you are measuring most
clearly** — not every run, and not reflexively the coldest one.

Pick it from the `reduce` table and the reduced PNGs, not from the run list.
For a scan, screen a run from inside the range the question is about — the
middle of a field scan, not its zero-field end, whose physics (static fields)
is the one the field removes:

- The clearest run for a **relaxation** effect (a glass freezing, a dynamic
  rate) is where the relaxation is *fastest but still resolved* — usually the
  cold end, and usually not the run where A(0) has collapsed.
- **A scan that crosses a transition needs a screen on each side.** An
  ordered magnet below T_c and the same sample above it are described by
  different models (a precession below, a paramagnetic relaxation above); one
  recipe chained through both fits neither. Screen one run on each side, fit
  the two sides as separate series with their own recipes, and report where
  the ordered-state model stops fitting. Split the scan at the last run the
  ordered model fits unflagged: no run belongs to both series, because a
  relaxation fitted to a signal that still precesses is not a paramagnetic
  rate. Fit the paramagnetic side with a **relaxation-only** recipe (no
  oscillating term) and report its rate λ(T) — flat or rising towards T_c —
  as the dynamics observable; an oscillating recipe carried above T_c comes
  back `frequency_unresolved` and gives no rate at all. Never conclude that an ordered-state
  signal is unresolvable from a model that was screened above the transition:
  screen the coldest run and one just below the transition, and look at their
  reduced PNGs and `fourier` spectra before saying so.
- The clearest run for an **oscillation** is where the precession is slow
  enough to resolve. At a pulsed source (ISIS) a large internal field precesses
  far too fast to see, so a magnet deep in its ordered state shows no
  oscillation at all — only a collapse of A(0) in the `reduce` table as the
  fast-relaxing fraction disappears inside the pulse. **Screen just below the
  transition**, where the order parameter is small and the frequency is low
  enough to fit. A falling A(0) on cooling is loss of *resolvable* signal, not
  loss of magnetism; say so in those terms.

Always pass the `--geometry` you established in step 3a. Without it the wizard
falls back to the survey's geometry (the header line says `from survey`, `from
user`, `from field` or `from file`), which is right whenever `prec` settled it
and gives you *nothing* exactly where you had to reason — a refuted run reads
`geom -`, so the wizard is left unscoped unless you pass `--geometry LF`
yourself. The wizard scopes its candidate families by geometry, so screening a
decoupling LF run as `TF` puts precession models in front of it and nothing
else.

Read from the output:

- **the recommendation** (model key and title) and the **ranked table** of
  candidates with AICc, reduced χ² and parameter count;
- **`Spectral lines`** — every line the wizard's spectral search detected,
  with its SNR — and **`Recommended fit`**, the fitted values. When a
  detected line is not in the recommendation, the wizard also writes recipe
  `line-<run>`: the recommendation **plus** that line, its amplitude started
  small. Fit it next (`asymmetry fit … --recipe line-<run>`); a line amplitude
  several times its error is the line, measured. Do not replace the
  recommendation with a bare oscillation — a weak line sits on the relaxation,
  and without it the fit turns the relaxation into a spurious slow
  "frequency". A precession
  frequency here is a finding in its own right: if you later fit the scan with
  a different model, a frequency the wizard found on this run still has to be
  accounted for in the summary, never contradicted by an empty `fourier`
  table;
- **confidence**: `high` or `medium` are both fine to proceed on — medium is
  the normal outcome on real data and just means residual structure remains.
  **`low` or `none` means do not fit a series on this run**: screen a different
  run (one with more signal, or further from a transition), or conclude the
  data has no significant structure and say so;
- **verdict** and **caveat**, which tell you what is left unmodelled;
- the narrative: which physics families were expanded, what the spectral search
  found, and whether the best model actually beat the null baseline. If the
  recommended model is a `null_...` baseline or ties with one, there is no
  structure to fit — say so rather than trending a meaningless parameter.

`--scope <preset>` restricts the candidate families to the class you decided in
Step 3b: `zf-static-magnetism`, `tf-knight-precession`, `tf-superconductor`,
`lf-dynamics`, `fluoride-fmuf`, `muonium-radical`, `all`, or `auto` (the
default, for a sample you cannot place). `--include C,D` adds time-domain
components the preset leaves out — `Oscillatory` for a line in an LF run —
and `--exclude C,D` drops ones the physics rules out, e.g.
`--exclude VortexLattice,VortexLatticePowder` for a magnet in TF. An unknown
component name is refused with the full list.

When the candidate the physics calls for is in the ranked table but not the
recommendation, or the wizard has no template for it at all, **write the recipe
yourself** with `asymmetry recipe` (see "Writing a recipe" below) rather than
re-screening again. The same holds when a component you `--include`d was fitted
and then rejected — the narrative says its frequency sat at the resolution
floor, completed less than a cycle, or had no supporting spectral peak. That
means the wizard had no line to start the frequency from, not that the physics
is absent: write the recipe with a physically estimated starting frequency
(`--initial frequency=<MHz>`, from γ_μ/2π = 0.01355 MHz/G times the field you
expect the muon to see), fit it, and accept it only if the fitted amplitude is
several times its error and the frequency moves smoothly across the scan. A
starting value from a textbook is fine; quoting it as a result is not. One or two wizard calls per scan is the norm; more means
the class decision in Step 3b was skipped.

The scope note may say "sample name suggests fluorine": that is read from the
run's title or sample text (`CaF2`, `LiF`, `KTCNQF4`), and it promotes the
F–μ–F family. Trust it when the sample really is a fluoride; a fluoride
whose title does not name it still gets F–μ–F candidates from the spectral
search.

`wizard` writes `asymmetry-work/recipes/wizard-<run>.json` — the fit recipe,
the only contract between screening and fitting. `--plot` writes
`asymmetry-work/plots/wizard-<run>.png` (data, recommended curve, residuals).
Look at it.

The wizard's candidate search prints `AsymmetryScaleWarning` blocks on stderr.
They are benign noise from seeding trial models; ignore them.

### Step 5 — fit the scan as a chained series

```bash
asymmetry fit-series <folder> --runs 102-107 --recipe wizard-102 \
    --order temperature --start 102 --name zf-scan --plot
```

- `--recipe` takes a name in `asymmetry-work/recipes/` (no path, no `.json`)
  or a path.
- `--order temperature` or `--order field` — the quantity the scan varies, the
  axis of the trend. `--order sample_temperature_logged` when the logged sample
  temperature departs from the setpoint (see Step 1). When the scan varies
  something the files do not record — a concentration, a degrader foil count,
  a magnet current — name it and give every run's value:
  `--order concentration --x 101=0,102=0.25,103=0.5`. Take those values
  from the notes, logbook or titles and say where they came from. `--order run`
  only when nothing else applies.
- `--start <run>` is **the run you screened**. The series chains outward from
  it in both directions, so every fit warm-starts from a neighbour near the run
  the recipe actually describes. Starting from the cold end with a recipe
  fitted at the hot end loses the middle of the scan.
- `--fix NAME=VALUE` holds a parameter (repeatable). `--global P,Q` pins
  parameters at their recipe value for every run — it does **not** fit them
  jointly.
- `--plot` writes a PNG per run plus a trend PNG per free parameter.

The per-run table gives reduced χ², a verdict and quality flags:

| Flag | Meaning |
|---|---|
| `failed` | The minimiser did not converge. The row is not a result. |
| `large_rel_err` | A free parameter's σ/value is large — the data barely constrained it. |
| `bound_pinned` | A free parameter sat on a bound. |
| `spurious_reseeded` | The fit landed on the spurious branch (amplitude collapse or frequency jump) near a transition, whether or not a reseed rescued it. |
| `frequency_unresolved` | A fitted frequency completes under two cycles in the informative window: a relaxation masquerading as a line (typically a weak line fitted without the relaxation it sits on). Not a precession result. |
| `amplitude_exceeds_data` | The fitted amplitudes (backgrounds included) add up to several times the record's own asymmetry: two components cancelling to describe a signal the data do not hold. The fit's parameters are not physical. |

A `poor` χ² verdict is common on high-statistics ISIS data (the band is tight
with thousands of degrees of freedom) and is not by itself a reason to discard
a run; a flag is.

### Step 5a — fit a simultaneous field group when the physics requires it

Use `fit-global`, not `fit-series --global`, when several runs must constrain
one set of physical parameters. Reduce the runs, then get one recipe for the
group: `--recipe` takes either a path to a recipe file or the name of one
already in `recipes/`, so it is `wizard-<run>` from screening a member of the
group, or a name you wrote yourself by hand-editing that recipe into the model
the physics calls for (see "Hand-editing a recipe" below). Check it converges
on a single run before spending the group on it, then name every parameter
that should be fitted once across the group:

```bash
asymmetry wizard <folder> --run 51343 --geometry LF --scope lf-dynamics
# edit recipes/wizard-51343.json into recipes/dynamic-gkt.json if the family
# the wizard ranked first is not the one the physics calls for
asymmetry fit <folder> --run 51343 --recipe dynamic-gkt --fix B_L=10
asymmetry fit-global <folder> --runs 51341-51343 --recipe dynamic-gkt \
    --shared A_1,Delta,nu,A_bg --field-param B_L --name ionic-160K --plot
```

`--field-param B_L` sets `B_L` from each run's recorded field and fixes it for
that run. Other parameters not listed in `--shared` remain run-local. Check the
model's actual parameter names in the recipe; never copy the example names
blindly. A decoupling **triplet** — a few fields at one temperature, repeated
across temperatures — normally shares the dynamic relaxation parameters and
physically common amplitudes, while the applied LF differs. A **field scan**
over many fields at one temperature asks the opposite question — how the rate
changes with field — so it is not a `fit-global`: fit one rate per run with
`fit-series --order field` and then `trend --model Redfield`.
`fit-global` takes the same `--order`/`--x` as `fit-series` (default `run`),
and its table and stored trend carry every run-local parameter along that
axis — so `asymmetry trend <folder> --series <name>` reads it, and
`trend --model` can fit it: a muonium relaxation rate fitted per sample with a
shared amplitude, ordered by `--order concentration --x …`, gives the rate
constant from `trend --model Linear`. Repeat `fit-global` for each temperature
group. There is not yet a batch command that trends a sequence of global fits,
so quote each stored group's shared values and uncertainties directly rather
than presenting independent fits as a coupled analysis.

### Step 5b — build and fit an integral-asymmetry field scan

An ALC/QLCR dataset is integrated across time and fitted against applied field;
do not send its individual high-field runs through the time-domain wizard:

```bash
asymmetry integral-scan <folder> --runs 19489-19519 --alpha-from 19485 \
    --tmin 1 --tmax 10 --model "LorentzianLCR + Cubic" \
    --name tcnq-10K --plot
```

The JSON and stored `scans/<name>.json` contain every point, exclusion and fit
uncertainty. Use `--initial NAME=VALUE` or `--fix NAME=VALUE` when the automatic
resonance seed needs physical guidance. If the background should be determined
only away from the resonance, use, for example,
`--baseline Cubic --baseline-regions 2000:2600,4500:5000 --model LorentzianLCR`.
Only call a feature an ALC resonance when the fitted peak, its uncertainty and
the plotted field dependence support that interpretation.

Report the quantities `integral-scan` actually emits — the resonance field,
width, amplitude, uncertainties and fit quality. A field can *constrain* a
hyperfine coupling physically, but do not convert it into a coupling constant
with a paper formula, shell arithmetic or a calculator and then present that
derived value as Asymmetry output. If the command did not print the coupling,
leave it qualitative.

### Step 5c — inspect a reduced run in frequency space

```bash
asymmetry fourier <folder> --run 20721 --window none --fmax 5 --plot
```

`fourier` writes `spectra/<name>.npz`, a JSON provenance file and, with
`--plot`, a spectrum PNG. Its table reports detected frequency, amplitude,
width and SNR. Zero padding interpolates the display but does not improve the
quoted resolution, which comes from the selected time window. Compare peaks
with the time-domain fit and the expected muon Larmor frequency; treat peaks
closer than the reported resolution as unresolved. This command is an FFT,
not MaxEnt.

Peak detection is deliberately conservative, so use this evidence ladder:

1. **Detected line** — it appears in the peak table. Quote its tabulated
   frequency, width and SNR.
2. **Visual candidate** — the PNG contains a local maximum or shoulder that is
   absent or much weaker in a matched reference, but it did not pass the peak
   threshold. Read an approximate frequency from the plotted axis, label it
   explicitly as visual-only, and do not attach a fitted width or SNR to it.
   An empty peak table means *not detected*, not *featureless*. When nothing
   is detected, `fourier` lists the band's **strongest maxima** with their
   height over the noise floor: they are candidates of this kind. When the
   physics predicts a weak line, fit the time domain with a recipe started at
   the candidate frequency (`recipe --initial frequency=…`) and let the fitted
   amplitude and its error decide.
3. **Pattern-level interpretation** — several detected and/or visual features
   may form a recognisable physical pattern. State what the pattern is
   consistent with, while keeping the component frequencies and derived
   quantities provisional. Do not suppress a supported qualitative assignment
   merely because the same FFT cannot make it a precision measurement.

**Reconcile a missing line with what the session already knows.** By the time
you transform a run you have usually measured its precession twice over, so an
empty peak table is a claim to check, not a result to report. Before writing
that a line is weak, absent or "broadband", hold the spectrum against (a) the
`survey` precession column for that same run — a run reported as precessing at
the Larmor frequency *has* a line — and (b) the run's own time-domain fit: a
fitted oscillating amplitude of order the full asymmetry, damped at σ (or λ),
is a line of width of order σ/2π in MHz. Broadened past the conservative
detector, that line is still there, and the honest sentence is that it is too
broad to be tabulated at this resolution — never that the spectrum has no
structure. When those sources disagree with your reading of the plot, the
transform or the displayed band is at fault, not the sample: put `--fmin`
and `--fmax` a few linewidths either side of the expected frequency and look
again. A full-record 0–10 MHz view of a damped line spreads it over a handful
of bins among noise maxima of similar height, which is how a present line gets
written up as absent.

Note also that `fourier --run N --plot` writes `plots/run-N.png` every time,
so a second transform of the same run replaces the first PNG. Read each plot
before re-running, and say in the summary which window the spectrum you are
describing actually used.

Always compare like with like: the science and reference spectra need the same
reduction, transform window, time range and displayed frequency range. Features
that change with sample condition while the matched reference remains a simple
line are more informative than small lobes common to both plots. Conversely,
do not promote arbitrary noise or shared window sidelobes into components. A
useful report can therefore say both “the detector found no side peaks” and
“the image has candidate shoulders near ...”; those statements are not in
conflict.

Keep **detection** and **visual separation** distinct. The peak threshold asks
whether the automated detector accepts a component; the quoted FFT resolution
asks whether two plotted features can be separated at all. On the magnitude
trace, local maxima with an intervening minimum and spacing of several
resolution elements are visually distinct candidates even when the peak table
is empty. If they are absent from the identically processed reference, list
their approximate axis coordinates and call them visual-only. Do not collapse
such a spectrum to “broadening” solely because no row passed the detector. By
contrast, maxima less than about one resolution element apart are unresolved
and must not be counted as separate components.

Before finalising a Fourier-led summary, make a compact spectral inventory:

- identify the central or reference line and compare it with the expected
  frequency and the reported resolution; say when the mismatch is comparable
  to, smaller than, or larger than one resolution element;
- trace the magnitude curve across the displayed band and give an individual
  approximate x-coordinate for each prominent visual candidate relevant to the
  interpretation — a broad frequency range is not a substitute for listing the
  maxima or shoulders inside it;
- mark every listed feature as **table-detected** or **visual-only**, then say
  whether the candidate offsets are roughly symmetric at the available
  resolution;
- name the physical line pattern that those observations support, using
  “consistent with” when the evidence is qualitative rather than withholding
  the pattern name altogether;
- name the concrete spectrum JSON, NPZ and PNG products that support the
  report.

This inventory is a final-report gate: a statement such as “structured across
the band” or “shoulders are present” is incomplete when the axis permits
approximate coordinates to be read. Approximate visual coordinates are not
precision measurements; reporting them is what lets the reader distinguish a
candidate line pattern from generic noise. Likewise, give the exact artifact
basenames or paths rather than only naming their parent directory.

For a suspected central-plus-satellites pattern, do not replace the side
features with two broad frequency ranges. Anchor the central line, choose the
most defensible local maximum or shoulder on each side, report one approximate
coordinate per candidate, and compare their offsets from the anchor. If the
offsets agree to about the FFT resolution and the matched reference lacks the
side features, call it a visual three-feature or triplet-like pattern. When the
sample context supplies a standard physical name for that pattern, use the
name explicitly with “consistent with” (for example, a central diamagnetic
line with two roughly symmetric shallow-muonium satellites is “consistent with
a shallow-muonium hyperfine triplet”). Do not dilute that supported qualitative
assignment to only “muonium-related structure”; retain the visual-only and
non-precision caveats alongside the name.

If the plotted range is too broad to read those coordinates, regenerate a
tighter view with the same transform settings. If a coordinate still cannot be
read responsibly, state that limitation; do not turn it into a claim that the
feature is absent.

If a symmetric window warns that it removed early-time power, use `none` with
a physical `--tmax` crop or a `lorentzian` window with `--filter-tau` matched to
the damping rate.

Choose a **science run**, never the weak-TF alpha-calibration run, for the
representative spectrum. For a temperature series, start with the coldest
high-statistics science run where splitting or broadening should be clearest;
use the calibration run only to obtain alpha. When several runs share that
coldest temperature, inspect their event counts (from the survey record or
the human table's `events` column) and choose the highest-statistics one rather
than the first run number. The JSON field is `total_events`; it is the gross
count across the default period's raw detector histograms.
If neither command exposes an event count, use acquisition duration as the
statistics proxy within otherwise comparable runs. Do not substitute the
largest precession SNR: that can favour a calibration run or a warmer,
narrower line over the cold science spectrum the experiment calls for.

Treat that choice as a provenance gate, not a quick guess. If the nominal
temperature repeats across a chronological block, do **not** discard the
whole block or jump to the first later run with a different setpoint. Such a
block may be the cooldown itself. Inspect `info --json` for at least the first
and last members and for the highest-statistics member, record the logged
sample temperature when present, and rank the plausible cold-end runs by
physical temperature and then statistics. The representative FFT is not
chosen until that short candidate table is resolved. This rule applies to any
scan whose setpoint and logged sample condition can differ, not only to
temperature scans.
For a semiconductor/shallow-donor or muonium dataset, inspect the spectrum
around the Larmor line for a central line and symmetric satellites. A full
0–5 MHz plot can compress a sub-MHz multiplet until it looks like generic
broadening. Make a second plot over a tight, physically motivated window around
the Larmor frequency (for a 100 G CdS-like scan, approximately
`--fmin 1.15 --fmax 1.65`) and compare the cold high-statistics run with a warm
single-line run using the same reduction and FFT settings. If the conservative
peak table remains empty but the zoomed cold PNG shows a central maximum with
roughly symmetric side maxima or shoulders that are absent in the warm PNG,
report their approximate plotted frequencies and describe the three-feature
pattern as consistent with a shallow-muonium hyperfine triplet. This is a
pattern-level assignment, not a claim that every component was algorithmically
detected or sufficiently resolved for a precision splitting. Do not erase the
observed candidates by replacing the multiplet with only a generic “damping”
or “broadening” explanation; report the observation and its limitation
together. The warm line may supply the central-frequency anchor when the cold
peak table is empty: one cold visual candidate on each side of that anchor is
still a triplet-like pattern to report, provided the matched plots support it.

Keep the full usable time record for this first unwindowed comparison: the
transform duration, not zero padding, sets the physical frequency resolution.
Do not crop to the visually interesting first few microseconds when the aim is
to resolve lines separated by only about 0.1 MHz. An 8 μs transform has roughly
0.125 MHz resolution and cannot establish such a multiplet; a roughly 31 μs
record gives about 0.032 MHz. For the 100 G CdS case, the useful reconnaissance
command shape is therefore `--window none --fmin 1.15 --fmax 1.65 --plot` with
the default full time range, followed by the same view on a warm reference. A
short physical crop is appropriate only when late-time data are unusable, and
then the coarser reported resolution limits what may be called resolved.

In a CdS-like shallow-donor scan, the FFT is reconnaissance rather than a
precision hyperfine measurement. The physical pattern is a warm diamagnetic
line near the applied-field Larmor frequency and cold neutral-muonium side
structure growing around it. Frequency differences read from shoulders are
only estimates at the quoted FFT resolution. A defensible quantitative result
requires a raw-count MaxEnt reconstruction or a time-domain model whose side
frequencies are tied to `f_centre ± delta`; neither may be imitated with
shell arithmetic or an unconstrained multi-component fit when the CLI does not
provide it.

For a temperature-dependent multi-line fit, component amplitudes cease to be
physical fractions when the fitted separation collapses or is unresolved. Do
not carry those amplitudes into an Arrhenius or population trend merely because
the overall chi-squared is acceptable: coincident components can exchange
amplitude without changing the curve. Mark the point unresolved and require an
identifiability check before deriving a neutral fraction or activation energy.

If the folder's scientific aim is frequency-domain structure, the FFT is the
primary result: the final summary must call it an **FFT, not MaxEnt**, and
report its window, resolution, detected peaks and **visual inspection of the
PNG**. Use the available image-viewing tool; the conservative peak table alone
cannot establish that weak shoulders are absent. When choosing or attaching a
small set of plots for review, nominate the primary science spectrum first and
its identically processed reference second; secondary runs, fit diagnostics
and trend plots come later. Include the exact science run used for the headline
conclusion even when its peak table is empty. A generic time-domain series may
complement the FFT, but must not displace it.

Reconcile the image list **after** the provenance and run-selection gates. If
metadata changes the representative run, replace every stale preliminary plot
in the review list. Keep explicit track of the final science run and its FFT
PNG; before emitting an image manifest, verify that its first entry is that
exact PNG and its second is the matched reference FFT. With a four-image
budget, use the remaining slots for the corresponding science/reference
time-domain plots. Never spend those slots on an exploratory run while omitting
the spectrum that supports the headline conclusion.

### Step 5d — finish every specialised workflow you start

Period selection, `integral-scan`, `fit-global` and `fourier` are not side
probes. If you invoke one because it matches the experiment, its result and
provenance belong in the final summary. Before writing, check that each
specialised command you ran appears under Results or Files and that you did not
silently fall back to a generic `fit-series` narrative.

### Step 6 — read the trend

```bash
asymmetry trend <folder> --series zf-scan --plot
```

This is the table your Results section comes from: the scan variable and every
fitted parameter with its uncertainty, per run, with the flags carried
through. `--csv <path>` also writes it as CSV. `--plot` writes one PNG per
parameter, with flagged points drawn distinctly.

When the series fits a frequency, the table carries `survey_line_mhz`: the line
the survey measured in each run. A fitted frequency far from it, or a line
fitted where the survey found none (`-`), is the fit locking onto noise or an
artefact — trust the survey's line and say which runs disagree.

When the model has one Gaussian or exponential envelope, the `envelope` column
says which shape each run prefers (`either` when they fit alike). A change of
shape along the scan is a result: report it with the runs on each side.

**Look at the trend PNGs with the Read tool before writing anything.** A trend
that is flat, that jumps, or whose scatter swamps the error bars is telling you
something the table alone will not.

### Step 6a — fit the trend with a physical law

When the experiment's question is a number that a trend encodes — a
transition temperature, a critical exponent, an activation energy, a
correlation time, a gap — fit the law to the trend column instead of reading
it off the plot or computing it by hand:

```bash
asymmetry trend <folder> --series zf-scan --model OrderParameter \
    --param frequency --fix alpha=1 --xmax 69 --exclude 2958 --plot
```

- `--model` takes a parameter-vs-x expression: `OrderParameter` (a precession
  frequency or internal field below the transition), `CriticalDivergence` (a
  rate or width diverging towards a transition, fitted from one side),
  `Arrhenius` (a hop or fluctuation rate against temperature), `Redfield` (a
  relaxation rate against longitudinal field; hold its exponent with
  `--fix m=2` for the textbook form), `SC_SWave` and the other `SC_*` gap
  models (a superconducting σ against temperature), `Linear` (a rate against
  concentration), and sums such as `Redfield + Constant`. The class you
  decided in Step 3b picks the law; say which you used and why. A transition
  temperature or exponent is measured against the trend's x — say whether that
  was the setpoint or the logged sample temperature.
- `--param` is the one quantity the law is written for: a single relaxation
  rate for Redfield or Arrhenius, a single frequency for an order parameter.
  When the series model carries two components of that kind the command
  prints a `NOTE`. If they are two species (a muonium and a diamagnetic line)
  fit the one the law describes. If they are two rates splitting one
  relaxation between them because AICc preferred it, refit the series with a
  single-rate recipe first — the law describes the one rate.
- `--xmin`/`--xmax` set the fit range in the trend's x units. An order
  parameter is fitted **below** the transition, a Redfield law over the field
  range where one process dominates. Compare points measured under matched
  conditions (one temperature for a concentration series). State the range in
  the summary.
- `--fix NAME=VALUE` holds a law parameter (e.g. `alpha=1` for the simple
  power law); `--initial NAME=VALUE` moves a start value.
- **Every run with a value enters unless you exclude it.** The output lists the
  runs it left out and the *flagged runs it fitted*. Exclude a flagged run
  whose value is suspect — `failed`, `spurious_reseeded`, or `bound_pinned` on
  the parameter you are fitting — with `--exclude RUNS`, and say which you
  excluded and why. A `large_rel_err` run is weighted down by its own error bar
  and can usually stay.
- The fit is stored in `series/<name>.json` under `trend_fits`, and `--plot`
  draws the curve over the points it was fitted to. Read the PNG: a law that
  misses the points near the transition, or a parameter reported `at bound`,
  is not a result.

Quote the law's parameters with their uncertainties exactly as printed, and
its χ²ᵣ. When χ²ᵣ is well above 1 the output adds errors scaled by √χ²ᵣ:
quote those, and say the law describes the trend only approximately — a
converged fit with a poor χ²ᵣ is still the result, reported with its caveat,
never withheld. The
same rule as for integral scans applies: derive nothing further by hand (a
penetration depth from σ, an energy in meV from a gap in kelvin) and present
it as Asymmetry output.

### Step 7 — write the summary, audit its numbers, then send it

Template in section 6. Write the draft to `summary.md` in the project directory,
then run

```bash
asymmetry audit summary.md
```

Every command's printed output is logged in the work directory, and `audit`
lists each number in the draft that no command printed. Each one it lists is
almost always arithmetic on printed values — a percentage change, a ratio, a
difference of two columns, a unit conversion (MHz to gauss, relative to molar),
a significance in σ — or a value from memory. Remove it, quote the printed
value instead, or say the relation in words ("rises by several percent", "an
order of magnitude faster"). Re-run `audit` until it lists nothing you would
defend as printed. Do not mention the audit in the reply; it is a check on
your draft, not a finding.

**The user sees neither tool output nor files — only your final message.** So
your final message must *be* the summary: its full text, as audited, typed out
in the reply. Reading `summary.md` with a tool, pointing to the file, or
writing a shorter recap is not a reply — the first two show the user nothing,
and a recap drops results and brings back numbers the audit removed. A clean audit means each number
appears in some output, not that it is the right one — still quote values from
the command that produced them.

## 3. Decision rules

**AICc ranks the candidates; the physics picks the model.** The wizard's
recommendation is the best *statistical* fit to one run. When two candidates
score within a few AICc of each other — the table marks comparable ones with
`~` — the ranking has not decided anything, and you must. Choose the one the
system calls for, take it from the ranked table (hand-edit the recipe, see
below), and say in the summary which you chose and why:

- A **disordered or glassy magnet** — a spin glass, a frozen moment system —
  relaxes with a *distribution* of rates, not one rate. That is a **stretched
  exponential**, not a simple exponential, and the stretching exponent is a
  result worth reporting.
- A **vortex lattice** in a superconductor gives a **Gaussian** field
  distribution: fit the Gaussian width σ.
- **Static nuclear dipolar fields** give a **Kubo–Toyabe** shape with its
  characteristic dip and one-third recovery, not a plain decay. *Which* KT is
  a second physics choice the ranking cannot make for you. A **dense** array
  of nuclear moments — the ordinary case for a stoichiometric compound, where
  every muon site has many comparable neighbours (H, Li, F, Al, La, V, Nb,
  Cu …) — gives a **Gaussian** field distribution: `StaticGaussianKT`,
  `DynamicGaussianKT`, or `Keren` for a run in a longitudinal field. The
  **Lorentzian** KT describes *dilute*, randomly sited moments — a few percent
  of impurity or defect spins in an otherwise moment-free host — and its
  width `a_L` is not a Gaussian `Delta`. Motion (ionic hopping, diffusion,
  muon diffusion) is the fluctuation rate `nu` *on top of* the Gaussian
  width, not a reason to switch distributions. Taking `DynamicLorentzianKT`
  because AICc preferred it, in a compound whose nuclei are dense, reports a
  field distribution the sample does not have. Name the KT you used and why.
- A **fluoride** gives the F–μ–F three-spin beat.

Taking a simple exponential because AICc liked it by two points, when the
physics says stretched, is the commonest way this analysis goes wrong.

**Say what every amplitude means.** A fitted model's amplitudes are not
bookkeeping; they are the fractions of the muon ensemble doing different
things. In every summary, state for the chosen model:

- the **relaxing/oscillating amplitude** (`A_1`) — the signal from the part of
  the sample under study, and how it changes across the scan;
- the **`Constant` term** (`A_bg`) — the **non-relaxing fraction**: muons that
  stopped outside the sample (sample holder, cryostat, silver mask) or in a
  volume that is not participating. For a superconductor that is the
  **non-superconducting fraction**; for a magnet it is the unordered or
  background fraction. Say explicitly how you handled it — fitted freely, or
  fixed from a reference run — rather than fitting one relaxing term and
  ignoring where the rest of the asymmetry went.

**Many flagged runs.** If a large fraction of the series is flagged, the recipe
is wrong for part of the scan. In order of effort:

1. Read the per-run PNGs (`asymmetry-work/plots/<name>/<run>.png`) at the
   flagged temperatures. Compare them with the run you screened. What changed?
2. Try a narrower `--tmax`. Late-time noise drags a fit that the early-time
   structure would have constrained.
3. Take a different template from the wizard's ranked table. Either re-run
   `wizard --scope <preset>` to steer the family, or hand-edit the recipe (see
   below) to the template you want.
4. Fix a parameter from a reference run, the way an analyst would. The standard
   pattern: fit a **high**-temperature run where the sample is simple (say the
   stretching exponent is 1, or the sample is paramagnetic) and fix the full
   asymmetry A(0) from it; fit a **low**-temperature run and fix the flat
   background from it; then re-run the series with
   `--fix A_1=<value> --fix A_bg=<value>`. Every fixed value must come from a
   fit you actually ran, and the summary must say which run it came from.

   This is not only a rescue for flagged runs. Whenever the relaxing amplitude
   and the background **trade off against each other** at one end of a scan —
   the background drifting negative, the amplitude jumping, both errors
   ballooning while χ² stays fine — they are degenerate there, and fixing one
   of them from a run where the relaxation *is* well resolved is the fix.
   Noticing the degeneracy and reporting the numbers anyway is not.

**Writing a recipe.** When the model you want is not the wizard's
recommendation, build it from an expression:

```bash
asymmetry recipe <folder> --expression "Oscillatory * Exponential + Constant" \
    --name line --run <run> --initial frequency=1.9
```

`--run` seeds the amplitudes, background and applied field from that reduced
run — every amplitude starts at the run's whole early-time asymmetry, so for a
weak line beside a large background give both explicitly
(`--initial A_1=0.5 --initial A_bg=<the background level>`); `--initial NAME=VALUE` moves a start value, `--fix NAME=VALUE` holds one,
`--tmin`/`--tmax` set the window. The command prints **every parameter name**
— in a repeated-component expression they are numbered by component
(`Oscillatory * Exponential + Oscillatory * Exponential` has `A_1`,
`frequency_1`, `phase_1`, `Lambda_2`, `A_3`, `frequency_3`, `phase_3`,
`Lambda_4`) — so read them from its output rather than guessing. Then run
`asymmetry fit <folder> --run N --recipe <name>` on one run to check it
converges before spending a series on it. (A recipe is also small JSON in
`asymmetry-work/recipes/`, and editing it by hand still works.)

**A caveat is not a reason to withhold a result.** When the experiment asks
for a quantity — a rate constant, a transition temperature, a correlation
time — and the data allow a fit, do the fit and report it with its caveat (a
temperature offset between samples, a poor χ²ᵣ, a short range), rather than
declining because the comparison is imperfect. Declining is for a question the
data cannot answer at all.

**One negative run is not a negative folder.** When a feature the physics
predicts — a line, a dip, a step — is missing from one run, test the run where
it should be strongest before concluding it is absent: the least perturbed
sample (a deoxygenated blank, a pure reference), the coldest run, the field
where it is clearest. Read the survey's notes to find that run. A fit that
drives the expected component to zero on an unfavourable run says nothing
about the others.

**The wizard found nothing.** Confidence `low`/`none`, or the null baseline
winning, on the run you screened: screen a different run before concluding
anything. If several runs screen the same way, the honest finding is "no
significant structure in these spectra", and you should say so rather than
fitting a series to satisfy the request.

**Two scans.** Analyse each separately, end to end, with its own screening run,
its own recipe and its own series name. Report both. Do not put runs from two
geometries, two fields or two instruments into one `fit-series`.

**Flagged runs are never results.** Do not quote a flagged run's parameter as a
measurement. List the flagged runs, their flags, and what you think went wrong.

**TF superconductors.** The physically meaningful quantity is the Gaussian
width σ of the vortex-lattice field distribution, so prefer a Gaussian-envelope
template when the ranked table offers one, even if AICc prefers the exponential
— and say which scored better. Report σ(T) rising below Tc, the diamagnetic
shift of the precession frequency, **and** the non-superconducting fraction
(the `Constant` amplitude, and any second, weakly relaxing oscillating term the
ranked table offers): a σ(T) reported without saying what happened to the rest
of the asymmetry is an incomplete answer for a superconductor.

## 4. What things cost

| Command | Cost |
|---|---|
| `survey` | a few seconds for a whole folder |
| `alpha`, `reduce` | ~0.2 s per run |
| `integral-scan` | roughly the cost of reducing its runs, plus a quick scan fit |
| `fourier` | instant after reduction |
| `wizard` | 3–8 s per ISIS run; up to a few minutes for a long HIFI or PSI record, or with `--include` |
| `fit`, `fit-series`, `fit-global` | a few seconds for a scan or one coupled group |
| `trend` | instant (it reads stored results); `--model` a second or two |

So: screen **one or two** runs, not every run. Reduce and fit whole scans
freely — those are cheap.

**Let long commands finish.** Give a `wizard` or `fit-series` call several
minutes through your shell tool's own timeout setting (in Claude Code, the Bash
tool's `timeout` parameter, e.g. 600000 ms) — not a `timeout` command, which
macOS does not have — and do not pipe it through `tail` or `head` (you lose the
output if it is cut off). If a command is moved to the background,
wait for it to complete before doing anything that depends on it — and never
end your turn while one is still running: the analysis stops with it.

## 5. Physics to ask yourself

Match the question to the geometry. Background for each of these is in the
published workflow chapters; read the relevant one rather than guessing:

- <https://benhuddart.github.io/asymmetry/workflows/temperature_scan_magnetism.html>
- <https://benhuddart.github.io/asymmetry/workflows/fmuf_entangled_states.html>
- <https://benhuddart.github.io/asymmetry/workflows/dynamic_kt_copper.html>
- <https://benhuddart.github.io/asymmetry/workflows/lf_decoupling_dynamics.html>
- <https://benhuddart.github.io/asymmetry/workflows/superconductor_penetration_depth.html>
- <https://benhuddart.github.io/asymmetry/workflows/global_fit_ionic_motion.html>
- <https://benhuddart.github.io/asymmetry/workflows/alc_scan_tcnq.html>
- <https://benhuddart.github.io/asymmetry/workflows/photomusr_silicon_periods.html>
- <https://benhuddart.github.io/asymmetry/reference/fourier_analysis.html>
- <https://benhuddart.github.io/asymmetry/workflows/calibration_grouping_emu.html>

**Zero field.** Is there an oscillation? An oscillation in zero applied field
means static long-range magnetic order, and its frequency is proportional to
the ordered moment — the order parameter. Does it fall towards zero, and is it
lost, as the sample warms through a transition? Is there more than one
frequency (more than one muon site or more than one sublattice)? With no
oscillation: is the relaxation Gaussian-like with a recovering tail
(Kubo–Toyabe, static nuclear dipolar fields) or a simple decay? In a fluorine
compound, is it the F–μ–F three-spin beat pattern? Note that the F–μ–F family
parameterises the dipolar coupling as the muon–fluorine distance `r_muF` in
ångström (the coupling frequency ω_D ∝ r⁻³), so a constant `r_muF` across a
scan *is* a temperature-independent dipolar coupling — say so in those terms.

**Longitudinal field.** An LF is applied to decouple static fields, so what
relaxation survives is dynamic — even in a sample whose zero-field spectrum
looked static (a Kubo–Toyabe shape): the field removes the static part, and
the rate left over as the field rises is the fluctuating part. Does the rate rise on cooling towards a
freezing or glass transition? Is a stretched exponential needed (a distribution
of rates, as in a spin glass) rather than a single exponential? Does the
recovered asymmetry increase with field, as decoupling predicts? A **field scan
at fixed temperature** of a fluctuating system asks one quantitative question:
does the rate follow Redfield's law, λ(B) ∝ τ / (1 + γ_μ²B²τ²)? Fit a single
exponential rate per run and then `trend --model Redfield` on λ(B); its `D`
and `nu` (MHz) are the width of the fluctuating field and the fluctuation
rate. A change of slope in λ(B) away from the fitted law is a finding —
a field-induced change of state (a magnetisation plateau's edges, a
spin-flop) — worth pointing at.

**A superconductor that is not in a vortex state.** A type-I superconductor
(pure Sn, Pb, In, Al, Hg) in a field below its critical field H_c, with a
large demagnetising factor (a foil across the field), splits into normal and
superconducting domains, and the field inside the normal domains is H_c, not
the applied field. Muons stopping there precess at γ_μ·H_c = 13.55 kHz/G × H_c
— a line that is **not** at the applied field's Larmor frequency, so the
survey reports `prec none`, and one that is present in an LF geometry when
the foil is tilted. Its frequency falls to zero as the sample warms to T_c. The
line is small — a fraction of a percent against the background — so screen at
the field and temperature where it is clearest (the middle of the field range,
the coldest logged temperature), look with `fourier` over a few MHz, and when
the wizard cannot seed it, write the recipe (`Oscillatory * Exponential +
Constant`, `--initial frequency=<0.01355 × H_c estimate>`) and fit the scan from
that. The H_c(T) law is `OrderParameter` with `alpha=2`, `beta=1`.

**Weak-TF muonium.** In water, solutions and many insulators a fraction of the
muons form muonium (Mu). In a weak transverse field its triplet precesses at
1.394 MHz/G — 103 times the bare muon's 13.55 kHz/G — so a 2 G run shows a
line near 2.8 MHz from Mu and a diamagnetic line (27 kHz) that completes less
than a cycle in the record. The 100 G runs beside them are for the diamagnetic
fraction and for alpha. The survey reports the 2 G runs as `prec other` or
`none` (the Mu line is not the applied field's Larmor line), and a full-record
Fourier transform may show nothing because Mu relaxes quickly; look with
`fourier --tmax 4`. **Look first in the sample with the least scavenger** — the
run the notes call **deoxygenated** — where the Mu line lives longest; the
survey shows it as `other@` near 1.394 MHz/G × B. Untreated water is **not** a
blank: its dissolved O₂ is a scavenger. Untreated water
carries dissolved O₂, which relaxes Mu too fast to see; a concentrated solution
likewise. Absence of a line there says nothing about the blank. Reduce that
blank with the right alpha for its block (Step 2) before concluding anything. The wizard has no dependable template for this; write the
recipe:

```bash
asymmetry recipe <folder> --name mu --run <2 G run> \
    --expression "Oscillatory * Exponential + Oscillatory * Exponential" \
    --fix frequency_1=2.79 --fix frequency_3=0.0279 --initial Lambda_2=0.5
```

(frequency_1 = 1.394 MHz/G × B for Mu and frequency_3 = 0.01355 MHz/G × B for
the diamagnetic muon, at the run's recorded field B — here 2 G.) `Lambda_2` is the Mu
relaxation rate λ_Mu. In a reaction-kinetics experiment λ_Mu = λ₀ + k_Mu[x],
so fit λ_Mu for each sample at one temperature — `fit-global` with the Mu and
diamagnetic amplitudes and phases shared, ordered by
`--order concentration --x <run>=<value>,…` with the concentrations read from
the titles or notes — and take k_Mu from `trend --model Linear`. Concentrations
given as "quarter", "half", "full" or "0.25" are **relative**: keep k_Mu per
unit of that relative concentration, write the values as the titles do (no "M"
after them), and never invent a molarity. Faster Mu relaxation in untreated
than in deoxygenated water is dissolved O₂.

**Transverse field.** The precession frequency gives the local field at the
muon: a shift relative to the applied field is a Knight shift. The relaxation
is the width of the field distribution. Below a superconducting Tc, the vortex
lattice broadens it: a Gaussian σ(T) rising below Tc, related to the London
penetration depth. Is the line shape Gaussian or exponential, and does it
change with temperature (a nuclear-dipolar Gaussian narrowing to a Lorentzian
as the muon starts to hop)?

## 6. The summary template

Use these headings. Fill only from command output.

> **Every number in the summary must appear in a command's output.** A
> temperature, field, frequency, rate, exponent, run number, alpha, χ² or count
> that no command printed does not go in. Textbook or literature values may be
> *discussed* — clearly attributed as such ("the textbook value for bulk nickel
> is far above the range scanned here") — never presented as a result of this
> session. Nor may a unit the data does not give be attached to a number: a
> concentration written "0.25" or "quarter" in a title is not 0.25 M.

This also excludes numbers you calculated in Python, PowerShell, a spreadsheet
or by applying a literature conversion to a stored JSON value — and numbers you
worked out in your head: a significance in σ, a percentage change, a ratio or a
sum of two printed values. The evidence must be an `asymmetry` command's own
output. If a useful derived quantity is not printed by the CLI, explain the
qualitative relation and leave the number out.

**Fits you flag do not carry physics.** A series whose runs are
`frequency_unresolved` or `amplitude_exceeds_data`, or that you call unreliable
yourself, supports no conclusion about the system — not even a qualitative
one ("consistent with critical slowing"). Say what failed and why.

**A law that did not fit does not get to tell the story.** When `trend
--model` prints `LAW NOT ESTABLISHED` — it did not converge, a parameter sits at
a bound, or an error is as large as its value — the physics that law
stands for — critical slowing down, activated hopping, a Redfield correlation
time, an order-parameter exponent — is **not established**. Say the fit
failed and describe the trend in plain words; do not borrow the law's
vocabulary. And fit a law only to the quantity it is written for: an Arrhenius
law belongs to a rate constant or hop rate, not to whatever rate a model
happened to fit.

**Experiment** — what the survey shows: instrument(s), run count and range, the
scans and their field/temperature span, anything that is not part of a scan,
and whether run order tracks the scan variable. For **each** distinct scan,
write its explicit run-number range or member list beside its geometry and
field/temperature span; a total folder range does not substitute for the
per-scan provenance. State each scan's geometry (ZF/TF/LF) **and how you
confirmed it**, not what the file stamp said.

**Calibration** — where alpha came from (which run, what value, or that it was
assumed 1.0 and why), the detector-grouping/profile provenance, and the
deadtime treatment.

**Model** — the wizard's recommendation on the run you screened, its confidence
and caveat, which run you screened and why, and any departure you made from the
recommendation (a different template, a fixed parameter) with your reason. Say
what each term of the model represents physically, the `Constant` included.

**Results** — a table from `trend`: the scan variable and, per key parameter,
value ± error. Include the amplitudes, not only the rate or frequency, and say
what fraction of the asymmetry is *not* in the component you are interpreting.
Only rows and values the tool produced. Also close the qualitative loop across
the scan endpoints: state plainly whether the defining spectral feature is
present, weakened, unresolved, heavily damped or absent at each end, using the
survey and inspected reduced/fit PNGs as evidence. If the feature disappears
and its fit becomes flagged, report that physical non-observation explicitly;
do not leave it implicit in an excluded-run list.

**Excluded or flagged runs** — every flagged run, its flag, and what you think
went wrong. State plainly that these values are not results.

**What this suggests** — the physics, in words: what the trend means, what it is
consistent with, what would need checking. Interpretation belongs here, and
here it may be qualitative.

**Files** — the work directory (`asymmetry-work/`), any stored `scans/` or
`spectra/` products, and the paths of the PNGs worth looking at.

## 7. Worked example

A synthetic folder of eight simulated runs: one weak-TF calibration run at
100 G, a six-run zero-field temperature scan, 10–60 K, the last of which carries
no signal at all, and one 110 G longitudinal decoupling run. Every command is
run from the project directory, with the data folder (here `runs`, but as often
an absolute path to a share) as its argument; everything the commands write
lands in `asymmetry-work/` beside them, and nothing is written into `runs`.
Output below is real, trimmed.

```console
$ asymmetry survey runs
8 run(s) in runs — SIM

run  T/K    B/G     geom  prec    orient        hist  points  dt   title
---  -----  ------  ----  ------  ------------  ----  ------  ---  -------------------------------------
101  5.00   100.00  TF*   larmor  Longitudinal  8     500     no   Calibrant T=5.0 K B=100.0 G
102  10.00  0.00    ZF    -       Longitudinal  8     500     no   Sample T=10.0 K B=0.0 G
...
107  60.00  0.00    ZF    -       Longitudinal  8     500     no   Sample T=60.0 K B=0.0 G
108  2.00   110.00  -     none    Longitudinal  8     500     no   Sample T=2.0 K B=110.0 G (decoupling)

Alpha-calibration candidates:
  run 101 (best) [measured]: precession at the Larmor frequency of the recorded 100 G (SNR 93)

Scans:
  temperature scan, SIM, ZF, B = 0 G: 6 runs, 10 to 60 K (run 102 -> 107)
```

One calibration run, one ZF temperature scan, one decoupling run. The files
stamp `TF` on every run: the survey reads the ZF runs from their zero field, and
run 101 from its *measured* precession (`TF*`). Run 108's stamp is refuted —
110 G with nothing precessing — so its geometry reads `-`, and it is not offered
as a calibration candidate.

```console
$ asymmetry alpha runs --run 101
Run 101 (SIM00000101.nxs)
  alpha           : 1.2521
  method          : per_run_estimate
  calibration run : yes — precession at the Larmor frequency of the recorded 100 G (SNR 93)
  precession      : at the Larmor frequency: yes — 1.363 MHz (SNR 93) against a Larmor 1.355 MHz

$ asymmetry reduce runs --runs 102-107 --alpha-from 101 --deadtime from_file --plot
run  T/K    B/G   points  A(0)/%   err/%  alpha   deadtime
---  -----  ----  ------  -------  -----  ------  --------
102  10.00  0.00  500     8.537    2.366  1.2521  file
...
107  60.00  0.00  500     -11.301  2.321  1.2521  file
```

Run 107's A(0) is already odd. Screen the coldest run, which has the most
signal:

```console
$ asymmetry wizard runs --run 102 --plot
Run 102 — geometry ZF (from survey), scope auto

Recommendation: exp_constant
  Recommended: Exponential + Constant by AICc (medium confidence).
  confidence : medium
  verdict    : structured

   key           title                  category  AICc   chi2_red  params
-  ------------  ---------------------  --------  -----  --------  ------
*  exp_constant  Exponential + Constant  General  487.7  0.969     3
```

Then chain the series outward from that run and read the trend:

```console
$ asymmetry fit-series runs --runs 102-107 --recipe wizard-102 \
      --order temperature --start 102 --name zf-scan --plot
run  temperature  chi2_red  verdict  flags
---  -----------  --------  -------  --------------------------------
102  10.000       0.969     good     -
...
107  60.000       1.004     good     large_rel_err, spurious_reseeded

flagged  : 1 of 6 run(s) — none were dropped

$ asymmetry trend runs --series zf-scan --plot
run  x   A_1       A_1_err   Lambda    Lambda_err  flags
---  --  --------  --------  --------  ----------  --------------------------------
102  10  19.7178   2.06445   0.143275  0.0228251   -
103  20  19.6304   1.20123   0.192625  0.0205074   -
104  30  20.6347   0.881581  0.229794  0.0187425   -
105  40  19.7195   0.627128  0.27763   0.0187841   -
106  50  20.0489   0.523803  0.311425  0.0186131   -
107  60  0.198726  0.232033  1.06863   1.67424     large_rel_err, spurious_reseeded
```

The summary reports Λ rising from 0.143 ± 0.023 μs⁻¹ at 10 K to
0.311 ± 0.019 μs⁻¹ at 50 K, and lists run 107 under flagged runs: its amplitude
collapsed to 0.199 ± 0.232 % and its rate is unconstrained, so its value is not
a result.

## 8. Installing the skill elsewhere

`asymmetry skill install --agent claude` (add `--project` for the current
directory), `asymmetry skill check` to verify the CLI, loaders, matplotlib and
the installed skill version, `asymmetry skill uninstall` to remove it.

`asymmetry skill install --agent claude --link` symlinks the packaged skill
instead of copying it, so edits in a checkout reach the agent without
reinstalling — for developing this skill, not for using it. `check` reports such
an install as `linked (development)` and always current.
