---
name: asymmetry-analysis
description: Analyse muon-spin spectroscopy (μSR/muSR) run folders with the Asymmetry CLI — ISIS/PSI .nxs, .bin, .mdu or .root runs. Use to survey, calibrate alpha, reduce to asymmetry, screen fit models, fit or summarise a ZF/TF/LF temperature or field scan of muon runs.
---

# Preliminary μSR analysis with the `asymmetry` CLI

You have a folder of muon-spin-spectroscopy runs and a command-line tool that
surveys them, reduces them to forward–backward asymmetry, screens fit models,
fits a scan, and trends the results. This skill is the recipe for driving it
end to end and writing a summary a spectroscopist would recognise.

Full flag reference: `references/commands.md` beside this file. Every command
also has `--help`.

## 1. What this does, and what it does not

**In scope.** Preliminary, time-domain forward–backward asymmetry analysis of:

- zero-field (ZF), transverse-field (TF) and longitudinal-field (LF) runs;
- a temperature scan or a field scan at one geometry;
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
| ALC / avoided-level-crossing resonance | Many runs at one fixed temperature with the LF stepped over a wide range (hundreds to thousands of gauss), often a chemistry sample. The analysis is integral asymmetry *versus field*, fit as a resonance line shape; there is no command for that. |
| Count-domain fitting | You need per-detector counts with N₀ and a relaxation term, not asymmetry. `reduce` only produces asymmetry. |
| Multi-group / orientation-resolved analysis | The run has many detector groups that must be fit together (angle-dependent Knight shift, crystal rotations). This CLI reduces exactly one forward/backward pair. |
| Fourier or maximum-entropy spectra | Anything asking for a frequency spectrum rather than a time-domain fit. There is no transform command. |
| Negative-muon (μ⁻) elemental analysis | Gamma spectra, elemental lines. Not asymmetry data. |
| Rotating-reference-frame or RF-resonance runs | Titles or notes naming RF; data modulated at a reference frequency. |
| Muonium chemistry / reaction rates | Rates versus concentration across samples, not a spin-relaxation trend. |
| Simultaneous multi-field fits | Several runs at *the same* temperature and *different* fields that must be fit jointly with shared parameters (a decoupling triplet per temperature). `fit-series --global` **pins** a parameter at one value for the whole series — it fits nothing jointly — so it cannot represent this. Fitting each field run independently and presenting the rates as if they were shared parameters is wrong; say the structure needs a simultaneous fit and stop. |
| A fragment of a published multi-field campaign | Disjoint blocks of run numbers with large gaps, multi-tesla fields, no self-contained scan. You cannot reconstruct the campaign's field log from what is on disk. |

A folder the tool can *load* is not automatically a folder the tool can
*analyse*. Loadability is not scope.

## 2. The workflow

Run every command from the directory holding the data (or pass an absolute
path). Every command writes into `<folder>/.asymmetry/` — the work directory —
so the next command picks the state up. **Do not pass `--workdir`**; the
default is right and a mismatched one silently loses your reduced runs.

Add `--json` when you need to parse a payload; the default human table is
usually easier to read and is what these examples show.

### Step 1 — survey the folder

```bash
asymmetry survey <folder>
```

Read it properly before doing anything else. It gives you, per run:
temperature, field, geometry, **`prec`**, detector orientation, point count,
whether the file carries deadtime values, title and notes; then the
**alpha-calibration candidates**, then **`scans`** — runs grouped into
temperature scans at fixed field and field scans at fixed temperature.

**`geom` and `prec` together are the geometry evidence.** For every run with a
non-zero recorded field the survey reduces the spectrum and looks for precession
at that field's Larmor frequency (γ_μ/2π × B = 13.55 kHz/G). `prec` reports what
it found:

| `prec` | Means |
|---|---|
| `larmor` | A strong line at the Larmor frequency of the recorded field. The field **is** transverse, and `geom` reads `TF*` — the `*` says the spectrum decided it, not the file. |
| `other` | A strong line somewhere else: the muon is precessing in an **internal** field that beats the applied one. An ordered magnet, below its transition. |
| `none` | No line worth the name. Whatever the file stamps, **this field is not precessing the muon** — in a field scan that means longitudinal decoupling. A row reading `geom TF` with `prec none` is a file stamp the data does not support; do not believe the `TF`. |
| `-` | Not measurable: zero field (nothing to look for), or a Larmor frequency above the record's Nyquist frequency (a kilogauss-scale field at a pulsed source). |

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

`asymmetry info <file>` prints one file's metadata if you need to check a
single file directly.

### Step 2 — decide alpha, and never guess it

Alpha is the forward/backward detector balance. It is measured on a **weak
transverse-field** run, where the precession is undamped and the balance shows
up cleanly.

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

- **The survey lists calibration candidates** → use one and report the value.
  When several are listed, prefer the one the survey marks `(best)` — the
  strongest measured precession — unless it is a member of the scan you are
  about to analyse and a separate run is available. A dedicated run at 20 G is
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
- `--plot` writes `<folder>/.asymmetry/plots/reduced-<run>.png`. Read a few of
  them with the Read tool — the lowest-temperature, the highest, and one in the
  middle. That is how you learn whether there is an oscillation, a Kubo–Toyabe
  dip, or featureless relaxation, before any model is chosen.
- `--tmax` trims a noisy tail; `--rebin k` merges bins. Both are available on
  `reduce`, and `fit`/`fit-series` also take `--tmin`/`--tmax` per fit.

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
- `prec none` on a non-zero field → **longitudinal**. The muon is not precessing
  in the applied field, so the field is along the beam. This holds even when
  `geom` still reads `TF`: that is the file's stamp, and the measurement refutes
  it. A field scan at fixed temperature over a wide range is LF decoupling, and
  this is how you confirm it.
- `prec other` → an ordered magnet precessing in its own internal field. That
  says nothing about the applied field's direction; fall back to the file's
  `geom`, the scan the run belongs to, and the reduced PNG.
- `prec -` → nothing was measurable. Judge it from the PNG as below.

**Confirm on the reduced PNG** — always for `other` and `-`, and as a sanity
check otherwise. A **transverse** field precesses the muon at γ_μ/2π × B —
13.55 kHz/G, so 20 G ≈ 0.27 MHz, 110 G ≈ 1.5 MHz, 400 G ≈ 5.4 MHz — a plainly
visible oscillation filling the early-time window. No oscillation at that
frequency means the field is longitudinal. (Use `reduce --tmax 2 --plot` to zoom
the early window if the full range is too compressed to judge.)

Then pass the right `--geometry ZF|TF|LF` to `wizard` for every run you screen,
and say in the summary what each scan's geometry is and how you established it.
Calling a decoupling LF series "TF" misreads the whole experiment even when the
fitted model is right.

### Step 4 — screen one run with the wizard

```bash
asymmetry wizard <folder> --run 102 --geometry ZF --plot
```

The wizard fits a scoped set of candidate models to one reduced run and ranks
them. Screen **the run whose spectrum shows the effect you are measuring most
clearly** — not every run, and not reflexively the coldest one.

Pick it from the `reduce` table and the reduced PNGs, not from the run list:

- The clearest run for a **relaxation** effect (a glass freezing, a dynamic
  rate) is where the relaxation is *fastest but still resolved* — usually the
  cold end, and usually not the run where A(0) has collapsed.
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
and wrong exactly where you had to reason — a `prec none` run the file stamps
`TF`. The wizard scopes its candidate families by geometry, so screening a
decoupling LF run as `TF` puts precession models in front of it and nothing
else.

Read from the output:

- **the recommendation** (model key and title) and the **ranked table** of
  candidates with AICc, reduced χ² and parameter count;
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

`--scope <preset>` restricts the candidate families when you know the physics:
`auto` (default), `zf-static-magnetism`, `tf-knight-precession`,
`tf-superconductor`, `lf-dynamics`, `fluoride-fmuf`, `muonium-radical`, `all`.
`auto` expands a family only when the spectral search finds support for it, so
a magnet whose oscillation is fast and heavily damped can come back as a bare
`Exponential + Constant`. If you expect static order and `auto` gives you a
plain relaxation, re-screen with `--scope zf-static-magnetism` and compare.

The scope note may say "sample name suggests fluorine": that is read from the
run's title or sample text (`CaF2`, `LiF`, `KTCNQF4`), and it promotes the
F–μ–F family. Trust it when the sample really is a fluoride; a fluoride
whose title does not name it still gets F–μ–F candidates from the spectral
search.

`wizard` writes `recipes/wizard-<run>.json` — the fit recipe, the only
contract between screening and fitting. `--plot` writes
`plots/wizard-<run>.png` (data, recommended curve, residuals). Look at it.

The wizard's candidate search prints `AsymmetryScaleWarning` blocks on stderr.
They are benign noise from seeding trial models; ignore them.

### Step 5 — fit the scan as a chained series

```bash
asymmetry fit-series <folder> --runs 102-107 --recipe wizard-102 \
    --order temperature --start 102 --name zf-scan --plot
```

- `--recipe` takes a name in `recipes/` (no path, no `.json`) or a path.
- `--order temperature` or `--order field` — the quantity the scan varies, the
  axis of the trend. `--order run` only when neither applies.
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

A `poor` χ² verdict is common on high-statistics ISIS data (the band is tight
with thousands of degrees of freedom) and is not by itself a reason to discard
a run; a flag is.

### Step 6 — read the trend

```bash
asymmetry trend <folder> --series zf-scan --plot
```

This is the table your Results section comes from: the scan variable and every
fitted parameter with its uncertainty, per run, with the flags carried
through. `--csv <path>` also writes it as CSV. `--plot` writes one PNG per
parameter, with flagged points drawn distinctly.

**Look at the trend PNGs with the Read tool before writing anything.** A trend
that is flat, that jumps, or whose scatter swamps the error bars is telling you
something the table alone will not.

### Step 7 — write the summary

Template in section 6.

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
  characteristic dip and one-third recovery, not a plain decay.
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

1. Read the per-run PNGs (`plots/<name>/<run>.png`) at the flagged
   temperatures. Compare them with the run you screened. What changed?
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

**Hand-editing a recipe.** A recipe is small JSON. To swap one component, edit
three things: `expression`, the entry in `model.component_names`, and the
parameter list (name and starting value). For example, `Exponential` →
`Gaussian` means `Lambda` → `sigma`; `Constant` carries `A_bg`. Write it to
`recipes/<name>.json` and pass `--recipe <name>`. Run `asymmetry fit --run N
--recipe <name>` on one run first to check it converges before spending a
series on it.

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
| `wizard` | 3–8 s per ISIS run |
| `fit`, `fit-series` | a few seconds for a whole scan |
| `trend` | instant (it reads stored results) |
| global/simultaneous screening | minutes, and not exposed in this CLI |

So: screen **one or two** runs, not every run. Reduce and fit whole scans
freely — those are cheap.

## 5. Physics to ask yourself

Match the question to the geometry. Background for each of these is in the
published workflow chapters; read the relevant one rather than guessing:

- <https://benhuddart.github.io/asymmetry/workflows/temperature_scan_magnetism.html>
- <https://benhuddart.github.io/asymmetry/workflows/fmuf_entangled_states.html>
- <https://benhuddart.github.io/asymmetry/workflows/dynamic_kt_copper.html>
- <https://benhuddart.github.io/asymmetry/workflows/lf_decoupling_dynamics.html>
- <https://benhuddart.github.io/asymmetry/workflows/superconductor_penetration_depth.html>
- <https://benhuddart.github.io/asymmetry/workflows/global_fit_ionic_motion.html>
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
relaxation survives is dynamic. Does the rate rise on cooling towards a
freezing or glass transition? Is a stretched exponential needed (a distribution
of rates, as in a spin glass) rather than a single exponential? Does the
recovered asymmetry increase with field, as decoupling predicts?

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
> session.

**Experiment** — what the survey shows: instrument(s), run count and range, the
scans and their field/temperature span, anything that is not part of a scan,
and whether run order tracks the scan variable. State each scan's geometry
(ZF/TF/LF) **and how you confirmed it**, not what the file stamp said.

**Calibration** — where alpha came from (which run, what value, or that it was
assumed 1.0 and why), and the deadtime treatment.

**Model** — the wizard's recommendation on the run you screened, its confidence
and caveat, which run you screened and why, and any departure you made from the
recommendation (a different template, a fixed parameter) with your reason. Say
what each term of the model represents physically, the `Constant` included.

**Results** — a table from `trend`: the scan variable and, per key parameter,
value ± error. Include the amplitudes, not only the rate or frequency, and say
what fraction of the asymmetry is *not* in the component you are interpreting.
Only rows and values the tool produced.

**Excluded or flagged runs** — every flagged run, its flag, and what you think
went wrong. State plainly that these values are not results.

**What this suggests** — the physics, in words: what the trend means, what it is
consistent with, what would need checking. Interpretation belongs here, and
here it may be qualitative.

**Files** — the work directory, and the paths of the PNGs worth looking at.

## 7. Worked example

A synthetic folder of eight simulated runs: one weak-TF calibration run at
100 G, a six-run zero-field temperature scan, 10–60 K, the last of which carries
no signal at all, and one 110 G longitudinal decoupling run. Output below is
real, trimmed.

```console
$ asymmetry survey runs
8 run(s) in runs — SIM

run  T/K    B/G     geom  prec    orient        hist  points  dt   title
---  -----  ------  ----  ------  ------------  ----  ------  ---  -------------------------------------
101  5.00   100.00  TF*   larmor  Longitudinal  8     500     no   Calibrant T=5.0 K B=100.0 G
102  10.00  0.00    ZF    -       Longitudinal  8     500     no   Sample T=10.0 K B=0.0 G
...
107  60.00  0.00    ZF    -       Longitudinal  8     500     no   Sample T=60.0 K B=0.0 G
108  2.00   110.00  TF    none    Longitudinal  8     500     no   Sample T=2.0 K B=110.0 G (decoupling)

Alpha-calibration candidates:
  run 101 (best) [measured]: precession at the Larmor frequency of the recorded 100 G (SNR 93)

Scans:
  temperature scan, ZF, B = 0 G: 6 runs, 10 to 60 K (run 102 -> 107)
```

One calibration run, one ZF temperature scan, one decoupling run. The files
stamp `TF` on every run: the survey reads the ZF runs from their zero field, and
run 101 from its *measured* precession (`TF*`). Run 108 keeps the file's `TF` but
`prec none` refutes it — 110 G with nothing precessing is a longitudinal field,
and it is not offered as a calibration candidate.

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
