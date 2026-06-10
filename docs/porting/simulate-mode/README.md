# Simulate mode — study

Date: 2026-06-10. Branch: `feat/simulate-mode`. Status: **implemented**
(same day; four phases, each validate-green — see the as-implemented notes
in [implementation-options.md](implementation-options.md)). Shipped:
`core/simulate.py`, `core/io/nexus_writer.py`,
`gui/windows/simulate_dialog.py`, the Data Browser degrade action and
provenance badging, 53 new tests, and `docs/user_guide/simulation.rst`.

Promotes the `simulate-mode` candidate (tier "now", score 16) under the
`wimda-parity-gap` umbrella (Wave A, project 8). The original candidate docs
are preserved for scoring history in [candidate/](candidate/) — they had to
move out of `docs/porting/candidates/` because the structural harness requires
unique slugs across candidate and study entries, and this promotion keeps the
slug `simulate-mode` (unlike the `maxent-spectrum` → `maxent` precedent, where
the names differed).

## What this feature is

Synthetic μSR dataset generation: take a fit model + parameter values + a
loaded run as instrument template, generate per-detector Poisson count
histograms, and return a first-class run that can be plotted, fitted, FFT'd
and saved as a loadable HDF5 NeXus file. Plus **degrade statistics**: Poisson
thinning of a loaded run by a factor ("what would this look like with half
the beam time"), sharing the same sampling core.

Three needs at once: teaching, fit validation (does the pipeline recover
known parameters?), and test-data manufacture for the rest of the parity
portfolio (double-pulse, pulse-shape and binning-mode verification all want
synthetic runs).

## Scientific grounding

The decay positron counts in each detector time bin are Poisson distributed;
the expected counts for detector $d$ follow the lifetime envelope modulated
by the asymmetry signal:

N_d(t) = N_{0,d} · exp(−t/τ_μ) · [1 + a_d(t)] + b_d

with τ_μ = 2.1969811 μs, a_d(t) the (signed, fractional) asymmetry signal
seen by that detector, and b_d a time-independent background rate (the
uncorrelated background of continuous sources; negligible at pulsed sources).
Simulation draws Poisson variates of these *expected counts* — never Gaussian
noise added to an asymmetry curve — so the per-bin errors propagate correctly
through the real reduction chain (grouping → α-balanced asymmetry → error
formula) *by construction*. The α calibration enters as the relative
efficiency of the forward and backward groups: for total rate 2·N₀ split
across the pair, N₀_F = 2N₀α/(1+α) and N₀_B = 2N₀/(1+α), so that the
reduction (F − αB)/(F + αB) recovers a(t) with α restored.

References:

- S. J. Blundell, R. De Renzi, T. Lancaster, and F. L. Pratt (eds.),
  *Muon Spectroscopy: An Introduction* (Oxford University Press, 2022) —
  Ch. 15 (Poisson statistics of counts per bin and the Gaussian
  approximation; the time-independent uncorrelated background; time zero and
  detector phases; the multi-component asymmetry model and the local
  (per-group count rate, phase, background) vs global (frequency, relaxation)
  parameter split; the χ²ᵣ "Goldilocks zone"), and Ch. 14 (pulsed vs
  continuous source characteristics: deadtime distortion, double-pulse
  structure, background levels).

## WiMDA reference behaviour (verified against the Pascal source)

`Simulate.pas` (172 lines) and `DegradeStats.pas` (50 lines), with supporting
machinery in `muondata.pas`, `Group.pas`, `nexusunit_.pas`, `numlib.pas`,
`Analyse.pas`. Verified 2026-06-10 by direct reading; the umbrella brief was
accurate but incomplete. Mechanics:

- **Template**: requires the previously loaded run to be `.nxs`; warns
  otherwise. Forces the deadtime ("count correction") checkbox off before
  generating.
- **Events**: the user enters total events in MEv. Expected counts per bin
  at t = 0 per histogram: `n0 = ntot/nhis · Δt/τ_μ` (exactly the lifetime
  envelope normalisation), further divided by `multifactor` (the number of
  raw detectors summed into each stored WiMDA histogram, derived in the
  NeXus loader as `ndet div 32`, with an EMU special case).
- **α split** (F/B asymmetry mode): `nf = 2·n0·α/(1+α)`, `nb = 2·n0/(1+α)`;
  forward histograms get `nf·exp(−t/τ)·(1 + 0.01·a(t))`, the backward group
  `nb·exp(−t/τ)·(1 − 0.01·a(t))` with `a(t) = musrfunc(t, g, p)` in percent.
- **Per-group t0**: `t = (i − tzero[g])·Δt` — bins before t0 evaluate the
  same formula at negative t (a growing exponential; unphysical, see
  comparison.md).
- **Double-pulse halving**: if the double-pulse flag is set, expected counts
  are halved for `t < dpsep/2` (only the first of the two proton pulses has
  implanted muons before the second arrives; `dpsep2 = dpsep/2` in
  `Analyse.pas`).
- **Non-FB fit modes** (single-histogram / count fits): the model already
  returns count-scale values; WiMDA uses `nn = a(t)/ghists · evfactor ·
  exp(−t/τ)` with `evfactor = ntot/eventstot` scaling relative to the loaded
  run's real event total.
- **Sampling**: `poisson(m)` in `numlib.pas` is the Numerical Recipes
  `poidev` rejection sampler driven by Delphi's global `random` — there is
  **no seed control** anywhere in the UI.
- **Sub-histograms**: each stored histogram gets `multifactor` independent
  Poisson draws written to the raw per-detector array `counts1`, summed for
  the display histogram; then `regroup` rebuilds the analysis groups.
- **Save Simulation**: copies the loaded `.nxs` file byte-for-byte, then
  re-opens it read-write and overwrites: `title`/`notes`/`sample/name` →
  "Simulation", `/run/instrument/detector/deadtimes` → zeros,
  `/run/histogram_data_1/counts` → `counts1`, and `time_zero` → t0 of group
  1 **quantised to whole μs**. Only `histogram_data_1` is written (one
  period). The NeXus API is opened in `NXacc_rdwr` mode, which cannot change
  dataset shapes — the simulation is therefore locked to the template's
  detector count and bin count.
- **DegradeStats**: in-place, one-shot (button disables itself):
  `histos[d,i] := poisson(histos[d,i] · factor)` for every bin, then
  `regroup` + `histototals`. The factor is a free positive number (> 1
  "upgrades"). This is Poisson resampling of the *measured* counts, not
  binomial thinning (see comparison.md).

## Scope

**Core (new, Qt-free `core/simulate.py`)**

- Model + parameters + instrument template (grouping, per-detector t0s, α,
  good frames, bin width, detector count — all from a loaded run) + total
  events → per-detector count histograms; Poisson draws of the expected
  counts; deterministic seeding throughout.
- Returns a first-class `Run` (+ reduced `MuonDataset`) with provenance
  metadata clearly marking it synthetic, including the generating model
  expression, parameter values, seed, and template identity.
- Optional flat background rate per detector (default 0) — an improvement
  over WiMDA, which has no background term.
- **Degrade statistics**: thinning of a loaded run by a factor, same
  sampling core, returning a **new derived run** (decision: not in-place).
- Promote the `docs/screenshots/data/archetypes.py` synthesis helpers into
  the new core module where they overlap (`_build_run_with_detector_asymmetries`,
  `_poisson_errors`); the screenshot generators then import from core.

**NeXus round-trip**

- Write the synthetic (or degraded) run as a loadable HDF5 `.nxs` via a
  minimal standalone ISIS muon NeXus V1 writer (decision; see
  implementation-options.md) that produces exactly what `NexusLoader._load_v1`
  consumes, plus a `/run/simulation` provenance group.

**GUI**

- File menu → "Generate Synthetic Run…" dialog: template picker (loaded
  runs), model picker (reusing `FitFunctionBuilderDialog`), parameter table
  seeded from the current fit when one exists, events spinner, optional
  fixed RNG seed, "Save as NeXus…" action; result appears in the Data
  Browser like any run, clearly badged.
- Data Browser context-menu action "Degrade Statistics…" (factor + seed).
- `mainwindow.py` / `data_browser.py` touches minimal and additive (other
  Wave A projects edit these files).

## Decisions log

| # | Question | Decision (with Ben) |
|---|---|---|
| 1 | Instrument template when no run is loaded | **Require a loaded run** (WiMDA parity; matches the out-of-scope line). Built-in ideal-instrument template recorded as a follow-on for teaching. |
| 2 | Persistence of synthetic runs in `.asymp` projects | **Via saved NeXus file** — a synthetic run lives in memory until saved as `.nxs`; the project references the file like any run. Zero schema churn (`core/project/schema.py` is shared-touch across Wave A). |
| 3 | Degrade statistics output | **New derived run** beside the original, badged with factor + seed, saveable via the same writer. WiMDA's in-place overwrite rejected. |
| 4 | NeXus writing strategy | **Minimal standalone V1 writer** (not WiMDA's template-copy). Works for templates loaded from PSI `.bin`/`.mdu` too; no stale sample logs; shape freedom; clean provenance. |
| 5 | Dialog model sourcing | **Seed from the current fit when one exists**, free choice via the builder otherwise. |
| 6 | Per-group amplitude handling | Core API takes **per-group signal specs** (the promoted archetypes pattern); the v1 dialog exposes a **single model with the WiMDA F/B α split**. Multi-group TF simulation with per-group phases recorded as a follow-on. |
| 7 | Injected deadtime/background | **Deadtimes written as zeros** (WiMDA parity; the synthetic counts contain no deadtime distortion so zero is the *correct* value, not a shortcut). **Optional flat background** rate (default 0) is in scope. Simulating deadtime distortion recorded as a follow-on. |

## Out of scope (with rationale)

- **Sample-environment log simulation** — `nexus_time_series` stays empty;
  nothing in the analysis chain needs it, and faking logs invites confusion
  with real provenance.
- **Event-mode simulation** — Asymmetry has no event-mode pipeline to feed.
- **Instrument templates beyond "copy a loaded run"** — a built-in ideal
  instrument is a recorded follow-on, not in this project (decision 1).
- **PSI `.bin` / ROOT output formats** — the NeXus V1 writer covers the
  loadable round trip; other writers add maintenance surface with no new
  capability.
- **Non-FB count-mode simulation** (WiMDA's `evfactor` path) — meaningful
  only once count-domain fit modes exist; deferred to
  `count-domain-fit-modes`, which can reuse the sampling core.
- **Double-pulse and two-period simulation** — recorded as follow-ons for
  `count-domain-fit-modes` to claim unless they fall out naturally.

## Documents

- [comparison.md](comparison.md) — WiMDA vs Asymmetry behaviour, every
  divergence with both behaviours stated; musrfit/Mantid columns.
- [implementation-options.md](implementation-options.md) — options analysis,
  chosen design, ordered implementation plan, file-by-file touch list, test
  plan, follow-ons.
- [follow-ons.md](follow-ons.md) — addendum: built-in instrument templates,
  the archetype gallery, the pull-distribution diagnostic, multi-group
  simulation and the project-save warning, built on the implemented feature.
- [test-data.md](test-data.md) — corpus runs and synthetic fixtures.
- [verification-plan.md](verification-plan.md) — round-trip refit recovery,
  pull-distribution check, degrade error scaling, determinism.
- [candidate/](candidate/) — the original candidate docs (scoring history).
