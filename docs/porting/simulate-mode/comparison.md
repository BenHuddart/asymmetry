# Simulate mode: comparison

Parity of functionality, not implementation: every divergence from WiMDA is
listed with both behaviours. WiMDA line references are to
`/Users/bhuddart/Source/WiMDA/src` (read directly 2026-06-10; `__history/`
and `__recovery/` ignored).

## Cross-program overview

| Aspect | WiMDA | musrfit | Mantid | Asymmetry (this design) |
|---|---|---|---|---|
| GUI simulate dialog | ✅ `Simulate.pas` form | ❌ | ❌ | ✅ `SimulateDialog` (File menu) |
| Hand-scripted simulation | ❌ | ✅ `.msr` generate mode | ✅ Python + `Fit` | ✅ `core/simulate.py` public API |
| Output as live dataset | ❌ (regroups in place) | ❌ (writes to disk) | ✅ (ADS workspace) | ✅ Data Browser entry, badged |
| Loadable file output | ✅ template-copy `.nxs` | ✅ | ✅ | ✅ standalone NeXus V1 writer |
| Per-bin Poisson counts | ✅ | ✅ | ✅ (`Stats=Poisson`) | ✅ `rng.poisson` of expected counts |
| Deterministic seeding | ❌ (Delphi global `random`) | ◐ | ◐ | ✅ explicit seed, default fixed |
| Degrade statistics | ✅ `DegradeStats.pas`, in place | ❌ | ❌ | ✅ derived run, exact thinning |
| Provenance recorded | ◐ title = "Simulation" | n/a | ◐ workspace history | ✅ model + params + seed + template in metadata and in the saved file |

musrfit and Mantid are verification oracles only (GPL); neither has a
muon-specific simulate UI, so WiMDA is the sole behavioural reference.

## Forward-model mechanics: WiMDA ↔ Asymmetry

| Mechanic | WiMDA (`Simulate.pas:38–109`) | Asymmetry |
|---|---|---|
| Template | Previously loaded run; must be `.nxs` | Any loaded run with histograms (NeXus, PSI `.bin`/`.mdu`, ROOT) — the standalone writer removes the format restriction |
| Envelope normalisation | `n0 = ntot/nhis · Δt/τ_μ`, ÷ `multifactor` | Same physics: expected counts per detector per bin at t = 0 from the total-events budget, divided across raw detectors |
| α split | `nf = 2n0α/(1+α)`, `nb = 2n0/(1+α)` (lines 58–59) | Identical formula, α from the template grouping |
| Signal | F: `(1 + 0.01·a(t))`, B: `(1 − 0.01·a(t))`, `a` in percent (lines 70–76) | Identical (models are percent-scale; core converts to fractional internally) |
| Per-group t0 | `t = (i − tzero[g])·Δt` (lines 64–68) | Per-detector `Histogram.t0_bin` from the template (finer than WiMDA's per-group value) |
| Sub-histograms | `multifactor` independent draws per stored histogram (lines 82–88) | One draw per physical detector histogram — Asymmetry stores raw detectors directly, so the multifactor layer is unnecessary; the statistics are identical (sum of independent Poissons) |
| Double-pulse halving | Expected counts halved for `t < dpsep/2` (lines 78–79) | `simulate_double_pulse_run` (the two pulses carry the polarization at `t ± dpsep/2`, weighted `e^{∓dpsep/2τ}`); round-trips the double-pulse single-histogram fit (delivered with `count-domain-fit-modes`) |
| Non-FB count modes | `nn = a(t)/ghists · evfactor · e^{−t/τ}` (line 76) | `simulate_count_run` (follow-on, [follow-ons.md](follow-ons.md)): the same `+a(t)` on every group as independent single-histogram counts, fittable by the PR #41 single-histogram mode |

## Divergences (both behaviours stated)

1. **Background term.** WiMDA: none — expected counts are pure
   envelope × signal. Asymmetry: optional flat background rate per detector
   (default 0), matching the time-independent uncorrelated background of
   continuous-source data described in the textbook. With the default the
   two agree exactly.
2. **Bins before t0.** WiMDA evaluates the same formula at negative t,
   producing a *growing* exponential before t0 (unphysical; a large t0 gives
   huge counts in bin 1). The screenshot archetypes use a flat N₀ plateau.
   Asymmetry: pre-t0 bins contain **background only** (zero when b_d = 0),
   matching what real pulsed-source histograms look like. Affects no
   analysis (good-bin windows start after t0) but matters for raw-count
   display realism.
3. **RNG and seeding.** WiMDA: Numerical Recipes `poidev` rejection sampler
   (`numlib.pas:821–858`) on Delphi's global `random`; no seed control; runs
   are unreproducible. Asymmetry: `numpy.random.default_rng(seed)`
   (PCG64); the seed is a first-class argument, recorded in provenance;
   fixed seed → bit-identical run.
4. **Degrade sampling law.** WiMDA: `new = Poisson(k·f)` where k is the
   *measured* count (`DegradeStats.pas:42–44`). For k ~ Poisson(λ) the
   marginal of Poisson(k·f) is over-dispersed relative to Poisson(λf)
   (variance λf(1+f), not λf). Asymmetry: **binomial thinning** for f < 1 —
   each recorded count survives with probability f, which gives *exactly*
   Poisson(λf) marginals, the statistically correct "shorter run". For
   f > 1 there is no exact construction from recorded data; Asymmetry keeps
   WiMDA's `Poisson(k·f)` with the over-dispersion caveat documented in the
   user guide.
5. **Degrade output.** WiMDA: overwrites the loaded histograms in place,
   one-shot (button disables; only reloading the file undoes it).
   Asymmetry: returns a new derived run beside the original (decision 3),
   repeatable with different factors/seeds.
6. **File writing.** WiMDA: byte-copies the loaded `.nxs` and overwrites
   counts/title/notes/sample-name/deadtimes/time_zero in `NXacc_rdwr` mode
   (shape-locked; stale logs and sample metadata from the template survive
   into the "simulation" file; `time_zero` is quantised to whole μs at
   `Simulate.pas:157`). Asymmetry: writes a fresh minimal ISIS muon NeXus V1
   file containing exactly what `NexusLoader` reads, with exact `time_zero`,
   synthetic-marked title, and a `/run/simulation` provenance group; no
   inherited logs.
7. **Deadtimes.** WiMDA writes zeros into the copied file's deadtime table
   and forces the deadtime-correction checkbox off before generating.
   Asymmetry does the same (zeros, `deadtime_correction: False` in
   grouping) — and this is recorded as *correct*, not parity-for-its-own
   sake: the synthetic counts contain no deadtime distortion, so a zero
   deadtime is the true instrument description. Simulating deadtime
   distortion (then correcting it back out) is a follow-on.
8. **Event-total accounting.** WiMDA's `ntot` sets the lifetime-envelope
   budget; the realised total fluctuates (Poisson) and the oscillatory terms
   make the F/B split only approximately even. Asymmetry keeps the same
   convention (document: "total events" is the expected envelope budget, not
   a guaranteed realised count). No divergence, but WiMDA never documents it.
9. **α on reload.** ISIS muon NeXus V1 has no α field, so a reloaded
   synthetic run gets the loader's default α — in both programs. Asymmetry
   records the generating α in `/run/simulation` (human-readable) and in
   run metadata, but does not invent a non-standard loader path for it.
   Round-trip refit tests therefore either simulate at α = 1 or re-enter α
   after reload, exactly as with real data.
10. **Run metadata.** WiMDA blanks temperature/field/start/stop in the main
    window after generating (display only) but the *copied file* keeps the
    template's sample values. Asymmetry: the synthetic run carries the
    template's field/temperature explicitly (they parameterise models like
    precession frequencies) plus an unambiguous synthetic marker; nothing
    stale survives by accident.
11. **Source of the instrument template.** WiMDA can only simulate from the
    previously loaded `.nxs` run — no run, no Simulate. Asymmetry (follow-on,
    [follow-ons.md](follow-ons.md)) additionally simulates from **built-in
    idealised instrument templates** (an ideal pulsed F/B and an ideal
    continuous F/B, no run loaded) and from one-click **textbook archetype
    presets** (Ag KT, EuO T-scan, F-μ-F, YBCO TF) that generate badged
    synthetic runs deterministically and refit to their stated physics. WiMDA
    has no analogue of either, nor of the multi-group per-phase ring
    (`simulate_multi_group_run`) or the in-GUI pull-distribution diagnostic.
    No conflict with WiMDA behaviour — these are pure additions; the
    loaded-run template path still behaves as WiMDA's does.

## Settled exclusions

- **Sample-environment logs, event mode, extra instrument templates, PSI/ROOT
  writers** — see README scope rationale.
- **Two-period NeXus *writing*** — the writer emits one period
  (`histogram_data_1`), as WiMDA does. Two-period **synthesis** itself shipped
  as the in-memory `simulate_two_period_run` (follow-on,
  [follow-ons.md](follow-ons.md)): it builds the loader's two-period payload so
  `select_period` and the green∓red combination work, but Save-as-NeXus still
  flattens to the red period. Promoting period support into the writer (and the
  loader round-trip) is the remaining deferral.
