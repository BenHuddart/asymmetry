# Simulate mode: follow-ons (addendum)

Date: 2026-06-10. Branch: `feat/simulate-mode-followons` (off `main`). This
addendum records the follow-on work built on top of the implemented
simulate-mode feature ([README.md](README.md),
[implementation-options.md](implementation-options.md)). It is **not** a new
study — the decisions, divergences and scope of the original port still bind;
this file documents what the recorded follow-ons became.

Settled with Ben at the session checkpoint:

- **Cut**: built-in ideal-instrument templates; the archetype gallery; the
  pull-distribution diagnostic; multi-group dialog support; plus the small
  project-save warning for unsaved synthetic runs.
- **Built-in template inventory**: a *pair* — an ideal pulsed F/B and an ideal
  continuous F/B (the two source archetypes Ch. 14 contrasts).
- **Deadtime-distortion injection**: stays deferred (recorded follow-on below).
- **Gallery presets**: Ag ZF/LF Kubo–Toyabe, EuO temperature scan, F-μ-F
  (PbF₂), YBCO TF.

## What shipped

### 1. Built-in ideal-instrument templates (`core/simulate.py`)

`InstrumentTemplate` + `BUILTIN_TEMPLATES` register two idealised instruments
that stand in for a loaded run, so the dialog can simulate with **nothing
loaded** (closing decision 1's recorded follow-on):

| Key | Detectors | Bin width | Window | t0 bin | Background |
|---|---|---|---|---|---|
| `ideal_pulsed_fb` | 32 F + 32 B | 16 ns | 32 μs | 100 | 0 |
| `ideal_continuous_fb` | 1 F + 1 B | 1 ns | 10 μs | 1000 | 10 counts/bin |

A template carries empty histograms plus a complete grouping; `simulate_run`
reads only structure (detector count, bin width, per-detector t0, good-bin
window, grouping), so the existing forward model drives it unchanged. The
continuous instrument carries a non-zero flat background — the time-independent
uncorrelated background characteristic of continuous sources (Ch. 14); the
pulsed one does not. `build_builtin_template(key)` materialises the `Run`.

The `SimulateDialog` template combo lists the built-ins (combo data is the
string registry key, distinguishing them from a loaded run number) and seeds
its event-budget and background spinners from the instrument's defaults;
Generate is no longer blocked when no run is loaded.

### 2. Archetype gallery (`core/simulate_presets.py`)

`ARCHETYPE_PRESETS` + `build_preset_runs(key)` expose the promoted textbook
archetypes as one-click **badged synthetic runs** generated through the full
simulate pipeline (Poisson histograms, not Gaussian-noised curves), using the
canonical literature parameters that also feed the documentation screenshots:

| Preset | Physics | Chapter |
|---|---|---|
| `ag_zf_kt` | Ag ZF Gaussian Kubo–Toyabe, Δ = 0.39 μs⁻¹ | Ch. 5 |
| `ag_lf_decoupling` | Ag LF decoupling series (0/10/25/50 G) | Ch. 5 |
| `euo_tscan` | EuO ferromagnet through Tc = 69 K (5 temperatures) | Ch. 6 |
| `fmuf_pbf2` | PbF₂ F-μ-F entanglement, r ≈ 1.17 Å | Ch. 4 |
| `ybco_tf` | YBCO TF Knight-shifted Larmor precession | Ch. 8 |

Each preset generates deterministically from a fixed seed and, refitted with
its generating model, recovers its stated physics within the fit errors
(Δ, precession frequency, B_L decoupling field, F-μ-F separation — all verified
in `tests/test_simulate_presets.py`). Scan presets emit a whole family in one
click. References cite the textbook by name and chapter, **never** by equation
number (study standing rule). Launched from **File → Simulate Preset**.

### 3. Pull-distribution diagnostic (`core/pull_diagnostic.py`)

`run_pull_distribution(template, model, parameters, refit, …)` re-simulates a
fit over N seeds at matched statistics, refits each, and collects per-parameter
pulls (θ̂ − θ_true)/σ. `ParameterPull`/`PullDistribution` expose the mean,
width and their N(0, 1) standard errors plus a calibration verdict
(well-calibrated / over-estimated / under-estimated). The core is
**engine-agnostic** — the caller injects a `refit` callable, so core never
imports the fitting engine or iminuit.

The GUI window (`gui/windows/pull_diagnostic_window.py`) histograms the pulls
against an N(0, 1) overlay with a verdict line; a **Pull diagnostic…** button on
the single-fit tab enables after a converged time-domain fit and re-simulates
from that run, model and fitted values over the same fit window.

Pulls centre on N(0, 1) with **no** (1 + A²)/(1 − A²) correction — `main` now
carries the exact Poisson error propagation (PR #35,
[asymmetry-error-propagation](../asymmetry-error-propagation/)). The original
simulate-mode refit test that documented the (1 − A²)/(1 + A²) χ²ᵣ bias was
already updated when #35 landed; this diagnostic confirms it in the GUI and in
`tests/test_pull_diagnostic.py`.

### 4. Multi-group simulation (`core/simulate.py`)

The recorded decision-6 follow-on. `GroupSignalSpec` (amplitude, relative
phase, N₀ weight per group) + `build_group_signals(model, specs,
base_parameters)` build the per-group fractional signals
`a_g(t) = amplitude · P(t)` on a **normalised** polarisation model (the
per-group amplitude owns the scale, exactly as in the grouped time-domain fit
contract); a non-zero relative phase requires a phase-capable model.
`simulate_multi_group_run(...)` wraps the existing
`simulate_run_from_group_signals` seam with provenance, and
`group_specs_from_grouped_fit(result)` lifts the per-group nuisance block out of
a grouped fit so a fitted ring can be re-simulated.

The `MultiGroupSimulateDialog` (File → Generate Multi-Group Run…) shows a
per-group amplitude/phase/N₀ table seeded from the active run's last grouped
fit when one is cached (`GlobalFitTab.grouped_simulate_seed_for_run` via
`MultiGroupFitWindow`), otherwise from the template's groups with evenly spread
phases. Verified bin-exact: each group's expected counts de-modulate to a clean
lifetime envelope reproducing the seeded `a_g(t)`.

### 5. Project-save warning

`MainWindow._unsaved_synthetic_run_labels()` finds synthetic/degraded runs whose
`run.source_file` is empty; `_write_project` warns before saving (Save anyway /
Cancel) that those runs will not reload with the project — closing the
as-implemented note on decision 2 (persistence is via Save-as-NeXus).

## New divergence from WiMDA

One genuinely new behaviour, recorded in [comparison.md](comparison.md)
(divergence 11): WiMDA can only simulate from a previously loaded `.nxs` run;
Asymmetry now also simulates from **built-in idealised instrument templates**
(no run loaded) and from **one-click textbook archetype presets**. WiMDA has no
analogue of either, nor of the multi-group per-phase ring or the in-GUI
pull-distribution diagnostic.

## Delivered later (Wave A strays, 2026-06-11)

The three count-domain-owned simulation modes the original deferral named all
landed once the count-domain fit modes existed (`feat/wave-a-strays`, off
`main`). Decisions settled with Ben at that session's checkpoint: two-period
runs stay **in-memory only** (the NeXus writer keeps emitting one period);
count-mode covers **per-group single-histogram count synthesis** for the PR #41
fits.

- **Double-pulse simulation** — `simulate_double_pulse_run` (shipped with
  `count-domain-fit-modes` itself; round-trips the double-pulse single-histogram
  fit).
- **Count-mode simulation** — `simulate_count_run` (`core/simulate.py`): imprints
  the **same** `+a(t)` on every detector group as independent single-histogram
  counts (`N₀·e^{−t/τ}(1+a)+b`), the modern equivalent of WiMDA's `evfactor`
  non-FB branch. Round-trips through `fit_single_histogram` (recovers N₀, A,
  background; `tests/test_simulate.py::TestCountModeSimulation`). The α-free
  `fit_fb_alpha` is fed by the existing antisymmetric `simulate_run` (it needs
  the F/B pair), verified in the same class.
- **Two-period simulation** — `simulate_two_period_run` + `PeriodSpec`
  (`core/simulate.py`): per-period model/parameters/α/event-budget, sampled from
  one seeded generator. Emits the loader's two-period payload
  (`period_histograms`/`period_reduced`/`period_good_frames`/
  `period_dead_time_us`/`period_mode`) with `run.histograms` cloned from red, so
  `select_period`, the green∓red combination in `reduce_run_to_dataset`, and
  `degrade_run` all work; pulls are unit-normal (`TestTwoPeriodSimulation`).
- **Simulate dialog** — a single additive **Generation** combo (forward/backward
  asymmetry · count histograms · two-period red/green) plus a green-amplitude
  factor shown only for two-period (0 → a flat reference period, so G−R recovers
  the red signal).

## Still deferred (unchanged)

- **Deadtime-distortion injection** (simulate the non-paralyzable count loss,
  write real deadtimes, exercise the correction end-to-end) — re-confirmed
  deferred.
- **NeXus two-period writer** — a two-period synthetic run is in-memory only;
  Save-as-NeXus emits the red period (the writer's `histogram_data_1`), matching
  WiMDA. Promoting period support into the writer + loader round-trip is a future
  follow-on.
- **Event-mode simulation; sample-environment logs; PSI/ROOT writers.**

## Possible next follow-ons (new)

- More built-in instruments (a high-field octagon; a GPS-style multi-detector
  continuous instrument with up/down/left/right groups).
- Gallery presets for the dynamic/relaxation archetypes (dynamic Kubo–Toyabe
  hopping series; stretched-exponential glass).
- A pull-distribution batch mode that scans statistics levels and plots the
  pull width vs events (the "how much beam time do I need" planning curve).
