# Asymmetry roadmap

> **Note (2026-06-10):** the WiMDA-sourced entries of this roadmap are
> superseded by the umbrella gap study at
> [`docs/porting/wimda-parity-gap/`](wimda-parity-gap/README.md), which
> partitions all remaining WiMDA functionality into a 12-project parallel
> portfolio (absorbing `simulate-mode`, `moments-analysis`, `rrf-transform`,
> `python-user-functions`, `minos-error-analysis`, the WiMDA slice of
> `phase-auto-calibration`, and the `period-arithmetic` remainder).
> Non-WiMDA entries (`msr-import`, `lem-depth-profiling`,
> `structural-transitions`, `bpp-relaxation`) are unaffected.

This roadmap aggregates Phase 3 candidate scoring into a single
prioritised plan. It is the canonical "what next?" document for
Asymmetry maintainers.

The comparison work that feeds this roadmap lives at:

- `docs/porting/reference/{wimda,musrfit,mantid}/inventory.md` —
  per-program inventories
- `docs/porting/comparison-matrix.md` — unified feature matrix
- `docs/porting/candidates/<slug>/` — per-candidate study folders
  with motivation, comparison, and scoring

A public-facing distillation of this roadmap appears in
`docs/user_guide/comparison.rst` (the μSR-community summary).

## Methodology

Every candidate carries two scores (1-5 each):

- **Impact**: breadth of user benefit × pedagogical value ×
  alignment with Asymmetry's existing strengths. 1 = niche tool;
  5 = foundational, every workflow benefits.
- **Ease**: registry / API readiness × model complexity × GUI
  surface required × test-data availability. 1 = years of work;
  5 = days, mostly glue code.

The product `score = impact × ease` (range 1–25) drives the tier
assignment. Ties are broken by impact — when in doubt, prefer the
higher-impact candidate.

| Tier | Score range | Horizon | Action |
|---|---|---|---|
| **Now** | ≥ 15 | 0-4 months | Promote to a full study folder under `docs/porting/<slug>/`; add `index.json` entry with `"status": "study"`. |
| **Next** | 9 – 14 | 4-9 months | Keep in `docs/porting/candidates/<slug>/`; surface in the public chapter; not yet scheduled. |
| **Later** | ≤ 8 | 9-12+ months | Catalogue for visibility; revisit at roadmap refresh. |

## Ranked candidates

| Rank | Slug | Impact | Ease | Score | Tier |
|---:|---|---:|---:|---:|---|
| 1 | [minos-error-analysis](candidates/minos-error-analysis/) | 4 | 5 | 20 | Now |
| 2 | [dynamic-kubo-toyabe](candidates/dynamic-kubo-toyabe/) | 5 | 4 | 20 | Now |
| 3 | [theory-library-expansion](candidates/theory-library-expansion/) | 4 | 4 | 16 | Now |
| 4 | [simulate-mode](candidates/simulate-mode/) | 4 | 4 | 16 | Now |
| 5 | [maxent-spectrum](candidates/maxent-spectrum/) | 4 | 3 | 12 | Next |
| 6 | [rrf-transform](candidates/rrf-transform/) | 3 | 4 | 12 | Next |
| 7 | [python-user-functions](candidates/python-user-functions/) | 4 | 3 | 12 | Next |
| 8 | [moments-analysis](candidates/moments-analysis/) | 2 | 5 | 10 | Later |
| 9 | [bpp-relaxation](candidates/bpp-relaxation/) | 2 | 5 | 10 | Later |
| 10 | [phase-auto-calibration](candidates/phase-auto-calibration/) | 3 | 3 | 9 | Next |
| 11 | [period-arithmetic](candidates/period-arithmetic/) | 3 | 3 | 9 | Next |
| 12 | [msr-import](candidates/msr-import/) | 3 | 3 | 9 | Next |
| 13 | [muonium-radical-hyperfine](candidates/muonium-radical-hyperfine/) | 3 | 3 | 9 | Next |
| 14 | [alc-avoided-level-crossing](candidates/alc-avoided-level-crossing/) | 3 | 2 | 6 | Later |
| 15 | [structural-transitions](candidates/structural-transitions/) | 2 | 3 | 6 | Later |
| 16 | [lem-depth-profiling](candidates/lem-depth-profiling/) | 2 | 3 | 6 | Later |

## Now tier (top 4) — promote to study folders

These four candidates get full `docs/porting/<slug>/` study folders
(matching the existing `deadtime-correction`,
`fourier-transform`, `multi-group-time-domain-fitting` template) and
appear in `docs/porting/index.json` with `"status": "study"`. Each
is expected to ship within the next 4 months.

### 1. MINOS asymmetric error analysis (score: 20)

Expose iminuit's `Minuit.minos()` through the fit engine and panel
UI. Per-parameter `+err / -err` triples surfaced in the parameter
table. Highest-leverage roadmap item: existing dependency, one
method call, immediate quality improvement.

### 2. Dynamic Kubo–Toyabe (score: 20)

Strong-collision dynamic KT polarisation function. Time-step
convolution implementation in numpy; ~30 lines plus registry entry.
Closes the most commonly-flagged "missing model" against musrfit /
Mantid.

### 3. Theory library expansion (score: 16)

Umbrella for porting Keren, Abragam, Bessel, SpinGlass, Meier,
MuoniumDecouplingCurve, and SuperconductorVortexLattice (time-domain
form). One PR per function; each ~20–60 lines numpy plus a registry
entry.

### 4. Simulate mode (score: 16)

GUI dialog that synthesises a `MuonDataset` from a chosen model +
parameter values + noise level. Reuses the existing
`docs/screenshots/data/archetypes.py` synthesis helpers (promoted to
`core/simulate.py`).

## Next tier (5-11)

These candidates stay in `docs/porting/candidates/<slug>/` until
roadmap refresh. They are flagged for public visibility in the
comparison chapter but are not yet scheduled.

- **MaxEnt spectrum (12)** — now studied in full at
  `docs/porting/maxent/` (slug `maxent`). The study reverses the
  earlier Burg-first plan: implement the MULTIMAX-lineage joint
  MaxEnt once (WiMDA `Wimdamax.pas` contract, Mantid `MaxentTools`
  as oracle); Burg MEM is a different method and is excluded.
  Heaviest algorithm in this roadmap.
- **RRF transform (12)** — rotating reference-frame demodulation
  algorithm. Useful for vortex-lattice and high-TF studies; ~30
  lines numpy plus a low-pass.
- **Python user functions (12)** — decorator-based plugin API.
  Lower-friction analogue of musrfit's C++ plugins. Best landed
  after theory-library-expansion so the user-facing example library
  is rich.
- **Phase auto-calibration (9)** — fit-driven per-detector phase
  estimation. Re-uses the multi-group fit machinery; a quality-of-
  life win after MINOS / dynamic-KT.
- **Period arithmetic (9)** — pulsed-source multi-period support
  in the IO layer + data browser combo. Critical for ISIS users.
- **`.msr` import (9)** — read musrfit `.msr` files. Cross-tool
  interop. Best after theory-library-expansion lands so imports
  don't silently lose functionality.
- **Muonium-radical hyperfine (9)** — port Mantid's
  `*Muonium*` fit functions and decoupling-curve parametric
  model. Best after theory-library-expansion lands.

## Later tier (12-)

Catalogued for visibility; revisit at the next roadmap refresh.

- **Moments analysis (10)** — quick lineshape characterisation
  in the Fourier panel. Low impact but trivial to ship — promote
  opportunistically.
- **BPP relaxation (10)** — single-page parametric model for
  muon-diffusion analysis; trivial to ship.
- **ALC interface (6)** — full Avoided Level Crossing workflow.
  Heaviest GUI surface in this roadmap; closes Mantid exclusivity
  but takes 1500-3000 lines of code.
- **Structural-transition toolkit (6)** — kink-finder helpers in
  the parameter-trending panel for non-magnetic phase transitions.
- **LEM depth profiling (6)** — implantation-energy → depth
  conversion plus a depth-axis trending mode. Narrow audience but
  high scientific leverage where it applies.

## What's deliberately not on the roadmap

- **Mantid elemental analysis** (μ-XRF) — out of Asymmetry's
  current scientific scope.
- **WiMDA's eigenvalue spectral estimator** — niche; MaxEnt
  covers the resolution-improvement use case.
- **musrfit's `musrview` ROOT-canvas export** — Asymmetry's GLE
  export is already publication-quality.
- **musrfit's `musrWiz` / `musrStep` editors** — Asymmetry's
  composite-model expression syntax + Fit Wizard cover the same
  workflow more ergonomically.
- **Mantid's MVP boilerplate** — Asymmetry's PySide6 single-process
  GUI is cleaner.

## Refresh policy

This roadmap is point-in-time. Suggested refresh cadence:

- **Quarterly**: re-score top-of-tier candidates against actual
  progress. Demote anything that turned out harder than expected;
  promote anything that became easier.
- **Annually**: re-run Phase 1 (per-program inventories) to capture
  upstream changes in WiMDA / musrfit / Mantid. Update the
  comparison matrix accordingly.

When a candidate is promoted from Next to Now, it gets a full
study folder at `docs/porting/<slug>/` and an `index.json` entry.
When a Now-tier feature ships, its `index.json` entry moves to
`"status": "implemented"`.
