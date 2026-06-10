# Implementation options — partitioning the WiMDA parity gap

This umbrella study's "implementation" is the **partition of the gap
inventory into independently-executable projects**. Per-feature
implementation options live in each project's own future study; this document
records how the portfolio was shaped and why.

## Option space considered

### Option A — partition by scientific theme only

One project per analysis area (reduction, fitting, frequency domain, …),
regardless of code surface. Cleanest narratives, but several themes converge
on the same GUI files (`grouping_dialog.py`, `fit_panel.py`,
`plot_panel.py`), which guarantees merge conflicts between parallel
worktrees.

### Option B — partition by module surface only

One project per touched module cluster. Maximises parallelism but splits
coherent science across projects (e.g. background-run subtraction lands in a
different project from background tail-fitting), making each study pass
shallower and verification harder.

### Option C — theme-first, surface-constrained (CHOSEN)

Group gaps by scientific coherence, then adjust boundaries so that projects
scheduled in the same wave touch disjoint modules; where coherence and
disjointness conflict, keep coherence and push the project to a later wave.
This is the partition in the README portfolio table.

Decision drivers:

- Execution model is **parallel Claude sessions in separate git worktrees**
  (decision 2026-06-10), so file overlap is the binding constraint, not
  reviewer bandwidth.
- Ben prefers **long sessions**: projects sized M target one long session;
  L projects are explicitly phased so each phase is one session with a
  shippable end state.
- Each project performs its own mandatory study pass before implementation,
  so briefs here stay at scope/direction altitude.

## Conflict analysis

Primary-surface map (files a project is expected to modify substantially):

| Project | Core surfaces | GUI surfaces |
|---|---|---|
| 1 data-reduction-parity | `transform/asymmetry.py`, `transform/background.py`, `transform/rebin.py`, `transform/grouping.py`, `io/periods.py` | `windows/grouping_dialog.py` |
| 2 run-arithmetic | `core/data/dataset.py`, new `core/data/combine.py` | `panels/data_browser.py` |
| 3 count-domain-fit-modes | `fitting/grouped_time_domain.py`, new `fitting/count_domain.py` | `windows/multi_group_fit_window.py`, light `panels/fit_panel.py` |
| 4 fit-workflow-diagnostics | `fitting/engine.py`, `fitting/result_summary.py` | `panels/fit_panel.py` |
| 5 frequency-domain-finishers | `fourier/fft.py`, new `fourier/burg.py`, `fourier/spectrum.py` | `panels/fourier_panel.py` |
| 6 maxent-completion | `maxent/engine.py` | `panels/maxent_panel.py` |
| 7 spectral-moments | new `fourier/moments.py` | small addition near `maxent_panel.py` |
| 8 simulate-mode | new `core/simulate.py` | new dialog, `mainwindow.py` hook |
| 9 model-function-parity | `fitting/parameter_models.py` | `panels/model_fit_dialog.py` |
| 10 rrf | new `transform/rrf.py` | `panels/plot_panel.py` |
| 11 workflow-visualisation | `io/` helpers, export helpers | `panels/plot_panel.py`, `panels/data_browser.py`, `mainwindow.py` |
| 12 python-user-functions | `fitting/composite.py` + `fitting/parameter_models.py` registries, new `core/plugins.py` | minor (picker refresh) |

Pairwise collisions and their resolution:

- **3 ↔ 4** (`fit_panel.py`): 3's panel changes are light (mode plumbing);
  4 owns the panel. Sequenced A → B.
- **6 ↔ 7** (`maxent_panel.py`): moments reads the finished spectrum; its
  panel hook is small but lands in the same file. Sequenced A → B.
- **9 ↔ 12** (`parameter_models.py` registry): 12 generalises the registries
  9 extends. Sequenced A → B.
- **10 ↔ 11** (`plot_panel.py`) and **2 ↔ 11** (`data_browser.py`): 11 is a
  basket of GUI touches by nature; it runs alone in Wave C.
- **Cross-cutting shared files** no partition removes: `mainwindow.py`
  (menu/toolbar hooks), `core/project/schema.py` (state persistence),
  `core/transform/__init__.py` / registry `__init__`s, `docs/porting/index.json`,
  user-guide toctrees. Rule for all projects: changes to these files must be
  small, additive, and placed at the end of the relevant block to keep merges
  trivial.

## Wave plan

- **Wave A**: 1, 3, 5, 6, 8, 9 — pairwise disjoint primary surfaces.
- **Wave B**: 2, 4, 7, 10, 12 — disjoint among themselves; each depends on a
  Wave A neighbour only through sequencing (not API): 4←3, 7←6, 12←9.
  2 and 10 could join Wave A if more parallel capacity is wanted — their only
  Wave A contact is none (2) and none (10); they are in B purely to bound
  simultaneous sessions.
- **Wave C**: 11 (GUI basket over files 2 and 10 also edit), plus
  `negative-muon-analysis` if promoted from deferred.

Hard dependencies (API, not just file overlap):

- `fit-workflow-diagnostics` optional phases (in-batch co-add, re-fit
  co-added selections) **depend on `run-arithmetic`** for count-level
  combination. Core scope (MINOS, χ² band, warm-start, abort) does not.
- `spectral-moments` consumes MaxEnt/FFT spectra as they exist today; no
  dependency on `maxent-completion` features (only the file-level sequencing
  noted above).
- `simulate-mode` reuses the grouping/asymmetry kernels read-only.

## Phasing of large projects

Phasing detail lives in each brief; the principle: each phase ends in a
shippable, tested state, ordered so the highest-value physics lands first
(e.g. maxent-completion ships the time-domain reconstruction overlay before
ZF/LF mode; data-reduction-parity ships alpha estimation + backgrounds before
binning modes).

## Process requirements carried into every project

- Study-first: full five-doc study under `docs/porting/<slug>/` before
  implementation (`docs/porting/README.md` workflow); these briefs seed the
  study but do not replace it.
- Core/GUI split: analysis behaviour in `asymmetry.core` first, GUI calls in
  (AGENTS.md invariant).
- GPL references (Mantid, musrfit) are verification oracles only — never
  vendored (MIT licence rule recorded in the maxent and time-integral
  studies).
- Physically-correct over WiMDA-literal where they conflict; every deliberate
  deviation documented in the project's comparison.md (precedent: LF-KT exact
  Hayano expression replacing WiMDA's 2Δ interpolation; positive-frequency
  muonium convention).
