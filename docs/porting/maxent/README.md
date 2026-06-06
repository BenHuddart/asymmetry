# Maximum Entropy Spectral Estimation Study

Status: study pass complete; implementation not started

This study records how Maximum Entropy (MaxEnt) spectral estimation is
implemented in WiMDA and Mantid, confirms its absence in musrfit, and compares
the candidate feature sets before porting MaxEnt into Asymmetry. No code is
written in this pass.

## Headline Findings

1. There are **four** distinct "maximum entropy" implementations across the
   reference programs, not three, and they are not all the same method:

   | # | Program | Where | What it actually is |
   |---|---|---|---|
   | 1 | WiMDA | `src/Wimdamax.pas` | Pratt/MULTIMAX joint multi-group MaxEnt on raw counts (Skilling–Bryan, 3 search directions, outer phase/amplitude/background/deadtime refinement) |
   | 2 | Mantid | `scripts/Muon/MaxentTools/` + `MuonMaxent.py` | **The same MULTIMAX lineage** — a numpy line-by-line port of the ISIS legacy FORTRAN; near-identical algorithm to #1 |
   | 3 | Mantid | `Framework/Algorithms/src/MaxEnt.cpp` | Generic Skilling–Bryan MaxEnt (2 search directions, SVD) on preprocessed data; PosNeg entropy and complex images |
   | 4 | WiMDA | `src/MaxEnt.pas` | **Not MaxEnt in the same sense at all** — Burg all-poles MEM (Numerical Recipes `memcof`/`evlmem`) with an FPE pole scan |

2. musrfit has **no** MaxEnt. Its own roadmap (`doc/musrfit.dox:68`) lists
   "add an interface to maxent" as a missing feature. Its entropy code
   (`PFTPhaseCorrection`) is an FFT *phase optimizer*, not spectral estimation.

3. Because #1 and #2 share one FORTRAN ancestor (the Southampton/Birmingham/
   St Andrews MULTIMAX program; F.L. Pratt, Physica B 289–290 (2000) 710),
   offering them as separate user-facing choices would be confusing. They are
   one algorithm to implement once — with Mantid's plain-numpy port available
   as an executable verification oracle (GPL-3, so an oracle only; Asymmetry
   is MIT and must not copy its code).

4. Both repo-local reference inventories contained errors about MaxEnt that
   this study corrects (see `comparison.md` §"Corrections To Prior
   Inventories").

## Scope

- WiMDA Pratt/MULTIMAX MaxEnt: algorithm, control dialog, outputs, moments
- WiMDA Burg MEM: documented as a *separate, out-of-scope* method
- Mantid `MuonMaxent` + `MaxentTools`: algorithm, properties, GUI tab, tests
- Mantid generic `MaxEnt-v1`: what it offers that the muon-specific path
  does not, and whether to expose it
- musrfit: verified absence; Fourier-only alternative characterized
- Feature-choice analysis: which behaviors must be a single chosen approach,
  which can be user-selectable options, and which should be excluded to avoid
  user confusion

## Study Files

- `comparison.md`: full cross-program comparison, feature matrix,
  choose-one vs offer-choice analysis, recommendation, and open research
  questions for the implementing agent
- `implementation-options.md`: candidate implementation strategies and the
  selected direction
- `test-data.md`: synthetic cases, reference datasets, and oracle strategy
- `verification-plan.md`: staged validation for the implementation pass

## Current Asymmetry Baseline

- Stub core entry point: `src/asymmetry/core/fourier/maxent.py` —
  `maxent(dataset, n_freq, f_max)` raises `NotImplementedError`.
- Reserved representation slot: `FrequencyMaxEnt`
  (`src/asymmetry/core/representation/frequency.py:85`,
  `RepresentationType.FREQ_MAXENT` in
  `src/asymmetry/core/representation/base.py:32`), already wired into the
  plot workspace view tokens
  (`src/asymmetry/gui/panels/plot_workspace_panel.py:20`).
- The representation project model deliberately leaves unimplemented
  MaxEnt representations uncomputed
  (`src/asymmetry/core/representation/project_model.py:232`).
- No MaxEnt tests exist yet.

Two baseline gaps matter for the implementation pass:

1. The current stub signature takes a processed `MuonDataset` (asymmetry),
   but the MULTIMAX algorithm consumes **raw grouped counts** with per-group
   statistics — the core seam will need access to grouped raw histograms,
   frames, t0, and good-bin ranges, not just asymmetry.
2. The reserved representation is frequency-domain; WiMDA's MaxEnt also
   produces a time-domain reconstruction overlay, which has no slot yet.

## Reference Lineage (for citation in later docs)

- J. Skilling and R.K. Bryan, Mon. Not. R. Astron. Soc. 211 (1984) 111 —
  the search-subspace algorithm both muon implementations use.
- F.L. Pratt, "WIMDA: a muon data analysis program for the Windows PC",
  Physica B 289–290 (2000) 710 — describes WiMDA's MaxEnt.
- T.M. Riseman and E.M. Forgan, Physica B 289–290 (2000) 718 (and companion
  papers) — the MULTIMAX/MaxEnt μSR methodology family. The implementing
  agent should verify the exact MULTIMAX provenance papers during its own
  research.
- A. Markvardsen, DPhil thesis (cited by Mantid `MaxEnt-v1.rst`) — basis of
  Mantid's generic MaxEnt.
