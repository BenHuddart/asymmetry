# WiMDA feature inventory

**Upstream:** `/Users/bhuddart/Source/WiMDA`  
**Author of original:** Francis L. Pratt (ISIS) — see F. L. Pratt,
*Physica B* 289-290, 710 (2000).  
**Language:** Object Pascal (Delphi) + supporting DLLs  
**Pre-existing catalogs reused:** `FEATURE_MAP.json`, `SYMBOL_MAP.json`,
`TEST_MAP.json`, `WiMDA.agent-manifest.json`. Slugs in those files are
quoted verbatim below where applicable.

WiMDA is the legacy Windows-era μSR analysis program from the ISIS
Muon Group. It is form-driven (TForm Pascal units) with significant
global state and DLL-based extensibility. Features below follow the
unified 10-category taxonomy used across all reference programs in
this comparison.

---

## 1. Data ingestion

- **`run-ingest`** — Dispatches NeXus, MUD, and PSI binary formats; parses
  instrument headers and populates run-level metadata (run number, title,
  periods, deadtime). Mutates module globals (`ALCmode`, `FileVals`,
  `prefix`).  
  Flagship: `src/muondata.pas`, `src/nexusunit.pas`, `src/mudunit.pas`.
  Optional NeXus / MUD compile-time switches.

- **Deadtime correction** — Embedded in the ingest path rather than as a
  separate algorithm; conventions vary by instrument format.  
  Flagship: `src/muondata.pas` (look for `Deadtime` references).

## 2. Asymmetry calculation

- **`asymmetry-fitting`** (data-preparation half) — Groups detector
  counts into forward / backward pairs, subtracts background, prepares
  time arrays for downstream fits. Background flags and group config
  live in form controls.  
  Flagship: `src/Analyse.pas`, `src/AsymFitFunction.pas`.

- **Period mapping (`period-mapping`)** — Reads `muonrun.periods`
  metadata and assigns detectors to red (forward) / green (backward)
  buckets, computing frame fractions for pulsed-source acquisitions.  
  Flagship: `src/PeriodMappingUnit.pas`.
  Implementation quirk: UI-driven; eight-period radio-button assumption
  baked into the form; mutates globals (`Redset`, `Greenset`,
  `UsingPeriods`).

## 3. Time-domain fitting

- **FITE engine** — Levenberg-Marquardt loop. Parameter bounds + tie
  logic stored inside `TAsymFitFunction` objects. Supports optional
  threading for long fits.  
  Flagship: `src/Fitucode.pas`.

- **`musr-function-registry`** — Dynamic loader for muon
  oscillation / relaxation theory functions exported by
  `musrfunctions.dll` (or user libraries). Supports both `stdcall` and
  Fortran-style exports. Library discovery mutates GUI control bounds.  
  Flagship: `src/MusrFunctionUnit.pas`, `src/Analyse.pas`
  (`LoadMusrfunctions`).
  Quirk: distinct from the FITE engine itself — the registry just
  exposes named callbacks the engine consumes.

- **Kubo–Toyabe relaxation** — Pre-computed integral tables for fast
  evaluation during fits. Static / dynamic KT variants accessed via the
  function registry.  
  Flagship: `src/KuboToyabe.pas`.

## 4. Multi-spectrum / global fitting

- **`multifit`** — Batch fitting across a sequence of runs (LF-sweep or
  delay-sweep). Sequential rather than simultaneous; loads runs, applies
  the shared model, advances. UI-centric (run list lives in form
  controls).  
  Flagship: `src/Multifit.pas`.
  Gap vs modern tools: no truly simultaneous global fit with linked
  parameters across runs.

## 5. Fourier / frequency-domain

- **`fft-spectrum`** — Zero-padded complex FFT (`ComplexFT`, `PowerFT`,
  `RealFT`, `CosFT`). Signal preprocessing (extrapolation, filtering)
  lives in `Plot.pas`, not in `Fourier.pas` itself.  
  Flagship: `src/Fourier.pas`.

- **`maxent-spectrum`** — Maximum-entropy reconstruction via Burg's
  method (`memcof`). Iterates pole count using the final-prediction-
  error (FPE) criterion; reconstructs the time-domain signal. Reads
  FFT parameters straight from UI controls (`FFTParams`); buffers
  allocated in finalisation.  
  Flagship: `src/MaxEnt.pas`, `src/Fourier.pas` (`CosFT`, `Four1`).
  **Notable:** WiMDA is one of only two reference programs that ships
  a working MaxEnt (the other is Mantid). musrfit's MaxEnt support is
  limited; Asymmetry's is a stub.

- **`spectrum background subtraction`** — Interactive tool for local
  background removal in frequency-domain plots.  
  Flagship: `src/SpecBG.pas`.

- **Eigenvalue-based spectral analysis** — Alternative frequency
  estimator via covariance eigendecomposition.  
  Flagship: `src/Eigen.pas`, `src/Eigenuni.pas`, `src/EigenTyps.pas`.
  Niche; rarely used in practice.

## 6. Parameter trending

- **`fit-table-processing`** — Sorts, rebins, and resamples fit tables
  for parameter-vs-T or parameter-vs-B trending plots. Tabular rows
  live in a `TRichEdit` widget; rebin re-enters run loading rather
  than transforming arrays in-place.  
  Flagship: `src/FitTableUnit.pas`, `src/Rebinning.pas`,
  `src/Resampling.pas`.

- **`model-fitting`** — Fits a parametric model (built-in or
  DLL-loaded) to user-supplied tabular data (x, y, error). Used to
  model already-extracted parameter trends.  
  Flagship: `src/Model.pas`, `src/fitfunctions.pas`,
  `src/FitTableUnit.pas`.

## 7. Logbook / multi-run management

- **Logbook unit** — Displays run headers, supports filtering by
  metadata, navigates prev/next runs in a session. Calls `ReadRun` for
  header extraction.  
  Flagship: `src/LogbookUnit.pas`, `src/BGform.pas`.

## 8. Visualisation

- **Plot panel** — Renders grouped histograms, asymmetry, model fits,
  FFTs, MaxEnt spectra, residuals as overlaid views.  
  Flagship: `src/Plot.pas`, `src/PlotPar.pas`, `src/PlotModel.pas`,
  `src/tlogplotunit.pas`.
  Quirk: `Plot.pas` combines preprocessing, FFT setup, MaxEnt
  invocation, and rendering — the seams between these stages are
  embedded inside the plot logic rather than separated into algorithms.

- **Moments analysis** — Computes m0, m1, m2 and derivatives (alpha,
  beta widths); parabolic peak extrapolation; averages across runs.  
  Flagship: `src/Moments.pas`.

- **Phase-table visualisation** — Associates MaxEnt spectral peaks
  with precession-phase angles; bidirectional sync with MaxEnt
  outputs.  
  Flagship: `src/PhaseTableUnit.pas`.

## 9. Project files / persistence

- **Session state** — Saved implicitly via form properties and
  registry entries (Windows registry). Recent file list and current
  run pointer tracked at the application level.  
  Flagship: `src/WiMDA_Main.pas`.
  Gap: no explicit JSON / XML serialisation layer; WiMDA cannot
  round-trip its full state to a portable project file.

## 10. Workflow utilities

- **Simulate mode** — Generates synthetic count histograms from a
  model + parameters. Form-driven; outputs to disk via
  `SaveSimulation` for validation and teaching.  
  Flagship: `src/Simulate.pas`.

- **`shared-state-runtime`** — Process-wide module globals (`groupd`,
  `groupr`, `timed`, `cgrp`, `nbin`, etc.) coupling features. Not a
  user-visible feature; a refactoring constraint for any port.

- **`legacy-peripheral-surface`** — Various small utilities (recent
  files, registry helpers, status-bar updates) catalogued under this
  slug in the FEATURE_MAP.

---

## Notable WiMDA-only depth

These features are richer in WiMDA than in the other reference
programs:

- **MaxEnt with Burg pole-scan and FPE criterion** — `src/MaxEnt.pas`.
  Mantid's MaxEnt is a different formulation; musrfit's is limited.
- **Simulate mode** — `src/Simulate.pas`. Mantid has no equivalent
  built-in tool; musrfit relies on user-written msr files.
- **Moments analysis** — `src/Moments.pas`. None of the other tools
  ship moments calculation.
- **Eigenvalue spectral estimator** — `src/Eigen.pas`. Niche but
  available.

## Obvious WiMDA gaps

- No truly simultaneous global fit (Multifit is sequential only).
- No batch processing framework / pipeline DSL.
- Limited metadata tagging — no per-run user annotations beyond what
  the data file provides.
- No formal uncertainty propagation to derived quantities.
- No model composition that combines built-in and user functions in a
  single expression (musrfit's FUNCTIONS block and Asymmetry's
  composite-model syntax are richer).
- Project state is not portable (Windows registry).

## Python port status

`python_port/` contains a partial WiMDA port with module structure
mirroring the Pascal units. Treat it as **prior art** for an
Asymmetry-side reimplementation, not as a complete reference: not
every Pascal unit has a corresponding Python module, and the port
itself documents the porting policy under
`docs/porting/README.md` (Asymmetry-side).
