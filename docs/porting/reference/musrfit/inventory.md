# musrfit feature inventory

**Upstream:** `/Users/bhuddart/Source/musrfit`  
**Authors of original:** Andreas Suter, Bastian Wojek (PSI)  
**Language:** C++ on top of ROOT + Minuit2; Qt5/Qt6 for editors  
**Pre-existing catalog reused:** `FEATURE_MAP.json` (slugs:
`msr_handler`, `run_data_io`, `data_conversion`, `fitting_engine`,
`theory_functions`, `fourier_transform`, `visualization_canvas`,
`ui_tools`, `musrroot_validation`). Slugs quoted verbatim below.

musrfit is the PSI open-source μSR analysis framework. Its
distinguishing feature is the `.msr` workflow file — a hand-editable
plain-text artifact that captures the complete fit setup. The viewer
(`musrview`), editor (`musredit`), and fit engine (`musrfit`) are
separate processes communicating via files. Features below follow the
unified 10-category taxonomy.

---

## 1. Data ingestion

- **`run_data_io`** — Run loaders for PSI-BIN, MUSR ROOT, NeXus/HDF5,
  MUD, and WKM formats. Auto-detects format via file signature.
  Handles histogram extraction, deadtime, detector grouping at load
  time.  
  Flagship: `src/classes/PRunDataHandler.cpp`.
  Each reader is a separate method; NeXus branch is compile-time
  optional.

- **`musrroot_validation`** — Introspects ROOT TFile structure and
  validates against `MusrRoot.xsd`; standalone CLI utility.  
  Flagship: `src/musrRootValidation.cpp`. Low porting priority.

- **`data_conversion`** — Format conversion utilities under
  `src/classes/PMsr2Data.cpp` and the `msr2data` CLI.

## 2. Asymmetry calculation

- **Forward/Backward grouping + α/β** — Implemented inside the
  per-run-type classes rather than a separate asymmetry module.
  Validates α and β presence in the RUN block, computes asymmetry
  from paired detector histograms with packing-aware binning.  
  Flagship: `src/classes/PRunAsymmetry.cpp`,
  `src/classes/PRunAsymmetryBNMR.cpp`,
  `src/classes/PRunAsymmetryRRF.cpp` (RRF variant — see
  `rrf-transform` candidate).

## 3. Time-domain fitting

- **`fitting_engine`** — Wraps Minuit2 with MIGRAD (gradient descent),
  MINOS (asymmetric error analysis), HESSE (Hessian),
  SCAN / CONTOUR (profile likelihoods). Objective is χ² or maximum
  likelihood. Driven by the COMMANDS block in the `.msr` file.
  OpenMP parallelisation is an optional compile-time switch.  
  Flagship: `src/classes/PFitter.cpp`, `src/classes/PFitterFcn.cpp`.

- **`theory_functions`** — Built-in library of ~34 theory functions
  covering exponentials, Gaussians, static/dynamic KT variants,
  precession (cosine, Bessel), superconductor vortex lattice, spin
  glass, Abragam, mu-minus, polynomial baselines, more.  
  Flagship: `src/classes/PTheory.cpp`, `src/include/PTheory.h`.
  Combine via `+` (independent additive channels) and `*` (cascaded
  multiplicative effects).
  **Notable:** the breadth of this library is musrfit's primary
  pedagogical strength relative to the other tools.

- **User functions** — Dynamic C++ plugin: users inherit from
  `PUserFcnBase`, compile to a `.so` / `.dll`, load via ROOT's plugin
  system. Signature is
  `operator()(Double_t t, const std::vector<Double_t>& par)`.  
  Flagship: `src/tests/userFcn/`.
  High friction (requires C++ + ROOT toolchain) but extremely
  flexible.

## 4. Multi-spectrum / global fitting

- **Batch fits with shared parameters** — `PFitter` runs a single
  parameter set across multiple RUN blocks of any type (asymmetry,
  single histogram, mu-minus). Sharing is achieved by composing the
  run objects against a shared `PMusrParamList`. No separate "global
  fit" class.

## 5. Fourier / frequency-domain

- **`fourier_transform`** — FFTW3-backed forward real FFT with Hann
  and Kaiser apodisation, zero-padding, and Minuit2-driven phase
  optimisation (`PFTPhaseCorrection::Minimize`). Returns magnitude
  and power spectra; complex output available.  
  Flagship: `src/classes/PFourier.cpp`, `src/classes/PPrepFourier.cpp`.

- **`visualization_canvas`** — `musrview` (ROOT canvas) integrates FFT
  computation interactively. `PMusrCanvas::HandleFourier` and
  `HandleDifferenceFourier` allow real-time phase sweeps and overlay
  modes.  
  Flagship: `src/classes/PMusrCanvas.cpp`.

- **MaxEnt** — Limited; musrfit ships a Cython-style entropy
  computation in some utility paths but does not expose a full-feature
  MaxEnt UI comparable to WiMDA or Mantid.

## 6. Parameter trending

- **`data_conversion` / `msr2data`** — Template-based batch
  generation of `.msr` files and run-series fit-parameter extraction.
  Given a template `.msr` and a run-number list, expands the template,
  iterates, collects fit parameters into tabular output. Implements
  parameter-vs-field / parameter-vs-temperature trending.  
  Flagship: `src/classes/PMsr2Data.cpp`, CLI in `src/msr2data.cpp`.
  Procedural rather than interactive; users edit templates and invoke
  `msr2data` from the shell.

## 7. Logbook / multi-run management

- **Run lists** — `PRunListCollection` and
  `PRunDataHandler::ReadFilesMsr` manage lists of runs declared in
  `.msr` files or external text. No integrated logbook GUI; multi-run
  setups are described via the COMMANDS block of the `.msr` file or
  via wrappers around `msr2data`.

## 8. Visualisation

- **`visualization_canvas`** — `musrview` is a separate ROOT-based
  viewer process. Displays data, theory overlay, FFT power, residuals,
  difference plots. PNG export for publications. Menu-driven canvas
  interaction.  
  Flagship: `src/classes/PMusrCanvas.cpp`, `src/musrview.cpp`.

- **`ui_tools`** — `musredit` (Qt5/Qt6 text editor for `.msr` files),
  `musrWiz` (parameter-setup wizard), `musrStep` (theory-selection
  walkthrough), `mupp` (post-fit parameter editor).  
  Flagship: `src/musredit_qt6/`, `src/musrgui/`.
  The UI shells are thin; core logic remains in `src/classes/`.

## 9. Project files / persistence

- **`msr_handler`** — `.msr` is the canonical user artifact encoding
  RUN, THEORY, FUNCTIONS, COMMANDS, PLOT, STATISTIC, GLOBAL blocks.
  Block-by-block parsing with error recovery; comments and line order
  preserved on round-trip.  
  Flagship: `src/classes/PMsrHandler.cpp`,
  `src/include/PMsrHandler.h`.
  **Notable:** the `.msr` file is musrfit's defining design choice.
  No other tool's project format is hand-editable to the same degree.

- **`msr2msr`** — Forward-converter for old `.msr` versions to the
  current schema.  
  Flagship: `src/msr2msr.cpp`.

## 10. Workflow utilities

- **FUNCTIONS block** — Inline arithmetic / transcendental
  expressions inside the `.msr` file, parsed by a Boost.Spirit
  recursive-descent grammar. Allows composing fit models without
  writing C++. Less expressive than Asymmetry's composite-model
  expression syntax.  
  Flagship: `src/classes/PFunctionHandler.cpp`,
  `src/include/PFunctionGrammar.h`, `src/include/PFunctionAst.h`.

- **`musrroot_validation`** — see §1.

---

## Notable musrfit-only depth

- **Theory function library**: ~34 functions (most of any reference
  program); covers superconductor vortex lattice, dynamic KT,
  Abragam, Bessel oscillation, mu-minus.
- **MIGRAD/MINOS/HESSE/SCAN/CONTOUR**: full Minuit2 surface exposed,
  including asymmetric MINOS errors and likelihood profiles. Asymmetry
  exposes only MIGRAD-style fits via iminuit.
- **`.msr` file**: hand-editable workflow artifact — strong
  reproducibility property no other tool replicates.
- **`musrview`**: publication-quality ROOT canvas exports.
- **Phase optimisation via Minuit2 inside the Fourier path**: a
  separately-optimised phase, not just an interactive slider.

## Obvious musrfit gaps

- **Interactive parameter trending**: `msr2data` is batch / CLI.
  Asymmetry's parameter-trending GUI is materially richer.
- **User-function friction**: C++ plugin loading is harder than
  Asymmetry's Python decorator-style composite expressions.
- **GUI fragmentation**: separate processes for `musrview`,
  `musredit`, `musrfit` itself, `musrWiz`, `musrStep`, `mupp`. Less
  cohesive than the Asymmetry single-window experience.
- **Logbook**: no integrated multi-run manager.
- **MaxEnt**: limited compared to WiMDA / Mantid.
