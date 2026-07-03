# Comparison matrix: WiMDA × musrfit × Mantid × Asymmetry

> **Note (2026-06-10):** the WiMDA and Asymmetry columns of this matrix are
> superseded by the full-source sweep in
> [`docs/porting/wimda-parity-gap/comparison.md`](wimda-parity-gap/comparison.md)
> (umbrella gap study). Consult that document for current WiMDA-vs-Asymmetry
> status; this matrix remains authoritative only for the musrfit and Mantid
> columns.

This document is the canonical side-by-side feature matrix across the
three reference programs and Asymmetry. It derives from the
per-program inventories under
`docs/porting/reference/{wimda,musrfit,mantid}/inventory.md` and from
a fresh scan of `src/asymmetry/` for the Asymmetry column.

## Symbols

- ✅ **Present** — feature exists and is usable.
- ◐ **Partial** — feature is implemented but materially less rich than
  the strongest reference, OR exists as infrastructure / stub.
- ❌ **Absent** — feature does not exist.
- ★ **Distinctive strength** — implementation is materially richer or
  more ergonomic than the alternatives.

## Categories

The ten categories below are the canonical taxonomy used across this
roadmap. Each subsection introduces the category, then carries a
feature-vs-program table. Rows that are clearly Asymmetry-only (no
analogue in the reference programs) are deferred to the closing
"Asymmetry-only innovations" section.

---

### 1. Data ingestion

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| ISIS NeXus (current) | ✅ `nexusunit.pas` | ✅ `PRunDataHandler.cpp` (compile-time optional) | ✅ `LoadMuonNexus`/`v2`/`v3` | ✅ `core/io/nexus.py` |
| PSI BIN | ❌ | ✅ `PRunDataHandler.cpp` | ✅ `LoadPSIMuonBin` | ✅ `core/io/psi.py` |
| MUSR ROOT | ❌ | ✅ `PRunDataHandler.cpp` | ❌ (TRIUMF MUD instead) | ✅ `core/io/root.py` |
| MUD (TRIUMF) | ✅ `mudunit.pas` | ✅ `PRunDataHandler.cpp` | ✅ `LoadMUD` | ❌ |
| WKM | ❌ | ✅ `PRunDataHandler.cpp` | ❌ | ❌ |
| Auto format detection | ◐ extension-based | ✅ signature-based | ✅ algorithm dispatcher | ◐ via per-loader probe |
| Sample-log loading | ◐ embedded in run | ◐ via metadata | ✅ `LoadMuonLog` | ◐ via `MuonDataset.metadata` |
| Period handling | ◐ `period-mapping` (UI-driven) | ❌ (single-pulse focus) | ★ `SummedPeriodSet` / `SubtractedPeriodSet` | ❌ |
| Deadtime as first-class | ◐ embedded in ingest | ◐ in RUN block | ✅ `ApplyDeadTimeCorr`, `CalMuonDeadTime` | ✅ `core/transform/deadtime.py` |

### 2. Asymmetry calculation

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Forward / backward grouping | ✅ `Analyse.pas` | ✅ `PRunAsymmetry.cpp` | ✅ `MuonPairingAsymmetry` | ✅ `core/transform/asymmetry.py`, `grouping.py` |
| α estimation | ◐ form-driven | ◐ in `.msr` RUN | ✅ `AlphaCalc`, `EstimateMuonAsymmetryFromCounts` | ◐ `core/transform/grouping.py` (manual entry) |
| β (asymmetric grouping) | ◐ form | ✅ `PRunAsymmetry.cpp` | ◐ via group pair α | ❌ |
| Auto phase calibration | ❌ | ❌ | ★ `CalMuonDetectorPhases` | ❌ |
| Background subtraction | ◐ embedded | ◐ via theory | ✅ `RemoveExpDecay`, `PSIBackgroundSubtraction` | ✅ `core/transform/background.py` |
| Rotating Reference Frame | ❌ | ◐ `PRunAsymmetryRRF.cpp` | ★ `RRFMuon` | ❌ |

### 3. Time-domain fitting

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Fit engine | ✅ FITE (LM) | ★ Minuit2 (MIGRAD / MINOS / HESSE / SCAN / CONTOUR) | ✅ general `Fit` algorithm | ◐ iminuit (Migrad only) |
| Asymmetric error analysis | ❌ | ★ MINOS | ◐ via Mantid `Fit` errors output | ❌ |
| Number of theory functions | ◐ ~12 via registry | ★ ~34 built-in | ★ ~15 muon + general lib | ◐ 6 MODELS + 11 components |
| Static Kubo–Toyabe (ZF) | ✅ `KuboToyabe.pas` | ✅ in `PTheory` | ✅ `StaticKuboToyabe` | ✅ `StaticGKT_ZF` |
| LF Kubo–Toyabe | ✅ via registry | ✅ in `PTheory` | ✅ `StaticKuboToyabe` (LF parameter) | ✅ `LFKuboToyabe` |
| Dynamic Kubo–Toyabe | ✅ via registry | ✅ in `PTheory` | ★ `DynamicKuboToyabe` (strong + weak collision) | ❌ |
| Keren | ❌ | ❌ | ★ `Keren` | ❌ |
| Meier (exchange-coupled) | ❌ | ❌ | ★ `Meier` | ❌ |
| Muonium decoupling curve | ❌ | ◐ (composable from Bessel etc.) | ★ `MuoniumDecouplingCurve` | ❌ |
| Muonium TF / ZF specialised | ❌ | ◐ via theory | ★ `HighTFMuonium`, `LowTFMuonium`, `TFMuonium`, `ZFMuonium` | ❌ |
| Superconductor vortex lattice | ❌ | ✅ in `PTheory` | ◐ via composite | ◐ parameter-domain only (`SC_TwoGap_SS`) |
| Abragam | ❌ | ✅ in `PTheory` | ❌ | ❌ |
| Bessel oscillation | ❌ | ✅ in `PTheory` | ❌ | ❌ |
| Muon F (μ−F nuclear coupling) | ❌ | ❌ | ★ `MuonFInteraction` | ✅ `MuF`, `FmuF_Linear`, `FmuF_General` |
| Stretched exponential | ✅ | ✅ | ✅ `StretchExpMuon` | ✅ |
| User-defined functions | ◐ DLL registry | ◐ C++ plugin (high friction) | ◐ via plugin or Python | ◐ via composite-expression syntax |
| Composite expression syntax | ❌ | ◐ FUNCTIONS block (limited) | ◐ `CompositeFunction` (procedural) | ★ free-form arithmetic + fraction groups |

### 4. Multi-spectrum / global fitting

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Sequential batch fit | ✅ `multifit` (LF / delay sweep) | ✅ via `msr2data` | ✅ `MuonSequentialFitDialog` | ✅ Fit Wizard / Global tab |
| Simultaneous global fit (shared params) | ❌ | ★ via shared `PMusrParamList` | ✅ "Simultaneous fit" mode | ✅ Global tab |
| Per-group nuisance + shared physics | ❌ | ✅ | ✅ Composite mapping | ✅ Multi-Group Fit window |
| Shared-parameter visualisation | ◐ table view | ◐ in `.msr` | ✅ Results Tab | ★ Parameter-classification UI |

### 5. Fourier / frequency-domain

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Real FFT | ✅ `Fourier.pas` | ✅ FFTW3 in `PFourier.cpp` | ✅ `FFT` algorithm | ✅ `core/fourier/fft.py` |
| Apodisation (Hann / Gauss / Lorentz) | ✅ | ✅ Hann, Kaiser | ✅ `PaddingAndApodization` (Lorentz, Gauss) | ✅ `core/fourier/window.py` |
| Zero-padding | ✅ | ✅ | ✅ | ✅ |
| Phase optimisation | ◐ manual + phase table | ★ Minuit2-driven (`PFTPhaseCorrection`) | ✅ via MaxEnt phases | ◐ manual + auto entropy mode |
| Group-resolved spectra | ◐ via plot | ◐ via canvas | ✅ as workspace groups | ✅ `core/fourier/grouped.py` |
| MaxEnt | ★ MULTIMAX joint MaxEnt (`Wimdamax.pas`); also a separate Burg MEM (`MaxEnt.pas`) | ❌ none (roadmap item only) | ★ `MuonMaxent` (same MULTIMAX lineage) + generic `MaxEnt-v1` | ◐ stub in `core/fourier/maxent.py`; study at `docs/porting/maxent/` |
| Eigenvalue spectral estimator | ✅ `Eigen.pas` | ❌ | ❌ | ❌ |

### 6. Parameter trending

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Post-fit parameter scan (T / B) | ◐ `fit-table-processing` (text-table) | ◐ `msr2data` (template-based CLI) | ◐ `PlotAsymmetryByLogValue` | ★ interactive trending panel |
| Parametric model fit to extracted parameters | ✅ `Model.pas` + DLL registry | ◐ requires custom workflow | ◐ via Mantid Fit on table | ★ Parameter-domain models (`SC_*`, `Lambda_bg`, etc.) |
| Avoided Level Crossing (ALC) workflow | ❌ | ❌ | ★ ALC interface (data load → baseline → peak fit, MVP) | ❌ |
| Results table (sortable, exportable) | ◐ TRichEdit table | ❌ | ✅ Results Tab | ◐ via Fit Parameters dock |

### 7. Logbook / multi-run management

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Multi-run table | ✅ `LogbookUnit.pas` | ◐ via run lists in `.msr` | ✅ "Loaded data" panel | ★ Data Browser with sort + filter + groups |
| Filter on column | ◐ basic | ❌ | ✅ | ★ Excel-style column filter dialog |
| Group runs by metadata | ❌ | ❌ | ◐ workspace groups | ✅ data groups (coadd) |
| Per-run annotations | ❌ | ◐ COMMENT lines in `.msr` | ❌ | ✅ via project file metadata |

### 8. Visualisation

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Time-domain plot | ✅ `Plot.pas` | ✅ `musrview` (ROOT canvas) | ✅ matplotlib in Workbench | ✅ `gui/panels/plot_panel.py` |
| Frequency-domain plot | ✅ `Plot.pas` | ✅ `musrview` | ✅ separate FDA interface | ✅ frequency-domain workspace |
| Residual / difference plot | ✅ | ✅ | ✅ | ✅ |
| Overlay multiple runs | ✅ | ✅ | ✅ | ✅ |
| Publication-quality export | ◐ PNG via Pascal canvas | ★ ROOT-native PDF / EPS / PNG | ◐ matplotlib export | ★ GLE-native PDF / EPS export |
| Moments analysis | ★ `Moments.pas` | ❌ | ❌ | ❌ |

### 9. Project files / persistence

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Portable project file | ❌ (Windows registry) | ★ `.msr` (hand-editable, plain text) | ✅ `.mantid` (HDF5, binary) | ✅ `.asymp` (JSON, schema-versioned) |
| Round-trip fidelity | ◐ session state only | ★ comments / line order preserved | ✅ full ADS state | ✅ schema-versioned JSON |
| Recent files | ✅ | ◐ via editor | ✅ | ✅ |
| Format conversion | ❌ | ✅ `msr2msr` (forward versioning) | ❌ | ◐ schema migration in `core/project` |
| Hand-editable | ❌ | ★ `.msr` | ❌ | ◐ JSON technically yes, but not designed for it |

### 10. Workflow utilities

| Feature | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Pipeline orchestration | ◐ form-driven (event handlers) | ◐ COMMANDS block in `.msr` | ★ `MuonProcess`, `MuonPreProcess` workflow algorithms | ◐ implicit via GUI signals |
| Synthetic data simulation | ★ `Simulate.pas` | ❌ (users write `.msr` by hand) | ❌ | ❌ |
| Inline arithmetic expressions | ❌ | ✅ FUNCTIONS block (Boost.Spirit) | ❌ | ★ composite-model expressions |
| Model-recommendation wizard | ❌ | ❌ | ❌ | ★ Fit Wizard (AICc / BIC ranking) |
| Plugin extensibility | ✅ DLL (musrfunctions / DLLs) | ◐ C++ plugins (high friction) | ◐ via Mantid algorithm framework | ◐ Python (no formal user-plugin API yet) |
| μ-XRF (negative-muon elemental analysis) | ❌ | ❌ | ✅ Elemental Analysis interface | ❌ (out of scope) |

---

## Asymmetry-only innovations

Features that are richer in Asymmetry than in any of the three
reference programs:

- **Fit Wizard** (`gui/windows/fit_wizard_window.py`,
  `core/fitting/fit_wizard.py`) — AICc / BIC-ranked
  model-recommendation portfolio over the entire MODELS registry
  given a single dataset. No analogue in WiMDA / musrfit / Mantid.

- **Composite-model expression syntax** (`core/fitting/composite.py`)
  — Free-form arithmetic over component names with fraction
  groups (`(...){frac}`). More expressive than musrfit's FUNCTIONS
  block; not available in WiMDA or Mantid.

- **Interactive parameter trending panel**
  (`gui/panels/fit_parameters_panel.py`) — Visualises
  per-run-fit parameters across a series, with parametric model
  fitting (e.g. `SC_TwoGap_SS`) directly in the same GUI.
  Mantid's ALC is more specialised; musrfit's `msr2data` is
  CLI / batch.

- **Schema-versioned project files** (`.asymp` JSON;
  `core/project/`) — Forward-compatible state serialisation with
  documented schema migrations. Mantid's `.mantid` files are HDF5
  binary; `.msr` files lack a versioned schema.

- **Modern PySide6 + matplotlib stack** — single-process GUI
  rather than musrfit's three-process model or Mantid's heavy
  C++/Python hybrid. Lower install friction; easier contributor
  ramp.

## Asymmetry's main gaps

Where the reference programs are materially richer:

- **Theory-function library breadth** — Asymmetry has ~17 model
  components total; musrfit has ~34; Mantid has ~15 specialised
  muon functions including `Keren`, `Meier`, `MuonFInteraction`,
  and four `Muonium*` variants.
- **Dynamic Kubo–Toyabe** — present in WiMDA, musrfit, and
  (especially) Mantid; absent in Asymmetry.
- **MaxEnt** — production implementations in WiMDA (`Wimdamax.pas`,
  Pratt/MULTIMAX) and Mantid (`MuonMaxent`, same lineage); musrfit has
  none; Asymmetry has a placeholder stub. WiMDA's `MaxEnt.pas` is a
  separate Burg all-poles MEM, not the same method. Full study:
  `docs/porting/maxent/`.
- **ALC interface** — Mantid only.
- **Rotating Reference Frame** — Mantid algorithm; musrfit has a
  partial implementation inside the asymmetry classes.
- **Automatic phase calibration** — Mantid's
  `CalMuonDetectorPhases` only.
- **Period arithmetic** — Mantid only; needed for ISIS pulsed
  beams.
- **Simulate mode** — WiMDA only.
- **Moments analysis** — WiMDA only.
- **Phase optimisation via numerical minimisation** —
  musrfit's `PFTPhaseCorrection`; Asymmetry has auto-phase via
  entropy but not Minuit-driven optimisation.
- **MINOS / asymmetric error analysis** — musrfit only; iminuit
  supports this but Asymmetry doesn't expose it.

## Reduction numerics — source-audited conventions (2026-07)

A dedicated four-source correctness audit (WiMDA Pascal, musrfit C++,
Mantid C++/Python, Asymmetry) compared the *numerical conventions* of the
raw-data reduction, beyond the feature level of the tables above. The
verified facts, with the load-bearing source locations:

### Corrected time axis and t0

| Convention | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| t0 source (PSI BIN) | per-histogram header int (`muondata.pas` `nt0`) | per-histogram header int; float `realT0` parsed but unused (`MuSR_td_PSI_bin.cpp`) | prefers float `realT0`, else int; collapses to **max** across slots (`LoadPSIMuonBin.cpp:186–234`) | per-histogram header int (`core/io/psi.py`) |
| Fractional t0 | no (integer bins) | fractional t0 shifts time labels only; counts never resampled (`PRunAsymmetry.cpp:1143`) | yes (bin-edge shift) | no (integer bins) |
| Multi-detector alignment | each detector read at its own t0 (`Group.pas:1498`) | integer shift onto group leader's t0 (`PRunAsymmetry.cpp:801–812`) | single max-t0 shift for all spectra (per-detector table optional, unused by the GUI) | integer shift onto **max** member t0 (`transform/grouping.py:56–105`) |
| Bin time stamp | packed-bin centre (`Group.pas:1584`) | `(fgb−0.5) + pack/2 − t0` centre convention (`PRunAsymmetry.cpp:1143–1145`) | bin edges | `(k − t0)·Δt` left-edge stamps, mean under packing |

Numerically the WiMDA and Asymmetry axes coincide exactly on PSI GPS
headers; musrfit's centring convention differs by ≤ half a raw bin
(labels only).

### First / last good data

- **WiMDA**: single global offset from **histogram 1 only**
  (`tgood_beg[1] − tzero[1]`, `Group.pas:1700`).
- **musrfit**: PSI header first/last-good are parsed but **never consulted
  by any fit path** — the range comes from the `.msr` file or a
  `t0 + 10 ns` fallback (`PRunAsymmetry.cpp:2028–2078`); forward/backward
  ranges are forced to equal time-since-t0 by shifting one side.
- **Mantid**: `max(firstGood[0..15])` but `lastGood[0]` only
  (`LoadPSIMuonBin.cpp:183–222`); applied as a crop by the GUI, not the
  loader.
- **Asymmetry**: intersection across the grouped detectors — max
  first-good offset, min last-good offset (`core/io/psi.py:886–889`) —
  the most conservative of the four. All four drop (not mask) pre-window
  data.

### Count errors and the F−B asymmetry error

- All four assign Poisson √N at the count level and use an error ≈ 1
  sentinel/floor for empty or degenerate bins (WiMDA additionally
  truncates the display at the first zero-count packed bin,
  `Plot.pas:1642`).
- **Exact-Poisson σ_A** `2|α|·√(F·B·(F+B))/(F+αB)²` is shared by WiMDA
  (`Analyse.pas:846–849`, algebraically identical with a `+1`
  low-count regularisation: variance `(1+N)` per group), musrfit
  (`PRunAsymmetry.cpp:1160`, α/β applied to the *theory* so the data
  error omits α — an approximation on their side), and Asymmetry
  (`transform/asymmetry.py`).
- **Mantid is the outlier**: `AsymmetryCalc` uses the
  independent-propagation `√((F+α²B)(1+A²))/(F+αB)` over-estimate *and*
  discards the workspace error arrays entirely, so upstream corrections
  never reach its pair-asymmetry errors (`AsymmetryCalc.cpp:130–156`).
- **Packing order**: all four form the binned asymmetry from **summed
  counts** (counts-then-ratio). WiMDA `Group.pas:1443–1515`; musrfit
  `PRunAsymmetry.cpp:1079` ("first rebin the data, than calculate the
  asymmetry"); Mantid rebins counts in `MuonPreProcess` before
  grouping/asymmetry. Asymmetry's fixed bunching originally rebinned the
  *asymmetry* (value-domain quadrature) and was unified onto the
  counts-first order in `binned_fb_asymmetry`
  (`transform/rebin.py`) — the value-domain path inflated merged error
  bars wherever one-sided raw bins contributed σ = 1 sentinels.
  `rebin()` remains the curve-level combiner for histogram-less data.

### Deadtime

- Same non-paralyzable formula in all four:
  `N/(1 − N·τ/(Δt·good_frames))` (WiMDA `Group.pas ccorrect`; musrfit
  `PRunBase.cpp:212`; Mantid `ApplyDeadTimeCorr.cpp:83`; Asymmetry
  `transform/deadtime.py`), applied per detector before grouping.
- For PSI BIN files it is effectively **off in every program** (the
  header carries no deadtime): WiMDA scales by `framestotal = 1`;
  musrfit's file-mode gate fails; Mantid synthesises an all-zero table
  (and mis-sets `goodfrm` to the *bin count* for PSI,
  `LoadPSIMuonBin.cpp:521`).
- **None of the four rescales errors after deadtime correction**:
  Asymmetry takes √(N_corr), Mantid and musrfit keep the stale
  √(N_raw), WiMDA mixes raw variance with the corrected denominator
  (its errors *shrink* under correction). All are approximations of the
  correct `c·√N`.

### Background

- Asymmetry's `range` mode is musrfit's pre-t0 window (same
  `0.1·t0–0.6·t0` default and beam-period snapping constants,
  `PRunAsymmetry.cpp:823/956`); `tail_fit` is WiMDA's `estBG/BGfit`
  (Poisson MLE instead of WiMDA's σ = 10¹⁰ amputation of ≤ 4-count
  bins — divergence D4); `reference_run` is WiMDA's FileBG with variance
  propagation WiMDA lacks (D7). Ordering (group → subtract → asymmetry)
  matches musrfit.
- Asymmetry is the only one of the four that propagates the estimated
  background's uncertainty into per-bin errors on all estimated paths.
  musrfit does so only in its asymmetry estimated-background path (its
  fixed-background path documents `√(f+bkg)` but implements `√f` —
  upstream doc/code mismatch, `PRunAsymmetry.cpp:884/913`). Mantid's
  background handling lives in the GUI corrections layer and its A0
  uncertainty is discarded by `AsymmetryCalc` (see above).

## How this matrix was verified

- WiMDA and musrfit rows seeded from each program's
  `FEATURE_MAP.json` (catalogued slugs).
- Mantid rows derived from a directed walk of `Framework/Muon/`,
  `qt/scientific_interfaces/Muon/`, and
  `docs/source/{algorithms,interfaces,techniques}`.
- Asymmetry rows grep-verified against the live codebase at
  `src/asymmetry/` on the same day this matrix was authored:

  - MODELS registry: 6 entries in
    `core/fitting/models.py`.
  - COMPONENTS registry: 11 entries in
    `core/fitting/composite.py`.
  - Fourier modules: `fft.py`, `grouped.py`, `maxent.py` (stub),
    `window.py`.
  - Transform modules: `asymmetry.py`, `background.py`,
    `deadtime.py`, `grouping.py`, `rebin.py`.
  - IO loaders: `nexus.py`, `psi.py`, `root.py`.
  - GUI windows: `fit_wizard_window.py`,
    `multi_group_fit_window.py`,
    `global_parameter_fit_window.py`.

Re-run that scan when the matrix is revisited to keep cells honest.
