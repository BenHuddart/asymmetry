# Mantid (muon slice) feature inventory

**Upstream:** `$MANTID_SRC`  
**Authors of original:** Mantid project team (ISIS + ORNL +
contributors)  
**Language:** C++ core (Framework) + Python/PySide2 (GUI) +
`matplotlib` plots  
**Pre-existing catalog reused:** none (Mantid has no FEATURE_MAP;
this inventory walks `Framework/Muon/`,
`qt/scientific_interfaces/Muon/`, and
`docs/source/{techniques,algorithms,interfaces,fitting/fitfunctions}`
explicitly).

Mantid is a general-purpose data-reduction framework with a
production-grade μSR analysis slice maintained for ISIS pulsed-muon
and (via PSI plugins) continuous-muon users. Its design choices —
workspaces as the data abstraction, MVP architecture in the GUI,
algorithms with type-checked properties — differ noticeably from
WiMDA's form-driven model and musrfit's `.msr`-centric model.
Features below follow the unified 10-category taxonomy and focus on
the muon-specific surface.

---

## 1. Data ingestion

- **`LoadMuonNexus` (v1–v3), `LoadPSIMuonBin`, `LoadMUD`,
  `LoadMuonLog`** — Multi-format loaders for NeXus (ISIS standard,
  multiple schema versions), PSI binary, MUD (TRIUMF), and sample
  logs.  
  Flagship: `Framework/Muon/src/LoadMuonNexus*.cpp`,
  `Framework/Muon/src/LoadPSIMuonBin.cpp`.
  NeXus loaders parse period data natively; ISO8601 timestamp
  parsing; detector grouping tables extracted at load.

- **Period handling** — Automatic period grouping on load;
  `SummedPeriodSet` / `SubtractedPeriodSet` syntax in `MuonProcess`
  for period arithmetic on pulsed-source data.
  **Notable:** Mantid is the only reference tool with first-class
  period arithmetic.

- **`ApplyDeadTimeCorr`, `CalMuonDeadTime`** — Deadtime as a
  first-class correction algorithm (not embedded in ingest), applied
  before asymmetry calculation. `CalMuonDeadTime` refines the
  estimate.  
  Flagship: `Framework/Muon/src/ApplyDeadTimeCorr.cpp`.

## 2. Asymmetry calculation

- **`AsymmetryCalc`, `MuonPairingAsymmetry`,
  `MuonGroupingAsymmetry`, `AlphaCalc`,
  `EstimateMuonAsymmetryFromCounts`** — Pipeline-driven asymmetry
  computation. Base formula
  `A(t) = (N_F - α·N_B) / (N_F + α·N_B)` with normalisation and
  error propagation.  
  Flagship: `Framework/Muon/src/AsymmetryCalc.cpp`,
  `Framework/Muon/src/MuonPairingAsymmetry.cpp`,
  `Framework/Muon/src/MuonGroupingAsymmetry.cpp`,
  `Framework/Muon/src/AlphaCalc.cpp`.

- **`ApplyMuonDetectorGrouping`,
  `ApplyMuonDetectorGroupPairing`,
  `LoadAndApplyMuonDetectorGrouping`, `MuonGroupDetectors`** —
  Grouping defined via XML files mapping detector index → group;
  group pairs carry their own α. Grouping tables flow through the
  workflow rather than living inside per-run objects.  
  Flagship: `Framework/Muon/src/ApplyMuonDetectorGrouping.cpp`,
  `qt/scientific_interfaces/Muon/IO_MuonGrouping.cpp`.

- **`CalMuonDetectorPhases`** — Estimates per-detector phases by
  fitting early-time data to a sinusoid; outputs a phase table.
  **Notable:** no equivalent in WiMDA, musrfit, or Asymmetry — phase
  estimation in those tools is manual.

- **`RRFMuon`** — Rotating Reference Frame transform; demodulates
  high-frequency oscillation into a slowly-varying envelope, useful
  for very-high-TF and vortex-lattice studies.  
  Flagship: `Framework/Muon/src/RRFMuon.cpp`.
  Mantid-only; musrfit has an `RRF` variant inside its asymmetry
  classes but does not expose it as a separate algorithm.

- **`PSIBackgroundSubtraction`, `RemoveExpDecay`** — Background
  removal helpers. PSI-specific paths handled separately.

## 3. Time-domain fitting

- **General Mantid `Fit` algorithm + muon fit-function library** —
  Fit functions live in `Framework/CurveFitting/src/Functions/` and
  are exposed via the global `FunctionFactory`. The muon-specific
  set:

  - `ExpDecayMuon`, `StretchExpMuon`
  - `StaticKuboToyabe`, `DynamicKuboToyabe`
  - `StaticKuboToyabeTimesExpDecay`,
    `StaticKuboToyabeTimesGausDecay`,
    `StaticKuboToyabeTimesStretchExp`
  - `ExpDecayOsc`, `GausDecay`
  - `Keren`, `Meier`
  - `MuonFInteraction`
  - `HighTFMuonium`, `LowTFMuonium`, `TFMuonium`, `ZFMuonium`
  - `MuoniumDecouplingCurve`

  Flagship: `Framework/CurveFitting/src/Functions/Keren.cpp`,
  `Meier.cpp`, etc.; helpers in
  `Framework/CurveFitting/inc/MantidCurveFitting/MuonHelpers.h`.
  **Notable:** Mantid has the broadest muonium-related coverage
  (four TF/ZF/decoupling functions), the only Meier and Keren
  implementations of the three programs, and the only dedicated
  dynamic KT (musrfit has a dynamic KT but it's part of the larger
  theory library; WiMDA's is registry-loaded).

- **Type-checked function properties** — Mantid parameters carry
  type and bounds metadata that the Fit algorithm validates.

## 4. Multi-spectrum / global fitting

- **`MuonSequentialFitDialog` (C++ widget) + "Simultaneous fit"
  mode in the Muon Analysis GUI** — Sequential mode loops over
  workspaces with shared parameters; simultaneous mode wraps
  multiple datasets in a `CompositeFunction` with linked
  parameters.  
  Flagship: `qt/scientific_interfaces/Muon/MuonSequentialFitDialog.cpp`;
  Python context in
  `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/Common/contexts/basic_fitting_context.py`,
  `general_fitting_context.py`.

## 5. Fourier / frequency-domain

- **Frequency Domain Analysis interface** — A *separate GUI tab*
  from the main Muon Analysis. FFT path uses
  `PaddingAndApodization` (Lorentz / Gaussian apodisation) →
  `FFT` (numpy backend through Mantid).  
  Flagship:
  `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/FrequencyDomainAnalysis/`.

- **`MuonMaxent`** — Production-grade maximum-entropy
  reconstruction. Iteratively refines the frequency spectrum and
  phases; outputs reconstructed time-domain data.  
  Flagship: `Framework/Muon/src/MuonMaxent.cpp`.
  **Notable:** along with WiMDA, the only working MaxEnt of the
  three reference programs.

## 6. Parameter trending

- **ALC (Avoided Level Crossing) interface** — Dedicated GUI for
  muon ALC measurements: data loading → baseline modelling
  (polynomial / spline) → peak fitting (Lorentz / Gaussian on
  baseline-corrected data). Strict MVP architecture.  
  Flagship: `qt/scientific_interfaces/Muon/ALCInterface.cpp`,
  `ALCDataLoading{Presenter,Model,View}.{cpp,h}`,
  `ALCBaselineModelling{Presenter,Model,View}.{cpp,h}`,
  `ALCPeakFitting{Presenter,Model,View}.{cpp,h}`.
  **Notable:** unique to Mantid. ALC is a niche but important
  ISIS technique for hyperfine-coupling studies and would be a
  flagship addition for Asymmetry.

- **`PlotAsymmetryByLogValue`** — Time-domain asymmetry as a
  function of a sample log (field, temperature) across a run
  series; outputs a trending table.  
  Flagship: `Framework/Muon/src/PlotAsymmetryByLogValue.cpp`.

- **Results Tab** — Sortable, filterable table of all fits in
  the session with χ², parameter values, uncertainties. Export to
  CSV / Excel.  
  Flagship:
  `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/Common/results_tab_widget/results_tab_model.py`.

## 7. Logbook / multi-run management

- **Workspace groups** — Mantid's data abstraction lets a "run"
  expand to a `WorkspaceGroup` containing per-period or
  per-detector-group workspaces, enabling sequential processing
  without bespoke run-list management.  
  Flagship: `Framework/API/src/WorkspaceGroup.cpp` (general); the
  Muon Analysis "Loaded data" panel.

## 8. Visualisation

- **Muon Analysis plot widget** — Multi-pane plot: raw data
  (time-domain), processed (asymmetry / counts), fit residuals;
  dual-pane mode for raw + fit; matplotlib backend integrated into
  the Workbench canvas. Observer pattern triggers replot on
  context change.  
  Flagship:
  `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/Common/plot_widget/`,
  `plotting_dock_widget.py`.

## 9. Project files / persistence

- **Mantid project (`.mantid`) files** — Workspace groups persist
  session-to-session if saved. Plus AnalysisDataService (ADS)
  exposes all live workspaces via Python and the GUI.
  Project files are richer than `.msr` but less hand-editable;
  binary HDF5 underneath.

## 10. Workflow utilities

- **`MuonProcess`, `MuonPreProcess`** — Pipeline-orchestration
  algorithms combining deadtime → grouping → offset / crop / rebin
  → asymmetry / counts.  
  Flagship: `Framework/WorkflowAlgorithms/src/MuonProcess.cpp`,
  `Framework/Muon/src/MuonPreProcess.cpp`.
  Diagram: `docs/source/diagrams/MuonProcess-v1_wkflw.dot`.

- **`ConvertFitFunctionForMuonTFAsymmetry`** — Helper that wraps
  any fit function for TF asymmetry analysis (handles the standard
  TF prefactor).  
  Flagship:
  `Framework/Muon/src/ConvertFitFunctionForMuonTFAsymmetry.cpp`.

- **Elemental Analysis interface** — Specialist GUI for
  negative-muon X-ray emission spectroscopy (μ-XRF). Separate
  from the main Muon Analysis interface.  
  Flagship: `qt/python/mantidqtinterfaces/mantidqtinterfaces/Muon/GUI/ElementalAnalysis/`.

---

## Notable Mantid-only depth

- **ALC interface** (Avoided Level Crossing) — niche but
  scientifically valuable.
- **`MuonMaxent`** — robust MaxEnt with iterative phase refinement.
- **Period arithmetic** (`SummedPeriodSet`, `SubtractedPeriodSet`).
- **`RRFMuon`** as a standalone algorithm.
- **`CalMuonDetectorPhases`** — automated phase estimation.
- **Muonium-specific fit functions** (`HighTFMuonium`,
  `LowTFMuonium`, `TFMuonium`, `ZFMuonium`,
  `MuoniumDecouplingCurve`) — broadest muonium / radical coverage.
- **`Meier` and `Keren` fit functions** — exchange-coupled and
  hopping-rate models not in WiMDA or musrfit's standard library.
- **Elemental Analysis** — μ-XRF interface (out of scope for
  Asymmetry).

## Obvious Mantid gaps

- **No interactive parameter-trending GUI** — ALC is restricted
  to peak fitting; general parameter trending requires manual
  workspace handling.
- **No Fit Wizard / model-recommendation tool** — Asymmetry's AICc
  ranking has no Mantid analogue.
- **No composite-model expression syntax** — fit functions
  compose via `CompositeFunction` but not via a free-form arithmetic
  expression string.
- **Heavyweight install** — Mantid is a 1+ GB build; not casual to
  set up.
- **MVP friction** — three-file presenter/model/view pattern in C++
  is more boilerplate than the Asymmetry Python+Qt approach.
