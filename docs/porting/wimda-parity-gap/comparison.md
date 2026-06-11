# WiMDA → Asymmetry functionality gap inventory

Date: 2026-06-10. Branch: `study/wimda-parity-gap`.

This is the consolidated record of a full sweep of the WiMDA source
(`$WIMDA_SRC/src`, all `.pas` units, ignoring
`__history/`/`__recovery/`) cross-referenced against the Asymmetry codebase
as of main commit `474534e` (post wimda-fit-function-parity merge). It
supersedes the WiMDA column of `docs/porting/comparison-matrix.md`, which
predates several merges.

The goal is **parity of functionality, not parity of implementation**: if an
analysis can be done in WiMDA, it should be possible in Asymmetry — using
modern, efficient, physically-correct approaches where these differ from
WiMDA's.

Method note: WiMDA's own `FEATURE_MAP.json` has known blind spots (it routes
"maxent" to the Burg code and misses `Wimdamax.pas` entirely), so this sweep
read the actual units rather than navigating by the map.

## Status legend

- ✅ PRESENT — capability exists in Asymmetry (cited).
- ◐ PARTIAL — exists but materially less capable; remainder noted.
- ❌ ABSENT — does not exist.
- Verdict for niche items: **port** / **adapt** (re-imagine natively) /
  **drop** (with rationale).
- `→ project` points at the brief in `projects/` that owns the gap.

---

## 1. Data ingestion & run handling

| Capability | WiMDA source | Asymmetry status | Verdict / project |
|---|---|---|---|
| ISIS NeXus v1/v2 (HDF5, periods, logs, deadtimes) | `nexusunit.pas` | ✅ `core/io/nexus.py` | done |
| PSI .bin / .mdu (incl. HAL-9500 layout) | `muondata.pas:1504,1695` | ✅ `core/io/psi.py` | done |
| MUSR ROOT / LEM | (LEM ASCII branch only) | ✅ `core/io/root.py` (beyond WiMDA) | done |
| TRIUMF MUD | `mudunit.pas` — **non-functional stub in WiMDA itself** (`LoadLibrary` commented out) | ❌ | not a WiMDA-parity item; stays on the general roadmap |
| HDF4 .nxs | n/a | ❌ | **out of scope** (standing project decision) |
| Legacy formats: `.tri`, `.kek`, KEK binary, DeltaT `.dat`, VMS PSI, MCS, `.raw` 16ns ASCII, ARGUS/CHRONUS `.ral` MACS | `muondata.pas:841–2150` | ❌ | **drop** — superseded by NeXus; no active data sources |
| Zip/bz2 transparent decompression of runs | `WiMDA_Main.pas:1004` | ❌ | **drop** — modern archives serve uncompressed |
| Live current-run monitoring ("run 0" from DAE temp files) | `muondata.pas:1376` | ❌ | optional late phase → `workflow-visualisation` (needs beamline access to test) |
| Event-mode time-slice loading | `nexusunit.pas:1632` | ❌ | defer — specialist; revisit on user demand |
| Legacy `alc*` autoALC scan files | `muondata.pas:722–837` | ❌ (scan *building* from runs is ✅, `core/transform/integral.py`) | **drop** unless legacy files surface (small io adapter then) |
| Run-number stepping / filename-pattern walker (LOAD>>, STEP) | `WiMDA_Main.pas` `PrevRun/NextRun/GetPrefixSuffix` | ◐ browser table paradigm; no stepping | → `workflow-visualisation` |
| Histogram-level co-add (counts summed, frames accumulated, event-weighted T/B averaging) | `muondata.pas:2418–2490` | ◐ `data_browser.py:_coadd_datasets` averages reduced asymmetry curves (statistically wrong for low counts; loses histograms) | → `run-arithmetic` |
| Co-subtract mode (laser on/off, background runs) | `Cosubmode1Click` | ❌ | → `run-arithmetic` |
| Log handling: T/B/laser/aux logs, log-averaged values as run T/B | `LogbookUnit.pas`, `nexusunit getNexuslog` | ◐ NXlog plotting + "use T from log"; no B-from-log, laser/aux logs, external `tlog\` `.mon` files | B-from-log → `workflow-visualisation`; laser/aux/`.mon` **drop** (pre-NeXus legacy) |

## 2. Grouping, periods, corrections

| Capability | WiMDA source | Asymmetry status | Verdict / project |
|---|---|---|---|
| Detector grouping editor + instrument presets | `Group.pas`, `Group2.pas` | ✅ grouping dialog, instrument presets incl. HAL-9500 | done |
| Per-detector exclude list (dead/hot detectors) | `Group2.pas ExcludeDetectors`, `default.exclude` | ❌ | → `data-reduction-parity` |
| Grouping file save/load + per-directory defaults (`default.mgp`) | `Group.pas:1877,2153`; `WiMDA_Main:1190` | ◐ project persistence + `.grp` I/O; no per-directory auto-defaults | → `data-reduction-parity` (minor) |
| RG mode Red/Green/G−R/G+R | `Group.dfm RGModeBox` | ✅ `core/io/periods.py` | done |
| Arbitrary N-period → red/green subset mapping (up to 8, Ignore option) | `PeriodMappingUnit.pas`, `MapPeriods` | ◐ per-period select + 2-set G±R; no subset summation | → `data-reduction-parity` |
| t0 / first-good / last-good (file vs manual) | `Group.pas` | ✅ loaders + grouping dialog | done |
| Automatic t0 search (prompt-peak scan) | `Group.pas:2225 SearchT0ButtonClick` | ❌ | → `data-reduction-parity` |
| Fixed bunching rebin | `Group.pas Regroup` | ✅ `core/transform/rebin.py` | done |
| Variable binning (log-friendly growth) & constant-error binning (width ∝ e^{λt}) | `Group.pas:1411–1418` | ❌ fixed factor only | → `data-reduction-parity` |
| Deadtime apply (non-paralyzable) | `Group.pas ccorrect` | ✅ `core/transform/deadtime.py` | done |
| Deadtime auto-estimate + per-detector calibration | `Group.pas:1340,2750` | ✅ `estimate/calibrate_deadtime_from_histograms` | done |
| Deadtime file auto-discovery (`dt*.dat`, `cal\`) + stale-calibration warning | `WiMDA_Main:925–975` | ◐ parse/load only | → `workflow-visualisation` (minor) |
| Extended deadtime models (polynomial, power-law, KEK spill) | `Group.pas ccorrect` Model panel | ❌ simple non-paralyzable only | → `count-domain-fit-modes` (with count-loss fitting) |
| Background: manual / region-average | `Group.pas aveBG` | ✅ `core/transform/background.py` | done |
| Background: auto fit of decay+flat to spectrum tail (pulsed sources have no pre-t0 region) | `Group.pas estBG/BGfit` | ❌ | → `data-reduction-parity` |
| Background-run subtraction (frame-ratio scaled co-loaded run) | `Group.pas Regroup FileBG`, `BGform.pas` | ❌ | → `data-reduction-parity` |
| Fitted-baseline Set BG / Unset BG (freeze fit curve as background) | `Analyse.pas:5877` | ❌ | defer — specialist two-step workflow; revisit with `count-domain-fit-modes` |
| Alpha estimation: diamagnetic χ² grid-search; lifetime-corrected "General" method (works on LF/ZF data) | `Group.pas:1775 EstimateButtonClick` | ◐ Mantid ΣF/ΣB ratio only (`estimate_alpha`) | → `data-reduction-parity` |
| F−B asymmetry + error model; G−R asymmetry | `Analyse.pas Getdata` | ✅ `core/transform/asymmetry.py`, periods | done |
| Multichannel N→32 mapping, ARGUS kicker-noise/ACC-shift fixers, KEK port select | `Group.pas`, `muondata.pas:2367` | n/a | **drop** — workarounds for WiMDA's fixed arrays / dead hardware eras |

## 3. Time-domain fitting

Fit *functions* reached parity in `wimda-fit-function-parity` (merged
2026-06-10); the fitting-slice sweep verified **nothing in the WiMDA function
source goes beyond what that study covered**. Remaining gaps are machinery
and workflow.

| Capability | WiMDA source | Asymmetry status | Verdict / project |
|---|---|---|---|
| Oscillation/relaxation function library (incl. plugin-DLL contents) | `FitTyps/AsymFitFunction/KuboToyabe/Extrafunctions` | ✅ 34 components + 10 models; parity confirmed | done |
| Minimisation engine | `Fitucode.pas FITE` (1972 Gauss–Newton) | ✅ iminuit migrad/simplex — superior | done |
| MINOS asymmetric errors / explicit HESSE | (absent in WiMDA too; musrfit strength) | ❌ | → `fit-workflow-diagnostics` (existing top-scored candidate) |
| Composite expressions, fractions, dependent amplitudes | `Analyse` grid + `MusrFun` | ✅ composite syntax + `{frac}` groups | done |
| Equality link groups | `LinkGroupForm.pas` | ✅ `parameters.py link_group` (PR #27) | done |
| All-groups simultaneous count-level fit (per-group N0/BG/Ampl/Phase) | `Analyse fgAll` | ✅ `grouped_time_domain.py` + MultiGroupFitWindow | done |
| α as free fit parameter (simultaneous F+B raw-count fit) | `Analyse fgFB` | ❌ | → `count-domain-fit-modes` |
| Single-histogram fitting (F-only/B-only, N0 + BG) | `Analyse fgForward/fgBackward` | ❌ | → `count-domain-fit-modes` |
| Interior exclude time range (second window) | `Analyse SecondRange` | ❌ | → `count-domain-fit-modes` |
| Fittable t0 offset; stretched-exp baseline drift | `Analyse pname[BG_base..]` | ❌ | → `count-domain-fit-modes` |
| Count-loss (deadtime) parameters inside the fit, pushed back to grouping | `Analyse CountLossModelling`, `SendToGroupClick` | ❌ | → `count-domain-fit-modes` |
| Double-pulse fitting (ISIS double-pulse mode) | `Analyse DoublePulse`, `ArrayMusrFunc:170–237` | ❌ | → `count-domain-fit-modes` |
| RRF fitting | `PlotPar RRFon` + `MusrFun` | ❌ | → `rrf` |
| Multifit (multi-run simultaneous, x2 = field/delay) | `Multifit.pas` | ✅ Global tab global/local roles — strictly more general; add an "LF-scan global fit" docs recipe | done (docs recipe) |
| Batch fitting over run lists | `BatchFit.pas` | ✅ batch series fits | done |
| Sequential warm-start (run N seeds run N+1, "itPrevious") | `BatchFit.pas` | ◐ single-fit seeds batch; no carry-forward chaining | → `fit-workflow-diagnostics` |
| χ²/dof statistical target band + good/poor/**overdone** flag | `FitOpt.Rgoodfit`, `Chi2Update` | ❌ χ²/dof number only | → `fit-workflow-diagnostics` |
| Mid-fit abort | `FitStatusForm` + `StopFitting` | ❌ (worker threads, no cancel) | → `fit-workflow-diagnostics` |
| Persistent on-disk fit log (`.fit`/`.mfit`/`.bfit`) | `Fitting.pas` | ◐ in-app results/log panel only | → `fit-workflow-diagnostics` |
| In-batch run co-adding (Smooth/Bin) | `BatchFit.pas` | ❌ | optional phase → `fit-workflow-diagnostics` (depends on `run-arithmetic`) |
| Re-fit co-added selection / rebin-fit-table-by-refit | `FitTableUnit.pas:718`, `Rebinning.pas` | ❌ | optional phase → `fit-workflow-diagnostics` (depends on `run-arithmetic`) |
| Fit-table grid resampling (for table FFTs) | `Resampling.pas` | ❌ | **drop** — served WiMDA's text-table FFT workflow |
| Animated per-iteration fitting | `Fitucode.pas:518` | ❌ | **drop** — legacy spectacle; abort + status suffice |
| Two-stage fit-vars vs save-vars selections | `Analyse.FITMouseDown` | ❌ | **drop** — parameter roles cover the need |
| Statistics degradation (Poisson thinning) | `DegradeStats.pas` | ❌ | **adapt** → fold into `simulate-mode` |

## 4. Parameter trending / Model layer

| Capability | WiMDA source | Asymmetry status | Verdict / project |
|---|---|---|---|
| Trend fitting of fit-series parameters vs field/T/run | `Model.pas` + `FitTableUnit.pas` | ✅ `parameter_models.py` (31 components incl. SC library — beyond WiMDA), model_fit_dialog, cross-group fits | done |
| Built-in model-function library (polynomial, power laws, 2-component Arrhenius, order parameter, critical divergence, Mu repolarisation, 2-Lorentzians+cubic) | `fitfunctions.pas:216–305` | ◐ several present (OrderParameter, CriticalDivergence, Arrhenius, PowerLaw); gaps enumerated in the project brief | → `model-function-parity` |
| Error modes Column/Percent/Absolute/None/Estimate (post-fit √(χ²/dof) rescale) | `Model.pas:685` | ◐ propagated errors + floor only | → `model-function-parity` |
| x2 second model variable (column or fixed) — e.g. λ(T) surfaces at several fields | `Model.pas` | ❌ | → `model-function-parity` |
| Arbitrary x column / external table import for trend fitting | `Model.pas columnscan` | ❌ x ∈ {field, temperature, run} | → `model-function-parity` |
| Multi-range x selection | `Modelxrange.pas` | ✅ multiple fit ranges in model_fit_dialog | done |
| Model `*fit.dll` user libraries (Delphi + FORTRAN) | `UserUnit.pas` | ❌ | → `python-user-functions` |
| Second-level Model Fit Table + GLE model plots | `ModelFitTableUnit.pas`, `PlotModel.pas` | ✅ global_parameter_fit_window + GLE export | done |
| Differential ALC (dA/dB) + ALC scan build + baseline/peak fit | `FitTableUnit`, `muondata` alc path | ✅ `integral.py`, `field_scan.py`, `alc_panel.py` | done |
| Kramers–Kronig optical-constants transform | `KramKron.pas` | ❌ | **drop** — optical spectroscopy, not μSR (decision 2026-06-10) |

## 5. Frequency domain

| Capability | WiMDA source | Asymmetry status | Verdict / project |
|---|---|---|---|
| FFT engine, apodisation (+start/τ), padding, average-subtract, time window | `Fourier.pas`, `FFTPar.pas` | ✅ `core/fourier/` (exact filter formulas) | done |
| Phase modes ×5, per-group phase table, auto-phase (peak/ave), t0 offset | `FFTPar Sinmode` | ✅ + entropy `phase_opt_real` beyond WiMDA | done |
| Group-averaged spectra, group exclusion, variance errors | `Plot.pas:1773+` | ✅ `grouped.py` | done |
| Field-axis (Gauss) spectrum; Tesla unit | `globals.pas:54 FFTb/AvFFTb` | ❌ MHz only | → `frequency-domain-finishers` |
| Frequency-range exclusion ×10 + diamag slot + PSI RF-harmonics preset | `FFTPar.pas:327–377` | ◐ core fn exists (`exclude_frequency_ranges`), not wired to GUI | → `frequency-domain-finishers` |
| Fit-and-subtract diamagnetic signal pre-FFT | `Plot.pas:1832–1890` | ❌ | → `frequency-domain-finishers` |
| Frequency-response compensation (ISIS pulse rolloff) | `Plot.pas:1931–1944` | ❌ | → `frequency-domain-finishers` |
| Spectrum BG offset (2σ-clipped baseline) | `Plot.pas:1959–1984` | ❌ | → `frequency-domain-finishers` |
| S/N + average-error readouts | `Plot.pas:1352–1385` | ◐ | → `frequency-domain-finishers` |
| Real+imag simultaneous view; N0-normalised single-histogram input | fourier study deferred list | ❌ | → `frequency-domain-finishers` |
| **Burg all-poles MEM pole scan** (FPE-optimised AR spectrum) | `MaxEnt.pas` | ❌ | **include as diagnostic** (decision 2026-06-10) → `frequency-domain-finishers`; present as a qualitative resolution-enhancement view with documented caveats (spurious-splitting pathology, no errors), never the quantitative result |
| Muonium/radical correlation spectrum (Breit–Rabi pair matching → hyperfine axis) + radical cursor tools | `Plot.pas:515–523,1387,2149–2230` | ❌ | optional phase → `frequency-domain-finishers` (niche; radical chemistry community) |
| FB t=0 extrapolation pre-FFT | `Plot.pas:1330–1349` | ❌ | **drop** — Asymmetry's Fourier source model (grouped counts) makes it moot |
| Eigenvalue units | `Eigen*.pas` | n/a | **exclude permanently** (decision 2026-06-10): mislabelled in old roadmap — these are Hermitian eigensolvers serving the F–μ–F plugin models, superseded by `np.linalg.eigh`; not a spectral estimator |

## 6. MaxEnt

Engine + GUI shipped (PRs #16, #26). Remaining WiMDA surface:

| Capability | WiMDA source | Asymmetry status | Verdict / project |
|---|---|---|---|
| MULTIMAX joint MaxEnt engine, run controls, convergence guard | `Wimdamax.pas`, `MaxControl.pas` | ✅ `core/maxent/engine.py` (improved stopping) | done |
| Time-domain reconstruction overlay (per-group reconstruction vs data) | `Wimdamax.pas:1148–1163` | ❌ (`opus` exists; not exposed) | → `maxent-completion` — **top shipped-MaxEnt gap** |
| ISIS pulse-shape response (single/double pulse, half-width, separation) | `Wimdamax.pas:266–319` | ❌ | → `maxent-completion` — needed for pulsed-source work ≳5 MHz |
| ZF/LF two-group mode + SpecBG zero-frequency lineshape subtraction | `Wimdamax`, `SpecBG.pas` | ❌ | → `maxent-completion` — ZF field-distribution workflow |
| Deadtime fitting inside MaxEnt; editable phase/deadtime tables; fitted-phase ↔ fit exchange | `DEADFIT`, `MaxEdit.pas`, `PhaseTableUnit.pas` | ◐ phases/amps in diagnostics; no editing/exchange | → `maxent-completion` (subsumes the `phase-auto-calibration` candidate's WiMDA slice) |
| Field-axis display + shift units | `MaxControl.pas:157–314` | ❌ | → `maxent-completion` (shares units work with `frequency-domain-finishers`) |
| Exclusion time window; error apodisation; smooth errors; looseness/phase-accel knobs | `Wimdamax.pas` | ❌/◐ | → `maxent-completion` (evaluate per-knob; some are numerical-era cruft) |
| Spectral deconvolution (Lor/Gau/measured) | `Wimdamax.pas:321–357` | ❌ | defer — numerically hazardous (study flagged `1/Sconv`); revisit on demand |
| Spectrum file export + rich log | `MaxControl GObuttonClick` | ◐ project recipe only | → `maxent-completion` (minor) |
| Spectral moments (B_pk, B_ave, B_rms, skewness α, lineshape β, run averaging, trend export) | `Moments.pas` | ❌ | → `spectral-moments` — the main quantitative consumer of the MaxEnt spectrum (penetration depth, vortex lattice) |

## 7. Visualisation, export, workflow utilities

| Capability | WiMDA source | Asymmetry status | Verdict / project |
|---|---|---|---|
| Time/frequency plots, per-group grids, overlays, residuals, fit curves | `Plot.pas` | ✅ plot_panel / workspace | done |
| Raw-count & log-count display modes | `Plot.pas GetDataGroup` | ❌ | → `workflow-visualisation` (t0/deadtime diagnostics) |
| F,B overlay view (α calibration aid) | `Plot.pas FBOverlay` | ❌ | → `workflow-visualisation` |
| Snapped data cursor: index/x/y±err, S/N, parabolic peak readout, windowed average ± error | `Plot.pas:1159–1228,2962+` | ◐ free cursor coords only | → `workflow-visualisation` |
| RRF display (MHz/G, phase, smoothing bins) | `PlotPar` RRF*, `Plot.pas:1652+` | ❌ | → `rrf` |
| ASCII export: data / data+fit / fit-only, x-range restricted, batch over run range, provenance headers | `SaveAsItemClick`, `SaveRange.pas`, `Plot.pas:3082` | ◐ GLE `.dat` sidecars only | → `workflow-visualisation` |
| Cursor-point → fit table (single + batch) | `Plot.pas:3034–3099` | ◐ fit-based trending supersedes | **drop** (superseded); revisit only if users miss it |
| GLE publication export (data, trends, model curves) | `GLEUnit`, `PlotModel`, `MakeGLE` | ✅ all panels, with preview | done |
| In-app GLE source editor | `GLEUnit.pas` | ❌ | **drop** — external editors; `.gle` files are readable |
| Printing (plot / with fit details) | `Plot.pas:2762–2860` | ❌ | **drop** — PDF export supersedes |
| Error-bar toggle, marker styles, tick spacing, ns/bins x-units | `PlotPar.pas` | ◐ | → `workflow-visualisation` (cosmetic basket, low priority) |
| Simulate mode (model → Poisson histograms → loadable NeXus) | `Simulate.pas` | ❌ (existing "now" candidate) | → `simulate-mode` |
| Logbook / multi-run table, filters, export | `LogbookUnit.pas` | ✅ Data Browser (beyond WiMDA) | done |
| Events-MEv / events-per-frame columns | `LogbookUnit:595–696` | ❌ | → `workflow-visualisation` (small) |
| Log time-series plots (T/B/aux tabs) | `tlogplotunit.pas` | ✅ run-info NXlog → LogPlotDialog | done |
| Histogram report (per-detector totals across runs; detector health) | `HistReport.pas` | ❌ | **adapt** → data-browser export columns, optional item in `workflow-visualisation` |
| Negative-muon elemental analysis (multi-τ capture fits, 67-element lifetime table, F/B capture ratios, BG subtract) | `NegMuAnalyse.pas` (2474 lines) | ❌ | **deferred brief** (decision 2026-06-10): adapt-not-port — see `projects/negative-muon-analysis.md` |
| User fit-function plugins (`musrfunctions.dll` osc/rel + `*fit.dll` models, FORTRAN variant) | `MusrFunctionUnit/`, `UserUnit/` | ❌ | → `python-user-functions` |
| ALCscans run-list form, SetALCthresh, CommentForm | `ALCscans.pas` etc. | n/a | **drop** — dead/vestigial in WiMDA itself |
| Registration/licensing, About | `security.pas` | n/a | **drop** |

---

## Settled exclusions (with rationale)

| Item | Decision | Rationale |
|---|---|---|
| HDF4 `.nxs` | Out (standing decision, reaffirmed 2026-06-10) | Coverage boundary, not a bug; all loadable corpus data is HDF5/PSI/ROOT |
| Eigen.pas eigensolvers | Out, permanently | Old roadmap mislabel: not a spectral estimator; eigensolver infrastructure for F–μ–F models, superseded by `np.linalg.eigh` (FmuF components shipped) |
| Kramers–Kronig | Out | Optical-spectroscopy utility (reflectivity → optical constants); transforms no μSR observable |
| Burg all-poles MEM | **In** (reversal of prior exclusion) | Cheap (~100 lines numpy); genuine niche (super-resolution of close lines from short windows + FPE pole-count diagnostic); to be framed as a diagnostic view with documented pathologies, never the quantitative result |
| Legacy file formats, zip loading, printing, GLE editor, ARGUS/KEK hardware fixers, multichannel mapping, fit-table resampling, animated fitting, dead WiMDA forms | Out | Legacy/cruft; no modern data source or superseded by Asymmetry equivalents (see tables above) |
| MUD loader | Out of *this* study | WiMDA's own MUD support is a non-functional stub — not a parity item; remains on the general (non-WiMDA) roadmap |

## Confirmed parity (headline)

Loaders for all live formats; grouping + instrument presets; deadtime
apply/estimate/calibrate; region/manual background; asymmetry + error model;
periods/RG; rebin (fixed); the **entire fit-function library**; link groups;
multi-group count fits; global/batch fitting (more general than Multifit);
trend fitting + ALC workflow (more capable than WiMDA's); FFT + phases +
windows (plus entropy phase optimisation WiMDA lacks); MULTIMAX MaxEnt core;
GLE export; data browser/logbook; log plotting. Asymmetry-only strengths
(Fit Wizard, global search, composite expressions, SC model library,
schema-versioned projects) have no WiMDA counterpart.
