# Time-integral asymmetry — cross-program comparison

## At a glance

| Aspect | WiMDA (ALC mode) | Mantid (`PlotAsymmetryByLogValue`) | musrfit |
| --- | --- | --- | --- |
| Native integral observable | **Yes** (count-integral) | **Yes** (Integral *and* Differential) | **No** — fit time-differential, trend params |
| Default formula | `(F − B)/(F + B)` from summed counts | Integral: `(F − αB)/(F + αB)` from integrated counts | n/a |
| Alpha balance | **Not applied** in ALC | **Applied** (`Alpha`, default 1.0) | n/a (α fitted in time domain) |
| Time window | Full good-bin window per run | Single `[TimeMin, TimeMax]` (full range if unset) | n/a |
| Red/green periods | Red/Green/G−R/G+R | `Red`/`Green` indices → diff & sum spectra | per-run blocks |
| x-axis variable | Field (Tesla), from fit-table column | `LogValue` sample log + `Function` (Mean/Min/Max/First/Last) | run metadata via `mupp` |
| Error model | Poisson, `\|A\|·[1/√\|F−B\| + 1/√(F+B)]` | Poisson (Mantid `AsymmetryCalc` algebra) | n/a |
| Differential transform | `dA/dB = (A₂−A₁)·1000/(B₂−B₁)` %/kG | "Differential" type integrates `A(t)` | n/a |
| Downstream | ALC scan plot + `dA/dB` | ALC interface: load → baseline → peak fit | `mupp` parameter plot |

## WiMDA — ALC mode (`src/muondata.pas`, `src/FitTableUnit.pas`)

- Data structure `ALCpoint` (`muondata.pas:317-350`) holds per-scan-point summed
  forward/backward counts (`Ffc`, `Bbc`), the integral asymmetry (`asym`,
  `asymerr`), and the field `B_IPS`.
- **Integral formula** (`muondata.pas:807-815`): forward detector columns 20–26
  and backward columns 32–38 are summed across the **full good-bin window**, then

  ```
  A = (Ffc − Bbc) / (Ffc + Bbc)
  σ_A = |A| · [ 1/√|Ffc − Bbc| + 1/√(Ffc + Bbc) ]
  ```

  Note: **no alpha balance** — raw geometric asymmetry. (Time-domain diamag fits
  in `Group.pas:1802` *do* use `(f − αb)/(f + αb)`; ALC does not.)
- **No explicit time window control** — integration spans `tgood_beg..tgood_end`
  read from NeXus `first_good_bin`/`last_good_bin` (`nexusunit_.pas:1042-1050`).
- **Scan assembly**: a list of run numbers (`ALCscans.pas`) is batch-fitted; the
  fit table's rows are then read back, each yielding one `(field, A)` point.
  Field is column 1 of the table (`muondata.pas:816`).
- **Differential ALC** (`FitTableUnit.pas:322-402`): forward-difference
  `dA/dB = (A₂−A₁)·1000/(B₂−B₁)` in %/kG, computed only for adjacent points with
  `|ΔB| < 1000 G`; error in quadrature. Used to sharpen resonance features.
- **No deadtime correction** in the ALC path (responsibility of the fit-table
  generation stage). Co-adding multiple scans re-derives `A` from summed counts.

## Mantid — `PlotAsymmetryByLogValue` (`Framework/Muon/...`)

The most complete and most directly portable reference (alpha-aware, matches
Asymmetry's existing `compute_asymmetry` error model).

- **Two reduction types** (`Type` property):
  - **Integral** (`PlotAsymmetryByLogValue.cpp:775-795`): integrate counts over
    `[TimeMin, TimeMax]`, then `Y = (F_int − α·B_int)/(F_int + α·B_int)` using the
    shared `AsymmetryCalc` formula (`AsymmetryCalc.cpp:134-151`), **alpha applied**.
  - **Differential** (`:760-773`): form `A(t) = (F(t) − αB(t))/(F(t) + αB(t))`
    per bin, then integrate the asymmetry curve over `[TimeMin, TimeMax]`.
  - *Key distinction*: Integral integrates **counts then forms asymmetry**;
    Differential forms **asymmetry then integrates**. WiMDA's method ≈ Mantid
    *Integral* (minus alpha).
- **Single time window** `[TimeMin, TimeMax]` (µs); full range if unset. There is
  **no** second integration range.
- **Red/green periods** (`Red`, `Green` 1-based indices): for dual-period runs the
  output spectra are Red−Green difference, Red, Green, and Red+Green sum
  (`:805-855`); errors summed in quadrature.
- **x-axis**: `LogValue` (sample-log name, e.g. `sample_magn_field`) reduced by
  `Function ∈ {Mean, Min, Max, First, Last}` (default `Last`)
  (`getLogValue`, `:867-940`); special-cased `run_start`/`run_end`.
- **Dead-time** (`None`/`FromRunData`/`FromSpecifiedFile`) and **grouping**
  (auto from file or explicit forward/backward spectra) applied **before**
  asymmetry. Results cached in the ADS so extending the run range reuses prior
  runs.
- **ALC interface** (`qt/scientific_interfaces/Muon/ALC*`): three steps —
  (1) `ALCDataLoadingModel` calls `PlotAsymmetryByLogValue` to build the scan,
  (2) `ALCBaselineModelling` fits/subtracts a polynomial baseline,
  (3) `ALCPeakFitting` fits resonance peaks. Steps 2–3 are the *analysis* of the
  observable, separable from producing it.

## musrfit — negative result (`src/...`, `src/musredit_qt6/mupp/`)

- **No integral fittype.** The fittype enum (`PMusr.h:84-97`) is single-histo (0),
  single-histo RRF (1), asymmetry (2), asymmetry RRF (3), µ⁻ (4), β-NMR (5),
  non-µSR (8). There is no integral/scan type and no per-run single-number
  reduction; `PRunAsymmetry` produces a time series and χ² is computed bin-by-bin.
- **ALC is only an instrument label** detected from the filename
  (`PRunDataHandler.cpp:2688`); such data is fitted as ordinary fittype-2
  asymmetry.
- **`mupp`** (parameter plotter) collates **fitted** parameters across a run
  collection and plots one vs another (e.g. field vs temperature):
  `PParamDataHandler::GetValues(coll, param)` returns one value per run
  (`Pmupp.cpp:959-992`). It reads the x-variable from run metadata; it never
  integrates counts. So the musrfit ALC/repolarisation workflow is *fit each run,
  extract a parameter (amplitude/rate), plot vs field* — not an integral observable.

## Mapping onto Asymmetry's existing seams

| Need | Existing Asymmetry seam |
| --- | --- |
| `(F − αB)/(F + αB)` + Mantid error model | `core/transform/asymmetry.py::compute_asymmetry` (already alpha-aware, Mantid-compatible error) |
| Summing counts over a window | pattern in `core/transform/asymmetry.py::estimate_alpha` (good-bin window sum) |
| Slice to `[t_min, t_max]` | `MuonDataset.time_range()` (`core/data/dataset.py:144-157`) |
| Red/green selection | `core/io/periods.py::select_period` / `combine_period_asymmetry` |
| Per-run x-variable (field/T) | `Run.field`, `Run.temperature`; `metadata["field"]` (G), `["temperature"]` (K) |
| Order a series by field/T/run | `FitSeries.order_key` + `sort_members()` (`core/representation/series.py:35,172`) |
| Trend plot vs field/T/run | `gui/panels/fit_parameters_panel.py` (x-axis combo: Auto/B/T/Run) |

The decisive observation: Asymmetry already has every primitive (alpha-aware
asymmetry, good-bin windowing, period selection, field/T metadata, series
ordering, trend plotting). The port is mostly **composition** — a new per-run
reduction plus a new series/representation type that reuses the trend surface.
