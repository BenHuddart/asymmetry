# Data-reduction parity — test data

Corpus root: `~/Documents/WiMDA muon school/` (see the corpus
`INDEX.md` and `docs/testing/wimda-corpus.md`). All `.nxs` paths below are
the HDF5 conversions (`data_hdf5/`/`Data_hdf5/` siblings) Asymmetry loads.
Corpus-dependent tests use the established `skipif` pattern
(`tests/test_period_selection.py:346`): synthetic fixtures always run; corpus
tests run when the files exist locally.

## Phase 1 — alpha estimation

| Purpose | Data | Notes |
|---|---|---|
| Diamagnetic α vs WiMDA-transcribed grid arithmetic | One weak-TF calibration run from the corpus (Magnetism/Ferromagnetic nickel set is native HDF5; pick a 20 G TF run above T_C, recorded during implementation) | Oracle: Python transcription of the grid walk + objective from `Group.pas:1775`; assert continuous optimiser lands within the final grid step (±0.001) of the transcribed result on identical input |
| General α self-consistency on relaxing LF data | **HIFI runs 118222–118240** — corannulene LF repolarisation scan at 350 K, fields 0–5000 G (`Chemistry/Molecular dynamics of corannulene/data_hdf5/HIFI00118222.nxs`–`...240.nxs`). The 10–100 G subset is 118228–118232 | ⚠ The umbrella brief calls this series "EMU"; the files and embedded `instrument/name` are HIFI. α is a detector/sample property: estimates across the field series must agree within uncertainties; compare scatter of General α vs ΣF/ΣB α across the series |
| ΣF/ΣB bias demonstration (docs figure) | Same LF series | General vs ratio estimate on a strongly relaxing run; documents *why* the new method exists |
| Synthetic oracle | Poisson draws from F = N_F e^{−λt}(1+a₀P(t)), B = N_B e^{−λt}(1−a₀P(t)) with known α = N_F/N_B, P(t) ∈ {cos ωt, e^{−σ²t²}, constant} | Both estimators must recover α within stated uncertainty; ratio estimator must show the predicted ≈ a₀⟨P⟩ bias on relaxing P |

## Phase 2 — backgrounds

| Purpose | Data | Notes |
|---|---|---|
| Tail-fit on pulsed ISIS data | Long-window pulsed runs: FmuF/PTFE ZF runs (`Nuclear magnetism and ionic motion/The FmuF state in PTFE/data_hdf5/`) and/or a nickel run | Oracle: transcription of `BGfit`/`estBG` (bin-integrated model, √N weights, ≤ 4-count deletion, late-half window) run on identical binned input — assert the Poisson-MLE result agrees within the transcribed fit's uncertainty scale; plus synthetic truth tests |
| Tail-fit synthetic | Poisson histograms with known flat rate added to exponential decay, low-count tails (counts/bin ≪ 10 at late t) | MLE must be unbiased where √N-weighting is demonstrably biased low — this is the D4 justification test |
| Background-run subtraction | Pairs from the corpus sharing instrument/geometry with different frame counts (e.g. two photo-μSR silicon runs, or sample + silver-reference if present; recorded during implementation) | Verify frame-ratio scale, error growth σ² = N + s²·N_BG, and that subtracting a run from itself yields zeros with √(2N) errors |

## Phase 3 — binning, t0, exclusion, periods

| Purpose | Data | Notes |
|---|---|---|
| Constant-error binning yields ~flat σ per output bin | Any long-window run (PTFE ZF) | Assert per-bin asymmetry errors vary by < ~2× across the window vs ≳ e^{λt/2} growth for raw binning |
| Variable binning law | Synthetic + any run | Edges follow bin0·(bin10/bin0)^(t/10 μs); WiMDA-formula cross-check within 0.2% (D8) |
| t0 search recovers loader t0 | Good files across formats: nickel `.nxs` (pulsed), EuO `.bin` (PSI, prompt peak), HAL `.mdu` | Found t0 within ±1 bin (continuous argmax) / ±2 bins (pulsed edge-midpoint vs file value); also synthetic pulses with known centre |
| Detector exclusion | Any multi-detector run + synthetic | Excluding a detector equals removing it from its group: group sums, α estimate, asymmetry all consistent; WiMDA-style range-text parser round-trips ("1,5,10-15") |
| Period subset mapping | **Photo-μSR silicon runs 103277–103298** (`Semiconductors/Photo-muSR in silicon/Data_hdf5/HIFI00103277.nxs` …) — the period-selection study's validated set | Mapping {1→red, 2→green} must reproduce the existing 2-period reduction bit-for-bit; subset sums verified against manual count addition; reuse `docs/porting/period-selection/validate_photomusr.py` expectations |

## Regression gate (all phases)

Previously-verified corpus results that exercise the shared reduction
kernels and must not move:

- CdS 5.12 K link-groups fit, χ²ᵣ = 1.35 (multi-line fit UX study, PR #27).
- EuO order-parameter β extraction (PR #15 study).
- EMU/HIFI repolarisation curve from the time-integral study (PR #23).

Existing suite coverage (`tests/test_*`) plus the corpus-gated tests above;
any change to `grouping.py`, `background.py`, `rebin.py`, `periods.py`
re-runs the full suite (`python tools/harness.py validate`).

## Oracle policy

Decision (Ben, 2026-06-10): **transcribed oracles only** — WiMDA arithmetic
is transcribed from the Pascal into test code (the
`tests/test_wimda_parity_components.py` pattern); no dependency on a live
WiMDA session. Live WiMDA spot-checks remain a possible later follow-on.
