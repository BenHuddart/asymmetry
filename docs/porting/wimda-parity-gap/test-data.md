# Test data — WiMDA parity gap portfolio

Corpus references for verifying the portfolio's projects. The shared corpus
is the WiMDA Muon School data under `~/Documents` (see `docs/testing/` for
the loading guide). Standing constraint: all of the "Basics" set is HDF4 and
out of scope; loadable coverage = HDF5 `.nxs` (Nickel, Nuclear/ionic,
pulsedTD sets), PSI `.bin`/`.mdu` (EuO, Chemistry), ROOT.

Each project's own study pass owns its detailed test-data document; this maps
the known-good corpus series to projects so studies start warm.

| Project | Corpus / data ideas |
|---|---|
| data-reduction-parity | TF calibration runs for the diamagnetic alpha estimator vs `estimate_alpha` baseline; LF/ZF relaxing runs (HIFI LF series, runs 118222–118240, 10–100 G @ 350 K — corannulene, `Chemistry/Molecular dynamics of corannulene/data_hdf5/`; already used for the integral study) for the "General" alpha method, which the ΣF/ΣB estimator cannot handle; pulsed ISIS runs for tail-fit background (no pre-t0 region); photo-μSR silicon runs 103277–103298 (multi-period) for subset→R/G mapping; long-time relaxation runs for variable/constant-error binning. |
| run-arithmetic | Pairs/triples of repeated runs from any corpus series: co-add at count level, compare against single long run statistics; laser-on/off photo-μSR pairs for co-subtract; verify error propagation vs Poisson expectation on synthetic histograms. |
| count-domain-fit-modes | TF calibration run: α from fgFB-style fit vs grouping-dialog estimate; single-histogram N0/BG fits on any continuous-source PSI run; double-pulse: ISIS double-pulse-mode runs if available, else synthetic two-pulse data from simulate-mode. |
| fit-workflow-diagnostics | Any temperature scan crossing a transition (EuO series — OrderParameter precedent, PR #15) to demonstrate warm-start value; MINOS vs HESSE asymmetry on a deliberately non-parabolic likelihood (low-stat KT fit). |
| frequency-domain-finishers | High-TF runs (HiFi/HAL) for field-axis spectra; known-frequency synthetic data for pulse-rolloff compensation; closely-spaced doublet synthetic series for Burg pole-scan characterisation (document its spurious-splitting onset as part of the diagnostic framing). |
| maxent-completion | The MaxEnt testing artifacts in the Asymmetry-testing worktree (from PR #16/#26 work); ISIS pulsed runs ≳5 MHz for pulse-shape response; ZF runs with internal fields for ZF/LF mode. |
| spectral-moments | Vortex-lattice / TF superconductor data if available; else synthetic asymmetric field distributions with known analytic moments (skewed Gaussian mixtures) — moments have closed forms, so synthetic-first verification is strong here. WiMDA's `Moments.pas` arithmetic as behavioural oracle. |
| simulate-mode | Round-trip: simulate from a fitted model of a real corpus run → reload through `NexusLoader` → refit recovers parameters within errors. Degrade-statistics: thin a high-stat run, verify error scaling ∝ 1/√N. |
| model-function-parity | Existing verified trend series: CdS 5.12 K (link-groups study, χ²ᵣ=1.35), EuO β-extraction (OrderParameter), EMU repolarisation curve (integral study); WiMDA `fitfunctions.pas` formulas transcribed as numerical oracles (same approach as `tests/test_wimda_parity_components.py`). |
| rrf | High-TF run: RRF-demodulated envelope must match the directly-fitted relaxation envelope; synthetic single-frequency data for exactness. |
| workflow-visualisation | Any corpus directory for run stepping/pattern walking; ASCII export golden files; events-column values vs WiMDA logbook output for the same runs. |
| python-user-functions | A worked example plugin reproducing one shipped component (e.g. Keren) bit-for-bit through the plugin path; registry isolation tests (synthetic only). |
| negative-muon-analysis (deferred) | No μ⁻ corpus locally; acquiring ISIS μ⁻ elemental-analysis data is a prerequisite flagged in the brief. |

Cross-cutting oracles:

- **WiMDA arithmetic as contract**: where behaviour is ported, transcribe the
  Pascal into the study and test against tabulated values (precedent:
  `tests/test_wimda_parity_components.py`). Note the known WiMDA `KTBArray`
  low-field bug (`KuboToyabe.pas:151–166`) — never oracle against WiMDA
  array-mode static KT at 0 < B < 2Δ.
- **Mantid as oracle only** (GPL): `MuonMaxent`, `PlotAsymmetryByLogValue`
  precedents already establish the pattern; never vendor.
- `python tools/harness.py validate` green is the gate for every phase.
