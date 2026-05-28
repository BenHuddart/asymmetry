# Per-page upgrade brief

Companion to `workflow-catalogue.md`. For each existing user-guide
page, this brief records:

- Current content character (from the Phase 1 audit).
- The 1-3 practical scenarios to add as a "When to Use This" block.
- Source textbook chapters.
- Screenshot need (re-use existing if possible; new only if listed
  in Phase 5).
- Cross-references to the new case-study chapters in
  `docs/user_guide/workflows/`.

Format is intentionally terse — this drives Phase 4 implementation.

---

## High-investment pages (currently API-only)

### `loading_data.rst`

- **Character**: API reference (NeXus / PSI / ROOT loaders).
- **Scenarios to add**:
  - "I have a sample run from ISIS — which format am I likely to
    see?" (NeXus current vs legacy).
  - "I have multi-period data — what's loaded by default?" (default
    period only; cross-link to the period-arithmetic roadmap
    candidate).
  - "My load failed — common causes and fixes" (file corruption,
    HDF5 lib mismatch, missing fields).
- **Textbook**: Blundell Ch 14 (sources), Ch 15.1 (experimental
  setup); Amato-Morenzoni Ch 3.
- **Screenshot**: none new — referenced workflows show the data
  browser populated after a successful load.
- **Workflow links**: all case studies open with "load your data";
  cross-reference workflows/index.

### `project_files.rst`

- **Character**: Schema reference (JSON structure, version
  migrations).
- **Scenarios to add**:
  - "I want to share my analysis with a collaborator" (save
    project + bundle the data files).
  - "I'm coming from musrfit `.msr` — where's the equivalent?"
    (cross-link to `msr-import` candidate; explain `.asymp` is
    JSON and machine-readable but not designed for hand-editing).
  - "Schema migrations: what happens when I open an older
    project?"
- **Textbook**: N/A — workflow-management topic.
- **Screenshot**: new `project_save_restore` (Phase 5) — save dialog
  + restored state.
- **Workflow links**: workflow-catalogue
  `project-reproducibility`.

### `data_processing.rst`

- **Character**: API reference + one rebinning example.
- **Scenarios to add**:
  - "I just rebinned my data — how do I know I didn't oversmooth?"
    (Nyquist on the precession frequency).
  - "Deadtime: when do I need to apply it?" (PSI BIN data
    typically; NeXus often pre-corrected).
  - "Background subtraction: when?" (after deadtime, before
    grouping).
- **Textbook**: Blundell Ch 15.3 (data characteristics);
  Amato-Morenzoni Ch 3.
- **Screenshot**: re-use existing `data_processing_rebin`; new
  `bunching_comparison` (Phase 5) showing ×1 vs ×4 vs ×16.
- **Workflow links**: every workflow's "prepare data" step.

### `fourier_analysis.rst`

- **Character**: API reference + workflow notes + FFT setup
  (MaxEnt stubbed).
- **Scenarios to add**:
  - "I have a TF run — should I look at the time domain or the
    frequency domain first?" (FFT for multi-frequency
    identification; time domain for single-frequency fits).
  - "Choosing apodisation: when Lorentz vs Gauss vs none?"
    (Lorentz when peaks are exponentially-damped, Gauss for
    Gaussian-damped, none if S/N is high).
  - "MaxEnt is currently a stub" — pointer to
    `maxent-spectrum` candidate.
- **Textbook**: Blundell Ch 15.5 (frequency domain), Ch 9.5 (vortex
  P(B)); Amato-Morenzoni Ch 4 (Fourier in μSR).
- **Screenshot**: re-use existing `fourier_tf`; new
  `apodisation_comparison` (Phase 5).
- **Workflow links**: workflow-catalogue `vortex-tf-fourier`,
  `fmuf-identification` (FFT to recognise the beat pattern),
  `muonium-radical-hyperfine` (resolve the hyperfine pair).

### `parameter_trending.rst`

- **Character**: API reference + SC gap-model catalog.
- **Scenarios to add**:
  - "I have σ(T) from a superconductor — which gap model do I
    start with?" (decision tree based on low-T trend).
  - "When to use the parameter-trending panel vs writing a Python
    script" (interactive exploration; export to script for
    publication).
  - "How do I propagate uncertainties through the gap-model fit?"
    (cross-link to `minos-error-analysis` candidate).
- **Textbook**: Blundell Ch 9 (superconductors), Ch 8 (dynamic
  trending); Amato-Morenzoni Ch 6.
- **Screenshot**: re-use existing `parameter_trending_mgb2`; new
  `temperature_trend_fit` (Phase 5) for the EuO order parameter.
- **Workflow links**: `workflows/superconductor_penetration_depth`,
  `workflows/temperature_scan_magnetism`.

---

## Medium-investment pages

### `composite_models.rst`

- **Character**: API + expression-syntax reference + fraction groups.
- **Scenarios to add**:
  - "When do I multiply components vs add them?" (multiplicative =
    cascaded physical effect, e.g. Gaussian envelope × Oscillatory;
    additive = independent channels).
  - "When to use fraction groups: linking volume fractions across
    components."
  - "I built a model and the fit fails — debugging strategies."
- **Textbook**: Blundell Ch 5 (polarisation function composites);
  Amato-Morenzoni Ch 4.
- **Screenshot**: re-use existing `composite_models_builder`; new
  `composite_fractions_dialog` (Phase 5).
- **Workflow links**: All case studies use composites; cross-link
  workflows/index.

### `logbook.rst`

- **Character**: API + GUI export.
- **Scenarios to add**:
  - "I loaded 50 runs — how do I tag and filter them?" (data
    browser column filters; group definitions).
  - "Exporting the logbook for a paper: TSV vs RTF" (TSV for
    further processing; RTF for direct paste).
- **Textbook**: N/A — workflow-management.
- **Screenshot**: re-use existing `logbook_view`.
- **Workflow links**: every case study opens with logbook setup.

### `global_fit_wizard.rst`

- **Character**: Workflow reference + two-phase design.
- **Scenarios to add**:
  - "I have an LF decoupling series — global fit or one-at-a-time?"
    (global preferred when Δ is shared across runs).
  - "Temperature series: do I share or split parameters?"
    (depends on whether the model is qualitatively the same
    across the series).
- **Textbook**: Blundell Ch 5.2 (LF decoupling); Hayano 1979.
- **Screenshot**: re-use existing `global_fit_lfkt`.
- **Workflow links**: `workflows/lf_decoupling_dynamics`.

### `grouped_time_domain_fitting.rst`

- **Character**: Workflow reference + per-group vs shared.
- **Scenarios to add**:
  - "When to use grouped fitting vs the standard asymmetry
    workflow" (Knight shift; vortex-lattice second-moment
    analysis).
  - "Reading per-group fit parameters: what does the Local /
    Global classification mean?"
- **Textbook**: Sonier RMP 72, 769 (2000); Blundell Ch 9.5.
- **Screenshot**: re-use existing `grouped_fit_ybco_knight`.
- **Workflow links**: workflow-catalogue
  `paramagnetic-knight-shift`.

### `gui_usage.rst`

- **Character**: GUI tour (already comprehensive at ~1000 lines).
- **Scenarios to add**: new top-level "Analysis Workflows" section
  pointing into the new `workflows/` subdir as the recommended
  onboarding path for new users coming from real experiments.
- **Textbook**: N/A.
- **Screenshot**: re-use existing.
- **Workflow links**: explicit pointer to `workflows/index`.

---

## Light-touch pages (already strong)

### `lf_kubo_toyabe.rst`

- **Character**: Theory + worked examples (already strong).
- **Add**: short pointer to `workflows/lf_decoupling_dynamics`
  case study under "Further reading".
- **Textbook**: Blundell Ch 5.2 (Hayano 1979).
- **Workflow links**: `workflows/lf_decoupling_dynamics`.

### `muon_fluorine.rst`

- **Character**: Theory + decision tree (already strong).
- **Add**: pointer to workflow-catalogue
  `fmuf-identification`; note the FFT diagnostic for
  recognising F–μ–F vs simple two-frequency oscillation.
- **Textbook**: Blundell Ch 4.6.
- **Workflow links**: optional
  `workflows/muon_fluorine_identification` (Phase 3 stretch goal).

### `sc_penetration_depth.rst`

- **Character**: Theory + model-selection table (already strong).
- **Add**: pointer to `workflows/superconductor_penetration_depth`
  case study.
- **Textbook**: Blundell Ch 9; Amato-Morenzoni Ch 6.
- **Workflow links**: `workflows/superconductor_penetration_depth`.

### `fit_wizard.rst`

- **Character**: Workflow narrative + worked example (already
  strong).
- **Add**: section "When to call the wizard" — unfamiliar
  spectrum, choosing between two candidate models that look
  visually similar.
- **Textbook**: N/A (Asymmetry-specific tool).
- **Workflow links**: all case studies use the wizard as a
  diagnostic.

### `detector_grouping.rst`

- **Character**: GUI tour + instrument schematics (already strong).
- **Add**: pointer to workflow-catalogue
  `paramagnetic-knight-shift` for the use case where
  per-detector grouping is critical.
- **Textbook**: Amato-Morenzoni Ch 3.
- **Workflow links**: workflow-catalogue
  `paramagnetic-knight-shift`.

### `vector_polarization.rst`

- **Character**: Feature reference.
- **Add**: brief note on when vector polarisation matters
  (anisotropic samples; oriented single crystals).
- **Textbook**: Blundell Ch 6.3 (local field anisotropy).
- **Workflow links**: none direct.

### `fitting.rst`

- **Character**: API reference + theory exposition.
- **Add**: "When to use the API vs the GUI" — Python for
  reproducibility / scripting; GUI for exploration / fitting
  unfamiliar data.
- **Textbook**: Blundell Ch 5.
- **Workflow links**: workflows/index for full examples.

### `comparison.rst`

- **Character**: positioning + roadmap (already strong).
- **Add**: pointer to workflows/index — "if you've used WiMDA /
  musrfit / Mantid, here's how their typical workflow maps onto
  Asymmetry".
- **Textbook**: N/A.
- **Workflow links**: workflows/index.

---

## Summary

- **5 pages** need substantial practical-guidance additions
  (loading_data, project_files, data_processing, fourier_analysis,
  parameter_trending).
- **5 pages** need medium additions (composite_models, logbook,
  global_fit_wizard, grouped_time_domain_fitting, gui_usage).
- **8 pages** need light pointers only (lf_kubo_toyabe,
  muon_fluorine, sc_penetration_depth, fit_wizard,
  detector_grouping, vector_polarization, fitting, comparison).
- **New screenshots needed (Phase 5)**: project_save_restore,
  bunching_comparison, apodisation_comparison,
  temperature_trend_fit, composite_fractions_dialog,
  fit_diagnostic_failure, fit_results_table, grouping_dialog_full.
  Plus optional period_selection_dialog, muonium_radical_demo.
  Total: 8-10 new scenarios.
- **New case studies (Phase 3)**:
  `workflows/temperature_scan_magnetism.rst`,
  `workflows/superconductor_penetration_depth.rst`,
  `workflows/lf_decoupling_dynamics.rst`. Optional fourth on
  muon-fluorine identification.
- **New roadmap candidates (Phase 6)**:
  `muonium-radical-hyperfine`, `bpp-relaxation`,
  `structural-transitions`, `lem-depth-profiling`.
