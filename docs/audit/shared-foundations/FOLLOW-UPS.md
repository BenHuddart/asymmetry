# Shared-Foundations Audit — Follow-ups

Out-of-scope discoveries surfaced during the audit. These are **not** actioned in
this PR; they are recorded here (and summarized in the PR description) for later
work. Do not widen the audit diff to address these.

Format: `- [ ] <area> — <what/why> (surfaced in Phase N)`

---

## Known at plan time (from PLAN.md decisions)

- [ ] `plot_panel.py` (7,293 lines) decomposition — explicitly out of scope for
  this audit; migrates onto shared widgets only.
- [ ] `mainwindow.py` (13,032 lines) decomposition — out of scope; migrates onto
  shared widgets only.
- [ ] Retire the `fit_panel.py` deprecation shim once all imports move to
  `panels/fit/` (created in Phase 2).
- [ ] Re-port `feat/fit-wizard-scope` (11 unmerged local commits: Scope tabs in
  both wizards, physics tags, tiered screening, FFT peak editing) onto the new
  wizard base (Phase 3) after this PR merges.

## Surfaced during the audit

<!-- append discoveries below -->

- [x] Phase 2 note (H2 param-table factory — NOT warranted): PLAN.md proposed
  folding the two tabs' "inline ~80-line parameter-table setup" into one
  `_build_parameter_table()` factory on `FitTabBase`. On inspection the setup is
  NOT duplicated: `SingleFitTab` uses a single `FitParameterTable()` whose setup
  already lives in the shared `FitParameterTable` class (`tab_base.py`), while
  `GlobalFitTab` builds several bespoke, differently-configured tables (a
  parameter-classification table + per-group model tables with hidden columns,
  custom column widths, and grouped item-changed wiring). The genuinely-shared
  foundation — `FitParameterTable`, the `_CommitOnTabDelegate` /
  `_ValueUncertaintyDelegate` delegates, and `_apply_param_table_style` — is
  ALREADY shared in `tab_base`. A cross-tab factory would be over-abstraction
  over divergent uses, so H2 was intentionally skipped. (surfaced in Phase 2)

- [ ] Phase 0 note: `global_parameter_fit_window.py` has no TSV export — only
  a GLE-plot export (`_export_plot_gle` / `_export_fit_subplot_gle` /
  `_export_local_parameters_gle`). PLAN.md's Phase 0 task description assumed
  a second TSV writer there for Phase 1c's shared-TSV-writer extraction; the
  only fit-parameters TSV export in the codebase is
  `FitParametersPanel._export_tsv` in `src/asymmetry/gui/panels/fit_parameters_panel.py`.
  If Phase 1c intends to unify TSV export across more than one call site,
  confirm which second call site (if any) is meant — as of 2026-07-03 there
  is only one. (surfaced in Phase 0)

- [ ] Phase 2 note: the `fit_panel.py` → `panels/fit/` split re-exports 23
  public + private symbols through the shim. Downstream `from
  ...fit_panel import <private helper>` call sites (e.g.
  `multi_group_fit_window.py` importing `_get_file_value_for_parameter`;
  tests importing `_set_tie_button_value`, `_tie_button_value`,
  `_dataset_representation_domain`, `_fit_domain_mismatch_message`,
  `_model_without_trailing_background`, `_bounded_phase_seed_padding`,
  `_MAX_PHASE_SEED_FFT_POINTS`) should migrate to importing from the owning
  submodule (`panels.fit.tab_base` / `panels.fit.seeding`) when the shim is
  retired. (surfaced in Phase 2)

- [x] Phase 1c scoped down from PLAN.md's description (shared TSV writer + GLE
  wrapper + export-path/binary-discovery helper) to just a shared GLE
  subprocess-invocation wrapper. Export-path caching (`default_export_path`,
  `remember_export_path`, `resolve_gle_export_paths` in
  `src/asymmetry/gui/export_paths.py`) and GLE binary discovery
  (`get_gle_executable` in `src/asymmetry/gui/gle_settings.py`) were already
  shared before this audit — nothing to extract there. TSV writing is a single
  call site (`FitParametersPanel._export_tsv`, see the Phase 0 note above), so
  there was no duplication to consolidate. The only real duplication was the
  `subprocess.run([_gle, "-d", <fmt>, <file>], capture_output=True, ...,
  check=True, cwd=...)` invocation copy-pasted across 6 GLE export/preview call
  sites in `fit_parameters_panel.py`, `plot_panel.py`, and
  `global_parameter_fit_window.py`; that is now `compile_gle()` in
  `src/asymmetry/gui/utils/export.py`, covered by `tests/test_export_utils.py`.
  (surfaced in Phase 1c)

- [ ] Phase 2 H3 — candidate (c) `_on_stop_fit` deliberately NOT unified. The
  two implementations diverge on worker selection (single cancels
  `self._fit_worker`; global cancels `self._count_fit_worker or self._fit_worker`
  — global has a second worker) AND on the label widget they write "Cancelling
  fit…" into (`_result_label` vs `_result_text`). Two independent divergences
  make the shared inner step (disable stop, set label, `worker.cancel()`) not
  worth extracting behind two hooks. Left in each subclass. (surfaced in Phase 2 H3)

- [ ] Phase 2 H3 — candidate (d) wizard-result caching NOT hoisted. The three
  cache attrs (`_cached_wizard_recommendation` / `_cached_wizard_signature` /
  `_cached_wizard_log_text`) and `_fit_wizard_window` carry different concrete
  types per tab (`FitWizardWindow`/`FitWizardRecommendation` vs
  `GlobalFitWizardWindow`/`GlobalFitWizardRecommendation`), and the
  signature-matching logic is entirely different: single's
  `_wizard_base_signature_matches` compares `run_number` + `model` and caches via
  `copy.deepcopy`; global's compares `run_numbers`/`model`/`values`/`types`/`bounds`
  and normalizes via `_normalized_wizard_signature`. The cache-clear triplet
  (`= None/None/""`) exists ONLY in single (`restore_state`), so there is nothing
  to dedupe. The attr-init split alone does not justify adding an `__init__` to
  `FitTabBase` (which currently defines none). Left entirely in each subclass. If
  a future pass adds a base `__init__`, the shared `_cached_wizard_log_text = ""`
  seed could move there. (surfaced in Phase 2 H3)

- [ ] Phase 2 H3 — candidate (e) shared fit-precondition validation NOT hoisted.
  The pre-fit preambles of `SingleFitTab._run_fit` and
  `GlobalFitTab._run_global_fit` diverge in more than the dataset expression: they
  write to different label widgets (`_result_label` vs `_result_text`), use
  different user-visible message prefixes (`"ERROR: "` vs `"Error: "`), different
  guard checks (single: `_current_dataset is None` + a `missing_component_names`
  check; global: an early `is_grouped_time_domain_mode()` return + a
  `len(_datasets) < 2` check), and different dataset pickers
  (`_current_dataset` vs `_datasets[0]`). Only the `_fit_domain_mismatch_message`
  call and the `_fit_blocked` bail are structurally similar, and unifying even
  those would risk changing the observable `ERROR:`/`Error:` text. Forcing a
  shared `_validate_fit_preconditions()` would need ~4 hooks and still not fit
  global's grouped-mode early return — over-abstraction. Left in each subclass.
  (surfaced in Phase 2 H3)
