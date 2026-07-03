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

- [ ] Phase 0 note: `global_parameter_fit_window.py` has no TSV export — only
  a GLE-plot export (`_export_plot_gle` / `_export_fit_subplot_gle` /
  `_export_local_parameters_gle`). PLAN.md's Phase 0 task description assumed
  a second TSV writer there for Phase 1c's shared-TSV-writer extraction; the
  only fit-parameters TSV export in the codebase is
  `FitParametersPanel._export_tsv` in `src/asymmetry/gui/panels/fit_parameters_panel.py`.
  If Phase 1c intends to unify TSV export across more than one call site,
  confirm which second call site (if any) is meant — as of 2026-07-03 there
  is only one. (surfaced in Phase 0)
