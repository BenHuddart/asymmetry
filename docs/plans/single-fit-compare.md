# Saved single fits and a Compare window

Status: in progress on `feat/single-fit-compare` (2026-10-05).

## Problem

A batch fit is recorded as a `FitSeries`. A different model or window records a
new series beside the old one. The Batch tab's **Series** row opens, renames and
deletes them, and the plot's **Fits · N** menu overlays several on one run.

A single fit has none of this. Each `(run, representation, projection)` holds
one `FitSlot`, and every Fit overwrites it, together with the plot's `"single"`
curve. To compare two models on one run today, the user has to note the first
fit's numbers down before running the second. Nothing in the app compares
goodness of fit (χ², AICc, BIC) between fits the user has kept.

## Settled design (decision log)

Ben chose D1 and D2 on 2026-10-05: recording is automatic, mirroring series D3,
and comparison is the plot overlay plus a Compare window. D3–D9 are lead
decisions inside those answers, recorded here so that review can overturn them.

- **D1: recording is automatic, like a series.**
  - **Matching rule.** A Fit whose *identity* (D3) matches the open fit
    replaces that fit in place: same `fit_id`, same label. Otherwise, if a fit
    on this slot shares the identity, the newest such fit is replaced and
    becomes the open fit. Otherwise a new fit is recorded beside the others
    and becomes the open fit.
  - **New fit.** Pressing **New fit** detaches the form, so the next Fit
    records a new fit even when nothing changed. This is how the user keeps a
    second minimum of the same model. Detaching is transient tab state, not
    project state.
- **D2: compare by overlay and by a window.**
  - **Overlay.** Every saved fit on the run appears in the plot's **Fits · N**
    menu under its own name, alongside any series covering the run.
  - **Window.** **Compare…** opens a window that hosts the shared
    `ModelComparePanel` with N = 1 run. Its footer button reads **Open A in the
    Single tab**.
- **D3: what identity covers.**
  - It covers the normalised composite model, the fit window, and each
    parameter's fixed flag, fixed value, link group and tie.
  - Seeds and bounds do not count. A single fit is exploration: iterating on
    starting guesses or tightening a bound must not grow the list. **New fit**
    covers the case where the user wants both kept.
  - The function is `single_fit_identity` in core, used both by the recorder
    and by the tab's "Edited" tag.
- **D4: storage.**
  - Each `(representation, projection)` holds a `SingleFitSet`: an ordered list
    of `FitSlot`s plus the open fit's id.
  - `FitSlot` gains three fields: `fit_id` (unique across the project, so the
    plot can key curves by it), `label` (a user rename or a disambiguating
    suffix only; the default is rendered on demand), and `fit_range`
    (`{"min", "max"}`, numeric, in the domain's unit).
  - `Representation.fit_for` / `fit` keep returning the *open* fit, so every
    existing reader (Add to series, the Diagnostic, the grouped surface, the
    restore mediator) is unchanged.
  - Schema v24 → v25 moves `fit` / `projection_fits` into `single_fits`. Each
    stored slot becomes the sole, open fit of its set.
- **D5: names.**
  - The default name is `<model> · <window>`, built from the same pieces as a
    series name (`composite_model_label`, `fit_window_label`).
  - A default that collides with another fit on the slot is pinned with
    `" (2)"`, as series D10 does.
  - **Rename…** stores a label. A Fit that replaces in place keeps the label.
- **D6: the plot.**
  - The open fit stays the plot's `"single"` curve, so every existing path
    (fresh fit, project restore, legacy stores) is untouched.
  - Every other saved fit draws under the plot id `single:<fit_id>`. Its curve
    is derived from its model and stored parameters over its own window, as
    `_overlay_series` does for series.
  - Switching the open fit re-derives the two curves that swap roles.
  - Deleting a fit clears its id.
  - The menu reads the open fit as `Single fit · <name>`.
- **D7: comparability.**
  - An information criterion compares fits of the *same data*. A fit's data
    key is its window plus its point count, `n = ndof + npar`.
  - The Compare window ranks fits within each data key: Δ and evidence
    weights are taken per group. Groups are listed with the open fit's group
    first. Each row's title carries its window, so the grouping reads on the
    board.
  - A fit with no recorded `npar` (failed HESSE) or no window (a pre-v25 fit)
    has no data key. It is listed with Δ "—" and is never ranked against
    others.
- **D8: the metric.**
  - The default is AICc, with a combo offering AIC / AICc / BIC. These are the
    wizard's `SelectionMetric` and `compute_information_criteria`, computed from
    the stored χ² with k = `npar`.
  - The residual gate is not re-run on saved fits. `gate_passed` is true, and
    the parameter flags (at bound, poorly determined) carry the warnings.
- **D9: curves off the GUI thread.**
  - The window builds the ranking (cheap) immediately.
  - Dense curves and normalised residuals are built on demand in a
    `TaskRunner` worker, through the panel's existing
    `curves_required` / `refresh_curves` protocol.

## Code map

- `core/representation/base.py`: `FitSlot` fields, `SingleFitSet`,
  `Representation` storage and the record/open/rename/delete API.
- `core/representation/single_fits.py` (new): `single_fit_identity`,
  `single_fit_default_name`, `single_fit_data_key`.
- `core/fitting/model_comparison.py`: `SavedFitInput`, `summarise_saved_fits`.
- `core/project/schema.py`: v24 → v25 migration.
- `gui/panels/fit/single_tab.py`: the **Fit** section (selector, New fit,
  Rename…, Delete…, Compare…), the Edited / New-fit tag, and signals.
- `gui/panels/fit/panel.py`: forwards the signals and the catalogue provider.
- `gui/mainwindow.py`: the recorder, open/rename/delete handlers, the plot
  overlays for saved fits, and the fit report including every saved fit.
- `gui/windows/single_fit_compare_window.py` (new): the Compare window.
- `gui/panels/plot_panel.py`: the `single:` fit-id labels in the Fits menu.
- Docs: `docs/reference/gui_usage.rst` (Single fitting),
  `docs/reference/project_files.rst` (schema v25), screenshot scenario,
  `CHANGELOG.md`.

## Phases

1. Core: storage, identity, names, the schema migration and the compare
   adapter, with tests.
2. GUI recording: the recorder, the Single tab's Fit section, the plot
   overlays and the fit report.
3. The Compare window.
4. Docs, screenshot scenario and changelog; then `validate` and the PR.
