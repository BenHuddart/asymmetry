# Comparison: current implementation vs unified projections model

This is an internal unification, so the "reference" being compared against is
Asymmetry's own current code, not an external program.

## The two cases today

| | Vector polarization | TF dual grouping |
|---|---|---|
| Where | EMU `Vector Polarization` preset | MuSR / HiFi transverse presets |
| Channels | 3 (`P_x`,`P_y`,`P_z`) | 2 (e.g. `Top–Bottom`, `Fwd–Back`) |
| Defined by | 6 groups, canonical **names** | separate **presets**, one F/B pair each |
| Switching | rewrite F/B pair, recompute; `vector_axis` | re-open dialog, re-apply preset |
| View together | "All" → stacked subplots | not possible |
| Per-channel alpha | `alpha_x/y/z` (+ legacy `alpha_px/py/pz`) | n/a (single alpha) |

Both are the same abstraction (a set of `(fwd, bwd, alpha)` projections of the
same run); only vector mode partially implements it, and does so by string
inference rather than declaration.

## Current vector-mode code seams

- **Preset definition (name-encoded):**
  `src/asymmetry/core/instrument.py:556` — EMU `Vector Polarization` declares six
  `GroupDefinition`s named `Pz Forward/Backward`, `Py Top/Bottom`,
  `Px Left/Right`. The axis identity lives only in those strings.
- **String-match inference, ×3 sites:**
  - `gui/windows/grouping_dialog.py:1543` `_detect_vector_axis_pairs`
  - `gui/mainwindow.py:1577` `_vector_axis_pairs_for_grouping`
  - `gui/mainwindow.py:1618` `_vector_axis_state_for_dataset`
  Each lowercases group names and looks for `"pz forward"` etc. → fragile,
  duplicated.
- **Axis switching mutates grouping:**
  `gui/mainwindow.py:1866` `_synchronize_targets_to_axis` rewrites
  `forward_group`/`backward_group` to the selected axis's pair and recomputes the
  ordinary F/B asymmetry. So under the hood an "axis" is just a F/B pair + alpha.
- **"All" clones per axis:**
  `gui/mainwindow.py:1839` `_build_vector_axis_datasets` deep-copies the dataset
  once per axis and stacks subplots — precedent for parallel per-projection
  datasets; generalize from all-or-one to the selected subset.
- **Header selector:**
  `gui/panels/plot_panel.py:272` `_polarization_combo` (`x/y/z/All`,
  mutually-exclusive). Replaced by the multi-select chip bar.
- **Per-axis alpha:** stored in the grouping payload as `alpha_x/y/z`; resolved
  with legacy fallback in `mainwindow._resolve_vector_alpha_values` and the
  grouping dialog's `_alpha_value_for_axis`.

## Fit-storage seams (per-projection persistence)

- **Persistent slot is axis-agnostic:** `core/representation/base.py:54` `FitSlot`
  — one per `Representation`, container keyed `(run, rep_type)`
  (`core/representation/container.py`). No projection dimension. Switching axis
  yields a new dataset; the old fit is simply not reloaded.
- **Display layer is already axis-aware (transient):**
  `gui/panels/plot_panel.py:345` caches `_fit_curves_by_key`,
  `_fit_components_by_key`, `_fit_metadata_by_key` by `(run, axis_key)` with
  `axis_key ∈ {None,P_x,P_y,P_z}`; `_fit_curve_for_dataset`
  (`plot_panel.py:2075`) resolves with axis-aware fallback. **Not persisted.**
- **Series/global fits:** `core/representation/series.py:58` `FitSeries`
  (`batch_id`, `member_run_numbers`, `results_by_run`, `param_roles`). This is
  the structure joint-across-projections fitting will mirror as
  `results_by_projection`.
- **Project schema:** currently v8, no axis/projection awareness in saved fits.
  Unified model needs v9 (projection-keyed `FitSlot` map) + forward migration.

## What changes

| Concern | Today | Unified |
|---|---|---|
| Channel identity | inferred from group-name strings (×3 sites) | declared `AsymmetryProjection` on preset/schema |
| Selector | `x/y/z/All` combo (mutually exclusive) | multi-select chip bar, floor 1, no "All" chip |
| View | all-or-one stacked subplots | any selected subset, one subplot each |
| Colour | (n/a — combo) | frame-tint identity vs run-colour (RG) preserved |
| Fit target | implicit / ambiguous | selectable subplot + fit-panel echo |
| Fit storage | `(run, rep_type)` single slot | `(run, rep_type, projection)` map, schema v9 |
| TF dual-grouping | mutually-exclusive presets | two projections, same machinery |
